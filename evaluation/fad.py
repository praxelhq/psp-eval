"""Fréchet Audio Distance (FAD) — corpus-level distributional accent distance.

FAD complements the per-phoneme PSP probes by capturing accent signals at the
corpus level: timbre, co-articulation, prosodic envelope, and distributional
phoneme frequency --- things individual phoneme-span probes miss entirely.

**Formal definition.** Model each corpus (generated, native-reference) as a
multivariate Gaussian in a speech embedding space (we use Wav2Vec2-XLS-R
layer-9, matching the space PSP uses). The Fréchet distance between two
Gaussians :math:`\\mathcal{N}(\\mu_g, \\Sigma_g)` and
:math:`\\mathcal{N}(\\mu_n, \\Sigma_n)` is

.. math::
   d^2 = \\| \\mu_g - \\mu_n \\|^2 + \\mathrm{tr}(\\Sigma_g + \\Sigma_n - 2(\\Sigma_g \\Sigma_n)^{1/2})

Lower is better; 0 means the two distributions coincide. This matches the
original Fréchet Inception Distance (FID, images) and Fréchet Audio Distance
(FAD, general audio) formulations.

**Why XLS-R layer 9.** PSP already uses layer 9 as its phonetic-discriminative
representation. Using the same layer for FAD ensures the two metrics live in
the same embedding space — we can interpret correlations between them rather
than confound embedding-space drift with metric-definition differences.

**Scale normalization.** Raw FAD values depend on the embedding dimension and
the absolute scale of the reference distribution. We report both:
- ``fad_raw``: the unnormalised Fréchet distance
- ``fad_silence_normalized``: fad_raw divided by fad(silence_corpus, native_corpus)
  — 0 ≈ native, 1 ≈ as bad as pure silence. This matches the ACCENT.md spec.

**Usage.**

    from evaluation.fad import compute_fad

    # Shape: (N_clips, D); D = 1024 for XLS-R-300m layer 9.
    generated_embs = ...  # mean-pool XLS-R layer 9 over each generated clip
    native_embs    = ...  # same, computed from native-reference corpus

    raw, info = compute_fad(generated_embs, native_embs)
    print(f"FAD = {raw:.3f}   (n_gen={info['n_gen']}, n_ref={info['n_ref']})")
"""

from __future__ import annotations

from typing import Any


def compute_fad(
    generated_embeddings: Any,  # np.ndarray shape (N, D)
    native_embeddings: Any,     # np.ndarray shape (M, D)
    eps: float = 1e-6,
) -> tuple[float, dict[str, Any]]:
    """Fréchet distance between two corpora represented as collections of
    utterance-level embeddings.

    Returns ``(fad, info_dict)`` where ``info_dict`` has:
    - ``n_gen``, ``n_ref``: sample counts per side
    - ``mean_norm``: ||μ_gen − μ_ref|| (component of the distance)
    - ``trace_term``: tr(Σ_g + Σ_n − 2 sqrt(Σ_g Σ_n))

    Numerical stability: we add ``eps * I`` to each covariance before the
    matrix-sqrt step to avoid negative eigenvalues from rank-deficient
    estimates when sample size is small (< D).

    Raises ValueError if either corpus has <2 samples (covariance undefined)
    or if embedding dimensions disagree.
    """
    import numpy as np
    from scipy import linalg

    g = np.asarray(generated_embeddings, dtype=np.float64)
    n = np.asarray(native_embeddings, dtype=np.float64)

    if g.ndim != 2 or n.ndim != 2:
        raise ValueError(f"Both inputs must be 2D; got g.shape={g.shape}, n.shape={n.shape}")
    if g.shape[1] != n.shape[1]:
        raise ValueError(f"Embedding dim mismatch: g.shape[1]={g.shape[1]} n.shape[1]={n.shape[1]}")
    if g.shape[0] < 2 or n.shape[0] < 2:
        raise ValueError(f"Need ≥2 samples per corpus for covariance; got n_gen={g.shape[0]} n_ref={n.shape[0]}")

    mu_g = g.mean(axis=0)
    mu_n = n.mean(axis=0)
    sig_g = np.cov(g, rowvar=False)
    sig_n = np.cov(n, rowvar=False)

    # Stabilise: add eps*I to each covariance (avoid negative eigenvalues on
    # sqrt when sample size is small relative to dimension).
    D = g.shape[1]
    sig_g += eps * np.eye(D)
    sig_n += eps * np.eye(D)

    # ||μ_g − μ_n||²
    diff = mu_g - mu_n
    mean_dist_sq = float(diff @ diff)

    # tr(Σ_g + Σ_n − 2 sqrt(Σ_g Σ_n))
    covmean = linalg.sqrtm(sig_g @ sig_n)
    if np.iscomplexobj(covmean):
        # Imaginary parts should be numerical noise — discard.
        if np.abs(covmean.imag).max() > 1e-3:
            # Significant imaginary component — regularise more aggressively
            sig_g_reg = sig_g + 1e-3 * np.eye(D)
            sig_n_reg = sig_n + 1e-3 * np.eye(D)
            covmean = linalg.sqrtm(sig_g_reg @ sig_n_reg)
        covmean = covmean.real

    trace_term = float(np.trace(sig_g) + np.trace(sig_n) - 2 * np.trace(covmean))

    fad = mean_dist_sq + trace_term

    return fad, {
        "n_gen": int(g.shape[0]),
        "n_ref": int(n.shape[0]),
        "mean_norm": float(np.sqrt(mean_dist_sq)),
        "trace_term": trace_term,
        "embedding_dim": int(D),
    }


def utterance_embedding_from_xlsr(
    audio_16k: Any,                    # np.ndarray (samples,) float32 at 16kHz
    embed_model: Any,                  # Wav2Vec2Model with output_hidden_states
    embed_processor: Any,              # Wav2Vec2FeatureExtractor
    device: str,
    layer: int = 9,
) -> Any:
    """Mean-pool XLS-R layer ``layer`` over the full utterance → (D,) np.float32.

    Matches the layer used by RetroflexProbe._embed_span, so FAD and PSP live
    in the same embedding space.
    """
    import numpy as np
    import torch

    inputs = embed_processor(audio_16k, sampling_rate=16_000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        out = embed_model(inputs, output_hidden_states=True)
    hs = out.hidden_states[layer]  # (1, T, D)
    emb = hs.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    return emb


# ---------------------------------------------------------------------------
# Minimal self-test: synthetic Gaussians should give a deterministic FAD.
# ---------------------------------------------------------------------------


def _test_identity_is_zero() -> None:
    """FAD(X, X) = 0 exactly."""
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 32))
    fad, info = compute_fad(x, x)
    # Not strictly zero because of eps regularisation, but should be tiny.
    assert fad < 1e-3, f"FAD(X, X) = {fad}, expected near 0"
    print(f"  FAD(X, X) = {fad:.2e} (near 0 as expected)")


def _test_well_separated_clouds_have_positive_fad() -> None:
    """Two clearly-separated Gaussians: FAD should be positive and reflect separation."""
    import numpy as np
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.0, scale=1.0, size=(200, 16))
    y = rng.normal(loc=3.0, scale=1.0, size=(200, 16))
    fad, info = compute_fad(x, y)
    # |mu_x - mu_y|^2 ~= 3^2 * 16 = 144 component-wise; actual is ||diff||^2 ~= 144
    assert info["mean_norm"] > 10.0, f"Expected mean-norm ~ 12, got {info['mean_norm']}"
    assert fad > 100.0, f"Expected FAD > 100, got {fad}"
    print(f"  FAD between well-separated clouds = {fad:.2f} (mean_norm={info['mean_norm']:.2f})")


def _test_more_separated_has_larger_fad() -> None:
    """Monotonicity: increasing separation should increase FAD."""
    import numpy as np
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 16))
    fads = []
    for loc in [0.5, 1.0, 2.0, 4.0]:
        y = rng.normal(loc=loc, size=(200, 16))
        fad, _ = compute_fad(x, y)
        fads.append(fad)
    # Should be monotonically increasing
    for i in range(len(fads) - 1):
        assert fads[i] < fads[i + 1], f"Non-monotone: fads[{i}]={fads[i]} > fads[{i+1}]={fads[i+1]}"
    print(f"  FAD monotone in separation: {[f'{f:.1f}' for f in fads]}")


def run_tests() -> None:
    _test_identity_is_zero()
    _test_well_separated_clouds_have_positive_fad()
    _test_more_separated_has_larger_fad()
    print("\nall FAD smoke tests passed")


if __name__ == "__main__":
    run_tests()
