"""Prosodic Signature Divergence (PSD) — corpus-level prosodic accent metric.

PSD captures the "music" of speech — F0 contour, rhythm, syllable timing —
signals that PSP's phoneme-span probes and FAD's spectral features miss.
These are the dimensions where Indic languages differ most from English
prosody: Telugu's descending declarative F0, Tamil's syllable-timed rhythm
(low nPVI), Hindi's Sanskrit-derived contour patterns.

**Formal definition.** For each utterance we extract a low-dimensional
prosodic feature vector (F0 statistics + onset rate + rhythm class), then
compute Fréchet distance between the generated corpus's feature distribution
and the native-reference corpus's distribution. Same math as FAD; different
embedding.

**Feature vector** (5 dimensions, per utterance):
1. mean log-F0 (pitch centre)
2. std log-F0 (pitch dynamism)
3. F0 range (max - min log-F0 over voiced frames)
4. onset rate (syllables per second, ≈ speech rate)
5. nPVI on inter-onset intervals (rhythm class: low = syllable-timed, high = stress-timed)

nPVI (normalized Pairwise Variability Index, Grabe & Low 2002) is the
standard rhythmic-class metric in phonological typology. Telugu / Tamil are
syllable-timed (low nPVI ≈ 40); English is stress-timed (nPVI ≈ 65);
Hindi is intermediate (nPVI ≈ 55).

**Why F0 stats on log-scale:** humans perceive pitch logarithmically; a
12-semitone range is the same *perceptually* whether centred at 120 Hz or
240 Hz. Log-space stats are speaker-rate-invariant in a way linear-Hz aren't.

**Usage.**

    from evaluation.psd import extract_prosodic_features, compute_psd

    generated_features = np.stack([extract_prosodic_features(wav) for wav in gen_wavs])
    native_features    = np.stack([extract_prosodic_features(wav) for wav in ref_wavs])

    psd, info = compute_psd(generated_features, native_features)
    print(f"PSD = {psd:.3f} (lower = closer to native prosody)")
"""

from __future__ import annotations

from typing import Any


# Feature-vector dimension. Keep in sync with extract_prosodic_features.
PSD_FEATURE_DIM = 5

# Minimum voiced-frame count for a clip to produce reliable F0 statistics.
MIN_VOICED_FRAMES = 20


def extract_prosodic_features(audio_16k: Any) -> Any:
    """Extract a 5-D prosodic feature vector from a 16 kHz mono clip.

    Returns np.ndarray shape (5,) float32. If F0 extraction fails or the clip
    has too few voiced frames, returns a feature vector with NaN entries; the
    caller should filter these before distribution-level computation.
    """
    import numpy as np
    import librosa

    audio = np.asarray(audio_16k, dtype=np.float32)
    if len(audio) < 16_000 * 0.5:
        # Too short to be meaningful (<500ms)
        return np.full(PSD_FEATURE_DIM, np.nan, dtype=np.float32)

    # --- F0 extraction via pyin (probabilistic yin, robust) ---
    f0, voiced_flag, _voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz
        fmax=librosa.note_to_hz("C6"),   # ~1050 Hz
        sr=16_000,
        frame_length=2048,
        hop_length=512,
    )
    voiced = f0[voiced_flag]
    voiced = voiced[~np.isnan(voiced)] if voiced.size > 0 else voiced

    if voiced.size < MIN_VOICED_FRAMES:
        return np.full(PSD_FEATURE_DIM, np.nan, dtype=np.float32)

    log_f0 = np.log(voiced)
    mean_logf0 = float(log_f0.mean())
    std_logf0 = float(log_f0.std())
    range_logf0 = float(log_f0.max() - log_f0.min())

    # --- Onset detection for syllable-rate proxy + nPVI ---
    onsets = librosa.onset.onset_detect(
        y=audio, sr=16_000, hop_length=512, backtrack=False, units="time",
    )
    duration_s = len(audio) / 16_000.0

    if len(onsets) >= 2:
        onset_rate = (len(onsets) - 1) / duration_s
        iois = np.diff(onsets)   # inter-onset intervals (s)
    else:
        onset_rate = 0.0
        iois = np.array([], dtype=np.float32)

    # nPVI = 100/(n-1) * sum_i |2*(d_i - d_{i+1}) / (d_i + d_{i+1})|
    if len(iois) >= 2:
        pairwise = 2 * np.abs(iois[:-1] - iois[1:]) / (iois[:-1] + iois[1:] + 1e-9)
        npvi = 100.0 * pairwise.mean()
    else:
        npvi = np.nan

    return np.array(
        [mean_logf0, std_logf0, range_logf0, onset_rate, npvi],
        dtype=np.float32,
    )


def compute_psd(
    generated_features: Any,  # np.ndarray shape (N, 5)
    native_features: Any,     # np.ndarray shape (M, 5)
    eps: float = 1e-4,
) -> tuple[float, dict[str, Any]]:
    """Prosodic Signature Divergence: Fréchet distance in 5-D prosodic space.

    Drops rows with NaN values before computation (clips where F0 extraction
    failed). Reuses FAD's Fréchet math.
    """
    import numpy as np

    from evaluation.fad import compute_fad

    g = np.asarray(generated_features, dtype=np.float64)
    n = np.asarray(native_features, dtype=np.float64)

    # Filter NaN rows
    g_mask = ~np.any(np.isnan(g), axis=1)
    n_mask = ~np.any(np.isnan(n), axis=1)
    g_clean = g[g_mask]
    n_clean = n[n_mask]

    if g_clean.shape[0] < 2 or n_clean.shape[0] < 2:
        raise ValueError(
            f"Not enough non-NaN samples after filter: n_gen={g_clean.shape[0]} "
            f"n_ref={n_clean.shape[0]} (need ≥2 each)"
        )

    psd, info = compute_fad(g_clean, n_clean, eps=eps)
    info["n_gen_dropped"] = int((~g_mask).sum())
    info["n_ref_dropped"] = int((~n_mask).sum())
    info["dim"] = PSD_FEATURE_DIM
    return psd, info


# ---------------------------------------------------------------------------
# Tests — synthetic feature vectors to verify math without audio deps.
# ---------------------------------------------------------------------------


def _test_identity_is_near_zero() -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 5))
    psd, info = compute_psd(x, x)
    assert psd < 1e-2, f"PSD(X,X) = {psd}, expected near 0"
    print(f"  PSD(X, X) = {psd:.2e}")


def _test_shifted_corpora_have_positive_psd() -> None:
    import numpy as np
    rng = np.random.default_rng(1)
    # Simulate: native ~ N(0, 1), generated ~ N(1.5, 1) — shifted prosody
    native = rng.normal(loc=0.0, scale=1.0, size=(100, 5))
    gen = rng.normal(loc=1.5, scale=1.0, size=(100, 5))
    psd, info = compute_psd(gen, native)
    assert info["mean_norm"] > 2.5, f"expected mean_norm > 2.5, got {info['mean_norm']}"
    assert psd > 5.0, f"expected psd > 5, got {psd}"
    print(f"  PSD between shifted corpora = {psd:.3f} (mean_norm={info['mean_norm']:.3f})")


def _test_nan_rows_are_dropped() -> None:
    import numpy as np
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 5))
    x_with_nan = x.copy()
    x_with_nan[0, 0] = np.nan
    x_with_nan[5, 2] = np.nan
    y = rng.normal(size=(50, 5))
    psd, info = compute_psd(x_with_nan, y)
    assert info["n_gen_dropped"] == 2, info
    assert info["n_gen"] == 48, info
    print(f"  NaN rows dropped correctly: n_gen_dropped={info['n_gen_dropped']}")


def run_tests() -> None:
    _test_identity_is_near_zero()
    _test_shifted_corpora_have_positive_psd()
    _test_nan_rows_are_dropped()
    print("\nall PSD smoke tests passed")


if __name__ == "__main__":
    run_tests()
