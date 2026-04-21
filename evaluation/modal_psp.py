"""Modal app: score a directory of .wav files with the PSP RetroflexProbe.

Mirrors evaluation/modal_eval.py::score_dir pattern. Reads centroid pickle
from /cache/psp_refs/{lang}_refs.pkl on praxy-voice-vol, emits one .psp.json
per .wav sibling. Scorecard aggregates the .psp.json files.

Run:
    uv run modal run evaluation/modal_psp.py::score_dir \\
        --audio-dir generated/<run_dir> --language te
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import modal

APP_NAME = "praxy-psp-score"
MODEL_VOLUME = "praxy-voice-vol"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)

psp_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        # torch 2.6 required: CVE-2025-32434 blocks legacy .bin CTC loads on
        # older torch. Keep pinned to match psp_bootstrap image.
        "torch>=2.6,<2.8", "torchaudio>=2.6,<2.8",
        "numpy<2", "scipy", "soundfile", "librosa",
        "huggingface_hub>=0.25", "transformers>=4.46",
        "accelerate>=0.34", "sentencepiece",
        "indic-transliteration>=2.3.82",
        "pyctcdecode",  # required by ai4bharat/indicwav2vec-hindi processor
    )
    .add_local_python_source("evaluation", "praxy")
)


_ALIGN_MODEL_BY_LANG = {
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "ai4bharat/indicwav2vec-hindi",
    "ta": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
}


@app.cls(
    image=psp_image,
    gpu="A10G",
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("praxy-hf")],
    timeout=1800,
    scaledown_window=300,
)
class PSPScorer:
    """Loads RetroflexProbe + native centroids per language on demand.

    Lazy-loads so one class instance can serve multiple languages without
    needing per-language parameterized classes (Modal 1.x parameter typing
    is awkward with str). Caches per-language across calls.
    """

    @modal.enter()
    def startup(self) -> None:
        import os
        os.environ.setdefault("HF_HOME", "/cache/hf")
        self._probes: dict = {}
        print("[PSPScorer] initialized, lazy-loading probes per language on first use")

    def _get_probe(self, language: str):
        """Lazy-load + cache a multi-dim probe for a language.

        We instantiate as AspirationProbe (subclass of RetroflexProbe) so one
        probe serves retroflex + aspiration. Length scoring uses a separate
        small LengthProbe that reuses the same loaded models via shared state.
        """
        if language in self._probes:
            return self._probes[language]

        from evaluation.psp import AspirationProbe

        align_repo = _ALIGN_MODEL_BY_LANG[language]
        probe = AspirationProbe(align_model=align_repo)
        probe.load()

        refs_path = Path(f"/cache/psp_refs/{language}_refs.pkl")
        if not refs_path.exists():
            raise FileNotFoundError(
                f"No PSP centroids at {refs_path}. Run psp_bootstrap for lang={language}"
            )
        with open(refs_path, "rb") as f:
            refs = pickle.load(f)

        n_embeddings = 0
        phonemes = []
        for phoneme, embs in refs.items():
            for e in embs:
                probe.add_native_reference(phoneme, e)
                n_embeddings += 1
            if embs:
                phonemes.append(phoneme)
        print(f"[PSPScorer:{language}] loaded {n_embeddings} embeddings across {len(phonemes)} phonemes ({phonemes})")

        self._probes[language] = probe
        return probe

    def _get_fad_natives(self, language: str):
        """Lazy-load + cache native utterance-level embeddings for FAD."""
        if not hasattr(self, "_fad_natives"):
            self._fad_natives = {}
        if language in self._fad_natives:
            return self._fad_natives[language]

        refs_path = Path(f"/cache/psp_refs/{language}_fad_natives.pkl")
        if not refs_path.exists():
            raise FileNotFoundError(
                f"No FAD natives at {refs_path}. Run psp_bootstrap::main_fad for lang={language}"
            )
        with open(refs_path, "rb") as f:
            matrix = pickle.load(f)
        print(f"[PSPScorer:{language}] loaded FAD natives shape={matrix.shape}")
        self._fad_natives[language] = matrix
        return matrix

    def _xlsr_utt_embed(self, audio_bytes: bytes, language: str):
        """Compute a single utterance-level XLS-R layer-9 embedding from audio bytes.

        Reuses the probe's loaded embed model (shared across languages since
        XLS-R-300m is language-agnostic). Returns np.float32 array shape (D,).
        """
        import numpy as np
        import torch

        probe = self._get_probe(language)
        from evaluation.psp import _load_audio_16k
        audio_16k = _load_audio_16k(audio_bytes)
        inputs = probe._embed_processor(
            audio_16k, sampling_rate=16_000, return_tensors="pt"
        ).input_values.to(probe.device)
        with torch.no_grad():
            out = probe._embed_model(inputs, output_hidden_states=True)
        hs = out.hidden_states[probe.embed_layer]  # (1, T, D)
        return hs.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    @modal.method()
    def embed_clip_for_fad(self, audio_bytes: bytes, language: str) -> dict:
        """Return the utterance-level XLS-R embedding for one clip.

        Caller accumulates these across a whole run, then invokes
        ``compute_corpus_fad`` to get the final scalar. Splitting this way
        lets the Modal container batch-process many clips in parallel with
        ``.map()`` before the final FAD math.
        """
        import numpy as np
        emb = self._xlsr_utt_embed(audio_bytes, language)
        return {"embedding": emb.tolist(), "dim": int(emb.shape[0])}

    @modal.method()
    def compute_corpus_fad(self, generated_embeddings: list, language: str) -> dict:
        """Given a list of utterance embeddings (as Python lists), compute FAD
        against this language's native reference matrix.
        """
        import numpy as np
        from evaluation.fad import compute_fad

        native = self._get_fad_natives(language)
        gen = np.asarray(generated_embeddings, dtype=np.float32)
        if gen.ndim != 2:
            raise ValueError(f"Expected 2D generated_embeddings, got shape {gen.shape}")
        fad, info = compute_fad(gen, native)
        return {"fad": fad, **info}

    def _get_psd_natives(self, language: str):
        """Lazy-load + cache native prosodic feature matrix for PSD."""
        if not hasattr(self, "_psd_natives"):
            self._psd_natives = {}
        if language in self._psd_natives:
            return self._psd_natives[language]
        refs_path = Path(f"/cache/psp_refs/{language}_psd_natives.pkl")
        if not refs_path.exists():
            raise FileNotFoundError(
                f"No PSD natives at {refs_path}. Run psp_bootstrap::main_psd for lang={language}"
            )
        with open(refs_path, "rb") as f:
            matrix = pickle.load(f)
        print(f"[PSPScorer:{language}] loaded PSD natives shape={matrix.shape}")
        self._psd_natives[language] = matrix
        return matrix

    @modal.method()
    def extract_psd_features(self, audio_bytes: bytes) -> dict:
        """Extract 5-D prosodic feature vector from one clip."""
        import numpy as np
        from evaluation.psd import extract_prosodic_features
        from evaluation.psp import _load_audio_16k

        audio_16k = _load_audio_16k(audio_bytes)
        feat = extract_prosodic_features(audio_16k)
        return {"features": feat.tolist(), "has_nan": bool(np.any(np.isnan(feat)))}

    @modal.method()
    def compute_corpus_psd(self, generated_features: list, language: str) -> dict:
        """Given a list of 5-D feature vectors, compute PSD against native."""
        import numpy as np
        from evaluation.psd import compute_psd

        native = self._get_psd_natives(language)
        gen = np.asarray(generated_features, dtype=np.float32)
        if gen.ndim != 2:
            raise ValueError(f"Expected 2D generated_features, got shape {gen.shape}")
        psd, info = compute_psd(gen, native)
        return {"psd": psd, **info}

    @modal.method()
    def score(self, audio_bytes: bytes, text: str, language: str, script: str | None = None) -> dict:
        """Score one clip against its ground-truth text. Returns retroflex +
        aspiration + length + per-phoneme breakdowns (incl. Tamil-zha) in one dict.
        """
        from evaluation.psp import LengthProbe, per_phoneme_breakdown

        probe = self._get_probe(language)
        retroflex = probe.score(audio_bytes, text, script=script, language_code=language)
        aspiration = probe.score_aspiration(audio_bytes, text, script=script, language_code=language)

        # Length probe reuses the same aligner + embedding models via shared
        # state — instantiate a LengthProbe that wraps the same underlying
        # loaded models so we don't re-load.
        length_probe = LengthProbe()
        length_probe._loaded = True
        length_probe._align_model = probe._align_model
        length_probe._align_processor = probe._align_processor
        length_probe._embed_model = probe._embed_model
        length_probe._embed_processor = probe._embed_processor
        length_probe.device = probe.device
        length_report = length_probe.score_length(audio_bytes, text, script=script, language_code=language)

        out = retroflex.to_dict()
        asp_dict = aspiration.to_dict()
        length_dict = length_report.to_dict()
        out["aspiration_fidelity"] = asp_dict["aspiration_fidelity"]
        out["aspiration_n_expected"] = asp_dict["n_expected"]
        out["aspiration_n_collapsed"] = asp_dict["n_collapsed"]
        out["aspiration_per_position"] = asp_dict["per_position"]
        out["length_fidelity"] = length_dict["length_fidelity"]
        out["length_measured_ratio"] = length_dict["measured_long_short_ratio"]
        out["length_native_ratio"] = length_dict["native_ratio"]
        out["length_n_long"] = length_dict["n_long"]
        out["length_n_short"] = length_dict["n_short"]

        # Per-phoneme breakdown of retroflex report. Tamil-zha (ḻ) is the paper's
        # named sub-dimension and gets elevated to a top-level field; the full
        # per-phoneme map is also returned for ablation analysis.
        phoneme_stats = per_phoneme_breakdown(retroflex.per_position)
        out["retroflex_per_phoneme"] = phoneme_stats
        zha = phoneme_stats.get("ḻ")
        if zha is not None:
            out["zha_fidelity"] = zha.get("fidelity")
            out["zha_n_expected"] = zha.get("n_expected")
            out["zha_n_collapsed"] = zha.get("n_collapsed")
        return out


@app.local_entrypoint()
def score_dir(audio_dir: str, language: str = "te") -> None:
    """Score every .wav in ``audio_dir``. Writes one .psp.json per audio file.

    Caches: skips entries that already have a .psp.json sibling.
    Ground-truth text comes from the run's manifest.json (utt_id → text).
    """
    import time

    repo_root = Path(__file__).resolve().parent.parent
    audio_path = (repo_root / audio_dir).resolve() if not Path(audio_dir).is_absolute() else Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio dir not found: {audio_path}")

    manifest_path = audio_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {audio_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    test_set_name = manifest["test_set"]

    # Load golden set to get per-utt text
    golden_path = repo_root / "evaluation" / "golden_test_sets" / f"{test_set_name}.json"
    with open(golden_path) as f:
        golden = json.load(f)
    ref_lookup = {u["id"]: u["text"] for u in golden["utterances"]}

    _SCRIPT_FOR_LANG = {"te": "telugu", "hi": "devanagari", "ta": "tamil"}
    script = _SCRIPT_FOR_LANG.get(language)

    wavs = sorted(audio_path.glob("*.wav"))
    if not wavs:
        print(f"[score_dir_psp] no .wav files in {audio_path}")
        return

    scorer = PSPScorer()
    print(f"[score_dir_psp] {len(wavs)} files in {audio_path} (lang={language}, script={script})")

    n_skipped = 0
    n_scored = 0
    for wav_path in wavs:
        out_path = wav_path.with_suffix(".psp.json")
        if out_path.exists():
            n_skipped += 1
            continue

        stem = wav_path.stem
        parts = stem.rsplit("__", 1)
        utt_id = parts[0]
        text = ref_lookup.get(utt_id)
        if not text:
            print(f"  skip {wav_path.name}: no ref text for utt_id={utt_id}")
            continue

        t0 = time.time()
        audio_bytes = wav_path.read_bytes()
        result = scorer.score.remote(audio_bytes=audio_bytes, text=text, language=language, script=script)
        result["audio_file"] = wav_path.name
        result["utt_id"] = utt_id
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        n_scored += 1
        dt = time.time() - t0
        fid = result.get("retroflex_fidelity")
        fid_str = f"fid={fid:.3f}" if fid is not None else "fid=N/A"
        n_exp = result.get("n_expected", 0)
        n_col = result.get("n_collapsed", 0)
        print(
            f"  [{n_scored}/{len(wavs) - n_skipped}] {wav_path.name} "
            f"→ {fid_str} (expected={n_exp}, collapsed={n_col}, {dt:.1f}s)"
        )

    print(f"[score_dir_psp] done. scored={n_scored} skipped={n_skipped}")


@app.local_entrypoint()
def score_fad(audio_dir: str, language: str = "te") -> None:
    """Compute corpus-level FAD for all .wav files in ``audio_dir`` vs this
    language's native reference distribution.

    Emits ``{audio_dir}/fad.json`` with the scalar FAD + info dict. Unlike the
    per-utterance PSP probes, FAD is one number per (system, language) — it's
    a distributional metric.
    """
    import time

    repo_root = Path(__file__).resolve().parent.parent
    audio_path = (repo_root / audio_dir).resolve() if not Path(audio_dir).is_absolute() else Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio dir not found: {audio_path}")

    wavs = sorted(audio_path.glob("*.wav"))
    if not wavs:
        print(f"[score_fad] no .wav files in {audio_path}")
        return
    if len(wavs) < 2:
        raise RuntimeError(f"FAD needs ≥2 utterances; {audio_path} has {len(wavs)}")

    scorer = PSPScorer()
    print(f"[score_fad] embedding {len(wavs)} utterances (lang={language})")

    # Extract per-utterance embeddings in parallel via Modal's .map().
    embeddings: list[list[float]] = []
    t0 = time.time()
    args = [(wav.read_bytes(), language) for wav in wavs]
    # Use starmap if the method accepts multiple args; we use .map with tuples
    for i, result in enumerate(scorer.embed_clip_for_fad.starmap(args)):
        embeddings.append(result["embedding"])
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(wavs)}] embedded ({dt:.1f}s elapsed)")

    # Final FAD computation (fast — pure numpy + scipy).
    print(f"[score_fad] computing FAD vs native corpus...")
    fad_result = scorer.compute_corpus_fad.remote(embeddings, language)

    out_path = audio_path / "fad.json"
    out_path.write_text(json.dumps(fad_result, indent=2))
    print(f"[score_fad] done. FAD = {fad_result['fad']:.3f}")
    print(f"[score_fad]   n_gen={fad_result['n_gen']} n_ref={fad_result['n_ref']}")
    print(f"[score_fad]   mean_norm={fad_result['mean_norm']:.3f} trace_term={fad_result['trace_term']:.3f}")
    print(f"[score_fad] → {out_path}")


@app.local_entrypoint()
def score_psd(audio_dir: str, language: str = "te") -> None:
    """Compute corpus-level PSD for all .wav files in ``audio_dir``.

    Emits ``{audio_dir}/psd.json`` with the scalar PSD + feature-means info.
    Like FAD, PSD is one number per (system, language) — distributional.
    """
    import time

    repo_root = Path(__file__).resolve().parent.parent
    audio_path = (repo_root / audio_dir).resolve() if not Path(audio_dir).is_absolute() else Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio dir not found: {audio_path}")

    wavs = sorted(audio_path.glob("*.wav"))
    if len(wavs) < 2:
        raise RuntimeError(f"PSD needs ≥2 utterances; {audio_path} has {len(wavs)}")

    scorer = PSPScorer()
    print(f"[score_psd] extracting prosodic features for {len(wavs)} utts (lang={language})")

    features: list[list[float]] = []
    n_nan = 0
    t0 = time.time()
    # .map() unwraps each item as a single positional arg — so we pass bytes
    # directly, not a tuple. (starmap would unwrap a tuple into *args.)
    args = [wav.read_bytes() for wav in wavs]
    for i, result in enumerate(scorer.extract_psd_features.map(args)):
        if result["has_nan"]:
            n_nan += 1
        features.append(result["features"])
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(wavs)}] extracted ({dt:.1f}s elapsed, nan_count={n_nan})")

    print(f"[score_psd] computing PSD vs native...")
    psd_result = scorer.compute_corpus_psd.remote(features, language)

    out_path = audio_path / "psd.json"
    out_path.write_text(json.dumps(psd_result, indent=2))
    print(f"[score_psd] done. PSD = {psd_result['psd']:.3f}")
    print(f"[score_psd]   n_gen={psd_result['n_gen']} n_ref={psd_result['n_ref']} (dropped_gen={psd_result.get('n_gen_dropped', 0)})")
    print(f"[score_psd] → {out_path}")
