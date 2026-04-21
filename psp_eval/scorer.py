"""User-facing PSP scoring API.

This is the ergonomic wrapper. All heavy lifting happens in the research
modules under ``evaluation/``.

**Design: local vs Modal.** The package supports two execution modes:

- **Local** (default if Modal credentials absent): loads models + centroids
  locally, runs scoring on CPU or local GPU. Suitable for quick single-clip
  evaluation. ~5-10s per clip on CPU, ~1s on GPU.
- **Modal** (default if Modal credentials present + --backend=modal): sends
  audio to Modal containers with pre-loaded models. ~10-30x faster for
  directory-scale evaluation; costs Modal compute credits.

The first call triggers centroid download from Hugging Face
(Praxel/psp-native-centroids) if not cached locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SUPPORTED_LANGUAGES = ("te", "hi", "ta")
CENTROIDS_HF_REPO = "Praxel/psp-native-centroids"  # TBD — release with paper
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "psp_eval"


def score_clip(
    audio_path: str | Path,
    text: str,
    language: str,
    backend: str = "local",
) -> dict[str, Any]:
    """Score a single audio clip against its expected transcript.

    Returns a dict with per-dimension fidelity scores:
    ``retroflex_fidelity``, ``aspiration_fidelity``, ``length_fidelity``,
    ``zha_fidelity``, plus counts + per-position breakdowns.

    FAD and PSD require corpus-level computation — use ``score_directory``.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"language must be one of {SUPPORTED_LANGUAGES}; got {language!r}")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    if backend == "modal":
        return _score_clip_modal(audio_path, text, language)
    elif backend == "local":
        return _score_clip_local(audio_path, text, language)
    else:
        raise ValueError(f"backend must be 'local' or 'modal'; got {backend!r}")


def score_directory(
    audio_dir: str | Path,
    language: str,
    backend: str = "local",
    include_corpus_metrics: bool = True,
) -> dict[str, Any]:
    """Score every .wav file in a directory, aggregate per-dimension fidelity,
    and optionally compute corpus-level FAD + PSD.

    Transcripts are resolved in order:
      1. ``manifest.json`` in the directory with an ``utterances: [{id, text}]`` schema
      2. Sibling ``.txt`` file per wav (same stem)

    Returns a single dict with aggregate PSP + (if requested) FAD + PSD.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"language must be one of {SUPPORTED_LANGUAGES}")

    audio_dir = Path(audio_dir)
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise NotADirectoryError(audio_dir)

    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"No .wav files in {audio_dir}")

    transcripts = _resolve_transcripts(audio_dir, wavs)

    per_clip = []
    for wav_path in wavs:
        text = transcripts.get(wav_path.stem)
        if text is None:
            print(f"  [psp_eval] skipping {wav_path.name}: no transcript")
            continue
        scores = score_clip(wav_path, text, language, backend=backend)
        scores["audio_file"] = wav_path.name
        per_clip.append(scores)

    if not per_clip:
        raise RuntimeError(f"No clips could be scored (no matching transcripts in {audio_dir})")

    agg = _aggregate(per_clip)

    if include_corpus_metrics:
        if backend == "modal":
            agg["fad"] = _compute_fad_modal(audio_dir, language)
            agg["psd"] = _compute_psd_modal(audio_dir, language)
        else:
            # Local corpus metrics require loading XLS-R + librosa for all wavs.
            # Skip with a warning on local backend to keep single-clip use fast.
            agg["fad"] = None
            agg["psd"] = None
            agg["corpus_metrics_skipped"] = (
                "FAD and PSD require backend='modal' (utterance-level XLS-R is too slow locally)"
            )

    return agg


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_transcripts(audio_dir: Path, wavs: list[Path]) -> dict[str, str]:
    """Resolve wav_stem → text from manifest.json or sibling .txt files."""
    import json

    transcripts: dict[str, str] = {}
    manifest_path = audio_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        utts = manifest.get("utterances", [])
        ref_lookup = {u["id"]: u["text"] for u in utts if "id" in u and "text" in u}
        # Match wav stems (with / without voice suffix) to utterance IDs
        for wav in wavs:
            stem = wav.stem
            # Praxy convention: <utt_id>__<voice>.wav
            utt_id = stem.rsplit("__", 1)[0]
            if utt_id in ref_lookup:
                transcripts[stem] = ref_lookup[utt_id]

    # Fill in from sibling .txt files where manifest didn't cover
    for wav in wavs:
        if wav.stem in transcripts:
            continue
        txt_path = wav.with_suffix(".txt")
        if txt_path.exists():
            transcripts[wav.stem] = txt_path.read_text(encoding="utf-8").strip()

    return transcripts


def _score_clip_local(audio_path: Path, text: str, language: str) -> dict[str, Any]:
    """Local scoring — loads models + centroids in-process."""
    from evaluation.psp import AspirationProbe, LengthProbe, per_phoneme_breakdown

    _LANG_ALIGN_MODEL = {
        "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
        "hi": "ai4bharat/indicwav2vec-hindi",
        "ta": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
    }
    _SCRIPT_FOR_LANG = {"te": "telugu", "hi": "devanagari", "ta": "tamil"}

    centroids = _ensure_centroids_downloaded(language)

    probe = AspirationProbe(align_model=_LANG_ALIGN_MODEL[language])
    probe.load()
    for phoneme, embs in centroids.items():
        for e in embs:
            probe.add_native_reference(phoneme, e)

    audio_bytes = audio_path.read_bytes()
    script = _SCRIPT_FOR_LANG[language]

    retroflex = probe.score(audio_bytes, text, script=script, language_code=language)
    aspiration = probe.score_aspiration(audio_bytes, text, script=script, language_code=language)

    length_probe = LengthProbe()
    length_probe._loaded = True
    length_probe._align_model = probe._align_model
    length_probe._align_processor = probe._align_processor
    length_probe._embed_model = probe._embed_model
    length_probe._embed_processor = probe._embed_processor
    length_probe.device = probe.device
    length_rep = length_probe.score_length(audio_bytes, text, script=script, language_code=language)

    out = retroflex.to_dict()
    phoneme_stats = per_phoneme_breakdown(retroflex.per_position)
    asp_dict = aspiration.to_dict()
    out["aspiration_fidelity"] = asp_dict["aspiration_fidelity"]
    out["length_fidelity"] = length_rep.length_fidelity
    zha = phoneme_stats.get("ḻ")
    if zha is not None:
        out["zha_fidelity"] = zha["fidelity"]
    return out


def _score_clip_modal(audio_path: Path, text: str, language: str) -> dict[str, Any]:
    """Modal-backed scoring — delegates to evaluation/modal_psp.py."""
    import modal

    from evaluation.modal_psp import PSPScorer

    _SCRIPT_FOR_LANG = {"te": "telugu", "hi": "devanagari", "ta": "tamil"}
    scorer = PSPScorer()
    audio_bytes = audio_path.read_bytes()
    return scorer.score.remote(
        audio_bytes=audio_bytes, text=text,
        language=language, script=_SCRIPT_FOR_LANG[language],
    )


def _compute_fad_modal(audio_dir: Path, language: str) -> float | None:
    """Call the Modal FAD pipeline on a directory. Returns scalar FAD."""
    from evaluation.modal_psp import PSPScorer

    scorer = PSPScorer()
    wavs = sorted(audio_dir.glob("*.wav"))
    embeddings = []
    for wav in wavs:
        result = scorer.embed_clip_for_fad.remote(
            audio_bytes=wav.read_bytes(), language=language,
        )
        embeddings.append(result["embedding"])
    if len(embeddings) < 2:
        return None
    return scorer.compute_corpus_fad.remote(embeddings, language)["fad"]


def _compute_psd_modal(audio_dir: Path, language: str) -> float | None:
    """Call the Modal PSD pipeline on a directory. Returns scalar PSD."""
    from evaluation.modal_psp import PSPScorer

    scorer = PSPScorer()
    wavs = sorted(audio_dir.glob("*.wav"))
    features = []
    for wav in wavs:
        result = scorer.extract_psd_features.remote(audio_bytes=wav.read_bytes())
        features.append(result["features"])
    if len(features) < 2:
        return None
    return scorer.compute_corpus_psd.remote(features, language)["psd"]


def _ensure_centroids_downloaded(language: str) -> dict[str, list]:
    """Download centroid pickle from HF if not cached; load and return.

    Post-release this fetches from huggingface.co/datasets/Praxel/psp-native-centroids.
    Until release, raises an informative error pointing at the local path.
    """
    import pickle

    cache_path = _DEFAULT_CACHE_DIR / f"{language}_refs.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(
        f"Centroids not found at {cache_path}. Pre-release builds require manual setup:\n"
        f"  mkdir -p {_DEFAULT_CACHE_DIR}\n"
        f"  modal volume get praxy-voice-vol /psp_refs/{language}_refs.pkl {cache_path}\n"
        f"Post-release, they'll auto-download from huggingface.co/datasets/{CENTROIDS_HF_REPO}"
    )


def _aggregate(per_clip: list[dict]) -> dict:
    """Mean each fidelity field; total expected/collapsed counts."""
    import statistics

    def _mean(key: str) -> float | None:
        vals = [c.get(key) for c in per_clip if c.get(key) is not None]
        if not vals:
            return None
        return round(statistics.mean(vals), 4)

    def _total(key: str) -> int:
        return sum(c.get(key, 0) or 0 for c in per_clip)

    return {
        "n_clips": len(per_clip),
        "retroflex_fidelity": _mean("retroflex_fidelity"),
        "retroflex_n_expected": _total("n_expected"),
        "retroflex_n_collapsed": _total("n_collapsed"),
        "aspiration_fidelity": _mean("aspiration_fidelity"),
        "length_fidelity": _mean("length_fidelity"),
        "zha_fidelity": _mean("zha_fidelity"),
    }


def _cli_main() -> None:
    """CLI entrypoint for `psp-score` console script.

        psp-score --audio-dir path/to/wavs --language te
    """
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Score a directory of TTS outputs with PSP.")
    ap.add_argument("--audio-dir", required=True, help="Directory of .wav files to score.")
    ap.add_argument("--language", required=True, choices=SUPPORTED_LANGUAGES)
    ap.add_argument("--backend", default="local", choices=["local", "modal"])
    ap.add_argument("--no-corpus-metrics", action="store_true",
                    help="Skip FAD / PSD (which require backend='modal').")
    args = ap.parse_args()

    scores = score_directory(
        audio_dir=args.audio_dir,
        language=args.language,
        backend=args.backend,
        include_corpus_metrics=not args.no_corpus_metrics,
    )
    print(json.dumps(scores, indent=2, ensure_ascii=False))
