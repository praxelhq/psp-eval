"""Bootstrap PSP native reference centroids from IndicTTS Telugu clips.

Run once per language pair. Output is a pickle of {phoneme: list[np.ndarray]}
saved to the praxy-voice-vol Modal volume. The scorecard loads this at eval
time so PSP can produce retroflex-fidelity scores.

    modal run evaluation/psp_bootstrap.py::bootstrap --language te --n-clips 80
"""

from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import modal

APP_NAME = "praxy-psp-bootstrap"
MODEL_VOLUME = "praxy-voice-vol"
DATA_VOLUME = "praxy-data"

app = modal.App(APP_NAME)
model_vol = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)

psp_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        # torch>=2.6 required: CVE-2025-32434 means transformers now refuses
        # to load legacy .bin checkpoints (like anuragshas/wav2vec2-large-
        # xlsr-53-telugu which isn't safetensors) on older torch. 2.6 unlocks
        # torch.load for this. Other Praxy components stay on torch 2.4;
        # PSP bootstrap is a separate image so this upgrade is isolated.
        "torch>=2.6,<2.8", "torchaudio>=2.6,<2.8",
        "numpy<2", "scipy", "soundfile", "librosa",
        "huggingface_hub>=0.25", "transformers>=4.46",
        "accelerate>=0.34", "sentencepiece",
        "indic-transliteration", "pyarrow>=17",
        "pyctcdecode",  # required by ai4bharat/indicwav2vec-hindi processor
    )
    .add_local_python_source("evaluation", "praxy")
)


# Phonemes we collect centroids for (retroflex + dental-collapse targets).
# Mirrors evaluation/psp.py RETROFLEX_PHONEMES + DENTAL_COLLAPSE_TARGETS values.
_TARGET_PHONEMES = {"ṭ", "ṭh", "ḍ", "ḍh", "ṇ", "ṣ", "ḷ", "ḻ",
                    "t", "th", "d", "dh", "n", "s", "l"}

_LANG_ALIGN_MODEL = {
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "ai4bharat/indicwav2vec-hindi",
    "ta": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
}


@app.function(
    image=psp_image,
    gpu="A10G",
    volumes={"/cache": model_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("praxy-hf")],
    timeout=3600,
)
def bootstrap(
    language: str = "te",
    n_clips: int = 80,
    embed_layer: int = 9,
    output_suffix: str = "",
) -> dict:
    """Sample N native clips from the manifest for ``language`` and populate
    RetroflexProbe's native-reference dict. Serialize the result.

    Returns a summary dict with per-phoneme embedding counts.
    """
    import os
    import random
    import sys

    import numpy as np
    import soundfile as sf
    from collections import defaultdict

    os.environ.setdefault("HF_HOME", "/cache/hf")

    # Inline the PSP module path — we mount the eval dir at /root/psp and use
    # it plus praxy/linguistics from the project tree.
    sys.path.insert(0, "/root")

    # We can't mount local Python modules through a Modal function by default.
    # Instead, include minimal inlined logic: BUPS romanization + phoneme
    # positions + XLS-R embedding extraction — mirroring evaluation/psp.py.
    # The exact scorecard path uses the full psp.py at eval time, but for
    # centroid bootstrap we only need embeddings per phoneme span.

    import torch
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    from transformers import (
        AutoModelForCTC,
        AutoProcessor,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Model,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    align_repo = _LANG_ALIGN_MODEL[language]
    print(f"[bootstrap] loading align model {align_repo}")
    align_processor = AutoProcessor.from_pretrained(align_repo)
    align_model = AutoModelForCTC.from_pretrained(align_repo).to(device)
    align_model.eval()

    print("[bootstrap] loading XLS-R embed model")
    embed_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    embed_model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-xls-r-300m", output_hidden_states=True
    ).to(device)
    embed_model.eval()

    # Pick N native clips from the train manifest — source=indictts_telugu
    # (studio-quality native speakers, not Shrutilipi's news-reader noise).
    manifest_path = Path("/data/manifests/train.jsonl")
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    # Pick the cleanest native source per language. Hindi has no IndicTTS
    # subset in our manifest — Rasa (studio-quality, native speakers) is the
    # best substitute. Telugu/Tamil use their IndicTTS subsets.
    _NATIVE_SOURCE_BY_LANG = {
        "te": "indictts_telugu",
        "ta": "indictts_tamil",
        "hi": "rasa",
    }
    target_source = _NATIVE_SOURCE_BY_LANG.get(language)
    candidate_rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["language_code"] != language:
                continue
            if target_source and r.get("source") != target_source:
                continue
            candidate_rows.append(r)

    if not candidate_rows:
        raise RuntimeError(f"No native source rows for language={language}")
    print(f"[bootstrap] {len(candidate_rows)} candidate native rows; sampling {n_clips}")
    rng = random.Random(1337)
    selected = rng.sample(candidate_rows, min(n_clips, len(candidate_rows)))

    # Script → ISO-15919 romanization; we locate target phonemes in the
    # ISO form (which matches alignment-model character inventory better
    # than raw Brahmic for many Indic CTC checkpoints).
    _SANS = {"te": sanscript.TELUGU, "hi": sanscript.DEVANAGARI, "ta": sanscript.TAMIL}

    def _load_wav(audio_path: str) -> np.ndarray:
        if audio_path.startswith("parquet-ref://"):
            # Resolve parquet-ref
            import pyarrow.parquet as pq
            body = audio_path[len("parquet-ref://"):]
            parquet_path, frag = body.rsplit("#row=", 1)
            row_idx = int(frag)
            table = pq.read_table(parquet_path)
            col = "audio_filepath" if "audio_filepath" in table.column_names else "audio"
            ast = table[col][row_idx].as_py()
            audio_bytes = ast.get("bytes")
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        else:
            audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        return audio

    # Use the native-script retroflex table from psp.py — CTC aligners for
    # Indic langs emit native-script tokens, not ISO-romanized ones. Matching
    # in native script is the only path that actually works.
    from evaluation.psp import RETROFLEX_CHARS_BY_SCRIPT

    _SCRIPT_FOR_LANG = {"te": "telugu", "hi": "devanagari", "ta": "tamil"}
    script = _SCRIPT_FOR_LANG[language]
    retroflex_table = RETROFLEX_CHARS_BY_SCRIPT[script]  # {native_char: bups_phoneme}

    # Also include dental-collapse target chars in native script so we can
    # extract those centroids too. Dental chars per script:
    _DENTAL_CHARS_BY_SCRIPT = {
        "telugu":     {"త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n", "స": "s", "ల": "l"},
        "devanagari": {"त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "स": "s", "ल": "l"},
        "tamil":      {"த": "t", "ந": "n", "ன": "n", "ஸ": "s", "ல": "l"},
    }
    # Aspiration-pair chars: aspirated vs unaspirated stops. Hindi contrasts
    # these phonemically; Tamil has none; Telugu has them but rarely in modern
    # usage. We collect both members of each pair for centroid-based probes.
    _ASPIRATION_CHARS_BY_SCRIPT = {
        "devanagari": {
            "क": "k", "ख": "kh", "ग": "g", "घ": "gh",
            "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
            "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
        },
        "telugu": {
            "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh",
            "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh",
            "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh",
        },
        "tamil": {
            # Tamil has no phonemic aspirated stops; include canonical letters
            # for completeness so scoring has substitute centroids if needed.
            "க": "k", "ச": "ch", "ஜ": "j", "ப": "p",
        },
    }
    dental_table = _DENTAL_CHARS_BY_SCRIPT[script]
    aspiration_table = _ASPIRATION_CHARS_BY_SCRIPT.get(script, {})
    char_to_phoneme: dict[str, str] = {**retroflex_table, **dental_table, **aspiration_table}

    # Diagnostic: log which target chars are actually in the aligner's vocab.
    vocab = align_processor.tokenizer.get_vocab()
    hits = [ch for ch in char_to_phoneme if ch in vocab]
    misses = [ch for ch in char_to_phoneme if ch not in vocab]
    print(f"[bootstrap] aligner vocab hits ({len(hits)}): {hits}")
    if misses:
        print(f"[bootstrap] aligner vocab MISSES ({len(misses)}): {misses}")

    def _align_and_extract(audio_16k: np.ndarray, text: str) -> list[tuple[str, np.ndarray]]:
        """Returns list of (bups_phoneme, embedding) for target native-script
        chars in ``text`` that we can align against the CTC output.
        """
        # Early-out if text has no target characters.
        if not any(ch in char_to_phoneme for ch in text):
            return []

        inputs = align_processor(audio_16k, sampling_rate=16_000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = align_model(input_values).logits[0]  # (T, V)

        pred_ids = logits.argmax(dim=-1).cpu().numpy()
        id_to_char = {v: k for k, v in vocab.items()}

        # Walk frames, collapsing blanks/repeats: when a target native char is
        # emitted, open a span; when a different char (or blank) appears, close it.
        phoneme_spans: list[tuple[str, int, int]] = []  # (native_char, f_start, f_end)
        in_span = False
        span_start = 0
        span_char: str | None = None
        for frame_idx, tid in enumerate(pred_ids):
            ch = id_to_char.get(int(tid), "")
            if ch in char_to_phoneme:
                if not in_span:
                    in_span = True
                    span_start = frame_idx
                    span_char = ch
                elif ch != span_char:
                    phoneme_spans.append((span_char, span_start, frame_idx))  # type: ignore[arg-type]
                    span_start = frame_idx
                    span_char = ch
            else:
                if in_span:
                    phoneme_spans.append((span_char, span_start, frame_idx))  # type: ignore[arg-type]
                    in_span = False
        if in_span:
            phoneme_spans.append((span_char, span_start, len(pred_ids)))  # type: ignore[arg-type]

        if not phoneme_spans:
            return []

        # XLS-R embedding for the full clip; slice per span via frame indices.
        feats = embed_processor(audio_16k, sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            out = embed_model(feats.input_values.to(device))
        layer_9 = out.hidden_states[9][0].cpu().numpy()  # (T, D)
        scale = layer_9.shape[0] / max(1, len(pred_ids))

        results: list[tuple[str, np.ndarray]] = []
        for native_ch, f0, f1 in phoneme_spans:
            e0, e1 = int(f0 * scale), int(f1 * scale)
            if e1 <= e0:
                continue
            emb_slice = layer_9[e0:e1]
            if emb_slice.size == 0:
                continue
            emb = emb_slice.mean(axis=0)
            bups_phoneme = char_to_phoneme[native_ch]
            results.append((bups_phoneme, emb))
        return results

    # Collect embeddings per phoneme
    refs: dict[str, list[np.ndarray]] = defaultdict(list)
    kept = 0
    for i, row in enumerate(selected):
        try:
            wav = _load_wav(row["audio_path"])
        except Exception as e:
            print(f"[bootstrap]  skip {i}: load error {e}")
            continue
        if len(wav) < 16_000 * 1.0 or len(wav) > 16_000 * 15:
            continue  # too short/long
        try:
            pairs = _align_and_extract(wav, row["transcript"])
        except Exception as e:
            print(f"[bootstrap]  skip {i}: align error {e}")
            continue
        for ph, emb in pairs:
            refs[ph].append(emb)
        kept += 1
        if kept % 10 == 0:
            print(f"[bootstrap] {kept}/{len(selected)} clips processed; "
                  f"phoneme counts: {[(p, len(v)) for p, v in sorted(refs.items())]}")

    # Save
    out_path = Path(f"/cache/psp_refs/{language}_refs.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({ph: [e.astype(np.float32) for e in embs] for ph, embs in refs.items()}, f)
    model_vol.commit()

    summary = {
        "language": language,
        "clips_processed": kept,
        "phoneme_counts": {p: len(v) for p, v in sorted(refs.items())},
        "output_path": str(out_path),
    }
    print(f"[bootstrap] DONE {summary}")
    return summary


@app.function(
    image=psp_image,
    gpu="A10G",
    volumes={"/cache": model_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("praxy-hf")],
    timeout=7200,  # 2 hrs — FAD needs ~1000 clips, longer than centroid bootstrap
)
def bootstrap_fad_natives(language: str = "te", n_clips: int = 1000) -> dict:
    """Extract utterance-level XLS-R layer-9 embeddings from native reference
    audio, for FAD computation.

    Output: ``/cache/psp_refs/{language}_fad_natives.pkl`` containing a single
    ``np.ndarray`` of shape ``(N, D=1024)``. This is the reference distribution
    FAD compares generated TTS corpora against.

    We sample ``n_clips`` from the same native source used for PSP centroids
    (indictts_telugu / indictts_tamil / rasa-hindi). 1000 clips is enough for
    a stable 1024-D Gaussian covariance (~rank-full for 1024 dimensions).

    Runs as a separate Modal function (not merged with ``bootstrap``) because
    FAD and centroid bootstraps may be rerun independently at different scales.
    """
    import os
    import random

    import numpy as np
    import soundfile as sf

    os.environ.setdefault("HF_HOME", "/cache/hf")

    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fad-bootstrap] loading XLS-R embed model on {device}")
    embed_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    embed_model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-xls-r-300m", output_hidden_states=True
    ).to(device)
    embed_model.eval()

    _NATIVE_SOURCE_BY_LANG = {"te": "indictts_telugu", "ta": "indictts_tamil", "hi": "rasa"}
    target_source = _NATIVE_SOURCE_BY_LANG[language]

    manifest_path = Path("/data/manifests/train.jsonl")
    candidate_rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["language_code"] == language and r.get("source") == target_source:
                candidate_rows.append(r)

    if not candidate_rows:
        raise RuntimeError(f"No native rows for language={language} source={target_source}")
    print(f"[fad-bootstrap] {len(candidate_rows)} candidate rows; sampling {n_clips}")
    rng = random.Random(1337)
    selected = rng.sample(candidate_rows, min(n_clips, len(candidate_rows)))

    def _load_wav(audio_path: str) -> np.ndarray:
        if audio_path.startswith("parquet-ref://"):
            import io as _io

            import pyarrow.parquet as pq
            body = audio_path[len("parquet-ref://"):]
            parquet_path, frag = body.rsplit("#row=", 1)
            row_idx = int(frag)
            table = pq.read_table(parquet_path)
            col = "audio_filepath" if "audio_filepath" in table.column_names else "audio"
            ast = table[col][row_idx].as_py()
            audio_bytes = ast.get("bytes")
            audio, sr = sf.read(_io.BytesIO(audio_bytes), dtype="float32")
        else:
            audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        return audio

    embeddings: list[np.ndarray] = []
    kept = 0
    for i, row in enumerate(selected):
        try:
            wav = _load_wav(row["audio_path"])
        except Exception as e:
            print(f"[fad-bootstrap]  skip {i}: load error {e}")
            continue
        if len(wav) < 16_000 * 1.0 or len(wav) > 16_000 * 20:
            continue

        try:
            inputs = embed_processor(wav, sampling_rate=16_000, return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                out = embed_model(inputs, output_hidden_states=True)
            hs = out.hidden_states[9]  # (1, T, D=1024)
            emb = hs.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            embeddings.append(emb)
            kept += 1
            if kept % 50 == 0:
                print(f"[fad-bootstrap] {kept}/{len(selected)} clips embedded")
        except Exception as e:
            print(f"[fad-bootstrap]  skip {i}: embed error {e}")
            continue

    if not embeddings:
        raise RuntimeError("No embeddings extracted")

    matrix = np.stack(embeddings, axis=0)
    print(f"[fad-bootstrap] final matrix shape: {matrix.shape}")

    out_path = Path(f"/cache/psp_refs/{language}_fad_natives.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(matrix, f)
    model_vol.commit()

    summary = {
        "language": language,
        "clips_processed": kept,
        "shape": list(matrix.shape),
        "output_path": str(out_path),
    }
    print(f"[fad-bootstrap] DONE {summary}")
    return summary


@app.function(
    image=psp_image,
    volumes={"/cache": model_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("praxy-hf")],
    timeout=3600,  # CPU-only (librosa) — no GPU needed for PSD features
)
def bootstrap_psd_natives(language: str = "te", n_clips: int = 500) -> dict:
    """Extract prosodic feature vectors from native reference audio.

    Output: ``/cache/psp_refs/{language}_psd_natives.pkl`` containing an
    ``np.ndarray`` of shape ``(N, 5)`` — one prosodic feature vector per clip.
    The 5 dims: [mean_logf0, std_logf0, range_logf0, onset_rate, nPVI]. See
    ``evaluation/psd.extract_prosodic_features`` for details.

    No GPU needed; librosa-based pyin + onset detection is CPU-bound.
    """
    import os
    import random

    import numpy as np
    import soundfile as sf

    os.environ.setdefault("HF_HOME", "/cache/hf")

    import sys
    sys.path.insert(0, "/root")

    from evaluation.psd import extract_prosodic_features

    _NATIVE_SOURCE_BY_LANG = {"te": "indictts_telugu", "ta": "indictts_tamil", "hi": "rasa"}
    target_source = _NATIVE_SOURCE_BY_LANG[language]

    manifest_path = Path("/data/manifests/train.jsonl")
    candidate_rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["language_code"] == language and r.get("source") == target_source:
                candidate_rows.append(r)

    if not candidate_rows:
        raise RuntimeError(f"No native rows for language={language} source={target_source}")
    print(f"[psd-bootstrap] {len(candidate_rows)} candidate rows; sampling {n_clips}")
    rng = random.Random(1337)
    selected = rng.sample(candidate_rows, min(n_clips, len(candidate_rows)))

    def _load_wav(audio_path: str) -> np.ndarray:
        if audio_path.startswith("parquet-ref://"):
            import io as _io
            import pyarrow.parquet as pq
            body = audio_path[len("parquet-ref://"):]
            parquet_path, frag = body.rsplit("#row=", 1)
            row_idx = int(frag)
            table = pq.read_table(parquet_path)
            col = "audio_filepath" if "audio_filepath" in table.column_names else "audio"
            ast = table[col][row_idx].as_py()
            audio_bytes = ast.get("bytes")
            audio, sr = sf.read(_io.BytesIO(audio_bytes), dtype="float32")
        else:
            audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        return audio

    features = []
    kept = 0
    skipped_nan = 0
    for i, row in enumerate(selected):
        try:
            wav = _load_wav(row["audio_path"])
        except Exception as e:
            print(f"[psd-bootstrap]  skip {i}: load error {e}")
            continue
        if len(wav) < 16_000 * 1.0 or len(wav) > 16_000 * 20:
            continue
        feat = extract_prosodic_features(wav)
        if np.any(np.isnan(feat)):
            skipped_nan += 1
            continue
        features.append(feat)
        kept += 1
        if kept % 50 == 0:
            print(f"[psd-bootstrap] {kept}/{len(selected)} clips processed (skipped_nan={skipped_nan})")

    if not features:
        raise RuntimeError("No clean features extracted")

    matrix = np.stack(features, axis=0)
    print(f"[psd-bootstrap] final matrix shape: {matrix.shape}")
    print(f"[psd-bootstrap] feature means: mean_logf0={matrix[:,0].mean():.2f} "
          f"std_logf0={matrix[:,1].mean():.2f} range={matrix[:,2].mean():.2f} "
          f"onset_rate={matrix[:,3].mean():.2f}/s nPVI={matrix[:,4].mean():.1f}")

    out_path = Path(f"/cache/psp_refs/{language}_psd_natives.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(matrix, f)
    model_vol.commit()

    summary = {
        "language": language,
        "clips_processed": kept,
        "clips_skipped_nan": skipped_nan,
        "shape": list(matrix.shape),
        "feature_means": [float(x) for x in matrix.mean(axis=0)],
        "output_path": str(out_path),
    }
    print(f"[psd-bootstrap] DONE {summary}")
    return summary


@app.function(
    image=psp_image,
    gpu="A10G",
    volumes={"/cache": model_vol, "/data": data_vol},
    secrets=[modal.Secret.from_name("praxy-hf")],
    timeout=3600,
)
def sanity_check_native(language: str = "te", n_clips: int = 50, seed: int = 999) -> dict:
    """Score held-out native clips through the full PSP pipeline.

    This is the strongest possible internal calibration signal: if our metrics
    return near-perfect fidelity on native audio (by construction native IS
    the target), then we know the metric doesn't have a systematic "always
    collapse" bias. If native audio DOESN'T score ≈ 1.0, we have a bug.

    Uses seed=999 (vs bootstrap's seed=1337) so held-out clips are disjoint
    from the centroid bootstrap — important for FAD/PSD since generating and
    reference distributions must not be identical.

    Output: a dict with aggregate per-dim fidelity + FAD + PSD for native audio.
    Saved to /cache/psp_refs/{language}_sanity.json on the volume.
    """
    import os
    import random

    import numpy as np
    import soundfile as sf

    os.environ.setdefault("HF_HOME", "/cache/hf")

    import sys
    sys.path.insert(0, "/root")

    from evaluation.psp import AspirationProbe, LengthProbe, per_phoneme_breakdown
    from evaluation.fad import compute_fad
    from evaluation.psd import compute_psd, extract_prosodic_features

    _ALIGN_MODEL_BY_LANG = {
        "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
        "hi": "ai4bharat/indicwav2vec-hindi",
        "ta": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
    }
    _SCRIPT_FOR_LANG = {"te": "telugu", "hi": "devanagari", "ta": "tamil"}
    _NATIVE_SOURCE_BY_LANG = {"te": "indictts_telugu", "ta": "indictts_tamil", "hi": "rasa"}

    # Load probes + centroids + FAD native embeddings + PSD native features
    print(f"[sanity] loading PSP infrastructure for {language}")
    probe = AspirationProbe(align_model=_ALIGN_MODEL_BY_LANG[language])
    probe.load()

    import pickle as _pk
    with open(f"/cache/psp_refs/{language}_refs.pkl", "rb") as f:
        refs = _pk.load(f)
    for phoneme, embs in refs.items():
        for e in embs:
            probe.add_native_reference(phoneme, e)

    with open(f"/cache/psp_refs/{language}_fad_natives.pkl", "rb") as f:
        fad_natives = _pk.load(f)
    with open(f"/cache/psp_refs/{language}_psd_natives.pkl", "rb") as f:
        psd_natives = _pk.load(f)

    length_probe = LengthProbe()
    length_probe._loaded = True
    length_probe._align_model = probe._align_model
    length_probe._align_processor = probe._align_processor
    length_probe._embed_model = probe._embed_model
    length_probe._embed_processor = probe._embed_processor
    length_probe.device = probe.device

    # Pick held-out native clips (seed=999 disjoint from bootstrap's seed=1337)
    target_source = _NATIVE_SOURCE_BY_LANG[language]
    manifest_path = Path("/data/manifests/train.jsonl")
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("language_code") == language and r.get("source") == target_source:
                rows.append(r)
    rng = random.Random(seed)
    selected = rng.sample(rows, min(n_clips, len(rows)))
    print(f"[sanity] scoring {len(selected)} held-out native clips")

    def _load_wav(audio_path):
        if audio_path.startswith("parquet-ref://"):
            import io as _io
            import pyarrow.parquet as pq
            body = audio_path[len("parquet-ref://"):]
            parquet_path, frag = body.rsplit("#row=", 1)
            row_idx = int(frag)
            table = pq.read_table(parquet_path)
            col = "audio_filepath" if "audio_filepath" in table.column_names else "audio"
            ast = table[col][row_idx].as_py()
            audio, sr = sf.read(_io.BytesIO(ast.get("bytes")), dtype="float32")
        else:
            audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        return audio

    # Score each clip; accumulate per-dim stats + corpus-level features
    script = _SCRIPT_FOR_LANG[language]
    rr_fids, af_fids, lf_fids, zf_fids = [], [], [], []
    rr_exp_total = rr_col_total = 0
    af_exp_total = af_col_total = 0
    zf_exp_total = zf_col_total = 0
    utt_embs = []
    psd_feats = []

    import soundfile as _sf
    import io as _io

    for i, row in enumerate(selected):
        try:
            audio = _load_wav(row["audio_path"])
        except Exception as e:
            print(f"[sanity] skip {i}: load error {e}")
            continue
        if len(audio) < 16_000 * 1.0 or len(audio) > 16_000 * 20:
            continue
        text = row.get("transcript", "").strip()
        if not text:
            continue

        # Convert to bytes for probe API (probes expect audio_bytes)
        buf = _io.BytesIO()
        _sf.write(buf, audio, 16_000, format="WAV")
        audio_bytes = buf.getvalue()

        try:
            rr_rep = probe.score(audio_bytes, text, script=script, language_code=language)
            af_rep = probe.score_aspiration(audio_bytes, text, script=script, language_code=language)
            lf_rep = length_probe.score_length(audio_bytes, text, script=script, language_code=language)
        except Exception as e:
            print(f"[sanity] skip {i}: score error {e}")
            continue

        if rr_rep.retroflex_fidelity is not None and rr_rep.n_expected > 0:
            rr_fids.append(rr_rep.retroflex_fidelity)
            rr_exp_total += rr_rep.n_expected
            rr_col_total += rr_rep.n_collapsed
        if af_rep.aspiration_fidelity is not None and af_rep.n_expected > 0:
            af_fids.append(af_rep.aspiration_fidelity)
            af_exp_total += af_rep.n_expected
            af_col_total += af_rep.n_collapsed
        if lf_rep.length_fidelity is not None:
            lf_fids.append(lf_rep.length_fidelity)

        # Tamil-zha sub-dim
        phoneme_stats = per_phoneme_breakdown(rr_rep.per_position)
        zha = phoneme_stats.get("ḻ")
        if zha and zha.get("n_expected", 0) > 0:
            zf_exp_total += zha["n_expected"]
            zf_col_total += zha.get("n_collapsed", 0)
            if zha.get("fidelity") is not None:
                zf_fids.append(zha["fidelity"])

        # Utterance-level XLS-R layer 9 for FAD
        try:
            feats = probe._embed_processor(audio, sampling_rate=16_000, return_tensors="pt").input_values.to(probe.device)
            import torch
            with torch.no_grad():
                out = probe._embed_model(feats, output_hidden_states=True)
            hs = out.hidden_states[9]  # (1, T, D)
            emb = hs.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            utt_embs.append(emb)
        except Exception as e:
            print(f"[sanity] skip FAD emb {i}: {e}")

        # Prosodic features for PSD
        pf = extract_prosodic_features(audio)
        if not np.any(np.isnan(pf)):
            psd_feats.append(pf)

        if (i + 1) % 10 == 0:
            print(f"[sanity] {i+1}/{len(selected)} clips scored")

    # Aggregate
    def _mean(xs):
        return float(np.mean(xs)) if xs else None

    summary = {
        "language": language,
        "n_clips_scored": len(utt_embs),
        "retroflex_fidelity_mean": _mean(rr_fids),
        "retroflex_collapse_rate": rr_col_total / rr_exp_total if rr_exp_total else None,
        "retroflex_n_expected": rr_exp_total,
        "retroflex_n_collapsed": rr_col_total,
        "aspiration_fidelity_mean": _mean(af_fids),
        "aspiration_collapse_rate": af_col_total / af_exp_total if af_exp_total else None,
        "aspiration_n_expected": af_exp_total,
        "aspiration_n_collapsed": af_col_total,
        "length_fidelity_mean": _mean(lf_fids),
        "zha_fidelity_mean": _mean(zf_fids),
        "zha_collapse_rate": zf_col_total / zf_exp_total if zf_exp_total else None,
        "zha_n_expected": zf_exp_total,
    }

    # FAD: compare held-out native corpus against centroid bootstrap corpus
    if len(utt_embs) >= 2:
        gen_matrix = np.stack(utt_embs, axis=0)
        fad, fad_info = compute_fad(gen_matrix, fad_natives)
        summary["fad"] = fad
        summary["fad_mean_norm"] = fad_info["mean_norm"]
        summary["fad_trace_term"] = fad_info["trace_term"]

    # PSD: same logic but on 5-D prosodic features
    if len(psd_feats) >= 2:
        psd, psd_info = compute_psd(np.stack(psd_feats, axis=0), psd_natives)
        summary["psd"] = psd
        summary["psd_mean_norm"] = psd_info["mean_norm"]

    out_path = Path(f"/cache/psp_refs/{language}_sanity.json")
    out_path.write_text(json.dumps(summary, indent=2))
    model_vol.commit()
    print(f"[sanity] DONE {json.dumps(summary, indent=2)}")
    return summary


@app.local_entrypoint()
def main_sanity(language: str = "te", n_clips: int = 50) -> None:
    """Native-audio sanity check entrypoint. Expected: fidelities ≈ 1.0, FAD/PSD low."""
    summary = sanity_check_native.remote(language=language, n_clips=n_clips)
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def main_psd(language: str = "te", n_clips: int = 500) -> None:
    """Entrypoint for PSD native-feature bootstrap.

        modal run evaluation/psp_bootstrap.py::main_psd --language te --n-clips 500
    """
    summary = bootstrap_psd_natives.remote(language=language, n_clips=n_clips)
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def main_fad(language: str = "te", n_clips: int = 1000) -> None:
    """Entrypoint for FAD native-embedding bootstrap.

        modal run evaluation/psp_bootstrap.py::main_fad --language te --n-clips 1000
    """
    summary = bootstrap_fad_natives.remote(language=language, n_clips=n_clips)
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def main(language: str = "te", n_clips: int = 80) -> None:
    summary = bootstrap.remote(language=language, n_clips=n_clips)
    print(json.dumps(summary, indent=2))
