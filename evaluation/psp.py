"""Phoneme Substitution Profile (PSP) — retroflex-fidelity prototype.

PSP is Component 2 of the Accent Naturalness Score (see ``evaluation/ACCENT.md``).
It decomposes "accent" into measurable per-dimension substitution rates against
a native-speaker acoustic prior. The composite ANS is ultimately a weighted mean
over FAD, PSP and PSD; this module implements the most novel slice: **retroflex
collapse detection** for Indic TTS.

What this file contains
-----------------------
A working, CPU-testable prototype of the retroflex dimension. The full MFA-based
per-phoneme pipeline described in ``ACCENT.md`` is a Sprint-6 dependency; here
we ship a *lightweight* acoustic approach that works today without MFA:

1. Parse the expected text via BUPS → a sequence of phoneme ids, so we know
   where retroflexes should appear (ṭ ḍ ṇ ṣ ḷ — ids in ``RETROFLEX_PHONEMES``).
2. Use a Wav2Vec2 CTC ASR model to build a frame-level emission matrix over the
   audio, then run ``torchaudio.functional.forced_align`` to get per-character
   timestamps. We then map each expected retroflex grapheme to a (t0, t1) span.
3. For each expected retroflex span, extract a Wav2Vec2-XLS-R embedding and
   classify it against two prototypes: a native-retroflex centroid and a
   dental-substitute centroid. The retroflex-fidelity at that position is
   ``sim(retroflex) / (sim(retroflex) + sim(dental))`` in [0, 1].
4. Aggregate: the retroflex-fidelity score for the clip is the mean over
   expected retroflex positions; the overall PSP.retroflex is 1.0 when every
   expected retroflex lands closer to the native centroid than the dental one.

Design choices worth flagging
-----------------------------
- **Prototype-based, not MFA.** Prototypes are mean XLS-R embeddings of a tiny
  reference set of retroflex- and dental-only clips. These can be bootstrapped
  from IndicTTS native speakers in Sprint 3; for now we expose a clean
  interface (``RetroflexProbe``) that accepts any native reference audio.
- **Forced-alignment on graphemes, not phonemes.** We use a *character-level*
  CTC model (the standard IndicWav2Vec style) rather than a phoneme-level one,
  since per-Indic-phoneme CTC models aren't universally available. Indic scripts
  have a near 1:1 grapheme:phoneme correspondence, so for the retroflex
  dimension this works — ṭ, ḍ, ṇ maps cleanly to ట, డ, ణ in Telugu, etc.
- **Script-aware retroflex detection.** Because we want this testable without
  audio, the text-side retroflex scanner uses BUPS + script-native Unicode
  ranges in parallel; they must agree.
- **Import-testable.** No GPU required to import this module. Model loading is
  deferred to ``RetroflexProbe.load()`` and will pick GPU if available.

The public surface is small:

    from evaluation.psp import RetroflexProbe, expected_retroflex_positions

    probe = RetroflexProbe()           # lazy-constructs; no model load yet
    score = probe.score(audio_bytes, text, script="telugu")
    # score: {"retroflex_fidelity": float in [0,1], "n_expected": int,
    #         "per_position": [{"phoneme": "ṭ", "char": "ట", "t0": 0.31,
    #                           "t1": 0.37, "native_sim": 0.83, ...}]}

For now the tests validate the text-side scanner + scoring math on synthetic
embeddings; the audio-side components are exercised only when ``torch`` and
``transformers`` are available, so the suite stays green on a CPU-only dev box.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from praxy.linguistics.bups import (
    BUPS,
    PHONEME_TO_ID,
    SANSCRIPT_NAMES,
    detect_script,
    get_bups,
)

if TYPE_CHECKING:  # pragma: no cover — only for type hints
    import numpy as np

# -----------------------------------------------------------------------------
# Retroflex inventory.
#
# These are the BUPS ids that represent retroflex articulation. ṭh and ḍh
# count too — they're aspirated retroflexes and equally collapsible. ḻ (Tamil
# zha) is a retroflex approximant; treated here as its own fidelity sub-score.
#
# The "dental collapse" targets are the phonemes a non-native speaker is most
# likely to substitute for each retroflex. These are used to build the
# dental-centroid reference for the classifier probe.
# -----------------------------------------------------------------------------

RETROFLEX_PHONEMES: tuple[str, ...] = ("ṭ", "ṭh", "ḍ", "ḍh", "ṇ", "ṣ", "ḷ", "ḻ")
DENTAL_COLLAPSE_TARGETS: dict[str, str] = {
    "ṭ": "t",
    "ṭh": "th",
    "ḍ": "d",
    "ḍh": "dh",
    "ṇ": "n",
    "ṣ": "s",   # retroflex sibilant typically flattens to /s/ or /ʃ/
    "ḷ": "l",
    "ḻ": "l",   # Tamil zha → /l/ or /r/ (we use /l/ as the primary collapse)
}

RETROFLEX_PHONEME_IDS: frozenset[int] = frozenset(
    PHONEME_TO_ID[p] for p in RETROFLEX_PHONEMES if p in PHONEME_TO_ID
)
DENTAL_PHONEME_IDS: frozenset[int] = frozenset(
    PHONEME_TO_ID[p] for p in set(DENTAL_COLLAPSE_TARGETS.values()) if p in PHONEME_TO_ID
)


# -----------------------------------------------------------------------------
# Script-native retroflex grapheme tables.
#
# We keep a direct char→retroflex-phoneme table per script so we can locate the
# *actual character positions* of retroflex sounds in the input string. BUPS
# gives us phonemes (scripts normalized); we need chars (for forced-alignment
# time spans). Both views must agree, which is enforced in the tests.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Aspiration inventory.
#
# Aspirated stops vs their unaspirated substitutes. Hindi contrasts these
# phonemically (kal /kal/ vs khal /khal/ = "time" vs "skin"); Tamil has no
# aspirated stops natively; Telugu aspirated forms appear in Sanskrit loans
# but are rare in modern speech.
# -----------------------------------------------------------------------------

ASPIRATED_PHONEMES: tuple[str, ...] = (
    "kh", "gh", "chh", "jh", "ṭh", "ḍh", "th", "dh", "ph", "bh",
)
UNASPIRATED_COLLAPSE_TARGETS: dict[str, str] = {
    "kh": "k", "gh": "g", "chh": "ch", "jh": "j",
    "ṭh": "ṭ", "ḍh": "ḍ", "th": "t", "dh": "d",
    "ph": "p", "bh": "b",
}

ASPIRATED_CHARS_BY_SCRIPT: dict[str, dict[str, str]] = {
    "devanagari": {
        "ख": "kh", "घ": "gh", "छ": "chh", "झ": "jh",
        "ठ": "ṭh", "ढ": "ḍh", "थ": "th", "ध": "dh",
        "फ": "ph", "भ": "bh",
    },
    "telugu": {
        "ఖ": "kh", "ఘ": "gh", "ఛ": "chh", "ఝ": "jh",
        "ఠ": "ṭh", "ఢ": "ḍh", "థ": "th", "ధ": "dh",
        "ఫ": "ph", "భ": "bh",
    },
    # Tamil: no aspirated stops in native phonology. Empty table → aspiration
    # score for Tamil is trivially 1.0 (no expected aspirations in text).
    "tamil": {},
    "kannada": {
        "ಖ": "kh", "ಘ": "gh", "ಛ": "chh", "ಝ": "jh",
        "ಠ": "ṭh", "ಢ": "ḍh", "ಥ": "th", "ಧ": "dh",
        "ಫ": "ph", "ಭ": "bh",
    },
    "bengali": {
        "খ": "kh", "ঘ": "gh", "ছ": "chh", "ঝ": "jh",
        "ঠ": "ṭh", "ঢ": "ḍh", "থ": "th", "ধ": "dh",
        "ফ": "ph", "ভ": "bh",
    },
    "gujarati": {
        "ખ": "kh", "ઘ": "gh", "છ": "chh", "ઝ": "jh",
        "ઠ": "ṭh", "ઢ": "ḍh", "થ": "th", "ધ": "dh",
        "ફ": "ph", "ભ": "bh",
    },
}


RETROFLEX_CHARS_BY_SCRIPT: dict[str, dict[str, str]] = {
    "devanagari": {
        "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ",
        "ष": "ṣ", "ळ": "ḷ",
    },
    "telugu": {
        "ట": "ṭ", "ఠ": "ṭh", "డ": "ḍ", "ఢ": "ḍh", "ణ": "ṇ",
        "ష": "ṣ", "ళ": "ḷ",
    },
    "tamil": {
        # Tamil only distinguishes a subset; ட carries both /ʈ/ and /ɖ/ allophonically.
        "ட": "ṭ", "ண": "ṇ", "ஷ": "ṣ", "ள": "ḷ", "ழ": "ḻ",
    },
    "kannada": {
        "ಟ": "ṭ", "ಠ": "ṭh", "ಡ": "ḍ", "ಢ": "ḍh", "ಣ": "ṇ",
        "ಷ": "ṣ", "ಳ": "ḷ",
    },
    "malayalam": {
        # Malayalam retroflex aspirated stop ("MALAYALAM LETTER TTHA") is
        # visually reminiscent of Latin letter-O but semantically distinct;
        # keep as-is.
        "ട": "ṭ", "ഠ": "ṭh", "ഡ": "ḍ", "ഢ": "ḍh", "ണ": "ṇ",  # noqa: RUF001
        "ഷ": "ṣ", "ള": "ḷ", "ഴ": "ḻ",
    },
    "bengali": {
        "ট": "ṭ", "ঠ": "ṭh", "ড": "ḍ", "ঢ": "ḍh", "ণ": "ṇ",
        "ষ": "ṣ",
    },
    "gujarati": {
        "ટ": "ṭ", "ઠ": "ṭh", "ડ": "ḍ", "ઢ": "ḍh", "ણ": "ṇ",
        "ષ": "ṣ", "ળ": "ḷ",
    },
}


# -----------------------------------------------------------------------------
# Data classes for reporting.
# -----------------------------------------------------------------------------


@dataclass
class RetroflexPosition:
    """One expected retroflex occurrence in the input text."""

    char: str            # the actual grapheme in the script
    phoneme: str         # BUPS phoneme token, e.g. "ṭ"
    phoneme_id: int
    char_index: int      # index into the original text (char-based)
    t0: float | None = None   # alignment start (s) — filled after forced-align
    t1: float | None = None   # alignment end (s)
    native_sim: float | None = None
    dental_sim: float | None = None
    fidelity: float | None = None    # native_sim / (native_sim + dental_sim)
    collapsed: bool | None = None    # True iff fidelity < 0.5


@dataclass
class RetroflexReport:
    """Aggregate retroflex-fidelity report for one audio clip."""

    retroflex_fidelity: float       # scalar in [0, 1], 1.0 = fully native
    n_expected: int                 # expected retroflexes in text
    n_collapsed: int                # how many fell closer to dental prototype
    per_position: list[RetroflexPosition] = field(default_factory=list)
    language: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "retroflex_fidelity": self.retroflex_fidelity,
            "n_expected": self.n_expected,
            "n_collapsed": self.n_collapsed,
            "language": self.language,
            "notes": self.notes,
            "per_position": [
                {
                    "char": p.char,
                    "phoneme": p.phoneme,
                    "char_index": p.char_index,
                    "t0": p.t0,
                    "t1": p.t1,
                    "native_sim": p.native_sim,
                    "dental_sim": p.dental_sim,
                    "fidelity": p.fidelity,
                    "collapsed": p.collapsed,
                }
                for p in self.per_position
            ],
        }


# -----------------------------------------------------------------------------
# Text-side: find expected retroflex positions.
#
# Works off-line, no model required. Script is detected if not provided.
# -----------------------------------------------------------------------------


def expected_retroflex_positions(
    text: str,
    script: str | None = None,
) -> list[RetroflexPosition]:
    """Return a list of retroflex occurrences in ``text``, in char order.

    ``script`` is detected from the first Indic character if not supplied. This
    uses a direct grapheme table (``RETROFLEX_CHARS_BY_SCRIPT``) rather than
    BUPS — graphemes give stable char indices, which we need for aligning
    against the audio side.
    """
    if script is None:
        # Detect from the first Indic char.
        for ch in text:
            s = detect_script(ch)
            if s != "latin":
                script = s
                break
        if script is None:
            return []

    table = RETROFLEX_CHARS_BY_SCRIPT.get(script)
    if table is None:
        return []

    out: list[RetroflexPosition] = []
    for i, ch in enumerate(text):
        phoneme = table.get(ch)
        if phoneme is None:
            continue
        pid = PHONEME_TO_ID.get(phoneme, PHONEME_TO_ID["<unk>"])
        out.append(
            RetroflexPosition(
                char=ch,
                phoneme=phoneme,
                phoneme_id=pid,
                char_index=i,
            )
        )
    return out


def bups_retroflex_phonemes_in(text: str, script: str | None = None) -> list[str]:
    """Cross-check: return the BUPS retroflex phonemes derived from ``text``.

    The grapheme scanner (``expected_retroflex_positions``) gives us one view;
    BUPS encoding gives us another. They must agree for the pipeline to be
    trustworthy (see tests).
    """
    bups = get_bups()
    if script is None:
        for ch in text:
            s = detect_script(ch)
            if s in SANSCRIPT_NAMES:
                script = s
                break
        if script is None:
            return []
    tokens = bups.encode_tokens(text, script=script)
    return [t.phoneme for t in tokens if t.phoneme in RETROFLEX_PHONEMES]


# -----------------------------------------------------------------------------
# Scoring math — unit-testable without any audio model.
# -----------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically-safe cosine similarity. Accepts 1-D np arrays."""
    import numpy as np

    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def fidelity_from_similarities(native_sim: float, dental_sim: float) -> float:
    """Softmax-free normalization: s / (s + d). Both sims remapped to [0, 1].

    Cosine sim is already in [-1, 1]; we rectify with max(x, 0) so that two
    embeddings with no positive alignment to either prototype don't silently
    produce a meaningful score (they produce 0.5, i.e. "unknown").
    """
    n = max(native_sim, 0.0)
    d = max(dental_sim, 0.0)
    total = n + d
    if total < 1e-9:
        return 0.5  # ambiguous — neither centroid wins
    return n / total


def aggregate_retroflex_fidelity(positions: list[RetroflexPosition]) -> tuple[float, int]:
    """Mean fidelity across scored positions + count of collapsed positions.

    If no positions have fidelity computed, returns (1.0, 0): absent data is
    treated as "no retroflex errors observed" so the score doesn't penalise
    languages/utterances without retroflexes.
    """
    scored = [p for p in positions if p.fidelity is not None]
    if not scored:
        return 1.0, 0
    mean_fid = sum(p.fidelity for p in scored) / len(scored)  # type: ignore[misc]
    n_collapsed = sum(1 for p in scored if p.collapsed)
    return mean_fid, n_collapsed


def per_phoneme_breakdown(
    positions: list[RetroflexPosition],
) -> dict[str, dict[str, float | int | None]]:
    """Break down a list of scored retroflex positions by phoneme.

    Returns ``{phoneme: {"fidelity": mean, "n_expected": int, "n_collapsed": int}}``.
    Used to pull sub-dimensions out of an aggregate retroflex report — e.g.,
    the Tamil-zha fidelity (phoneme ``ḻ``) is a sub-slice of the overall
    retroflex report and the paper wants it reported separately.
    """
    by_phoneme: dict[str, list[RetroflexPosition]] = {}
    for p in positions:
        by_phoneme.setdefault(p.phoneme, []).append(p)

    out: dict[str, dict[str, float | int | None]] = {}
    for phoneme, ps in by_phoneme.items():
        scored = [p for p in ps if p.fidelity is not None]
        n_expected = len(ps)
        n_collapsed = sum(1 for p in scored if p.collapsed)
        mean_fid = (
            sum(p.fidelity for p in scored) / len(scored)  # type: ignore[misc]
            if scored else None
        )
        out[phoneme] = {
            "fidelity": mean_fid,
            "n_expected": n_expected,
            "n_collapsed": n_collapsed,
        }
    return out


# -----------------------------------------------------------------------------
# Audio-side: the RetroflexProbe class.
#
# GPU-capable; cleanly import-testable on CPU. Model loading is deferred until
# ``load()`` is called (or the first scoring call). In Modal, instantiate once
# and reuse across calls.
# -----------------------------------------------------------------------------


# Default models — chosen for CPU/GPU parity and for the availability of both
# a CTC head (for alignment) and embeddings (for the probe).
#
# - XLS-R 300m: strong Indic coverage (pretrained on 128 languages incl. IndicSUPERB
#   derivatives). We pull hidden states at a mid-layer — see ACCENT.md note that
#   middle layers carry the most phonetic/accent-discriminative info (Probing
#   for Phonology, arxiv 2506.17542).
# - For forced-alignment we reuse the same model when an AI4Bharat char-CTC
#   checkpoint is unavailable; otherwise prefer a char-CTC model of the target
#   language.
DEFAULT_EMBED_MODEL = "facebook/wav2vec2-xls-r-300m"
DEFAULT_ALIGN_MODEL_BY_LANG = {
    "hi": "ai4bharat/indicwav2vec-hindi",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "ta": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
}
XLSR_EMBED_LAYER = 9  # middle-ish layer of a 24-layer XLS-R; overrideable


class RetroflexProbe:
    """Prototype-based retroflex-fidelity probe.

    Lifecycle
    ---------
    - Construct: cheap, no model loaded.
    - ``load()``: loads XLS-R (+ a lang-specific align model). GPU if available.
    - ``add_native_reference(phoneme, audio_segments)``: register native-speaker
      audio for one or more phonemes; builds centroid per phoneme.
    - ``add_dental_reference(phoneme, audio_segments)``: register matched
      dental-substitute audio (collected from L1-English Telugu reads, or
      synthesised via pronunciation-swap).
    - ``score(audio, text, script)``: produce a ``RetroflexReport``.

    Operational note
    ----------------
    The prototypes are the weak link of a *prototype-based* probe. Sprint 6
    replaces this with an MFA-aligned native acoustic model; until then, even
    a dozen good native clips per phoneme beats no probe at all. We deliberately
    expose a small, manual reference API so the native-corpus curator can
    bootstrap iteratively.
    """

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        align_model: str | None = None,
        embed_layer: int = XLSR_EMBED_LAYER,
        device: str | None = None,
    ) -> None:
        self.embed_model_name = embed_model
        self.align_model_name = align_model
        self.embed_layer = embed_layer
        self.device = device
        self._loaded = False
        self._embed_model = None
        self._embed_processor = None
        self._align_model = None
        self._align_processor = None
        self._bups: BUPS | None = None

        # Prototype registries: phoneme -> list of embedding vectors (np arrays).
        self._native_refs: dict[str, list[np.ndarray]] = {}
        self._dental_refs: dict[str, list[np.ndarray]] = {}

    # ------------------------ Model loading ------------------------

    def load(self) -> None:
        """Load embedding + alignment models. Idempotent."""
        if self._loaded:
            return
        import torch
        from transformers import (
            AutoModelForCTC,
            AutoProcessor,
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Embedding model (XLS-R): we need hidden states, not logits.
        self._embed_processor = Wav2Vec2FeatureExtractor.from_pretrained(self.embed_model_name)
        self._embed_model = Wav2Vec2Model.from_pretrained(
            self.embed_model_name,
            output_hidden_states=True,
        ).to(device)
        self._embed_model.eval()

        # Alignment model (CTC head required). Falls back to the embed model
        # only if a real CTC checkpoint isn't provided; callers should supply
        # a language-appropriate alignment model.
        if self.align_model_name is not None:
            self._align_processor = AutoProcessor.from_pretrained(self.align_model_name)
            self._align_model = AutoModelForCTC.from_pretrained(self.align_model_name).to(device)
            self._align_model.eval()

        self._bups = get_bups()
        self._loaded = True

    # ------------------------ Reference prototypes ------------------------

    def add_native_reference(self, phoneme: str, embedding: np.ndarray) -> None:
        """Append one native embedding for ``phoneme`` (e.g. "ṭ"). ndarray shape (D,)."""
        self._native_refs.setdefault(phoneme, []).append(embedding)

    def add_dental_reference(self, phoneme: str, embedding: np.ndarray) -> None:
        """Append one dental-substitute embedding for the retroflex ``phoneme``."""
        self._dental_refs.setdefault(phoneme, []).append(embedding)

    def native_centroid(self, phoneme: str) -> np.ndarray | None:
        import numpy as np

        refs = self._native_refs.get(phoneme)
        if not refs:
            return None
        return np.mean(np.stack(refs, axis=0), axis=0)

    def dental_centroid(self, phoneme: str) -> np.ndarray | None:
        import numpy as np

        refs = self._dental_refs.get(phoneme)
        if not refs:
            # Fall back to dental-collapse-target's own native centroid
            # (i.e. "what a /t/ looks like") if dental substitutes weren't
            # explicitly provided.
            target = DENTAL_COLLAPSE_TARGETS.get(phoneme)
            if target is not None and target in self._native_refs:
                refs = self._native_refs[target]
        if not refs:
            return None
        return np.mean(np.stack(refs, axis=0), axis=0)

    # ------------------------ Scoring ------------------------

    def score_positions(
        self,
        positions: list[RetroflexPosition],
        embeddings_per_position: list[np.ndarray | None],
    ) -> RetroflexReport:
        """Score pre-extracted embeddings against the prototype centroids.

        Callers can use this directly when they already have spans + embeddings,
        bypassing audio preprocessing. Mostly here to keep unit tests hermetic.
        """
        assert len(positions) == len(embeddings_per_position)
        for pos, emb in zip(positions, embeddings_per_position, strict=True):
            if emb is None:
                continue
            n_centroid = self.native_centroid(pos.phoneme)
            d_centroid = self.dental_centroid(pos.phoneme)
            if n_centroid is None or d_centroid is None:
                # No prototype for this phoneme → mark ambiguous, don't count.
                continue
            n_sim = cosine_similarity(emb, n_centroid)
            d_sim = cosine_similarity(emb, d_centroid)
            fid = fidelity_from_similarities(n_sim, d_sim)
            pos.native_sim = n_sim
            pos.dental_sim = d_sim
            pos.fidelity = fid
            pos.collapsed = fid < 0.5

        mean_fid, n_collapsed = aggregate_retroflex_fidelity(positions)
        return RetroflexReport(
            retroflex_fidelity=mean_fid,
            n_expected=len(positions),
            n_collapsed=n_collapsed,
            per_position=positions,
        )

    def score(
        self,
        audio_bytes: bytes,
        text: str,
        script: str | None = None,
        language_code: str | None = None,
    ) -> RetroflexReport:
        """Full pipeline: audio+text → RetroflexReport.

        Requires ``load()`` to have been called (or called here on first use).
        Raises RuntimeError if alignment model isn't configured — alignment
        is mandatory for the audio side of this probe.
        """
        if not self._loaded:
            self.load()
        if self._align_model is None:
            raise RuntimeError(
                "No alignment model loaded. Pass align_model= to RetroflexProbe "
                "or call .load() after setting align_model_name."
            )

        positions = expected_retroflex_positions(text, script=script)
        if not positions:
            return RetroflexReport(
                retroflex_fidelity=1.0,
                n_expected=0,
                n_collapsed=0,
                per_position=[],
                language=language_code,
                notes="No retroflex phonemes expected in input.",
            )

        audio_16k = _load_audio_16k(audio_bytes)
        spans = self._align_text_to_audio(audio_16k, text)
        # Fill t0, t1 for each retroflex position from spans (char-index keyed).
        for pos in positions:
            span = spans.get(pos.char_index)
            if span is not None:
                pos.t0, pos.t1 = span

        # Extract XLS-R embedding per retroflex span.
        embeddings = [
            self._embed_span(audio_16k, pos.t0, pos.t1)
            if pos.t0 is not None and pos.t1 is not None
            else None
            for pos in positions
        ]
        report = self.score_positions(positions, embeddings)
        report.language = language_code
        return report

    # ------------------------ Audio helpers ------------------------

    def _align_text_to_audio(
        self,
        audio_16k: np.ndarray,
        text: str,
    ) -> dict[int, tuple[float, float]]:
        """Forced-align text to audio; return {char_index: (t0, t1)} in seconds.

        Uses the loaded CTC model (``self._align_model``). If alignment fails
        mid-utterance, skipped chars simply don't appear in the returned dict —
        caller handles missing spans gracefully.

        Implementation detail: we run CTC, then use torchaudio's forced_align
        on the emission matrix using the target grapheme sequence.
        """
        import torch
        import torchaudio.functional as taF  # noqa: N812 — torchaudio convention uses F

        # Encode waveform.
        inputs = self._align_processor(  # type: ignore[union-attr]
            audio_16k, sampling_rate=16_000, return_tensors="pt"
        ).input_values.to(self.device)
        with torch.no_grad():
            logits = self._align_model(inputs).logits  # type: ignore[union-attr]  # (1, T, V)
        emission = torch.log_softmax(logits, dim=-1)[0]  # (T, V)
        n_frames = emission.shape[0]
        duration_s = len(audio_16k) / 16_000.0
        frame_s = duration_s / n_frames

        # Build grapheme targets using the align tokenizer's vocab. Chars not
        # in the vocab are dropped from the alignment but preserved in the
        # char_index returned by expected_retroflex_positions → we just won't
        # get a span for them. That's OK for this prototype.
        tokenizer = self._align_processor.tokenizer  # type: ignore[union-attr]
        vocab = tokenizer.get_vocab()
        blank_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        target_ids: list[int] = []
        # We keep an index map so each target id knows its source char index.
        target_to_char: list[int] = []
        for i, ch in enumerate(text):
            tok_id = vocab.get(ch)
            if tok_id is None:
                # try lowercase (some tokenizers only store lower)
                tok_id = vocab.get(ch.lower())
            if tok_id is None or tok_id == blank_id:
                continue
            target_ids.append(tok_id)
            target_to_char.append(i)

        if not target_ids:
            return {}

        targets = torch.tensor([target_ids], dtype=torch.int32, device=emission.device)
        try:
            alignments, scores = taF.forced_align(
                emission.unsqueeze(0), targets, blank=blank_id
            )
        except Exception:
            # fallback: no alignment available → empty span map
            return {}

        spans = taF.merge_tokens(alignments[0], scores[0])
        out: dict[int, tuple[float, float]] = {}
        # spans is in 1:1 order with target_ids
        for tok_idx, span in enumerate(spans):
            if tok_idx >= len(target_to_char):
                break
            char_idx = target_to_char[tok_idx]
            t0 = span.start * frame_s
            t1 = span.end * frame_s
            out[char_idx] = (t0, t1)
        return out

    def _embed_span(
        self,
        audio_16k: np.ndarray,
        t0: float,
        t1: float,
    ) -> np.ndarray | None:
        """Mean-pool the XLS-R hidden state at layer ``self.embed_layer`` over (t0, t1)."""
        import numpy as np
        import torch

        # Slice audio; pad if too short (XLS-R wants min ~400 samples for context).
        s0 = max(0, int(t0 * 16_000))
        s1 = min(len(audio_16k), int(t1 * 16_000))
        if s1 - s0 < 160:  # <10 ms → unreliable
            # Widen by 50 ms on either side to pick up co-articulation context.
            pad = int(0.05 * 16_000)
            s0 = max(0, s0 - pad)
            s1 = min(len(audio_16k), s1 + pad)
        if s1 - s0 < 400:
            return None

        segment = audio_16k[s0:s1]
        inputs = self._embed_processor(  # type: ignore[union-attr]
            segment, sampling_rate=16_000, return_tensors="pt"
        ).input_values.to(self.device)
        with torch.no_grad():
            out = self._embed_model(inputs, output_hidden_states=True)  # type: ignore[union-attr]
        # hidden_states: tuple(len = n_layers+1)
        layer = self.embed_layer
        hs = out.hidden_states[layer]  # (1, T, D)
        emb = hs.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        return emb


# -----------------------------------------------------------------------------
# Aspiration probe — second PSP dimension.
#
# Parallel architecture to RetroflexProbe: same alignment + XLS-R embedding +
# cosine-centroid scoring, but the contrast is (aspirated, unaspirated) rather
# than (retroflex, dental). Reuses RetroflexProbe's loaded models by sharing
# the same probe instance when wrapped in PSPScorer; also usable standalone.
# -----------------------------------------------------------------------------


@dataclass
class AspirationPosition:
    """One expected aspirated phoneme occurrence in the input text."""

    char: str
    phoneme: str         # e.g. "kh"
    phoneme_id: int
    char_index: int
    t0: float | None = None
    t1: float | None = None
    aspirated_sim: float | None = None
    unaspirated_sim: float | None = None
    fidelity: float | None = None    # aspirated_sim / (aspirated + unaspirated)
    collapsed: bool | None = None    # True iff fidelity < 0.5


@dataclass
class AspirationReport:
    """Aggregate aspiration-fidelity report for one audio clip."""

    aspiration_fidelity: float
    n_expected: int
    n_collapsed: int
    per_position: list[AspirationPosition] = field(default_factory=list)
    language: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspiration_fidelity": self.aspiration_fidelity,
            "n_expected": self.n_expected,
            "n_collapsed": self.n_collapsed,
            "language": self.language,
            "notes": self.notes,
            "per_position": [
                {
                    "char": p.char,
                    "phoneme": p.phoneme,
                    "char_index": p.char_index,
                    "t0": p.t0,
                    "t1": p.t1,
                    "aspirated_sim": p.aspirated_sim,
                    "unaspirated_sim": p.unaspirated_sim,
                    "fidelity": p.fidelity,
                    "collapsed": p.collapsed,
                }
                for p in self.per_position
            ],
        }


def expected_aspiration_positions(
    text: str,
    script: str | None = None,
) -> list[AspirationPosition]:
    """Return a list of aspirated-phoneme occurrences in ``text``, in char order.

    Mirrors ``expected_retroflex_positions`` but scans for aspirated graphemes
    (ख/ग/छ/झ/थ/ध/फ/भ in Devanagari etc.). Tamil returns [] since Tamil has no
    phonemic aspirated stops.
    """
    if script is None:
        for ch in text:
            s = detect_script(ch)
            if s != "latin":
                script = s
                break
        if script is None:
            return []

    table = ASPIRATED_CHARS_BY_SCRIPT.get(script)
    if not table:
        return []

    out: list[AspirationPosition] = []
    for i, ch in enumerate(text):
        phoneme = table.get(ch)
        if phoneme is None:
            continue
        pid = PHONEME_TO_ID.get(phoneme, PHONEME_TO_ID["<unk>"])
        out.append(
            AspirationPosition(
                char=ch,
                phoneme=phoneme,
                phoneme_id=pid,
                char_index=i,
            )
        )
    return out


def aggregate_aspiration_fidelity(positions: list[AspirationPosition]) -> tuple[float, int]:
    """Mean fidelity across scored positions + collapsed count.

    Mirrors ``aggregate_retroflex_fidelity``. Returns (1.0, 0) if no positions
    were scored — absence of aspiration isn't penalised.
    """
    scored = [p for p in positions if p.fidelity is not None]
    if not scored:
        return 1.0, 0
    mean_fid = sum(p.fidelity for p in scored) / len(scored)  # type: ignore[misc]
    n_collapsed = sum(1 for p in scored if p.collapsed)
    return mean_fid, n_collapsed


# -----------------------------------------------------------------------------
# Length-fidelity probe — third PSP dimension (ratio-based, not centroid-based).
# -----------------------------------------------------------------------------


# Long ↔ short vowel pairs, BUPS-phoneme form. In Devanagari / Telugu / Tamil
# these are phonemic: "kal" (time) vs "kāl" (death), "kal" vs "kāla".
LONG_SHORT_VOWEL_PAIRS: dict[str, str] = {
    "ā": "a", "ī": "i", "ū": "u", "ē": "e", "ō": "o",
}

# Native-speaker long/short duration ratios from Indic phonology literature.
# Used as defaults when a corpus-derived prior isn't available. Sources:
# IndicTTS corpus statistics, various Indic phonology references.
# Paper will report both literature-prior and corpus-derived-prior results.
NATIVE_LONG_SHORT_RATIO_DEFAULT: dict[str, float] = {
    "telugu": 1.85,
    "hindi": 2.00,
    "tamil": 1.90,
    "devanagari": 2.00,  # alias for hindi when script is known but lang isn't
}

# Native-script char tables for long vowels (we want positions in original
# text). Short-vowel chars are inferred via LONG_SHORT_VOWEL_PAIRS lookup
# on the BUPS phoneme.
LONG_VOWEL_CHARS_BY_SCRIPT: dict[str, dict[str, str]] = {
    "devanagari": {
        "आ": "ā", "ई": "ī", "ऊ": "ū", "ए": "ē", "ओ": "ō",
        # Matras (vowel signs — attached to consonants)
        "ा": "ā", "ी": "ī", "ू": "ū", "े": "ē", "ो": "ō",
    },
    "telugu": {
        "ఆ": "ā", "ఈ": "ī", "ఊ": "ū", "ఏ": "ē", "ఓ": "ō",
        "ా": "ā", "ీ": "ī", "ూ": "ū", "ే": "ē", "ో": "ō",
    },
    "tamil": {
        "ஆ": "ā", "ஈ": "ī", "ஊ": "ū", "ஏ": "ē", "ஓ": "ō",
        "ா": "ā", "ீ": "ī", "ூ": "ū", "ே": "ē", "ோ": "ō",
    },
    "kannada": {
        "ಆ": "ā", "ಈ": "ī", "ಊ": "ū", "ಏ": "ē", "ಓ": "ō",
        "ಾ": "ā", "ೀ": "ī", "ೂ": "ū", "ೇ": "ē", "ೋ": "ō",
    },
}

SHORT_VOWEL_CHARS_BY_SCRIPT: dict[str, dict[str, str]] = {
    "devanagari": {
        "अ": "a", "इ": "i", "उ": "u", "ए": "e", "ओ": "o",
        # Note: no short-vowel matras are distinct; schwa is implicit in consonants.
    },
    "telugu": {
        "అ": "a", "ఇ": "i", "ఉ": "u", "ఎ": "e", "ఒ": "o",
        "ి": "i", "ు": "u", "ె": "e", "ొ": "o",
    },
    "tamil": {
        "அ": "a", "இ": "i", "உ": "u", "எ": "e", "ஒ": "o",
        "ி": "i", "ு": "u", "ெ": "e", "ொ": "o",
    },
    "kannada": {
        "ಅ": "a", "ಇ": "i", "ಉ": "u", "ಎ": "e", "ಒ": "o",
        "ಿ": "i", "ು": "u", "ೆ": "e", "ೊ": "o",
    },
}


@dataclass
class LengthPosition:
    """One long- or short-vowel occurrence in the input text."""

    char: str
    phoneme: str              # BUPS: "ā" / "a" / "ī" / "i" / ...
    is_long: bool
    char_index: int
    t0: float | None = None
    t1: float | None = None
    duration_s: float | None = None


@dataclass
class LengthReport:
    """Aggregate length-fidelity report for one audio clip."""

    length_fidelity: float           # [0, 1], 1 = native-like ratio
    measured_long_short_ratio: float | None
    native_ratio: float              # reference ratio used
    n_long: int
    n_short: int
    per_position: list[LengthPosition] = field(default_factory=list)
    language: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_fidelity": self.length_fidelity,
            "measured_long_short_ratio": self.measured_long_short_ratio,
            "native_ratio": self.native_ratio,
            "n_long": self.n_long,
            "n_short": self.n_short,
            "language": self.language,
            "notes": self.notes,
            "per_position": [
                {
                    "char": p.char,
                    "phoneme": p.phoneme,
                    "is_long": p.is_long,
                    "char_index": p.char_index,
                    "t0": p.t0,
                    "t1": p.t1,
                    "duration_s": p.duration_s,
                }
                for p in self.per_position
            ],
        }


def expected_length_positions(
    text: str,
    script: str | None = None,
) -> list[LengthPosition]:
    """Return long- and short-vowel positions in ``text``, marked with is_long."""
    if script is None:
        for ch in text:
            s = detect_script(ch)
            if s != "latin":
                script = s
                break
        if script is None:
            return []

    long_table = LONG_VOWEL_CHARS_BY_SCRIPT.get(script, {})
    short_table = SHORT_VOWEL_CHARS_BY_SCRIPT.get(script, {})
    if not long_table and not short_table:
        return []

    out: list[LengthPosition] = []
    for i, ch in enumerate(text):
        if ch in long_table:
            phoneme = long_table[ch]
            pid = PHONEME_TO_ID.get(phoneme, PHONEME_TO_ID["<unk>"])
            out.append(LengthPosition(
                char=ch, phoneme=phoneme, is_long=True, char_index=i,
            ))
        elif ch in short_table:
            phoneme = short_table[ch]
            pid = PHONEME_TO_ID.get(phoneme, PHONEME_TO_ID["<unk>"])
            out.append(LengthPosition(
                char=ch, phoneme=phoneme, is_long=False, char_index=i,
            ))
    return out


def length_fidelity_from_ratio(
    measured: float | None, native: float, tolerance: float = 0.5
) -> float:
    """Fidelity = 1 - |measured - native| / (native * tolerance), clamped [0, 1].

    With tolerance=0.5, a measured ratio within 50% of the native ratio scores
    1.0; a ratio at 1.5x native (or 0.5x native) scores 0.0. Generous because
    corpus-wise speech-rate variation is large.
    """
    if measured is None:
        return 1.0
    error = abs(measured - native) / max(native * tolerance, 1e-6)
    return max(0.0, min(1.0, 1.0 - error))


class LengthProbe(RetroflexProbe):
    """Measure vowel-length fidelity via long-to-short duration ratio.

    Unlike RetroflexProbe/AspirationProbe (centroid-based classification), this
    probe is *distributional*: we measure duration ratios and compare against
    a native-speaker prior. Same loaded models (alignment + embedding), but
    the scoring path is different.

    Native prior: ``native_long_short_ratio`` (default from literature).
    """

    def __init__(
        self,
        *args,
        native_long_short_ratio: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._native_ratio_override = native_long_short_ratio

    def _resolve_native_ratio(self, language_code: str | None, script: str | None) -> float:
        if self._native_ratio_override is not None:
            return self._native_ratio_override
        lang_map = {"te": "telugu", "hi": "hindi", "ta": "tamil"}
        lang_name = lang_map.get(language_code or "")
        if lang_name in NATIVE_LONG_SHORT_RATIO_DEFAULT:
            return NATIVE_LONG_SHORT_RATIO_DEFAULT[lang_name]
        if script in NATIVE_LONG_SHORT_RATIO_DEFAULT:
            return NATIVE_LONG_SHORT_RATIO_DEFAULT[script]
        return 1.9  # generic Indic fallback

    def score_length(
        self,
        audio_bytes: bytes,
        text: str,
        script: str | None = None,
        language_code: str | None = None,
    ) -> LengthReport:
        """Full length-fidelity pipeline: audio+text → LengthReport."""
        if not self._loaded:
            self.load()
        if self._align_model is None:
            raise RuntimeError(
                "No alignment model loaded. Pass align_model= to LengthProbe "
                "or call .load() after setting align_model_name."
            )

        positions = expected_length_positions(text, script=script)
        native_ratio = self._resolve_native_ratio(language_code, script)

        if not positions:
            return LengthReport(
                length_fidelity=1.0,
                measured_long_short_ratio=None,
                native_ratio=native_ratio,
                n_long=0, n_short=0,
                per_position=[], language=language_code,
                notes="No vowels expected in input.",
            )

        audio_16k = _load_audio_16k(audio_bytes)
        spans = self._align_text_to_audio(audio_16k, text)
        for pos in positions:
            span = spans.get(pos.char_index)
            if span is not None:
                pos.t0, pos.t1 = span
                pos.duration_s = max(0.0, span[1] - span[0])

        long_durations = [p.duration_s for p in positions if p.is_long and p.duration_s]
        short_durations = [p.duration_s for p in positions if not p.is_long and p.duration_s]

        if not long_durations or not short_durations:
            # Can't compute a ratio; return neutral score with counts.
            return LengthReport(
                length_fidelity=1.0,
                measured_long_short_ratio=None,
                native_ratio=native_ratio,
                n_long=len(long_durations),
                n_short=len(short_durations),
                per_position=positions,
                language=language_code,
                notes="Insufficient vowel pairs to compute ratio.",
            )

        mean_long = sum(long_durations) / len(long_durations)
        mean_short = sum(short_durations) / len(short_durations)
        measured_ratio = mean_long / max(mean_short, 1e-6)
        fid = length_fidelity_from_ratio(measured_ratio, native_ratio)

        return LengthReport(
            length_fidelity=fid,
            measured_long_short_ratio=measured_ratio,
            native_ratio=native_ratio,
            n_long=len(long_durations),
            n_short=len(short_durations),
            per_position=positions,
            language=language_code,
        )


class AspirationProbe(RetroflexProbe):
    """Second-dimension probe for aspiration fidelity.

    Subclasses RetroflexProbe to reuse model loading, alignment, and span
    embedding. The difference is in scoring: for each expected aspirated
    phoneme (e.g. "kh"), we compare its XLS-R embedding against the native
    centroid for the same phoneme ("kh" in _native_refs) vs the centroid for
    the unaspirated substitute ("k" in _native_refs). Both are present in the
    same bootstrap pickle, so we don't need a separate centroid file.
    """

    def unaspirated_centroid(self, aspirated_phoneme: str):
        """The unaspirated counterpart's native centroid — serves as the substitute prototype."""
        target = UNASPIRATED_COLLAPSE_TARGETS.get(aspirated_phoneme)
        if target is None:
            return None
        return self.native_centroid(target)

    def score_aspiration_positions(
        self,
        positions: list[AspirationPosition],
        embeddings_per_position: list[np.ndarray | None],
    ) -> AspirationReport:
        """Score pre-extracted embeddings against aspirated/unaspirated centroids."""
        assert len(positions) == len(embeddings_per_position)
        for pos, emb in zip(positions, embeddings_per_position, strict=True):
            if emb is None:
                continue
            aspirated_c = self.native_centroid(pos.phoneme)
            unaspirated_c = self.unaspirated_centroid(pos.phoneme)
            if aspirated_c is None or unaspirated_c is None:
                continue  # no prototype — mark ambiguous, don't count
            a_sim = cosine_similarity(emb, aspirated_c)
            u_sim = cosine_similarity(emb, unaspirated_c)
            fid = fidelity_from_similarities(a_sim, u_sim)
            pos.aspirated_sim = a_sim
            pos.unaspirated_sim = u_sim
            pos.fidelity = fid
            pos.collapsed = fid < 0.5

        mean_fid, n_collapsed = aggregate_aspiration_fidelity(positions)
        return AspirationReport(
            aspiration_fidelity=mean_fid,
            n_expected=len(positions),
            n_collapsed=n_collapsed,
            per_position=positions,
        )

    def score_aspiration(
        self,
        audio_bytes: bytes,
        text: str,
        script: str | None = None,
        language_code: str | None = None,
    ) -> AspirationReport:
        """Full aspiration pipeline: audio+text → AspirationReport.

        Separate method name (not override of ``score``) so a single probe
        instance can compute both dimensions: ``.score()`` returns retroflex,
        ``.score_aspiration()`` returns aspiration. Both share the same loaded
        models and centroid dict.
        """
        if not self._loaded:
            self.load()
        if self._align_model is None:
            raise RuntimeError(
                "No alignment model loaded. Pass align_model= to AspirationProbe "
                "or call .load() after setting align_model_name."
            )

        positions = expected_aspiration_positions(text, script=script)
        if not positions:
            return AspirationReport(
                aspiration_fidelity=1.0,
                n_expected=0,
                n_collapsed=0,
                per_position=[],
                language=language_code,
                notes="No aspirated phonemes expected in input.",
            )

        audio_16k = _load_audio_16k(audio_bytes)
        spans = self._align_text_to_audio(audio_16k, text)
        for pos in positions:
            span = spans.get(pos.char_index)
            if span is not None:
                pos.t0, pos.t1 = span

        embeddings = [
            self._embed_span(audio_16k, pos.t0, pos.t1)
            if pos.t0 is not None and pos.t1 is not None
            else None
            for pos in positions
        ]
        report = self.score_aspiration_positions(positions, embeddings)
        report.language = language_code
        return report


# -----------------------------------------------------------------------------
# Small audio utility, isolated so tests can mock it.
# -----------------------------------------------------------------------------


def _load_audio_16k(audio_bytes: bytes) -> np.ndarray:
    """Decode bytes → mono float32 at 16 kHz. Imports soundfile/librosa lazily."""
    import numpy as np
    import soundfile as sf

    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16_000:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        except ImportError as e:
            raise RuntimeError(
                f"Audio sample rate is {sr}, need 16 kHz. Install librosa for resampling."
            ) from e
    return audio


# -----------------------------------------------------------------------------
# Modal GPU entrypoint (reuses the pattern in evaluation/modal_eval.py).
# Import-testable on CPU — actual execution only occurs when Modal runs it.
# -----------------------------------------------------------------------------


def _build_modal_app():  # pragma: no cover — executed only on Modal
    """Returns a configured Modal app. Kept in a function so ``import evaluation.psp``
    doesn't actually construct the app unless asked.
    """
    import modal

    app = modal.App("praxy-psp")
    volume = modal.Volume.from_name("praxy-voice-vol", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "git", "libsndfile1")
        .pip_install(
            "torch==2.4.0",
            "torchaudio==2.4.0",
            "numpy<2",
            "scipy",
            "soundfile",
            "librosa",
            "transformers>=4.46",
            "accelerate>=0.34",
            "sentencepiece",
            "huggingface_hub>=0.25",
            "indic-transliteration>=2.3.82",
        )
    )

    @app.cls(
        image=image,
        gpu="A10G",
        volumes={"/cache": volume},
        secrets=[modal.Secret.from_name("praxy-hf")],
        timeout=1800,
        scaledown_window=300,
    )
    class ModalRetroflexProbe:
        align_model_name: str = "ai4bharat/indicwav2vec-hindi"

        @modal.enter()
        def startup(self) -> None:
            import os

            os.environ.setdefault("HF_HOME", "/cache/hf")
            self.probe = RetroflexProbe(align_model=self.align_model_name)
            self.probe.load()

        @modal.method()
        def score(self, audio_bytes: bytes, text: str, script: str | None = None) -> dict:
            report = self.probe.score(audio_bytes, text, script=script)
            return report.to_dict()

    return app, ModalRetroflexProbe


# -----------------------------------------------------------------------------
# CLI — for quick local inspection of the text-side scanner.
#
#     uv run python -m evaluation.psp --text "విశ్వవిద్యాలయం" --script telugu
#
# Emits JSON with expected retroflex positions; no audio model required.
# -----------------------------------------------------------------------------


def _cli_main() -> None:  # pragma: no cover — human-driven
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Inspect expected retroflexes in a text.")
    ap.add_argument("--text", required=True)
    ap.add_argument("--script", default=None)
    args = ap.parse_args()

    positions = expected_retroflex_positions(args.text, script=args.script)
    bups_view = bups_retroflex_phonemes_in(args.text, script=args.script)
    report = {
        "text": args.text,
        "script": args.script,
        "n_retroflex_expected": len(positions),
        "grapheme_view": [
            {"char": p.char, "phoneme": p.phoneme, "char_index": p.char_index}
            for p in positions
        ],
        "bups_view": bups_view,
        "agrees": [p.phoneme for p in positions] == bups_view,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    _cli_main()
