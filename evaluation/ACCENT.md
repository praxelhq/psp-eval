# Accent Naturalness Score (ANS) — spec + implementation status

**Status (2026-04-21):** v1 shipping. All six dimensions implemented and tested; preliminary tri-lingual benchmark across 4-5 systems complete. See `paper/psp/psp.tex` for paper draft.

**Implementation status per dimension**:
- ✅ Retroflex fidelity (RR) — `evaluation/psp.py::RetroflexProbe`, 16 tests passing
- ✅ Aspiration fidelity (AF) — `evaluation/psp.py::AspirationProbe`, 8 tests passing
- ✅ Length fidelity (LF) — `evaluation/psp.py::LengthProbe`, 10 tests passing
- ✅ Tamil-zha fidelity (ZF) — sub-dimension of RR via `per_phoneme_breakdown()`
- ✅ FAD — `evaluation/fad.py`, 3 smoke tests passing, 1000-clip native refs for Te/Hi/Ta
- ✅ PSD — `evaluation/psd.py`, 3 smoke tests passing, 500-clip native refs for Te/Hi/Ta
- ⏳ Conjunct epenthesis (CER) — scaffolded in text-scanning tables; probe deferred to v2
- ⏳ Formal MOS calibration — deferred to v2

## Related work (must-cite, updated 2026-04-21)

- **Quantifying Speaker Embedding Phonological Rule Interactions (PSR)** — arXiv 2601.14417 (Jan 2026). Introduces the *Phoneme Shift Rate* metric for American↔British English disentanglement. Conceptual sibling of PSP, but:
  - **English-only**, rule-based (flapping, rhoticity, vowel mapping); PSP is **Indic-first** with acoustic probes.
  - PSR is a *disentanglement probe* for speaker embeddings; PSP is a **TTS-system evaluation benchmark** with released artifacts.
  - PSR produces a single scalar; PSP decomposes into **per-phonological-dimension** scores (retroflex / aspiration / length / zha).
  - Both should coexist — PSR for English accent research, PSP for Indic. We cite PSR prominently to frame PSP as orthogonal.
- **Learning-free L2-Accented Speech Generation** — arXiv 2603.07550 (2026). Phonological-rule-based *generator*, not an evaluator. Different artifact.
- **Pairwise Evaluation of Accent Similarity** — arXiv 2505.14410 (Interspeech 2025). PPG + vowel-formant distance; single-number metric, not per-dimension.
- **Rasmalai** (AI4Bharat, Interspeech 2025). 13k-hr Indic *dataset* with accent/intonation descriptions; uses MUSHRA human eval — we use its native subset as our reference corpus.
- **Frechet Speech Distance** — arXiv 2601.21386. Validates FAD for TTS quality but not accent-specific; our FAD component (ANS.1) aligns.
- **Sanas AI** — real-time accent *conversion* generator, 50M-utterance training set, no published evaluation metric. Adjacent commercial prior-art, not prior publication.

**No one has published:** per-phonological-dimension substitution rates for Indic TTS, native-centroid probes using XLS-R mid-layer embeddings, retroflex-collapse/aspiration-loss/length-MAE/zha-fidelity as named, released metrics, or the term *Phoneme Substitution Profile*. That's our wedge.

## The gap we're filling

Traditional TTS evaluation measures intelligibility (WER/CER) and overall naturalness (MOS / UTMOS). Neither captures **accent**. A model can be fully intelligible ("foreigner speaking Telugu" — Pushpak's first-listen verdict on Chatterbox) yet scorebadly on nativeness. MOS raters do pick up accent subjectively, but there's no **automatic, reproducible accent benchmark** for Indic TTS — a gap big enough to be its own contribution.

The Praxy paper's narrative is strongest if we can say: "our fine-tuned model not only transcribes correctly (LLM-WER) but is indistinguishable from native speakers by our accent benchmark." That requires the benchmark to exist.

## Indic accent — what actually varies

Accent is not one dimension. For Indian languages, these are the systematic signatures a non-native speaker drops or flattens:

| Dimension | Native signature | Non-native collapse | Languages most affected |
|---|---|---|---|
| **Retroflex consonants** ṭ ḍ ṇ ṣ ḷ | Clean retroflex articulation | Collapsed to dental (/t/, /d/, /n/) or alveolar | All Brahmic — especially Telugu, Tamil |
| **Aspiration** kh gh ph bh | Distinct aspirated burst | Collapsed to unaspirated | Hindi, Bengali |
| **Vowel length** a/ā i/ī u/ū | Phonemic (words differ by length) | Length-merged | All Indic |
| **Schwa deletion** | Hindi/Marathi drop word-final schwa (राम = /ɾaːm/ not /ɾaːmə/) | Schwa retained as inherent 'a' | Hindi, Marathi |
| **Gemination** | Doubled consonants are phonemic (Tamil poṇṇu vs poṇu) | Single consonant | Tamil, Telugu |
| **Uniquely Tamil** ழ (zha) | /ɻ/ — retroflex approximant | Mapped to /l/ or /r/ | Tamil only |
| **F0 contour** | Language-specific: Telugu drops on declaratives, Tamil has syllable-timed rhythm, Hindi has lexical tone in some dialects | English stress-timed pattern imposed | All |
| **Conjuncts** (क्ष, శ్చ) | Fluent cluster articulation | Inserted schwa / epenthesis | All |

These are *measurable* — not just "sounds foreign." That's the opening.

## Proposed compound Accent Naturalness Score (ANS)

Three independent components, each normalized to [0, 1] where 1.0 = native. Composite ANS is the weighted mean.

### Component 1 — Distributional nativeness: Fréchet Audio Distance (FAD)

Fréchet distance between the generated audio's embedding distribution and a native-speaker reference distribution, in a self-supervised audio embedding space (Wav2Vec2-XLS-R-300m works well for Indic since its pretraining includes IndicSUPERB).

- For each language, build a native reference set: ~50h per language from IndicTTS + Rasa (native speakers, studio quality).
- Compute mean + covariance of Wav2Vec2-XLS-R embeddings over frames.
- For a candidate TTS, compute its mean + covariance over the same evaluation set, then Fréchet distance.
- **Lower is better.** Normalize by (FAD of silence-padding baseline) so 0 = perfect match, 1 = as bad as silence.

Strengths: captures everything — mel, phoneme, prosody — in one number. Known metric from image (FID) and audio (FAD-original) literature.
Weaknesses: not interpretable per-dimension. Needs a clean native reference set.

### Component 2 — Phoneme substitution profile (PSP)

Force-align the generated audio to its ground-truth text using Montreal Forced Aligner (MFA) with a **native-speaker-trained Indic acoustic model**. Compare the actual acoustic features at each expected-phoneme position against the native model's expected distribution for that phoneme.

For each of the dimensions above (retroflex, aspiration, vowel length, etc.), compute a **substitution rate**: fraction of instances where the generated acoustics are closer to the non-native substitute than to the native target.

- **Retroflex rate (RR)** = 1 − P(dental | expected retroflex in generation).
- **Aspiration fidelity (AF)** = P(aspirated burst | expected aspirated in generation).
- **Length fidelity (LF)** = mean absolute error on phoneme duration vs expected vowel-length prior.
- **Conjunct epenthesis rate (CER_conj)** = fraction of clusters where schwa insertion is detected.
- **Tamil zha fidelity (ZF)** = P(retroflex approximant | ழ present in reference).

Aggregate: PSP = weighted mean of (1 − each substitution rate), weights set so dimensions missing in a language (e.g., ZF doesn't apply to Hindi) don't count.

Strengths: **novel**, per-dimension interpretable, directly targets Indic accent signatures. The paper contribution lives here.
Weaknesses: requires MFA acoustic models fine-tuned on Indic native speech. Buildable — AI4Bharat has released similar models, and we have enough native data to fine-tune if needed.

### Component 3 — Prosodic signature divergence (PSD)

For each language, compute native-speaker distributions of three prosodic features across a reference corpus:
- F0 contour shape (normalized mean contour over sentences, clustered by punctuation type: declarative / question / exclamation)
- Syllable duration distribution
- Inter-syllable energy envelope (speech-rhythm class — syllable-timed vs stress-timed signature)

For a candidate audio, compute the same features and measure **symmetrized KL divergence** against the native reference.

Strengths: captures the "music" of the language — the hardest thing to fake. Telugu's characteristic descending F0, Tamil's even syllable timing, Hindi's Sanskrit-derived intonation patterns.
Weaknesses: needs careful per-category normalization; punctuation-conditioning matters.

## Composite ANS

```
ANS = 0.4 × (1 − FAD_norm) + 0.4 × PSP + 0.2 × PSD
```

(Weights tuned on a calibration set of ~100 utterances with human MOS ratings; initial weights are heuristic.)

Validation: correlate ANS against crowd-sourced MOS ratings on the same audio. If ρ > 0.80 we have a defensible automatic benchmark.

## Implementation plan

Order of operations — each step ships independently.

1. **Native reference corpora curation** — per language, select ~50 hrs from IndicTTS + Rasa where the speaker is a confirmed native. Build a JSONL manifest. (~1 day CPU.)
2. **FAD pipeline** (`evaluation/fad.py`) — Wav2Vec2-XLS-R embedding + mean/cov computation + Fréchet distance. Use existing `fadtk` or `frechet_audio_distance` PyPI package. (~1 day.)
3. **MFA Indic acoustic model training** (`evaluation/mfa_indic.py`) — Bootstrap from English MFA model, adapt with native Indic data. (~3 days.)
4. **PSP computation** (`evaluation/psp.py`) — per-dimension rates using MFA alignments + classifier probes. (~2 days.)
5. **PSD computation** (`evaluation/psd.py`) — prosodic feature extraction + KL divergence. (~1 day.)
6. **Calibration study** — 100 utterances × 5 raters × MOS on a paid panel. Tune ANS weights. (~1 week wall-clock.)
7. **Benchmark release** — publish ANS as a standalone PyPI package + HF dataset + leaderboard page. (~1 week.)

**Total effort: ~3 weeks of focused work.** Sequence after Sprint 5 (so we have a real Praxy model to benchmark).

## Companion-paper angle

Working title: **"ANS: An Automatic Accent Naturalness Score for Indic Text-to-Speech"**

- Section 1: the WER gap — why intelligibility is not enough
- Section 2: Indic accent phonology, systematic non-native signatures
- Section 3: ANS design and components
- Section 4: calibration against human MOS
- Section 5: benchmarking existing Indic TTS (Chatterbox / IndicF5 / Parler / commercial APIs)
- Section 6: case studies — where existing systems fail on specific accent dimensions
- Section 7: release and leaderboard

Venue fit: Interspeech is the natural home. ICASSP works too. Could also be a NeurIPS Evaluations & Datasets Track submission if framed as a benchmark.

## Risks

- **MFA native model quality is the critical dependency.** If the acoustic model isn't well-calibrated on Indic native data, PSP gives noise. Mitigation: we have clean native data (IndicTTS); we just need to train MFA carefully. AI4Bharat has already done related work we can borrow from.
- **Weights (0.4/0.4/0.2) are arbitrary until calibrated.** Calibration study is essential; otherwise the composite is hand-waving.
- **This is a three-week project.** Not tonight, not this week — Sprint 6+. Scheduling discipline matters: accent eval is valuable but secondary to shipping the main contribution.
