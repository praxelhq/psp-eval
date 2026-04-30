# PSP: An Interpretable Per-Dimension Accent Benchmark for Indic Text-to-Speech

**Working title.** Target: Interspeech 2026 rolling / arXiv v1 by D10 (2026-04-30).

## Thesis

Existing TTS evaluation — WER, CER, UTMOS, MOS — answers "is it intelligible" and "does it sound natural," but **does not measure accent**. A synthesiser can score well on all four and still sound like a non-native speaker. For languages whose phonology contains features systematically collapsed by non-native speech (retroflex vs. dental stops, aspiration, vowel length, Tamil retroflex approximant /ɻ/), accent is a first-order quality dimension that the field has not benchmarked.

**PSP (Phoneme Substitution Profile)** is an automatic, interpretable, per-phonological-dimension benchmark for Indic TTS. For each dimension where native speakers systematically differ from L2 speakers, PSP estimates the rate at which a TTS system's output collapses the native form toward its L2 substitute — via forced-alignment + native-speaker-centroid acoustic probes over Wav2Vec2-XLS-R mid-layer embeddings.

## Contributions

1. **Formal definition** of per-dimension substitution rates for Indic phonology: retroflex→dental (RR), aspiration loss (AF), vowel-length merger (LF), conjunct epenthesis (CER), and Tamil zha fidelity (ZF).
2. **Probe-based measurement**: prototype-centroid similarity over XLS-R layer-9 embeddings, aligned via CTC forced-alignment. No MFA models required; reproducible from open checkpoints.
3. **Released native reference set** — centroids per phoneme per language (te, hi, ta), bootstrapped from IndicTTS + Rasa.
4. **Calibration study**: ρ against native-speaker MOS on 50 utterances × 5 raters.
5. **Benchmark table**: Parler-TTS, Chatterbox, IndicF5, ElevenLabs v3, Cartesia Sonic-3, Sarvam, Praxy Voice — first time these systems are compared on accent dimensions directly.
6. **Open-source leaderboard**: PyPI package + HF dataset card + GitHub leaderboard.

## Related work — differentiation (verbatim)

- **PSR (arXiv 2601.14417, Jan 2026)** introduced Phoneme Shift Rate for American↔British English speaker-embedding disentanglement. PSR is rule-based and English-specific; PSP is acoustic-probe-based and Indic-first. PSR is a single scalar; PSP decomposes per phonological dimension.
- **Pairwise Accent Similarity (arXiv 2505.14410)** uses PPG + vowel-formant distances. Not interpretable per-dimension; not Indic-specific.
- **Rasmalai (AI4Bharat, IS 2025)** — dataset of 13k-hr Indic audio with accent labels, evaluated via MUSHRA. Dataset, not metric. We reuse a native subset as reference corpus.
- **Sanas AI** — accent generator (commercial), not a published evaluator. Adjacent prior art only.
- **Frechet Speech Distance (arXiv 2601.21386)** — single-scalar quality metric; we include FAD as a companion component (ANS.1) but argue PSP is the interpretable complement.

## Method

### 1. Per-dimension definitions (§3)
- **Retroflex rate (RR).** For each expected retroflex phoneme (ṭ ḍ ṇ ṣ ḷ in BUPS), compute `native_centroid_sim / (native_centroid_sim + dental_centroid_sim)` over the aligned frame span. Fidelity in [0,1]; 1.0 = native.
- **Aspiration fidelity (AF).** For each expected aspirated stop (kh gh ph bh), cosine-sim between probe embedding and aspirated-centroid vs. unaspirated-centroid.
- **Vowel length fidelity (LF).** Per long/short vowel (a/ā, i/ī, u/ū): MAE between measured phoneme duration and native-distribution mean for that phoneme.
- **Tamil zha fidelity (ZF).** For ழ (Tamil zha): probe embedding similarity to retroflex-approximant centroid vs. /l/ centroid.
- **Conjunct epenthesis rate (CER_conj).** Per conjunct cluster (क्ष, శ్చ), detect inserted schwa via duration/energy probe.

### 2. Native centroid construction (§4)
- For each language, sample N = 80 native clips from IndicTTS Telugu / IndicTTS Tamil / Rasa Hindi.
- Force-align with language-specific CTC models (`anuragshas/wav2vec2-large-xlsr-53-telugu`, `Harveenchadha/vakyansh-wav2vec2-tamil-tam-250`, `ai4bharat/indicwav2vec-hindi`).
- Extract XLS-R layer-9 embeddings over aligned phoneme spans; centroid = mean per phoneme.

### 3. Scoring pipeline (§5)
- Input: (audio, ground-truth text, language).
- Expected retroflex positions scanned from text (native-script table).
- Force-align text → audio → per-character time spans.
- XLS-R embedding per retroflex span → cosine sim vs. native-centroid and dental-centroid → per-position fidelity.
- Aggregate: PSP-RR = mean over positions in utterance; corpus-level = mean over utterances.

### 4. Calibration (§6)
- 50 utterances × 5 TTS systems × 5 MOS raters (native speakers, Prolific).
- Report Pearson ρ between PSP-RR and MOS accent-naturalness rating, system-level and utterance-level.
- Report inter-rater reliability (Krippendorff's α).

## Experiments

### Systems benchmarked (§7)
| System | License | RR Te | RR Hi | RR Ta | AF Hi | LF all | ZF Ta |
|---|---|---|---|---|---|---|---|
| Parler-TTS | Apache-2.0 | | | | | | |
| Chatterbox (zero-shot) | MIT | | | | | | — |
| IndicF5 | Apache-2.0 | | | | | | |
| ElevenLabs v3 | commercial | | | | | | |
| Cartesia Sonic-3 | commercial | | | | | | |
| Sarvam Bulbul | proprietary | | | | | | |
| Praxy Voice (ours, fine-tuned) | Apache-2.0 | **bold** | **bold** | **bold** | | | |
| Human reference (native) | — | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |

Expected reading: existing open-source systems have RR-Te ≈ 0.3-0.5 (significant retroflex collapse); commercial systems 0.6-0.8; Praxy 0.8+ after LoRA fine-tune.

## Release

- **Code**: `evaluation/psp.py` + `evaluation/psp_bootstrap.py` in github.com/praxelhq/praxy-tts under MIT.
- **Centroids**: HF dataset `Praxel/psp-native-centroids` under CC-BY.
- **Leaderboard**: `praxel.in/psp-leaderboard` with submission pipeline.
- **Install**: `pip install git+https://github.com/praxelhq/psp-eval.git` (PyPI publish planned post-paper-acceptance).

## Limitations

- Prototype-centroid approach; an MFA-trained native acoustic model (not yet available for Telugu at production quality) would give sharper per-dimension signals.
- 80-clip reference sets per language; scale to 500+ in v2.
- Current benchmark covers Telugu / Hindi / Tamil; Kannada / Malayalam / Bengali / Gujarati scaffolding present in code, reference sets TBD.
- Calibration against MOS only; perceptual studies with trained phoneticians would strengthen.

## Open questions

- Optimal XLS-R layer for each dimension (we use layer 9 for retroflex; aspiration may prefer earlier layers — phonology probing literature suggests middle-early, layers 5-8).
- How much does the alignment-model mismatch affect measurement? Char-level CTC is a practical choice — a phoneme-level aligner could yield more precise spans.

## Timeline to v1

- D0-D2 (now): bootstrap all 3 langs, wire into scorecard, pilot on 3 systems.
- D3-D5: aspiration + length dimensions, benchmark 7 systems.
- D6-D7: calibration study (50 utt × 5 raters, Prolific).
- D8-D10: write-up, ArXiv submission.
