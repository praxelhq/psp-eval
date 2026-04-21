# psp-eval

**PSP: An interpretable per-dimension accent benchmark for Indic text-to-speech.**

[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/TBD)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Centroids: CC-BY](https://img.shields.io/badge/centroids-CC--BY--4.0-green.svg)](https://huggingface.co/datasets/Praxel/psp-native-centroids)

Standard TTS evaluation measures **intelligibility** (WER, CER) and **overall naturalness** (MOS, UTMOS) but does not quantify **accent**. A synthesiser may score well on all four yet sound non-native on features that are phonemic in the target language.

**PSP** (Phoneme Substitution Profile) decomposes accent into six interpretable dimensions:

| # | Metric | What it captures | Native = |
|---|---|---|---|
| 1 | **Retroflex fidelity (RR)** | ṭ/t, ḍ/d, ṇ/n, ṣ/s, ḷ/l distinctions in Indic phonology | 1.0 |
| 2 | **Aspiration fidelity (AF)** | kh/k, gh/g, bh/b, dh/d, ph/p distinctions (Hindi-primary) | 1.0 |
| 3 | **Length fidelity (LF)** | Long/short vowel duration ratio (ā vs a, ī vs i, ū vs u) | 1.0 |
| 4 | **Tamil-zha fidelity (ZF)** | Tamil ழ /ɻ/ vs /l/ collapse | 1.0 |
| 5 | **Fréchet Audio Distance (FAD)** | Corpus-level distributional distance in XLS-R space | 0.0 |
| 6 | **Prosodic Signature Divergence (PSD)** | F0 contour + syllable rate + rhythm (nPVI) | 0.0 |

## Quick start

```bash
pip install psp-eval
```

```python
from psp_eval import score_directory

scores = score_directory("path/to/my_tts_outputs/", language="te")
print(scores)
# {'n_clips': 20,
#  'retroflex_fidelity': 0.787,
#  'retroflex_collapse_rate': 0.333,
#  'aspiration_fidelity': ...,
#  'length_fidelity': ...,
#  'zha_fidelity': ...,
#  'fad': 250.4,
#  'psd': 11.06}
```

Each `.wav` in the directory is scored against its expected transcript. Transcripts are resolved from either:
1. A `manifest.json` in the directory with `{utterances: [{id, text}]}` schema, or
2. Sibling `.txt` files next to each `.wav`.

## Supported languages (v1)

`te` (Telugu), `hi` (Hindi), `ta` (Tamil). Extension to Kannada / Malayalam / Bengali / Gujarati / Punjabi planned for v2.

## Preliminary benchmark (v1, pilot sets)

A selection from the released `paper/benchmark_results.json`:

| System | Retroflex collapse (Te) | FAD (Te) | PSD (Te) |
|---|---|---|---|
| Sarvam Bulbul | **33.3%** | **250** | 11.1 |
| Indic Parler-TTS | **33.3%** | 325 | **10.4** |
| Praxy R5 (ours) | 40.0% | 534 | 14.1 |
| ElevenLabs v3 | 40.0% | 329 | 154.4 |
| Cartesia Sonic-3 | 50.0% | 458 | 33.8 |

Key finding: **PSP ordering diverges from WER ordering.** Commercial WER-leaders do not uniformly lead on retroflex or prosodic fidelity. See the paper for Hindi / Tamil tables + analysis.

## Repository structure

```
psp-eval/
├── README.md              # You are here
├── LICENSE                # MIT (code) · centroids CC-BY-4.0
├── pyproject.toml
├── psp_eval/              # pip-installable user-facing package
│   ├── __init__.py
│   ├── scorer.py          # score_directory(), score_clip()
│   └── pyproject.toml
├── evaluation/            # Research-grade modules (imported by psp_eval)
│   ├── psp.py             # RetroflexProbe, AspirationProbe, LengthProbe + helpers
│   ├── fad.py             # Fréchet Audio Distance (corpus-level)
│   ├── psd.py             # Prosodic Signature Divergence (corpus-level)
│   ├── psp_bootstrap.py   # Native centroid + native-reference extraction
│   ├── modal_psp.py       # Modal-backed scorer (GPU batch mode)
│   ├── build_300_utt_golden.py  # Test set construction
│   ├── test_psp.py        # 34 unit tests
│   ├── ACCENT.md          # Design spec for all 6 dimensions
│   └── golden_test_sets/  # Te/Hi/Ta × (300-utt golden + 10-utt smoke)
├── praxy/linguistics/
│   └── bups.py            # Brahmic Unified Phoneme Space (shared dep)
├── paper/
│   ├── psp.tex            # LaTeX source (IEEEtran)
│   ├── refs.bib           # Bibliography
│   ├── benchmark_results.json  # All v1 numbers (reproducibility artefact)
│   └── README.md          # How to build the PDF
└── huggingface/
    └── dataset_card.md    # For Praxel/psp-native-centroids release
```

## Running local backend

The PyPI package's `backend="local"` runs the probe on your machine. First time it downloads centroids from Hugging Face (`Praxel/psp-native-centroids`, ~400 MB per language) to `~/.cache/psp_eval/`.

CPU works but is slow (~5-10s per clip for per-phoneme probes). GPU (CUDA or MPS) recommended for directory-scale runs.

## Running Modal backend (recommended for large runs)

```bash
pip install psp-eval[modal]
modal setup  # one-time auth
```

```python
from psp_eval import score_directory

scores = score_directory("my_wavs/", language="te", backend="modal")
```

The Modal backend offloads scoring to GPU containers with pre-loaded models. ~10-30× faster than local for directory-scale evaluation. Uses Modal's free-tier credits for light use.

## Building the paper PDF

```bash
cd paper/
pdflatex psp
bibtex psp
pdflatex psp
pdflatex psp  # second pass for forward refs
```

See `paper/README.md` for details.

## Citation

If you use PSP in your research, please cite:

```bibtex
@misc{teja2026psp,
  title={{PSP}: An Interpretable Per-Dimension Accent Benchmark for Indic Text-to-Speech},
  author={Teja, Pushpak},
  year={2026},
  eprint={TBD},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Honest caveats (v1)

- **Smoke-set scale**: v1 benchmarks use 10-utterance pilot sets per (system, language). Full 300-utterance benchmarks land in v2.
- **Noise floor on per-phoneme probes for Telugu/Tamil**: on native audio, Telugu retroflex fidelity registers 0.54 (not 1.0); Tamil 0.47. Hindi registers 1.00 (perfect). Likely driven by aligner quality variance. FAD and PSD do not share this noise floor. Treat per-phoneme scores as **relative rankings across systems**, not absolute distances from a 1.0 ceiling. See paper §6 for details.
- **MOS correlation**: formal human-perception calibration is deferred to v2. v1 reports five internal-consistency signals supporting metric validity.

## Related work

- **PSR** ([Lertpetchpun et al., ICASSP 2026](https://arxiv.org/abs/2601.14417)) introduced the Phoneme Shift Rate for American↔British English — conceptually adjacent, but rule-based and English-only. PSP complements it for Indic.
- **FAD** ([Kilgour et al., 2019](https://arxiv.org/abs/1812.08466)) — foundational corpus-level audio metric. Used as one of our 6 dimensions.
- **UTMOS** ([Saeki et al., 2022](https://arxiv.org/abs/2204.02152)), **VoiceMOS Challenge** ([Huang et al., 2022](https://arxiv.org/abs/2203.11389)) — established MOS-prediction baselines. Orthogonal to PSP; measures overall quality, not per-phonological-dimension accent.

## License

- **Code**: MIT (see `LICENSE`)
- **Native-speaker centroids + reference data**: CC-BY-4.0 at [Praxel/psp-native-centroids](https://huggingface.co/datasets/Praxel/psp-native-centroids) (release upon paper acceptance)
- **Paper text**: CC-BY-4.0 (arXiv)

## Acknowledgments

PSP v1 was developed independently without external API credit grants. Thanks to Aakanksha Naik for arXiv endorsement; thanks to the IndicTTS, Rasa, and FLEURS teams for the native-reference corpora this benchmark is built on.

## Contact

Pushpak Teja — pushpak@praxel.ai — [praxel.ai](https://praxel.ai)
