# psp-eval

Reference implementation and benchmark for **PSP: An Interpretable Per-Dimension Accent Benchmark for Indic Text-to-Speech** (Teja, 2026).

[![Paper](https://img.shields.io/badge/paper-arXiv:2604.25476-b31b1b.svg)](https://arxiv.org/abs/2604.25476)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Centroids: CC-BY](https://img.shields.io/badge/centroids-CC--BY--4.0-green.svg)](https://huggingface.co/datasets/Praxel/psp-native-centroids)

Standard text-to-speech evaluation measures intelligibility (WER, CER) and overall naturalness (MOS, UTMOS) but does not quantify accent. A synthesiser may score well on all four and still sound non-native on features that are phonemic in the target language.

PSP (Phoneme Substitution Profile) decomposes accent into six interpretable dimensions:

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

## Preliminary results (v1, pilot sets)

Selected Telugu results from `paper/benchmark_results.json`:

| System | RR collapse | FAD | PSD |
|---|---|---|---|
| Sarvam Bulbul | 33.3% | 250 | 11.1 |
| Indic Parler-TTS | 33.3% | 325 | 10.4 |
| Praxy Voice R5 (ours, in progress) | 40.0% | 534 | 14.1 |
| ElevenLabs v3 | 40.0% | 329 | 154.4 |
| Cartesia Sonic-3 | 50.0% | 458 | 33.8 |

The principal observation is that PSP ordering is not monotonic with WER ordering: systems that lead on intelligibility do not uniformly lead on retroflex or prosodic fidelity. See the paper for full Hindi / Tamil tables and cross-language analysis.

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
  author={Menta, Venkata Pushpak Teja},
  year={2026},
  eprint={2604.25476},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.25476}
}
```

## Limitations (v1)

- **Pilot-scale benchmarks.** v1 reports numbers on 10-utterance pilot sets per (system, language). Full benchmarks on the 300-utterance golden sets released with this repository appear in v2.
- **Per-phoneme probe noise floor on Telugu and Tamil.** Held-out native audio registers retroflex fidelity of 0.54 on Telugu and 0.47 on Tamil (not 1.0); Hindi native audio registers 1.00. The discrepancy is consistent with language-specific CTC aligner quality: our Hindi aligner (AI4Bharat) is trained on substantially larger and cleaner data than the community Telugu and Tamil aligners. Per-phoneme scores should be interpreted as relative rankings across systems evaluated on the same test set, not as absolute distances from a theoretical 1.0 ceiling. FAD and PSD are not affected and remain absolute-interpretable. See paper §6.
- **MOS correlation deferred to v2.** v1 supplies five internal-consistency signals supporting metric validity. Formal human-perception calibration is a v2 deliverable.

## Related work

- **PSR** ([Lertpetchpun et al., ICASSP 2026](https://arxiv.org/abs/2601.14417)) introduced the Phoneme Shift Rate for American↔British English — conceptually adjacent, but rule-based and English-only. PSP complements it for Indic.
- **FAD** ([Kilgour et al., 2019](https://arxiv.org/abs/1812.08466)) — foundational corpus-level audio metric. Used as one of our 6 dimensions.
- **UTMOS** ([Saeki et al., 2022](https://arxiv.org/abs/2204.02152)), **VoiceMOS Challenge** ([Huang et al., 2022](https://arxiv.org/abs/2203.11389)) — established MOS-prediction baselines. Orthogonal to PSP; measures overall quality, not per-phonological-dimension accent.

## License

- **Code**: MIT (see `LICENSE`)
- **Native-speaker centroids + reference data**: CC-BY-4.0 at [Praxel/psp-native-centroids](https://huggingface.co/datasets/Praxel/psp-native-centroids) (release upon paper acceptance)
- **Paper text**: CC-BY-4.0 (arXiv)

## Acknowledgments

PSP v1 was developed independently without external API credit grants. Native reference data is derived from the IndicTTS, Rasa, and FLEURS speech corpora, used under their respective licenses.

## Contact

Pushpak Teja — pushpak@praxel.in
