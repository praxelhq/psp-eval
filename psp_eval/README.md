# psp-eval

**Interpretable per-phonological-dimension accent benchmark for Indic TTS.**

## Install

```bash
pip install psp-eval
```

## Quick start

```python
from psp_eval import score_directory

scores = score_directory("path/to/my_tts_outputs/", language="te")

print(f"Retroflex fidelity:   {scores['retroflex_fidelity']:.3f}")   # 0-1, native=1
print(f"Aspiration fidelity:  {scores['aspiration_fidelity']:.3f}")
print(f"Length fidelity:      {scores['length_fidelity']:.3f}")
print(f"Tamil-zha fidelity:   {scores['zha_fidelity']}")             # None if language != 'ta'
print(f"FAD (corpus-level):   {scores['fad']:.3f}")                  # lower is better
print(f"PSD (prosodic):       {scores['psd']:.3f}")
```

Each `.wav` in the directory is scored against its expected transcript.
Transcripts are resolved from either:
1. A `manifest.json` with `{utterances: [{id, text}]}` schema, or
2. Sibling `.txt` files next to each wav.

## Dimensions

| Metric | What it measures | Native = |
|---|---|---|
| Retroflex fidelity (RR) | Retroflex vs dental articulation (ṭ/t, ḍ/d, ...) | 1.0 |
| Aspiration fidelity (AF) | Aspirated vs unaspirated stops (kh/k, bh/b, ...) | 1.0 |
| Length fidelity (LF) | Long/short vowel duration ratio preservation (ā/a) | 1.0 |
| Tamil-zha fidelity (ZF) | Tamil ழ /ɻ/ vs /l/ collapse (Tamil only) | 1.0 |
| FAD | Corpus-level Fréchet distance in XLS-R space (timbre + co-articulation) | 0.0 |
| PSD | Prosodic divergence: F0 stats + rhythm (nPVI) + speech rate | 0.0 |

## Supported languages

`te` (Telugu), `hi` (Hindi), `ta` (Tamil). More Indic languages in v2.

## Backends

- `backend="local"` (default): runs on your machine. CPU works but slow (~5-10s/clip).
  GPU 10-30x faster.
- `backend="modal"`: offloads to Modal cloud. Recommended for directory-scale runs.
  Requires `modal` installed + authenticated.

## Cite

If you use PSP in your research, please cite:

```bibtex
@inproceedings{psp2026,
  title={PSP: An Interpretable Per-Dimension Accent Benchmark for Indic Text-to-Speech},
  author={Teja, Pushpak},
  booktitle={Interspeech},
  year={2026}
}
```

## License

MIT. Centroids and reference data under CC-BY-4.0.

## Links

- Paper: *link TBD on ArXiv upload*
- Code: https://github.com/praxelhq/praxy-tts
- Centroids: https://huggingface.co/datasets/Praxel/psp-native-centroids
