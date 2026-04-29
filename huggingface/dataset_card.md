---
license: cc-by-4.0
language:
- te
- hi
- ta
task_categories:
- text-to-speech
- audio-classification
tags:
- accent-evaluation
- phoneme-probe
- indic
- retroflex
- wav2vec2
- fad
- psd
size_categories:
- 1K<n<10K
---

# Praxel/psp-native-centroids

Native-speaker reference artefacts for the **PSP** (Phoneme Substitution Profile)
benchmark for Indic text-to-speech accent evaluation. Companion to the paper
[PSP: An Interpretable Per-Dimension Accent Benchmark for Indic Text-to-Speech](https://arxiv.org/abs/2604.25476)
(Menta, 2026).

This dataset is a **scoring reference**, not a training corpus. It contains
pre-computed acoustic references extracted from publicly-licensed native-speaker
speech corpora, used by the [`psp-eval` PyPI package](https://github.com/praxelhq/psp-eval)
to score TTS outputs on six accent dimensions.

## Contents

Per-language files for Telugu (`te`), Hindi (`hi`), and Tamil (`ta`):

| File | Shape / size | Description |
|---|---|---|
| `{lang}_refs.pkl` | `{phoneme: [ndarray (1024,)]}` | Per-phoneme Wav2Vec2-XLS-R layer-9 centroid bags (500-clip bootstrap) |
| `{lang}_fad_natives.pkl` | `ndarray (1000, 1024)` | Utterance-level XLS-R embeddings for FAD computation |
| `{lang}_psd_natives.pkl` | `ndarray (500, 5)` | Prosodic feature vectors (F0 mean/std/range, onset-rate, nPVI) for PSD |
| `{lang}_sanity.json` | small JSON | Held-out native-audio sanity-check scores (§6 paper Signal 5) |

## Provenance

All centroids and reference distributions are derived from:
- **Telugu**: [IndicTTS](https://www.iitm.ac.in/donlab/tts/) (Telugu subset) — CC-BY-4.0
- **Hindi**: [Rasa](https://github.com/AI4Bharat/Rasa) (Hindi subset) — CC-BY-4.0
- **Tamil**: IndicTTS (Tamil subset) — CC-BY-4.0

500 clips per language sampled from the full corpus with seed `1337`. FAD references
sample 1000 clips from the same pool with the same seed. PSD references sample 500
clips. Held-out sanity-check clips sample from the same pool with disjoint seed `999`.

Each pickle was produced by `evaluation/psp_bootstrap.py` in the
[praxelhq/psp-eval repository](https://github.com/praxelhq/psp-eval); see that
script for the exact extraction pipeline and the alignment-model checkpoints used.

## Usage

```python
from psp_eval import score_directory

# Centroids auto-download from this repo on first use.
scores = score_directory("my_tts_outputs/", language="te")
```

Or load directly in Python:

```python
import pickle
from huggingface_hub import hf_hub_download

path = hf_hub_download("Praxel/psp-native-centroids", "te_refs.pkl", repo_type="dataset")
with open(path, "rb") as f:
    refs = pickle.load(f)
# refs: {"ṭ": [np.ndarray (1024,), ...], "ḍ": [...], ...}
```

## Known caveats

- **Per-phoneme probe noise floor**: native Telugu / Tamil audio registers
  0.47–0.54 retroflex fidelity when scored against these centroids (not 1.0).
  This reflects speaker variance between centroid and held-out native corpora,
  aligner quality, and the strictness of the 0.5 collapse threshold. Interpret
  per-phoneme scores as **relative rankings across systems**, not absolute
  distances from a theoretical 1.0 ceiling. See paper §6 Signal 5 for details.
- **FAD / PSD** do not share this noise floor (native audio correctly scores
  5–50× lower than commercial-TTS outputs).
- **Unnormalised Fréchet across mixed-scale PSD dimensions**: nPVI has numeric
  range ~$10^2$ while log-$F_0$ is ~$10^0$. A z-scored variant is planned for
  the v2 release.

## Citation

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

## License

CC-BY-4.0 — matching the originating corpus licenses (IndicTTS, Rasa).

## Related

- **Code**: https://github.com/praxelhq/psp-eval (MIT)
- **PyPI**: `pip install psp-eval`
- **Paper**: https://arxiv.org/abs/2604.25476

## Contact

Pushpak Teja — pushpak@praxel.in — [praxel.in](https://praxel.in)
