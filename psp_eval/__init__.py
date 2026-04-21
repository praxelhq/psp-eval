"""psp-eval — interpretable per-phonological-dimension accent benchmark for Indic TTS.

The user-facing API for PSP. Power-users who want the full internals can
still import from ``evaluation.psp`` / ``evaluation.fad`` / ``evaluation.psd``;
this thin wrapper exists to make the common case trivially easy.

Quick start::

    from psp_eval import score_directory

    scores = score_directory("path/to/wavs", language="te")
    print(scores["retroflex_fidelity"])
    print(scores["aspiration_fidelity"])
    print(scores["length_fidelity"])
    print(scores["fad"])
    print(scores["psd"])

Each wav file in the directory must have a sibling `.txt` file containing
the expected transcript, OR a `manifest.json` in the directory mapping
utterance IDs to reference text (Praxy-style).
"""

from __future__ import annotations

from psp_eval.scorer import score_directory, score_clip

__all__ = ["score_directory", "score_clip"]
__version__ = "0.1.0"
