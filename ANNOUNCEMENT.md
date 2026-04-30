# Launch announcement drafts for ArXiv v1 release

Short-form drafts for social / network sharing. Customize tone per platform.

---

## X / Twitter thread (5 posts)

**Tweet 1 (hook)**
New preprint: **PSP — an interpretable per-dimension accent benchmark for Indic TTS** 🧵

WER tells you if the words are right. PSP tells you if the *accent* is right. Turns out those two questions have surprisingly different answers.

paper: [arxiv link]
code: github.com/praxelhq/psp-eval

**Tweet 2 (the problem)**
Indic TTS has a silent quality gap that WER doesn't capture. ElevenLabs Hindi WER is <1%, yet the prosody is noticeably flat. Sarvam's FAD is 212 on Hindi but 267 on Cartesia. Same intelligibility, very different "native-ness".

**Tweet 3 (the method)**
PSP decomposes accent into 6 measurable dimensions:
• Retroflex fidelity (ṭ/t collapse)
• Aspiration fidelity (kh/k collapse)
• Vowel-length fidelity
• Tamil-zha fidelity (ழ → /l/)
• FAD (corpus-level timbre/co-articulation)
• PSD (F0 + rhythm divergence)

**Tweet 4 (the finding)**
Benchmarked 4 commercial + open-source TTS on Te/Hi/Ta. Retroflex collapse grows monotonically with language difficulty:
• Hindi ~1%
• Telugu ~40%
• Tamil ~68%

No single system is Pareto-optimal across all 6 dimensions. Indic-first systems (Sarvam, Parler) generalise better to Tamil than Western-built ones.

**Tweet 5 (the ask)**
v1 is preliminary (10-utt pilot sets, no MOS yet). v2 coming ~6 weeks with 300-utt + MOS calibration.

If you build Indic TTS, I'd love to include your system in v2 — drop me a line.

Centroids on HF: huggingface.co/datasets/Praxel/psp-native-centroids

---

## LinkedIn post (longer form, paragraph)

🎙️ New preprint out: **PSP — a benchmark that measures Indic accent in TTS, not just word-accuracy.**

Modern text-to-speech for Indian languages is surprisingly good on intelligibility. ElevenLabs, Cartesia, and Sarvam all achieve sub-5% WER on Hindi. Yet anyone who listens can tell something's off, especially for Telugu and Tamil — the retroflexes blur, the pitch range collapses, the rhythm is slightly wrong. These are *phonemic* differences that word-error-rate completely misses.

We propose **PSP** (Phoneme Substitution Profile) — an interpretable, per-phonological-dimension benchmark that decomposes accent into six measurable components: retroflex fidelity, aspiration fidelity, vowel-length fidelity, Tamil-zha fidelity, Fréchet Audio Distance, and prosodic signature divergence. The first four use acoustic probes over Wav2Vec2-XLS-R embeddings with native-speaker centroid references. The last two are corpus-level distributional distances.

Three findings from v1:
1. **Retroflex collapse grows with language difficulty**: ~1% on Hindi, ~40% on Telugu, ~68% on Tamil. Matches community knowledge; validates the metric.
2. **PSP ordering ≠ WER ordering**. ElevenLabs leads WER on Hindi but places second on FAD; Cartesia places last on FAD despite strong WER.
3. **No single system is Pareto-optimal** across all six dimensions. Per-dimension decomposition reveals system-specific failure modes.

Everything open-source: code (MIT), centroids (CC-BY-4.0), 300-utt golden test sets, reproducibility artefact. Install from source with `pip install git+https://github.com/praxelhq/psp-eval.git` (PyPI publish planned post-paper-acceptance).

v2 with formal MOS correlation + 300-utt full benchmarks planned for June 2026.

If you build Indic TTS and want your system included in v2, please reach out.

Paper → [arxiv link]
Code → github.com/praxelhq/psp-eval
Centroids → huggingface.co/datasets/Praxel/psp-native-centroids

#TTS #IndianLanguages #SpeechSynthesis #MachineLearning #AI #Research

---

## Email to AI4Bharat / Sarvam / ElevenLabs / Cartesia (post-upload)

**Subject**: Published: PSP benchmark for Indic TTS — your system featured; v2 scaling ask

Hi [name],

I published a preprint today on an accent-evaluation benchmark for Indic TTS that includes [system name] among four benchmarked systems: [arxiv link]. Your system [brief honest summary: "leads on FAD for all three languages" / "top on retroflex collapse for Telugu" / etc.].

v1 uses 10-utterance pilot sets. For v2 (target ~6 weeks out), I'm scaling to the 300-utterance held-out golden sets released with v1 + adding MOS calibration via Karya.

Small ask: would your team be willing to issue a temporary rate-limit exemption (no paid quota needed; I have trial credits) for a ~2-week synthesis window? This is the only thing blocking a larger-N v2 run for [system name].

Full disclosure: I built PSP independently; no vendor has reviewed it. v2 will acknowledge any resources you provide in that version's Acknowledgments section. No strings.

Happy to share the draft ahead of v2 upload if helpful.

Best,
Pushpak Teja
Praxel Ventures · [pushpak@praxel.in](mailto:pushpak@praxel.in)

---

## Praxel internal blog post (longer, optional)

**Announcing PSP: a new benchmark for Indic accent in TTS**

[~500 words, to be drafted closer to launch]

Key points:
- Why accent ≠ intelligibility
- Why Indic specifically
- What makes PSP different from prior metrics
- Preliminary findings + honest caveats
- How to use it (pip install, score your TTS)
- Roadmap to v2
