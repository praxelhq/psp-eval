# Karya research-partnership outreach — draft

**To**: research@karya.in (or contact form at https://karya.in)
**Subject**: Research partnership for speech-TTS accent benchmark (Indic, Interspeech submission)
**From**: Pushpak Teja, Praxel HQ (pushpak@praxel.in)

---

Hi Karya team,

I'm reaching out to explore a research partnership for a small-scale MOS (Mean Opinion Score) rating study on Indic text-to-speech audio. We're building an open-source accent-evaluation benchmark for Indic TTS (Telugu / Hindi / Tamil) targeting Interspeech 2026, and we need human MOS ratings to calibrate our automatic metric against native-speaker perception.

**Study snapshot**
- **Task**: native speakers listen to a ~5-second audio clip and rate it on a 1-5 scale for *accent naturalness* (does this sound like a native speaker of this language?).
- **Scale**: 50 audio clips × 5 raters × 3 languages (Telugu, Hindi, Tamil) = **750 individual ratings**
- **Rater time per session**: ~2 hours (50 clips × ~2.5 min each, including a short training/attention block)
- **Languages**: Telugu, Hindi, Tamil — raters must be native speakers of the language they rate
- **Audio source**: outputs from 7 publicly benchmarkable TTS systems (e.g., Parler-TTS, Chatterbox, ElevenLabs, Cartesia) plus our own fine-tuned model. All audio is synthesized (no PII, no sensitive content).
- **Content**: neutral declarative and interrogative sentences (e.g., "the library opens at nine"). No politically or religiously sensitive material.

**Timeline**
- Ready-to-launch: **2026-04-28** (1 week from now)
- Target completion: **2026-05-10**
- Analysis + writeup: **2026-05-10 → 2026-05-12** (ArXiv v1 target)

**Budget**
- Up to **USD $500 total** for rater compensation + platform fees
- Flexible on rate structure — we want to meet your ethical-floor wage
- Willing to cover Karya's overhead / administrative cost within this envelope

**Open-science commitment**
- Study protocol, raw anonymized ratings, analysis code, and the full benchmark itself will be released publicly under MIT / CC-BY
- Karya will be acknowledged in paper + release artifacts (with whatever attribution you prefer)
- Aggregate data is intended to benefit the broader Indic speech-research community

**What we'd need from Karya**
- Rater recruitment + scheduling for Te/Hi/Ta native speakers
- Rating UI that supports short audio playback + 1-5 slider + optional free-text comment
- Attention-check quality control (e.g., catch-trial clips of clearly good/bad TTS)
- Anonymized data export (CSV / JSON) post-study

**Our background**
- Praxel HQ (praxel.in) is an early-stage Indian speech-AI company focused on open-source Indic TTS / STT
- GitHub: github.com/praxelhq/praxy-tts (private for now; public on ArXiv upload)
- Founder: Pushpak Teja (ex-Groww, Masters' Union); reachable at pushpak@praxel.in

Could we schedule a 20-minute call this week to see if Karya would be a fit for this partnership? Happy to share the study protocol draft + the audio samples we'd use.

Thanks,
Pushpak

---

## Internal notes (not for email)

- If Karya replies with a higher quote than $500, ask for the specific cost breakdown before negotiating — they may charge differently for the platform + rater pool.
- If Karya says our 1-week lead time is too tight, offer to push launch to 2026-05-05 (still gives us a week for analysis pre-ArXiv).
- If Karya rejects (unlikely — we're well within their ethical scope), pivot to Prolific India-filter same day. Script is already in the codeswitch/ MOS sub-spec.
- Attention-check protocol: 5 calibration clips mixed into the 50 — 2 obviously bad (robot voice at wrong language), 2 obviously good (native-speaker source audio), 1 ambiguous. Filter raters whose calibration accuracy <80%.
