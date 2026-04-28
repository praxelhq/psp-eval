# Building the PSP paper PDF

## Prerequisites
TeXLive (or MacTeX) with `IEEEtran` class. On macOS:
```bash
brew install --cask mactex-no-gui
```

## Build
Run **four commands in order**. Citations need two latex passes after bibtex:
```bash
cd paper/psp/
xelatex psp
bibtex psp
xelatex psp
xelatex psp
```

You'll get `psp.pdf` in the same directory.

## Building on Overleaf / online services
Upload both `psp.tex` and `refs.bib` to the same project. Most online services
auto-detect the bibliography and run the full `pdflatex → bibtex → pdflatex ×2`
sequence. If citations show as `[?]`, force a full rebuild.

## Known warnings (non-fatal)
- Several `Underfull \hbox` warnings on long URLs — cosmetic, ignore.
- `Overfull \hbox` on the long aligner repo name `Harveenchadha/vakyansh-...`
  — cosmetic, ignore.

## Dependencies (all standard in TeXLive)
- IEEEtran document class
- amsmath / amssymb / booktabs / graphicx / hyperref / xcolor / newunicodechar

## Non-ASCII characters
The paper uses Unicode IAST diacritics (ṭ ḍ ṇ ṣ ḷ ḻ), Tamil letter ழ, and
IPA /ɻ/. All are declared via `\newunicodechar` in the preamble and render
natively — no `tipa` or `fontspec` required.
