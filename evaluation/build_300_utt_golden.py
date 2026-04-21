"""Build 300-utterance golden test sets from the manifest, by category.

Categories (phonological density):
- ``retroflex_heavy``: utterances with ≥3 retroflex chars
- ``aspiration_heavy``: utterances with ≥2 aspirated chars (Hindi primarily)
- ``length_heavy``: utterances with ≥2 long-vowel marks
- ``conjunct_heavy``: utterances with ≥1 conjunct sign (halanta / virama)
- ``general``: remaining diversity — sampled to fill out 300

Reads the train.jsonl we downloaded from Modal at /tmp/train_manifest.jsonl.
Writes one JSON per language to evaluation/golden_test_sets/{lang}_golden_300.json
matching the existing test-set schema.

Run:
    uv run python -m evaluation.build_300_utt_golden --language te
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evaluation.psp import (
    ASPIRATED_CHARS_BY_SCRIPT,
    LONG_VOWEL_CHARS_BY_SCRIPT,
    RETROFLEX_CHARS_BY_SCRIPT,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LANG_TO_SCRIPT = {"te": "telugu", "hi": "devanagari", "ta": "tamil"}
LANG_NAME = {"te": "telugu", "hi": "hindi", "ta": "tamil"}

# Indic virama / halanta characters — presence indicates consonant cluster
VIRAMA_BY_SCRIPT = {"telugu": "్", "devanagari": "्", "tamil": "்"}


def _count_chars(text: str, table: dict[str, str]) -> int:
    return sum(1 for ch in text if ch in table)


def categorize(text: str, script: str) -> dict[str, int]:
    retroflex_t = RETROFLEX_CHARS_BY_SCRIPT.get(script, {})
    aspirated_t = ASPIRATED_CHARS_BY_SCRIPT.get(script, {})
    long_t = LONG_VOWEL_CHARS_BY_SCRIPT.get(script, {})
    virama = VIRAMA_BY_SCRIPT.get(script, "")

    return {
        "retroflex": _count_chars(text, retroflex_t),
        "aspiration": _count_chars(text, aspirated_t),
        "length_long": _count_chars(text, long_t),
        "conjuncts": text.count(virama) if virama else 0,
        "chars": len(text),
    }


def build_golden(language: str, n_utt: int = 300) -> dict:
    manifest_path = Path("/tmp/train_manifest.jsonl")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing — run "
            "`modal volume get praxy-data /manifests/train.jsonl /tmp/train_manifest.jsonl` first."
        )

    script = LANG_TO_SCRIPT[language]
    retroflex_t = RETROFLEX_CHARS_BY_SCRIPT.get(script, {})
    aspirated_t = ASPIRATED_CHARS_BY_SCRIPT.get(script, {})
    long_t = LONG_VOWEL_CHARS_BY_SCRIPT.get(script, {})

    # Prefer clean studio sources over news readers for text quality.
    preferred_sources = {
        "te": {"indictts_telugu", "fleurs"},
        "hi": {"rasa", "fleurs"},
        "ta": {"indictts_tamil", "fleurs"},
    }[language]

    pool: list[dict] = []
    seen_texts = set()
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("language_code") != language:
                continue
            if r.get("source") not in preferred_sources:
                continue
            text = (r.get("transcript") or "").strip()
            if len(text) < 20 or len(text) > 300:
                continue
            # Deduplicate by text (IndicTTS has some near-duplicate reads)
            tkey = text[:80]
            if tkey in seen_texts:
                continue
            seen_texts.add(tkey)
            stats = categorize(text, script)
            pool.append({"text": text, "stats": stats})

    print(f"[build_golden] {language}: pool size = {len(pool)}")

    # Category buckets
    retroflex_heavy = sorted(
        [p for p in pool if p["stats"]["retroflex"] >= 3],
        key=lambda p: -p["stats"]["retroflex"],
    )
    aspiration_heavy = sorted(
        [p for p in pool if p["stats"]["aspiration"] >= 2],
        key=lambda p: -p["stats"]["aspiration"],
    )
    length_heavy = sorted(
        [p for p in pool if p["stats"]["length_long"] >= 2],
        key=lambda p: -p["stats"]["length_long"],
    )
    conjunct_heavy = sorted(
        [p for p in pool if p["stats"]["conjuncts"] >= 1],
        key=lambda p: -p["stats"]["conjuncts"],
    )

    # Quotas (language-specific: Tamil has no phonemic aspirated stops, so its
    # aspiration quota is redistributed to retroflex + length. This asymmetry
    # IS the linguistic reality and matches the paper's claim.)
    if language == "ta":
        quotas = {
            "retroflex_heavy": 150,
            "aspiration_heavy": 0,
            "length_heavy": 75,
            "conjunct_heavy": 25,
            "general": 50,
        }
    else:
        quotas = {
            "retroflex_heavy": 100,
            "aspiration_heavy": 75,
            "length_heavy": 50,
            "conjunct_heavy": 25,
            "general": 50,
        }

    def _take(source: list[dict], n: int, used: set) -> list[dict]:
        out = []
        for p in source:
            key = p["text"][:80]
            if key in used:
                continue
            out.append(p)
            used.add(key)
            if len(out) >= n:
                break
        return out

    used = set()
    selected: dict[str, list[dict]] = {}
    selected["retroflex_heavy"] = _take(retroflex_heavy, quotas["retroflex_heavy"], used)
    selected["aspiration_heavy"] = _take(aspiration_heavy, quotas["aspiration_heavy"], used)
    selected["length_heavy"] = _take(length_heavy, quotas["length_heavy"], used)
    selected["conjunct_heavy"] = _take(conjunct_heavy, quotas["conjunct_heavy"], used)

    # Fill the general bucket with random diverse sentences (not already used)
    rng = random.Random(1337)
    remaining = [p for p in pool if p["text"][:80] not in used]
    rng.shuffle(remaining)
    selected["general"] = _take(remaining, quotas["general"], used)

    # Assemble utterances list in the test-set schema
    utterances = []
    utt_ctr = 1
    for category, items in selected.items():
        for item in items:
            utterances.append({
                "id": f"{language}_golden_{utt_ctr:03d}",
                "category": category,
                "text": item["text"],
                "stats": item["stats"],
            })
            utt_ctr += 1
            if utt_ctr - 1 >= n_utt:
                break
        if utt_ctr - 1 >= n_utt:
            break

    # Summary
    by_cat = {c: sum(1 for u in utterances if u["category"] == c) for c in quotas}
    print(f"[build_golden] {language}: produced {len(utterances)} utts → {by_cat}")

    return {
        "test_set": f"{language}_golden_300",
        "language": LANG_NAME[language],
        "language_code": language,
        "source_description": "Sampled from IndicTTS + Rasa + FLEURS manifest, categorized by phonological density for PSP benchmarking.",
        "utterances": utterances,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=["te", "hi", "ta"])
    ap.add_argument("--n-utt", type=int, default=300)
    ap.add_argument("--out-dir", default="evaluation/golden_test_sets")
    args = ap.parse_args()

    golden = build_golden(args.language, args.n_utt)
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{golden['test_set']}.json"
    out_path.write_text(json.dumps(golden, indent=2, ensure_ascii=False))
    print(f"[build_golden] wrote {out_path}")


if __name__ == "__main__":
    main()
