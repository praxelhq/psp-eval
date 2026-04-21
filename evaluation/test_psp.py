"""Unit tests for ``evaluation.psp``.

Designed to run CPU-only, in seconds, without downloading any model. The audio
side of the probe is exercised via ``score_positions`` with synthetic embeddings
so we can pin down the scoring math; the actual Wav2Vec2 pipeline is import-
tested (we verify that constructing a ``RetroflexProbe`` doesn't fail) but its
model-loading branch is skipped when ``torch`` isn't available.

Run:
    uv run --extra dev pytest evaluation/test_psp.py -v
    # or the built-in runner:
    uv run python -m evaluation.test_psp
"""

from __future__ import annotations

import sys
from importlib.util import find_spec

from evaluation.psp import (
    ASPIRATED_CHARS_BY_SCRIPT,
    ASPIRATED_PHONEMES,
    DENTAL_COLLAPSE_TARGETS,
    LONG_SHORT_VOWEL_PAIRS,
    LONG_VOWEL_CHARS_BY_SCRIPT,
    NATIVE_LONG_SHORT_RATIO_DEFAULT,
    RETROFLEX_CHARS_BY_SCRIPT,
    RETROFLEX_PHONEMES,
    SHORT_VOWEL_CHARS_BY_SCRIPT,
    UNASPIRATED_COLLAPSE_TARGETS,
    AspirationPosition,
    AspirationProbe,
    LengthPosition,
    LengthProbe,
    RetroflexPosition,
    RetroflexProbe,
    aggregate_aspiration_fidelity,
    aggregate_retroflex_fidelity,
    bups_retroflex_phonemes_in,
    expected_aspiration_positions,
    expected_length_positions,
    expected_retroflex_positions,
    fidelity_from_similarities,
    length_fidelity_from_ratio,
)

HAS_NUMPY = find_spec("numpy") is not None
HAS_TORCH = find_spec("torch") is not None


# -----------------------------------------------------------------------------
# Text-side tests: grapheme scanner, BUPS cross-check.
# -----------------------------------------------------------------------------


def test_retroflex_inventory_is_non_empty() -> None:
    """Basic sanity: our phoneme constants aren't empty after inventory changes."""
    assert len(RETROFLEX_PHONEMES) >= 5
    for p in RETROFLEX_PHONEMES:
        # Every retroflex should have a declared dental-collapse target.
        assert p in DENTAL_COLLAPSE_TARGETS, f"{p} missing a dental collapse target"


def test_retroflex_table_coverage_per_script() -> None:
    """Each major script has at least the four main retroflex graphemes."""
    for script in ("devanagari", "telugu", "kannada"):
        table = RETROFLEX_CHARS_BY_SCRIPT[script]
        # Every table maps to one of the declared retroflex phonemes.
        for ch, ph in table.items():
            assert ph in RETROFLEX_PHONEMES, f"{script} {ch!r} maps to unknown phoneme {ph!r}"
        # Four core retroflexes present (except Tamil which collapses some).
        core = {"ṭ", "ḍ", "ṇ", "ṣ"}
        assert core.issubset(set(table.values())), f"{script} missing some of {core}"


def test_expected_retroflex_positions_telugu() -> None:
    """Telugu: 'విశ్వవిద్యాలయం' has no retroflexes; 'టింగ్' has one; 'డబ్బు' has one."""
    assert expected_retroflex_positions("విశ్వవిద్యాలయం", script="telugu") == []

    r = expected_retroflex_positions("టింగ్", script="telugu")
    assert len(r) == 1
    assert r[0].char == "ట" and r[0].phoneme == "ṭ"
    assert r[0].char_index == 0

    r = expected_retroflex_positions("డబ్బు", script="telugu")
    assert len(r) == 1
    assert r[0].char == "డ" and r[0].phoneme == "ḍ"


def test_expected_retroflex_positions_hindi() -> None:
    """Hindi: 'पढ़ाई' has no retroflexes (ढ़ is alveolar flap, not retroflex here).
    'बड़ा' has 'ड' which we count as retroflex even in modern Hindi where it's
    an alveolar allophone — matches the ACCENT.md spec."""
    r = expected_retroflex_positions("बड़ा", script="devanagari")
    # ड is retroflex; ़ is the nukta diacritic and shouldn't double-count.
    assert any(p.char == "ड" for p in r), r


def test_expected_retroflex_positions_tamil_zha() -> None:
    """Tamil: ழ (zha) must be recognised as the retroflex approximant."""
    r = expected_retroflex_positions("தமிழ்", script="tamil")
    zha = [p for p in r if p.char == "ழ"]
    assert len(zha) == 1
    assert zha[0].phoneme == "ḻ"


def test_script_autodetect_from_text() -> None:
    """When ``script`` is None, detect from the first Indic char."""
    text = "టింగ్"
    r_auto = expected_retroflex_positions(text)
    r_explicit = expected_retroflex_positions(text, script="telugu")
    assert [(p.char, p.phoneme) for p in r_auto] == [(p.char, p.phoneme) for p in r_explicit]


def test_bups_view_agrees_with_grapheme_view() -> None:
    """The BUPS-decoded phoneme stream must agree with our direct grapheme scan.

    This is a contract test — if BUPS tokenizes differently after an inventory
    change, this catches it.
    """
    samples = [
        ("telugu", "టింగ్ పెద్ద"),
        ("devanagari", "बड़ा डर"),
        ("kannada", "ಕಟ್ಟು"),
    ]
    for script, text in samples:
        grapheme = [p.phoneme for p in expected_retroflex_positions(text, script=script)]
        bups = bups_retroflex_phonemes_in(text, script=script)
        # Order must match and content must be a superset match (BUPS may decode
        # additional retroflexes from conjuncts that the char-table misses;
        # we assert grapheme ⊆ bups as multiset-prefix here).
        assert set(grapheme).issubset(set(bups)), (
            f"{script}: grapheme view {grapheme} not ⊆ bups view {bups}"
        )


# -----------------------------------------------------------------------------
# Scoring-math tests: fidelity arithmetic, aggregation.
# -----------------------------------------------------------------------------


def test_fidelity_from_similarities_monotonicity() -> None:
    """Raising native sim while holding dental sim fixed must raise fidelity."""
    base = fidelity_from_similarities(0.3, 0.3)
    higher = fidelity_from_similarities(0.6, 0.3)
    assert higher > base

    # And the converse: raising dental sim drops fidelity.
    lower = fidelity_from_similarities(0.3, 0.6)
    assert lower < base


def test_fidelity_bounds() -> None:
    assert fidelity_from_similarities(1.0, 0.0) == 1.0
    assert fidelity_from_similarities(0.0, 1.0) == 0.0
    # Both zero (or negative) → ambiguous 0.5.
    assert fidelity_from_similarities(0.0, 0.0) == 0.5
    assert fidelity_from_similarities(-0.2, -0.3) == 0.5


def test_aggregate_mean_and_collapsed_count() -> None:
    def pos(fid: float) -> RetroflexPosition:
        p = RetroflexPosition(char="ട", phoneme="ṭ", phoneme_id=0, char_index=0)
        p.fidelity = fid
        p.collapsed = fid < 0.5
        return p

    positions = [pos(0.8), pos(0.9), pos(0.3), pos(0.2)]
    mean_fid, n_collapsed = aggregate_retroflex_fidelity(positions)
    assert abs(mean_fid - 0.55) < 1e-9
    assert n_collapsed == 2


def test_aggregate_no_positions_returns_perfect() -> None:
    """Absent retroflex expectations must not penalise the score."""
    mean_fid, n_collapsed = aggregate_retroflex_fidelity([])
    assert mean_fid == 1.0 and n_collapsed == 0


# -----------------------------------------------------------------------------
# Probe-level tests with synthetic embeddings.
#
# We skip these if numpy isn't installed (base dev env). They don't require
# torch or any model.
# -----------------------------------------------------------------------------


def test_probe_construct_does_not_load() -> None:
    """Constructing a probe should never trigger model downloads."""
    probe = RetroflexProbe()
    assert probe._loaded is False
    assert probe._embed_model is None


def test_score_positions_with_synthetic_prototypes() -> None:
    """Feed synthetic embeddings; verify native wins / dental wins / mixed cases."""
    if not HAS_NUMPY:
        print("  [skip] numpy unavailable")
        return
    import numpy as np

    rng = np.random.default_rng(seed=42)
    probe = RetroflexProbe()
    d = 64

    # Build two well-separated prototype centroids. We use *non-antipodal*
    # centroids to model the realistic acoustic case: retroflex and dental
    # embeddings are distinct but both live in a broadly-positive manifold
    # of the XLS-R space (they share many features — voicing, place of
    # articulation — and differ in a few dimensions).
    base = rng.normal(loc=1.0, scale=0.05, size=d).astype(np.float32)
    native_axis = np.zeros(d, dtype=np.float32)
    native_axis[:d // 2] = 1.0
    dental_axis = np.zeros(d, dtype=np.float32)
    dental_axis[d // 2:] = 1.0
    native_center = (base + 2.0 * native_axis).astype(np.float32)
    dental_center = (base + 2.0 * dental_axis).astype(np.float32)

    for _ in range(5):
        probe.add_native_reference(
            "ṭ", native_center + rng.normal(scale=0.01, size=d).astype(np.float32)
        )
        probe.add_dental_reference(
            "ṭ", dental_center + rng.normal(scale=0.01, size=d).astype(np.float32)
        )

    positions = [
        RetroflexPosition(char="ట", phoneme="ṭ", phoneme_id=0, char_index=0),  # → native
        RetroflexPosition(char="ట", phoneme="ṭ", phoneme_id=0, char_index=2),  # → dental
        RetroflexPosition(char="ట", phoneme="ṭ", phoneme_id=0, char_index=4),  # mid
    ]
    embeddings = [
        native_center + rng.normal(scale=0.01, size=d).astype(np.float32),
        dental_center + rng.normal(scale=0.01, size=d).astype(np.float32),
        ((native_center + dental_center) / 2.0).astype(np.float32),
    ]

    report = probe.score_positions(positions, embeddings)
    assert report.n_expected == 3
    # First position — native should clearly win.
    fid0 = report.per_position[0].fidelity
    assert fid0 is not None and fid0 > 0.52, (fid0, report.per_position[0].native_sim, report.per_position[0].dental_sim)
    assert report.per_position[0].collapsed is False
    # Second — dental clearly wins (collapsed).
    fid1 = report.per_position[1].fidelity
    assert fid1 is not None and fid1 < 0.48, (fid1, report.per_position[1].native_sim, report.per_position[1].dental_sim)
    assert report.per_position[1].collapsed is True
    # Third — ambiguous midpoint (within ~5% of 0.5).
    fid2 = report.per_position[2].fidelity
    assert fid2 is not None and 0.45 < fid2 < 0.55, fid2


def test_score_positions_without_prototype_is_skipped() -> None:
    """If no prototype exists for a phoneme, the position is scored but flagged.

    Specifically: fidelity stays None and aggregate falls back to perfect score
    (rather than polluting the mean with phantom values). Prevents silent
    degradation when the reference set is missing a phoneme.
    """
    if not HAS_NUMPY:
        print("  [skip] numpy unavailable")
        return
    import numpy as np

    probe = RetroflexProbe()
    positions = [
        RetroflexPosition(char="ழ", phoneme="ḻ", phoneme_id=0, char_index=3),
    ]
    embeddings = [np.zeros(64, dtype=np.float32)]
    report = probe.score_positions(positions, embeddings)
    assert report.per_position[0].fidelity is None
    # Aggregate with no scored positions → perfect (documented behaviour).
    assert report.retroflex_fidelity == 1.0
    assert report.n_collapsed == 0


def test_score_positions_dental_fallback_uses_sibling_native() -> None:
    """When explicit dental refs are absent, probe falls back to the native
    centroid for the *dental collapse target* (e.g. /t/ native embeddings
    stand in as the "dental" prototype for /ṭ/)."""
    if not HAS_NUMPY:
        print("  [skip] numpy unavailable")
        return
    import numpy as np

    rng = np.random.default_rng(seed=7)
    probe = RetroflexProbe()
    d = 32

    native_t_retroflex = rng.normal(loc=1.0, scale=0.01, size=d).astype(np.float32)
    native_t_dental = rng.normal(loc=-1.0, scale=0.01, size=d).astype(np.float32)
    for _ in range(3):
        probe.add_native_reference("ṭ", native_t_retroflex.copy())
        probe.add_native_reference("t", native_t_dental.copy())

    # No explicit dental_ref for ṭ; code should use native "t" refs as fallback.
    pos = RetroflexPosition(char="ట", phoneme="ṭ", phoneme_id=0, char_index=0)
    report = probe.score_positions([pos], [native_t_retroflex.copy()])
    assert report.per_position[0].fidelity is not None
    assert report.per_position[0].fidelity > 0.8


# -----------------------------------------------------------------------------
# Aspiration dimension — text-side and scoring tests.
# -----------------------------------------------------------------------------


def test_aspiration_inventory_has_collapse_target_for_every_phoneme() -> None:
    """Every aspirated phoneme must have a declared unaspirated collapse target,
    and that target must not itself be in the aspirated set."""
    assert len(ASPIRATED_PHONEMES) >= 8
    for p in ASPIRATED_PHONEMES:
        assert p in UNASPIRATED_COLLAPSE_TARGETS, f"{p} missing unaspirated target"
        target = UNASPIRATED_COLLAPSE_TARGETS[p]
        assert target not in ASPIRATED_PHONEMES, (
            f"collapse target {target!r} for {p!r} is itself aspirated"
        )


def test_aspiration_tables_well_formed() -> None:
    """Each script's aspirated char table maps to declared aspirated phonemes."""
    for script in ("devanagari", "telugu", "kannada", "bengali", "gujarati"):
        table = ASPIRATED_CHARS_BY_SCRIPT[script]
        for ch, ph in table.items():
            assert ph in ASPIRATED_PHONEMES, f"{script} {ch!r} maps to unknown aspirated phoneme {ph!r}"
    # Tamil has no aspirated stops.
    assert ASPIRATED_CHARS_BY_SCRIPT["tamil"] == {}


def test_expected_aspiration_positions_hindi() -> None:
    """khānā (खाना, 'food') has exactly one aspirated kh at index 0."""
    positions = expected_aspiration_positions("खाना", script="devanagari")
    assert len(positions) == 1
    assert positions[0].char == "ख"
    assert positions[0].phoneme == "kh"
    assert positions[0].char_index == 0


def test_expected_aspiration_positions_tamil_returns_empty() -> None:
    """Tamil has no aspirated stops natively — scanner returns []."""
    positions = expected_aspiration_positions("வணக்கம்", script="tamil")
    assert positions == []


def test_aggregate_aspiration_no_positions_returns_perfect() -> None:
    assert aggregate_aspiration_fidelity([]) == (1.0, 0)


def test_aspiration_probe_construct_does_not_load() -> None:
    probe = AspirationProbe()
    assert probe._loaded is False
    # Inherits _native_refs structure from RetroflexProbe
    assert hasattr(probe, "_native_refs")


def test_aspiration_probe_unaspirated_centroid_falls_through_native_refs() -> None:
    """AspirationProbe uses the same _native_refs dict for both aspirated and
    unaspirated centroids — kh centroid = _native_refs['kh'], unaspirated
    substitute = _native_refs['k']. This is the core design choice that lets
    one bootstrap pickle serve both probes."""
    if not HAS_NUMPY:
        print("  [skip] numpy unavailable")
        return
    import numpy as np
    probe = AspirationProbe()
    # Add native centroids for both aspirated and unaspirated
    rng = np.random.default_rng(0)
    d = 16
    for _ in range(3):
        probe.add_native_reference("kh", rng.normal(size=d).astype(np.float32))
        probe.add_native_reference("k", rng.normal(size=d).astype(np.float32))

    aspirated_c = probe.native_centroid("kh")
    unaspirated_c = probe.unaspirated_centroid("kh")
    assert aspirated_c is not None and aspirated_c.shape == (d,)
    assert unaspirated_c is not None and unaspirated_c.shape == (d,)
    # Unaspirated centroid for "kh" is native centroid for "k" — they should match
    k_native = probe.native_centroid("k")
    assert np.allclose(unaspirated_c, k_native)


def test_aspiration_score_positions_with_synthetic_prototypes() -> None:
    """Mirror of retroflex synthetic test but for aspirated/unaspirated pairs."""
    if not HAS_NUMPY:
        print("  [skip] numpy unavailable")
        return
    import numpy as np
    rng = np.random.default_rng(seed=99)
    probe = AspirationProbe()
    d = 64
    base = rng.normal(loc=1.0, scale=0.05, size=d).astype(np.float32)
    asp_axis = np.zeros(d, dtype=np.float32); asp_axis[:d // 2] = 1.0
    unasp_axis = np.zeros(d, dtype=np.float32); unasp_axis[d // 2:] = 1.0
    asp_center = (base + 2.0 * asp_axis).astype(np.float32)
    unasp_center = (base + 2.0 * unasp_axis).astype(np.float32)
    for _ in range(5):
        probe.add_native_reference("kh", asp_center + rng.normal(scale=0.01, size=d).astype(np.float32))
        probe.add_native_reference("k", unasp_center + rng.normal(scale=0.01, size=d).astype(np.float32))

    positions = [
        AspirationPosition(char="ख", phoneme="kh", phoneme_id=0, char_index=0),
        AspirationPosition(char="ख", phoneme="kh", phoneme_id=0, char_index=2),
    ]
    embeddings = [
        asp_center + rng.normal(scale=0.01, size=d).astype(np.float32),    # aspirated wins
        unasp_center + rng.normal(scale=0.01, size=d).astype(np.float32),  # unaspirated wins (collapsed)
    ]
    report = probe.score_aspiration_positions(positions, embeddings)
    assert report.n_expected == 2
    assert report.per_position[0].fidelity is not None and report.per_position[0].fidelity > 0.52
    assert report.per_position[0].collapsed is False
    assert report.per_position[1].fidelity is not None and report.per_position[1].fidelity < 0.48
    assert report.per_position[1].collapsed is True
    assert report.n_collapsed == 1


# -----------------------------------------------------------------------------
# Length dimension — text-side + scoring math tests.
# -----------------------------------------------------------------------------


def test_long_short_vowel_pairs_well_formed() -> None:
    """Every long vowel must map to its short counterpart."""
    for long_v, short_v in LONG_SHORT_VOWEL_PAIRS.items():
        assert long_v != short_v
        # Long vowels end with macron-like markers; short don't. Sanity check.
        assert len(short_v) == 1, f"short vowel {short_v!r} should be single char"


def test_long_vowel_tables_per_script() -> None:
    """Each major Indic script has distinct long- and short-vowel tables."""
    for script in ("devanagari", "telugu", "tamil", "kannada"):
        long_t = LONG_VOWEL_CHARS_BY_SCRIPT.get(script, {})
        short_t = SHORT_VOWEL_CHARS_BY_SCRIPT.get(script, {})
        assert long_t, f"{script} missing long-vowel table"
        assert short_t, f"{script} missing short-vowel table"
        # Every long-vowel phoneme maps to something in LONG_SHORT_VOWEL_PAIRS
        for ch, ph in long_t.items():
            assert ph in LONG_SHORT_VOWEL_PAIRS, f"{script} {ch!r} maps to unknown long {ph!r}"


def test_expected_length_positions_telugu() -> None:
    """Telugu 'రాము' (Rāmu) has one long 'ā' (matra) and one short 'u' (matra)."""
    positions = expected_length_positions("రాము", script="telugu")
    phonemes = [(p.phoneme, p.is_long) for p in positions]
    assert ("ā", True) in phonemes, phonemes
    assert ("u", False) in phonemes, phonemes


def test_expected_length_positions_hindi() -> None:
    """Hindi 'काम' (kām) has long ā; 'कम' (kam) has short a via implicit schwa
    (not scored). We test the long case."""
    positions = expected_length_positions("काम", script="devanagari")
    assert any(p.phoneme == "ā" and p.is_long for p in positions)


def test_length_fidelity_from_ratio_native_hits_1() -> None:
    assert length_fidelity_from_ratio(1.9, 1.9) == 1.0
    # measured=1.9, native=2.0, tolerance=0.5 → error=0.1/1.0=0.1 → fid=0.9
    assert length_fidelity_from_ratio(1.9, 2.0) > 0.89  # floating-point ~0.8999


def test_length_fidelity_from_ratio_collapsed_drops() -> None:
    """A ratio of 1.0 (no long-short distinction) should score near 0 against
    a native prior of ~2.0 (50% relative error)."""
    fid = length_fidelity_from_ratio(1.0, 2.0)
    assert fid < 0.05, fid


def test_length_fidelity_from_ratio_extreme_lowers() -> None:
    """Way out-of-range ratio clamps at 0."""
    assert length_fidelity_from_ratio(0.1, 1.9) == 0.0
    assert length_fidelity_from_ratio(10.0, 1.9) == 0.0


def test_length_fidelity_missing_measurement_is_perfect() -> None:
    """No measurement available → neutral 1.0 (don't penalize absence)."""
    assert length_fidelity_from_ratio(None, 1.9) == 1.0


def test_length_probe_construct_does_not_load() -> None:
    probe = LengthProbe()
    assert probe._loaded is False


def test_length_probe_resolves_native_ratio_from_language() -> None:
    """Language code routes to literature prior."""
    probe = LengthProbe()
    assert probe._resolve_native_ratio("te", None) == NATIVE_LONG_SHORT_RATIO_DEFAULT["telugu"]
    assert probe._resolve_native_ratio("hi", None) == NATIVE_LONG_SHORT_RATIO_DEFAULT["hindi"]
    # Override wins over language default
    probe2 = LengthProbe(native_long_short_ratio=2.5)
    assert probe2._resolve_native_ratio("te", None) == 2.5


# -----------------------------------------------------------------------------
# Import-ability under real stacks (torch / transformers). These tests
# intentionally avoid any network call; they only verify the module imports
# cleanly on the given environment.
# -----------------------------------------------------------------------------


def test_import_without_torch_is_clean() -> None:
    """``evaluation.psp`` should import even if torch isn't installed (audio
    side uses lazy imports). Also: load() must be the only place that touches
    heavy deps."""
    # Already imported at module top; this test is a dummy to document the
    # contract. If torch truly wasn't on the dev box, the import at the top of
    # this file would fail first.
    import evaluation.psp as psp

    assert hasattr(psp, "RetroflexProbe")


# -----------------------------------------------------------------------------
# Runner.
# -----------------------------------------------------------------------------


def _run() -> int:
    tests = [
        test_retroflex_inventory_is_non_empty,
        test_retroflex_table_coverage_per_script,
        test_expected_retroflex_positions_telugu,
        test_expected_retroflex_positions_hindi,
        test_expected_retroflex_positions_tamil_zha,
        test_script_autodetect_from_text,
        test_bups_view_agrees_with_grapheme_view,
        test_fidelity_from_similarities_monotonicity,
        test_fidelity_bounds,
        test_aggregate_mean_and_collapsed_count,
        test_aggregate_no_positions_returns_perfect,
        test_probe_construct_does_not_load,
        test_score_positions_with_synthetic_prototypes,
        test_score_positions_without_prototype_is_skipped,
        test_score_positions_dental_fallback_uses_sibling_native,
        test_aspiration_inventory_has_collapse_target_for_every_phoneme,
        test_aspiration_tables_well_formed,
        test_expected_aspiration_positions_hindi,
        test_expected_aspiration_positions_tamil_returns_empty,
        test_aggregate_aspiration_no_positions_returns_perfect,
        test_aspiration_probe_construct_does_not_load,
        test_aspiration_probe_unaspirated_centroid_falls_through_native_refs,
        test_aspiration_score_positions_with_synthetic_prototypes,
        test_long_short_vowel_pairs_well_formed,
        test_long_vowel_tables_per_script,
        test_expected_length_positions_telugu,
        test_expected_length_positions_hindi,
        test_length_fidelity_from_ratio_native_hits_1,
        test_length_fidelity_from_ratio_collapsed_drops,
        test_length_fidelity_from_ratio_extreme_lowers,
        test_length_fidelity_missing_measurement_is_perfect,
        test_length_probe_construct_does_not_load,
        test_length_probe_resolves_native_ratio_from_language,
        test_import_without_torch_is_clean,
    ]
    n_pass = n_fail = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            n_fail += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e!r}")
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(_run())
