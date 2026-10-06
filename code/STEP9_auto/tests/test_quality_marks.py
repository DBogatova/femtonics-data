#!/usr/bin/env python3
"""test_quality_marks.py - quality ratings (run_quality.csv) on the synthetic fixture.

Reuses the fixture of test_imaging_only.py (2 runs, 2 mice, in /tmp) and asserts:
  - ratings round-trip (set / get / aliases / clear / invalid rejected)
  - a rated run is NOT set aside, run_marks.csv and load_marks() are unchanged
  - femto_status attaches run['rating']
  - cohort_stats: 'rating' column, per-run rating list, SENSITIVITY block with (a) and (b),
    stats rows for main / without_questionable / very_good_only
  - paper_stats.quality_sensitivity: markdown table + JSON for both subsets
  - `femto rate` CLI (main_rate) writes only the patched file

Never touches the real run_quality.csv / run_marks.csv: both paths are pointed at the fixture.
Run with the analysis venv:  $PY code/STEP9_auto/tests/test_quality_marks.py
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "code"))
sys.path.insert(0, str(PROJECT / "code" / "STEP7_workflow"))
sys.path.insert(0, str(PROJECT / "code" / "STEP8_stats"))
sys.path.insert(0, str(PROJECT / "code" / "STEP9_auto"))

import test_imaging_only as tio          # noqa: E402  (fixture builder + metrics setup)

B1 = "rbp4_test1_26-01-15_Run001"        # behavior run, mouse rbp4_test1


def _bytes(p: Path):
    return p.read_bytes() if p.exists() else None


def test_round_trip(rm):
    print("\n=== test_round_trip ===")
    assert rm.get_quality(B1) is None
    for given, want in (("very_good", "very_good"), ("very good", "very_good"), ("VG", "very_good"),
                        ("good", "good"), ("questionable", "questionable")):
        assert rm.set_quality(B1, given, "why") == want
        assert rm.get_quality(B1) == want, (given, rm.get_quality(B1))
        assert rm.quality_of(B1) == (want, "why")
    rm.set_quality(B1, "clear")
    assert rm.get_quality(B1) is None and B1 not in rm.load_quality()
    try:
        rm.set_quality(B1, "great")
        raise AssertionError("invalid rating accepted")
    except ValueError:
        pass
    assert rm.rating_label("very_good") == "very good" and rm.rating_label(None) == ""
    print("  PASS")


def test_rating_does_not_set_aside(rm):
    print("\n=== test_rating_does_not_set_aside ===")
    before_csv, before = _bytes(rm.MARKS_CSV), rm.load_marks()
    for q in rm.RATINGS:
        rm.set_quality(B1, q, "t")
        assert not rm.is_set_aside(B1), f"rating {q} set the run aside"
        assert rm.mark_of(B1) == (None, "")
    assert rm.load_marks() == before and _bytes(rm.MARKS_CSV) == before_csv, "run_marks.csv changed"
    import importlib, femto_status as fs
    importlib.reload(fs)
    runs = fs.build_status(Path(os.environ["FEMTO_ROOT"]))
    assert all("rating" in r for r in runs)
    r1 = [r for r in runs if r.get("behavior_base") == B1]
    assert r1 and r1[0]["rating"] == rm.RATINGS[-1] and not r1[0].get("mark")
    from cohort_stats import collect
    d = collect()
    assert B1 in set(d.behavior_base), "rated run missing from the cohort"
    assert d.set_index("behavior_base").loc[B1, "rating"] == "questionable"
    rm.set_quality(B1, None)
    print("  PASS")


def test_cohort_sensitivity(rm):
    print("\n=== test_cohort_sensitivity ===")
    import cohort_stats as cs
    rm.set_quality(B1, "questionable", "fixture")
    d = cs.collect(); dd = cs.collect_distance()
    assert "rating" in d.columns
    with contextlib.redirect_stdout(io.StringIO()):
        txt, tab = cs.tests(d, dd)
    assert "Quality rating per run" in txt and B1 in txt.split("Quality rating per run")[1]
    if tab is not None:                  # the fixture is tiny; the family may be empty
        assert "SENSITIVITY TO YOUR QUALITY RATINGS" in txt
        assert "(a) without 'questionable'" in txt and "(b) 'very good' runs only" in txt
        assert {r["subset"] for r in cs._SENS} == {"main", "without_questionable", "very_good_only"}
        # main block is unchanged by ratings
        rm.set_quality(B1, None)
        with contextlib.redirect_stdout(io.StringIO()):
            _, tab0 = cs.tests(cs.collect(), dd)
        assert tab0[["test", "p_mouse", "q_bh"]].equals(tab[["test", "p_mouse", "q_bh"]]), "ratings changed the main result"
        print(f"  PASS ({len(tab)} primary tests; sensitivity rows {len(cs._SENS)})")
    else:
        print("  PASS (no primary family on the fixture; rating list checked)")
    rm.set_quality(B1, None)


def test_paper_stats_sensitivity(rm):
    print("\n=== test_paper_stats_sensitivity ===")
    import pandas as pd
    import cohort_stats as cs
    import paper_stats as ps
    rm.set_quality(B1, "very_good", "fixture")
    d = cs.collect()
    d["rating"] = d.behavior_base.map(lambda b: rm.get_quality(b) or "")
    lines, j = ps.quality_sensitivity(d, pd.DataFrame(), pd.DataFrame(), [], ["test02_coupling"], [0.01])
    md = "\n".join(lines)
    assert "## Sensitivity to Daria's quality ratings" in md
    assert "| Test | main p (q) | (a) no questionable p (q) | (b) very good only p (q) |" in md
    assert set(j["subsets"]) == {"without_questionable", "very_good_only"} and j["counts"]["very_good"] == 1
    assert j["subsets"]["without_questionable"]["n_cells"] == len(d)      # nothing questionable -> identical
    assert "not computable" in j["subsets"]["very_good_only"]["note"]       # 1 mouse
    rm.set_quality(B1, None)
    print("  PASS")


def test_cli(rm):
    print("\n=== test_cli ===")
    with contextlib.redirect_stdout(io.StringIO()) as out:
        rm.main_rate([B1, "good", "--reason", "cli"])
        rm.main_rate([B1])
        rm.main_rate([])
        rm.main_rate([B1, "clear"])
    s = out.getvalue()
    assert f"{B1}: rated good - cli" in s and "1 rated run(s)" in s and "rating cleared" in s, s
    assert rm.get_quality(B1) is None
    print("  PASS")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="femto_test_quality_"))
    real_q = PROJECT / "run_quality.csv"; real_q_before = _bytes(real_q)
    real_m = PROJECT / "run_marks.csv"; real_m_before = _bytes(real_m)
    import common.run_marks as rm
    old = (rm.MARKS_CSV, rm.QUALITY_CSV, rm.PROJECT)
    passed = failed = 0
    try:
        root = tio.build_fixture(tmp)["root"]
        os.environ["FEMTO_ROOT"] = str(root)
        rm.MARKS_CSV, rm.QUALITY_CSV, rm.PROJECT = root / "run_marks.csv", root / "run_quality.csv", root
        with contextlib.redirect_stdout(io.StringIO()):       # metrics for both fixture runs
            tio.test_run_metrics_behavior_run(); tio.test_run_metrics_imaging_only()
        for t in (test_round_trip, test_rating_does_not_set_aside, test_cohort_sensitivity,
                  test_paper_stats_sensitivity, test_cli):
            try:
                t(rm); passed += 1
            except Exception as e:
                import traceback
                print(f"\n  FAIL: {t.__name__}: {e}"); traceback.print_exc(); failed += 1
    finally:
        rm.MARKS_CSV, rm.QUALITY_CSV, rm.PROJECT = old
        os.environ.pop("FEMTO_ROOT", None)
        shutil.rmtree(tmp, ignore_errors=True)
    assert _bytes(real_q) == real_q_before, "the real run_quality.csv was modified!"
    assert _bytes(real_m) == real_m_before, "the real run_marks.csv was modified!"
    print(f"\n{'=' * 60}\n  RESULTS: {passed} passed, {failed} failed out of {passed + failed}\n{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
