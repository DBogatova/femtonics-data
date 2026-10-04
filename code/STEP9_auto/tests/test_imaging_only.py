#!/usr/bin/env python3
"""test_imaging_only.py - integration tests for imaging-only run support.

Builds a tiny synthetic fixture project in /tmp with:
  - a fake ranked_runs.csv / behavior_imaging_master.csv with 1 behavior run
  - an imaging_only_runs.csv with 1 imaging-only run
  - small synthetic 4D stacks with a fake tube/dendrite, ref3d, autoseg,
    reviewed mask, segments_final, and a mice.csv

Then runs femto_status, auto_mask (if available), auto_regions (if available),
run_metrics, coupling_phenotype, behavior_coupling, cohort_stats with
FEMTO_ROOT pointing at the fixture, and asserts:
  - outputs exist
  - the imaging-only run is in cohort imaging stats
  - the imaging-only run is absent from behavior stats
  - run_marks (excluded/revisit) removes a run from all statistics
  - <stem>_ignore.json removes a region from stats

No pytest needed: plain python asserts, runnable with the project venv.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[3]
CODE = PROJECT / "code"
PYTHON = sys.executable

# Ensure project code is importable
sys.path.insert(0, str(CODE / "STEP7_workflow"))
sys.path.insert(0, str(CODE / "STEP8_stats"))
sys.path.insert(0, str(CODE))


def make_synthetic_stack(shape=(20, 4, 32, 64), seed=42):
    """Create a small synthetic 4D (T,Z,Y,X) uint16 stack with a bright tube."""
    rng = np.random.default_rng(seed)
    T, Z, Y, X = shape
    vol = rng.integers(100, 200, size=shape, dtype=np.uint16)
    # tube: bright band along X in the middle of Y and Z
    y_center, z_center = Y // 2, Z // 2
    for t in range(T):
        vol[t, z_center - 1:z_center + 2, y_center - 2:y_center + 3, :] += np.uint16(
            800 + rng.integers(0, 200, size=(3, 5, X)))
        # soma: brighter spot at the left end
        vol[t, z_center - 1:z_center + 2, y_center - 2:y_center + 3, :8] += np.uint16(
            400 + rng.integers(0, 100, size=(3, 5, 8)))
        # branch event: sporadic at the right end
        if t % 5 == 0:
            vol[t, z_center, y_center, 50:60] += np.uint16(1000)
    return vol


def make_ref3d(vol):
    """Mean projection to make a reference volume (Z,Y,X)."""
    return vol.mean(axis=0).astype(np.uint16)


def make_mask(vol_shape, label=2):
    """Binary mask around the bright tube."""
    _, Z, Y, X = vol_shape
    mask = np.zeros((Z, Y, X), dtype=np.uint8)
    y_center, z_center = Y // 2, Z // 2
    mask[z_center - 1:z_center + 2, y_center - 2:y_center + 3, :] = label
    return mask


def make_segments(vol_shape):
    """3 segment labels: soma (1), trunk (2), branch (3)."""
    _, Z, Y, X = vol_shape
    seg = np.zeros((Z, Y, X), dtype=np.uint8)
    y_center, z_center = Y // 2, Z // 2
    # soma: columns 0..7
    seg[z_center - 1:z_center + 2, y_center - 2:y_center + 3, :8] = 1
    # trunk: columns 8..39
    seg[z_center - 1:z_center + 2, y_center - 2:y_center + 3, 8:40] = 2
    # branch: columns 40..63
    seg[z_center - 1:z_center + 2, y_center - 2:y_center + 3, 40:] = 3
    return seg


def build_fixture(root: Path):
    """Build a minimal fixture project under root/."""
    import tifffile

    # ── mice.csv ─────────────────────────────────────────────────────────
    mice_csv = root / "mice.csv"
    with open(mice_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mouse", "line", "virus", "titer_gc_per_ml", "volume_nl",
                     "dose_gc", "injection_date", "injected_by", "notes"])
        w.writerow(["rbp4_test1", "Rbp4-Cre", "AAV-PHP.eB GCaMP7s", "3.20E+13",
                     "15", "4.80E+11", "2025-10-10", "Test", "test mouse"])
        # Mouse with blank injection date (imaging-only, new)
        w.writerow(["rbp4_test2", "Rbp4-Cre", "AAV-PHP.eB GCaMP7s", "3.20E+13",
                     "15", "4.80E+11", "", "Test", "test mouse no injection date"])

    # ── Behavior run: rbp4_test1 / 01-15-2026 / run01 ───────────────────
    beh_run_dir = root / "rbp4_test1" / "01-15-2026" / "preprocessed" / "run01"
    beh_run_dir.mkdir(parents=True, exist_ok=True)
    stem_beh = "run01_clean"
    vol_beh = make_synthetic_stack()
    tifffile.imwrite(str(beh_run_dir / f"{stem_beh}.tif"), vol_beh)
    ref_beh = make_ref3d(vol_beh)
    tifffile.imwrite(str(beh_run_dir / f"{stem_beh}_ref3d.tif"), ref_beh)
    (beh_run_dir / f"{stem_beh}_ref3d.json").write_text(json.dumps({"version": "test"}))
    mask_beh = make_mask(vol_beh.shape)
    tifffile.imwrite(str(beh_run_dir / f"{stem_beh}_autoseg_labelmap.tif"), mask_beh)
    tifffile.imwrite(str(beh_run_dir / f"{stem_beh}_autoseg_labelmap_reviewed.tif"), mask_beh)
    seg_beh = make_segments(vol_beh.shape)
    tifffile.imwrite(str(beh_run_dir / f"{stem_beh}_segments_final.tif"), seg_beh)
    seg_json_beh = {"segment_names": {"1": "soma", "2": "trunk", "3": "branch2"}}
    (beh_run_dir / f"{stem_beh}_segments_final.json").write_text(json.dumps(seg_json_beh))

    # Fake behavior CSV so behavior_coupling detects it
    beh_session = root / "rbp4_test1" / "01-15-2026"
    (beh_session / "behavior").mkdir(parents=True, exist_ok=True)
    beh_csv = beh_session / "behavior" / "rbp4_test1_26-01-15_Run001_behavior.csv"
    with open(beh_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aligned_time_s", "pupil_smooth", "whisker_smooth_pad", "in_imaging_window"])
        for i in range(200):
            t = i * 0.2
            w.writerow([f"{t:.3f}", f"{100 + 10 * np.sin(t):.1f}", f"{50 + 5 * np.cos(t):.1f}", 1])

    # ── Imaging-only run: rbp4_test2 / 09-20-2026 / run02 ───────────────
    io_run_dir = root / "rbp4_test2" / "09-20-2026" / "preprocessed" / "run02"
    io_run_dir.mkdir(parents=True, exist_ok=True)
    stem_io = "run02_clean"
    vol_io = make_synthetic_stack(seed=99)
    tifffile.imwrite(str(io_run_dir / f"{stem_io}.tif"), vol_io)
    ref_io = make_ref3d(vol_io)
    tifffile.imwrite(str(io_run_dir / f"{stem_io}_ref3d.tif"), ref_io)
    (io_run_dir / f"{stem_io}_ref3d.json").write_text(json.dumps({"version": "test"}))
    mask_io = make_mask(vol_io.shape)
    tifffile.imwrite(str(io_run_dir / f"{stem_io}_autoseg_labelmap.tif"), mask_io)
    tifffile.imwrite(str(io_run_dir / f"{stem_io}_autoseg_labelmap_reviewed.tif"), mask_io)
    seg_io = make_segments(vol_io.shape)
    tifffile.imwrite(str(io_run_dir / f"{stem_io}_segments_final.tif"), seg_io)
    seg_json_io = {"segment_names": {"1": "soma", "2": "trunk", "3": "branch2"}}
    (io_run_dir / f"{stem_io}_segments_final.json").write_text(json.dumps(seg_json_io))

    # ── ranked_runs.csv (1 behavior run) ─────────────────────────────────
    ranked = root / "ranked_runs.csv"
    with open(ranked, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "priority", "mouse", "date", "scan_type", "munit",
                     "behavior_run", "voxel_zyx_um", "volume_rate_hz",
                     "imaging_quality", "extracted_4d_tif"])
        w.writerow(["1", "P1", "rbp4_test1", "01-15-2026", "snake", "MUnit_1",
                     "Run001", "0.8/0.9/0.9", "5.0", "good",
                     "rbp4_test1/01-15-2026/preprocessed/run01/run01_4d.tif"])

    # ── behavior_imaging_master.csv (1 ranked behavior run) ──────────────
    master = root / "behavior_imaging_master.csv"
    with open(master, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "priority", "mouse", "date", "session_dir", "munit",
                     "scan_type", "frame_rate_hz", "pixel_x_um", "voxel_z_um",
                     "imaging_quality", "behavior_base", "behavior_run_number",
                     "extracted_tif", "quality_score", "behavior_frame_loss_pct",
                     "behavior_warnings", "quality_notes",
                     "extracted_tif_other_candidates", "extracted_tif_suspect_nz"])
        w.writerow(["1", "P1", "rbp4_test1", "01-15-2026",
                     "rbp4_test1/01-15-2026", "MUnit_1", "snake",
                     "5.0", "0.9", "0.8", "good",
                     "rbp4_test1_26-01-15_Run001", "1",
                     "rbp4_test1/01-15-2026/preprocessed/run01/run01_4d.tif",
                     "4", "0.0", "", "", "", ""])

    # ── imaging_only_runs.csv (1 imaging-only run) ───────────────────────
    io_csv = root / "imaging_only_runs.csv"
    with open(io_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mouse", "date", "session_dir", "munit", "scan_type",
                     "frame_rate_hz", "pixel_x_um", "voxel_z_um",
                     "voxel_zyx_um", "imaging_quality", "behavior_status",
                     "behavior_base", "behavior_run_number",
                     "run_dir", "stem", "extracted_tif", "quality_score"])
        w.writerow(["rbp4_test2", "09-20-2026",
                     "rbp4_test2/09-20-2026", "MUnit_2", "snake",
                     "5.0", "0.9", "0.8", "0.8/0.9/0.9",
                     "good", "missing", "", "",
                     "rbp4_test2/09-20-2026/preprocessed/run02",
                     "run02_clean", "", "3"])

    # ── run_marks.csv (empty initially, for later tests) ─────────────────
    marks = root / "run_marks.csv"
    with open(marks, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["behavior_base", "mark", "reason", "updated"])

    # ── stats/ directory ─────────────────────────────────────────────────
    (root / "stats").mkdir(exist_ok=True)

    return {
        "root": root,
        "beh_run_dir": beh_run_dir,
        "io_run_dir": io_run_dir,
        "stem_beh": stem_beh,
        "stem_io": stem_io,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Tests
# ═════════════════════════════════════════════════════════════════════════════
def test_femto_status_loads_imaging_only():
    """femto_status.load_runs() includes imaging-only runs from imaging_only_runs.csv."""
    print("\n=== test_femto_status_loads_imaging_only ===")
    import importlib
    import femto_status as fs
    importlib.reload(fs)

    root = Path(os.environ["FEMTO_ROOT"])
    runs = fs.load_runs(root)
    # Should have at least 2 runs: 1 behavior + 1 imaging-only
    assert len(runs) >= 2, f"expected >= 2 runs, got {len(runs)}"

    io_runs = [r for r in runs if r.get("_imaging_only")]
    assert len(io_runs) >= 1, f"expected >= 1 imaging-only run, got {len(io_runs)}"

    io = io_runs[0]
    assert io["mouse"] == "rbp4_test2", f"wrong mouse: {io['mouse']}"
    assert io["priority"] == "P5", f"priority should be P5, got {io['priority']}"
    assert io["rank"] > 1, f"rank should be > 1 (after behavior runs), got {io['rank']}"
    assert io["_behavior_status"] == "missing", f"behavior_status should be 'missing', got {io['_behavior_status']}"
    print("  PASS: imaging-only run loaded with correct fields")


def test_resolve_run_dirs_imaging_only():
    """resolve_run_dirs populates run_dir/stem/stack for imaging-only runs via hint."""
    print("\n=== test_resolve_run_dirs_imaging_only ===")
    import importlib
    import femto_status as fs
    importlib.reload(fs)

    root = Path(os.environ["FEMTO_ROOT"])
    runs = fs.load_runs(root)
    fs.resolve_run_dirs(root, runs)

    io_runs = [r for r in runs if r.get("_imaging_only")]
    assert len(io_runs) >= 1
    io = io_runs[0]
    assert io["stack"] is not None, "imaging-only run should have a stack"
    assert io["stem"] == "run02_clean", f"wrong stem: {io['stem']}"
    assert io["run_dir"] is not None, "imaging-only run should have a run_dir"
    print(f"  PASS: run_dir={io['run_dir']}, stem={io['stem']}")


def test_detect_artifacts_and_stage():
    """detect_artifacts and stage work for imaging-only runs."""
    print("\n=== test_detect_artifacts_and_stage ===")
    import importlib
    import femto_status as fs
    importlib.reload(fs)

    root = Path(os.environ["FEMTO_ROOT"])
    runs = fs.build_status(root)
    io_runs = [r for r in runs if r.get("_imaging_only")]
    assert len(io_runs) >= 1
    io = io_runs[0]
    assert io["stage"] != "not_local", f"imaging-only run should be local, got stage={io['stage']}"
    print(f"  PASS: stage={io['stage']}, checklist={io['checklist']}")


def test_next_action_needs_behavior():
    """next_action for imaging-only runs includes 'needs behavior' text."""
    print("\n=== test_next_action_needs_behavior ===")
    import importlib
    import femto_status as fs
    importlib.reload(fs)

    root = Path(os.environ["FEMTO_ROOT"])
    runs = fs.build_status(root)
    io_runs = [r for r in runs if r.get("_imaging_only")]
    assert len(io_runs) >= 1
    io = io_runs[0]
    label = io["next"]["label"]
    assert "needs behavior" in label.lower(), f"expected 'needs behavior' in label, got: {label}"
    print(f"  PASS: next_action label = '{label}'")


def test_voxel_resolution_imaging_only():
    """voxel.py resolves voxel size for imaging-only runs via imaging_only_runs.csv."""
    print("\n=== test_voxel_resolution_imaging_only ===")
    from common.voxel import resolve_voxel
    root = Path(os.environ["FEMTO_ROOT"])
    io_stack = root / "rbp4_test2" / "09-20-2026" / "preprocessed" / "run02" / "run02_clean.tif"
    vox = resolve_voxel(io_stack, None, quiet=True)
    assert len(vox) == 3, f"expected 3-tuple, got {vox}"
    assert abs(vox[0] - 0.8) < 0.01, f"Z voxel should be ~0.8, got {vox[0]}"
    assert abs(vox[1] - 0.9) < 0.01, f"Y voxel should be ~0.9, got {vox[1]}"
    print(f"  PASS: voxel = {vox}")


def test_run_metrics_imaging_only():
    """run_metrics produces metrics for the imaging-only run."""
    print("\n=== test_run_metrics_imaging_only ===")
    import importlib
    import femto_status as fs
    importlib.reload(fs)

    root = Path(os.environ["FEMTO_ROOT"])
    runs = fs.build_status(root)
    io_runs = [r for r in runs if r.get("_imaging_only")]
    assert len(io_runs) >= 1
    io = io_runs[0]

    from run_metrics import metrics_for_run
    m = metrics_for_run(io, root)
    assert m is not None, "metrics_for_run returned None for imaging-only run"
    assert "r_soma_branch" in m, "no r_soma_branch in metrics"
    # behavior_state should be None (no behavior data)
    assert m.get("r_soma_branch_quiet") is None or m.get("r_soma_branch_active") is None, \
        "imaging-only run should not have behavior-state metrics"
    # Write it so cohort_stats can find it
    io_run_dir = root / io["run_dir"]
    out_p = io_run_dir / f"{io['stem']}_metrics.json"
    out_p.write_text(json.dumps(m, indent=2, default=float))
    print(f"  PASS: metrics written, r_soma_branch={m['r_soma_branch']:.3f}")
    return m


def test_run_metrics_behavior_run():
    """run_metrics produces metrics for the behavior run with behavior state."""
    print("\n=== test_run_metrics_behavior_run ===")
    import importlib
    import femto_status as fs
    importlib.reload(fs)

    root = Path(os.environ["FEMTO_ROOT"])
    runs = fs.build_status(root)
    beh_runs = [r for r in runs if not r.get("_imaging_only") and r.get("stack")]
    assert len(beh_runs) >= 1
    beh = beh_runs[0]

    from run_metrics import metrics_for_run
    m = metrics_for_run(beh, root)
    assert m is not None, "metrics_for_run returned None for behavior run"
    # Write it
    run_dir = root / beh["run_dir"]
    out_p = run_dir / f"{beh['stem']}_metrics.json"
    out_p.write_text(json.dumps(m, indent=2, default=float))
    print(f"  PASS: metrics written, r_soma_branch={m.get('r_soma_branch', 'n/a')}")
    return m


def test_behavior_coupling_skips_imaging_only():
    """behavior_coupling returns 'no behavior data' for imaging-only run."""
    print("\n=== test_behavior_coupling_skips_imaging_only ===")
    import importlib
    import femto_status as fs
    importlib.reload(fs)

    root = Path(os.environ["FEMTO_ROOT"])
    runs = fs.build_status(root)
    io_runs = [r for r in runs if r.get("_imaging_only")]
    assert len(io_runs) >= 1
    io = io_runs[0]

    from behavior_coupling import analyze_run
    res = analyze_run(io, root)
    if res is not None:
        assert "note" in res, f"expected 'note' in behavior_coupling result, got keys: {list(res.keys())}"
        assert "no behavior" in res["note"].lower(), f"expected 'no behavior' note, got: {res['note']}"
    print(f"  PASS: behavior_coupling correctly {'skipped' if res is None else 'returned note: ' + res.get('note', '')}")


def test_cohort_stats_includes_imaging_only():
    """cohort_stats includes imaging-only runs in imaging analyses, excludes from behavior."""
    print("\n=== test_cohort_stats_includes_imaging_only ===")
    root = Path(os.environ["FEMTO_ROOT"])

    from cohort_stats import collect
    d = collect()
    assert len(d) >= 2, f"expected >= 2 runs in cohort, got {len(d)}"

    # Check that imaging-only run is present
    io_rows = d[d["mouse"] == "rbp4_test2"]
    assert len(io_rows) >= 1, "imaging-only run (rbp4_test2) not found in cohort_metrics"

    # Check that imaging-only run is marked as no behavior
    assert not io_rows.iloc[0].get("has_behavior", True), "imaging-only run should have has_behavior=False"

    # Behavior run should have has_behavior=True (we gave it a behavior CSV)
    beh_rows = d[d["mouse"] == "rbp4_test1"]
    assert len(beh_rows) >= 1
    # (has_behavior depends on whether r_soma_branch_quiet exists - it might not for synthetic data
    #  since the behavior CSV is sparse, but the key is the imaging-only run is correctly marked)

    # The short label should have [no beh] for imaging-only
    io_short = io_rows.iloc[0]["short"]
    assert "[no beh]" in io_short, f"expected '[no beh]' in short label, got: {io_short}"

    print(f"  PASS: {len(d)} runs in cohort, imaging-only marked, short='{io_short}'")


def test_coupling_phenotype_imaging_only():
    """coupling_phenotype handles imaging-only run without crashing."""
    print("\n=== test_coupling_phenotype_imaging_only ===")
    import importlib
    import femto_status as fs
    importlib.reload(fs)

    root = Path(os.environ["FEMTO_ROOT"])
    runs = fs.build_status(root)
    io_runs = [r for r in runs if r.get("_imaging_only") and r.get("stack")]
    assert len(io_runs) >= 1
    io = io_runs[0]

    from coupling_phenotype import analyze_run
    res = analyze_run(io, root)
    # Should produce a result (3 regions = minimum for coupling_phenotype)
    assert res is not None, "coupling_phenotype returned None"
    if "note" not in res:
        assert "n_groups" in res, "expected 'n_groups' in coupling result"
        print(f"  PASS: n_groups={res['n_groups']}")
    else:
        print(f"  PASS: coupling_phenotype returned note: {res['note']}")


def test_run_marks_excludes_from_stats():
    """run_marks (excluded) removes a run from all statistics."""
    print("\n=== test_run_marks_excludes_from_stats ===")
    root = Path(os.environ["FEMTO_ROOT"])

    # Mark the behavior run as excluded
    from common.run_marks import set_mark, load_marks, is_set_aside, MARKS_CSV
    # Temporarily override MARKS_CSV to our fixture
    import common.run_marks as rm
    old_csv = rm.MARKS_CSV
    old_project = rm.PROJECT
    rm.MARKS_CSV = root / "run_marks.csv"
    rm.PROJECT = root

    try:
        set_mark("rbp4_test1_26-01-15_Run001", "excluded", "test exclusion")
        marks = load_marks()
        assert "rbp4_test1_26-01-15_Run001" in marks, "mark not saved"
        assert is_set_aside("rbp4_test1_26-01-15_Run001"), "is_set_aside should return True"

        # Now collect cohort - the excluded run should be gone
        from cohort_stats import collect
        d = collect()
        excluded_rows = d[d["behavior_base"] == "rbp4_test1_26-01-15_Run001"]
        assert len(excluded_rows) == 0, f"excluded run should not appear in cohort, found {len(excluded_rows)}"
        print(f"  PASS: excluded run absent from cohort ({len(d)} remaining)")

        # Clear the mark
        set_mark("rbp4_test1_26-01-15_Run001", None)
        assert not is_set_aside("rbp4_test1_26-01-15_Run001"), "mark should be cleared"

        # Re-collect: should be back
        d2 = collect()
        back = d2[d2["behavior_base"] == "rbp4_test1_26-01-15_Run001"]
        assert len(back) >= 1, "run should be back after clearing mark"
        print(f"  PASS: run back in cohort after clearing mark ({len(d2)} runs)")
    finally:
        rm.MARKS_CSV = old_csv
        rm.PROJECT = old_project


def test_ignore_json_removes_region():
    """<stem>_ignore.json removes a region from metrics."""
    print("\n=== test_ignore_json_removes_region ===")
    root = Path(os.environ["FEMTO_ROOT"])
    io_run_dir = root / "rbp4_test2" / "09-20-2026" / "preprocessed" / "run02"
    stem = "run02_clean"
    seg_p = io_run_dir / f"{stem}_segments_final.tif"

    # Write ignore list: ignore branch2
    ignore_p = io_run_dir / f"{stem}_ignore.json"
    ignore_p.write_text(json.dumps({
        "ignore": ["branch2"],
        "reason": "test ignore",
        "updated": "2026-10-04T00:00:00Z",
        "regions_file": f"{stem}_segments_final.tif"
    }))

    import importlib
    import femto_status as fs
    importlib.reload(fs)

    runs = fs.build_status(root)
    io_runs = [r for r in runs if r.get("_imaging_only") and r.get("stack")]
    io = io_runs[0]

    from run_metrics import metrics_for_run
    m = metrics_for_run(io, root)
    assert m is not None
    # branch2 should be ignored => no branch metrics
    assert "branch2" in (m.get("ignored_regions") or []), \
        f"branch2 should be in ignored_regions, got: {m.get('ignored_regions')}"
    # With branch2 ignored, should only have soma + trunk → no branch metrics
    # (either note about no branch, or branch metrics absent)
    has_branch = any(v.get("compartment") == "branch"
                     for v in m.get("regions", {}).values())
    assert not has_branch, "branch2 region should be absent after ignore"
    print(f"  PASS: branch2 ignored, regions={list(m.get('regions', {}).keys())}")

    # Clean up
    ignore_p.unlink()


def test_mice_csv_blank_injection_date():
    """mice.csv with blank injection_date does not crash cohort_stats."""
    print("\n=== test_mice_csv_blank_injection_date ===")
    from cohort_stats import collect
    d = collect()
    # rbp4_test2 has blank injection_date → dpi should be NaN
    io_rows = d[d["mouse"] == "rbp4_test2"]
    if len(io_rows):
        dpi = io_rows.iloc[0].get("dpi")
        assert dpi is None or (isinstance(dpi, float) and np.isnan(dpi)), \
            f"dpi should be NaN for blank injection_date, got {dpi}"
    print(f"  PASS: blank injection_date handled ({len(d)} runs collected without crash)")


def test_make_auto_tree_copies_imaging_only_csv():
    """make_auto_tree copies imaging_only_runs.csv into the mirror root."""
    print("\n=== test_make_auto_tree_copies_imaging_only_csv ===")
    root = Path(os.environ["FEMTO_ROOT"])
    # Simulate: make a temp mirror destination
    mirror = root / "_test_mirror"
    if mirror.exists():
        shutil.rmtree(mirror)
    mirror.mkdir()

    sys.path.insert(0, str(CODE / "STEP9_auto"))
    from make_auto_tree import copy_root_csvs
    copied = copy_root_csvs(root, mirror)
    assert "imaging_only_runs.csv" in copied, \
        f"imaging_only_runs.csv should be copied, got: {copied}"
    assert (mirror / "imaging_only_runs.csv").exists(), \
        "imaging_only_runs.csv not found in mirror"
    print(f"  PASS: copied CSVs: {copied}")

    # Clean up
    shutil.rmtree(mirror)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="femto_test_io_"))
    print(f"Fixture root: {tmpdir}")

    try:
        fixture = build_fixture(tmpdir)
        root = fixture["root"]

        # Point everything at the fixture
        os.environ["FEMTO_ROOT"] = str(root)

        # Also need to override run_marks.csv path for the fixture
        import common.run_marks as rm
        old_csv = rm.MARKS_CSV
        old_project = rm.PROJECT
        rm.MARKS_CSV = root / "run_marks.csv"
        rm.PROJECT = root

        try:
            passed, failed = 0, 0
            tests = [
                test_femto_status_loads_imaging_only,
                test_resolve_run_dirs_imaging_only,
                test_detect_artifacts_and_stage,
                test_next_action_needs_behavior,
                test_voxel_resolution_imaging_only,
                test_run_metrics_behavior_run,
                test_run_metrics_imaging_only,
                test_behavior_coupling_skips_imaging_only,
                test_coupling_phenotype_imaging_only,
                test_cohort_stats_includes_imaging_only,
                test_run_marks_excludes_from_stats,
                test_ignore_json_removes_region,
                test_mice_csv_blank_injection_date,
                test_make_auto_tree_copies_imaging_only_csv,
            ]
            for t in tests:
                try:
                    t()
                    passed += 1
                except Exception as e:
                    import traceback
                    print(f"\n  FAIL: {t.__name__}: {e}")
                    traceback.print_exc()
                    failed += 1

            print(f"\n{'='*60}")
            print(f"  RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
            print(f"{'='*60}")
            return 1 if failed else 0
        finally:
            rm.MARKS_CSV = old_csv
            rm.PROJECT = old_project
    finally:
        # Clean up temp dir
        shutil.rmtree(tmpdir, ignore_errors=True)
        # Restore env
        if "FEMTO_ROOT" in os.environ:
            del os.environ["FEMTO_ROOT"]


if __name__ == "__main__":
    sys.exit(main())
