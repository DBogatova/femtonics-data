#!/usr/bin/env python3
"""qc_outliers.py — per-run visual QC features and automatic outlier flagging.

v2.0: Redesigned from visual inspection of contact sheets for all 52 runs.

Computes 4 features from the ref3d (anatomy_mean, activity_p99.5,
neighbour_corr, cofire_mean) and the auto mask. None uses correlation with the
soma/reference or with behavior (no circularity).

Features:
  1. activity_outside_ratio  — mean transient activity (p99.5 − anatomy) outside
     the mask / inside. High → mask missed the active cell.
  2. boundary_fraction_yz    — fraction of mask X columns where the mask centroid
     is within 2 voxels of the Z or Y tube boundary. High → cell out of frame.
  3. mask_fill_fraction      — fraction of the tube (Z*Y*X) filled by the mask.
     Used as supporting evidence (very high + high act_out → mask on noise).
  4. multi_y_band_fraction   — fraction of X columns with multiple disconnected
     Y bands in the mask at any Z slice. Used as supporting evidence.

Outlier rules (conservative; designed to have 0 false positives on 3 curated
ground-truth cells):
  - activity_outside_ratio > median + 4 MAD: mask missed the active structure.
  - boundary_fraction_yz > 0.75 (absolute): cell largely out of frame.
  - mask_fill_fraction > median + 4 MAD AND activity_outside_ratio > 1.5:
    mask captured noise/everything (supporting composite).
  - multi_y_band_fraction > median + 4 MAD AND activity_outside_ratio > 1.0:
    multiple cells in the mask (supporting composite).

A run is flagged only when at least one primary rule triggers.
Marks are always 'revisit' (never 'excluded'). Reason starts with '[auto-QC]'.

Outputs:
  - JSON per run: <stem>_qc.json
  - Cohort table: auto_pipeline/qc_review/qc_table.csv
  - qc_review.json update if called with --write-review
  - propose_marks() returns [(behavior_base, reason)]

Usage:
  python code/STEP9_auto/qc_outliers.py                     # all mirror runs
  python code/STEP9_auto/qc_outliers.py --run-dir <path>    # single run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

_SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if (_SCRIPT_ROOT / "code").is_dir():
    _ROOT = _SCRIPT_ROOT
elif "FEMTO_ROOT" in os.environ:
    _ROOT = Path(os.environ["FEMTO_ROOT"]).resolve()
else:
    _ROOT = _SCRIPT_ROOT
sys.path.insert(0, str(_ROOT / "code"))

__version__ = "2.0.0"


def _project_root():
    fr = os.environ.get("FEMTO_ROOT")
    return Path(fr).resolve() if fr else _ROOT


# ── Feature computation ───────────────────────────────────────────────────────

def load_ref3d(ref_path: Path) -> dict:
    """Load ref3d -> dict of channel name -> 3D array (Z, Y, X)."""
    raw = tifffile.imread(str(ref_path))
    json_path = ref_path.with_suffix(".json")
    if json_path.exists():
        meta = json.loads(json_path.read_text())
        ch_names = list(meta.get("channels", {}).keys())
    else:
        ch_names = ["anatomy_mean", "activity_p99.5", "neighbour_corr", "cofire_mean"]
    channels = {}
    for i, name in enumerate(ch_names):
        if i < raw.shape[1]:
            channels[name] = raw[:, i, :, :]
    return channels


def load_mask(mask_path: Path) -> np.ndarray:
    """Load auto mask -> bool array (Z, Y, X)."""
    raw = tifffile.imread(str(mask_path))
    return (raw == 2) if raw.max() == 2 else (raw > 0)


def compute_qc_features(run_dir: str | Path, stem: str = None) -> dict:
    """Compute QC features for a single run from ref3d + mask.

    Returns a dict with all feature values.
    """
    run_dir = Path(run_dir)

    # Find files
    if stem is None:
        masks = sorted(run_dir.glob("*_autoseg_labelmap_reviewed.tif"))
        if not masks:
            return {"error": f"no reviewed mask in {run_dir}"}
        stem = masks[0].name.replace("_autoseg_labelmap_reviewed.tif", "")

    ref_path = run_dir / f"{stem}_ref3d.tif"
    mask_path = run_dir / f"{stem}_autoseg_labelmap_reviewed.tif"

    if not ref_path.exists():
        return {"error": f"missing {ref_path.name}"}
    if not mask_path.exists():
        return {"error": f"missing {mask_path.name}"}

    # Load data
    channels = load_ref3d(ref_path)
    mask = load_mask(mask_path)
    Z, Y, X = mask.shape

    anatomy = channels.get("anatomy_mean", np.zeros_like(mask, dtype=float)).astype(np.float64)
    activity = channels.get("activity_p99.5", np.zeros_like(mask, dtype=float)).astype(np.float64)
    transient = np.clip(activity - anatomy, 0, None)

    features = {
        "run_dir": str(run_dir), "stem": stem, "version": __version__,
        "shape": [int(Z), int(Y), int(X)],
        "mask_voxels": int(mask.sum()),
    }

    # ── Feature 1: activity_outside_ratio ──────────────────────────────────
    inside = mask
    outside = ~mask
    if inside.sum() > 0 and outside.sum() > 0:
        mean_in = float(transient[inside].mean())
        mean_out = float(transient[outside].mean())
        features["activity_outside_ratio"] = float(mean_out / max(mean_in, 1e-6))
        features["mean_transient_inside"] = mean_in
        features["mean_transient_outside"] = mean_out
    else:
        features["activity_outside_ratio"] = 0.0
        features["mean_transient_inside"] = 0.0
        features["mean_transient_outside"] = 0.0

    # ── Feature 2: boundary_fraction_yz ────────────────────────────────────
    n_boundary_cols = 0
    n_mask_cols = 0
    for xi in range(X):
        col_mask = mask[:, :, xi]
        if not col_mask.any():
            continue
        n_mask_cols += 1
        zz, yy = np.where(col_mask)
        cz = zz.mean()
        cy = yy.mean()
        if cz < 2 or cz > Z - 3 or cy < 2 or cy > Y - 3:
            n_boundary_cols += 1
    features["boundary_fraction_yz"] = float(n_boundary_cols / max(n_mask_cols, 1))

    # ── Feature 3: mask_fill_fraction ──────────────────────────────────────
    tube_voxels = Z * Y * X
    features["mask_fill_fraction"] = float(mask.sum() / tube_voxels)

    # ── Feature 4: multi_y_band_fraction ───────────────────────────────────
    multi_cols = 0
    total_cols = 0
    for xi in range(X):
        col = mask[:, :, xi]
        if not col.any():
            continue
        total_cols += 1
        for zi in range(Z):
            row = col[zi, :]
            if row.any():
                lab, n_y = ndi.label(row)
                if n_y > 1:
                    multi_cols += 1
                    break
    features["multi_y_band_fraction"] = float(multi_cols / max(total_cols, 1))

    features["computed"] = datetime.now(timezone.utc).isoformat()
    return features


# ── Outlier detection ─────────────────────────────────────────────────────────

# Absolute thresholds (from visual review of 52 runs):
# - boundary_fraction_yz > 0.75: cell clearly clipped by FOV
#   (visual: every run above 0.75 shows the cell extending beyond the tube edge;
#    runs at 0.6-0.7 are at the edge but the cell is still traceable)
BOUNDARY_ABSOLUTE = 0.75


def detect_outliers(feature_table: list[dict], mad_factor: float = 4.0) -> list[dict]:
    """Flag runs as outliers.

    Uses median + mad_factor * MAD for activity_outside_ratio, and an absolute
    threshold for boundary_fraction_yz. mask_fill_fraction and
    multi_y_band_fraction are supporting evidence only (composite rules).

    Returns list of dicts with 'run_dir', 'stem', 'flagged', 'reasons', 'features'.
    """
    # Compute cohort statistics
    def _stats(key):
        vals = np.array([f[key] for f in feature_table
                         if key in f and isinstance(f[key], (int, float))])
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)) * 1.4826)
        return {"median": med, "MAD": mad,
                "threshold": med + mad_factor * max(mad, 1e-9)}

    stats_act = _stats("activity_outside_ratio")
    stats_fill = _stats("mask_fill_fraction")
    stats_my = _stats("multi_y_band_fraction")

    results = []
    for f in feature_table:
        reasons = []

        act_out = f.get("activity_outside_ratio", 0)
        bnd = f.get("boundary_fraction_yz", 0)
        fill = f.get("mask_fill_fraction", 0)
        multi_y = f.get("multi_y_band_fraction", 0)

        # Primary rule 1: activity_outside_ratio > med + 4 MAD
        if act_out > stats_act["threshold"]:
            reasons.append(
                f"[auto-QC] mask missed the active cell: activity outside auto mask "
                f"{act_out:.1f}x activity inside "
                f"(cohort median + {mad_factor:.0f} MAD = {stats_act['threshold']:.2f})"
            )

        # Primary rule 2: boundary_fraction_yz > 0.75
        if bnd > BOUNDARY_ABSOLUTE:
            reasons.append(
                f"[auto-QC] cell largely out of frame: mask at tube boundary for "
                f"{bnd*100:.0f}% of its X extent (threshold {BOUNDARY_ABSOLUTE*100:.0f}%)"
            )

        # Supporting composite: very high fill + moderate act_out
        if fill > stats_fill["threshold"] and act_out > 1.5:
            reasons.append(
                f"[auto-QC] mask may have captured noise: fills {fill*100:.0f}% of tube "
                f"(threshold {stats_fill['threshold']*100:.0f}%) and activity is "
                f"{act_out:.1f}x outside"
            )

        # Supporting composite: multi-Y bands + act_out
        # Require multi_y > 10% (not just above MAD threshold) to avoid marginal flags
        if multi_y > max(stats_my["threshold"], 0.10) and act_out > 1.0:
            reasons.append(
                f"[auto-QC] possible multiple cells in mask: disconnected Y bands "
                f"in {multi_y*100:.0f}% of columns and activity {act_out:.1f}x outside"
            )

        results.append({
            "run_dir": f.get("run_dir", ""),
            "stem": f.get("stem", ""),
            "flagged": len(reasons) > 0,
            "reasons": reasons,
            "features": {
                "activity_outside_ratio": act_out,
                "boundary_fraction_yz": bnd,
                "mask_fill_fraction": fill,
                "multi_y_band_fraction": multi_y,
            },
        })

    return results


def propose_marks(outlier_results: list[dict]) -> list[tuple]:
    """Return [(behavior_base, reason)] for flagged runs.

    Resolves behavior_base from imaging_only_runs.csv or
    behavior_imaging_master.csv. Never marks 'excluded'.
    """
    root = _project_root()
    # Build a run_dir -> behavior_base map
    rd_to_base = {}

    io_path = root / "imaging_only_runs.csv"
    if io_path.exists():
        with open(io_path) as fh:
            for row in csv.DictReader(fh):
                rd = row.get("run_dir", "")
                base = row.get("behavior_base", "")
                if rd and base:
                    rd_to_base[rd] = base

    bim_path = root / "behavior_imaging_master.csv"
    if bim_path.exists():
        with open(bim_path) as fh:
            for row in csv.DictReader(fh):
                rd = row.get("run_dir", "")
                base = row.get("behavior_base", "")
                if rd and base and rd not in rd_to_base:
                    rd_to_base[rd] = base

    marks = []
    for r in outlier_results:
        if not r["flagged"]:
            continue
        # Try to find behavior_base from the run_dir
        rd = r["run_dir"]
        # Normalize: strip auto_pipeline prefix if present
        for prefix in [str(root) + "/", "auto_pipeline/"]:
            if rd.startswith(prefix):
                rd = rd[len(prefix):]
        base = rd_to_base.get(rd, "")
        if base:
            reason = "; ".join(r["reasons"][:3])
            marks.append((base, reason))
    return marks


# ── Main ──────────────────────────────────────────────────────────────────────

def run_qc_all(output_dir: str = None):
    """Run QC on all mirror runs and write outputs."""
    root = _project_root()

    if output_dir is None:
        output_dir = root / "qc_review"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all runs with auto masks and ref3d
    run_dirs = []
    for mask_file in sorted(root.rglob("*_autoseg_labelmap_reviewed.tif")):
        rd = mask_file.parent
        if "old" in str(rd) or "_e2e" in str(rd) or "/_" in str(rd):
            continue
        stem = mask_file.name.replace("_autoseg_labelmap_reviewed.tif", "")
        ref = rd / f"{stem}_ref3d.tif"
        if ref.exists():
            run_dirs.append((rd, stem))

    print(f"[qc] Found {len(run_dirs)} runs to QC")

    # Compute features
    feature_table = []
    for rd, stem in run_dirs:
        print(f"  {rd.name}/{stem}...", end=" ", flush=True)
        t0 = time.time()
        features = compute_qc_features(rd, stem)
        elapsed = time.time() - t0

        if "error" not in features:
            feature_table.append(features)
            # Write per-run JSON
            qc_path = rd / f"{stem}_qc.json"
            qc_path.write_text(json.dumps(features, indent=2))
            print(f"{elapsed:.1f}s  act_out={features['activity_outside_ratio']:.3f} "
                  f"bnd={features['boundary_fraction_yz']:.3f} "
                  f"fill={features['mask_fill_fraction']:.3f}")
        else:
            print(f"SKIP: {features['error']}")

    # Detect outliers
    results = detect_outliers(feature_table)

    # Write cohort table
    csv_path = output_dir / "qc_table.csv"
    fieldnames = ["run_dir", "stem", "flagged",
                  "activity_outside_ratio", "boundary_fraction_yz",
                  "mask_fill_fraction", "multi_y_band_fraction", "reasons"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {"run_dir": r["run_dir"], "stem": r["stem"],
                   "flagged": r["flagged"]}
            row.update(r["features"])
            row["reasons"] = "; ".join(r["reasons"])
            w.writerow(row)

    # Summary
    n_flagged = sum(1 for r in results if r["flagged"])
    print(f"\n[qc] {n_flagged}/{len(results)} runs flagged")
    for r in results:
        if r["flagged"]:
            print(f"  FLAGGED: {Path(r['run_dir']).name}/{r['stem']}")
            for reason in r["reasons"]:
                print(f"    {reason}")

    marks = propose_marks(results)
    if marks:
        print(f"\n[qc] Proposed marks ({len(marks)}):")
        for base, reason in marks:
            print(f"  revisit {base}: {reason}")

    return results, feature_table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="single run directory to QC")
    ap.add_argument("--output-dir", help="output directory for cohort table")
    args = ap.parse_args(argv)

    if args.run_dir:
        features = compute_qc_features(args.run_dir)
        print(json.dumps(features, indent=2))
    else:
        run_qc_all(args.output_dir)


if __name__ == "__main__":
    main()
