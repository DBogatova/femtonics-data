#!/usr/bin/env python3
"""qc_review_sheets.py — render contact sheets and compute visual QC features.

For each mirror run, loads the ref3d (anatomy_mean, activity_p99.5,
neighbour_corr, cofire_mean) and the auto mask, then:
  1. Renders a compact contact sheet: anatomy/activity/cofire MIPs (XY and XZ),
     mask outline overlaid, with run label.
  2. Computes features that capture what makes a run unanalyzable:
     - activity_outside_ratio: ratio of activity (99.5th pct − median) outside
       the mask vs inside. High → other active cells in the FOV.
     - boundary_fraction_yz: fraction of the mask's XZ/XY path length where the
       traced cell sits at the Z or Y boundary of the tube. High → cell partly
       out of frame.
     - cofire_fragmentation: whether the cofire MIP shows spatially separated
       bright structures (multiple cells that fire together differently).
     - activity_gap_fraction: fraction of the X range where the cell's mask
       is present but activity is very low (cell disappears over a stretch).

Outputs:
  - AUTO_ROOT/qc_review/sheets/<stem>_sheet.png  (per run)
  - Returns feature dicts for aggregation.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

warnings.filterwarnings("ignore", category=UserWarning)

_SCRIPT_ROOT = Path(__file__).resolve().parents[2]
AUTO_ROOT = Path(os.environ.get("FEMTO_ROOT", str(_SCRIPT_ROOT / "auto_pipeline"))).resolve()


def load_ref3d(ref_path: Path) -> dict:
    """Load ref3d and return dict of channel name -> 3D array (Z, Y, X)."""
    raw = tifffile.imread(str(ref_path))
    # Shape: (Z, 4, Y, X)
    json_path = ref_path.with_suffix(".json")
    channels = {}
    if json_path.exists():
        meta = json.loads(json_path.read_text())
        ch_names = list(meta.get("channels", {}).keys())
    else:
        ch_names = ["anatomy_mean", "activity_p99.5", "neighbour_corr", "cofire_mean"]

    for i, name in enumerate(ch_names):
        if i < raw.shape[1]:
            channels[name] = raw[:, i, :, :]
    return channels


def load_mask(mask_path: Path) -> np.ndarray:
    """Load auto mask -> bool array (Z, Y, X)."""
    raw = tifffile.imread(str(mask_path))
    return (raw == 2) if raw.max() == 2 else (raw > 0)


def compute_features(channels: dict, mask: np.ndarray, stem: str,
                     run_dir: str) -> dict:
    """Compute the 4 visual QC features.

    Features designed to separate run01 (multiple cells, partly out of frame)
    from the 3 good ground-truth cells. None uses soma correlation or behavior.

    1. activity_outside_ratio:
       (mean activity outside mask) / (mean activity inside mask).
       Activity = the activity_p99.5 channel minus the anatomy_mean (transient
       component). A normal single-cell recording has very low activity outside
       the mask; multiple active cells → high ratio.

    2. boundary_fraction_yz:
       For each X column with mask voxels, check if the mask centroid in (Z,Y)
       is within 2 voxels of the tube's Z or Y boundary. High → cell at edge,
       partly out of frame.

    3. cofire_fragmentation:
       Number of distinct bright connected components in the cofire MIP after
       thresholding at the 90th pct of the cofire channel, minus 1 (the main
       cell). High → multiple separate co-firing structures.

    4. activity_gap_fraction:
       Fraction of X columns that (a) have mask voxels AND (b) the mean
       activity inside the mask in that column is below the 25th pct of
       across-column mask activity. A long dim stretch → the cell disappears
       for part of the tube.
    """
    from scipy import ndimage as ndi

    activity = channels.get("activity_p99.5", np.zeros_like(mask, dtype=float))
    anatomy = channels.get("anatomy_mean", np.zeros_like(mask, dtype=float))
    cofire = channels.get("cofire_mean", np.zeros_like(mask, dtype=float))

    activity = activity.astype(np.float64)
    anatomy = anatomy.astype(np.float64)
    cofire = cofire.astype(np.float64)

    # Transient activity = activity − anatomy (the part that fluctuates)
    transient = activity - anatomy
    transient = np.clip(transient, 0, None)

    Z, Y, X = mask.shape
    feats = {"stem": stem, "run_dir": run_dir, "shape": [Z, Y, X]}

    # 1. activity_outside_ratio
    inside = mask
    outside = ~mask
    if inside.sum() > 0 and outside.sum() > 0:
        mean_in = transient[inside].mean()
        mean_out = transient[outside].mean()
        feats["activity_outside_ratio"] = float(mean_out / max(mean_in, 1e-6))
    else:
        feats["activity_outside_ratio"] = 0.0

    # 2. boundary_fraction_yz
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
        # Check if centroid is within 2 voxels of boundary
        if cz < 2 or cz > Z - 3 or cy < 2 or cy > Y - 3:
            n_boundary_cols += 1
    feats["boundary_fraction_yz"] = float(n_boundary_cols / max(n_mask_cols, 1))

    # 3. cofire_fragmentation
    # Project cofire to XY MIP and XZ MIP, threshold, count components
    cofire_xy = cofire.max(axis=0)  # Y, X
    cofire_xz = cofire.max(axis=1)  # Z, X
    combined = np.zeros((Y + Z + 2, X), dtype=np.float64)
    combined[:Y, :] = cofire_xy
    combined[Y + 2:, :] = cofire_xz

    # Use a higher threshold to find truly bright regions
    if combined.max() > 0:
        thr = np.percentile(combined[combined > 0], 85) if (combined > 0).sum() > 10 else 0
        bright = combined > thr
        lab, n_comp = ndi.label(bright)
        # Filter by size — only count components > 50 voxels in the projection
        if n_comp > 0:
            sizes = np.bincount(lab.ravel())[1:]
            n_sig = int((sizes > 30).sum())
        else:
            n_sig = 0
    else:
        n_sig = 0
    feats["cofire_n_components"] = n_sig
    feats["cofire_fragmentation"] = max(0, n_sig - 1)

    # 4. activity_gap_fraction
    col_activity = np.zeros(X)
    for xi in range(X):
        col_mask = mask[:, :, xi]
        if col_mask.any():
            col_activity[xi] = transient[:, :, xi][col_mask].mean()
        else:
            col_activity[xi] = np.nan

    valid = ~np.isnan(col_activity)
    if valid.sum() >= 4:
        vals = col_activity[valid]
        q25 = np.percentile(vals, 25)
        gap_cols = np.sum(col_activity[valid] < q25 * 0.5)
        feats["activity_gap_fraction"] = float(gap_cols / valid.sum())
    else:
        feats["activity_gap_fraction"] = 0.0

    # Additional raw stats
    feats["mask_voxels"] = int(mask.sum())
    feats["mask_fill_fraction"] = float(mask.sum() / (Z * Y * X))
    feats["mean_transient_inside"] = float(transient[mask].mean()) if mask.any() else 0.0
    feats["mean_transient_outside"] = float(transient[~mask].mean()) if (~mask).any() else 0.0

    return feats


def render_sheet(channels: dict, mask: np.ndarray, stem: str,
                 out_path: Path, features: dict = None):
    """Render a compact contact sheet for visual inspection.

    Layout (2 rows × 3 cols):
      [anatomy XY MIP]  [activity XY MIP]  [cofire XY MIP]
      [anatomy XZ MIP]  [activity XZ MIP]  [cofire XZ MIP]

    Mask outline overlaid in red on all panels.
    Title: stem + key feature values.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    anatomy = channels.get("anatomy_mean", np.zeros_like(mask, dtype=float)).astype(float)
    activity = channels.get("activity_p99.5", np.zeros_like(mask, dtype=float)).astype(float)
    cofire = channels.get("cofire_mean", np.zeros_like(mask, dtype=float)).astype(float)
    transient = np.clip(activity - anatomy, 0, None)

    # MIPs
    anat_xy = anatomy.max(axis=0)    # Y, X
    anat_xz = anatomy.max(axis=1)    # Z, X
    act_xy = transient.max(axis=0)
    act_xz = transient.max(axis=1)
    cof_xy = cofire.max(axis=0)
    cof_xz = cofire.max(axis=1)

    # Mask outlines
    from scipy import ndimage as ndi
    mask_xy = mask.max(axis=0)
    mask_xz = mask.max(axis=1)
    outline_xy = mask_xy.astype(float) - ndi.binary_erosion(mask_xy, iterations=1).astype(float)
    outline_xz = mask_xz.astype(float) - ndi.binary_erosion(mask_xz, iterations=1).astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(14, 5), constrained_layout=True)

    panels = [
        (anat_xy, "anatomy MIP (XY)", outline_xy),
        (act_xy, "activity (transient) XY", outline_xy),
        (cof_xy, "cofire MIP (XY)", outline_xy),
        (anat_xz, "anatomy MIP (XZ)", outline_xz),
        (act_xz, "activity (transient) XZ", outline_xz),
        (cof_xz, "cofire MIP (XZ)", outline_xz),
    ]

    for idx, (data, title, outline) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        # Normalize to 1st-99.5th percentile
        vmin = np.percentile(data, 1)
        vmax = np.percentile(data, 99.5)
        ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto",
                  interpolation="nearest")
        # Red outline
        if outline.any():
            ax.contour(outline, levels=[0.5], colors=["red"], linewidths=0.5)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    # Title
    if features:
        title_str = (
            f"{stem}  |  act_out/in={features.get('activity_outside_ratio', 0):.3f}  "
            f"bnd_frac={features.get('boundary_fraction_yz', 0):.3f}  "
            f"cofire_frag={features.get('cofire_fragmentation', 0)}  "
            f"gap_frac={features.get('activity_gap_fraction', 0):.3f}"
        )
    else:
        title_str = stem
    fig.suptitle(title_str, fontsize=9, weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_all_runs(root: Path) -> list[tuple[Path, str]]:
    """Find all runs with ref3d + reviewed mask."""
    runs = []
    for mask_file in sorted(root.rglob("*_autoseg_labelmap_reviewed.tif")):
        rd = mask_file.parent
        if "old" in str(rd):
            continue
        stem = mask_file.name.replace("_autoseg_labelmap_reviewed.tif", "")
        ref = rd / f"{stem}_ref3d.tif"
        if ref.exists():
            runs.append((rd, stem))
    return runs


def main():
    import time

    root = AUTO_ROOT
    out_dir = root / "qc_review" / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = find_all_runs(root)
    print(f"[qc_review] Found {len(runs)} runs")

    all_features = []
    for rd, stem in runs:
        print(f"  {rd.relative_to(root)}/{stem}...", end=" ", flush=True)
        t0 = time.time()

        ref_path = rd / f"{stem}_ref3d.tif"
        mask_path = rd / f"{stem}_autoseg_labelmap_reviewed.tif"

        channels = load_ref3d(ref_path)
        mask = load_mask(mask_path)

        features = compute_features(channels, mask, stem, str(rd.relative_to(root)))
        all_features.append(features)

        sheet_path = out_dir / f"{stem}_sheet.png"
        render_sheet(channels, mask, stem, sheet_path, features)

        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s  act_out={features['activity_outside_ratio']:.3f} "
              f"bnd={features['boundary_fraction_yz']:.3f} "
              f"frag={features['cofire_fragmentation']} "
              f"gap={features['activity_gap_fraction']:.3f}")

    # Write features JSON
    feat_path = root / "qc_review" / "qc_review_features.json"
    feat_path.write_text(json.dumps(all_features, indent=2))
    print(f"\n[qc_review] Features: {feat_path}")
    print(f"[qc_review] Sheets: {out_dir}")

    return all_features


if __name__ == "__main__":
    main()
