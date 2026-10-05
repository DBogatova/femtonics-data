#!/usr/bin/env python
"""auto_regions.py — headless automatic region picking from a cell mask.

Given a cleaned 4D stack and a binary or labeled cell mask, this script:
  1. Skeletonizes the mask, builds the tree (reuses wrap_segments_napari math).
  2. Identifies the soma as the thick blob (if present).
  3. Orients the dendrite: column 0 (deep end) is the soma/proximal side.
  4. Computes geodesic distance from the proximal tip through the mask.
  5. Detects the SIDE BRANCH as a zone of multi-lobe cross-sections in the
     mask: columns where the YZ slice has 2+ connected components indicate
     the main path and a diverging branch. The bifurcation is placed at the
     start of this zone, and branch2 is the smaller lobe.
  6. Detects the CELL END as the farthest mask extent along the path (the
     mask itself may end before the scan tube does).
  7. Places regions as FULL CROSS-SECTION slices of the mask along the path:
       - reference (soma blob or proximal trunk_soma_end) at the proximal end
       - trunk1, trunk2: intermediate trunk regions before the fork
       - bifurcation: small region at the fork junction
       - branch2: the side lobe leaving the main path
       - main_branch1 [, main_branch2]: continuation of the main path after
         the fork, with the last region at the cell end
     Each region spans the full mask cross-section over its distance window.
     Within each placement zone the exact window is tuned by split-half
     reliability (never by correlation with the reference — no circularity).
  8. Naming convention (Daria's decision):
       soma | trunk_soma_end (reference when no soma)
       trunk1, trunk2 (before the fork)
       bifurcation (at the fork)
       branch2 (the side branch lobe)
       main_branch1, main_branch2 (continuing main path — compartment branch)
     compatible with compartment_of in run_metrics.py. If no fork is found,
     regions are trunk1..N only, with the last being "branch" (distal end).

Usage:
  python code/STEP9_auto/auto_regions.py <run_dir>/runNN_clean.tif \\
      --mask <mask.tif> [--exclude <exclude.tif>] --out-dir DIR [--voxel Z Y X]

Inputs are read-only; all outputs go to --out-dir.

Outputs:
  <stem>_segments_final.tif   uint8 labels 1..N
  <stem>_segments_final.json  compatible with wrap_segments_napari.py --check and
                               run_metrics.py / coherence_with_behavior.py
  <stem>_autoregions_qc.png   QC figure
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    convolve,
    distance_transform_edt,
    label as cc_label,
    uniform_filter1d,
)
from scipy.signal import find_peaks
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

# ---- path setup ----
HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "STEP7_workflow"))
sys.path.insert(0, str(CODE / "STEP8_stats"))

from common.voxel import add_voxel_arg, resolve_voxel
from common.cleanup import drop_small_islands, describe

__version__ = "0.5.1"

CC26 = np.ones((3, 3, 3), np.uint8)
CC8_2D = np.ones((3, 3), np.uint8)

# ---- Parameters (same for every cell — no per-run tuning) ----
MIN_ARC_VOX = 3              # drop skeleton arcs shorter than this
SOMA_FACTOR = 2.0            # radius factor for soma blob detection
MIN_ISLAND_VOX = 20          # drop small islands at save
SPUR_THRESHOLD_UM = 5.0      # skeleton branches shorter than this (um) are spurs
BRANCH_MIN_TERRITORY = 30    # territory voxels: branches smaller than this are spurs
SOMA_MIN_MAX_EDT_UM = 3.0    # soma blob must have max EDT >= this (um)
SOMA_MIN_VOXELS = 400        # soma blob must have at least this many voxels

# Region placement — matched to Daria's pattern (from 7 GT cells)
REGION_LENGTH_UM = 20.0       # preferred path span per region (um)
REGION_MIN_LENGTH_UM = 5.0    # smallest acceptable region span
BIF_REGION_LENGTH_UM = 8.0    # path span for the bifurcation region
MIN_REGION_VOXELS = 25        # minimum total voxels for a region
PROXIMAL_REF_LENGTH_UM = 12.0 # path span for proximal reference (no soma)

# Side-branch detection by multi-lobe cross-section analysis
MULTI_LOBE_MIN_SIZE = 2       # minimum voxels per lobe to count as a real lobe
MULTI_LOBE_RUN_MIN = 5        # need >= this many consecutive multi-lobe columns
SIDE_LOBE_MIN_COLS = 3        # the side lobe must span >= this many columns
SIDE_LOBE_MIN_VOXELS = 20     # the side lobe territory must be >= this
BIF_ZONE_PAD_COLS = 3         # how many columns before the multi-lobe start = bif center

# Cell-end detection
CELL_END_TAPER_THRESHOLD = 0.3  # column count / peak count below this = tapered off


# ==============================================================================
# Skeleton & tree math
# ==============================================================================
def skeleton_branch_points(skel):
    deg = convolve(skel.astype(np.uint8), CC26, mode="constant") - skel.astype(np.uint8)
    return skel & (deg >= 3)


def partition_skeleton_into_arcs(skel, min_arc_vox=1):
    nodes = skeleton_branch_points(skel)
    edges = skel & ~nodes
    lab, n = cc_label(edges, structure=CC26)
    if min_arc_vox > 1:
        sizes = np.bincount(lab.ravel())
        keep = [i for i in range(1, n + 1) if sizes[i] >= min_arc_vox]
        out = np.zeros_like(lab)
        for new, old in enumerate(keep, start=1):
            out[lab == old] = new
        return out.astype(np.int32), len(keep)
    return lab.astype(np.int32), int(n)


def geodesic_arc_partition(mask, arc_labels, voxel):
    costs = np.where(mask, 1.0, np.inf).astype(float)
    best = np.full(mask.shape, np.inf)
    part = np.zeros(mask.shape, np.int32)
    for a in (int(i) for i in np.unique(arc_labels) if i > 0):
        mcp = MCP_Geometric(costs, sampling=tuple(voxel))
        cc, _ = mcp.find_costs([tuple(p) for p in np.argwhere(arc_labels == a)])
        take = cc < best
        best[take] = cc[take]
        part[take] = a
    part[~mask] = 0
    part[np.isinf(best)] = 0
    return part, best


def arc_radii(edt, arc_labels):
    return {int(a): float(np.median(edt[arc_labels == a]))
            for a in np.unique(arc_labels) if a > 0}


def soma_blob_from_thickness(mask, edt, median_arc_radius, voxel, factor=SOMA_FACTOR):
    if median_arc_radius <= 0:
        return np.zeros(mask.shape, bool)
    core = mask & (edt >= factor * median_arc_radius)
    if not core.any():
        return np.zeros(mask.shape, bool)
    lab, n = cc_label(core, structure=CC26)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    blob = lab == int(sizes.argmax())
    soma_r = float(edt[blob].max())
    allowed = mask & (edt >= 0.5 * soma_r)
    grown = blob.copy()
    for _ in range(64):
        nxt = binary_dilation(grown, structure=CC26) & allowed
        if nxt.sum() == grown.sum():
            break
        grown = nxt
    steps = int(np.ceil(soma_r / float(min(voxel))))
    rind = grown.copy()
    for _ in range(steps):
        rind = binary_dilation(rind, structure=CC26) & mask
    dist_to_body = distance_transform_edt(~grown, sampling=tuple(voxel))
    rind &= dist_to_body <= soma_r
    return rind


# ==============================================================================
# Side-branch detection via multi-lobe cross-sections
# ==============================================================================
def detect_side_branch(mask, voxel, deep_end_first=True):
    """Detect a side branch by finding columns where the YZ cross-section
    has multiple connected components (lobes).

    Returns dict with:
      found: bool
      bif_x_range: (x_lo, x_hi) of the bifurcation zone
      branch2_mask: bool volume of the side-lobe voxels
      main_lobe_mask: bool volume of the main-lobe voxels in the multi-lobe zone
      multi_lobe_cols: list of (x, n_components, lobe_sizes)
      branch2_x_range: (x_lo, x_hi)
    or None if not found.
    """
    nZ, nY, nX = mask.shape
    col_counts = np.array([mask[:, :, x].sum() for x in range(nX)])

    # Find nonzero columns
    nonzero = np.where(col_counts > 0)[0]
    if len(nonzero) < 10:
        return None

    x_start, x_end = int(nonzero[0]), int(nonzero[-1])

    # Analyze each column for multi-lobe cross-sections
    multi_lobe_info = {}  # x -> (n_components, lobe_sizes, lobe_labels_2d)
    for x in range(x_start, x_end + 1):
        cs = mask[:, :, x]
        if cs.sum() < 4:
            continue
        lab2d, n = cc_label(cs, structure=CC8_2D)
        if n < 2:
            continue
        sizes = []
        for i in range(1, n + 1):
            s = int((lab2d == i).sum())
            if s >= MULTI_LOBE_MIN_SIZE:
                sizes.append((s, i))
        if len(sizes) >= 2:
            sizes.sort(reverse=True)
            multi_lobe_info[x] = (len(sizes), sizes, lab2d)

    if not multi_lobe_info:
        return None

    # Find runs of consecutive multi-lobe columns
    ml_cols = sorted(multi_lobe_info.keys())
    runs = []
    current_run = [ml_cols[0]]
    for i in range(1, len(ml_cols)):
        if ml_cols[i] - ml_cols[i - 1] <= 2:  # allow 1-column gap
            current_run.append(ml_cols[i])
        else:
            if len(current_run) >= MULTI_LOBE_RUN_MIN:
                runs.append(current_run)
            current_run = [ml_cols[i]]
    if len(current_run) >= MULTI_LOBE_RUN_MIN:
        runs.append(current_run)

    if not runs:
        return None

    # Pick the best run: the one with the clearest secondary lobe
    # (most columns and largest secondary lobe relative to primary)
    # Filter: median secondary lobe must be >= 5 voxels to exclude noise specks
    # (real bifurcations have median secondary >= 7, false positives ~3)
    MEDIAN_SECONDARY_MIN = 5
    best_run = None
    best_score = 0
    for run in runs:
        # Score: number of columns * median size of secondary lobe
        secondary_sizes = []
        for x in run:
            if x in multi_lobe_info:
                sizes = multi_lobe_info[x][1]
                if len(sizes) >= 2:
                    secondary_sizes.append(sizes[1][0])
        if not secondary_sizes:
            continue
        med_secondary = float(np.median(secondary_sizes))
        if med_secondary < MEDIAN_SECONDARY_MIN:
            continue  # secondary lobes too small — noise, not a real branch
        score = len(run) * med_secondary
        if score > best_score:
            best_score = score
            best_run = run

    if best_run is None:
        return None

    ml_start = best_run[0]
    ml_end = best_run[-1]

    # Build the side-lobe mask: for each multi-lobe column, identify the
    # secondary (smaller) lobe as the side branch, and the primary as main path
    branch2_mask = np.zeros(mask.shape, bool)
    main_lobe_mask = np.zeros(mask.shape, bool)

    # Determine which lobe is the "side" one by tracking continuity:
    # The main lobe is the one that connects to columns before the multi-lobe zone.
    # Use the centroid of each lobe to decide.

    # Get the main path centroid from columns just before the multi-lobe zone
    pre_cols = range(max(x_start, ml_start - 10), ml_start)
    pre_centroids_yz = []
    for x in pre_cols:
        cs = mask[:, :, x]
        if cs.sum() > 0:
            pts = np.argwhere(cs)
            pre_centroids_yz.append(pts.mean(0))
    if pre_centroids_yz:
        main_centroid = np.mean(pre_centroids_yz, axis=0)  # (Z, Y) of main path
    else:
        # Fallback: use the largest lobe at the first multi-lobe column
        main_centroid = None

    for x in best_run:
        if x not in multi_lobe_info:
            # Gap column — assign all mask voxels to main lobe
            main_lobe_mask[:, :, x] = mask[:, :, x]
            continue
        n, sizes, lab2d = multi_lobe_info[x]
        cs = mask[:, :, x]

        if main_centroid is not None:
            # Assign lobes: the one whose centroid is closest to the main path
            # is the main lobe; others are side branch
            lobe_centroids = {}
            for size, lbl in sizes:
                pts = np.argwhere(lab2d == lbl)
                lobe_centroids[lbl] = pts.mean(0)

            # Find the lobe closest to main_centroid
            main_lbl = min(lobe_centroids, key=lambda lbl:
                           np.linalg.norm(lobe_centroids[lbl] - main_centroid))

            for size, lbl in sizes:
                lobe_voxels = (lab2d == lbl)
                if lbl == main_lbl:
                    main_lobe_mask[:, :, x] |= lobe_voxels
                else:
                    branch2_mask[:, :, x] |= lobe_voxels
        else:
            # No pre-columns: largest lobe is main
            main_lbl = sizes[0][1]
            for size, lbl in sizes:
                lobe_voxels = (lab2d == lbl)
                if lbl == main_lbl:
                    main_lobe_mask[:, :, x] |= lobe_voxels
                else:
                    branch2_mask[:, :, x] |= lobe_voxels

    # Check that the side lobe is substantial enough
    branch2_vox = int(branch2_mask.sum())
    branch2_cols = np.where(branch2_mask.any(axis=(0, 1)))[0]

    if branch2_vox < SIDE_LOBE_MIN_VOXELS or len(branch2_cols) < SIDE_LOBE_MIN_COLS:
        return None

    branch2_x_range = (int(branch2_cols.min()), int(branch2_cols.max()))

    # The bifurcation zone: placed where the cross-section first widens
    # (start of the multi-lobe zone). Look for the column where the mask
    # cross-section count first jumps up relative to the pre-zone baseline.
    col_counts = np.array([mask[:, :, x].sum() for x in range(nX)])
    pre_zone = range(max(x_start, ml_start - 15), ml_start)
    pre_median = float(np.median([col_counts[x] for x in pre_zone if col_counts[x] > 0])) if len(pre_zone) > 0 else 1.0
    # Find first column where count is >=1.3x the pre-zone median
    bif_center = ml_start
    for x in range(max(x_start, ml_start - 5), min(nX, ml_start + 10)):
        if col_counts[x] > 1.3 * pre_median:
            bif_center = x
            break
    bif_half = BIF_ZONE_PAD_COLS
    bif_x_lo = max(x_start, bif_center - bif_half)
    bif_x_hi = min(x_end, bif_center + bif_half)

    return {
        "found": True,
        "bif_x_range": (bif_x_lo, bif_x_hi),
        "branch2_mask": branch2_mask,
        "main_lobe_mask": main_lobe_mask,
        "multi_lobe_cols": [(x, multi_lobe_info[x][0],
                             [s for s, _ in multi_lobe_info[x][1]])
                            for x in best_run if x in multi_lobe_info],
        "branch2_x_range": branch2_x_range,
        "multi_lobe_start": ml_start,
        "multi_lobe_end": ml_end,
        "branch2_voxels": branch2_vox,
    }


# ==============================================================================
# Cell-end detection
# ==============================================================================
def find_cell_end_x(mask, deep_end_first=True):
    """Find the X coordinate where the cell ends, which may be before the
    scan tube ends. Uses the mask's own extent.

    Returns (cell_start_x, cell_end_x).
    """
    col_counts = np.array([mask[:, :, x].sum() for x in range(mask.shape[2])])
    nonzero = np.where(col_counts > 0)[0]
    if len(nonzero) == 0:
        return 0, mask.shape[2] - 1
    return int(nonzero[0]), int(nonzero[-1])


# ==============================================================================
# Geodesic distance
# ==============================================================================
def geodesic_distance_from_seeds(mask, seed_voxels, voxel):
    """Geodesic distance from seed_voxels through the mask."""
    costs = np.where(mask, 1.0, np.inf).astype(float)
    mcp = MCP_Geometric(costs, sampling=tuple(voxel))
    seeds = [tuple(p) for p in seed_voxels]
    if not seeds:
        return np.full(mask.shape, np.inf)
    dist, _ = mcp.find_costs(seeds)
    return dist


# ==============================================================================
# Reliability
# ==============================================================================
def dff(t, f0_pct=10.0):
    f0 = np.percentile(t, f0_pct)
    return (t - f0) / max(f0, 1e-6)


def split_half_reliability(flat, idx, seed=0):
    """Spearman-Brown corrected split-half r."""
    if len(idx) < 8:
        return float("nan")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    a = dff(flat[:, perm[0::2]].mean(1).astype(np.float64))
    b = dff(flat[:, perm[1::2]].mean(1).astype(np.float64))
    r = float(np.corrcoef(a, b)[0, 1])
    return float(2 * r / (1 + r)) if r > -0.99 else float("nan")


# ==============================================================================
# Region placement helpers
# ==============================================================================
def slice_by_x(mask, x_lo, x_hi):
    """Boolean mask of all voxels in mask with X in [x_lo, x_hi]."""
    out = np.zeros(mask.shape, bool)
    x_lo = max(0, int(x_lo))
    x_hi = min(mask.shape[2] - 1, int(x_hi))
    out[:, :, x_lo:x_hi + 1] = mask[:, :, x_lo:x_hi + 1]
    return out


def slice_by_distance(mask, dist_from_root, d_lo, d_hi):
    """Boolean mask of all mask voxels with geodesic distance in [d_lo, d_hi)."""
    return mask & np.isfinite(dist_from_root) & (dist_from_root >= d_lo) & (dist_from_root < d_hi)


def best_window_in_zone_x(flat, mask, x_lo, x_hi, voxel,
                           target_cols=None,
                           min_voxels=MIN_REGION_VOXELS,
                           step_cols=2):
    """Within an X-column zone [x_lo, x_hi], find the best sub-window by
    split-half reliability.

    Returns (x_lo_best, x_hi_best, reliability, n_voxels) or None.
    """
    nX = mask.shape[2]
    x_lo = max(0, int(x_lo))
    x_hi = min(nX - 1, int(x_hi))
    zone_cols = x_hi - x_lo + 1
    if zone_cols < 3:
        return None

    if target_cols is None:
        vx = voxel[2]
        target_cols = max(5, int(round(REGION_LENGTH_UM / vx)))

    # If zone is close to target, use the whole zone
    if zone_cols <= int(target_cols * 1.3):
        w = slice_by_x(mask, x_lo, x_hi)
        n = int(w.sum())
        if n < min_voxels:
            return None
        idx = np.flatnonzero(w.ravel())
        rel = split_half_reliability(flat, idx)
        return (x_lo, x_hi, rel, n)

    # Search for best sub-window
    best = None
    best_rel = -1.0

    for length in sorted(set([target_cols,
                              max(5, int(target_cols * 0.7)),
                              int(target_cols * 1.3),
                              min(zone_cols, int(target_cols * 1.5))])):
        if length > zone_cols or length < 3:
            continue
        for start in range(x_lo, x_hi - length + 2, step_cols):
            end = start + length - 1
            w = slice_by_x(mask, start, end)
            n = int(w.sum())
            if n < min_voxels:
                continue
            idx = np.flatnonzero(w.ravel())
            rel = split_half_reliability(flat, idx)
            if rel == rel and rel > best_rel:
                best_rel = rel
                best = (start, end, rel, n)

    return best


def best_window_in_zone(flat, mask, dist_from_root, zone_lo, zone_hi, voxel,
                         target_length_um=REGION_LENGTH_UM,
                         min_length_um=REGION_MIN_LENGTH_UM,
                         min_voxels=MIN_REGION_VOXELS,
                         step_um=2.0):
    """Within a geodesic-distance zone [zone_lo, zone_hi), find the best
    full-cross-section sub-window by split-half reliability.

    Returns (d_lo_best, d_hi_best, reliability, n_voxels) or None.
    """
    zone_length = zone_hi - zone_lo
    if zone_length < min_length_um:
        return None

    if zone_length <= target_length_um * 1.3:
        w = slice_by_distance(mask, dist_from_root, zone_lo, zone_hi)
        n = int(w.sum())
        if n < min_voxels:
            return None
        idx = np.flatnonzero(w.ravel())
        rel = split_half_reliability(flat, idx)
        return (zone_lo, zone_hi, rel, n)

    best = None
    best_rel = -1.0

    lengths = sorted(set([
        target_length_um,
        target_length_um * 0.8,
        target_length_um * 1.2,
        max(min_length_um, target_length_um * 0.6),
        min(zone_length, target_length_um * 1.5),
    ]))

    for length in lengths:
        if length > zone_length or length < min_length_um:
            continue
        for start in np.arange(zone_lo, zone_hi - length + 0.01, step_um):
            end = min(start + length, zone_hi)
            w = slice_by_distance(mask, dist_from_root, start, end)
            n = int(w.sum())
            if n < min_voxels:
                continue
            idx = np.flatnonzero(w.ravel())
            rel = split_half_reliability(flat, idx)
            if rel == rel and rel > best_rel:
                best_rel = rel
                best = (float(start), float(end), rel, n)

    return best


# ==============================================================================
# Main region-picking logic
# ==============================================================================
def pick_regions(stack_path, mask, voxel, exclude=None, deep_end_first=True,
                 mask_meta=None):
    """Automatic region picking. Returns (segments_vol, metadata_dict).

    mask_meta: dict from the auto_mask JSON sidecar (reviews[0].params).
       When present, trusts soma_detected / soma_end_x / cell_start_x /
       cell_end_x from the mask stage rather than re-detecting from the EDT
       (avoids interaction bugs where the EDT-based detection finds a thick zone
       at the wrong end of the cell).
    """
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"auto_regions v{__version__}")
    log(f"Loading stack: {stack_path}")
    stack = tifffile.imread(str(stack_path))
    T = stack.shape[0]
    flat = stack.reshape(T, -1)
    log(f"  shape: {stack.shape}, T={T}")

    mask = mask.astype(bool)
    if exclude is not None:
        excl_bool = exclude.astype(bool)
    else:
        excl_bool = None

    # ================================================================
    # 1. CELL END: the mask's own extent along X
    # ================================================================
    cell_start_x, cell_end_x = find_cell_end_x(mask, deep_end_first)
    nX = mask.shape[2]
    log(f"  cell X extent: [{cell_start_x}, {cell_end_x}] of {nX}")

    # ================================================================
    # 2. SOMA detection via EDT thickness
    # ================================================================
    edt = distance_transform_edt(mask, sampling=tuple(voxel))
    skel = skeletonize(mask)
    arc_labels, n_arcs = partition_skeleton_into_arcs(skel, MIN_ARC_VOX)
    radii = arc_radii(edt, arc_labels)
    median_arc_radius = float(np.median(list(radii.values()))) if radii else 0.0

    # --- Use mask_meta to guide soma detection (avoids re-detection at wrong end) ---
    mask_soma_hint = None
    if mask_meta is not None:
        _soma_det = mask_meta.get("soma_detected", None)
        _soma_end_x = mask_meta.get("soma_end_x", None)
        if _soma_det is True and _soma_end_x is not None:
            mask_soma_hint = int(_soma_end_x)
            log(f"  mask_meta: soma_detected=True, soma_end_x={mask_soma_hint}")
        elif _soma_det is False:
            mask_soma_hint = "no_soma"
            log(f"  mask_meta: soma_detected=False")

    soma_blob = soma_blob_from_thickness(mask, edt, median_arc_radius, voxel)

    has_soma = False
    if mask_soma_hint == "no_soma":
        # Trust the mask: no soma
        log(f"  no soma (mask_meta override)")
        soma_blob = np.zeros(mask.shape, bool)
    elif soma_blob.any():
        # If we have a soma hint, constrain the blob to the correct end
        if mask_soma_hint is not None and isinstance(mask_soma_hint, int):
            blob_x = np.argwhere(soma_blob)[:, 2]
            blob_x_center = float(blob_x.mean())
            # soma_end_x is in columns from the deep end. Check if the blob is
            # at the correct end (near soma_end_x, which is proximal / low X
            # when deep_end_first).
            soma_zone_x = mask_soma_hint  # the X column where the soma ends
            if deep_end_first:
                # soma should be at low X (near column 0 to soma_end_x)
                if blob_x_center > soma_zone_x + 30:
                    # Blob is at the WRONG end — rebuild soma from the correct end
                    log(f"  soma blob at X={blob_x_center:.0f} is beyond soma_end_x={soma_zone_x}; rebuilding at proximal end")
                    # Use only the proximal columns
                    prox_zone = mask.copy()
                    prox_zone[:, :, soma_zone_x + 20:] = False
                    soma_blob = soma_blob_from_thickness(prox_zone, edt * prox_zone, median_arc_radius, voxel)
                    if not soma_blob.any():
                        # Fallback: all mask voxels up to soma_end_x
                        soma_blob = mask & (np.arange(mask.shape[2])[None, None, :] <= soma_zone_x + 5)
                        log(f"  soma rebuilt from X≤{soma_zone_x + 5}: {soma_blob.sum()} vox")
            else:
                if blob_x_center < nX - soma_zone_x - 30:
                    log(f"  soma blob at wrong end (deep_end_first=False); rebuilding")
                    prox_zone = mask.copy()
                    prox_zone[:, :, :nX - soma_zone_x - 20] = False
                    soma_blob = soma_blob_from_thickness(prox_zone, edt * prox_zone, median_arc_radius, voxel)

        max_edt = float(edt[soma_blob].max()) if soma_blob.any() else 0
        n_blob = int(soma_blob.sum())
        n_mask = int(mask.sum())
        frac = n_blob / max(n_mask, 1)
        if soma_blob.any() and max_edt >= SOMA_MIN_MAX_EDT_UM and n_blob >= SOMA_MIN_VOXELS and frac < 0.30:
            has_soma = True
            log(f"  soma blob: {n_blob} vox ({frac:.1%} of mask), max_edt={max_edt:.2f} um")
        elif soma_blob.any() and mask_soma_hint is not None and isinstance(mask_soma_hint, int):
            # Mask said there's a soma — relax criteria if the blob has enough voxels
            if n_blob >= 50 and max_edt >= 1.5:
                has_soma = True
                log(f"  soma blob (mask_meta relaxed): {n_blob} vox ({frac:.1%}), max_edt={max_edt:.2f} um")
            else:
                reason = f"too small even with hint: vox={n_blob}, max_edt={max_edt:.2f}"
                log(f"  soma blob rejected: {reason}")
                soma_blob = np.zeros(mask.shape, bool)
        else:
            reason = ""
            if max_edt < SOMA_MIN_MAX_EDT_UM:
                reason += f"max_edt={max_edt:.2f} < {SOMA_MIN_MAX_EDT_UM}; "
            if n_blob < SOMA_MIN_VOXELS:
                reason += f"voxels={n_blob} < {SOMA_MIN_VOXELS}; "
            if frac >= 0.30:
                reason += f"frac={frac:.1%} >= 30% (thick tube, not a soma); "
            log(f"  soma blob rejected: {n_blob} vox, {reason.rstrip('; ')}")
            soma_blob = np.zeros(mask.shape, bool)
    else:
        log(f"  no soma blob found")

    # ================================================================
    # 3. SIDE-BRANCH detection via cross-section lobe analysis
    # ================================================================
    sb = detect_side_branch(mask, voxel, deep_end_first)
    if sb is not None:
        log(f"  side branch FOUND: multi-lobe X=[{sb['multi_lobe_start']},{sb['multi_lobe_end']}], "
            f"branch2 X={sb['branch2_x_range']}, {sb['branch2_voxels']} vox")
        log(f"  bifurcation X range: {sb['bif_x_range']}")
    else:
        log(f"  no side branch detected")

    # ================================================================
    # 4. GEODESIC DISTANCE from the proximal tip
    # ================================================================
    if has_soma:
        root_voxels = np.argwhere(soma_blob)
    else:
        # Proximal tip: the mask voxels at the deepest X columns
        if deep_end_first:
            tip_x = cell_start_x
        else:
            tip_x = cell_end_x
        tip_vox = np.argwhere(mask[:, :, tip_x])
        if len(tip_vox) == 0:
            # Find the nearest nonempty column
            for dx in range(1, nX):
                for xx in [tip_x + dx, tip_x - dx]:
                    if 0 <= xx < nX:
                        v = np.argwhere(mask[:, :, xx])
                        if len(v) > 0:
                            tip_vox = v
                            tip_x = xx
                            break
                if len(tip_vox) > 0:
                    break
        root_voxels = np.column_stack([tip_vox, np.full(len(tip_vox), tip_x)])

    dist_from_root = geodesic_distance_from_seeds(mask, root_voxels, voxel)

    # Path stats
    main_dists = dist_from_root[mask & np.isfinite(dist_from_root)]
    if has_soma:
        main_dists = main_dists[main_dists > 0]
    total_path_um = float(main_dists.max()) if len(main_dists) > 0 else 0.0
    log(f"  total path length: {total_path_um:.1f} um")

    # ================================================================
    # 5. BUILD REGIONS
    # ================================================================
    regions = []  # list of (name, mask_bool, role, reliability)

    vx = voxel[2]  # X pixel size in um

    # ---- Reference region ----
    if has_soma:
        n_soma = int(soma_blob.sum())
        if n_soma >= MIN_REGION_VOXELS:
            idx = np.flatnonzero(soma_blob.ravel())
            rel = split_half_reliability(flat, idx)
            regions.append(("soma", soma_blob.copy(), "soma", rel))
            log(f"  REGION soma: {n_soma} vox, rel={rel:.4f}")
            # The reference extends to the end of the soma in X
            soma_x_max = int(np.argwhere(soma_blob)[:, 2].max())
    else:
        # trunk_soma_end: proximal reference
        ref_cols = max(5, int(round(PROXIMAL_REF_LENGTH_UM / vx)))
        if deep_end_first:
            ref_x_lo = cell_start_x
            ref_x_hi = min(cell_end_x, cell_start_x + ref_cols - 1)
        else:
            ref_x_hi = cell_end_x
            ref_x_lo = max(cell_start_x, cell_end_x - ref_cols + 1)
        w = best_window_in_zone_x(flat, mask, ref_x_lo, ref_x_hi, voxel,
                                  target_cols=ref_cols)
        if w is not None:
            rmask = slice_by_x(mask, w[0], w[1])
            regions.append(("trunk_soma_end", rmask, "trunk", w[2]))
            log(f"  REGION trunk_soma_end: {w[3]} vox, X=[{w[0]},{w[1]}], rel={w[2]:.4f}")
        else:
            rmask = slice_by_x(mask, ref_x_lo, ref_x_hi)
            n = int(rmask.sum())
            if n >= MIN_REGION_VOXELS:
                idx = np.flatnonzero(rmask.ravel())
                rel = split_half_reliability(flat, idx)
                regions.append(("trunk_soma_end", rmask, "trunk", rel))
                log(f"  REGION trunk_soma_end (fallback): {n} vox, rel={rel:.4f}")

    # ---- Determine the main-path zones ----
    # Proximal end: end of soma or start of cell
    if has_soma:
        prox_x = soma_x_max + 1
    elif regions:
        # End of the trunk_soma_end region
        ref_pts = np.argwhere(regions[-1][1])
        if deep_end_first:
            prox_x = int(ref_pts[:, 2].max()) + 1
        else:
            prox_x = int(ref_pts[:, 2].min()) - 1
    else:
        prox_x = cell_start_x if deep_end_first else cell_end_x

    # Distal end: cell end
    distal_x = cell_end_x if deep_end_first else cell_start_x

    # If side branch found, determine the bifurcation location
    bif_x = None
    branch2_region_mask = None
    if sb is not None:
        bif_x_lo, bif_x_hi = sb["bif_x_range"]
        bif_x = (bif_x_lo + bif_x_hi) // 2
        branch2_region_mask = sb["branch2_mask"]

    # Main path length (in columns, then um)
    if deep_end_first:
        main_cols = distal_x - prox_x + 1
    else:
        main_cols = prox_x - distal_x + 1

    main_path_um = main_cols * vx
    log(f"  main path after reference: {main_cols} cols, {main_path_um:.1f} um")

    # ---- Trunk regions (before the fork, or along the full path if no fork) ----
    if sb is not None and bif_x is not None:
        # Trunk extends from prox_x to just before the bifurcation
        if deep_end_first:
            trunk_x_lo = prox_x
            trunk_x_hi = bif_x_lo - 1
        else:
            trunk_x_hi = prox_x
            trunk_x_lo = bif_x_hi + 1
    else:
        # No fork: trunk is the full main path
        if deep_end_first:
            trunk_x_lo = prox_x
            trunk_x_hi = distal_x
        else:
            trunk_x_lo = distal_x
            trunk_x_hi = prox_x

    trunk_cols = trunk_x_hi - trunk_x_lo + 1
    trunk_um = trunk_cols * vx

    # How many trunk regions? ~ one per 80-100 um, minimum 1, maximum 3
    target_cols_per_region = max(10, int(round(REGION_LENGTH_UM / vx)))
    if trunk_um < 50:
        n_trunk = 1
    elif trunk_um < 180:
        n_trunk = 2
    else:
        n_trunk = min(3, max(2, int(round(trunk_um / 100.0))))

    log(f"  trunk zone: X=[{trunk_x_lo},{trunk_x_hi}], {trunk_um:.0f} um -> {n_trunk} trunk regions")

    # Place trunk regions spread along the trunk zone
    # Pattern from GT: roughly at 25-30% and 55-60% of the trunk zone
    trunk_name_start = 1
    if has_soma:
        trunk_name_start = 1  # trunk, trunk2
    else:
        trunk_name_start = 1  # trunk_soma_end already placed, but we already named it
        # Next trunk regions named trunk, trunk2

    # For naming: if reference is trunk_soma_end, trunks are "trunk", "trunk2"
    # If reference is soma, trunks are "trunk", "trunk2"
    trunk_fracs = np.linspace(0.15, 0.85, n_trunk + 2)[1:-1]  # skip endpoints

    trunk_count = 0
    for fi, frac in enumerate(trunk_fracs):
        center_x = int(trunk_x_lo + frac * trunk_cols)
        half = target_cols_per_region // 2

        x_lo = max(trunk_x_lo, center_x - half)
        x_hi = min(trunk_x_hi, center_x + half)

        w = best_window_in_zone_x(flat, mask, x_lo, x_hi, voxel,
                                  target_cols=target_cols_per_region)
        if w is not None:
            rmask = slice_by_x(mask, w[0], w[1])
            if has_soma:
                rmask &= ~soma_blob
            n = int(rmask.sum())
            if n >= MIN_REGION_VOXELS:
                trunk_count += 1
                if trunk_count == 1:
                    name = "trunk" if has_soma else "trunk"
                else:
                    name = f"trunk{trunk_count}"
                regions.append((name, rmask, "trunk", w[2]))
                log(f"  REGION {name}: {n} vox, X=[{w[0]},{w[1]}], rel={w[2]:.4f}")
                continue

        # Fallback
        rmask = slice_by_x(mask, x_lo, x_hi)
        if has_soma:
            rmask &= ~soma_blob
        n = int(rmask.sum())
        if n >= MIN_REGION_VOXELS:
            trunk_count += 1
            if trunk_count == 1:
                name = "trunk" if has_soma else "trunk"
            else:
                name = f"trunk{trunk_count}"
            idx = np.flatnonzero(rmask.ravel())
            rel = split_half_reliability(flat, idx)
            regions.append((name, rmask, "trunk", rel))
            log(f"  REGION {name} (fallback): {n} vox, X=[{x_lo},{x_hi}], rel={rel:.4f}")

    # ---- Rename trunk regions to be consistent ----
    # Daria's pattern: trunk1, trunk2 (or trunk, trunk2 when soma present)
    # With trunk_soma_end as reference, the trunks should be "trunk", "trunk2"
    # With soma as reference, the trunks should be "trunk", "trunk2"
    # Let's just fix the naming sequentially
    trunk_regions = [(i, r) for i, r in enumerate(regions) if r[2] == "trunk" and r[0] not in ("trunk_soma_end", "soma")]
    for seq, (idx, r) in enumerate(trunk_regions):
        if seq == 0:
            new_name = "trunk"
        else:
            new_name = f"trunk{seq + 1}"
        regions[idx] = (new_name, r[1], r[2], r[3])

    # ---- Bifurcation region ----
    if sb is not None:
        bif_x_lo, bif_x_hi = sb["bif_x_range"]
        bif_mask = slice_by_x(mask, bif_x_lo, bif_x_hi)
        if has_soma:
            bif_mask &= ~soma_blob
        n_bif = int(bif_mask.sum())
        if n_bif >= MIN_REGION_VOXELS:
            idx = np.flatnonzero(bif_mask.ravel())
            rel = split_half_reliability(flat, idx)
            regions.append(("bifurcation", bif_mask, "branch", rel))
            log(f"  REGION bifurcation: {n_bif} vox, X=[{bif_x_lo},{bif_x_hi}], rel={rel:.4f}")

    # ---- Branch2 region (the side lobe) ----
    if sb is not None and branch2_region_mask is not None:
        b2_vox = int(branch2_region_mask.sum())
        if b2_vox >= MIN_REGION_VOXELS:
            idx = np.flatnonzero(branch2_region_mask.ravel())
            rel = split_half_reliability(flat, idx)
            regions.append(("branch2", branch2_region_mask.copy(), "branch", rel))
            b2_x = sb["branch2_x_range"]
            log(f"  REGION branch2 (side lobe): {b2_vox} vox, X=[{b2_x[0]},{b2_x[1]}], rel={rel:.4f}")

    # ---- Main-branch regions (after the fork) ----
    if sb is not None:
        # Main branch starts after the multi-lobe zone ends
        # The main path continues; the side branch has been separated
        if deep_end_first:
            mb_x_lo = sb["multi_lobe_end"] + 1
            mb_x_hi = distal_x
        else:
            mb_x_hi = sb["multi_lobe_start"] - 1
            mb_x_lo = distal_x

        mb_cols = mb_x_hi - mb_x_lo + 1
        mb_um = mb_cols * vx

        if mb_cols > 3:
            # How many main_branch regions?
            if mb_um < 60:
                n_mb = 1
            elif mb_um < 180:
                n_mb = 2
            else:
                n_mb = min(3, max(2, int(round(mb_um / 100.0))))

            log(f"  main_branch zone: X=[{mb_x_lo},{mb_x_hi}], {mb_um:.0f} um -> {n_mb} regions")

            mb_fracs = np.linspace(0.1, 0.9, n_mb + 2)[1:-1]

            mb_count = 0
            for fi, frac in enumerate(mb_fracs):
                center_x = int(mb_x_lo + frac * mb_cols)
                half = target_cols_per_region // 2

                # Last region should be anchored near the distal end
                if fi == len(mb_fracs) - 1:
                    if deep_end_first:
                        x_hi = mb_x_hi
                        x_lo = max(mb_x_lo, x_hi - int(target_cols_per_region * 1.5))
                    else:
                        x_lo = mb_x_lo
                        x_hi = min(mb_x_hi, x_lo + int(target_cols_per_region * 1.5))
                else:
                    x_lo = max(mb_x_lo, center_x - half)
                    x_hi = min(mb_x_hi, center_x + half)

                # Exclude branch2 voxels from main_branch regions
                w = best_window_in_zone_x(flat, mask & ~branch2_region_mask, x_lo, x_hi, voxel,
                                          target_cols=target_cols_per_region)
                if w is not None:
                    rmask = slice_by_x(mask & ~branch2_region_mask, w[0], w[1])
                    if has_soma:
                        rmask &= ~soma_blob
                    n = int(rmask.sum())
                    if n >= MIN_REGION_VOXELS:
                        mb_count += 1
                        if mb_count == 1:
                            name = "main_branch"
                        elif mb_count == 2:
                            name = "main_branch2"
                        else:
                            name = f"main_branch{mb_count}"
                        regions.append((name, rmask, "branch", w[2]))
                        log(f"  REGION {name}: {n} vox, X=[{w[0]},{w[1]}], rel={w[2]:.4f}")
                        continue

                # Fallback
                rmask = slice_by_x(mask & ~branch2_region_mask, x_lo, x_hi)
                if has_soma:
                    rmask &= ~soma_blob
                n = int(rmask.sum())
                if n >= MIN_REGION_VOXELS:
                    mb_count += 1
                    if mb_count == 1:
                        name = "main_branch"
                    elif mb_count == 2:
                        name = "main_branch2"
                    else:
                        name = f"main_branch{mb_count}"
                    idx = np.flatnonzero(rmask.ravel())
                    rel = split_half_reliability(flat, idx)
                    regions.append((name, rmask, "branch", rel))
                    log(f"  REGION {name} (fallback): {n} vox, X=[{x_lo},{x_hi}], rel={rel:.4f}")
    else:
        # No fork: the distal end is "branch" (Daria's run07 pattern: distal = branch)
        # Place a distal region
        if deep_end_first:
            dist_x_hi = distal_x
            dist_x_lo = max(trunk_x_lo, distal_x - int(target_cols_per_region * 1.5))
        else:
            dist_x_lo = distal_x
            dist_x_hi = min(trunk_x_hi, distal_x + int(target_cols_per_region * 1.5))

        w = best_window_in_zone_x(flat, mask, dist_x_lo, dist_x_hi, voxel,
                                  target_cols=target_cols_per_region)
        if w is not None:
            rmask = slice_by_x(mask, w[0], w[1])
            if has_soma:
                rmask &= ~soma_blob
            n = int(rmask.sum())
            if n >= MIN_REGION_VOXELS:
                # Check if this overlaps with any trunk region already placed
                overlap = False
                for rn, rm, rr, _ in regions:
                    if rr == "trunk" and rn not in ("trunk_soma_end",):
                        if (rmask & rm).sum() > 0.3 * n:
                            overlap = True
                            break
                if not overlap:
                    regions.append(("branch", rmask, "branch", w[2]))
                    log(f"  REGION branch (distal): {n} vox, X=[{w[0]},{w[1]}], rel={w[2]:.4f}")

    if not regions:
        raise ValueError("no regions found")

    # ================================================================
    # 6. BUILD OUTPUT
    # ================================================================
    seg = np.zeros(mask.shape, np.uint8)
    region_meta = []

    # Sort regions by their median X coordinate
    def region_median_x(r):
        name, rmask, role, rel = r
        pts = np.argwhere(rmask)
        return float(pts[:, 2].mean()) if len(pts) > 0 else 0.0

    if deep_end_first:
        regions.sort(key=region_median_x)
    else:
        regions.sort(key=lambda r: -region_median_x(r))

    label_num = 1
    for name, rmask, role, rel in regions:
        seg[rmask] = label_num
        pts = np.argwhere(rmask)
        x_lo_r = int(pts[:, 2].min()) if len(pts) > 0 else 0
        x_hi_r = int(pts[:, 2].max()) if len(pts) > 0 else 0
        region_meta.append({
            "label": label_num,
            "name": name,
            "role": role,
            "voxels": int(rmask.sum()),
            "reliability": round(rel, 4) if rel == rel else 0.0,
            "x_range": [x_lo_r, x_hi_r],
            "x_span_um": round((x_hi_r - x_lo_r) * vx, 1),
        })
        label_num += 1

    # Clamp to mask
    seg[~mask] = 0
    if excl_bool is not None:
        seg[excl_bool] = 0

    # Drop small islands
    seg, rep = drop_small_islands(seg, min_voxels=MIN_ISLAND_VOX)
    if rep:
        log(f"  island cleanup: {describe(rep)}")

    # Determine reference
    ref_label = None
    for rm in region_meta:
        if rm["name"] == "soma" or rm["name"] == "trunk_soma_end":
            ref_label = rm["label"]
            break
    if ref_label is None:
        trunk_regions = [rm for rm in region_meta if rm["role"] == "trunk"]
        if trunk_regions:
            ref_label = trunk_regions[0]["label"]

    # Compute geodesic distances from reference
    if ref_label is not None:
        ref_voxels = np.argwhere(seg == ref_label)
        if len(ref_voxels) > 0:
            ref_dist = geodesic_distance_from_seeds(mask, ref_voxels, voxel)
            for rm in region_meta:
                d = ref_dist[seg == rm["label"]]
                d = d[np.isfinite(d)]
                rm["distance_um"] = round(float(np.median(d)), 2) if len(d) > 0 else 0.0
    else:
        for rm in region_meta:
            rm["distance_um"] = 0.0

    region_meta.sort(key=lambda rm: rm.get("distance_um", 0))

    # Reason strings
    for rm in region_meta:
        reasons = []
        if rm["name"] == "soma":
            reasons.append("thick blob (EDT >= 2x median arc radius)")
        elif rm["name"] == "trunk_soma_end":
            reasons.append("proximal reference (deep end, no soma detected)")
        elif rm["name"] == "bifurcation":
            reasons.append("junction where cross-section splits into multiple lobes")
        elif rm["name"] == "branch2":
            reasons.append("side lobe diverging from main path in cross-section")
        elif rm["name"].startswith("main_branch"):
            reasons.append("main-path continuation beyond the side-branch fork")
        elif rm["name"] == "branch":
            reasons.append("distal end of the dendrite (no fork detected)")
        elif "trunk" in rm["name"]:
            reasons.append("main-path trunk region")
        else:
            reasons.append("region")
        reasons.append(f"full cross-section, {rm['voxels']} vox, rel={rm['reliability']:.4f}")
        rm["reason"] = "; ".join(reasons)

    reference_type = "soma" if has_soma else "proximal_trunk"
    reference_name = None
    for rm in region_meta:
        if rm["label"] == ref_label:
            reference_name = rm["name"]
            break

    flags = []
    if sb is None:
        flags.append("no_side_branch_detected")
    if not has_soma:
        flags.append("no_soma_detected")

    return seg, {
        "regions": region_meta,
        "reference": reference_type,
        "reference_region": reference_name,
        "reference_label": ref_label,
        "side_branch": {
            "found": sb is not None,
            "multi_lobe_start": sb["multi_lobe_start"] if sb else None,
            "multi_lobe_end": sb["multi_lobe_end"] if sb else None,
            "bif_x_range": list(sb["bif_x_range"]) if sb else None,
            "branch2_x_range": list(sb["branch2_x_range"]) if sb else None,
            "branch2_voxels": sb["branch2_voxels"] if sb else None,
        } if sb else {"found": False},
        "cell_end_x": cell_end_x,
        "cell_start_x": cell_start_x,
        "flags": flags,
        "tree_info": {
            "has_soma": has_soma,
            "soma_blob_voxels": int(soma_blob.sum()) if soma_blob is not None else 0,
            "median_arc_radius_um": median_arc_radius,
            "deep_end_first": deep_end_first,
            "total_path_length_um": round(total_path_um, 2),
            "n_arcs": n_arcs,
        },
        "log": log_lines,
    }


# ==============================================================================
# JSON sidecar (wrap_segments_napari compatible)
# ==============================================================================
def build_sidecar(seg, voxel, meta, stack_path, mask_path, version=__version__):
    labels = sorted(int(v) for v in np.unique(seg) if v > 0)
    region_map = {rm["label"]: rm for rm in meta["regions"]}

    segment_names = {}
    voxels_per_label = {}
    for lb in labels:
        rm = region_map.get(lb, {})
        segment_names[str(lb)] = rm.get("name", f"seg{lb}")
        voxels_per_label[str(lb)] = int((seg == lb).sum())

    sidecar = {
        "created": datetime.datetime.now().astimezone().isoformat(),
        "tool": "auto_regions.py",
        "voxel_zyx_um": list(voxel),
        "labels": {str(lb): segment_names.get(str(lb), f"seg{lb}") for lb in labels},
        "voxels_per_label": voxels_per_label,
        "source_clean": str(stack_path),
        "source_mask": str(mask_path),
        "accepted_cells": [1],
        "segment_names": segment_names,
        "islands_removed": [],
        "min_island_voxels": MIN_ISLAND_VOX,
        "wrap_clicks": [],
        "auto_regions": {
            "version": version,
            "params": {
                "region_length_um": REGION_LENGTH_UM,
                "region_min_length_um": REGION_MIN_LENGTH_UM,
                "min_region_voxels": MIN_REGION_VOXELS,
                "bif_region_length_um": BIF_REGION_LENGTH_UM,
                "proximal_ref_length_um": PROXIMAL_REF_LENGTH_UM,
                "soma_factor": SOMA_FACTOR,
                "multi_lobe_run_min": MULTI_LOBE_RUN_MIN,
                "side_lobe_min_cols": SIDE_LOBE_MIN_COLS,
                "side_lobe_min_voxels": SIDE_LOBE_MIN_VOXELS,
            },
            "reference": meta["reference"],
            "reference_region": meta.get("reference_region"),
            "regions": {
                rm["name"]: {
                    "label": rm["label"],
                    "name": rm["name"],
                    "voxels": rm["voxels"],
                    "distance_um": rm.get("distance_um", 0.0),
                    "reliability": rm["reliability"],
                    "x_range": rm.get("x_range", [0, 0]),
                    "reason": rm.get("reason", ""),
                }
                for rm in meta["regions"]
            },
            "side_branch": meta.get("side_branch", {"found": False}),
            "cell_end_x": meta.get("cell_end_x"),
            "tree_info": meta["tree_info"],
            "flags": meta.get("flags", []),
        },
    }
    return sidecar


# ==============================================================================
# QC figure
# ==============================================================================
def make_qc_figure(seg, mask, voxel, meta, stack_path, out_path, flat=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if flat is None:
        stack = tifffile.imread(str(stack_path))
        T = stack.shape[0]
        flat = stack.reshape(T, -1)
    else:
        T = flat.shape[0]

    stack_3d = flat.mean(0).reshape(mask.shape)
    labels = sorted(int(v) for v in np.unique(seg) if v > 0)
    region_map = {rm["label"]: rm for rm in meta["regions"]}
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # XZ MIP
    ax = axes[0, 0]
    anat_xz = stack_3d.max(1)
    ax.imshow(anat_xz, cmap="gray", aspect="auto",
              extent=[0, anat_xz.shape[1] * voxel[2], anat_xz.shape[0] * voxel[0], 0])
    seg_xz = seg.max(1)
    for i, lb in enumerate(labels):
        rm = seg_xz == lb
        rgba = np.zeros((*rm.shape, 4))
        rgba[rm] = list(colors[i][:3]) + [0.45]
        ax.imshow(rgba, aspect="auto",
                  extent=[0, anat_xz.shape[1] * voxel[2], anat_xz.shape[0] * voxel[0], 0])
        pts = np.argwhere(rm)
        if len(pts):
            cy, cx = pts.mean(0)
            name = region_map.get(lb, {}).get("name", f"{lb}")
            ax.text(cx * voxel[2], cy * voxel[0], name, fontsize=7, color="white",
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=colors[i][:3], alpha=0.7))
    ax.set_title("XZ MIP: regions")
    ax.set_xlabel("X (um)"); ax.set_ylabel("Z (um)")

    # XY MIP
    ax = axes[0, 1]
    anat_xy = stack_3d.max(0)
    ax.imshow(anat_xy, cmap="gray", aspect="auto",
              extent=[0, anat_xy.shape[1] * voxel[2], anat_xy.shape[0] * voxel[1], 0])
    seg_xy = seg.max(0)
    for i, lb in enumerate(labels):
        rm = seg_xy == lb
        rgba = np.zeros((*rm.shape, 4))
        rgba[rm] = list(colors[i][:3]) + [0.45]
        ax.imshow(rgba, aspect="auto",
                  extent=[0, anat_xy.shape[1] * voxel[2], anat_xy.shape[0] * voxel[1], 0])
    ax.set_title("XY MIP: regions")
    ax.set_xlabel("X (um)"); ax.set_ylabel("Y (um)")

    # Traces
    ax = axes[1, 0]
    offset = 0
    for i, lb in enumerate(labels):
        idx = np.flatnonzero((seg == lb).ravel())
        if len(idx) == 0:
            continue
        trace = flat[:, idx].mean(1).astype(np.float64)
        f0 = np.percentile(trace, 10)
        trace_dff = (trace - f0) / max(f0, 1e-6)
        rm = region_map.get(lb, {})
        name = rm.get("name", f"seg{lb}")
        rel = rm.get("reliability", 0)
        ax.plot(np.arange(T), trace_dff + offset, color=colors[i][:3], linewidth=0.5,
                label=f"{name} (rel={rel:.3f}, {rm.get('voxels', '?')}vox)")
        offset += max(trace_dff.max() - trace_dff.min(), 0.5) + 0.3
    ax.set_xlabel("Frame"); ax.set_ylabel("dF/F (stacked)")
    ax.set_title("Region traces")
    ax.legend(fontsize=7, loc="upper right")

    # Summary table
    ax = axes[1, 1]; ax.axis("off")
    headers = ["Region", "Voxels", "X range", "Dist(um)", "Rel", "Role"]
    table_data = [[rm["name"], str(rm["voxels"]),
                   f"{rm.get('x_range', [0,0])[0]}-{rm.get('x_range', [0,0])[1]}",
                   f"{rm.get('distance_um', 0):.1f}",
                   f"{rm['reliability']:.4f}", rm["role"]]
                  for rm in meta["regions"]]
    if table_data:
        table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.3)

    sb_info = meta.get("side_branch", {})
    title_extra = ""
    if sb_info.get("found"):
        title_extra = f" | bif X={sb_info.get('bif_x_range')}"
    ax.set_title(f"auto_regions v{__version__}: {Path(stack_path).stem}{title_extra}", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# JSONL logging
# ==============================================================================
def log_jsonl(stage_file, entry):
    os.makedirs(os.path.dirname(stage_file), exist_ok=True)
    entry["time"] = datetime.datetime.now().astimezone().isoformat()
    with open(stage_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ==============================================================================
# Main entry point
# ==============================================================================
def run_auto_regions(stack_path, mask_path, exclude_path=None, out_dir=None,
                     voxel=None, deep_end_first=True):
    stack_path = Path(stack_path)
    mask_path = Path(mask_path)
    stem = stack_path.stem

    mask_raw = tifffile.imread(str(mask_path))
    if mask_raw.max() == 2 and (mask_raw == 2).any():
        mask = (mask_raw == 2)
    elif mask_raw.max() == 1:
        mask = mask_raw.astype(bool)
    else:
        mask = mask_raw > 0

    exclude = None
    if exclude_path and Path(exclude_path).exists():
        exclude = tifffile.imread(str(exclude_path)).astype(bool)

    if voxel is None:
        voxel = resolve_voxel(stack_path, None, quiet=False)
    else:
        voxel = tuple(float(v) for v in voxel)
        print(f"[voxel] Z Y X = {voxel[0]:.3f} {voxel[1]:.3f} {voxel[2]:.3f} um  (source: command line)")

    if out_dir is None:
        out_dir = stack_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = ROOT / "auto_pipeline" / "logs" / "regions_v5.jsonl"
    log_jsonl(log_file, {"stage": "regions_v5", "what": "start",
                         "result": f"stack={stack_path.name} mask={mask_path.name} v={__version__}"})

    # Load the mask JSON sidecar (if auto_mask produced it) for soma hints
    mask_meta = None
    mask_json_candidates = [
        mask_path.parent / mask_path.name.replace("_autoseg_labelmap_reviewed.tif", "_autoseg_reviewed.json"),
        mask_path.parent / mask_path.name.replace("_labelmap_reviewed.tif", "_reviewed.json"),
    ]
    for mjp in mask_json_candidates:
        if mjp.exists():
            try:
                with open(mjp) as f:
                    mjdata = json.load(f)
                reviews = mjdata.get("reviews", [])
                if reviews:
                    mask_meta = reviews[-1].get("params", {})
                    print(f"[auto_regions] loaded mask_meta from {mjp.name}: soma_detected={mask_meta.get('soma_detected')}")
                break
            except Exception:
                pass

    seg, meta = pick_regions(stack_path, mask, voxel, exclude=exclude,
                             deep_end_first=deep_end_first, mask_meta=mask_meta)

    seg_path = out_dir / f"{stem}_segments_final.tif"
    json_path = out_dir / f"{stem}_segments_final.json"
    qc_path = out_dir / f"{stem}_autoregions_qc.png"

    for p in (seg_path, json_path, qc_path):
        if p.exists():
            old_dir = out_dir / "old"; old_dir.mkdir(exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            shutil.move(str(p), str(old_dir / f"{p.stem}.{stamp}{p.suffix}"))

    tifffile.imwrite(str(seg_path), seg.astype(np.uint8))
    sidecar = build_sidecar(seg, voxel, meta, stack_path, mask_path)
    json_path.write_text(json.dumps(sidecar, indent=2))
    make_qc_figure(seg, mask.astype(bool) if not isinstance(mask, np.ndarray) else mask,
                   voxel, meta, stack_path, qc_path)

    log_jsonl(log_file, {"stage": "regions_v5", "what": "done",
                         "result": f"seg={seg_path.name} regions={[rm['name'] for rm in meta['regions']]} v={__version__}"})

    print(f"\nOutputs:")
    print(f"  segments: {seg_path}")
    print(f"  sidecar:  {json_path}")
    print(f"  QC:       {qc_path}")

    return seg, sidecar, {"seg": seg_path, "json": json_path, "qc": qc_path}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", help="path to runNN_clean.tif (4D, T,Z,Y,X)")
    ap.add_argument("--mask", required=True, help="path to the cell mask TIFF")
    ap.add_argument("--exclude", default=None, help="path to the exclude labelmap")
    ap.add_argument("--out-dir", required=True, help="output directory")
    add_voxel_arg(ap)
    ap.add_argument("--deep-end-first", type=int, default=1,
                    help="1 if column 0 is the deep (soma) end (default 1)")
    args = ap.parse_args(argv)

    run_auto_regions(args.stack, args.mask, exclude_path=args.exclude,
                     out_dir=args.out_dir, voxel=args.voxel,
                     deep_end_first=bool(args.deep_end_first))


if __name__ == "__main__":
    main()
