#!/usr/bin/env python
"""auto_regions.py — headless automatic region picking from a cell mask.

Given a cleaned 4D stack and a binary or labeled cell mask, this script:
  1. Skeletonizes the mask, builds the tree (reuses wrap_segments_napari math).
  2. Identifies the soma as the thick blob (if present).
  3. Orients the dendrite: column 0 (deep end) is the soma/proximal side.
  4. Computes geodesic distance from the proximal tip through the mask.
  5. Detects bifurcations via skeleton topology.
  6. Places 3-6 regions as FULL CROSS-SECTION slices of the mask, spaced
     along the dendrite path:
       - reference (soma blob or proximal ~10 um) at the proximal end
       - 1-2 trunk regions spread evenly along the main path before the
         bifurcation (or along the full path if no bifurcation)
       - bifurcation region (~5 um) centered on the first real fork
       - 1-2 branch regions beyond the bifurcation, with the last touching
         the distal tip of the scanned path
       - side-branch region (branch2) if a side path is detected
     Each region spans the full mask cross-section over its distance window.
     Within each placement zone the exact window is tuned by split-half
     reliability (never by correlation with the reference — no circularity).
  7. Names: soma, trunk1, trunk2, bifurcation, branch1, branch1far, branch2 ...
     compatible with compartment_of in run_metrics.py.  A region is "branch"
     only if it lies beyond a detected bifurcation; otherwise it is named
     "trunk" and the flag "no_bifurcation_detected" is set.

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

__version__ = "0.4.0"

CC26 = np.ones((3, 3, 3), np.uint8)

# ---- Parameters (same for every cell — no per-run tuning) ----
MIN_ARC_VOX = 3              # drop skeleton arcs shorter than this
SOMA_FACTOR = 2.0            # radius factor for soma blob detection
MIN_ISLAND_VOX = 20          # drop small islands at save
SPUR_THRESHOLD_UM = 5.0      # skeleton branches shorter than this (um) are spurs
BRANCH_MIN_TERRITORY = 30    # territory voxels: branches smaller than this are spurs
SOMA_MIN_MAX_EDT_UM = 3.2    # soma blob must have max EDT >= this
SOMA_MIN_VOXELS = 400        # soma blob must have at least this many voxels

# Region placement — matched to Daria's pattern
REGION_LENGTH_UM = 25.0       # preferred path span per region (um)
REGION_MIN_LENGTH_UM = 8.0    # smallest acceptable region span
BIF_REGION_LENGTH_UM = 6.0    # path span for the bifurcation region
WIDTH_BIN_UM = 25.0           # path bin for the width profile (>= one scan chunk)
WIDTH_SKIP_FRAC = 0.15        # ignore the proximal 15 % (soma taper, dim deep end)
WIDTH_DROP_RATIO = 0.70       # sustained width after/before <= this -> trunk->branch
MIN_REGION_VOXELS = 30        # minimum total voxels for a region
SIDE_BRANCH_MIN_TERRITORY = 40  # side-branch territory must be >= this
PROXIMAL_REF_LENGTH_UM = 12.0   # path span for proximal reference (no soma)

# How many trunk/branch regions:  determined by the available path length.
# Roughly one trunk region per 80-100 um of pre-bifurcation path,
# and one branch region per 80-100 um of post-bifurcation path.
TRUNK_REGION_SPACING_UM = 100.0
BRANCH_REGION_SPACING_UM = 100.0


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
# Tree analysis
# ==============================================================================
def build_skeleton_graph(mask, voxel, exclude=None, min_arc_vox=MIN_ARC_VOX):
    """Build skeleton, arcs, partition, adjacency, and the soma blob."""
    mask = mask.astype(bool)
    if exclude is not None:
        mask = mask & ~exclude.astype(bool)
    if not mask.any():
        raise ValueError("mask is empty after excluding intruders")

    skel = skeletonize(mask)
    if not skel.any():
        raise ValueError("skeleton is empty")

    arc_labels, n_arcs = partition_skeleton_into_arcs(skel, min_arc_vox)
    partition, geo_cost = geodesic_arc_partition(mask, arc_labels, voxel)
    edt = distance_transform_edt(mask, sampling=tuple(voxel))
    radii = arc_radii(edt, arc_labels)
    median_arc_radius = float(np.median(list(radii.values()))) if radii else 0.0
    soma_blob = soma_blob_from_thickness(mask, edt, median_arc_radius, voxel)

    bp_mask = skeleton_branch_points(skel)
    bp_lbl, n_bp = cc_label(bp_mask, structure=CC26)

    junctions = {}
    for b in range(1, n_bp + 1):
        bpc = bp_lbl == b
        dilated = binary_dilation(bpc, structure=CC26)
        touching = set(int(v) for v in np.unique(arc_labels[dilated]) if v > 0)
        junctions[b] = touching

    arc_adj = {a: set() for a in range(1, n_arcs + 1)}
    for j_id, arcs in junctions.items():
        for a in arcs:
            for other in arcs:
                if other != a:
                    arc_adj[a].add(other)

    arc_lengths = {}
    for a in range(1, n_arcs + 1):
        pts = np.argwhere(arc_labels == a)
        if len(pts) < 2:
            arc_lengths[a] = 0.0
            continue
        order = np.argsort(pts[:, 2])
        ordered = pts[order] * np.array(voxel)
        arc_lengths[a] = float(np.sum(np.linalg.norm(np.diff(ordered, axis=0), axis=1)))

    return {
        "mask": mask,
        "skel": skel,
        "arc_labels": arc_labels,
        "n_arcs": n_arcs,
        "partition": partition,
        "geo_cost": geo_cost,
        "edt": edt,
        "radii": radii,
        "arc_lengths": arc_lengths,
        "median_arc_radius": median_arc_radius,
        "soma_blob": soma_blob,
        "junctions": junctions,
        "arc_adj": arc_adj,
        "bp_labels": bp_lbl,
    }


def find_root_arc(graph, deep_end_first=True):
    """Find the root arc (soma side). Returns (arc_id, has_soma)."""
    soma_blob = graph["soma_blob"]
    edt = graph["edt"]
    n_arcs = graph["n_arcs"]

    real_soma = False
    if soma_blob.any():
        max_edt = float(edt[soma_blob].max())
        n_blob = int(soma_blob.sum())
        if max_edt >= SOMA_MIN_MAX_EDT_UM and n_blob >= SOMA_MIN_VOXELS:
            real_soma = True

    if real_soma:
        overlap = {}
        for a in range(1, n_arcs + 1):
            territory = graph["partition"] == a
            ov = (territory & soma_blob).sum()
            if ov > 0:
                overlap[a] = ov
        if overlap:
            return max(overlap, key=overlap.get), True

    mean_x = {}
    for a in range(1, n_arcs + 1):
        pts = np.argwhere(graph["partition"] == a)
        if len(pts) > 0:
            mean_x[a] = pts[:, 2].mean()

    if deep_end_first:
        return min(mean_x, key=mean_x.get), False
    else:
        return max(mean_x, key=mean_x.get), False


def walk_main_path(graph, root_arc, voxel):
    """Walk from root along the thickest/longest continuation, recording side branches."""
    visited = {root_arc}
    main_path = [root_arc]
    side_branches = []
    bifurcation_junctions = []

    current = root_arc
    while True:
        neighbors = graph["arc_adj"][current] - visited
        if not neighbors:
            break

        significant = []
        for n in neighbors:
            length = graph["arc_lengths"].get(n, 0)
            territory = int((graph["partition"] == n).sum())
            further = graph["arc_adj"][n] - visited - {n}
            if length < SPUR_THRESHOLD_UM and territory < BRANCH_MIN_TERRITORY and not further:
                visited.add(n)
                continue
            significant.append(n)

        if not significant:
            break

        if len(significant) == 1:
            current = significant[0]
            visited.add(current)
            main_path.append(current)
        else:
            for j_id, j_arcs in graph["junctions"].items():
                if current in j_arcs and any(n in j_arcs for n in significant):
                    bifurcation_junctions.append(j_id)
                    break

            def score(a):
                r = graph["radii"].get(a, 0)
                t = int((graph["partition"] == a).sum())
                l = graph["arc_lengths"].get(a, 0)
                return r * t + l
            scores = {a: score(a) for a in significant}
            continuation = max(scores, key=scores.get)
            for n in significant:
                if n != continuation:
                    side_branches.append(n)
                visited.add(n)
            current = continuation
            main_path.append(current)

    expanded_branches = []
    for sb in side_branches:
        branch_arcs = [sb]
        frontier = [sb]
        while frontier:
            a = frontier.pop()
            for n in graph["arc_adj"][a] - visited:
                visited.add(n)
                branch_arcs.append(n)
                frontier.append(n)
        expanded_branches.append(branch_arcs)

    return main_path, expanded_branches, bifurcation_junctions


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
def slice_by_distance(mask, dist_from_root, d_lo, d_hi):
    """Boolean mask of all mask voxels with geodesic distance in [d_lo, d_hi)."""
    return mask & np.isfinite(dist_from_root) & (dist_from_root >= d_lo) & (dist_from_root < d_hi)


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

    # If zone is already close to target, use the whole zone
    if zone_length <= target_length_um * 1.3:
        w = slice_by_distance(mask, dist_from_root, zone_lo, zone_hi)
        n = int(w.sum())
        if n < min_voxels:
            return None
        idx = np.flatnonzero(w.ravel())
        rel = split_half_reliability(flat, idx)
        return (zone_lo, zone_hi, rel, n)

    # Search for best sub-window
    best = None
    best_rel = -1.0

    # Try the target length first, then variations
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
def pick_regions(stack_path, mask, voxel, exclude=None, deep_end_first=True):
    """Automatic region picking. Returns (segments_vol, metadata_dict)."""
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

    # Build skeleton graph
    log("Building skeleton graph...")
    graph = build_skeleton_graph(mask, voxel, exclude=exclude, min_arc_vox=MIN_ARC_VOX)
    log(f"  skeleton: {graph['skel'].sum()} vox, {graph['n_arcs']} arcs, "
        f"median_radius={graph['median_arc_radius']:.2f} um")
    log(f"  soma blob: {graph['soma_blob'].sum()} vox")

    for a in range(1, graph['n_arcs'] + 1):
        terr = (graph['partition'] == a).sum()
        if terr > 0:
            log(f"    arc {a}: skel={(graph['arc_labels']==a).sum()} "
                f"terr={terr} radius={graph['radii'].get(a,0):.2f} "
                f"length={graph['arc_lengths'].get(a,0):.1f}um "
                f"mean_x={np.argwhere(graph['partition']==a)[:,2].mean():.0f}")

    root_arc, has_soma = find_root_arc(graph, deep_end_first)
    main_path, branches, bif_junctions = walk_main_path(graph, root_arc, voxel)
    log(f"  root: arc {root_arc}, has_soma={has_soma}")
    log(f"  main path: {main_path}")
    log(f"  side branches: {branches}")

    # Check soma validity
    edt = graph["edt"]
    soma_blob = graph["soma_blob"]
    real_soma = False
    if has_soma and soma_blob.any():
        max_edt = float(edt[soma_blob].max())
        n_blob = int(soma_blob.sum())
        if max_edt >= SOMA_MIN_MAX_EDT_UM and n_blob >= SOMA_MIN_VOXELS:
            real_soma = True
        else:
            log(f"  soma blob rejected: max_edt={max_edt:.2f}, voxels={n_blob}")
            has_soma = False

    # Geodesic distance from proximal tip
    if has_soma and soma_blob.any():
        root_voxels = np.argwhere(soma_blob)
    else:
        root_skel_pts = np.argwhere(graph["arc_labels"] == root_arc)
        if len(root_skel_pts) == 0:
            root_skel_pts = np.argwhere(graph["partition"] == root_arc)
        if deep_end_first:
            tip_idx = root_skel_pts[:, 2].argmin()
        else:
            tip_idx = root_skel_pts[:, 2].argmax()
        tip = root_skel_pts[tip_idx]
        dists_to_tip = np.linalg.norm((root_skel_pts - tip) * np.array(voxel), axis=1)
        root_voxels = root_skel_pts[dists_to_tip < 3.0]
        if len(root_voxels) == 0:
            root_voxels = tip.reshape(1, 3)

    dist_from_root = geodesic_distance_from_seeds(mask, root_voxels, voxel)

    # Build main-path territory and side-branch territories
    main_territory = np.zeros(mask.shape, bool)
    for a in main_path:
        main_territory |= (graph["partition"] == a)
    if exclude is not None:
        main_territory &= ~exclude.astype(bool)

    side_branch_territories = []
    for br_arcs in branches:
        br_terr = np.zeros(mask.shape, bool)
        for a in br_arcs:
            br_terr |= (graph["partition"] == a)
        if has_soma and soma_blob.any():
            br_terr &= ~soma_blob
        if exclude is not None:
            br_terr &= ~exclude.astype(bool)

        # Find the branch skeleton tip (farthest from root)
        br_skel_pts = []
        for a in br_arcs:
            br_skel_pts.extend(np.argwhere(graph["arc_labels"] == a).tolist())
        if not br_skel_pts:
            if br_terr.sum() >= SIDE_BRANCH_MIN_TERRITORY:
                side_branch_territories.append(br_terr)
            continue
        br_skel_pts = np.array(br_skel_pts)
        # The tip is the skeleton point farthest from the root
        tip_dists = np.array([dist_from_root[tuple(p)] for p in br_skel_pts])
        tip_dists[~np.isfinite(tip_dists)] = -1
        tip = br_skel_pts[tip_dists.argmax()]

        # Grow from the tip through the branch territory only
        # This gives us the "distinctly branch" voxels, excluding junction voxels
        # that are close to both the main path and the branch
        costs_br = np.where(br_terr, 1.0, np.inf).astype(float)
        from skimage.graph import MCP_Geometric
        mcp = MCP_Geometric(costs_br, sampling=tuple(voxel))
        dist_from_tip, _ = mcp.find_costs([tuple(tip)])
        # Also compute distance from the main path through the branch territory
        mp_boundary = binary_dilation(main_territory, structure=CC26) & br_terr & ~main_territory
        if mp_boundary.any():
            mp_boundary_pts = np.argwhere(mp_boundary)
            mcp2 = MCP_Geometric(costs_br, sampling=tuple(voxel))
            dist_from_mp, _ = mcp2.find_costs([tuple(p) for p in mp_boundary_pts])
            # Keep voxels closer to the branch tip than to the main path boundary
            refined = br_terr & np.isfinite(dist_from_tip) & (dist_from_tip < dist_from_mp * 1.2)
        else:
            refined = br_terr

        if refined.sum() >= SIDE_BRANCH_MIN_TERRITORY:
            side_branch_territories.append(refined)
            log(f"  side branch: {br_terr.sum()} -> {refined.sum()} vox (tip-based restriction)")
        elif br_terr.sum() >= SIDE_BRANCH_MIN_TERRITORY:
            side_branch_territories.append(br_terr)
            log(f"  side branch: {br_terr.sum()} vox (tip restriction dropped too many)")
        else:
            log(f"  dropping tiny side branch: {br_terr.sum()} vox")

    # Main-path distance range (excluding soma)
    main_usable = main_territory.copy()
    if has_soma and soma_blob.any():
        main_usable &= ~soma_blob

    main_dists = dist_from_root[main_usable]
    main_dists = main_dists[np.isfinite(main_dists)]
    if len(main_dists) == 0:
        raise ValueError("no usable main path voxels")

    mp_min = float(main_dists.min())
    mp_max = float(main_dists.max())
    mp_length = mp_max - mp_min
    log(f"  main path (excl soma): dist {mp_min:.1f} - {mp_max:.1f} um, length={mp_length:.1f} um")

    # Bifurcation distance
    bifurcation_dist = None
    real_bifurcation = False
    if bif_junctions:
        bif_dists = []
        for j_id in bif_junctions:
            bp_pts = np.argwhere(graph["bp_labels"] == j_id)
            for p in bp_pts:
                d = dist_from_root[tuple(p)]
                if np.isfinite(d):
                    bif_dists.append(d)
        if bif_dists:
            bifurcation_dist = float(np.min(bif_dists))

    # Only count as a real bifurcation if it has a substantial side branch
    if bifurcation_dist is not None and side_branch_territories:
        real_bifurcation = True
        log(f"  bifurcation at distance {bifurcation_dist:.1f} um from root")
    elif bifurcation_dist is not None:
        log(f"  bifurcation at {bifurcation_dist:.1f} um but no substantial side branches — treating as single path")
        bifurcation_dist = None

    # Second, weaker evidence for the trunk -> branch transition when the skeleton has
    # no fork inside the tube (the other daughter may leave the scanned volume): a
    # SUSTAINED step down in dendrite width along the path. Width = mask voxels per
    # WIDTH_BIN_UM of path (>= one scan chunk, so chunk-edge artifacts average out).
    # Never decided from activity. Daria's run03: ~14 -> ~8.5 vox/column after her
    # bifurcation; run05 shows no step, so it stays trunk (her 'branch2' there comes
    # from anatomy outside the tube; she can rename it in the region tool).
    width_drop_dist = None
    width_profile = []
    if not real_bifurcation:
        d_all = dist_from_root[main_usable]
        ok = np.isfinite(d_all)
        d_all = d_all[ok]
        edges = np.append(np.arange(mp_min, mp_max, WIDTH_BIN_UM), mp_max)
        span = np.diff(edges)
        counts = np.histogram(d_all, bins=edges)[0].astype(float) / np.maximum(span, 1e-9) * WIDTH_BIN_UM
        if len(counts) > 1 and span[-1] < 0.5 * WIDTH_BIN_UM:   # too short to judge width
            counts, edges = counts[:-1], edges[:-1]
        width_profile = [round(float(c), 1) for c in counts]
        n_b = len(counts)
        lo_i = int(np.ceil(WIDTH_SKIP_FRAC * n_b))           # skip soma taper / deep end
        best = None
        for i in range(max(lo_i, 2), n_b - 2):              # >= 2 bins on each side
            before, after = counts[lo_i:i], counts[i:]
            if len(before) < 2 or len(after) < 2 or np.median(before) <= 0:
                continue
            ratio = float(np.median(after) / np.median(before))
            step = float(np.median(counts[max(i - 2, 0):i]))
            local = float(np.median(counts[i:i + 2])) / step if step > 0 else 1.0
            if ratio <= WIDTH_DROP_RATIO and local <= WIDTH_DROP_RATIO + 0.1:
                if best is None or ratio < best[1]:
                    best = (i, ratio)
        if best is not None:
            width_drop_dist = float(edges[best[0]])
            log(f"  width drop to {best[1]:.2f}x at {width_drop_dist:.1f} um from root "
                f"(per-{WIDTH_BIN_UM:.0f}-um widths {width_profile})")
        else:
            log(f"  no sustained width drop (per-{WIDTH_BIN_UM:.0f}-um widths {width_profile})")

    # ====================================================================
    # REGION PLACEMENT
    #
    # Daria's pattern (from ground truth):
    #   - reference (soma or proximal trunk) at the very start
    #   - 1-2 intermediate trunk regions at roughly 25% and 55% of the path
    #   - the last region at the far distal tip (~93% of path)
    #   - if a bifurcation exists, a small bif region there and "branch"
    #     names for everything beyond it
    #   - the distal end is always "branch" (even without a skeleton fork):
    #     in L5 apicals the far end is branches by anatomy
    #   - regions do NOT tile the path — there are unlabeled gaps between
    #
    # Implementation: place region centers at fixed fractions of the total
    # main-path distance, then optimize the exact sub-window by split-half
    # reliability.
    # ====================================================================
    regions = []  # (name, mask, role, reliability)

    # ---- 1. SOMA ----
    if has_soma and soma_blob.any():
        n_soma = int(soma_blob.sum())
        if n_soma >= MIN_REGION_VOXELS:
            idx = np.flatnonzero(soma_blob.ravel())
            rel = split_half_reliability(flat, idx)
            regions.append(("soma", soma_blob.copy(), "soma", rel))
            log(f"  REGION soma: {n_soma} vox, rel={rel:.4f}")

    # ---- 2. PROXIMAL REFERENCE (if no soma) ----
    if not has_soma:
        ref_end = min(mp_min + PROXIMAL_REF_LENGTH_UM, mp_min + mp_length * 0.06)
        ref_end = max(ref_end, mp_min + REGION_MIN_LENGTH_UM)
        w = best_window_in_zone(flat, main_territory, dist_from_root,
                                mp_min, ref_end, voxel,
                                target_length_um=PROXIMAL_REF_LENGTH_UM,
                                min_length_um=REGION_MIN_LENGTH_UM)
        if w is not None:
            d_lo, d_hi, rel, n = w
            rmask = slice_by_distance(main_territory, dist_from_root, d_lo, d_hi)
            regions.append(("trunk1", rmask, "trunk", rel))
            log(f"  REGION trunk1 (reference): {n} vox, dist=[{d_lo:.1f},{d_hi:.1f}], rel={rel:.4f}")
        else:
            rmask = slice_by_distance(main_territory, dist_from_root,
                                       mp_min, mp_min + PROXIMAL_REF_LENGTH_UM)
            n = int(rmask.sum())
            if n >= MIN_REGION_VOXELS:
                idx = np.flatnonzero(rmask.ravel())
                rel = split_half_reliability(flat, idx)
                regions.append(("trunk1", rmask, "trunk", rel))
                log(f"  REGION trunk1 (reference, fallback): {n} vox, rel={rel:.4f}")

    # ---- 3. BIFURCATION region ----
    bif_region_placed = False
    if real_bifurcation:
        bif_half = BIF_REGION_LENGTH_UM / 2
        bif_lo = max(mp_min, bifurcation_dist - bif_half)
        bif_hi = min(mp_max, bifurcation_dist + bif_half)
        bif_mask = slice_by_distance(main_territory, dist_from_root, bif_lo, bif_hi)
        n_bif = int(bif_mask.sum())
        if n_bif >= MIN_REGION_VOXELS:
            idx = np.flatnonzero(bif_mask.ravel())
            rel = split_half_reliability(flat, idx)
            regions.append(("bifurcation", bif_mask, "branch", rel))
            bif_region_placed = True
            log(f"  REGION bifurcation: {n_bif} vox, dist=[{bif_lo:.1f},{bif_hi:.1f}], rel={rel:.4f}")

    # ---- 4. MAIN-PATH intermediate and distal regions ----
    # Determine the trunk/branch boundary
    if real_bifurcation:
        trunk_boundary = bifurcation_dist
    elif width_drop_dist is not None:
        trunk_boundary = width_drop_dist                  # weaker evidence; flagged below
    else:
        # No evidence of a transition inside the scan: everything on the main path is
        # trunk (v0.3 called the distal 30 % 'branch' by position, which mislabeled
        # trunk as branch in cells without a fork).
        trunk_boundary = np.inf

    # How many intermediate regions to place?
    # Pattern: at ~25% and ~55% of path for trunk, ~65% and ~93% for branch
    # (with the last region anchored at the distal tip).
    # Decide count from path length.
    n_intermediate = max(2, min(4, round(mp_length / 100.0)))
    # Typical: 2 for short dendrites (<200 um), 3 for medium (200-350), 4 for long (>350)

    # Generate region center fractions, evenly spaced between ~20% and ~95%
    # (leaving room at proximal end for the reference)
    frac_start = 0.22
    frac_end = 0.93
    fracs = np.linspace(frac_start, frac_end, n_intermediate)

    trunk_name_start = 1 if has_soma else 2  # trunk1 is the no-soma reference
    trunk_count = 0
    branch_main_count = 0  # branch regions on the main path

    for fi, frac in enumerate(fracs):
        center_dist = mp_min + frac * mp_length
        half = REGION_LENGTH_UM / 2

        # Is this region before or after the trunk/branch boundary?
        is_branch = center_dist > trunk_boundary

        # Widen window for the last region (distal tip)
        if fi == len(fracs) - 1:
            # Anchor the last region at the distal tip
            z_hi = min(mp_max + 0.5, mp_max)
            z_lo = max(z_hi - REGION_LENGTH_UM * 1.8, mp_min)
            # Make it big (Daria's distal regions are large)
            target_len = REGION_LENGTH_UM * 1.8
        else:
            z_lo = max(mp_min, center_dist - half)
            z_hi = min(mp_max, center_dist + half)
            target_len = REGION_LENGTH_UM

        # Skip if we'd overlap with the bifurcation region
        if bif_region_placed and not is_branch:
            bif_d = bifurcation_dist
            if z_lo < bif_d + BIF_REGION_LENGTH_UM and z_hi > bif_d - BIF_REGION_LENGTH_UM:
                # This window is near the bifurcation — shift it or skip
                if center_dist < bif_d:
                    z_hi = min(z_hi, bif_d - BIF_REGION_LENGTH_UM)
                else:
                    z_lo = max(z_lo, bif_d + BIF_REGION_LENGTH_UM)

        # Assign name
        if is_branch:
            if branch_main_count == 0:
                name = "branch1"
            elif branch_main_count == 1:
                name = "branch1far"
            else:
                name = f"branch1far{branch_main_count}"
            role = "branch"
        else:
            idx_num = trunk_name_start + trunk_count
            name = f"trunk{idx_num}" if idx_num > 1 else "trunk"
            role = "trunk"

        # For the last region, use a wider search zone
        if fi == len(fracs) - 1:
            w = best_window_in_zone(flat, main_territory, dist_from_root,
                                     z_lo, z_hi, voxel,
                                     target_length_um=target_len,
                                     min_length_um=REGION_MIN_LENGTH_UM)
        else:
            w = best_window_in_zone(flat, main_territory, dist_from_root,
                                     z_lo, z_hi, voxel,
                                     target_length_um=target_len,
                                     min_length_um=REGION_MIN_LENGTH_UM)
        if w is not None:
            d_lo, d_hi, rel, n = w
            rmask = slice_by_distance(main_territory, dist_from_root, d_lo, d_hi)
            if has_soma and soma_blob.any():
                rmask &= ~soma_blob
            n = int(rmask.sum())
            if n >= MIN_REGION_VOXELS:
                regions.append((name, rmask, role, rel))
                log(f"  REGION {name}: {n} vox, dist=[{d_lo:.1f},{d_hi:.1f}], rel={rel:.4f}")
                if is_branch:
                    branch_main_count += 1
                else:
                    trunk_count += 1
                continue

        # Fallback
        rmask = slice_by_distance(main_territory, dist_from_root, z_lo, z_hi)
        if has_soma and soma_blob.any():
            rmask &= ~soma_blob
        n = int(rmask.sum())
        if n >= MIN_REGION_VOXELS:
            idx = np.flatnonzero(rmask.ravel())
            rel = split_half_reliability(flat, idx)
            regions.append((name, rmask, role, rel))
            log(f"  REGION {name}: {n} vox, dist=[{z_lo:.1f},{z_hi:.1f}], rel={rel:.4f} (fallback)")
            if is_branch:
                branch_main_count += 1
            else:
                trunk_count += 1

    # ---- 5. (v0.4) no renaming: with no fork, main-path regions are trunk, and
    # beyond a width drop they keep the branch1 / branch1far names. ----

    # ---- 6. SIDE-BRANCH regions (branch2, branch3, ...) ----
    # Determine next branch number
    existing_branch_nums = set()
    for rname, _, rrole, _ in regions:
        if rrole == "branch" and "branch" in rname:
            import re
            m = re.search(r"branch(\d+)", rname)
            if m:
                existing_branch_nums.add(int(m.group(1)))
    branch_num = max(existing_branch_nums, default=1) + 1
    # But skip numbers already used
    while branch_num in existing_branch_nums:
        branch_num += 1

    for sb_terr in side_branch_territories:
        n_sb = int(sb_terr.sum())
        if n_sb < MIN_REGION_VOXELS:
            continue
        idx = np.flatnonzero(sb_terr.ravel())
        rel = split_half_reliability(flat, idx)
        name = f"branch{branch_num}"
        regions.append((name, sb_terr, "branch", rel))
        sb_dists = dist_from_root[sb_terr]
        sb_dists = sb_dists[np.isfinite(sb_dists)]
        log(f"  REGION {name} (side): {n_sb} vox, dist={np.median(sb_dists):.1f}, rel={rel:.4f}")
        existing_branch_nums.add(branch_num)
        branch_num += 1

    if not regions:
        raise ValueError("no regions found")

    # ====================================================================
    # BUILD OUTPUT
    # ====================================================================
    seg = np.zeros(mask.shape, np.uint8)
    region_meta = []

    # Sort by geodesic distance
    def region_dist(r):
        name, rmask, role, rel = r
        d = dist_from_root[rmask]
        d = d[np.isfinite(d)]
        return float(np.median(d)) if len(d) > 0 else 0.0

    regions.sort(key=region_dist)

    label_num = 1
    for name, rmask, role, rel in regions:
        seg[rmask] = label_num
        d = dist_from_root[rmask]
        d = d[np.isfinite(d)]
        med_dist = float(np.median(d)) if len(d) > 0 else 0.0
        region_meta.append({
            "label": label_num,
            "name": name,
            "role": role,
            "voxels": int(rmask.sum()),
            "reliability": round(rel, 4) if rel == rel else 0.0,
            "distance_um": round(med_dist, 2),
        })
        label_num += 1

    # Clamp to mask
    seg[~mask.astype(bool)] = 0
    if exclude is not None:
        seg[exclude.astype(bool)] = 0

    # Drop small islands
    seg, rep = drop_small_islands(seg, min_voxels=MIN_ISLAND_VOX)
    if rep:
        log(f"  island cleanup: {describe(rep)}")

    # Determine reference
    ref_label = None
    for rm in region_meta:
        if rm["role"] == "soma":
            ref_label = rm["label"]
            break
    if ref_label is None:
        trunk_regions = [rm for rm in region_meta if rm["role"] == "trunk"]
        if trunk_regions:
            ref_label = min(trunk_regions, key=lambda rm: rm["distance_um"])["label"]

    # Recompute distances from reference
    if ref_label is not None:
        ref_voxels = np.argwhere(seg == ref_label)
        if len(ref_voxels) > 0:
            ref_dist = geodesic_distance_from_seeds(mask, ref_voxels, voxel)
            for rm in region_meta:
                d = ref_dist[seg == rm["label"]]
                d = d[np.isfinite(d)]
                rm["distance_um"] = round(float(np.median(d)), 2) if len(d) > 0 else 0.0

    region_meta.sort(key=lambda rm: rm["distance_um"])

    # Reason strings
    for rm in region_meta:
        reasons = []
        if rm["role"] == "soma":
            reasons.append("thick blob (EDT >= 2x median arc radius)")
        elif rm["name"] == "trunk1" and not has_soma:
            reasons.append("proximal reference (deep end)")
        elif "trunk" in rm["name"]:
            reasons.append("main-path trunk region")
        elif rm["name"] == "bifurcation":
            reasons.append("skeleton bifurcation point")
        elif "branch" in rm["name"] and rm["name"] not in ("branch1", "branch1far") and not rm["name"].startswith("branch1far"):
            reasons.append("side branch leaving main path")
        elif real_bifurcation:
            reasons.append("main-path continuation beyond bifurcation")
        else:
            reasons.append(f"main path beyond a sustained width drop at {width_drop_dist:.0f} um (no fork in the scan)")
        reasons.append(f"full cross-section, {rm['voxels']} vox, rel={rm['reliability']:.4f}")
        rm["reason"] = "; ".join(reasons)

    reference_type = "soma" if any(rm["role"] == "soma" for rm in region_meta) else "proximal_trunk"
    reference_name = None
    for rm in region_meta:
        if rm["label"] == ref_label:
            reference_name = rm["name"]
            break

    flags = []
    if not real_bifurcation:
        flags.append("no_bifurcation_detected")
        if width_drop_dist is not None:
            flags.append(f"branch_from_width_drop_at_{width_drop_dist:.0f}um")
        else:
            flags.append("all_main_path_regions_trunk")

    return seg, {
        "regions": region_meta,
        "reference": reference_type,
        "reference_region": reference_name,
        "reference_label": ref_label,
        "trunk_branch_boundary": ("skeleton_fork" if real_bifurcation else
                                  "width_drop" if width_drop_dist is not None else "none"),
        "width_drop_um": width_drop_dist,
        "width_profile_vox_per_bin": width_profile,
        "width_bin_um": WIDTH_BIN_UM,
        "flags": flags,
        "tree_info": {
            "n_arcs": graph["n_arcs"],
            "n_skel_voxels": int(graph["skel"].sum()),
            "main_path_arcs": main_path,
            "branch_arc_groups": [[int(a) for a in br] for br in branches],
            "bifurcation_junctions": [int(j) for j in bif_junctions],
            "bifurcation_dist_um": round(bifurcation_dist, 2) if bifurcation_dist is not None else None,
            "has_soma": has_soma,
            "soma_blob_voxels": int(soma_blob.sum()) if soma_blob is not None else 0,
            "median_arc_radius_um": graph["median_arc_radius"],
            "deep_end_first": deep_end_first,
            "total_path_length_um": round(mp_length + (float(dist_from_root[soma_blob].max()) if has_soma and soma_blob.any() else 0), 2),
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
                "trunk_spacing_um": TRUNK_REGION_SPACING_UM,
                "branch_spacing_um": BRANCH_REGION_SPACING_UM,
                "bif_region_length_um": BIF_REGION_LENGTH_UM,
                "proximal_ref_length_um": PROXIMAL_REF_LENGTH_UM,
                "soma_factor": SOMA_FACTOR,
                "spur_threshold_um": SPUR_THRESHOLD_UM,
                "branch_min_territory": BRANCH_MIN_TERRITORY,
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
                    "reason": rm.get("reason", ""),
                }
                for rm in meta["regions"]
            },
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
    headers = ["Region", "Voxels", "Dist(um)", "Rel", "Role"]
    table_data = [[rm["name"], str(rm["voxels"]),
                   f"{rm.get('distance_um', 0):.1f}",
                   f"{rm['reliability']:.4f}", rm["role"]]
                  for rm in meta["regions"]]
    if table_data:
        table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.3)
    ax.set_title(f"auto_regions v{__version__}: {Path(stack_path).stem}", fontsize=9)

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

    log_file = ROOT / "auto_pipeline" / "logs" / "regioner.jsonl"
    log_jsonl(log_file, {"stage": "regioner", "what": "start",
                         "result": f"stack={stack_path.name} mask={mask_path.name} v={__version__}"})

    seg, meta = pick_regions(stack_path, mask, voxel, exclude=exclude,
                             deep_end_first=deep_end_first)

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

    log_jsonl(log_file, {"stage": "regioner", "what": "done",
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
