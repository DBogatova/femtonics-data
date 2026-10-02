#!/usr/bin/env python
"""
auto_segment.py - AUTOMATIC 3D neuron segmentation for Femtonics 2P volumetric
calcium imaging (thin-slab 4D stacks, GCaMP in Rbp4 pyramidal neurons).

Goal: replace the ~1 h/run of hand tracing (guideline_mask_napari.py) + hand
labeling (segment_mask_napari.py) with a zero-click first pass whose output the
human only has to *review* (~2 min accept/correct in review_autoseg_napari.py).

Pipeline (validated on real data, then reproduced here):

  1. AUTO-MASK from three per-voxel temporal statistics, computed in a single
     memory-bounded streaming pass over time:
        corr = max over the 4 x/y neighbor shifts of temporal Pearson
               correlation   (activity finds thin structure single frames miss)
        amax = temporal 99.5th percentile
        mean = temporal mean
     Each map is gaussian_filter(sigma=0.8)-smoothed, robust-z scored
        z(v) = (v - median(v)) / (p84(v) - median(v)),
     and score = elementwise max of the three z-maps. Threshold (~0.8-1.0),
     binary_closing(3^3), drop connected components < --min-voxels (60).

  2. CELL SPLIT by NMF on the masked voxels' time series. Per voxel:
     F0 = 10th percentile, dF/F, globally shifted non-negative. sklearn
     NMF(init='nndsvda', n_components=K). Assign each voxel to argmax loading;
     merge components whose TEMPORAL factors correlate r > --merge-r (0.8) - they
     are the same cell. Each merged group is one candidate CELL. (Distinct
     functional subtrees of ONE neuron correlate ~0.3, so a single neuron with
     weakly-coupled subtrees can be split into >1 candidate cell; the review
     step merges. This is reported, not hidden.)

  3. PER-CELL ANATOMY by skeleton graph. skeletonize(cell mask); build the
     26-neighbor skeleton graph (skan if importable, else neighbor-count).
     SOMA = skeleton region at the distance-transform peak (thickest point);
     TRUNK = longest/widest geodesic path from the soma; BRANCHES = the
     remaining skeleton sub-trees, largest first. Dense labels are grown to
     every cell voxel by nearest labeled-skeleton assignment.
     At this SNR / slab thickness the anatomy split is a heuristic; each cell
     carries an `anatomy_provisional` flag and the reasons behind it.

Output contract (the review tool and the auditor build against THIS). For input
<dir>/<stem>.tif:
  <dir>/<stem>_autoseg_labelmap.tif  uint8 (Z,Y,X); 0=bg; per cell i (1-based)
                                     label = local + (i-1)*10, local 1=soma
                                     2=trunk 3+=branches  (cell1 1-9, cell2 11-19)
  <dir>/<stem>_autoseg.json          cells[], params{}, stats{}
  <dir>/<stem>_autoseg_preview.png   anatomy MIP + mask + labels

Never overwrites hand-made labelmaps: every output carries the _autoseg suffix.

CLI examples
------------
  python auto_segment.py run7_clean.tif
  python auto_segment.py a.tif b.tif --thr 0.9 --k 6 --min-voxels 60
  python auto_segment.py --glob 'preprocessed/run*/run*_clean.tif'
  python auto_segment.py run7_clean.tif --report-only     # print plan, no writes
"""
import argparse
from pathlib import Path
import glob as globmod
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from common.voxel import add_voxel_arg, resolve_voxel
import sys as _psys, pathlib as _ppl
_psys.path.insert(0, str(_ppl.Path(__file__).resolve().parents[1]))
from common.progress import progress

__version__ = "1.0.0"

# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------


def _log(msg):
    print(msg, flush=True)


def _peak_mem_mb():
    """Peak RSS of this process in MB (macOS reports bytes, Linux KiB)."""
    try:
        import resource
        m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return m / (1024.0 * 1024.0)      # bytes -> MB
        return m / 1024.0                      # KiB -> MB
    except Exception:
        return float("nan")


def _shift_zero(a, dy, dx):
    """Return b with b[...,y,x] = a[...,y+dy,x+dx], zero-filled at invalid borders.

    Works for arrays whose last two axes are (Y, X); leading axes (T and/or Z)
    are preserved. Zero-fill (rather than np.roll's wrap) keeps border voxels
    from picking up a spurious high correlation with the opposite edge.
    """
    Y, X = a.shape[-2], a.shape[-1]
    out = np.zeros_like(a)
    y0d, y1d = max(0, -dy), Y - max(0, dy)
    x0d, x1d = max(0, -dx), X - max(0, dx)
    y0s, y1s = max(0, dy), Y - max(0, -dy)
    x0s, x1s = max(0, dx), X - max(0, -dx)
    if y1d > y0d and x1d > x0d:
        out[..., y0d:y1d, x0d:x1d] = a[..., y0s:y1s, x0s:x1s]
    return out


def robust_z(v):
    """(v - median) / (p84 - median): robust z where p84-median ~= 1 sigma."""
    med = float(np.median(v))
    p84 = float(np.percentile(v, 84))
    scale = p84 - med
    if scale <= 1e-9:
        scale = float(np.std(v)) or 1.0
    return (v - med) / scale


# ----------------------------------------------------------------------------
# step 1: activity statistics + mask
# ----------------------------------------------------------------------------


def activity_statistics(stack, tchunk=200, space_chunk=20000):
    """Streaming per-voxel temporal statistics on a (T,Z,Y,X) stack.

    Returns (corr, amax, mean) each (Z,Y,X) float32:
      corr = max over the 4 x/y neighbor shifts of temporal Pearson correlation
      amax = temporal 99.5th percentile
      mean = temporal mean
    Memory-bounded: one pass over time in chunks for mean/std/cross-products, one
    pass over space for the percentile. Peak temporaries ~ tchunk*Z*Y*X*4 bytes.
    """
    T, Z, Y, X = stack.shape
    s1 = np.zeros((Z, Y, X), np.float64)     # sum_t v
    s2 = np.zeros((Z, Y, X), np.float64)     # sum_t v^2
    shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    cross = {s: np.zeros((Z, Y, X), np.float64) for s in shifts}

    for t0 in range(0, T, tchunk):
        progress(t0, T, "auto-segment: activity statistics")
        c = stack[t0:t0 + tchunk].astype(np.float32)
        s1 += c.sum(0)
        s2 += np.square(c).sum(0)
        for sh in shifts:
            nb = _shift_zero(c, sh[0], sh[1])
            cross[sh] += (c * nb).sum(0)
        del c

    mean = (s1 / T).astype(np.float32)
    var = np.maximum(s2 / T - (s1 / T) ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)

    corr = np.zeros((Z, Y, X), np.float32)
    for sh in shifts:
        nbm = _shift_zero(mean, sh[0], sh[1])
        nbs = _shift_zero(std, sh[0], sh[1])
        denom = std * nbs
        c = np.zeros((Z, Y, X), np.float32)
        good = denom > 1e-9
        c[good] = ((cross[sh][good] / T) - mean[good] * nbm[good]) / (denom[good] + 1e-12)
        corr = np.maximum(corr, c)
    corr = np.clip(corr, 0.0, None)

    # temporal 99.5th percentile, chunked over space to bound memory
    flat = stack.reshape(T, -1)
    N = flat.shape[1]
    amax = np.empty(N, np.float32)
    for c0 in range(0, N, space_chunk):
        block = flat[:, c0:c0 + space_chunk].astype(np.float32)
        amax[c0:c0 + block.shape[1]] = np.percentile(block, 99.5, axis=0)
    amax = amax.reshape(Z, Y, X)

    return corr, amax, mean


def compute_score(corr, amax, mean, sigma=0.8):
    """gaussian smooth each stat, robust-z, take the elementwise max."""
    from scipy.ndimage import gaussian_filter
    maps = []
    for v in (corr, amax, mean):
        vs = gaussian_filter(v.astype(np.float32), sigma)
        maps.append(robust_z(vs))
    return np.maximum.reduce(maps).astype(np.float32)


def build_mask(score, thr, min_voxels, close_iter=1):
    """Threshold, 3x3x3 closing, drop small connected components."""
    from scipy.ndimage import binary_closing, generate_binary_structure, label as cc_label
    mask = score >= thr
    if close_iter > 0:
        mask = binary_closing(mask, structure=np.ones((3, 3, 3), bool), iterations=close_iter)
    struct = generate_binary_structure(3, 3)   # 26-connectivity
    lbl, n = cc_label(mask, structure=struct)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    keep = np.where(sizes >= min_voxels)[0]
    keep = keep[keep > 0]
    return np.isin(lbl, keep)


# ----------------------------------------------------------------------------
# step 2: NMF cell split
# ----------------------------------------------------------------------------


def _union_find_merge(n, pairs):
    """Group 0..n-1 into connected components given merge pairs."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def split_cells(stack, mask, k=6, merge_r=0.8, min_voxels=60, max_iter=300, seed=0):
    """NMF on masked voxel dF/F time series -> per-voxel cell id volume.

    Returns:
      cellvol : (Z,Y,X) int32, 0 = background, 1..ncells = cell id
      cell_traces : (ncells, T) mean dF/F trace per cell
      info : dict with nmf/merge diagnostics
    """
    T = stack.shape[0]
    coords = np.argwhere(mask)                       # (n_vox, 3)
    n_vox = coords.shape[0]
    cellvol = np.zeros(mask.shape, np.int32)
    if n_vox == 0:
        return cellvol, np.zeros((0, T), np.float32), {"n_vox": 0, "note": "empty mask"}

    # masked voxel traces (T, n_vox) -> dF/F
    zc, yc, xc = coords[:, 0], coords[:, 1], coords[:, 2]
    F = stack[:, zc, yc, xc].astype(np.float32).T    # (n_vox, T)
    F0 = np.percentile(F, 10, axis=1, keepdims=True)
    dff = (F - F0) / (np.abs(F0) + 1e-6)
    dff = dff - dff.min()                             # global shift -> non-negative

    keff = int(max(1, min(k, n_vox, T)))
    single = keff < 2 or n_vox < 2
    if single:
        comp_of_vox = np.zeros(n_vox, int)
        H = dff.mean(0, keepdims=True)
        n_comp = 1
        merged = [[0]]
    else:
        from sklearn.decomposition import NMF
        model = NMF(n_components=keff, init="nndsvda", random_state=seed,
                    max_iter=max_iter, tol=1e-4)
        W = model.fit_transform(dff)                  # (n_vox, keff)
        H = model.components_                          # (keff, T) NMF basis (not merged on)
        comp_of_vox = np.argmax(W, axis=1)
        n_comp = keff
        # Merge components whose representative TEMPORAL activity correlates
        # > merge_r. The representative signal is each component's mean dF/F
        # trace (the quantity the prototype measured at r~0.79-0.95 intra-cell,
        # ~0.3 across distinct subtrees). The raw NMF basis rows H are near-
        # orthogonal by construction and must NOT be used for this test.
        active = [c for c in range(n_comp) if np.any(comp_of_vox == c)]
        pairs = []
        if len(active) > 1:
            comp_trace = np.zeros((n_comp, T), np.float32)
            for c in active:
                comp_trace[c] = dff[comp_of_vox == c].mean(0)
            Cn = comp_trace - comp_trace.mean(1, keepdims=True)
            norm = np.linalg.norm(Cn, axis=1) + 1e-12
            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    a, b = active[ii], active[jj]
                    r = float(np.dot(Cn[a], Cn[b]) / (norm[a] * norm[b]))
                    if r > merge_r:
                        pairs.append((a, b))
        merged = _union_find_merge(n_comp, pairs)
        merged = [g for g in merged if any(np.any(comp_of_vox == c) for c in g)]

    # map component -> cell index (1-based), then voxel -> cell
    comp2cell = {}
    for ci, group in enumerate(merged, start=1):
        for c in group:
            comp2cell[c] = ci
    cell_of_vox = np.array([comp2cell[c] for c in comp_of_vox], dtype=np.int32)

    # drop / reassign cells smaller than min_voxels to nearest surviving cell
    ncells = len(merged)
    sizes = {ci: int(np.sum(cell_of_vox == ci)) for ci in range(1, ncells + 1)}
    big = [ci for ci, s in sizes.items() if s >= min_voxels]
    if not big:                                        # keep the single largest
        big = [max(sizes, key=sizes.get)] if sizes else []
    if big and len(big) < ncells:
        keepset = set(big)
        # spatial nearest reassignment for voxels of dropped cells
        tmp = np.zeros(mask.shape, np.int32)
        tmp[zc, yc, xc] = cell_of_vox
        drop_mask = np.isin(tmp, list(keepset), invert=True) & mask
        keep_seed = np.where(np.isin(tmp, list(keepset)), tmp, 0)
        if np.any(drop_mask) and np.any(keep_seed):
            from scipy.ndimage import distance_transform_edt
            _, idx = distance_transform_edt(keep_seed == 0, return_indices=True)
            tmp[drop_mask] = keep_seed[tuple(idx[:, drop_mask])]
        cell_of_vox = tmp[zc, yc, xc]
        # renumber surviving cells to 1..K contiguous
        remap = {old: new for new, old in enumerate(sorted(set(cell_of_vox[cell_of_vox > 0].tolist())), start=1)}
        cell_of_vox = np.array([remap.get(int(c), 0) for c in cell_of_vox], dtype=np.int32)

    cellvol[zc, yc, xc] = cell_of_vox
    ncells = int(cellvol.max())

    # mean dF/F trace per cell (for temporal_r_to_others)
    cell_traces = np.zeros((ncells, T), np.float32)
    for ci in range(1, ncells + 1):
        sel = cell_of_vox == ci
        if np.any(sel):
            cell_traces[ci - 1] = dff[sel].mean(0)

    info = {
        "n_vox": int(n_vox),
        "k_requested": int(k),
        "k_effective": int(keff),
        "n_components_active": int(len(set(comp_of_vox.tolist()))),
        "n_cells": ncells,
        "merge_r": merge_r,
        "single_component": bool(single),
    }
    return cellvol, cell_traces, info


# ----------------------------------------------------------------------------
# step 3: per-cell skeleton anatomy labeling
# ----------------------------------------------------------------------------


def _build_skeleton_graph(skel, vox):
    """26-neighbor graph of skeleton voxels.

    Returns coords (M,3), adjacency list adj[i]=[(j,dist_um),...], degree (M,).
    Uses skan if importable (for the summary), else a direct neighbor scan.
    """
    coords = np.argwhere(skel)
    M = coords.shape[0]
    if M == 0:
        return coords, [], np.zeros(0, int)
    index = {tuple(c): i for i, c in enumerate(coords)}
    offsets = [(dz, dy, dx)
               for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if not (dz == 0 and dy == 0 and dx == 0)]
    vz, vy, vx = vox
    dists = {o: float(np.sqrt((o[0] * vz) ** 2 + (o[1] * vy) ** 2 + (o[2] * vx) ** 2))
             for o in offsets}
    adj = [[] for _ in range(M)]
    for i, (z, y, x) in enumerate(coords):
        for o in offsets:
            nb = (z + o[0], y + o[1], x + o[2])
            j = index.get(nb)
            if j is not None:
                adj[i].append((j, dists[o]))
    degree = np.array([len(a) for a in adj], int)
    return coords, adj, degree


def _dijkstra(adj, source, M):
    """Geodesic distance (physical) and parent pointers from source."""
    import heapq
    dist = np.full(M, np.inf)
    parent = np.full(M, -1, int)
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent


def label_anatomy(cellmask, vox, max_branches=7):
    """Label one cell's voxels 1=soma, 2=trunk, 3+=branches.

    Returns (local_labels (Z,Y,X) uint8, info dict). If the skeleton is too
    small/ambiguous the split is heuristic and info['provisional'] is True.
    """
    from scipy.ndimage import distance_transform_edt, label as cc_label, generate_binary_structure
    from skimage.morphology import skeletonize

    info = {"n_skel": 0, "dt_max_um": 0.0, "n_branches": 0,
            "provisional": True, "provisional_reasons": []}
    out = np.zeros(cellmask.shape, np.uint8)
    if cellmask.sum() == 0:
        info["provisional_reasons"].append("empty cell mask")
        return out, info

    dt = distance_transform_edt(cellmask, sampling=vox).astype(np.float32)
    soma_center = np.unravel_index(int(np.argmax(dt)), dt.shape)
    dt_max = float(dt[soma_center])
    info["dt_max_um"] = round(dt_max, 3)

    skel = skeletonize(cellmask)
    coords, adj, degree = _build_skeleton_graph(skel, vox)
    M = coords.shape[0]
    info["n_skel"] = int(M)

    reasons = []
    thin_dim = min(cellmask.shape)                       # slab thinness (voxels)
    if M < 25:
        reasons.append(f"skeleton small ({M} voxels)")
    if dt_max < 2.0:
        reasons.append(f"max radius small ({dt_max:.2f} um)")
    if thin_dim < 8:
        reasons.append(f"thin slab (min dim {thin_dim} voxels)")

    # Fallbacks when there is essentially no graph to walk
    if M < 3:
        r_soma = max(dt_max, 1.0)
        dist_c = distance_transform_edt(
            ~_point_mask(cellmask.shape, soma_center), sampling=vox)
        out[cellmask & (dist_c <= r_soma)] = 1           # soma blob
        out[cellmask & (out == 0)] = 2                   # everything else = trunk
        info["n_branches"] = 0
        info["provisional"] = True
        info["provisional_reasons"] = reasons + ["degenerate skeleton"]
        return out, info

    # soma skeleton node = skeleton voxel nearest the distance-transform peak
    d2 = ((coords - np.asarray(soma_center)) * np.asarray(vox)).astype(np.float32)
    soma_node = int(np.argmin((d2 ** 2).sum(1)))
    gdist, parent = _dijkstra(adj, soma_node, M)
    gdist[~np.isfinite(gdist)] = np.inf

    # soma region on the skeleton: nodes within R_soma geodesic of the soma node
    r_soma = max(dt_max, 2.0 * float(min(vox)))
    node_label = np.zeros(M, np.uint8)                   # 0 unassigned
    soma_nodes = np.where(gdist <= r_soma)[0]
    node_label[soma_nodes] = 1

    # trunk = geodesic path from soma to the farthest reachable endpoint
    endpoints = np.where(degree == 1)[0]
    reach = [e for e in endpoints if np.isfinite(gdist[e]) and node_label[e] != 1]
    if not reach:
        reach = [int(np.argmax(np.where(np.isfinite(gdist), gdist, -1)))]
    trunk_tip = int(max(reach, key=lambda e: gdist[e]))
    path = []
    n = trunk_tip
    while n != -1:
        path.append(n)
        if n == soma_node:
            break
        n = parent[n]
    for nidx in path:
        if node_label[nidx] == 0:
            node_label[nidx] = 2                          # trunk

    # branches = remaining skeleton, split into connected sub-trees
    remaining = np.where(node_label == 0)[0]
    branch_id_of_node = {}
    n_branches = 0
    if remaining.size:
        rem_set = set(remaining.tolist())
        seen = set()
        comps = []
        for start in remaining:
            if start in seen:
                continue
            stack_ = [start]
            comp = []
            seen.add(start)
            while stack_:
                u = stack_.pop()
                comp.append(u)
                for v, _w in adj[u]:
                    if v in rem_set and v not in seen:
                        seen.add(v)
                        stack_.append(v)
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        # Keep only the largest `max_branches` sub-trees so per-cell labels stay
        # within 1..9 (soma=1, trunk=2, branches=3..2+max_branches) and never
        # collide with the next cell's +10 offset. Smaller fragments are left
        # unlabeled on the skeleton and fold into the nearest kept compartment
        # during the dense nearest-skeleton grow below.
        n_branches = min(len(comps), max_branches)
        for bi, comp in enumerate(comps[:n_branches], start=3):
            for u in comp:
                branch_id_of_node[u] = bi
    info["n_branches"] = int(n_branches)
    info["n_branch_fragments_total"] = int(len(comps)) if remaining.size else 0

    # paint skeleton labels, then grow to every cell voxel by nearest skeleton voxel
    skel_lab = np.zeros(cellmask.shape, np.uint8)
    for i, (z, y, x) in enumerate(coords):
        lab = node_label[i]
        if lab == 0:                                      # trunk/soma stragglers and
            lab = branch_id_of_node.get(i, 0)             # dropped fragments -> grow to nearest
        skel_lab[z, y, x] = lab
    if np.count_nonzero(skel_lab) == 0:
        skel_lab[soma_center] = 1
    _, idx = distance_transform_edt(skel_lab == 0, sampling=vox, return_indices=True)
    grown = skel_lab[tuple(idx)]
    out[cellmask] = grown[cellmask]

    # confidence: one clear soma, one dominant trunk, enough skeleton
    trunk_vox = int(np.sum(out == 2))
    soma_vox = int(np.sum(out == 1))
    if soma_vox == 0:
        reasons.append("no soma region resolved")
    info["provisional"] = bool(reasons)
    info["provisional_reasons"] = reasons
    return out, info


def _point_mask(shape, pt):
    m = np.zeros(shape, bool)
    m[pt] = True
    return m


# ----------------------------------------------------------------------------
# assembly, preview, IO
# ----------------------------------------------------------------------------


def assemble_labelmap(cellvol, vox, max_branches=7):
    """Per-cell anatomy labeling -> single uint8 labelmap with the offset scheme.

    cell i (1-based): label = local + (i-1)*10, local 1=soma 2=trunk 3+=branch.
    Returns (labelmap uint8, per-cell anatomy info list).
    """
    ncells = int(cellvol.max())
    out = np.zeros(cellvol.shape, np.uint8)
    anat = []
    for ci in range(1, ncells + 1):
        cellmask = cellvol == ci
        offset = (ci - 1) * 10
        if offset + 9 > 255:
            anat.append({"id": ci, "skipped": "label range exceeds uint8"})
            continue
        local, info = label_anatomy(cellmask, vox, max_branches=max_branches)
        painted = local > 0
        out[painted] = (local[painted].astype(np.int32) + offset).astype(np.uint8)
        # map local labels to global ids
        soma = 1 + offset
        trunk = 2 + offset
        branches = [b + offset for b in range(3, 3 + info["n_branches"])]
        info.update({"id": ci, "labels": {"soma": soma, "trunk": trunk, "branches": branches}})
        anat.append(info)
    return out, anat


def make_preview(stack, mask, labelmap, cells_json, out_png, stem):
    """anatomy MIP + mask + label overlay, publication-tidy (Agg, no show)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    anat = stack.max(0).astype(np.float32)               # temporal max
    z_top = anat.max(0)                                  # (Y,X)  look down Z
    y_side = anat.max(1)                                 # (Z,X)  look down Y
    lab_top = labelmap.max(0)
    lab_side = labelmap.max(1)
    mask_top = mask.max(0)

    lo, hi = np.percentile(anat, (2, 99.7))

    maxlab = int(labelmap.max())
    rng = np.random.default_rng(1)
    colors = [(0, 0, 0, 0)]
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    for i in range(1, maxlab + 1):
        colors.append(tuple(palette[(i - 1) % 20]))
    lcmap = ListedColormap(colors)

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].imshow(z_top, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
    ax[0, 0].set_title("anatomy  temporal-max MIP (look down Z)")
    ax[0, 1].imshow(z_top, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
    ax[0, 1].imshow(np.ma.masked_where(~mask_top, mask_top), cmap="autumn",
                    alpha=0.5, aspect="auto")
    ax[0, 1].set_title("auto-mask")
    ax[1, 0].imshow(z_top, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
    ax[1, 0].imshow(np.ma.masked_where(lab_top == 0, lab_top), cmap=lcmap,
                    vmin=0, vmax=maxlab, alpha=0.75, interpolation="nearest", aspect="auto")
    ax[1, 0].set_title("anatomy labels (XY)  1=soma 2=trunk 3+=branch  +10/cell")
    ax[1, 1].imshow(y_side, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
    ax[1, 1].imshow(np.ma.masked_where(lab_side == 0, lab_side), cmap=lcmap,
                    vmin=0, vmax=maxlab, alpha=0.75, interpolation="nearest", aspect="auto")
    ax[1, 1].set_title("anatomy labels (XZ side)")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])

    ncells = len(cells_json)
    prov = sum(1 for c in cells_json if c.get("anatomy_provisional"))
    fig.suptitle(f"{stem}   cells={ncells}   mask={int(mask.sum())} vox"
                 f"   provisional anatomy={prov}/{ncells}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def output_paths(stack_path, out_dir=None):
    d = os.path.dirname(os.path.abspath(stack_path))
    stem = os.path.basename(stack_path)
    if stem.lower().endswith(".tif"):
        stem = stem[:-4]
    elif stem.lower().endswith(".tiff"):
        stem = stem[:-5]
    dd = out_dir if out_dir else d
    return {
        "stem": stem,
        "dir": dd,
        "labelmap": os.path.join(dd, f"{stem}_autoseg_labelmap.tif"),
        "json": os.path.join(dd, f"{stem}_autoseg.json"),
        "preview": os.path.join(dd, f"{stem}_autoseg_preview.png"),
    }


# ----------------------------------------------------------------------------
# per-stack driver
# ----------------------------------------------------------------------------


def process_one(stack_path, args):
    import tifffile
    t0 = time.time()
    paths = output_paths(stack_path, args.out_dir)

    if args.report_only:
        _log(f"[report-only] {stack_path}")
        for key in ("labelmap", "json", "preview"):
            _log(f"    -> {paths[key]}")
        return True

    _log(f"[run] {stack_path}")
    # Prefer the DeepCAD-denoised sibling as the COMPUTE source when present:
    # measured on run7, the denoised input gives a tighter mask (48% vs 57%
    # stray voxels) with fewer spurious cell splits. Output naming stays on the
    # ORIGINAL stem, so downstream contracts (review tool, status tracker,
    # coherence) are unaffected; traces are extracted elsewhere from raw.
    sp = Path(stack_path)                           # callers pass str
    cand = sp.parent / (sp.stem + "_denoised.tif")
    src = args.mask_source
    if src == "auto":
        src = "denoised" if cand.exists() else "raw"
    if src == "denoised" and not cand.exists():
        raise FileNotFoundError(f"--mask-source denoised but {cand.name} does not exist")
    compute_path = cand if src == "denoised" else sp
    mask_source_record = {"mask_source": src, "computed_from": compute_path.name,
                          "analysis_note": "mask geometry only; traces/events use the raw registered stack"}
    _log(f"    mask computed from: {compute_path.name}  (--mask-source {args.mask_source} -> {src})")
    stack = tifffile.imread(compute_path)
    if stack.ndim == 3:                                  # (Z,Y,X): fake a length-1 T
        stack = stack[None]
    if stack.ndim != 4:
        raise ValueError(f"expected 4D (T,Z,Y,X) or 3D (Z,Y,X), got {stack.shape}")
    T, Z, Y, X = stack.shape
    vox = tuple(resolve_voxel(sp, args.voxel))

    corr, amax, mean = activity_statistics(stack, tchunk=args.tchunk)
    score = compute_score(corr, amax, mean, sigma=args.sigma)
    mask = build_mask(score, args.thr, args.min_voxels, close_iter=args.close)
    n_mask = int(mask.sum())
    _log(f"    mask: {n_mask} voxels ({100.0 * n_mask / mask.size:.2f}% of volume)")

    cellvol, cell_traces, split_info = split_cells(
        stack, mask, k=args.k, merge_r=args.merge_r,
        min_voxels=args.min_voxels, max_iter=args.nmf_iter, seed=args.seed)
    ncells = int(cellvol.max())
    _log(f"    cells: {ncells}  (k_eff={split_info.get('k_effective')}, "
         f"active_comp={split_info.get('n_components_active')})")

    labelmap, anat = assemble_labelmap(cellvol, vox, max_branches=args.max_branches)

    # temporal correlations between cells
    if ncells >= 2:
        Cn = cell_traces - cell_traces.mean(1, keepdims=True)
        nrm = np.linalg.norm(Cn, axis=1) + 1e-12
        rmat = (Cn @ Cn.T) / np.outer(nrm, nrm)
    else:
        rmat = np.ones((ncells, ncells), np.float32)

    cells_json = []
    for ci in range(1, ncells + 1):
        a = next((x for x in anat if x.get("id") == ci), {})
        r_others = {int(cj): round(float(rmat[ci - 1, cj - 1]), 3)
                    for cj in range(1, ncells + 1) if cj != ci}
        cells_json.append({
            "id": ci,
            "labels": a.get("labels", {"soma": None, "trunk": None, "branches": []}),
            "n_voxels": int(np.sum(cellvol == ci)),
            "temporal_r_to_others": r_others,
            "anatomy_provisional": bool(a.get("provisional", True)),
            "provisional_reasons": a.get("provisional_reasons", []),
            "n_skeleton_voxels": int(a.get("n_skel", 0)),
            "max_radius_um": float(a.get("dt_max_um", 0.0)),
            "n_branches": int(a.get("n_branches", 0)),
        })

    runtime = time.time() - t0
    peak_mb = _peak_mem_mb()
    params = {
        "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "input": os.path.abspath(stack_path),
        "input_shape_TZYX": [int(T), int(Z), int(Y), int(X)],
        "voxel_zyx_um": list(vox),
        **mask_source_record,
        "thr": args.thr, "k": args.k, "merge_r": args.merge_r,
        "min_voxels": args.min_voxels, "sigma": args.sigma,
        "max_branches": args.max_branches,
        "close_iter": args.close, "nmf_max_iter": args.nmf_iter, "seed": args.seed,
        "skan_available": _skan_available(),
    }
    stats = {
        "mask_voxels": n_mask,
        "mask_fraction": round(n_mask / float(mask.size), 5),
        "score_thr": args.thr,
        "n_cells": ncells,
        "nmf": split_info,
        "runtime_s": round(runtime, 2),
        "peak_mem_mb": round(peak_mb, 1),
        "any_provisional_anatomy": bool(any(c["anatomy_provisional"] for c in cells_json)),
    }
    doc = {"cells": cells_json, "params": params, "stats": stats}

    os.makedirs(paths["dir"], exist_ok=True)
    tifffile.imwrite(paths["labelmap"], labelmap.astype(np.uint8))
    with open(paths["json"], "w") as fh:
        json.dump(doc, fh, indent=2)
    make_preview(stack, mask, labelmap, cells_json, paths["preview"], paths["stem"])

    _log(f"    wrote {paths['labelmap']}")
    _log(f"    wrote {paths['json']}")
    _log(f"    wrote {paths['preview']}")
    _log(f"    runtime {runtime:.1f}s  peak_mem {peak_mb:.0f} MB")
    return True


def _skan_available():
    try:
        import skan  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def find_inputs(args):
    paths = list(args.stacks)
    if args.glob:
        paths += sorted(globmod.glob(args.glob, recursive=True))
    seen, uniq = set(), []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stacks", nargs="*", help="4D (T,Z,Y,X) stack path(s)")
    ap.add_argument("--glob", default=None, help="glob pattern for input stacks (recursive)")
    ap.add_argument("--thr", type=float, default=0.8, help="score threshold (0.8-1.0)")
    ap.add_argument("--k", type=int, default=6, help="NMF components")
    ap.add_argument("--merge-r", type=float, default=0.8,
                    help="merge NMF components whose temporal factors correlate above this")
    ap.add_argument("--min-voxels", type=int, default=60,
                    help="drop mask components / cells smaller than this")
    ap.add_argument("--max-branches", type=int, default=7,
                    help="max branch labels per cell (keeps per-cell labels within the +10 offset)")
    ap.add_argument("--sigma", type=float, default=0.8, help="gaussian smoothing sigma (voxels)")
    ap.add_argument("--close", type=int, default=1, help="binary_closing iterations (3x3x3)")
    add_voxel_arg(ap)
    ap.add_argument("--tchunk", type=int, default=200, help="time-chunk for streaming stats")
    ap.add_argument("--nmf-iter", type=int, default=300, help="NMF max_iter")
    ap.add_argument("--seed", type=int, default=0, help="NMF random_state")
    ap.add_argument("--out-dir", default=None, help="write outputs here instead of alongside input")
    ap.add_argument("--mask-source", choices=["auto", "raw", "denoised"], default="auto",
                    help="image the MASK is computed from: raw = <stem>.tif, denoised = "
                         "<stem>_denoised.tif (DeepCAD), auto = denoised if present else raw. "
                         "Recorded in the autoseg JSON. Traces always come from raw.")
    ap.add_argument("--report-only", action="store_true",
                    help="print planned output paths and exit (no computation, no writes)")
    args = ap.parse_args(argv)

    inputs = find_inputs(args)
    if not inputs:
        ap.error("no input stacks (give positional path(s) or --glob)")

    _log(f"auto_segment v{__version__}  |  {len(inputs)} stack(s)  |  "
         f"skan={'yes' if _skan_available() else 'no (neighbor-count fallback)'}")
    ok, fail = 0, 0
    for p in inputs:
        try:
            process_one(p, args)
            ok += 1
        except Exception as e:                            # batch-safe: keep going
            fail += 1
            _log(f"[FAIL] {p}: {type(e).__name__}: {e}")
            traceback.print_exc()
    _log(f"done: {ok} ok, {fail} failed")
    return 1 if fail and ok == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
