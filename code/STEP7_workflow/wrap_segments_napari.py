#!/usr/bin/env python
"""
wrap_segments_napari.py - ONE-CLICK anatomy wrapping for dendrite segments.

After the automatic cell proposal (auto_segment.py) you accept/reject whole cells,
lightly refine the mask, then WRAP anatomy with a single click: click the soma and
the whole soma blob becomes label 1; click the trunk and the trunk path becomes 2;
click each branch and that subtree becomes 3, 4, ...  The result is a plain sequential
labelmap (1..N, 0 = background) that feeds code/extra/segment_event_coherence.py
(which orders segments by X, soma-end -> branch-end, and needs nothing fancier).

This file has TWO entry points:
  * default        : opens napari (lazy import - napari is NOT imported at module load).
  * --check        : headless. Loads the run's inputs, runs the WRAP MATH on a synthetic
                     click (the mask voxel of maximum distance-transform radius), checks it
                     produces a non-empty, junction-bounded region, exercises the save
                     logic into a temp dir, prints PASS/FAIL and sets the exit code.
                     Imports only numpy / scipy / skimage / tifffile - never napari.

--------------------------------------------------------------------------------
KEYBINDINGS (interactive)
--------------------------------------------------------------------------------
  1..9        toggle cell #N in / out of the working mask (accept / reject a whole
              proposed cell). Title shows which cells are currently IN. Default: all in.
  a           cycle the projection axis of the anatomy MIP background (XY -> XZ -> ZY).
  w           toggle WRAP mode. While ON, a single click (no drag) wraps the region
              under the cursor; while OFF you can paint/erase to refine the mask.
  s           re-label the LAST wrap as soma  (label 1).
  t           re-label the LAST wrap as trunk (label 2).
  b           re-label the LAST wrap as the next free branch label (>= 3).
  i           toggle INTERVAL mode: click TWO points along the dendrite and the stretch
              between them (along the skeleton, bounded by the two cross-sections) becomes
              a region - e.g. 'proximal 20 um of branch 2' or 'trunk 50-80 um from soma'.
  k           toggle CUT mode: click on an existing region and it is split in two at the
              cross-section through the click (perpendicular to the local dendrite). The
              boundary is exactly where you clicked - correct any automatic boundary.
  e           toggle ERASE mode: napari brush on the mask removes voxels; regions covering
              them shrink immediately. ([ ] brush size.) Any other mode switches it off.
  m           merge the last two wraps into one region (fixes skeleton over-splits).
  n           name the last wrap (dialog); names go into the JSON sidecar and figures.
  u           undo the last wrap.
  Ctrl+S      save  <stem>_clean_segments_final.tif (uint8, labels 1..N) + JSON sidecar.

  Mask refine uses napari's own paint controls on the "mask (refine)" layer:
    P / E      paint / erase        [ / ]   smaller / larger brush        Ctrl+Z  undo stroke
  (Editing the mask invalidates the skeleton cache; the next wrap re-skeletonizes.)

--------------------------------------------------------------------------------
WRAP MATH (pure functions, numpy/scipy/skimage only - tested by --check)
--------------------------------------------------------------------------------
  1. skeletonize the working mask (skimage.morphology.skeletonize), 3D.
  2. count 26-neighbours on the skeleton; voxels with degree >= 3 are BRANCH POINTS.
  3. remove the branch points and connected-component label the remainder: each
     component is an ARC (one inter-junction branch of the skeleton).
  4. GEODESIC PARTITION: for every arc, flood its geodesic distance THROUGH the mask
     (skimage.graph.MCP_Geometric with cost 1 inside the mask, inf outside, anisotropic
     voxel sampling); assign each mask voxel to its nearest arc. This mirrors the g-key
     grow in STEP4_segments and bounds every arc's territory where the next arc's
     territory begins - growth stops at junctions.
  5. a click maps to its nearest skeleton arc; the WRAPPED REGION is that arc's territory.
  6. SOMA auto-suggestion: arc_radius = median EDT along an arc's skeleton (local tube
     radius); median_arc_radius = median over arcs. If the wrapped region's max EDT
     radius >= 2 * median_arc_radius the region is blob-like -> suggest soma (label 1);
     otherwise it is the next free label. s / t / b let you override.

Usage
-----
  # interactive
  python code/STEP7_workflow/wrap_segments_napari.py <run_dir_or_clean.tif> [--voxel Z Y X]
  # headless self-test on run7
  python code/STEP7_workflow/wrap_segments_napari.py <run_dir> --check

Never overwrites hand-made *_clean_segments*.tif or any *_autoseg* file: the output is
<stem>_clean_segments_final.tif, and if that already exists it is versioned (_v2, _v3, ...).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
# scipy / skimage are safe at module load: they do NOT import napari. Only the
# interactive launch() imports napari, so `import wrap_segments_napari` stays headless.
from scipy.ndimage import distance_transform_edt, convolve, label as cc_label, label, binary_dilation
from skimage.morphology import skeletonize
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from common.voxel import add_voxel_arg, resolve_voxel
from common.napari_panel import ActionPanel
from common.cleanup import drop_small_islands, describe
from skimage.graph import MCP_Geometric

DEFAULT_VOXEL = (0.85, 0.8, 0.8)          # fallback only; real value read from autoseg JSON
MIN_ISLAND_VOX = 20        # components smaller than this are dropped at save (per label)
SOMA_FACTOR = 2.0                         # region-max-radius >= SOMA_FACTOR * median-arc-radius -> soma
CC26 = np.ones((3, 3, 3), np.uint8)       # 26-connectivity structuring element


# ======================================================================================
# PURE WRAP-MATH (no napari, no I/O) - this is what --check exercises
# ======================================================================================
def cell_id_of_label(label: int) -> int:
    """Autoseg scheme is (cell-1)*10 + {1=soma,2=trunk,3+=branch}; recover the 1-based cell."""
    return (int(label) - 1) // 10 + 1


def accepted_cells_mask(autoseg: np.ndarray, accepted) -> np.ndarray:
    """Boolean union of the accepted whole cells (anatomy split ignored)."""
    accepted = set(int(a) for a in accepted)
    out = np.zeros(autoseg.shape, bool)
    labels = [int(v) for v in np.unique(autoseg) if v > 0]
    for lbl in labels:
        if cell_id_of_label(lbl) in accepted:
            out |= (autoseg == lbl)
    return out


def cell_ids(autoseg: np.ndarray):
    """Sorted list of 1-based cell ids present in an autoseg labelmap."""
    return sorted({cell_id_of_label(int(v)) for v in np.unique(autoseg) if v > 0})


def cell_id_volume(autoseg: np.ndarray) -> np.ndarray:
    """Map every voxel to its 1-based cell id (0 = background) for display / toggling."""
    out = np.zeros(autoseg.shape, np.uint16)
    for lbl in (int(v) for v in np.unique(autoseg) if v > 0):
        out[autoseg == lbl] = cell_id_of_label(lbl)
    return out


def skeleton_branch_points(skel: np.ndarray) -> np.ndarray:
    """Skeleton voxels with 26-degree >= 3 (junctions)."""
    deg = convolve(skel.astype(np.uint8), CC26, mode="constant") - skel.astype(np.uint8)
    return skel & (deg >= 3)


def partition_skeleton_into_arcs(skel: np.ndarray, min_arc_vox: int = 1):
    """Cut the skeleton at its branch points; connected-component label the pieces.

    Returns (arc_labels int32 [same shape, 0 = not-an-arc], n_arcs). Arcs smaller than
    min_arc_vox are dropped (default 1 keeps every inter-junction segment)."""
    nodes = skeleton_branch_points(skel)
    edges = skel & ~nodes                                   # cut junctions -> disjoint arcs
    lab, n = cc_label(edges, structure=CC26)
    if min_arc_vox > 1:
        sizes = np.bincount(lab.ravel())
        keep = [i for i in range(1, n + 1) if sizes[i] >= min_arc_vox]
        out = np.zeros_like(lab)
        for new, old in enumerate(keep, start=1):
            out[lab == old] = new
        return out.astype(np.int32), len(keep)
    return lab.astype(np.int32), int(n)


def geodesic_arc_partition(mask: np.ndarray, arc_labels: np.ndarray, voxel):
    """Assign every mask voxel to its geodesically nearest arc, measured THROUGH the mask.

    Cost is 1 inside the mask and inf outside, so paths never leave the dendrite; growth
    from each arc stops where a nearer arc's territory begins. Returns (partition int32,
    best_cost float). Mirrors the MCP_Geometric inflate used by STEP4 segment tools."""
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
    part[np.isinf(best)] = 0                                # unreachable (disconnected, no arc)
    return part, best


def arc_radii(edt: np.ndarray, arc_labels: np.ndarray):
    """Per-arc local tube radius = MEDIAN distance-transform value along the arc skeleton."""
    return {int(a): float(np.median(edt[arc_labels == a]))
            for a in np.unique(arc_labels) if a > 0}


def nearest_arc(point, arc_labels: np.ndarray, voxel) -> int:
    """Arc id of the skeleton voxel nearest (anisotropic) to `point` (z,y,x)."""
    av = np.argwhere(arc_labels > 0)
    if av.size == 0:
        raise ValueError("no skeleton arcs to click on (empty skeleton)")
    d2 = (((av - np.asarray(point, float)) * np.asarray(voxel, float)) ** 2).sum(1)
    return int(arc_labels[tuple(av[int(d2.argmin())])])


def soma_blob_from_thickness(mask, edt, median_arc_radius, voxel, factor=SOMA_FACTOR):
    """The soma as a BLOB, independent of the skeleton (which fragments a blob into many
    arcs so that off-centre clicks grab only a slice of it).

    Core = mask voxels whose distance-transform radius >= factor x the median arc radius
    (i.e. much wider than any branch). Keep the largest core component, then grow it
    outward through the mask while the local radius stays above the median branch
    radius, so the soma includes its whole rounded surface but stops where the trunk
    begins to look like a tube. Returns a boolean volume (all False if nothing is thick)."""
    if median_arc_radius <= 0:
        return np.zeros(mask.shape, bool)
    core = mask & (edt >= factor * median_arc_radius)
    if not core.any():
        return np.zeros(mask.shape, bool)
    lab, n = label(core, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    blob = lab == int(sizes.argmax())
    # grow from the core, but only through voxels whose local radius stays above HALF
    # the soma's own radius: the soma ends at the first neck where the tube narrows.
    # (Using the median branch radius here let the blob run down a thick proximal trunk.)
    soma_r = float(edt[blob].max())
    allowed = mask & (edt >= 0.5 * soma_r)
    grown = blob.copy()
    for _ in range(64):
        nxt = binary_dilation(grown, structure=np.ones((3, 3, 3))) & allowed
        if nxt.sum() == grown.sum():
            break
        grown = nxt
    # add the rounded surface: mask voxels within one soma radius of the thick body,
    # measured as a plain dilation by ceil(soma_r / voxel) steps but clipped to the mask
    steps = int(np.ceil(soma_r / float(min(voxel))))
    rind = grown.copy()
    for _ in range(steps):
        rind = binary_dilation(rind, structure=np.ones((3, 3, 3))) & mask
    # but never past the neck: drop rind voxels farther from the body than soma_r
    # along the tube (approximate with EDT-to-body)
    dist_to_body = distance_transform_edt(~grown, sampling=tuple(voxel))
    rind &= dist_to_body <= soma_r
    return rind


def build_wrap_cache(mask: np.ndarray, voxel, min_arc_vox: int = 1) -> dict:
    """Everything a click needs, computed once per mask state (skeleton -> arcs ->
    geodesic partition -> radii). Recompute after the mask is edited."""
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("working mask is empty (no cells accepted?)")
    skel = skeletonize(mask)
    if not skel.any():
        raise ValueError("skeleton is empty - mask too small?")
    arc_labels, n_arcs = partition_skeleton_into_arcs(skel, min_arc_vox)
    partition, _ = geodesic_arc_partition(mask, arc_labels, voxel)
    edt = distance_transform_edt(mask, sampling=tuple(voxel))
    radii = arc_radii(edt, arc_labels)
    median_arc_radius = float(np.median(list(radii.values()))) if radii else 0.0
    soma_blob = soma_blob_from_thickness(mask, edt, median_arc_radius, voxel)
    return {
        "soma_blob": soma_blob,
        "mask": mask,
        "skel": skel,
        "n_skel": int(skel.sum()),
        "arc_labels": arc_labels,
        "n_arcs": int(n_arcs),
        "partition": partition,
        "edt": edt,
        "arc_radius": radii,
        "median_arc_radius": median_arc_radius,
    }


def wrap_from_click(mask: np.ndarray, click_zyx, voxel, cache: dict = None,
                    min_arc_vox: int = 1, soma_factor: float = SOMA_FACTOR) -> dict:
    """Core one-click wrap. Returns the wrapped region and a soma suggestion.

    click_zyx is snapped to the nearest mask voxel if it lands outside the mask, then
    mapped to its nearest skeleton arc; the wrapped region is that arc's geodesic
    territory (bounded by junctions)."""
    if cache is None:
        cache = build_wrap_cache(mask, voxel, min_arc_vox)
    mask = cache["mask"]
    pt = np.clip(np.round(np.asarray(click_zyx)).astype(int), 0, np.array(mask.shape) - 1)
    if not mask[tuple(pt)]:                                 # snap to nearest mask voxel
        mv = np.argwhere(mask)
        d2 = (((mv - pt) * np.asarray(voxel, float)) ** 2).sum(1)
        pt = mv[int(d2.argmin())]
    med = cache["median_arc_radius"]
    blob = cache.get("soma_blob")
    if blob is not None and blob[tuple(pt)]:
        # click inside the thick blob -> the WHOLE soma, whichever skeleton arc is nearest
        arc = -1
        region = blob.copy()
        region_max_radius = float(cache["edt"][region].max())
        soma_suggested = True
    else:
        arc = nearest_arc(pt, cache["arc_labels"], voxel)
        region = (cache["partition"] == arc)
        if blob is not None:
            region = region & ~blob                          # arcs never eat into the soma
        region_max_radius = float(cache["edt"][region].max()) if region.any() else 0.0
        soma_suggested = bool(med > 0 and region_max_radius >= soma_factor * med)
    return {
        "click": tuple(int(x) for x in pt),
        "arc": int(arc),
        "region": region,
        "size": int(region.sum()),
        "click_radius": float(cache["edt"][tuple(pt)]),
        "region_max_radius": region_max_radius,
        "arc_radius": float(cache["arc_radius"].get(arc, 0.0)),
        "median_arc_radius": med,
        "soma_suggested": soma_suggested,
        "n_arcs": cache["n_arcs"],
        "n_skel": cache["n_skel"],
    }


def interval_from_clicks(mask: np.ndarray, a_zyx, b_zyx, voxel, cache: dict = None,
                         min_arc_vox: int = 1) -> dict:
    """TWO-CLICK INTERVAL: the piece of dendrite BETWEEN two points along the skeleton.

    1. snap both clicks to the nearest skeleton voxel;
    2. geodesic shortest path along the skeleton between them (MCP on skeleton voxels
       only, anisotropic sampling) -> the interval's centreline;
    3. every mask voxel whose geodesic-nearest skeleton voxel (measured THROUGH the mask)
       lies on that centreline belongs to the interval. Growth therefore stops exactly at
       the two clicked cross-sections, not at junctions - so 'proximal 20 um of branch 2'
       or 'trunk 50-80 um from the soma' are one gesture each.
    Returns dict(region, size, path (N,3), length_um, endpoints)."""
    if cache is None:
        cache = build_wrap_cache(mask, voxel, min_arc_vox)
    mask, skel = cache["mask"], cache["skel"]
    sk = np.argwhere(skel)
    def snap(pt):
        pt = np.clip(np.round(np.asarray(pt)).astype(int), 0, np.array(mask.shape) - 1)
        d2 = (((sk - pt) * np.asarray(voxel, float)) ** 2).sum(1)
        return tuple(int(x) for x in sk[int(d2.argmin())])
    a, b = snap(a_zyx), snap(b_zyx)
    if a == b:
        raise ValueError("both clicks snapped to the same skeleton voxel")
    cost = np.where(skel, 1.0, np.inf)
    mcp = MCP_Geometric(cost, sampling=tuple(voxel))
    cum, _ = mcp.find_costs([a], [b])
    if not np.isfinite(cum[b]):
        raise ValueError("the two points are not connected along the skeleton")
    path = np.array(mcp.traceback(b), int)
    on_path = np.zeros(mask.shape, bool); on_path[tuple(path.T)] = True
    # nearest skeleton voxel for every mask voxel, through the mask
    if "skel_owner" not in cache:
        c2 = np.where(mask, 1.0, np.inf)
        m2 = MCP_Geometric(c2, sampling=tuple(voxel))
        d, tb = m2.find_costs([tuple(q) for q in sk])
        offs = np.array(m2.offsets)
        idx = np.indices(mask.shape).reshape(3, -1).T
        cur = idx.copy(); tbf = tb.reshape(-1); src = skel.reshape(-1)
        for _ in range(int(np.linalg.norm(mask.shape)) + 5):
            fl = np.ravel_multi_index(cur.T, mask.shape)
            done = src[fl]
            if done.all():
                break
            st = tbf[fl]; mv = (~done) & (st >= 0)
            cur[mv] = cur[mv] - offs[st[mv]]
        cache["skel_owner"] = np.ravel_multi_index(cur.T, mask.shape).reshape(mask.shape)
    owner = cache["skel_owner"]
    region = mask & on_path.reshape(-1)[owner]
    seglen = np.linalg.norm(np.diff(path, axis=0) * np.asarray(voxel, float), axis=1).sum()
    return {"region": region, "size": int(region.sum()), "path": path,
            "length_um": float(seglen), "endpoints": (a, b)}


def cut_region_at(mask: np.ndarray, region: np.ndarray, click_zyx, voxel, cache: dict = None) -> tuple:
    """CUT HERE: split `region` (a boolean sub-volume of the mask) into two at the
    cross-section through the click, perpendicular to the local dendrite direction.

    1. snap the click to the nearest skeleton voxel inside the region;
    2. local direction = principal axis of the skeleton voxels within ~4 um of it;
    3. every region voxel is assigned to the side of the plane (through the snapped
       point, normal = that direction) it falls on.
    Returns (side_a, side_b) boolean volumes, both non-empty, or raises ValueError.
    Nothing about this depends on skeleton arcs or junctions: the cut is exactly where
    you clicked, so the user corrects an automatic boundary with one click."""
    if cache is None:
        cache = build_wrap_cache(mask, voxel)
    skel = cache["skel"] & region
    if not skel.any():
        skel = cache["skel"]
    sk = np.argwhere(skel)
    pt = np.clip(np.round(np.asarray(click_zyx)).astype(int), 0, np.array(mask.shape) - 1)
    w = np.asarray(voxel, float)
    d2 = (((sk - pt) * w) ** 2).sum(1)
    c = sk[int(d2.argmin())]
    near = sk[np.sqrt((((sk - c) * w) ** 2).sum(1)) <= 4.0]
    if len(near) < 3:
        near = sk[np.argsort(d2)[:7]]
    X = (near - c) * w
    _, _, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    normal = vt[0]
    rv = np.argwhere(region)
    side = ((rv - c) * w) @ normal
    a = np.zeros(mask.shape, bool); b = np.zeros(mask.shape, bool)
    a[tuple(rv[side < 0].T)] = True; b[tuple(rv[side >= 0].T)] = True
    if not a.any() or not b.any():
        raise ValueError("cut plane does not split the region (click nearer its middle)")
    return a, b, tuple(int(x) for x in c)


def merge_wraps(wraps, i: int, j: int) -> list:
    """Merge wrap j into wrap i (union of regions, i's label/name kept). Returns new list."""
    if i == j or not (0 <= i < len(wraps) and 0 <= j < len(wraps)):
        return wraps
    wi, wj = wraps[i], wraps[j]
    merged = dict(wi); merged["region"] = wi["region"] | wj["region"]
    merged["size"] = int(merged["region"].sum())
    merged["merged_from"] = wi.get("merged_from", []) + [wj.get("name", str(j))]
    out = [w for k, w in enumerate(wraps) if k not in (i, j)]
    out.insert(min(i, j), merged)
    return out


def suggest_label(soma_suggested: bool, used_labels) -> int:
    """soma -> 1 (if free); otherwise the smallest free positive label."""
    used = set(int(u) for u in used_labels)
    if soma_suggested and 1 not in used:
        return 1
    n = 1
    while n in used:
        n += 1
    return n


def next_branch_label(used_labels) -> int:
    """Smallest free branch label (>= 3)."""
    used = set(int(u) for u in used_labels)
    n = 3
    while n in used:
        n += 1
    return n


def label_name(label: int) -> str:
    label = int(label)
    if label == 1:
        return "soma"
    if label == 2:
        return "trunk"
    return f"branch{label - 2}"


def relabel_sequential(seg: np.ndarray) -> np.ndarray:
    """Remap present labels to a contiguous 1..N in ascending order (soma=1 stays first)."""
    out = np.zeros(seg.shape, np.uint8)
    for new, old in enumerate(sorted(int(v) for v in np.unique(seg) if v > 0), start=1):
        out[seg == old] = new
    return out


def segment_name_map(wraps, shape, final_seg) -> dict:
    """{final_label: name} for the saved labelmap. build_labelmap paints wraps in click
    order (later wins) and then renumbers the surviving labels 1..N in ascending order;
    this replays that so each final id gets the name of the last wrap painted with it."""
    raw = np.zeros(shape, np.int32)
    name_of = {}
    for w in wraps:
        raw[w["region"]] = w["label"]
        name_of[int(w["label"])] = w.get("name") or label_name(w["label"])
    out = {}
    for new, old in enumerate(sorted(int(v) for v in np.unique(raw) if v > 0), start=1):
        if (final_seg == new).any():
            out[str(new)] = name_of.get(old, f"seg{new}")
    return out


def build_labelmap(wraps, shape) -> np.ndarray:
    """Rebuild the segment labelmap by applying wraps in click order (later overrides
    earlier where regions overlap), then relabel to a contiguous 1..N."""
    seg = np.zeros(shape, np.uint8)
    for w in wraps:
        seg[w["region"]] = w["label"]
    return relabel_sequential(seg)


# ======================================================================================
# I/O helpers (tifffile / json only - still no napari)
# ======================================================================================
def find_inputs(target, autoseg_override=None, ref3d_override=None, out_override=None) -> dict:
    """Resolve a run directory OR a *_clean.tif into the set of files we need."""
    target = Path(target)
    if target.is_dir():
        rundir = target
        cleans = [c for c in sorted(rundir.glob("*_clean.tif"))
                  if not c.name.endswith("_denoised.tif")]
        if not cleans:
            raise FileNotFoundError(f"no *_clean.tif in {rundir}")
        clean = cleans[0]
    else:
        clean = target
        rundir = clean.parent
    name = clean.name
    stem = name[:-len("_clean.tif")] if name.endswith("_clean.tif") else clean.stem

    ref3d = Path(ref3d_override) if ref3d_override else rundir / f"{stem}_clean_ref3d.tif"

    autoseg = Path(autoseg_override) if autoseg_override else None
    if autoseg is None:
        for cand in (f"{stem}_clean_autoseg_labelmap_reviewed.tif",
                     f"{stem}_clean_autoseg_labelmap.tif",
                     f"{stem}_clean_denoised_autoseg_labelmap_reviewed.tif",
                     f"{stem}_clean_denoised_autoseg_labelmap.tif"):
            if (rundir / cand).exists():
                autoseg = rundir / cand
                break
    autoseg_json = None
    if autoseg is not None:
        j = Path(str(autoseg).replace("_labelmap_reviewed.tif", ".json")
                 .replace("_labelmap.tif", ".json"))
        autoseg_json = j if j.exists() else None

    # hand-made segment labelmaps (used only by --check for the rough agreement number)
    hand = [h for h in sorted(rundir.glob(f"{stem}_clean_segments*.tif"))
            if "final" not in h.name]

    out = Path(out_override) if out_override else rundir / f"{stem}_clean_segments_final.tif"
    return {"rundir": rundir, "stem": stem, "clean": clean, "ref3d": ref3d,
            "autoseg": autoseg, "autoseg_json": autoseg_json, "hand": hand, "out": out}


def load_voxel(autoseg_json, cli_voxel=None):
    """voxel_zyx_um from the autoseg JSON params, else CLI, else DEFAULT_VOXEL."""
    if cli_voxel:
        return tuple(float(x) for x in cli_voxel)
    if autoseg_json and Path(autoseg_json).exists():
        try:
            j = json.loads(Path(autoseg_json).read_text())
            v = j.get("params", {}).get("voxel_zyx_um")
            if v and len(v) == 3:
                return tuple(float(x) for x in v)
        except Exception:
            pass
    # no silent default: fall back to the run's acquisition metadata, which exits
    # loudly if it cannot be found
    stem = Path(autoseg_json).name.replace("_autoseg.json", ".tif") if autoseg_json else None
    return tuple(resolve_voxel(Path(autoseg_json).parent / stem, None)) if stem else DEFAULT_VOXEL


def anatomy_background(ref3d_path, clean_path=None) -> np.ndarray:
    """C0 (anatomy) of the (Z,C,Y,X) static reference as a (Z,Y,X) float32 volume.
    Falls back to the temporal mean of the 4D clean stack if no ref3d is present."""
    if ref3d_path and Path(ref3d_path).exists():
        r = tifffile.imread(str(ref3d_path))
        if r.ndim == 4:                                     # (Z, C, Y, X)
            return r[:, 0].astype(np.float32)
        return r.astype(np.float32)
    stack = tifffile.imread(str(clean_path))
    return (stack.mean(0) if stack.ndim == 4 else stack).astype(np.float32)


def _safe_out_path(out: Path) -> Path:
    """Never clobber an existing file (incl. hand-made maps). Version _v2, _v3, ... .
    Also refuse names that look hand-made or like autoseg output."""
    lname = out.name.lower()
    if "autoseg" in lname or ("segments" in lname and "final" not in lname):
        raise ValueError(f"refusing to write to a non-'final' segments/autoseg path: {out}")
    if not out.exists():
        return out
    stem, suf = out.stem, out.suffix
    i = 2
    while True:
        cand = out.with_name(f"{stem}_v{i}{suf}")
        if not cand.exists():
            return cand
        i += 1


def save_segments(out_path, seg, voxel, sidecar_extra=None) -> dict:
    """Write the uint8 labelmap (1..N, contiguous) + a JSON sidecar. Returns paths + ids."""
    out_path = _safe_out_path(Path(out_path))
    seg = relabel_sequential(seg).astype(np.uint8)
    tifffile.imwrite(str(out_path), seg)
    ids = [int(i) for i in np.unique(seg) if i > 0]
    sidecar = {
        "created": datetime.datetime.now().astimezone().isoformat(),
        "tool": "wrap_segments_napari.py",
        "voxel_zyx_um": list(voxel),
        "labels": {str(i): label_name(i) for i in ids},
        "voxels_per_label": {str(i): int((seg == i).sum()) for i in ids},
    }
    if sidecar_extra:
        sidecar.update(sidecar_extra)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(sidecar, indent=2))
    return {"tif": out_path, "json": json_path, "ids": ids}


# ======================================================================================
# HEADLESS SELF-TEST (--check) - imports nothing beyond numpy/scipy/skimage/tifffile
# ======================================================================================
def run_check(target, voxel_cli=None, min_arc_vox=1, soma_factor=SOMA_FACTOR) -> bool:
    print("=" * 78)
    print("wrap_segments_napari.py --check")
    print("=" * 78)
    ok = True
    inp = find_inputs(target)
    print(f"run dir : {inp['rundir']}")
    print(f"stem    : {inp['stem']}")
    print(f"autoseg : {inp['autoseg']}")
    print(f"ref3d   : {inp['ref3d']}  (exists={Path(inp['ref3d']).exists()})")
    if inp["autoseg"] is None:
        print("FAIL: no autoseg labelmap found")
        return False

    voxel = load_voxel(inp["autoseg_json"], voxel_cli)
    print(f"voxel   : {voxel} um (z,y,x)")

    autoseg = tifffile.imread(str(inp["autoseg"]))
    cells = cell_ids(autoseg)
    mask = accepted_cells_mask(autoseg, cells)              # default: ALL cells accepted
    print(f"cells   : {cells}  ->  working mask {int(mask.sum())} voxels "
          f"(shape {mask.shape})")

    # ---- core wrap math on a synthetic click = max distance-transform radius voxel ----
    cache = build_wrap_cache(mask, voxel, min_arc_vox)
    edt = cache["edt"]
    click = np.unravel_index(int(np.argmax(np.where(mask, edt, -1.0))), mask.shape)
    res = wrap_from_click(mask, click, voxel, cache=cache, soma_factor=soma_factor)

    print("-" * 78)
    print("WRAP MATH")
    print(f"  n skeleton voxels        : {res['n_skel']}")
    print(f"  n junction arcs          : {res['n_arcs']}")
    print(f"  branch-point voxels      : {int(skeleton_branch_points(cache['skel']).sum())}")
    print(f"  synthetic click (z,y,x)  : {res['click']}   radius={res['click_radius']:.3f} um")
    print(f"  wrapped-region size      : {res['size']} voxels")
    print(f"  wrapped arc id           : {res['arc']}")
    print(f"  region max radius        : {res['region_max_radius']:.3f} um")
    print(f"  median arc radius        : {res['median_arc_radius']:.3f} um "
          f"(soma if region max >= {soma_factor:g}x = {soma_factor*res['median_arc_radius']:.3f})")
    print(f"  soma auto-suggested?     : {res['soma_suggested']}")

    # region must be non-empty and junction-bounded (strictly inside the mask, >1 arc)
    non_empty = res["size"] > 0
    bounded = (res["size"] < int(mask.sum())) and (res["n_arcs"] >= 2)
    print(f"  non-empty region         : {non_empty}")
    print(f"  bounded by junctions     : {bounded}  "
          f"(region {res['size']} < mask {int(mask.sum())}, arcs {res['n_arcs']} >= 2)")
    if not (non_empty and bounded):
        ok = False
        print("  FAIL: region empty or not junction-bounded")
    if not res["soma_suggested"]:
        ok = False
        print("  FAIL: soma auto-suggestion did not fire on the max-radius region")

    # ---- INTERVAL MATH: two clicks along the longest skeleton arc ----
    print("-" * 78)
    print("INTERVAL MATH (two-click stretch selection)")
    arc_ids, counts = np.unique(cache["arc_labels"][cache["arc_labels"] > 0], return_counts=True)
    longest = int(arc_ids[counts.argmax()])
    pts = np.argwhere(cache["arc_labels"] == longest)
    # two points ~ at 25% and 75% along the arc's principal axis
    axis = pts[:, np.argmax(np.ptp(pts, axis=0))]
    order = np.argsort(axis); a_pt = pts[order[len(order) // 4]]; b_pt = pts[order[3 * len(order) // 4]]
    try:
        iv = interval_from_clicks(mask, a_pt, b_pt, voxel, cache=cache)
        whole_arc = int((cache["partition"] == longest).sum())
        print(f"  longest arc {longest}: {len(pts)} skel vox, territory {whole_arc} vox")
        print(f"  interval {iv['endpoints'][0]} -> {iv['endpoints'][1]}: path {len(iv['path'])} vox, "
              f"{iv['length_um']:.1f} um, region {iv['size']} vox")
        sub = 0 < iv["size"] < whole_arc
        inside = bool((iv["region"] & ~mask).sum() == 0)
        path_in = bool(iv["region"][tuple(iv["path"].T)].all())
        print(f"  strict sub-region of the arc : {sub}")
        print(f"  region inside mask           : {inside}")
        print(f"  centreline inside region     : {path_in}")
        if not (sub and inside and path_in):
            ok = False; print("  FAIL: interval region invalid")
        # cut: split the interval at its middle skeleton point -> two non-empty halves
        mid = iv["path"][len(iv["path"]) // 2]
        ca, cb, cpt = cut_region_at(mask, iv["region"], mid, voxel, cache=cache)
        print(f"  cut interval at {cpt}  : {int(ca.sum())} + {int(cb.sum())} vox "
              f"(= {iv['size']}: {int(ca.sum()) + int(cb.sum()) == iv['size']}, disjoint: {not (ca & cb).any()})")
        if not (ca.any() and cb.any() and int(ca.sum()) + int(cb.sum()) == iv["size"] and not (ca & cb).any()):
            ok = False; print("  FAIL: cut_region_at invalid")
        # merge: interval + the soma wrap -> union, size adds up minus overlap
        merged = merge_wraps([{"region": res["region"], "label": 1, "name": "soma", "size": res["size"]},
                              {"region": iv["region"], "label": 3, "name": "iv", "size": iv["size"]}], 0, 1)
        exp = int((res["region"] | iv["region"]).sum())
        print(f"  merge soma+interval          : {len(merged)} wrap(s), {merged[0]['size']} vox (expected {exp})")
        if not (len(merged) == 1 and merged[0]["size"] == exp):
            ok = False; print("  FAIL: merge_wraps wrong")
    except Exception as e:
        ok = False; print(f"  FAIL: interval raised {type(e).__name__}: {e}")

    # ---- rough agreement vs the user's hand segments labelmap (soma = label 1) ----
    print("-" * 78)
    print("ROUGH AGREEMENT vs hand segments (honest, un-tuned)")
    hand5 = [h for h in inp["hand"] if h.name.endswith("segments5.tif")]
    hand_file = hand5[0] if hand5 else (inp["hand"][-1] if inp["hand"] else None)
    if hand_file is None:
        print("  (no hand *_clean_segments*.tif found to compare)")
    else:
        hand = tifffile.imread(str(hand_file))
        if hand.shape == mask.shape:
            soma = hand == 1
            region = res["region"]
            inter = int((region & soma).sum())
            frac_region_in_soma = inter / max(1, region.sum())
            frac_soma_covered = inter / max(1, int(soma.sum()))
            print(f"  hand file                : {hand_file.name}")
            print(f"  hand soma (label 1) vox  : {int(soma.sum())}")
            print(f"  wrapped soma-click vox   : {int(region.sum())}")
            print(f"  fraction of wrap inside hand-soma : {frac_region_in_soma:.3f}")
            print(f"  fraction of hand-soma covered     : {frac_soma_covered:.3f}")
        else:
            print(f"  (hand {hand.shape} != mask {mask.shape}; skipping)")

    # ---- exercise the save logic into a temp dir (never touches the run dir) ----
    print("-" * 78)
    print("SAVE LOGIC (temp dir)")
    wraps = [{"region": res["region"], "label": 1}]        # pretend the user kept it as soma
    seg = build_labelmap(wraps, mask.shape)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"{inp['stem']}_clean_segments_final.tif"
        saved = save_segments(out, seg, voxel,
                              sidecar_extra={"accepted_cells": cells,
                                             "wrap_clicks": [res["click"]]})
        reloaded = tifffile.imread(str(saved["tif"]))
        seq_ok = list(np.unique(reloaded)) == [0, 1] and reloaded.dtype == np.uint8
        round_trip = np.array_equal(reloaded, seg)
        json_ok = saved["json"].exists() and json.loads(saved["json"].read_text())["labels"] == {"1": "soma"}
        print(f"  wrote     : {saved['tif'].name} (uint8, ids {saved['ids']})")
        print(f"  sidecar   : {saved['json'].name}")
        print(f"  round-trip equal          : {round_trip}")
        print(f"  labels sequential/uint8   : {seq_ok}")
        print(f"  sidecar labels correct    : {json_ok}")
        if not (round_trip and seq_ok and json_ok):
            ok = False
            print("  FAIL: save/reload/round-trip mismatch")
    # TemporaryDirectory auto-cleans; nothing left on disk.

    print("=" * 78)
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 78)
    return ok


# ======================================================================================
# INTERACTIVE (napari imported lazily here - NOT at module load)
# ======================================================================================
def launch(target, voxel_cli=None, min_arc_vox=1, soma_factor=SOMA_FACTOR):
    import napari                                            # lazy: keeps module import headless

    inp = find_inputs(target)
    if inp["autoseg"] is None:
        raise SystemExit(f"no autoseg labelmap found under {inp['rundir']}")
    voxel = load_voxel(inp["autoseg_json"], voxel_cli)
    vox = tuple(voxel)

    autoseg = tifffile.imread(str(inp["autoseg"]))
    cellvol = cell_id_volume(autoseg)
    all_cells = cell_ids(autoseg)
    bg = anatomy_background(inp["ref3d"], inp["clean"])

    AX = {0: "XY", 1: "XZ", 2: "ZY"}
    S = {
        "accepted": set(all_cells),                         # default: every cell in
        "last_base": accepted_cells_mask(autoseg, all_cells),
        "wraps": [],                                         # list of dict(region,label,name,meta)
        "cache": None,                                       # wrap cache; None => stale
        "cache_mask": None,                                  # mask snapshot the cache was built on
        "wrap_mode": False, "interval_mode": False, "pending": None, "cut_mode": False, "erase_mode": False,
        "axis": 0,
    }

    v = napari.Viewer(ndisplay=3)
    clim = (float(np.percentile(bg, 2)), float(np.percentile(bg, 99.7)))
    v.add_image(bg, name="anatomy", colormap="gray", scale=vox,
                contrast_limits=clim, rendering="attenuated_mip", opacity=0.6)
    cells_layer = v.add_labels(cellvol, name="cells (accept/reject)", scale=vox, opacity=0.35)
    mask_layer = v.add_labels(S["last_base"].astype(np.uint8), name="mask (refine)",
                              scale=vox, opacity=0.3)
    seg_layer = v.add_labels(np.zeros(cellvol.shape, np.uint8), name="segments (wrap)",
                             scale=vox, opacity=0.7)
    v.scale_bar.visible = True
    v.scale_bar.unit = "um"

    def working_mask():
        return np.asarray(mask_layer.data).astype(bool)

    def rebuild_mask_from_cells():
        """Recompute the base union for the accepted cells while preserving manual edits."""
        cur = working_mask()
        added = cur & ~S["last_base"]
        removed = (~cur) & S["last_base"]
        base = accepted_cells_mask(autoseg, S["accepted"])
        new = (base | added) & ~removed
        mask_layer.data = new.astype(np.uint8)
        S["last_base"] = base
        S["cache"] = None                                   # mask changed -> skeleton stale

    def refresh_segments():
        seg_layer.data = build_labelmap(S["wraps"], cellvol.shape)

    def status():
        acc = ",".join(str(c) for c in sorted(S["accepted"])) or "none"
        wr = "  ".join(f"{label_name(w['label'])}={w['label']}({w['size']}vx)"
                       for w in S["wraps"]) or "none yet"
        mode = ("ERASE (brush)" if S.get("erase_mode") else
                "CUT (click=split)" if S["cut_mode"] else
                "INTERVAL (2 clicks)" if S["interval_mode"] else
                "WRAP (click=wrap)" if S["wrap_mode"] else "refine (paint/erase)")
        title = f"cells in: {acc} | {mode} | wraps: {wr}"
        v.title = title
        print(title, flush=True)
        if S.get("panel") is not None:
            P = S["panel"]
            P.set_toggle("w", S["wrap_mode"] and not S["interval_mode"] and not S["cut_mode"])
            P.set_toggle("i", S["interval_mode"]); P.set_toggle("k", S["cut_mode"])
            P.set_toggle("e", S.get("erase_mode", False))
            if S.get("erase_mode"):
                P.hint("ERASE: brush over the mask to remove voxels; regions shrink with it. [ ] brush size")
            elif S["cut_mode"]:
                P.hint("CUT: click on a region where it should be split in two")
            elif S["interval_mode"]:
                P.hint("Click the SECOND point along the dendrite" if S["pending"] is not None
                       else "INTERVAL: click the FIRST point along the dendrite")
            elif S["wrap_mode"]:
                P.hint("WRAP: click a piece of dendrite to make it a region")
            elif not S["wraps"]:
                P.hint("Press [w] (or the button) and click the soma")
            else:
                P.hint("Refine mode: paint/erase the mask. [w] to go back to wrapping")
            lines = [f"{k+1}. {w['name']}  (label {w['label']}, {w['size']} vox"
                     + (f", {w['length_um']:.0f} um" if w.get("kind") == "interval" else "") + ")"
                     for k, w in enumerate(S["wraps"])]
            P.status("cells in: " + acc + "\n" + ("\n".join(lines) if lines else "no regions yet"))

    def ensure_cache():
        m = working_mask()
        if S["cache"] is None or S["cache_mask"] is None or not np.array_equal(m, S["cache_mask"]):
            print("skeletonizing working mask ...", flush=True)
            S["cache"] = build_wrap_cache(m, vox, min_arc_vox)
            S["cache_mask"] = m.copy()
            print(f"  {S['cache']['n_skel']} skeleton voxels, {S['cache']['n_arcs']} junction arcs",
                  flush=True)
        return S["cache"]

    # ---- cell accept/reject: number keys 1..9 ----
    def toggle_cell(cid):
        if cid not in all_cells:
            print(f"cell {cid} does not exist (cells: {all_cells})", flush=True)
            return
        if cid in S["accepted"]:
            S["accepted"].discard(cid)
        else:
            S["accepted"].add(cid)
        rebuild_mask_from_cells()
        status()

    for d in range(1, 10):
        v.bind_key(str(d), lambda vw, d=d: toggle_cell(d), overwrite=True)

    # ---- anatomy projection axis cycle ----
    def cycle_axis(vw):
        S["axis"] = (S["axis"] + 1) % 3
        print(f"anatomy view axis -> {AX[S['axis']]}", flush=True)
        try:
            v.dims.ndisplay = 3
        except Exception:
            pass
    v.bind_key("a", cycle_axis, overwrite=True)

    # ---- WRAP mode toggle ----
    def toggle_wrap(vw):
        if S.get("erase_mode"):
            S["erase_mode"] = False; mask_layer.mode = "pan_zoom"
        S["wrap_mode"] = not S["wrap_mode"]
        if S["wrap_mode"]:
            ensure_cache()
            for ly in (cells_layer, mask_layer, seg_layer):
                ly.mode = "pan_zoom"                         # clicks are wraps, drags rotate
            v.layers.selection.active = seg_layer
        else:
            v.layers.selection.active = mask_layer
        status()
    v.bind_key("w", toggle_wrap, overwrite=True)

    # ---- INTERVAL mode: two clicks select the stretch of dendrite between them ----
    def toggle_interval(vw):
        S["interval_mode"] = not S["interval_mode"]; S["pending"] = None; S["cut_mode"] = False
        if S["interval_mode"] and not S["wrap_mode"]:
            toggle_wrap(vw)                                    # needs click-to-select active
        print(f"interval mode {'ON: click two points along the dendrite' if S['interval_mode'] else 'OFF'}",
              flush=True)
        status()
    v.bind_key("i", toggle_interval, overwrite=True)

    # ---- CUT mode: one click splits the region under the cursor at that cross-section ----
    def toggle_cut(vw):
        S["cut_mode"] = not S["cut_mode"]
        if S["cut_mode"]:
            S["interval_mode"] = False; S["pending"] = None
            if not S["wrap_mode"]:
                toggle_wrap(vw)
        print(f"cut mode {'ON: click where a region should be split' if S['cut_mode'] else 'OFF'}", flush=True)
        status()
    v.bind_key("k", toggle_cut, overwrite=True)

    # ---- merge the last two wraps into one region ----
    def merge_last_two(vw):
        if len(S["wraps"]) < 2:
            print("need two wraps to merge", flush=True); return
        n = len(S["wraps"])
        S["wraps"] = merge_wraps(S["wraps"], n - 2, n - 1)
        refresh_segments()
        print(f"merged into {S['wraps'][-1]['name']} ({S['wraps'][-1]['size']} vox)", flush=True)
        status()
    v.bind_key("m", merge_last_two, overwrite=True)

    # ---- name the last wrap (recorded in the JSON sidecar and the figure legend) ----
    def name_last(vw):
        if not S["wraps"]:
            print("no wraps yet", flush=True); return
        try:
            from qtpy.QtWidgets import QInputDialog
            txt, ok = QInputDialog.getText(None, "name this region",
                                           f"name for label {S['wraps'][-1]['label']}:",
                                           text=S["wraps"][-1]["name"])
            if ok and txt.strip():
                S["wraps"][-1]["name"] = txt.strip(); status()
        except Exception as e:
            print(f"naming dialog unavailable: {e}", flush=True)
    v.bind_key("n", name_last, overwrite=True)

    def _click_to_voxel(mask, event):
        """Best-effort cursor -> data voxel. In 3D, march the view ray to the first mask hit."""
        try:
            dc = np.asarray(seg_layer.world_to_data(event.position), float)
        except Exception:
            dc = np.asarray(event.position, float) / np.asarray(vox, float)
        pt = np.clip(np.round(dc).astype(int), 0, np.array(mask.shape) - 1)
        if mask[tuple(pt)]:
            return tuple(pt)
        vd = getattr(event, "view_direction", None)
        if vd is not None:
            step = np.asarray(vd, float) / np.asarray(vox, float)
            n = np.linalg.norm(step)
            if n > 0:
                step /= n
                p = dc.copy()
                for _ in range(int(np.linalg.norm(mask.shape)) * 3):
                    ip = np.round(p).astype(int)
                    if np.all(ip >= 0) and np.all(ip < np.array(mask.shape)) and mask[tuple(ip)]:
                        return tuple(ip)
                    p += step
        mv = np.argwhere(mask)                              # global nearest fallback
        d2 = (((mv - pt) * np.asarray(vox, float)) ** 2).sum(1)
        return tuple(mv[int(d2.argmin())])

    @seg_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        if not S["wrap_mode"]:
            return
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged:                                         # a rotate/pan, not a wrap click
            return
        m = working_mask()
        cache = ensure_cache()
        click = _click_to_voxel(m, event)
        if S["cut_mode"]:
            # find which wrap (if any) contains the click; else cut the whole mask piece there
            idx = next((k for k in range(len(S["wraps"]) - 1, -1, -1) if S["wraps"][k]["region"][click]), None)
            if idx is None:
                print("cut: click on an existing region first (wrap it, then cut it)", flush=True); return
            try:
                a, b, cpt = cut_region_at(m, S["wraps"][idx]["region"], click, vox, cache=cache)
            except ValueError as e:
                print(f"cut failed: {e}", flush=True); return
            old = S["wraps"][idx]
            wa = dict(old); wa["region"] = a; wa["size"] = int(a.sum()); wa["kind"] = "cut"
            wb = dict(old); wb["region"] = b; wb["size"] = int(b.sum()); wb["kind"] = "cut"
            wb["label"] = next_branch_label([w["label"] for w in S["wraps"]])
            wb["name"] = f"{old['name']}-b"; wa["name"] = f"{old['name']}-a"
            S["wraps"][idx] = wa; S["wraps"].insert(idx + 1, wb)
            refresh_segments()
            print(f"cut {old['name']} at {cpt} -> {wa['name']} ({wa['size']} vox, label {wa['label']}) + "
                  f"{wb['name']} ({wb['size']} vox, label {wb['label']}). s/t/b relabel the LAST one.", flush=True)
            status(); return
        if S["interval_mode"]:
            if S["pending"] is None:
                S["pending"] = click
                print(f"interval: first point {click} - click the second point", flush=True)
                return
            try:
                res = interval_from_clicks(m, S["pending"], click, vox, cache=cache)
            except ValueError as e:
                print(f"interval failed: {e}", flush=True); S["pending"] = None; return
            S["pending"] = None
            label = next_branch_label([w["label"] for w in S["wraps"]])
            w = {"region": res["region"], "label": label, "name": f"interval{len(S['wraps'])+1}",
                 "size": res["size"], "arc": -1, "click": res["endpoints"][0],
                 "click2": res["endpoints"][1], "length_um": res["length_um"],
                 "region_max_radius": 0.0, "soma_suggested": False, "kind": "interval"}
            S["wraps"].append(w); refresh_segments()
            print(f"interval {res['endpoints'][0]} -> {res['endpoints'][1]}: {res['length_um']:.1f} um along "
                  f"the skeleton, {res['size']} vox -> label {label}. (s/t/b relabel, n name, m merge, u undo)",
                  flush=True)
            status(); return
        res = wrap_from_click(m, click, vox, cache=cache, soma_factor=soma_factor)
        label = suggest_label(res["soma_suggested"], [w["label"] for w in S["wraps"]])
        w = {"region": res["region"], "label": label, "name": label_name(label),
             "size": res["size"], "arc": res["arc"], "click": res["click"],
             "region_max_radius": res["region_max_radius"],
             "soma_suggested": res["soma_suggested"]}
        S["wraps"].append(w)
        refresh_segments()
        print(f"wrapped arc {res['arc']} @ {res['click']} -> {label_name(label)} (label {label}); "
              f"{res['size']} vox, region max radius {res['region_max_radius']:.2f} um, "
              f"soma_suggested={res['soma_suggested']}. (s=soma t=trunk b=branch u=undo)",
              flush=True)
        status()

    # ---- relabel the last wrap ----
    def _relabel_last(new_label):
        if not S["wraps"]:
            print("no wraps yet", flush=True)
            return
        S["wraps"][-1]["label"] = new_label
        if S["wraps"][-1].get("kind") != "interval" or S["wraps"][-1]["name"].startswith("interval"):
            S["wraps"][-1]["name"] = label_name(new_label)
        refresh_segments()
        status()
    v.bind_key("s", lambda vw: _relabel_last(1), overwrite=True)
    v.bind_key("t", lambda vw: _relabel_last(2), overwrite=True)
    v.bind_key("b", lambda vw: _relabel_last(
        next_branch_label([w["label"] for w in S["wraps"][:-1]])), overwrite=True)

    @v.bind_key("u", overwrite=True)
    def _undo(vw):
        if S["wraps"]:
            gone = S["wraps"].pop()
            refresh_segments()
            print(f"undid wrap {gone['name']} (label {gone['label']})", flush=True)
            status()
        else:
            print("nothing to undo", flush=True)

    @v.bind_key("Control-S", overwrite=True)
    def _save(vw):
        seg = build_labelmap(S["wraps"], cellvol.shape)
        seg[~working_mask()] = 0                            # clamp to the current mask
        seg, rep = drop_small_islands(seg, min_voxels=MIN_ISLAND_VOX)
        print(f"island cleanup (<{MIN_ISLAND_VOX} vox per label): {describe(rep)}", flush=True)
        if not (seg > 0).any():
            print("nothing to save (no wraps).", flush=True)
            return
        extra = {
            "source_clean": str(inp["clean"]),
            "source_autoseg": str(inp["autoseg"]),
            "accepted_cells": sorted(S["accepted"]),
            "segment_names": segment_name_map(S["wraps"], cellvol.shape, seg),
            "islands_removed": [{"label": lb, "components": n, "voxels": v} for lb, n, v in rep],
            "min_island_voxels": MIN_ISLAND_VOX,
            "wrap_clicks": [{"click_zyx": list(w["click"]), "arc": int(w["arc"]),
                             "label": int(w["label"]), "name": w["name"],
                             "size": int(w["size"]),
                             "soma_suggested": bool(w["soma_suggested"]),
                             "kind": w.get("kind", "arc"),
                             **({"click2_zyx": list(w["click2"]), "length_um": round(w["length_um"], 2)}
                                if w.get("kind") == "interval" else {}),
                             **({"merged_from": w["merged_from"]} if w.get("merged_from") else {})}
                            for w in S["wraps"]],
        }
        saved = save_segments(inp["out"], seg, vox, sidecar_extra=extra)
        print(f"saved -> {saved['tif']}  labels={saved['ids']}", flush=True)
        print(f"        sidecar -> {saved['json']}", flush=True)
        print("NEXT (coherence figure): "
              f"python code/extra/segment_event_coherence.py '{inp['clean']}' "
              f"'{saved['tif']}' --voxel {vox[0]} {vox[1]} {vox[2]}", flush=True)

    print(__doc__.split("Usage")[0].split("KEYBINDINGS")[1] if "KEYBINDINGS" in __doc__ else "")

    # ---- ERASE mode: brush on the mask layer removes voxels from the mask AND from any
    #      region covering them (regions are clamped to the mask at save anyway, but the
    #      display should agree immediately) ----
    def toggle_erase(vw=None):
        S["erase_mode"] = not S.get("erase_mode", False)
        if S["erase_mode"]:
            S["wrap_mode"] = False; S["interval_mode"] = False; S["cut_mode"] = False; S["pending"] = None
            v.layers.selection.active = mask_layer
            mask_layer.mode = "erase"; mask_layer.brush_size = 2; mask_layer.n_edit_dimensions = 3
            print("ERASE: brush over mask voxels to remove them ([ ] = brush size). Regions follow.", flush=True)
        else:
            mask_layer.mode = "pan_zoom"
        status()

    @mask_layer.mouse_drag_callbacks.append
    def _after_mask_edit(layer, event):
        if not S.get("erase_mode"):
            return
        yield
        while event.type == "mouse_move":
            yield
        m = working_mask()
        changed = False
        for w in S["wraps"]:                                 # trim regions to the edited mask
            if (w["region"] & ~m).any():
                w["region"] = w["region"] & m; w["size"] = int(w["region"].sum()); changed = True
        S["cache"] = None                                    # skeleton is stale
        if changed:
            refresh_segments()
        status()

    v.bind_key("e", toggle_erase, overwrite=True)

    # ---- side panel: every key as a button, same callbacks ----
    P = ActionPanel(v, title="pick regions")
    P.section("1. select regions")
    P.button("Wrap whole piece (click)", key="w", cb=lambda: toggle_wrap(v), toggle=True,
             tooltip="One click selects the junction-to-junction piece under the cursor")
    P.button("Interval between two clicks", key="i", cb=lambda: toggle_interval(v), toggle=True,
             tooltip="Click two points; the stretch between them becomes a region")
    P.button("Cut region here (click)", key="k", cb=lambda: toggle_cut(v), toggle=True,
             tooltip="Click on a region: it is split in two at that cross-section, exactly where you clicked")
    P.section("2. label the last region")
    P.button("Last = soma", key="s", cb=lambda: _relabel_last(1))
    P.button("Last = trunk", key="t", cb=lambda: _relabel_last(2))
    P.button("Last = next branch", key="b",
             cb=lambda: _relabel_last(next_branch_label([w["label"] for w in S["wraps"][:-1]])))
    P.button("Name last region...", key="n", cb=lambda: name_last(v))
    P.section("3. fix")
    P.button("Erase with brush (mask + regions)", key="e", cb=lambda: toggle_erase(v), toggle=True,
             tooltip="Brush removes voxels from the mask; any region covering them shrinks too")
    P.button("Merge last two regions", key="m", cb=lambda: merge_last_two(v))
    P.button("Undo last region", key="u", cb=lambda: _undo(v))
    P.note("Cells: keys 1-9 toggle a proposed cell in/out. [a] rotates the anatomy view. "
           "Paint/erase the mask with napari's brush when wrap mode is off.")
    P.section("4. done")
    P.button("Save regions", key="Ctrl+S", cb=lambda: _save(v))
    P.finish()
    S["panel"] = P

    status()
    napari.run()


# ======================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="run directory OR a <stem>_clean.tif path")
    ap.add_argument("--check", action="store_true",
                    help="headless self-test of the wrap math + save logic (no napari)")
    add_voxel_arg(ap)
    ap.add_argument("--autoseg", default=None, help="explicit autoseg labelmap path")
    ap.add_argument("--ref3d", default=None, help="explicit ref3d path")
    ap.add_argument("--out", default=None, help="explicit output labelmap path")
    ap.add_argument("--min-arc-vox", type=int, default=1,
                    help="drop skeleton arcs smaller than this many voxels (default 1 = keep all)")
    ap.add_argument("--soma-factor", type=float, default=SOMA_FACTOR,
                    help="region-max-radius >= factor * median-arc-radius -> suggest soma")
    args = ap.parse_args(argv)

    if args.check:
        ok = run_check(args.target, voxel_cli=args.voxel,
                       min_arc_vox=args.min_arc_vox, soma_factor=args.soma_factor)
        sys.exit(0 if ok else 1)
    launch(args.target, voxel_cli=args.voxel,
           min_arc_vox=args.min_arc_vox, soma_factor=args.soma_factor)


if __name__ == "__main__":
    main()
