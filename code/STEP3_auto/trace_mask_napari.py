#!/usr/bin/env python
"""trace_mask_napari.py - path-guided 3D dendrite mask with one-slider haze removal.

WHY
  Thresholding a reference volume cannot separate a faint branch from the scattered-
  light halo of the bright trunk: in absolute intensity they are the same. This tool
  instead defines the mask RELATIVE to a traced centerline: a voxel belongs to the
  dendrite when it is brighter than alpha x the intensity of the nearest centerline
  point. Halo (~30 % of its source) drops out at alpha ~ 0.5 for trunk and branch
  alike, while the faint branch itself stays. Thickness is then one number you drag.

WORKFLOW (napari)
  * The reference volume is shown (default channel: cofire_mean, the tree at the
    moments it fires together; 'c' cycles channels).
  * Centerline seeding: the bright ridges of the reference are skeletonized into the
    initial trace (--seed autoseg uses the autoseg skeleton instead; --seed none starts
    empty) so on a good run you only confirm. Delete an arc by clicking it with 'x' held; add an arc
    by clicking two points with the 'trace' tool ('t' held): the tool finds the
    brightest-ridge path between them THROUGH the reference (geodesic, cost ~ 1/I).
    Clicks in the 2D top view snap through the whole Z column, so one pair of clicks
    connects structure across slices - never trace slice by slice.
  * Growth sliders (dock panel, live):
      alpha       relative threshold: keep voxels >= alpha x local centerline intensity
      radius x    cap on distance from the centerline, as a multiple of the local
                  radius estimated at each centerline point (soma large, branch small)
      pad (vox)   final dilation, for when you want a safety margin
  * Other cells ('i' / button): trace a crossing or neighbouring cell with two clicks;
    its arcs are magenta. Its grown region is taken OUT of your mask (voxels both cells
    could claim go to whichever cell's centerline is brighter there) and saved as
    <stem>_exclude_labelmap.tif, which the movie and figure tools black out. 'f' flips
    any arc (including seeded ones) between your cell and the other cell.
  * Erase by hand ('e' / button): paint on the red 'erase' layer; those voxels are
    removed from the mask and STAY removed when you move the sliders (the mask is
    regenerated from the centerline, then your erasures are subtracted).
  * 'uncertain' layer (orange): centerline points where the reference itself is dim
    (below --dim-pct of centerline intensities) - a path was found but the structure
    is not clearly there. Shown, never auto-bridged. Decide by eye.
  * Reopening resumes where you saved: every arc (yours and other cells', seeded or
    traced), its owner, your erasures and additions are kept in <stem>_trace_session.npz,
    and the sliders/reference channel come back at the values you saved with. A mask
    saved before sessions existed is rebuilt so it opens voxel-for-voxel as saved.
  * Add by hand ('a' / button): paint on the green layer to force voxels into your mask.
    --fresh starts over from the seed (saved session and exclusion ignored until you save).
  * Ctrl+S writes <stem>_autoseg_labelmap_reviewed.tif (cell 1, class 2 = structure,
    label 2) + a record in <stem>_autoseg_reviewed.json, so femto_status advances to
    mask_reviewed and wrap_segments_napari.py takes it from there. Originals are never
    overwritten.

  Keys: c channel | t(hold)+click trace | x(hold)+click delete arc | u undo | r reseed
        d 2D<->3D | Ctrl+S save
"""
from __future__ import annotations
import argparse, json, os, sys, pathlib as _pl
from datetime import datetime, timezone
import numpy as np, tifffile
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

_sys_root = _pl.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_sys_root))
from common.voxel import add_voxel_arg, resolve_voxel        # noqa: E402
from common.napari_panel import ActionPanel                  # noqa: E402
from common.cleanup import drop_small_islands, describe      # noqa: E402

__version__ = "0.1.0"
CELL_OFFSET = 10
STRUCT_LABEL = 2                       # cell 1, class 2 (trunk/structure) per autoseg contract


# ----------------------------------------------------------------------------- paths
def derive_paths(stack_path):
    p = _pl.Path(stack_path)
    stem = p.with_suffix("")
    return {"stem": str(stem),
            "ref3d": str(stem) + "_ref3d.tif", "ref3d_json": str(stem) + "_ref3d.json",
            "autoseg": str(stem) + "_autoseg_labelmap.tif", "autoseg_json": str(stem) + "_autoseg.json",
            "out_tif": str(stem) + "_autoseg_labelmap_reviewed.tif",
            "out_json": str(stem) + "_autoseg_reviewed.json",
            "exclude_tif": str(stem) + "_exclude_labelmap.tif",
            "session": str(stem) + "_trace_session.npz"}


def load_reference(paths, channel="cofire_mean"):
    """(Z,Y,X) float32 reference in [0,1] + list of channel names + index used."""
    ref = tifffile.imread(paths["ref3d"]).astype(np.float32)            # (Z,C,Y,X)
    names = list(json.load(open(paths["ref3d_json"]))["channels"]) if os.path.exists(paths["ref3d_json"]) \
        else [f"ch{i}" for i in range(ref.shape[1])]
    if channel not in names:                                           # 3-channel legacy files
        channel = "activity_p99.5" if "activity_p99.5" in names else names[1 if len(names) > 1 else 0]
    ci = names.index(channel)
    vol = ref[:, ci]
    lo, hi = np.percentile(vol, [1, 99.9])
    return np.clip((vol - lo) / (hi - lo + 1e-9), 0, 1), names, ci, ref


# ----------------------------------------------------------------------------- tracing
def cost_volume(ref, gamma=2.0, eps=0.02):
    """Geodesic cost: cheap on bright ridges, expensive in the dark. cost = (eps + 1-I)^gamma."""
    return (eps + (1.0 - ref)) ** gamma


def geodesic_path(cost, a, b, voxel):
    mcp = MCP_Geometric(cost, sampling=tuple(voxel))
    mcp.find_costs([tuple(int(x) for x in a)], [tuple(int(x) for x in b)])
    try:
        return np.array(mcp.traceback(tuple(int(x) for x in b)), int)
    except Exception:
        return None


def _order_arc(pts):
    """Order skeleton voxels of one arc along the path (argwhere order is raster, not path)."""
    pts = np.asarray(pts)
    if len(pts) <= 2:
        return pts
    P = {tuple(p): i for i, p in enumerate(pts)}
    nbrs = [[] for _ in pts]
    for i, p in enumerate(pts):
        for d in _OFFS26:
            j = P.get((p[0]+d[0], p[1]+d[1], p[2]+d[2]))
            if j is not None:
                nbrs[i].append(j)
    ends = [i for i, nb in enumerate(nbrs) if len(nb) == 1]
    start = ends[0] if ends else 0
    order, seen, cur, prev = [start], {start}, start, -1
    while True:
        nxt = [j for j in nbrs[cur] if j not in seen]
        if not nxt:
            break
        # prefer continuing straight (small loops in 26-connectivity)
        cur = nxt[0]; seen.add(cur); order.append(cur)
    if len(order) < len(pts):                        # stray voxels (loops): append by distance
        rest = [i for i in range(len(pts)) if i not in seen]
        order += rest
    return pts[order]


_OFFS26 = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]


def _skeleton_arcs(mask, min_arc=3, prune_radius=None):
    """Skeletonize a boolean volume and split at junctions -> list of ordered (N,3) arcs.

    With prune_radius (= voxel size): inside thick regions the skeleton of a blob is a
    tangle of short arcs between clustered junctions (the soma). Arcs shorter than the
    local tube radius that end at a junction are removed and their junction cluster is
    left as a single point so neighboring arcs still connect through it."""
    skel = skeletonize(mask)
    if skel.sum() == 0:
        return []
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant") - skel
    junction = skel & (nb >= 3)
    lab, n = ndi.label(skel & ~junction, structure=np.ones((3, 3, 3)))
    arcs = [_order_arc(np.argwhere(lab == i)) for i in range(1, n + 1) if (lab == i).sum() >= 1]
    if prune_radius is not None and len(arcs):
        edt = ndi.distance_transform_edt(mask, sampling=tuple(prune_radius))
        jpts = np.argwhere(junction)
        step = float(np.mean(prune_radius))
        def touches(e) -> int:
            return int(len(jpts) > 0 and bool(np.min(np.abs(jpts - e).max(1)) <= 1))
        kept = []
        for a in arcs:
            n_touch = touches(a[0]) + touches(a[-1])
            local_r = float(np.median(edt[tuple(a.T)]))
            length_um = len(a) * step
            if n_touch >= 1 and length_um < 1.5 * local_r:
                continue                                  # spur or tangle inside a thick blob
            if n_touch == 2 and len(a) < 8:
                continue                                  # junction-to-junction link: soma tangle
            kept.append(a)
        arcs = kept
        # re-attach: extend each kept arc end that touched a junction by the junction voxel
        # so arcs meeting at a soma still share a point
        out = []
        for a in arcs:
            a = list(map(tuple, a))
            for e in (a[0], a[-1]):
                if len(jpts) and np.min(np.abs(jpts - e).max(1)) <= 1:
                    j = jpts[np.argmin(np.abs(jpts - e).max(1))]
                    if e == a[0]: a.insert(0, tuple(j))
                    else: a.append(tuple(j))
            out.append(np.array(a))
        arcs = out
    return [a for a in arcs if len(a) >= min_arc]


def seed_from_reference(ref, z=5.0, min_component=50, min_arc=3, voxel=None):
    """Initial centerline from the reference's own bright ridges: robust-z threshold
    (median + z * MAD) -> drop specks -> skeleton split at junctions. Independent of
    autoseg quality (which on some runs degenerates to most of the volume)."""
    sm = ndi.gaussian_filter(ref, sigma=(0.5, 0.8, 0.8))
    med = np.median(sm); mad = np.median(np.abs(sm - med)) * 1.4826 + 1e-9
    m = sm > med + z * mad
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    m = np.isin(lab, np.where(sizes >= min_component)[0])
    return _skeleton_arcs(m, min_arc, prune_radius=voxel)


def seed_from_autoseg(paths, shape, min_arc=3):
    """Optional: centerline from the autoseg labelmap skeleton."""
    if not os.path.exists(paths["autoseg"]):
        return []
    lm = tifffile.imread(paths["autoseg"])
    return _skeleton_arcs(lm > 0, min_arc) if lm.shape == shape else []


# ----------------------------------------------------------------------------- growth
def grow_cache(ref, arcs, voxel, max_radius_um=12.0):
    """Everything the sliders need, computed once per trace edit.

    Returns dict with, per voxel: nearest-centerline intensity, geodesic distance from the
    centerline (um), local radius at the nearest centerline point (um), and per-arc data.
    Geodesic distance is measured with cost = 1/(ref+eps) so the territory follows the
    structure rather than a Euclidean ball; capped at max_radius_um for speed.
    """
    shape = ref.shape
    cl = np.zeros(shape, np.int32)
    for k, pts in enumerate(arcs, start=1):
        cl[tuple(pts.T)] = k
    if cl.max() == 0:
        return None
    # per-centerline-point intensity (smoothed along the arc so a single dim voxel does not
    # punch a hole) and local radius via the reference's own half-max extent
    smooth = ndi.gaussian_filter(ref, sigma=(0.5, 0.8, 0.8))
    cl_int = np.zeros(shape, np.float32)
    cl_int[cl > 0] = smooth[cl > 0]
    # local radius: distance transform of ref > 0.5*local intensity, read at the centerline
    half = smooth >= 0.5 * np.maximum(ndi.maximum_filter(cl_int, size=5), 1e-3)
    edt = ndi.distance_transform_edt(half, sampling=tuple(voxel))
    cl_rad = np.zeros(shape, np.float32)
    cl_rad[cl > 0] = np.maximum(edt[cl > 0], float(min(voxel)))
    # geodesic nearest-centerline: propagate intensity and radius via MCP from all
    # centerline points at once, with per-source values carried by the traceback labels
    cost = (1.0 / (smooth + 0.05)).astype(float)
    mcp = MCP_Geometric(cost, sampling=tuple(voxel))
    src = np.argwhere(cl > 0)
    dist, trace = mcp.find_costs([tuple(p) for p in src], max_cumulative_cost=None)
    # follow tracebacks to assign each voxel its source centerline voxel (vectorised via
    # offsets: walk the traceback field until reaching a source)
    offsets = np.array(mcp.offsets)
    idx = np.indices(shape).reshape(3, -1).T
    cur = idx.copy()
    tr_flat = trace.reshape(-1)
    is_src = (cl > 0).reshape(-1)
    for _ in range(int(max_radius_um / min(voxel)) + 20):
        flat = np.ravel_multi_index(cur.T, shape)
        done = is_src[flat]
        if done.all():
            break
        step = tr_flat[flat]
        move = (~done) & (step >= 0)
        cur[move] = cur[move] - offsets[step[move]]
    flat = np.ravel_multi_index(cur.T, shape)
    near_int = cl_int.reshape(-1)[flat].reshape(shape)
    near_rad = cl_rad.reshape(-1)[flat].reshape(shape)
    # euclidean distance to the assigned centerline voxel (um), for the radius cap
    src_xyz = cur.reshape(shape + (3,)).astype(np.float32)
    own = np.indices(shape).transpose(1, 2, 3, 0).astype(np.float32)
    eucl = np.sqrt((((own - src_xyz) * np.asarray(voxel, np.float32)) ** 2).sum(-1))
    # dim centerline points: structure not clearly present under the path
    cl_vals = smooth[cl > 0]
    return {"cl": cl, "ref": ref, "smooth": smooth, "near_int": near_int, "near_rad": near_rad,
            "eucl": eucl, "geo": dist, "cl_vals": cl_vals}


def grow(cache, alpha=0.5, radius_x=1.5, pad=0, dim_pct=15.0):
    """Mask + uncertain-centerline flags from the cache and the three slider values."""
    if cache is None:
        return None, None
    ref, ni, nr, eu = cache["smooth"], cache["near_int"], cache["near_rad"], cache["eucl"]
    m = (ref >= alpha * ni) & (eu <= radius_x * nr) & (ni > 0)
    m |= cache["cl"] > 0                                    # centerline always inside
    # close one-voxel seams where arcs meet (junction voxels belong to no arc)
    m = ndi.binary_closing(m, structure=np.ones((1, 3, 3))) | m
    # keep only components touching the centerline (kills detached halo islands)
    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    keep = np.unique(lab[cache["cl"] > 0]); keep = keep[keep > 0]
    m = np.isin(lab, keep)
    if pad > 0:
        m = ndi.binary_dilation(m, iterations=int(pad))
    thr = np.percentile(cache["cl_vals"], dim_pct) if len(cache["cl_vals"]) else 0
    unc = (cache["cl"] > 0) & (cache["smooth"] < thr)
    return m, unc


def owner_caches(ref, arcs, owners, voxel):
    """The expensive part of grow_owned (geodesic maps), once per arc/owner/reference change."""
    own_arcs = [a for a, o in zip(arcs, owners) if o != "intruder"]
    int_arcs = [a for a, o in zip(arcs, owners) if o == "intruder"]
    return (grow_cache(ref, own_arcs, voxel) if own_arcs else None,
            grow_cache(ref, int_arcs, voxel) if int_arcs else None)


def grow_owned(ref, arcs, owners, voxel, alpha=0.5, radius_x=1.5, pad=0, dim_pct=15.0, erase=None, add=None,
               caches=None):
    """Grow the own-cell mask and the intruder mask from ONE set of centerline arcs.

    owners[k] is "own" or "intruder" for arcs[k]. Each owner's mask is grown with the
    same relative-threshold rule as grow(); where both claim a voxel it goes to the owner
    whose nearest centerline is brighter there (its halo is weaker than the other cell's
    body). Own-cell centerline voxels always stay own. Returns (own, intruder, uncertain).
    """
    own_arcs = [a for a, o in zip(arcs, owners) if o != "intruder"]
    int_arcs = [a for a, o in zip(arcs, owners) if o == "intruder"]
    shape = ref.shape
    empty = np.zeros(shape, bool)
    c_own, c_int = caches if caches is not None else owner_caches(ref, arcs, owners, voxel)
    m_own, unc = grow(c_own, alpha, radius_x, pad, dim_pct) if c_own else (empty.copy(), empty.copy())
    m_int, _ = grow(c_int, alpha, radius_x, pad, dim_pct) if c_int else (empty.copy(), None)
    if m_own is None: m_own = empty.copy()
    if m_int is None: m_int = empty.copy()
    both = m_own & m_int
    if both.any():
        ni_own = c_own["near_int"] if c_own else np.zeros(shape, np.float32)
        ni_int = c_int["near_int"]
        to_int = both & (ni_int > ni_own)
        if c_own:
            to_int &= ~(c_own["cl"] > 0)                     # own centerline never moves
        m_own &= ~to_int
        m_int &= ~(both & ~to_int)
    if add is not None:                                   # painted-in voxels are always yours
        m_own |= add; m_int &= ~add
    if erase is not None:
        m_own &= ~erase; m_int &= ~erase
    return m_own, m_int, (unc if unc is not None else empty)


def save_exclude(paths, intruder, owners, arcs, voxel, min_island=20):
    """Write <stem>_exclude_labelmap.tif (uint8, 1 = voxels of other cells) and add an
    'exclude' block to the reviewed JSON. Used only for display (black-out); regions and
    traces never read it. Removes the file content-wise (writes zeros) if nothing is left,
    rather than deleting anything."""
    lm = intruder.astype(np.uint8)
    lm, _ = drop_small_islands(lm, min_voxels=min_island)
    tifffile.imwrite(paths["exclude_tif"], lm)
    doc = load_json_safe(paths["out_json"])
    doc["exclude"] = {
        "file": os.path.basename(paths["exclude_tif"]),
        "voxels": int((lm > 0).sum()),
        "n_intruder_arcs": int(sum(o == "intruder" for o in owners)),
        "updated": datetime.now(timezone.utc).isoformat(),
        "note": "voxels of other (crossing/neighbouring) cells: blacked out in movies and figure MIPs; never used for traces",
    }
    json.dump(doc, open(paths["out_json"], "w"), indent=2)
    return paths["exclude_tif"], int((lm > 0).sum())


def load_json_safe(path):
    """Read a JSON object. If the file exists but cannot be parsed, keep a timestamped
    copy (<name>.corrupt-YYYYmmdd-HHMMSS) and warn, so nothing already recorded is lost
    silently; then return {} so a fresh record can be written."""
    if not os.path.exists(path):
        return {}
    try:
        doc = json.load(open(path))
        return doc if isinstance(doc, dict) else {"_previous_non_object": doc}
    except Exception as e:
        import shutil
        bak = f"{path}.corrupt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, bak)
        print(f"[trace] WARNING: could not read {os.path.basename(path)} ({e}); kept a copy as "
              f"{os.path.basename(bak)} and started a new record")
        return {}


def last_saved_params(paths):
    """Settings of the last mask saved with this tool (alpha, radius_x, pad, channel),
    or None."""
    if not os.path.exists(paths["out_json"]):
        return None
    try:
        revs = [r for r in json.load(open(paths["out_json"])).get("reviews", [])
                if r.get("tool") == "trace_mask_napari"]
        return revs[-1].get("params") if revs else None
    except Exception:
        return None


def save_session(paths, arcs, owners, erase, add=None):
    """Everything needed to reopen exactly where you left off: every centerline arc (own
    and other cell, traced or seeded), its owner, and the hand erasures."""
    lens = np.array([len(a) for a in arcs], np.int64)
    pts = np.concatenate(arcs).astype(np.int32) if arcs else np.zeros((0, 3), np.int32)
    np.savez_compressed(paths["session"], arc_points=pts, arc_lengths=lens,
                        owners=np.array(owners, dtype="U8"),
                        erase=np.packbits(np.asarray(erase, bool).ravel()),
                        add=np.packbits(np.asarray(add if add is not None else np.zeros_like(erase), bool).ravel()),
                        shape=np.array(np.asarray(erase).shape, np.int64))


def load_session(paths, shape):
    """(arcs, owners, erase) from a saved session, or None if absent/incompatible."""
    if not os.path.exists(paths["session"]):
        return None
    try:
        z = np.load(paths["session"])
        if tuple(z["shape"]) != tuple(shape):
            print("[trace] saved session is for a different volume shape; ignoring it"); return None
        lens, pts = z["arc_lengths"], z["arc_points"]
        arcs = np.split(pts.astype(int), np.cumsum(lens)[:-1]) if len(lens) else []
        owners = [str(o) for o in z["owners"]]
        n = int(np.prod(shape))
        erase = np.unpackbits(z["erase"])[:n].reshape(shape).astype(bool)
        add = np.unpackbits(z["add"])[:n].reshape(shape).astype(bool) if "add" in z.files else np.zeros(shape, bool)
        if len(owners) != len(arcs):
            return None
        return arcs, owners, erase, add
    except Exception as e:
        print(f"[trace] WARNING: could not read saved session ({e}); starting from the seed")
        return None


# ----------------------------------------------------------------------------- save
def save(paths, mask, arcs, params, voxel, min_island=20):
    for k in ("out_tif", "out_json"):
        if os.path.abspath(paths[k]) in (os.path.abspath(paths["autoseg"]), os.path.abspath(paths["autoseg_json"])):
            raise RuntimeError("refusing to overwrite the original autoseg files")
    lm = np.where(mask, STRUCT_LABEL, 0).astype(np.uint8)
    lm, rep = drop_small_islands(lm, min_voxels=min_island)
    print(f"[trace] island cleanup (<{min_island} vox): {describe(rep)}")
    tifffile.imwrite(paths["out_tif"], lm)
    doc = load_json_safe(paths["out_json"]) if os.path.exists(paths["out_json"]) \
        else load_json_safe(paths["autoseg_json"])
    if not isinstance(doc, dict):
        doc = {}
    doc.setdefault("reviews", []).append({
        "tool": "trace_mask_napari", "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "output": os.path.basename(paths["out_tif"]),
        "mask_voxels": int((lm > 0).sum()), "n_arcs": len(arcs),
        "islands_removed": [{"label": lb, "components": n, "voxels": v} for lb, n, v in rep],
        "min_island_voxels": min_island,
        "arc_lengths_vox": [int(len(a)) for a in arcs],
        "voxel_zyx_um": [float(v) for v in voxel],
        "params": params,
        "label_convention": "cell 1, class 2 (structure) -> label 2; anatomy assigned later by wrap_segments",
    })
    json.dump(doc, open(paths["out_json"], "w"), indent=2)
    return paths["out_tif"], paths["out_json"]


# ----------------------------------------------------------------------------- GUI
def launch(stack_path, voxel_cli=None, channel="cofire_mean", alpha=0.5, radius_x=1.5, pad=0,
           dim_pct=15.0, ndisplay=2, seed="reference", seed_z=5.0, resume=True):
    import napari

    paths = derive_paths(stack_path)
    voxel = resolve_voxel(stack_path, voxel_cli)
    saved = last_saved_params(paths) if resume else None
    if saved:
        alpha = float(saved.get("alpha", alpha)); radius_x = float(saved.get("radius_x", radius_x))
        pad = int(saved.get("pad", pad)); channel = saved.get("reference_channel", channel)
        print(f"[trace] restoring saved settings: alpha {alpha:.2f}, radius x {radius_x:.2f}, pad {pad}, "
              f"channel {channel}")
    ref, names, ci, ref_all = load_reference(paths, channel)
    def do_seed(r):
        if seed == "none": return []
        return seed_from_autoseg(paths, r.shape) if seed == "autoseg" else seed_from_reference(r, z=seed_z, voxel=voxel)
    arcs = do_seed(ref)
    print(f"[trace] reference {ref.shape} channel={names[ci]}; seeded {len(arcs)} arcs from {seed}")
    S = {"arcs": arcs, "owners": ["own"] * len(arcs), "hist": [], "alpha": alpha, "rx": radius_x,
         "pad": pad, "cache": None, "ci": ci, "cost": cost_volume(ref), "ref": ref, "pending": None}
    sess = load_session(paths, ref.shape) if resume else None
    S["erase0"] = None; S["add0"] = None
    if sess is not None:
        S["arcs"], S["owners"], S["erase0"], S["add0"] = sess
        print(f"[trace] resumed saved session: {len(S['arcs'])} arcs "
              f"({S['owners'].count('intruder')} other-cell), {int(S['erase0'].sum())} erased, "
              f"{int(S['add0'].sum())} added voxels")
    elif resume and os.path.exists(paths["out_tif"]):
        # a mask saved before sessions existed: rebuild a session that reproduces it EXACTLY.
        # centerline = skeleton of the saved mask; the voxels that growth would add or miss
        # become erase / add strokes, so what you see on opening is what you saved.
        saved_mask = tifffile.imread(paths["out_tif"]) > 0
        if saved_mask.shape == ref.shape and saved_mask.any():
            arcs0 = _skeleton_arcs(saved_mask, 3, prune_radius=voxel) or _skeleton_arcs(saved_mask, 1)
            owners0 = ["own"] * len(arcs0)
            excl0 = (tifffile.imread(paths["exclude_tif"]) > 0) if os.path.exists(paths["exclude_tif"]) else None
            g_own, _, _ = grow_owned(ref, arcs0, owners0, voxel, alpha, radius_x, pad)
            S["arcs"], S["owners"] = arcs0, owners0
            S["erase0"] = g_own & ~saved_mask
            S["add0"] = saved_mask & ~g_own
            print(f"[trace] no saved session: rebuilt one from your saved mask "
                  f"({int(saved_mask.sum())} vox, {len(arcs0)} arcs; {int(S['add0'].sum())} vox kept by 'add', "
                  f"{int(S['erase0'].sum())} by 'erase')")
    prev_excl = (tifffile.imread(paths["exclude_tif"]) > 0) if os.path.exists(paths["exclude_tif"]) else None
    if resume and sess is None and prev_excl is not None and prev_excl.shape == ref.shape and prev_excl.any():
        # arcs lying mostly inside a previously saved exclusion start as intruder arcs
        for k, a in enumerate(S["arcs"]):
            if prev_excl[tuple(a.T)].mean() > 0.5:
                S["owners"][k] = "intruder"
        print(f"[trace] loaded previous exclusion ({int(prev_excl.sum())} vox); "
              f"{S['owners'].count('intruder')} seed arcs start as intruder")

    v = napari.Viewer(title=f"trace mask - {os.path.basename(stack_path)}", ndisplay=ndisplay)
    ref_layer = v.add_image(ref, name=f"reference [{names[ci]}]", scale=voxel, colormap="gray",
                            contrast_limits=(0, 1))
    mask_layer = v.add_labels(np.zeros(ref.shape, np.uint8), name="mask", scale=voxel, opacity=0.45)
    erase_layer = v.add_labels(np.zeros(ref.shape, np.uint8), name="erase (paint here to remove)",
                               scale=voxel, opacity=0.6)
    erase_layer.colormap = {None: (0, 0, 0, 0), 1: (1.0, 0.2, 0.2, 1.0)}   # red = erased
    erase_layer.brush_size = 2; erase_layer.selected_label = 1; erase_layer.n_edit_dimensions = 3
    if S.get("erase0") is not None:
        erase_layer.data = S["erase0"].astype(np.uint8)
    add_layer = v.add_labels(np.zeros(ref.shape, np.uint8), name="add (paint here to include)",
                             scale=voxel, opacity=0.6)
    add_layer.colormap = {None: (0, 0, 0, 0), 1: (0.2, 1.0, 0.3, 1.0)}      # green = added
    add_layer.brush_size = 2; add_layer.selected_label = 1; add_layer.n_edit_dimensions = 3
    if S.get("add0") is not None:
        add_layer.data = S["add0"].astype(np.uint8)
    int_layer = v.add_labels(np.zeros(ref.shape, np.uint8), name="other cell (excluded)",
                             scale=voxel, opacity=0.5)
    int_layer.colormap = {None: (0, 0, 0, 0), 1: (1.0, 0.1, 0.9, 1.0)}      # magenta
    cl_layer = v.add_labels(np.zeros(ref.shape, np.int32), name="centerline", scale=voxel, opacity=1.0)
    unc_layer = v.add_labels(np.zeros(ref.shape, np.uint8), name="uncertain (dim centerline)",
                             scale=voxel, opacity=1.0)
    unc_layer.colormap = {None: (0, 0, 0, 0), 1: (1.0, 0.55, 0.0, 1.0)}   # orange; None = default (transparent)
    pts_layer = v.add_points(np.zeros((0, 3)), name="trace clicks", scale=voxel, size=2,
                             face_color="cyan")

    # ---- dock: shared action panel (buttons mirror the keys)
    P = ActionPanel(v, title="trace mask")
    S["mode"] = None                                   # sticky click mode: None | "trace" | "delete"
    class _Status:                                     # keep the old status.setText() call sites
        def setText(self, t): P.status(t)
    status = _Status()

    def set_mode(m):
        if S.get("erasing"):
            set_erase(False)
        if S.get("adding"):
            set_add(False)
        S["mode"] = None if S["mode"] == m else m
        S["pending"] = None; pts_layer.data = np.zeros((0, 3))
        P.set_toggle("t", S["mode"] == "trace"); P.set_toggle("x", S["mode"] == "delete")
        P.set_toggle("i", S["mode"] == "intruder"); P.set_toggle("f", S["mode"] == "flip")
        P.hint({"trace": "TRACE: click the FIRST point of the new arc",
                "delete": "DELETE: click an arc to remove it",
                "intruder": "OTHER CELL: click two points along the crossing cell (it turns magenta)",
                "flip": "FLIP: click an arc to switch it between your cell and the other cell",
                None: "Drag the sliders until the halo is gone; trace missing branches with [t]"}[S["mode"]])

    P.section("1. thickness (live)")
    P.slider("alpha - relative threshold", 5, 95, alpha, 100,
             lambda val: (S.__setitem__("alpha", val), regrow_soon()))
    P.note("higher alpha = thinner mask: keeps voxels brighter than alpha x the local centerline")
    P.slider("radius x local", 50, 400, radius_x, 100, lambda val: (S.__setitem__("rx", val), regrow_soon()))
    P.slider("pad (voxels)", 0, 3, pad, 1, lambda val: (S.__setitem__("pad", val), regrow_soon()), fmt="{:.0f}")
    P.section("2. edit the centerline")
    P.button("Trace arc between 2 clicks", key="t", cb=lambda: set_mode("trace"), toggle=True,
             tooltip="Click two points; the brightest path between them becomes a centerline arc")
    P.button("Delete arc under click", key="x", cb=lambda: set_mode("delete"), toggle=True)
    P.button("Undo", key="u", cb=lambda: undo())
    P.button("Re-seed centerline", key="r", cb=lambda: reseed())
    P.section("2a. other cells (blacked out in movies)")
    P.button("Trace other cell (2 clicks)", key="i", cb=lambda: set_mode("intruder"), toggle=True,
             tooltip="Trace a crossing/neighbouring cell; its voxels leave your mask and are blacked out in movies")
    P.button("Flip arc: mine <-> other", key="f", cb=lambda: set_mode("flip"), toggle=True,
             tooltip="Click an arc to move it between your cell (cyan) and the other cell (magenta)")
    P.section("2b. erase by hand")
    P.button("Erase with brush", key="e", cb=lambda: set_erase(not S.get("erasing", False)), toggle=True,
             tooltip="Paint on the red layer; those voxels are removed from the mask and stay removed")
    P.button("Clear all erasures", cb=lambda: clear_erase())
    P.button("Add with brush", key="a", cb=lambda: set_add(not S.get("adding", False)), toggle=True,
             tooltip="Paint on the green layer; those voxels are forced into your mask and stay in")
    P.section("3. view")
    P.button("Next reference channel", key="c", cb=lambda: cycle_channel())
    P.button("2D / 3D", key="d", cb=lambda: toggle_dims())
    P.note("Orange = centerline where the reference is dim (uncertain). Decide by eye; nothing is auto-bridged.")
    P.section("4. done")
    btn = P.button("Save mask", key="Ctrl+S")
    P.finish()

    # ---- core updates
    from qtpy.QtCore import QTimer
    _timer = QTimer(); _timer.setSingleShot(True); _timer.setInterval(60)
    _timer.timeout.connect(lambda: regrow())

    def rebuild_cache():
        # expensive geodesic maps: only when arcs, owners or the reference change
        S["caches"] = owner_caches(S["ref"], S["arcs"], S["owners"], voxel) if S["arcs"] else (None, None)
        cl = np.zeros(ref.shape, np.int32)
        for k, pts in enumerate(S["arcs"], start=1):
            cl[tuple(pts.T)] = k
        cl_layer.data = cl
        cmap = {None: (0, 0, 0, 0)}
        for k, o in enumerate(S["owners"], start=1):
            cmap[k] = (1.0, 0.1, 0.9, 1.0) if o == "intruder" else (0.1, 0.9, 1.0, 1.0)
        cl_layer.colormap = cmap

    def regrow_soon():
        _timer.start()                                      # restarts: only the last request runs

    def regrow():
        _timer.stop()
        if not S["arcs"]:
            mask_layer.data = np.zeros(ref.shape, np.uint8); unc_layer.data = np.zeros(ref.shape, np.uint8)
            int_layer.data = np.zeros(ref.shape, np.uint8)
            status.setText("no centerline - use Trace and click two points"); return
        er = np.asarray(erase_layer.data) > 0
        ad = np.asarray(add_layer.data) > 0
        m, mi, unc = grow_owned(S["ref"], S["arcs"], S["owners"], voxel, S["alpha"], S["rx"],
                                S["pad"], dim_pct, erase=er, add=ad, caches=S.get("caches"))
        mask_layer.data = m.astype(np.uint8); unc_layer.data = unc.astype(np.uint8)
        int_layer.data = mi.astype(np.uint8)
        n_int = S["owners"].count("intruder")
        status.setText(f"{len(S['arcs']) - n_int} arcs | mask {int(m.sum()):,} vox | "
                       f"uncertain centerline pts: {int(unc.sum())}"
                       + (f"\nother cell: {n_int} arcs, {int(mi.sum()):,} vox (excluded)" if n_int else ""))

    def push_hist():
        S["hist"].append(([a.copy() for a in S["arcs"]], list(S["owners"])))
        if len(S["hist"]) > 30: S["hist"].pop(0)

    def world_to_vox(pos):
        return tuple(int(round(p / s_)) for p, s_ in zip(pos[-3:], voxel))

    # ---- mouse: t+click trace, x+click delete
    def on_click(layer, event):
        mods = set(event.modifiers) if event.modifiers else set()
        held = set(S.get("held", set()))
        if S.get("mode") in ("trace", "intruder"): held.add("t")
        if S.get("mode") in ("delete", "flip"): held.add("x")
        pos = world_to_vox(v.cursor.position)
        if not all(0 <= p < n for p, n in zip(pos, ref.shape)):
            return
        if "x" in held:
            cl = cl_layer.data
            if cl[pos] == 0:                                # snap to nearest centerline voxel
                pts = np.argwhere(cl > 0)
                if len(pts) == 0: return
                w = np.asarray(voxel, float).copy()
                if v.dims.ndisplay == 2:
                    w[0] = 0.0                              # 2D view: ignore depth, pick by XY
                d = (((pts - np.array(pos)) * w) ** 2).sum(1)
                pos = tuple(pts[d.argmin()])
            k = int(cl[pos]) - 1
            if 0 <= k < len(S["arcs"]):
                push_hist()
                if S.get("mode") == "flip":
                    S["owners"][k] = "own" if S["owners"][k] == "intruder" else "intruder"
                else:
                    S["arcs"].pop(k); S["owners"].pop(k)
                rebuild_cache(); regrow()
        elif "t" in held:
            # snap the click to the brightest voxel within +-1 in Y/X and, in 2D slice
            # view, through the WHOLE Z column: you click on the top view and the path
            # is found in 3D, so a connection never has to be drawn slice by slice.
            # (In 3D view the click already carries a depth, so only +-1 in Z.)
            z, y, x = pos
            zs = slice(0, S["ref"].shape[0]) if v.dims.ndisplay == 2 else slice(max(0, z-1), z+2)
            sl = (zs, slice(max(0, y-1), y+2), slice(max(0, x-1), x+2))
            loc = np.unravel_index(np.argmax(S["ref"][sl]), S["ref"][sl].shape)
            pos = tuple(int(sl[i].start + loc[i]) for i in range(3))
            if S["pending"] is None:
                S["pending"] = pos; pts_layer.data = np.array([pos]); P.hint("TRACE: now click the SECOND point")
            else:
                path = geodesic_path(S["cost"], S["pending"], pos, voxel)
                S["pending"] = None; pts_layer.data = np.zeros((0, 3))
                if path is None or len(path) < 2:
                    status.setText("no path found"); return
                push_hist(); S["arcs"].append(path)
                S["owners"].append("intruder" if S.get("mode") == "intruder" else "own")
                rebuild_cache(); regrow()
                P.hint("TRACE: click the FIRST point of the next arc (or click the button again to stop)")
    for lyr in (ref_layer, mask_layer, cl_layer, unc_layer, int_layer):
        lyr.mouse_drag_callbacks.append(on_click)

    @erase_layer.mouse_drag_callbacks.append
    def _after_erase(layer, event):
        yield
        while event.type == "mouse_move":
            yield
        regrow()                                            # stroke finished -> subtract it

    # any other change to the erase layer (fill tool, undo, programmatic) also re-applies
    erase_layer.events.data.connect(lambda e: regrow_soon())

    @add_layer.mouse_drag_callbacks.append
    def _after_add(layer, event):
        yield
        while event.type == "mouse_move":
            yield
        regrow()
    add_layer.events.data.connect(lambda e: regrow_soon())

    def set_add(on):
        S["adding"] = on
        if on:
            if S.get("erasing"): set_erase(False)
            S["mode"] = None; P.set_toggle("t", False); P.set_toggle("x", False)
            v.layers.selection.active = add_layer; add_layer.mode = "paint"
            P.hint("ADD: paint voxels to include them in your mask (brush size: [ ]). Kept across slider changes.")
        else:
            add_layer.mode = "pan_zoom"; v.layers.selection.active = mask_layer
            P.hint("Drag the sliders until the halo is gone; trace missing branches with [t]")
        P.set_toggle("a", on)

    def set_erase(on):
        """Erase mode: select the erase layer with the brush; off: back to pan/zoom."""
        S["erasing"] = on
        if on and S.get("adding"):
            set_add(False)
        if on:
            S["mode"] = None; P.set_toggle("t", False); P.set_toggle("x", False)
            v.layers.selection.active = erase_layer; erase_layer.mode = "paint"
            P.hint("ERASE: paint over voxels to remove them (brush size: [ ]). Erasures survive slider changes.")
        else:
            erase_layer.mode = "pan_zoom"; v.layers.selection.active = mask_layer
            P.hint("Drag the sliders until the halo is gone; trace missing branches with [t]")
        P.set_toggle("e", on)
    def clear_erase():
        erase_layer.data = np.zeros(ref.shape, np.uint8); regrow()

    # napari key events: track held keys
    S["held"] = set()
    def held(key):
        def f(viewer):
            S["held"].add(key); yield; S["held"].discard(key)
        return f
    v.bind_key("t", held("t"), overwrite=True)
    v.bind_key("x", held("x"), overwrite=True)

    def undo(viewer=None):
        if S["hist"]:
            S["arcs"], S["owners"] = S["hist"].pop(); rebuild_cache(); regrow()
    def reseed(viewer=None):
        push_hist(); S["arcs"] = do_seed(S["ref"]); S["owners"] = ["own"] * len(S["arcs"])
        rebuild_cache(); regrow()
    def cycle_channel(viewer=None):
        S["ci"] = (S["ci"] + 1) % ref_all.shape[1]
        vol = ref_all[:, S["ci"]]; lo, hi = np.percentile(vol, [1, 99.9])
        S["ref"] = np.clip((vol - lo) / (hi - lo + 1e-9), 0, 1); S["cost"] = cost_volume(S["ref"])
        ref_layer.data = S["ref"]; ref_layer.name = f"reference [{names[S['ci']]}]"
        rebuild_cache(); regrow()
    def toggle_dims(viewer=None):
        v.dims.ndisplay = 3 if v.dims.ndisplay == 2 else 2
    def do_save(viewer=None):
        regrow()                                            # mask on disk == erasures applied now
        m = mask_layer.data > 0
        if not m.any():
            status.setText("nothing to save"); return
        params = {"alpha": S["alpha"], "radius_x": S["rx"], "pad": S["pad"], "dim_pct": dim_pct,
                  "reference_channel": names[S["ci"]],
                  "manually_erased_voxels": int((np.asarray(erase_layer.data) > 0).sum()),
                  "manually_added_voxels": int((np.asarray(add_layer.data) > 0).sum())}
        own_arcs = [a for a, o in zip(S["arcs"], S["owners"]) if o != "intruder"]
        params["n_other_cell_arcs"] = S["owners"].count("intruder")
        t, j = save(paths, m, own_arcs, params, voxel)
        save_session(paths, S["arcs"], S["owners"], np.asarray(erase_layer.data) > 0,
                     np.asarray(add_layer.data) > 0)
        mi = np.asarray(int_layer.data) > 0
        extra = ""
        if mi.any() or os.path.exists(paths["exclude_tif"]):
            et, nx = save_exclude(paths, mi, S["owners"], S["arcs"], voxel)
            extra = f"\nother cell: {nx:,} vox -> {os.path.basename(et)}"
            print(f"[trace] exclusion saved {et} ({nx} vox)")
        status.setText(f"saved {os.path.basename(t)}  ({int(m.sum()):,} vox){extra}\n"
                       "close napari, then Refresh the panel")
        print(f"[trace] saved {t}\n[trace] record appended to {j}")
    v.bind_key("u", undo, overwrite=True); v.bind_key("r", reseed, overwrite=True)
    v.bind_key("c", cycle_channel, overwrite=True); v.bind_key("d", toggle_dims, overwrite=True)
    v.bind_key("e", lambda vw: set_erase(not S.get("erasing", False)), overwrite=True)
    v.bind_key("a", lambda vw: set_add(not S.get("adding", False)), overwrite=True)
    v.bind_key("i", lambda vw: set_mode("intruder"), overwrite=True)
    v.bind_key("f", lambda vw: set_mode("flip"), overwrite=True)
    v.bind_key("Control-s", do_save, overwrite=True); btn.clicked.connect(lambda: do_save())

    rebuild_cache(); regrow(); set_mode(None)
    napari.run()


# ----------------------------------------------------------------------------- check
def run_check(stack_path, voxel_cli=None) -> bool:
    """Headless self-test on a real run: seed, grow at three alphas, trace one geodesic
    path, save to a temp dir with the exact contract, verify monotonic thinning."""
    import tempfile, shutil
    ok = True
    def rep(name, cond, extra=""):
        nonlocal ok; ok &= bool(cond); print(f"  {'PASS' if cond else 'FAIL'}: {name} {extra}")
    paths = derive_paths(stack_path)
    voxel = resolve_voxel(stack_path, voxel_cli)
    ref, names, ci, _ = load_reference(paths)
    rep("reference loaded", ref.ndim == 3, f"{ref.shape} ch={names[ci]}")
    arcs = seed_from_reference(ref, voxel=voxel)
    rep("seeded arcs from reference ridges", 3 <= len(arcs) <= 60, f"n={len(arcs)}")
    cache = grow_cache(ref, arcs, voxel)
    sizes = []
    for a in (0.3, 0.5, 0.7):
        m, unc = grow(cache, alpha=a, radius_x=1.5, pad=0); sizes.append(int(m.sum()))
    rep("alpha thins the mask monotonically", sizes[0] > sizes[1] > sizes[2], f"{sizes}")
    m5, _ = grow(cache, 0.5, 1.5, 0); m5b, _ = grow(cache, 0.5, 3.0, 0)
    rep("radius cap grows the mask", m5b.sum() >= m5.sum(), f"{int(m5.sum())} -> {int(m5b.sum())}")
    cl = cache["cl"] > 0
    rep("centerline always inside mask", bool((m5 & cl).sum() == cl.sum()))
    # one geodesic path between the two furthest centerline voxels of the longest arc
    longest = max(arcs, key=len); a, b = longest[0], longest[-1]
    path = geodesic_path(cost_volume(ref), a, b, voxel)
    rep("geodesic path found between arc ends", path is not None and len(path) >= 2,
        f"len={0 if path is None else len(path)}")
    if path is not None:
        rep("path stays on bright ridge", float(np.median(ref[tuple(path.T)])) > float(np.median(ref)),
            f"median on-path {np.median(ref[tuple(path.T)]):.2f} vs volume {np.median(ref):.2f}")
    # other-cell exclusion: flag the arc least like the longest one as an intruder
    if len(arcs) >= 3:
        own_mask_all, _ = grow(cache, 0.5, 1.5, 0)
        owners = ["own"] * len(arcs)
        k_int = int(np.argsort([len(a) for a in arcs])[-2])      # a real, long arc
        owners[k_int] = "intruder"
        mo, mi, _ = grow_owned(ref, arcs, owners, voxel, 0.5, 1.5, 0)
        rep("owner-aware growth: own and other cell disjoint", not (mo & mi).any(),
            f"own {int(mo.sum())} / other {int(mi.sum())} vox")
        rep("intruder arc voxels end up in the other-cell mask", bool(mi[tuple(arcs[k_int].T)].mean() > 0.9))
        own_cl = np.zeros(ref.shape, bool)
        for a, o in zip(arcs, owners):
            if o == "own": own_cl[tuple(a.T)] = True
        rep("own centerline stays in own mask", bool(mo[own_cl].all()))
        rep("excluding shrinks own mask", int(mo.sum()) < int(own_mask_all.sum()),
            f"{int(own_mask_all.sum())} -> {int(mo.sum())}")
    # save contract into a temp copy so the run dir is untouched
    tmp = tempfile.mkdtemp()
    try:
        tp = {k: (os.path.join(tmp, os.path.basename(v_)) if k in ("out_tif", "out_json") else v_)
              for k, v_ in paths.items()}
        t, j = save(tp, m5, arcs, {"alpha": 0.5}, voxel)
        lm = tifffile.imread(t)
        rep("saved labelmap uint8 with label 2 only", lm.dtype == np.uint8 and set(np.unique(lm)) <= {0, 2})
        rep("reviewed json has a review record", "reviews" in json.load(open(j)))
        if len(arcs) >= 3:
            tp["exclude_tif"] = os.path.join(tmp, os.path.basename(paths["exclude_tif"]))
            et, nx = save_exclude(tp, mi, owners, arcs, voxel)
            ex = tifffile.imread(et)
            # session round trip: reopen must reproduce the exact own + other-cell masks
            tp["session"] = os.path.join(tmp, os.path.basename(paths["session"]))
            er = np.zeros(ref.shape, bool); er[tuple(np.argwhere(mo)[:25].T)] = True
            save_session(tp, arcs, owners, er)
            sess = load_session(tp, ref.shape)
            ok_rt = sess is not None and len(sess[0]) == len(arcs) and sess[1] == owners \
                and all(np.array_equal(a, b) for a, b in zip(sess[0], arcs)) and np.array_equal(sess[2], er)
            mo2, mi2, _ = grow_owned(ref, sess[0], sess[1], voxel, 0.5, 1.5, 0, erase=sess[2]) if sess else (None, None, None)
            mo1, mi1, _ = grow_owned(ref, arcs, owners, voxel, 0.5, 1.5, 0, erase=er)
            rep("session round trip restores arcs, owners, erasures and identical masks",
                ok_rt and np.array_equal(mo1, mo2) and np.array_equal(mi1, mi2),
                f"{len(arcs)} arcs, other-cell {int(mi2.sum()) if mi2 is not None else '?'} vox")
            # corrupt JSON is preserved, not silently reset
            bad = os.path.join(tmp, "bad.json"); open(bad, "w").write("{not json")
            d0 = load_json_safe(bad)
            rep("corrupt JSON kept as a copy, not silently reset",
                d0 == {} and any(f.startswith("bad.json.corrupt-") for f in os.listdir(tmp)))
            rep("exclude labelmap written (uint8, 0/1) + JSON block",
                ex.dtype == np.uint8 and set(np.unique(ex)) <= {0, 1} and "exclude" in json.load(open(j)),
                f"{nx} vox")
        if os.path.exists(paths["autoseg"]):
            rep("original autoseg untouched",
                not os.path.exists(os.path.join(tmp, os.path.basename(paths["autoseg"]))))
        else:
            rep("no autoseg present - tool runs from the reference alone", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n==== --check {'PASS' if ok else 'FAIL'} ====")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", help="<stem>.tif cleaned stack; ref3d + autoseg looked up next to it")
    add_voxel_arg(ap)
    ap.add_argument("--channel", default="cofire_mean", help="ref3d channel to trace on")
    ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--radius-x", type=float, default=1.5)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--dim-pct", type=float, default=15.0,
                    help="centerline points below this percentile of centerline intensity are flagged uncertain")
    ap.add_argument("--ndisplay", type=int, choices=(2, 3), default=2)
    ap.add_argument("--seed", choices=("reference", "autoseg", "none"), default="reference",
                    help="initial centerline: bright ridges of the reference (default), the "
                         "autoseg skeleton, or nothing (trace everything by hand)")
    ap.add_argument("--seed-z", type=float, default=5.0,
                    help="robust-z threshold for reference seeding (higher = fewer, surer arcs)")
    ap.add_argument("--fresh", action="store_true",
                    help="start clean from the seed: ignore the saved session and the saved "
                         "other-cell exclusion (files on disk are only replaced when you save)")
    ap.add_argument("--check", action="store_true", help="headless self-test; writes nothing into the run dir")
    args = ap.parse_args(argv)
    if args.check:
        return 0 if run_check(args.stack, args.voxel) else 1
    launch(args.stack, args.voxel, args.channel, args.alpha, args.radius_x, args.pad, args.dim_pct,
           args.ndisplay, args.seed, args.seed_z, resume=not args.fresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
