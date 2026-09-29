"""Display masking: show only the cell you analyzed, on a black background.

For MOVIES AND FIGURES ONLY. Traces, events and every number in the analysis are
computed from the unmodified stack and never pass through this module.

weight(z, y, x) in [0, 1]:
  * 1 inside the own-cell mask (the reviewed mask, or the region labelmap);
  * a smooth falloff to 0 over `edge_um` micrometers outside it (cosine ramp on the
    anisotropic distance), so the cell keeps its natural soft edge instead of a
    cut-out look; edge_um = 0 gives a hard cut;
  * exactly 0 on voxels of other cells (<stem>_exclude_labelmap.tif), even if they
    fall inside the falloff band, so a crossing cell is blacked out completely.

Multiply any volume (T,Z,Y,X or Z,Y,X) by the weight BEFORE projecting (MIP /
rotation), so haze and neighbours cannot win the maximum projection.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi


def find_masks(stack_path):
    """Paths next to <stem>.tif: own-cell mask (reviewed, else regions) and exclusion.
    Returns (own_path or None, exclude_path or None)."""
    stack_path = Path(stack_path)
    stem = stack_path.with_suffix("")
    own = None
    for cand in (f"{stem}_autoseg_labelmap_reviewed.tif", f"{stem}_segments_final.tif"):
        if Path(cand).exists():
            own = Path(cand); break
    excl = Path(f"{stem}_exclude_labelmap.tif")
    return own, (excl if excl.exists() else None)


def display_weight(own_mask: np.ndarray, voxel, edge_um: float = 2.0,
                   exclude: np.ndarray | None = None, extra_own: np.ndarray | None = None) -> np.ndarray:
    """float32 (Z,Y,X) weight: 1 in the cell, cosine falloff over edge_um outside it,
    0 on excluded voxels. `extra_own` (e.g. the region labelmap > 0) is unioned in so
    every displayed region is fully bright even if it extends past the reviewed mask."""
    own = np.asarray(own_mask) > 0
    if extra_own is not None:
        own = own | (np.asarray(extra_own) > 0)
    if not own.any():
        w = np.ones(own.shape, np.float32)
    elif edge_um <= 0:
        w = own.astype(np.float32)
    else:
        d = ndi.distance_transform_edt(~own, sampling=tuple(float(v) for v in voxel))
        t = np.clip(d / float(edge_um), 0.0, 1.0)
        w = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
        w[own] = 1.0
    if exclude is not None:
        ex = np.asarray(exclude) > 0
        w[ex & ~own] = 0.0            # an own voxel is never blacked out
    return w


def load_display_weight(stack_path, labelmap=None, voxel=None, edge_um: float = 2.0):
    """Build the weight for a run from the files next to its stack.
    Returns (weight or None, description string). None means: nothing to mask with."""
    import tifffile
    own_p, ex_p = find_masks(stack_path)
    seg = tifffile.imread(str(labelmap)) if labelmap is not None and Path(labelmap).exists() else None
    if own_p is None and seg is None:
        return None, "no mask found - display unmasked"
    own = tifffile.imread(str(own_p)) if own_p is not None else seg
    ex = tifffile.imread(str(ex_p)) if ex_p is not None else None
    if voxel is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from common.voxel import resolve_voxel
        voxel = resolve_voxel(stack_path, None, quiet=True)
    w = display_weight(own, voxel, edge_um, exclude=ex, extra_own=seg)
    desc = (f"display mask: {own_p.name if own_p else 'regions'}"
            f", edge {edge_um:g} um"
            + (f", other cell blacked out ({int((ex > 0).sum())} vox from {ex_p.name})" if ex is not None else ""))
    return w, desc


def options_record(mask: bool, edge_um: float, hide_other: bool = False) -> dict:
    return {"mask": bool(mask), "edge_um": round(float(edge_um), 3), "hide_other": bool(hide_other)}


def options_match(output_path, mask: bool, edge_um: float, hide_other: bool = False) -> bool:
    """True if <output>.display.json records the same options. A missing sidecar counts
    as the historical default (unmasked) so existing outputs are not all rebuilt."""
    import json
    side = Path(str(output_path) + ".display.json")
    rec = options_record(mask, edge_um, hide_other)
    if not side.exists():
        return rec == options_record(False, 2.0)
    try:
        old = json.loads(side.read_text()); old.setdefault("hide_other", False)
        return old == rec
    except Exception:
        return False


def write_options(output_path, mask: bool, edge_um: float, hide_other: bool = False):
    import json
    Path(str(output_path) + ".display.json").write_text(json.dumps(options_record(mask, edge_um, hide_other)))


# ----------------------------------------------------------------------------- other cells
def background_donors(exclude, own, window_vox: int = 20, margin_vox: int = 2, seed: int = 0):
    """For every voxel of another cell (plus a 1-voxel rim), pick a DONOR background voxel:
    same Z plane, within +-window_vox columns, at least margin_vox from both cells.
    Returns (targets (N,3), donors (N,3), rim_flag (N,) bool). Deterministic (seeded)."""
    ex = np.asarray(exclude) > 0
    own = np.asarray(own) > 0 if own is not None else np.zeros_like(ex)
    ex &= ~own
    if not ex.any():
        z = np.zeros((0, 3), int)
        return z, z, np.zeros(0, bool)
    st = np.ones((3, 3, 3), bool)
    rim = ndi.binary_dilation(ex, st) & ~ex & ~own
    far = ~ndi.binary_dilation(ex | own, st, iterations=margin_vox)
    targets = np.argwhere(ex | rim)
    rng = np.random.default_rng(seed)
    donors = np.empty_like(targets)
    Z, Y, X = ex.shape
    for z in np.unique(targets[:, 0]):
        cand = np.argwhere(far[z])                    # (y, x) background in this plane
        sel = np.flatnonzero(targets[:, 0] == z)
        if len(cand) == 0:                            # no background in the plane: any plane
            cand3 = np.argwhere(far)
            pick = cand3[rng.integers(0, len(cand3), len(sel))] if len(cand3) else targets[sel]
            donors[sel] = pick
            continue
        order = np.argsort(cand[:, 1]); cx = cand[order, 1]; cy = cand[order, 0]
        for i in sel:
            x0 = targets[i, 2]
            w = window_vox
            while True:
                lo, hi = np.searchsorted(cx, x0 - w), np.searchsorted(cx, x0 + w, side="right")
                if hi > lo or w > X:
                    break
                w *= 2
            k = rng.integers(lo, hi) if hi > lo else rng.integers(0, len(cx))
            donors[i] = (z, cy[k], cx[k])
    rim_flag = rim[tuple(targets.T)]
    return targets, donors, rim_flag


def fill_other_cells(vol, targets, donors, rim_flag):
    """In place: other-cell voxels take their donor's value; rim voxels are a 50/50 blend.
    vol is (Z,Y,X) or (T,Z,Y,X). Returns vol."""
    if len(targets) == 0:
        return vol
    t, d = tuple(targets.T), tuple(donors.T)
    if vol.ndim == 4:
        src = vol[(slice(None),) + d].astype(np.float32)
        cur = vol[(slice(None),) + t].astype(np.float32)
        new = np.where(rim_flag[None, :], 0.5 * cur + 0.5 * src, src)
        vol[(slice(None),) + t] = new.astype(vol.dtype)
    else:
        src = vol[d].astype(np.float32); cur = vol[t].astype(np.float32)
        vol[t] = np.where(rim_flag, 0.5 * cur + 0.5 * src, src).astype(vol.dtype)
    return vol


def load_other_cell_fill(stack_path, labelmap=None):
    """(targets, donors, rim) for the run's saved exclusion, or None if there is none."""
    import tifffile
    own_p, ex_p = find_masks(stack_path)
    if ex_p is None:
        return None
    ex = tifffile.imread(str(ex_p))
    own = tifffile.imread(str(own_p)) if own_p is not None else None
    if labelmap is not None and Path(labelmap).exists():
        seg = tifffile.imread(str(labelmap)) > 0
        own = seg if own is None else ((np.asarray(own) > 0) | seg)
    if not (ex > 0).any():
        return None
    return background_donors(ex, own)
