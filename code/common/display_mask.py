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
