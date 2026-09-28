"""Remove tiny disconnected islands from masks and labelmaps before saving.

A 1-2 voxel speck floating away from the dendrite is never anatomy at 0.8 um
voxels: it is a noise voxel that passed a threshold, or a stray brush touch. Left in,
it becomes a region with a trace of pure noise, or a false 'branch' in a figure.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

_C26 = np.ones((3, 3, 3), bool)


def drop_small_islands(vol: np.ndarray, min_voxels: int = 20, per_label: bool = True):
    """Return (cleaned, report). Components (26-connected) smaller than `min_voxels`
    are zeroed. For a labelmap with per_label=True the test runs within each label, so
    a small piece of label 3 detached from the rest of label 3 is removed even if it
    touches label 2. report = list of (label, n_removed_components, n_removed_voxels)."""
    vol = np.asarray(vol)
    out = vol.copy()
    report = []
    labels = [int(v) for v in np.unique(vol) if v > 0] if per_label else [None]
    for lb in labels:
        sel = (vol == lb) if lb is not None else (vol > 0)
        cc, n = ndi.label(sel, structure=_C26)
        if n <= 1:
            continue
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        small = np.where((sizes > 0) & (sizes < min_voxels))[0]
        if len(small) == 0:
            continue
        kill = np.isin(cc, small)
        out[kill] = 0
        report.append((lb, int(len(small)), int(kill.sum())))
    return out, report


def describe(report) -> str:
    if not report:
        return "no small islands"
    return "; ".join(f"label {lb}: removed {n} island(s), {v} vox" if lb is not None
                     else f"removed {n} island(s), {v} vox" for lb, n, v in report)
