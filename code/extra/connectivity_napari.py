#!/usr/bin/env python
"""
connectivity_napari.py - Reconstruct a 3D mask from a 4D (T,Z,Y,X) hyperstack and
label connected components, to judge whether structures are physically connected
(same cell) or separate (a piece of another cell) -- the napari equivalent of
Imaris "Surfaces / connected components".

Concept
-------
"Connected vs separate" = 3D connected-component labeling on a binary mask:
  * voxels in the same component (given the connectivity + gap tolerance) -> same
    label id / color  -> CONNECTED
  * voxels in different components                                       -> SEPARATE

Pipeline
--------
  1. collapse time (max projection by default) -> structural volume (Z,Y,X)
  2. light smoothing + threshold -> binary mask
  3. optional morphological closing to bridge small gaps (the "merge" knob)
  4. 3D connected components (skimage.measure.label)
  5. drop tiny specks by physical volume
  6. open napari: image + colored Labels layer (hover shows the component id)
     and save the labelmap TIFF

Run from an env that has napari installed (your apical-dendrites env):
    python connectivity_napari.py path/to/stack_4d.tif
    python connectivity_napari.py stack.tif --close 2 --connectivity 3 --threshold 98
    python connectivity_napari.py stack.tif --no-view   # just save the labelmap

Interpreting the result
-----------------------
* If two branches share ONE color -> they are connected at the chosen settings.
* If raising --close (gap bridging) or --connectivity merges them, the "gap" is
  near the resolution limit -> treat connection as uncertain.
* Z is highly anisotropic (3.9 um vs ~1 um in XY), so connections across Z are
  the least certain; check them in the 3D view by rotating.
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, binary_closing, generate_binary_structure
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


def collapse_time(stack, mode, t0, t1):
    """4D (T,Z,Y,X) -> 3D (Z,Y,X)."""
    if stack.ndim == 3:
        return stack.astype(np.float32)
    seg = stack[t0:t1] if (t0 is not None or t1 is not None) else stack
    if mode == "mean":
        return np.nanmean(seg, axis=0).astype(np.float32)
    if mode == "sum":
        return np.clip(seg, 0, None).sum(axis=0).astype(np.float32)
    return np.nanmax(seg, axis=0).astype(np.float32)   # default: temporal MIP


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiff", help="4D (T,Z,Y,X) or 3D (Z,Y,X) TIFF")
    ap.add_argument("--voxel", nargs=3, type=float, default=[3.9, 1.0, 1.2],
                    metavar=("Z", "Y", "X"), help="voxel size in um (default 3.9 1.0 1.2)")
    ap.add_argument("--time-mode", choices=["max", "mean", "sum"], default="max",
                    help="how to collapse time into a structural volume")
    ap.add_argument("--t0", type=int, default=None, help="start frame of time window")
    ap.add_argument("--t1", type=int, default=None, help="end frame of time window")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="Gaussian sigma (in voxels, XY) before thresholding")
    ap.add_argument("--threshold", type=float, default=99.0,
                    help="intensity percentile for the binary mask (lower = more inclusive)")
    ap.add_argument("--close", type=int, default=0,
                    help="iterations of 3D morphological closing to bridge gaps "
                         "(the merge knob; 0 = strict)")
    ap.add_argument("--connectivity", type=int, choices=[1, 2, 3], default=1,
                    help="3D connectivity: 1=faces, 2=+edges, 3=+corners (more merging)")
    ap.add_argument("--min-vol", type=float, default=2000.0,
                    help="drop components smaller than this volume (um^3)")
    ap.add_argument("--save", default=None, help="output labelmap TIFF path")
    ap.add_argument("--no-view", action="store_true", help="don't open napari, just save")
    args = ap.parse_args()

    stack = tifffile.imread(args.tiff)
    print(f"loaded {args.tiff}  shape={stack.shape} dtype={stack.dtype}")

    vol = collapse_time(stack, args.time_mode, args.t0, args.t1)   # (Z,Y,X)

    if args.smooth > 0:
        vol = gaussian_filter(vol, sigma=(0.5, args.smooth, args.smooth))

    thr = np.nanpercentile(vol, args.threshold)
    mask = vol > thr
    print(f"threshold @ p{args.threshold} = {thr:.4g} -> {mask.sum()} voxels")

    if args.close > 0:
        st = generate_binary_structure(3, args.connectivity)
        mask = binary_closing(mask, structure=st, iterations=args.close)

    # connected components in 3D
    labels = label(mask, connectivity=args.connectivity)

    # drop tiny specks by physical volume
    voxel_vol = float(np.prod(args.voxel))
    min_voxels = int(args.min_vol / voxel_vol)
    if min_voxels > 0:
        kept = remove_small_objects(labels, min_size=min_voxels)
        labels = label(kept > 0, connectivity=args.connectivity)

    props = sorted(regionprops(labels), key=lambda r: r.area, reverse=True)
    print(f"\n{len(props)} connected components (each = one color in napari):")
    for r in props[:25]:
        vox = int(r.area)
        print(f"  id {r.label:>3}: {vox:>8d} vox = {vox*voxel_vol:>10.0f} um^3   "
              f"bbox(z,y,x)={r.bbox}")

    out = args.save or str(Path(args.tiff).with_name(Path(args.tiff).stem + "_components.tif"))
    tifffile.imwrite(out, labels.astype(np.uint16))
    print(f"\nsaved labelmap -> {out}")

    if args.no_view:
        return

    import napari
    scale = tuple(args.voxel)
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(vol, name="structure (time-collapsed)", scale=scale,
                     colormap="gray", rendering="attenuated_mip",
                     contrast_limits=(float(np.percentile(vol, 5)),
                                      float(np.percentile(vol, 99.5))))
    viewer.add_labels(labels, name="connected components", scale=scale)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    print("\nIn napari: hover a structure to read its component id (status bar).\n"
          "Same id/color = connected. Use the Labels layer's paint/fill tools to\n"
          "manually merge or split, then File > Save Selected Layer to export.")
    napari.run()


if __name__ == "__main__":
    main()
