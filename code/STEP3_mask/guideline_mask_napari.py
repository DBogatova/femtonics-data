#!/usr/bin/env python
"""
guideline_mask_napari.py - Draw the dendrite path on the MIP in napari; build a
thin 3D mask = bright structure within a corridor around your line.

Why: some branches fire only in certain events and never threshold cleanly from
one frame. Tracing them by hand on the temporal-max MIP lets you include exactly
what you can see, while the corridor keeps the mask thin and on-structure.

Workflow
--------
  python code/STEP3_mask/guideline_mask_napari.py run_clean.tif --voxel 0.8 0.9 0.9
  * a napari window opens with the Z-MIP and an (empty) 'guideline' Shapes layer
  * select 'guideline', use the PATH tool, click along the dendrite (incl. the
    faint branches you can see); add multiple paths for branches
  * press 'b' to build the mask (repeat after editing), Ctrl+S to save
  --radius controls corridor half-width (thinness); --thr-pct the structure threshold
"""
import argparse
import numpy as np
import tifffile


def highpass(vol, small=(0.5, 1, 1), big=(2, 6, 6)):
    from scipy.ndimage import gaussian_filter
    e = gaussian_filter(vol, small) - gaussian_filter(vol, big)
    e[e < 0] = 0
    return e


def build_corridor(paths, shape_yx, radius):
    from skimage.draw import line as skline
    from scipy.ndimage import binary_dilation
    Y, X = shape_yx
    corr = np.zeros((Y, X), bool)
    for p in paths:
        pts = np.round(np.asarray(p)[:, -2:]).astype(int)
        for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
            rr, cc = skline(int(y0), int(x0), int(y1), int(x1))
            rr = np.clip(rr, 0, Y - 1); cc = np.clip(cc, 0, X - 1)
            corr[rr, cc] = True
    if radius > 0:
        corr = binary_dilation(corr, iterations=int(radius))
    return corr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", help="4D (T,Z,Y,X) or 3D (Z,Y,X) stack")
    ap.add_argument("--voxel", nargs=3, type=float, default=[1.0, 1.0, 1.0], metavar=("Z", "Y", "X"))
    ap.add_argument("--agg", choices=["max", "mean"], default="max")
    ap.add_argument("--radius", type=int, default=5, help="corridor half-width in XY px (thinness)")
    ap.add_argument("--thr-pct", type=float, default=80.0, help="structure high-pass percentile")
    ap.add_argument("--close", type=int, default=2, help="3D closing iterations to fill gaps in the mask")
    ap.add_argument("--enhance", type=float, default=0.7,
                    help="cell (vesselness) enhancement added to the mean-MIP background (0 = plain mean)")
    ap.add_argument("--min-vox", type=int, default=200, help="drop components smaller than this")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import napari
    from skimage.measure import label

    stack = tifffile.imread(args.stack)
    if stack.ndim == 4:
        vmax = stack.max(0).astype(np.float32)
        vmean = stack.mean(0).astype(np.float32)
    else:
        vmax = vmean = stack.astype(np.float32)
    vol = vmax                                        # structure detection from temporal max
    Z, Y, X = vol.shape
    hp = highpass(vol)
    bright = hp > np.percentile(hp, args.thr_pct)     # (Z,Y,X)
    out = args.out or args.stack.rsplit(".", 1)[0] + "_guided_labelmap.tif"

    def znorm(a):
        lo, hi = np.percentile(a, (2, 99.7))
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    from skimage.filters import sato
    vess = sato(highpass(vmean), sigmas=[0.5, 1, 2, 3], black_ridges=False)  # cell only, ~0 in bg
    mip_max = znorm(vmax.max(0))                       # active branches
    mip_mean = znorm(vmean.max(0))                     # soma/trunk baseline anatomy
    mip_vess = znorm(vess.max(0))                      # tubular enhancement (no background)
    enhanced = znorm(mip_mean + args.enhance * mip_vess)   # cell enhanced, background untouched

    v = napari.Viewer()
    v.add_image(enhanced, name="anatomy (enhanced)", colormap="gray")
    v.add_image(mip_mean, name="soma/trunk (mean)", colormap="gray", visible=False)
    v.add_image(mip_max, name="activity (max)", colormap="gray", visible=False)
    shp = v.add_shapes(name="guideline", shape_type="path", edge_color="cyan", edge_width=1)
    state = {"mask3d": None}
    print("Draw path(s) along the dendrite on the MIP; press 'b' to build, Ctrl+S to save.")

    @v.bind_key("b")
    def _build(viewer):
        if len(shp.data) == 0:
            print("draw a path first"); return
        corr = build_corridor(shp.data, (Y, X), args.radius)
        mask = bright & corr[None, :, :]
        if args.close > 1:
            from scipy.ndimage import binary_closing, generate_binary_structure
            mask = binary_closing(mask, generate_binary_structure(3, 2), iterations=args.close)
            mask = mask & corr[None, :, :]
        lbl = label(mask)
        sizes = np.bincount(lbl.ravel())            # drop small components (no deprecation)
        keep = np.where(sizes >= args.min_vox)[0]; keep = keep[keep > 0]
        lbl = label(np.isin(lbl, keep))
        state["mask3d"] = lbl.astype(np.uint16)
        disp = lbl.max(0).astype(np.uint16)         # 2D Z-MIP footprint, aligned with the MIP
        if "dendrite mask" in viewer.layers:
            viewer.layers["dendrite mask"].data = disp
        else:
            viewer.add_labels(disp, name="dendrite mask", opacity=0.5)
        print(f"built mask: {int((state['mask3d']>0).sum())} voxels, "
              f"{int(state['mask3d'].max())} components (showing Z-MIP footprint)")

    @v.bind_key("Control-s")
    def _save(viewer):
        if state["mask3d"] is not None:
            tifffile.imwrite(out, state["mask3d"])
            print(f"saved -> {out}")
        else:
            print("build a mask first (press 'b')")

    napari.run()


if __name__ == "__main__":
    main()
