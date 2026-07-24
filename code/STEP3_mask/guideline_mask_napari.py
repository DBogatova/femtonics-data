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
  --proj-axis {z,y,x} sets the STARTING projection (z=XY top, y=XZ side, x=ZY end-on);
    switch live with a=cycle, j/k/l=XY/XZ/ZY. Each plane has its own 'trace' layer;
    'b' unions the corridors from every projection you drew on (extruded along each axis).
  --activity-pct N (e.g. 90) also pulls in corridor voxels that only fire briefly
    (transient max-mean), so episodically-active branches aren't lost. The magenta
    "transient branches" layer shows where those are, to help you trace them.
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
    ap.add_argument("--proj-axis", choices=["z", "y", "x"], default="z",
                    help="projection to trace on: z=XY (top), y=XZ (side), x=ZY (end-on). "
                         "The corridor is built in that plane and extruded along the projection axis.")
    ap.add_argument("--radius", type=int, default=5, help="corridor half-width in XY px (thinness)")
    ap.add_argument("--thr-pct", type=float, default=80.0, help="structure high-pass percentile")
    ap.add_argument("--activity-pct", type=float, default=0.0,
                    help="also include corridor voxels whose TRANSIENT activity (max-mean over time) "
                         "exceeds this percentile - catches branches that only fire briefly "
                         "(e.g. 90). 0 = off (structure only)")
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
    pa = {"z": 0, "y": 1, "x": 2}[args.proj_axis]     # projection axis (extrude along this)
    view = {"z": "XY (top)", "y": "XZ (side)", "x": "ZY (end-on)"}[args.proj_axis]
    plane_shape = tuple(s for i, s in enumerate((Z, Y, X)) if i != pa)
    hp = highpass(vol)
    bright = hp > np.percentile(hp, args.thr_pct)     # (Z,Y,X)
    transient = np.clip(vmax - vmean, 0, None)        # per-voxel episodic brightening
    hp_act = highpass(transient)                      # emphasizes thin transiently-firing branches
    out = args.out or args.stack.rsplit(".", 1)[0] + "_guided_labelmap.tif"

    def znorm(a):
        lo, hi = np.percentile(a, (2, 99.7))
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    from skimage.filters import sato
    from scipy.ndimage import binary_closing, generate_binary_structure
    vess3 = sato(highpass(vmean), sigmas=[0.5, 1, 2, 3], black_ridges=False)  # cell only, ~0 in bg

    def mip_set(a):                                    # znorm'd MIPs projected along axis a
        mm = znorm(vmean.max(a))
        return {"anatomy (enhanced)": znorm(mm + args.enhance * znorm(vess3.max(a))),
                "soma/trunk (mean)": mm,
                "activity (max)": znorm(vmax.max(a)),
                "transient branches (max-mean)": znorm(hp_act.max(a))}

    axname = {0: "XY (top)", 1: "XZ (side)", 2: "ZY (end-on)"}
    cur = {"pa": pa}
    ms = mip_set(pa)
    v = napari.Viewer()
    v.add_image(ms["anatomy (enhanced)"], name="anatomy (enhanced)", colormap="gray")
    v.add_image(ms["soma/trunk (mean)"], name="soma/trunk (mean)", colormap="gray", visible=False)
    v.add_image(ms["activity (max)"], name="activity (max)", colormap="gray", visible=False)
    v.add_image(ms["transient branches (max-mean)"], name="transient branches (max-mean)",
                colormap="magma", visible=True, blending="additive")
    shapes = {a: v.add_shapes(name=f"trace {['XY', 'XZ', 'ZY'][a]}", shape_type="path",
                              edge_color="cyan", edge_width=1, visible=(a == pa)) for a in (0, 1, 2)}
    state = {"mask3d": None}

    def set_proj(a):
        cur["pa"] = a % 3
        for nm, data in mip_set(cur["pa"]).items():
            v.layers[nm].data = data
        for k, layer in shapes.items():
            layer.visible = (k == cur["pa"])
        if state["mask3d"] is not None and "dendrite mask" in v.layers:
            v.layers["dendrite mask"].data = state["mask3d"].max(cur["pa"]).astype(np.uint16)
        v.layers.selection.active = shapes[cur["pa"]]
        v.reset_view()
        print(f"projection: {axname[cur['pa']]} (draw on the 'trace {['XY','XZ','ZY'][cur['pa']]}' layer)",
              flush=True)

    v.bind_key("a", lambda vw: set_proj(cur["pa"] + 1), overwrite=True)   # cycle projection
    v.bind_key("j", lambda vw: set_proj(0), overwrite=True)               # XY
    v.bind_key("k", lambda vw: set_proj(1), overwrite=True)               # XZ
    v.bind_key("l", lambda vw: set_proj(2), overwrite=True)               # ZY
    v.layers.selection.active = shapes[pa]
    print(f"projection: {axname[pa]}.  KEYS: a=cycle proj, j/k/l=XY/XZ/ZY, b=build, Ctrl+S=save.\n"
          "Trace branches on any projection (each plane has its own 'trace' layer); build unions them.",
          flush=True)

    @v.bind_key("b")
    def _build(viewer):
        combined = np.zeros((Z, Y, X), bool)
        anypath = False
        for a in (0, 1, 2):
            if len(shapes[a].data) == 0:
                continue
            anypath = True
            ps = tuple(s for i, s in enumerate((Z, Y, X)) if i != a)
            corr = build_corridor(shapes[a].data, ps, args.radius)
            combined |= np.expand_dims(corr, a)        # extrude along that projection axis, union
        if not anypath:
            print("draw a path on some projection first"); return
        sel = bright
        if args.activity_pct and args.activity_pct > 0:
            sel = bright | (hp_act > np.percentile(hp_act, args.activity_pct))
        mask = sel & combined
        if args.close > 1:
            mask = binary_closing(mask, generate_binary_structure(3, 2), iterations=args.close) & combined
        lbl = label(mask)
        sizes = np.bincount(lbl.ravel())
        keep = np.where(sizes >= args.min_vox)[0]; keep = keep[keep > 0]
        lbl = label(np.isin(lbl, keep))
        state["mask3d"] = lbl.astype(np.uint16)
        disp = lbl.max(cur["pa"]).astype(np.uint16)    # footprint in the current projection plane
        if "dendrite mask" in viewer.layers:
            viewer.layers["dendrite mask"].data = disp
        else:
            viewer.add_labels(disp, name="dendrite mask", opacity=0.5)
        print(f"built mask: {int((state['mask3d']>0).sum())} voxels, "
              f"{int(state['mask3d'].max())} components (footprint in {axname[cur['pa']]})", flush=True)

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
