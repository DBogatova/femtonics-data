#!/usr/bin/env python
"""
refine_mask_napari.py - View and REFINE a 3D dendrite mask over its volume in
napari: rotate/inspect in 3D, paint to add or erase to remove, Ctrl+S to save.

Loads a structural volume (temporal max/mean of the 4D stack, or a single frame)
and overlays the labelmap as an editable Labels layer.

Run from the project root with the project venv:
  .venv/bin/python code/STEP3_mask/refine_mask_napari.py \
      run_clean.tif run_dendrite_labelmap.tif --voxel 0.8 0.9 0.9 --ndisplay 2

Controls: select the mask layer -> paintbrush adds, eraser removes; Ctrl+S saves
(*_edited.tif). Draw on 2D slices in any orientation and flip to 3D to check:
  j / k / l : edit XY / XZ / ZY slices (scroll the slider through depth)
  d         : toggle 2D (edit) <-> 3D (inspect)
--ndisplay 2 starts in flat editing, 3 starts rotatable; --movie overlays on the 4D
time series; --frame N shows one timepoint.
--rot-x DEG rotates the volume around X (e.g. 45) so you can see/edit a branch hidden
in the XY/XZ views; painted additions are rotated back and unioned onto the mask on save.
Two reference layers are added from the 4D stack to reveal branches the mask may miss:
"activity (max)" (ever-bright) and "transient branches" (max-mean high-pass, magenta,
on by default) - toggle them with the eye icon and paint the mask to include them.
"""
import argparse
import numpy as np
import tifffile


def _highpass(vol, small=(0.5, 1, 1), big=(2, 6, 6)):
    from scipy.ndimage import gaussian_filter
    e = gaussian_filter(vol, small) - gaussian_filter(vol, big)
    e[e < 0] = 0
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", help="4D (T,Z,Y,X) or 3D (Z,Y,X) structural stack")
    ap.add_argument("labelmap", nargs="?", default=None, help="optional 3D (Z,Y,X) mask labelmap")
    ap.add_argument("--voxel", nargs=3, type=float, default=[3.9, 1.0, 1.2],
                    metavar=("Z", "Y", "X"), help="voxel size (um) for correct aspect ratio")
    ap.add_argument("--agg", choices=["max", "mean"], default="max",
                    help="temporal projection for the structural volume")
    ap.add_argument("--frame", type=int, default=None,
                    help="show this single timepoint instead of a projection")
    ap.add_argument("--rendering", default="attenuated_mip",
                    help="3D image rendering mode (attenuated_mip, mip, translucent, iso, ...)")
    ap.add_argument("--movie", action="store_true",
                    help="overlay the mask on the full 4D time series (adds a time slider to play)")
    ap.add_argument("--ndisplay", type=int, choices=[2, 3], default=3,
                    help="2 = flat view (best for painting segment labels), 3 = rotatable volume")
    ap.add_argument("--rot-x", type=float, default=0.0,
                    help="rotate the volume this many degrees around X (long axis) for the session, "
                         "to see/edit a branch hidden in XY/XZ; painted edits are rotated back and "
                         "unioned onto the mask on save (add-only in this mode)")
    ap.add_argument("--segments", action="store_true",
                    help="segment-painting mode: show the given labelmap as faint context and paint "
                         "segment numbers (1,2,3...) on a fresh layer with the SAME 3D/slice controls; "
                         "saves *_segments_labelmap.tif clamped to the mask (for STEP5 traces)")
    ap.add_argument("--no-refs", action="store_true",
                    help="don't add the temporal 'activity (max)' / 'transient branches' reference "
                         "layers (they are temporal-max based and can look noisy on some recordings; "
                         "use with --frame N for a clean single-frame view)")
    args = ap.parse_args()

    import sys
    print(f"[1/4] python = {sys.executable}", flush=True)
    try:
        import napari
    except ImportError:
        sys.exit("ERROR: napari is not installed in THIS python.\n"
                 "Run with your napari env, e.g.:\n"
                 "  /Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/"
                 ".venv/bin/python code/STEP3_mask/refine_mask_napari.py ...")
    print(f"[2/4] napari {napari.__version__}", flush=True)

    stack = tifffile.imread(args.stack)
    labels = tifffile.imread(args.labelmap).astype(np.uint16) if args.labelmap else None
    scale3 = tuple(args.voxel)

    # activity reference volumes to reveal episodically-firing branches (4D only)
    act_vols = {}
    if stack.ndim == 4 and not args.no_refs:
        vmax_a = stack.max(0).astype(np.float32)
        transient_a = _highpass(np.clip(vmax_a - stack.mean(0).astype(np.float32), 0, None))
        act_vols = {"activity (max)": vmax_a, "transient branches": transient_a}

    mask_info = (f"mask {labels.shape}, voxels {int((labels>0).sum())}"
                 if labels is not None else "no mask")
    if args.movie and stack.ndim == 4:
        # full 4D movie: image keeps the time axis; mask is 3D and broadcasts over time
        img = stack
        img_scale = (1.0,) + scale3
        samp = stack[:: max(1, stack.shape[0] // 30)]
        clim = (float(np.percentile(samp, 2)), float(np.percentile(samp, 99.7)))
        vol_shape = stack.shape[1:]
        print(f"[3/4] movie: {stack.shape}, {mask_info}", flush=True)
    else:
        if stack.ndim == 4:
            vol = stack[args.frame] if args.frame is not None else \
                (stack.mean(0) if args.agg == "mean" else stack.max(0))
        else:
            vol = stack
        img = vol.astype(np.float32)
        img_scale = scale3
        clim = (float(np.percentile(img, 2)), float(np.percentile(img, 99.7)))
        vol_shape = img.shape
        print(f"[3/4] volume {img.shape}, {mask_info}", flush=True)

    show_mask = labels is not None and labels.shape == vol_shape
    if labels is not None and labels.shape != vol_shape:
        print(f"WARNING: mask shape {labels.shape} != volume {vol_shape}; showing structure only")

    # optional oblique view: rotate structure (+mask) around X (axes Z,Y) for the session
    rot = float(args.rot_x)
    orig_labels = None
    if rot and not args.movie:
        from scipy.ndimage import rotate as nd_rotate
        img = nd_rotate(img, rot, axes=(0, 1), reshape=False, order=1)
        for k in list(act_vols):
            act_vols[k] = nd_rotate(act_vols[k], rot, axes=(0, 1), reshape=False, order=1)
        if show_mask:
            orig_labels = labels.copy()
            labels = (nd_rotate(labels.astype(np.float32), rot, axes=(0, 1),
                                reshape=False, order=0) > 0.5).astype(np.uint16)
        print(f"[rot] volume rotated {rot} deg around X for this session", flush=True)

    viewer = napari.Viewer(ndisplay=args.ndisplay)
    viewer.add_image(img, name="structure", scale=img_scale, colormap="gray",
                     rendering=args.rendering, contrast_limits=clim)
    # branch-revealing reference layers (mask stays on top for painting)
    for nm, cmap, vis, blend in [("activity (max)", "gray", False, "translucent"),
                                 ("transient branches", "magma", True, "additive")]:
        if nm in act_vols:
            va = act_vols[nm]
            lo, hi = (np.percentile(va[va > 0], (2, 99.7)) if (va > 0).any() else (0.0, 1.0))
            viewer.add_image(va, name=nm, scale=scale3, colormap=cmap, visible=vis,
                             blending=blend, rendering=args.rendering, contrast_limits=(lo, hi))
    if show_mask:
        import os
        if args.segments:
            # mask = faint context; paint segment numbers on a fresh editable layer
            viewer.add_labels(labels, name="mask (context)", scale=scale3, opacity=0.25)
            lbl_layer = viewer.add_labels(np.zeros_like(labels), name="segments (paint 1,2,3...)",
                                          scale=scale3, opacity=0.6)
            lbl_layer.mode = "paint"
            edit_out = args.labelmap.rsplit(".", 1)[0].replace("_labelmap", "") + "_segments_labelmap.tif"

            @viewer.bind_key("Control-s")
            def _save_seg(v):
                s = lbl_layer.data.astype(np.uint16)
                mref = labels.astype(bool)
                if rot and orig_labels is not None:
                    from scipy.ndimage import rotate as nd_rotate
                    s = nd_rotate(s.astype(np.float32), -rot, axes=(0, 1),
                                  reshape=False, order=0).astype(np.uint16)
                    mref = orig_labels.astype(bool)
                s[~mref] = 0                      # segments stay inside the mask
                tifffile.imwrite(edit_out, s)
                ids = [int(i) for i in np.unique(s) if i > 0]
                print(f"saved segments -> {edit_out}  labels={ids}", flush=True)

            print("SEGMENTS: on the 'segments' layer set the label number (1,2,3...) and paint each "
                  "part; erase to fix; a/j/k slice views, d=2D/3D; Ctrl+S saves ->", edit_out, flush=True)
        else:
            lbl_layer = viewer.add_labels(labels, name="dendrite mask", scale=scale3, opacity=0.5)
            edit_out = os.path.splitext(args.labelmap)[0] + "_edited.tif"

            @viewer.bind_key("Control-s")
            def _save_mask(v):
                m = lbl_layer.data.astype(np.uint16)
                if rot and orig_labels is not None:
                    from scipy.ndimage import rotate as nd_rotate
                    back = nd_rotate(m.astype(np.float32), -rot, axes=(0, 1),
                                     reshape=False, order=0) > 0.5
                    m = (orig_labels.astype(bool) | back).astype(np.uint16)
                tifffile.imwrite(edit_out, m)
                print(f"saved edited mask -> {edit_out}", flush=True)

            print("EDIT: select 'dendrite mask' layer, use paint (brush) / erase / fill tools;\n"
                  "      press Ctrl+S to save your corrected mask ->", edit_out, flush=True)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    # --- editing helpers: rotate around X (keep X horizontal) + flip to 3D ---
    if not args.movie:
        orders = [(0, 1, 2), (1, 0, 2)]                # XY (top, scroll Z), XZ (side, scroll Y)
        names = ["XY (top)", "XZ (side)"]
        st = {"i": 0}

        def _apply(i):
            st["i"] = i % len(orders)
            viewer.dims.ndisplay = 2
            viewer.dims.order = orders[st["i"]]
            viewer.reset_view()
            print(f"slice view: {names[st['i']]}", flush=True)

        viewer.bind_key("a", lambda v: _apply(st["i"] + 1), overwrite=True)  # rotate 90 deg around X
        viewer.bind_key("j", lambda v: _apply(0), overwrite=True)            # XY top
        viewer.bind_key("k", lambda v: _apply(1), overwrite=True)            # XZ side

        def _toggle3d(v):
            viewer.dims.ndisplay = 3 if viewer.dims.ndisplay == 2 else 2
            viewer.reset_view()
        viewer.bind_key("d", _toggle3d, overwrite=True)                      # flip 2D <-> 3D (drag to rotate)
        print("KEYS: a=rotate around X (XY<->XZ),  j=XY top,  k=XZ side,  d=toggle 2D/3D,  Ctrl+S=save",
              flush=True)
        if act_vols:
            print("BRANCHES: 'transient branches' (magenta) + 'activity (max)' layers show branches "
                  "the mask may miss — toggle via the eye icon and paint them into the mask.", flush=True)

    tip = " (press the play button on the time slider to run the movie)" if args.movie else ""
    print(f"[4/4] napari window opening{tip} — drag to rotate; close window to exit.", flush=True)
    napari.run()


if __name__ == "__main__":
    main()
