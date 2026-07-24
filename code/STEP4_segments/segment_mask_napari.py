#!/usr/bin/env python
"""
segment_mask_napari.py - Pick 3-4 segments on the mask by painting on MIP
projections; toggle the projection axis in-session; labels accumulate into one
3D segmentation and are saved as a labelmap for trace extraction.

Keys (this tool)
----------------
  a            : cycle projection axis (XY -> XZ -> ZY); in 3D edit, snap the camera
  j / k / l    : jump to XY / XZ / ZY directly (2D) or snap the camera there (3D)
                 (paint is committed to the 3D segmentation before switching, and
                 existing segments re-project so you keep building)
  d            : toggle 2D projection painting <-> ROTATABLE 3D edit (like refine's 3D
                 view). In 3D the mask is pre-filled so the brush always has a surface:
                 PAINT (P) a segment / ERASE (E) to unassign, HOLD SPACE + drag to rotate;
                 press d again to return to 2D (edits kept, segments stay in the mask)
  g            : GROW segments from your seeds THROUGH the mask (geodesic nearest-seed):
                 dab one stroke on each branch, then g fills each connected branch and
                 splits touching branches at their junction. Unseeded, disconnected
                 branches stay unlabeled. Erase to trim, add seeds and press g again.
  u            : undo the last grow (single level)
  s            : commit + save -> *_segments_labelmap.tif

Painting uses napari's own controls
------------------------------------
  P / E        : paint / erase mode           (also 2 / 1)
  - / =        : previous / next label id ;   M = start a new label
  [ / ]        : smaller / larger brush
  Ctrl+Z       : undo a manual paint / erase stroke

For MANY THIN branches that overlap in a projection, don't rely on 2D extrusion (it
labels every mask voxel through the depth, grabbing branches behind the one you meant).
Instead: dab a seed on each branch (2D paint or the 3D view), press 'g' to grow along the
mask, then rotate in 3D and paint/erase to fix. This keeps each branch precise.

Paint labels 1..4 on the 'segments' layer over the magenta footprint. Only mask
voxels are labeled; painting extrudes along the current projection axis, so use
the view where the parts you want to separate don't overlap.

Usage
-----
  python code/STEP4_segments/segment_mask_napari.py run_clean.tif run_labelmap.tif --voxel 0.8 0.9 0.9
Then:
  python code/STEP5_traces/extract_segment_traces.py run_clean.tif *_segments_labelmap.tif --voxel 0.8 0.9 0.9
"""
import argparse
import numpy as np
import tifffile

AX = {"z": 0, "y": 1, "x": 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack")
    ap.add_argument("labelmap")
    ap.add_argument("--voxel", nargs=3, type=float, default=[1.0, 1.0, 1.0], metavar=("Z", "Y", "X"))
    ap.add_argument("--axis", choices=["z", "y", "x"], default="z", help="starting projection axis")
    ap.add_argument("--agg", choices=["max", "mean"], default="max",
                    help="temporal projection for the anatomy background (max matches refine)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import napari

    stack = tifffile.imread(args.stack)
    mask = tifffile.imread(args.labelmap) > 0
    bgvol = (stack.max(0) if args.agg == "max" else stack.mean(0)).astype(np.float32) \
        if stack.ndim == 4 else stack.astype(np.float32)
    vox = list(args.voxel)
    out = args.out or args.labelmap.rsplit(".", 1)[0].replace("_labelmap", "") + "_segments_labelmap.tif"

    S = {"seg3d": np.zeros(mask.shape, np.uint8), "a": AX[args.axis]}

    def proj(vol, a, labels=False):
        return vol.max(axis=a)

    def scale_for(a):
        return tuple(v for i, v in enumerate(vox) if i != a)

    v = napari.Viewer()
    a0 = S["a"]
    lo, hi = np.percentile(bgvol.max(a0), (2, 99.7))
    img = v.add_image(bgvol.max(a0), name="anatomy", colormap="gray",
                      contrast_limits=(lo, hi), scale=scale_for(a0))
    foot = v.add_labels(mask.max(a0).astype(np.uint8), name="mask footprint",
                        opacity=0.3, scale=scale_for(a0))
    seg = v.add_labels(S["seg3d"].max(a0).astype(np.uint8), name="segments",
                       opacity=0.6, scale=scale_for(a0))
    seg.mode = "paint"
    v.scale_bar.visible = True
    v.scale_bar.unit = "um"

    S["mode"] = "2d"        # "2d" = projection painting, "edit" = 3D volume editing
    S["L3"] = {}
    S["eo"] = 0             # current edit-slice orientation index
    S["undo"] = None        # snapshot of segments taken before the last grow (for u)
    S["unset"] = 255        # sentinel pre-filled over the mask so 3D drawing has a surface
    S["base2d"] = np.asarray(seg.data).astype(np.uint8).copy()   # baseline for the delta commit

    def show_seg2d(a):
        """Project the current 3D segments onto axis a, show them, and record that as the
        delta-commit baseline - so only NEW 2D strokes get extruded (never the whole
        projection, which would flatten depth-specific 3D edits)."""
        d = S["seg3d"].max(a).astype(np.uint8)
        seg.data = d
        S["base2d"] = d.copy()

    def commit():
        """Extrude only the 2D pixels changed since the last baseline along the current
        axis (constrained to the mask); untouched columns keep their 3D structure."""
        a = S["a"]
        s2 = np.asarray(seg.data).astype(np.uint8)
        base = S["base2d"]
        if base.shape != s2.shape:
            base = np.zeros_like(s2)
        changed = s2 != base
        if changed.any():
            chg3 = np.broadcast_to(np.expand_dims(changed, a), mask.shape)
            new3 = np.broadcast_to(np.expand_dims(s2, a), mask.shape)
            paint = chg3 & mask & (new3 > 0)
            S["seg3d"][paint] = new3[paint]
            clear = chg3 & (new3 == 0)
            S["seg3d"][clear] = 0
        S["base2d"] = s2.copy()

    def to_edit():
        if S["mode"] == "edit":
            return
        commit()
        for ly in (img, foot, seg):
            ly.visible = False
        clim = (float(np.percentile(bgvol, 2)), float(np.percentile(bgvol, 99.7)))
        S["L3"]["anat"] = v.add_image(bgvol, name="anatomy 3D", colormap="gray",
                                      scale=tuple(vox), contrast_limits=clim,
                                      rendering="attenuated_mip")   # full opacity, like refine
        seg0 = S["seg3d"].astype(np.uint8).copy()
        seg0[mask & (seg0 == 0)] = S["unset"]              # pre-fill the mask so the 3D brush
        lyr = v.add_labels(seg0, name="segments (paint 1..N / erase)",   # always has a surface to hit
                           scale=tuple(vox), opacity=0.6)
        S["L3"]["seg"] = lyr
        S["mode"] = "edit"
        v.dims.ndisplay = 3                                # rotatable 3D cube (like refine's 3D view)
        v.dims.order = (0, 1, 2)
        lyr.n_edit_dimensions = 3                          # paint/erase a 3D sphere, not one slice
        lyr.brush_size = 4                                 # small brush for thin branches
        lyr.selected_label = 1
        lyr.mode = "paint"                                 # draw label 1 by default; press E to erase
        v.layers.selection.active = lyr
        snap_camera(S["eo"])
        print("3D edit (like refine): the whole mask is pre-filled so the brush always has a "
              "surface. PAINT (P) a segment - pick its label with -/= or M - and ERASE (E) to "
              "unassign; [ / ] resize the brush; HOLD SPACE + drag to rotate; g = grow from your "
              "seeds (u = undo); a/j/k/l snap camera; d returns to 2D painting.", flush=True)

    def to_paint():
        if S["mode"] != "edit":
            return
        s3 = np.asarray(S["L3"]["seg"].data).astype(np.uint8)
        s3[s3 == S["unset"]] = 0                           # sentinel -> unassigned (not a segment)
        s3[~mask] = 0                                      # segments stay inside the mask
        S["seg3d"] = s3
        for ly in list(S["L3"].values()):
            v.layers.remove(ly)
        S["L3"].clear()
        S["mode"] = "2d"
        v.dims.ndisplay = 2
        a = S["a"]
        foot.data = mask.max(a).astype(np.uint8)
        show_seg2d(a)
        for ly in (img, foot, seg):
            ly.visible = True
        v.layers.selection.active = seg
        v.reset_view()
        print("2D painting mode.", flush=True)

    def set_axis(a):
        commit()
        S["a"] = a
        img.data = bgvol.max(a); img.scale = scale_for(a)
        img.contrast_limits = tuple(np.percentile(bgvol.max(a), (2, 99.7)))
        foot.data = mask.max(a).astype(np.uint8); foot.scale = scale_for(a)
        show_seg2d(a); seg.scale = scale_for(a)
        v.reset_view()
        print(f"axis -> {a} ({'XY' if a==0 else 'XZ' if a==1 else 'ZY'} view)")

    # canonical 3D camera orientations: (view_direction, up_direction) in (Z, Y, X)
    CANON = {0: ((1, 0, 0), (0, 1, 0)),    # XY: look down Z
             1: ((0, 1, 0), (-1, 0, 0)),   # XZ: look down Y (Z up)
             2: ((0, 0, 1), (-1, 0, 0))}   # ZY: look down X (Z up)

    def snap_camera(a):
        v.dims.ndisplay = 3
        vd, up = CANON[a]
        try:
            v.camera.set_view_direction(view_direction=vd, up_direction=up)
        except Exception:
            v.reset_view()

    def view_axis(a):
        if S["mode"] == "edit":
            S["eo"] = a
            snap_camera(a)                                 # rotate the 3D cube, stay in 3D erase
            print(f"camera -> {'XY' if a==0 else 'XZ' if a==1 else 'ZY'} view "
                  "(hold SPACE + drag to rotate freely)", flush=True)
        else:
            set_axis(a)

    def cycle(vw):
        if S["mode"] == "edit":
            view_axis((S["eo"] + 1) % 3)
        else:
            set_axis((S["a"] + 1) % 3)

    # --- geodesic grow from seeds through the mask (label numbers, brush size and
    #     paint-vs-erase are all handled by napari's own controls) ---
    def _geodesic_fill(seeds, m):
        from scipy.ndimage import distance_transform_edt
        from skimage.segmentation import watershed
        seeds = seeds.astype(np.uint8).copy()
        seeds[~m] = 0
        edt = distance_transform_edt(seeds == 0)               # flood outward from every seed
        return watershed(edt, markers=seeds, mask=m).astype(np.uint8)

    def grow(vw):
        if S["mode"] == "edit":
            seeds = np.asarray(S["L3"]["seg"].data).astype(np.uint8).copy()
            seeds[seeds == S["unset"]] = 0                 # the pre-fill is not a seed
            m = mask
        else:
            commit()
            seeds, m = S["seg3d"], mask
        if not (seeds > 0).any():
            print("grow: paint at least one seed on each segment first.", flush=True)
            return
        S["undo"] = seeds.astype(np.uint8).copy()          # snapshot for one-key undo (u)
        grown = _geodesic_fill(seeds, m)
        if S["mode"] == "edit":
            out = grown.astype(np.uint8)
            out[mask & (out == 0)] = S["unset"]            # keep the pre-fill so drawing still anchors
            S["L3"]["seg"].data = out
            S["L3"]["seg"].refresh()
        else:
            S["seg3d"] = grown
            show_seg2d(S["a"])
        ids = {int(i): int((grown == i).sum()) for i in np.unique(grown) if i > 0}
        print(f"grew segments through the mask (geodesic nearest-seed): voxels={ids}. "
              "Touching branches split at their junction; press u to undo, erase to trim, "
              "add seeds and grow again.", flush=True)

    def undo(vw):
        snap = S.get("undo")
        if snap is None:
            print("nothing to undo (grow with g first; Ctrl+Z undoes manual paint/erase).",
                  flush=True)
            return
        if S["mode"] == "edit":
            out = snap.astype(np.uint8).copy()
            out[mask & (out == 0)] = S["unset"]
            S["L3"]["seg"].data = out
            S["L3"]["seg"].refresh()
        else:
            S["seg3d"] = snap.copy()
            show_seg2d(S["a"])
        S["undo"] = None                                       # single-level undo
        print("undid the last grow (segments restored to their pre-grow seeds).", flush=True)

    v.bind_key("a", cycle, overwrite=True)                 # cycle view (projection or slice)
    for key, ax in {"j": 0, "k": 1, "l": 2}.items():       # j=XY k=XZ l=ZY
        v.bind_key(key, lambda vw, ax=ax: view_axis(ax), overwrite=True)
    v.bind_key("d", lambda vw: (to_paint() if S["mode"] == "edit" else to_edit()),
               overwrite=True)                             # toggle paint <-> volume edit
    v.bind_key("g", grow, overwrite=True)                  # geodesic grow from seeds through the mask
    v.bind_key("u", undo, overwrite=True)                  # undo the last grow (single level)

    @v.bind_key("s", overwrite=True)
    def _save(vw):
        if S["mode"] == "edit":
            to_paint()                # writes exact 3D edits (sentinel stripped) into seg3d
        else:
            commit()                  # capture the current 2D strokes
        tifffile.imwrite(out, S["seg3d"].astype(np.uint16))
        ids = [int(i) for i in np.unique(S["seg3d"]) if i > 0]
        vx = {i: int((S["seg3d"] == i).sum()) for i in ids}
        print(f"saved -> {out}  segments={ids}  voxels={vx}")

    print("Segments: label #, brush size and paint/erase use napari's own controls "
          "(P/E, -/=, M, [ / ]). Tool keys: a=cycle view, j/k/l=XY/XZ/ZY, d=2D<->3D edit, "
          "g=grow-from-seeds, u=undo-grow, s=save.")
    napari.run()


if __name__ == "__main__":
    main()
