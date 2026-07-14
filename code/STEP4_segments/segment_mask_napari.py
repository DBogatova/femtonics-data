#!/usr/bin/env python
"""
segment_mask_napari.py - Pick 3-4 segments on the mask by painting on MIP
projections; toggle the projection axis in-session; labels accumulate into one
3D segmentation and are saved as a labelmap for trace extraction.

Keys (in napari)
----------------
  a            : cycle projection axis (XY -> XZ -> ZY)
  j / k / l    : jump to XY / XZ / ZY view directly
                 (paint is committed to the 3D segmentation before switching,
                 and existing segments re-project so you keep building)
  s            : commit + save -> *_segments_labelmap.tif

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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import napari

    stack = tifffile.imread(args.stack)
    mask = tifffile.imread(args.labelmap) > 0
    bgvol = (stack.mean(0) if stack.ndim == 4 else stack).astype(np.float32)
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

    def commit():
        a = S["a"]
        s2 = np.asarray(seg.data)
        painted = np.broadcast_to(np.expand_dims(s2 > 0, a), mask.shape) & mask
        vals = np.broadcast_to(np.expand_dims(s2, a), mask.shape)
        S["seg3d"][painted] = vals[painted]

    def set_axis(a):
        commit()
        S["a"] = a
        img.data = bgvol.max(a); img.scale = scale_for(a)
        img.contrast_limits = tuple(np.percentile(bgvol.max(a), (2, 99.7)))
        foot.data = mask.max(a).astype(np.uint8); foot.scale = scale_for(a)
        seg.data = S["seg3d"].max(a).astype(np.uint8); seg.scale = scale_for(a)
        v.reset_view()
        print(f"axis -> {a} ({'XY' if a==0 else 'XZ' if a==1 else 'ZY'} view)")

    def cycle(vw):
        set_axis((S["a"] + 1) % 3)
    v.bind_key("a", cycle, overwrite=True)                 # cycle XY -> XZ -> ZY
    direct = {"j": 0, "k": 1, "l": 2}                      # j=XY(z) k=XZ(y) l=ZY(x)
    for key, ax in direct.items():
        v.bind_key(key, lambda vw, ax=ax: set_axis(ax), overwrite=True)

    @v.bind_key("s", overwrite=True)
    def _save(vw):
        commit()
        tifffile.imwrite(out, S["seg3d"].astype(np.uint16))
        ids = [int(i) for i in np.unique(S["seg3d"]) if i > 0]
        vx = {i: int((S["seg3d"] == i).sum()) for i in ids}
        print(f"saved -> {out}  segments={ids}  voxels={vx}")

    print("Paint labels 1..4 on 'segments'. Keys: a = cycle view, j=XY k=XZ l=ZY, s = save.")
    napari.run()


if __name__ == "__main__":
    main()
