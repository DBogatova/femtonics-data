#!/usr/bin/env python3
"""
segment_3d_movie.py - 3D turntable movie of the dendrite rotating around the X
axis with the segments overlaid semi-transparently.

The structural volume (temporal max) and the segment labelmap are rotated around
X (the long axis) in small steps; each step is max-projected to a view and the
colored segments are blended on top. Exported as an MP4.

Usage
-----
  python code/STEP6_movie/segment_3d_movie.py run_clean.tif segments_labelmap.tif \
      --fps 8 --frames 90 --rotations 1 --alpha 0.5 --out seg3d.mp4
"""
import argparse
import numpy as np
import tifffile
import imageio.v2 as imageio
import matplotlib
from scipy.ndimage import rotate as nd_rotate


def pad_zy(v, D):
    Z, Y, X = v.shape
    out = np.zeros((D, D, X), v.dtype)
    z0, y0 = (D - Z) // 2, (D - Y) // 2
    out[z0:z0 + Z, y0:y0 + Y, :] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", help="4D (T,Z,Y,X) or 3D (Z,Y,X) stack")
    ap.add_argument("labelmap", help="3D (Z,Y,X) segment labelmap")
    ap.add_argument("--fps", type=int, default=8, help="playback frame rate")
    ap.add_argument("--frames", type=int, default=90, help="frames for a full rotation")
    ap.add_argument("--rotations", type=float, default=1.0, help="number of full turns over the movie")
    ap.add_argument("--alpha", type=float, default=0.5, help="segment overlay opacity")
    ap.add_argument("--scale", type=int, default=4, help="integer upscale for visibility")
    ap.add_argument("--agg", choices=["max", "mean"], default="max", help="temporal projection for the structure")
    ap.add_argument("--time", action="store_true",
                    help="play the activity over time in 3D (instead of a static turntable)")
    ap.add_argument("--dual", action="store_true",
                    help="stacked movie: top = structure, bottom = live activity (same 3D view, over time)")
    ap.add_argument("--angle", type=float, default=30.0,
                    help="viewing tilt around X for --time mode (degrees)")
    ap.add_argument("--time-subsample", type=int, default=1, help="use every Nth timepoint (--time)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stack = tifffile.imread(args.stack)
    seg = tifffile.imread(args.labelmap).astype(np.uint16)
    struct = (stack.max(0) if args.agg == "max" else stack.mean(0)).astype(np.float32) \
        if stack.ndim == 4 else stack.astype(np.float32)
    assert seg.shape == struct.shape, f"mask {seg.shape} != volume {struct.shape}"
    Z, Y, X = struct.shape

    D = int(np.ceil(np.hypot(Z, Y)))          # pad Z,Y so rotation around X never clips
    struct = pad_zy(struct, D)
    seg = pad_zy(seg, D)

    lo_s, hi_s = np.percentile(struct[struct > 0], (2, 99.7)) if (struct > 0).any() else (0.0, 1.0)
    N = int(seg.max())
    turbo = matplotlib.colormaps["turbo"]
    segcol = np.zeros((N + 1, 3), np.float32)
    for lbl in range(1, N + 1):
        segcol[lbl] = turbo((lbl - 0.5) / max(1, N))[:3]

    suffix = "_3d_dual.mp4" if args.dual else "_3d_time.mp4" if args.time else "_3d_rotX.mp4"
    out = args.out or args.labelmap.rsplit(".", 1)[0] + suffix
    writer = imageio.get_writer(out, fps=args.fps, macro_block_size=None)
    s = max(1, args.scale)

    def panel(view, segv, lo, hi):                       # blended RGB float (H,W,3)
        g = np.clip((view - lo) / (hi - lo + 1e-6), 0, 1)
        rgb = np.repeat(g[..., None], 3, axis=2)
        a = (segv > 0)[..., None] * args.alpha
        return rgb * (1 - a) + segcol[segv] * a

    def write_img(rgb):
        img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        img = np.repeat(np.repeat(img, s, 0), s, 1)
        h, w = img.shape[:2]
        if h % 2 or w % 2:
            img = np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)))
        writer.append_data(img)

    if (args.dual or args.time) and stack.ndim == 4:
        T = stack.shape[0]
        idx = list(range(0, T, max(1, args.time_subsample)))
        samp = stack[:: max(1, T // 40)]
        lo_a, hi_a = np.percentile(samp, (30, 99.7))     # activity contrast (events over baseline)
        for k, t in enumerate(idx):
            ang = args.angle + k * 360.0 * args.rotations / max(1, len(idx))
            segr = nd_rotate(seg, ang, axes=(0, 1), reshape=False, order=0).max(0)
            act = nd_rotate(pad_zy(stack[t].astype(np.float32), D), ang, axes=(0, 1),
                            reshape=False, order=1).max(0)
            if args.dual:
                strv = nd_rotate(struct, ang, axes=(0, 1), reshape=False, order=1).max(0)
                top = panel(strv, segr, lo_s, hi_s)      # structure
                bot = panel(act, segr, lo_a, hi_a)       # live activity
                sep = np.full((2, top.shape[1], 3), 0.2, np.float32)
                write_img(np.vstack([top, sep, bot]))
            else:
                write_img(panel(act, segr, lo_a, hi_a))
        mode = "dual (structure / activity)" if args.dual else "time series"
        print(f"saved {out}  ({len(idx)} timepoints @ {args.fps} fps, tilt {args.angle} deg, "
              f"{args.rotations} turn(s), {mode}, {N} segments)")
    else:
        # static turntable: rotate the structure around X
        for i in range(args.frames):
            ang = i * 360.0 * args.rotations / args.frames
            strv = nd_rotate(struct, ang, axes=(0, 1), reshape=False, order=1).max(0)
            segr = nd_rotate(seg, ang, axes=(0, 1), reshape=False, order=0).max(0)
            write_img(panel(strv, segr, lo_s, hi_s))
        print(f"saved {out}  ({args.frames} frames @ {args.fps} fps, {args.rotations} turn(s) "
              f"around X, {N} segments)")
    writer.close()


if __name__ == "__main__":
    main()
