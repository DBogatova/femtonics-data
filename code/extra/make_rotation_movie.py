#!/usr/bin/env python3
"""
make_rotation_movie.py - Build a "rotating while time plays" movie from a 4D
TIFF whose axes are (T, angle, Y, X) -- i.e. each timepoint already holds a
series of pre-rendered rotation views (as produced by an ImageJ 3D Project
saved over time).

For each output frame the timepoint advances and the rotation angle advances,
so the structure spins while the activity plays.

EXAMPLE
-------
.venv/bin/python code/make_rotation_movie.py \
    preprocessed/rbp4_132_2026-05-20_MUnit_13_4D_rotation.tif \
    --rotations 6 --fps 20 --scale 3 --counter \
    --out preprocessed/MUnit_13_rotating.mp4
"""
import argparse
import numpy as np
import tifffile
import imageio.v2 as imageio
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiff", help="4D TIFF, axes (T, angle, Y, X)")
    ap.add_argument("--out", default=None, help="output .mp4 (or .gif)")
    ap.add_argument("--angle-step", type=float, default=1.0,
                    help="rotation-view indices advanced per frame (default 1 = "
                         "exact native pairing: one stored angle per timepoint)")
    ap.add_argument("--rotations", type=float, default=None,
                    help="if set, overrides --angle-step to give this many full "
                         "spins across the whole movie")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--time-subsample", type=int, default=1,
                    help="use every Nth timepoint (default 1 = all)")
    ap.add_argument("--scale", type=int, default=3,
                    help="integer upscale for visibility (default 3)")
    ap.add_argument("--pmin", type=float, default=1.0, help="low contrast percentile")
    ap.add_argument("--pmax", type=float, default=99.7, help="high contrast percentile")
    ap.add_argument("--counter", action="store_true",
                    help="overlay the timepoint (frame) number")
    args = ap.parse_args()

    with tifffile.TiffFile(args.tiff) as tf:
        arr = tf.series[0].asarray()          # (T, A, Y, X)
    if arr.ndim != 4:
        raise SystemExit(f"expected 4D (T,angle,Y,X), got shape {arr.shape}")
    T, A, Y, X = arr.shape
    print(f"loaded {arr.shape} dtype={arr.dtype}")

    # global contrast from a time/angle sample so brightness is stable
    sample = arr[::max(1, T // 40)].astype(np.float32)
    lo, hi = np.percentile(sample, (args.pmin, args.pmax))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max() or 1)

    times = range(0, T, args.time_subsample)
    n_out = len(list(times))
    out = args.out or args.tiff.rsplit(".", 1)[0] + "_rotating.mp4"

    # angle advance per output frame (in rotation-view index units)
    if args.rotations is not None:
        step = A * args.rotations / max(1, n_out)
    else:
        step = args.angle_step

    writer = imageio.get_writer(out, fps=args.fps, macro_block_size=None) \
        if out.endswith(".mp4") else imageio.get_writer(out, fps=args.fps)

    for i, t in enumerate(range(0, T, args.time_subsample)):
        a = int(round(i * step)) % A
        frame = arr[t, a].astype(np.float32)
        frame = np.clip((frame - lo) / (hi - lo), 0, 1)
        img8 = (frame * 255).astype(np.uint8)

        if args.scale > 1:
            img8 = np.repeat(np.repeat(img8, args.scale, 0), args.scale, 1)

        if args.counter:
            im = Image.fromarray(img8).convert("L")
            d = ImageDraw.Draw(im)
            d.text((4, 2), f"t={t}", fill=255)
            img8 = np.asarray(im)

        # H.264 needs even width/height: pad with a black row/col if odd
        h, w = img8.shape[:2]
        if h % 2 or w % 2:
            img8 = np.pad(img8, ((0, h % 2), (0, w % 2)), mode="constant")

        writer.append_data(img8)

    writer.close()
    spins = n_out * step / A
    print(f"saved {out}  ({n_out} frames @ {args.fps} fps, "
          f"~{n_out/args.fps:.0f}s, {spins:.1f} rotations, {step:g} angle/frame)")


if __name__ == "__main__":
    main()
