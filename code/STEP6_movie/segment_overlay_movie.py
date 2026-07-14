#!/usr/bin/env python3
"""
segment_overlay_movie.py - Export an MP4 of the projected activity movie with the
segment labels overlaid (same look as the segment trace figure, but over time).

Each frame is the per-timepoint MIP (along --proj-axis) in grayscale with the
static segment colors overlaid, so you watch each segment light up.

Usage
-----
  python code/STEP6_movie/segment_overlay_movie.py run_clean.tif segments_labelmap.tif \
      --proj-axis y --fps 20 --scale 4 --out seg_movie.mp4
"""
import argparse
import numpy as np
import tifffile
import imageio.v2 as imageio
import matplotlib


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", help="4D (T,Z,Y,X) stack")
    ap.add_argument("labelmap", help="3D (Z,Y,X) segment labelmap")
    ap.add_argument("--proj-axis", choices=["z", "y", "x"], default="y",
                    help="projection: z=XY, y=XZ, x=ZY")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--scale", type=int, default=4, help="integer upscale for visibility")
    ap.add_argument("--alpha", type=float, default=0.5, help="segment overlay opacity")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pa = {"z": 0, "y": 1, "x": 2}[args.proj_axis]
    stack = tifffile.imread(args.stack)
    seg = tifffile.imread(args.labelmap)
    assert stack.ndim == 4 and seg.shape == stack.shape[1:], "shape mismatch"
    T = stack.shape[0]

    proj = stack.max(axis=pa + 1)            # (T, H, W) per-frame MIP along spatial axis pa
    segmip = seg.max(axis=pa)                # (H, W) segment labels
    N = int(segmip.max())

    # fixed global contrast so activity changes are visible
    samp = proj[:: max(1, T // 40)]
    lo, hi = np.percentile(samp, (2, 99.7))

    # static segment overlay (turbo colors + alpha)
    turbo = matplotlib.colormaps["turbo"]
    segrgb = np.zeros((*segmip.shape, 3), np.float32)
    for lbl in range(1, N + 1):
        segrgb[segmip == lbl] = turbo((lbl - 0.5) / max(1, N))[:3]
    amask = ((segmip > 0).astype(np.float32) * args.alpha)[..., None]

    out = args.out or args.labelmap.rsplit(".", 1)[0] + f"_overlay_{args.proj_axis}.mp4"
    writer = imageio.get_writer(out, fps=args.fps, macro_block_size=None)
    s = max(1, args.scale)
    for t in range(T):
        g = np.clip((proj[t].astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
        rgb = np.repeat(g[..., None], 3, axis=2)
        frame = rgb * (1 - amask) + segrgb * amask
        img = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        img = np.repeat(np.repeat(img, s, 0), s, 1)
        h, w = img.shape[:2]
        if h % 2 or w % 2:
            img = np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)))
        writer.append_data(img)
    writer.close()
    print(f"saved {out}  ({T} frames @ {args.fps} fps, {N} segments, {args.proj_axis}-projection)")


if __name__ == "__main__":
    main()
