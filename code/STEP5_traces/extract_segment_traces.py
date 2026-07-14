#!/usr/bin/env python3
"""
extract_segment_traces.py - Split a 3D dendrite mask into segments along its
length and extract per-segment dF/F traces + event timestamps ("what each part
does, and when").

Steps
-----
  1. mask voxels -> physical coords (voxel-scaled) -> PCA principal axis
  2. bin voxels along that axis into N contiguous segments
  3. per segment: mean fluorescence over time -> dF/F (F0 = 10th percentile)
  4. detect events per segment (prominence peaks) -> timestamps
  5. save: per-segment CSVs (Slice,Mean=dF/F), combined CSV, events CSV,
     a segment-label TIFF, and a preview (segment map + stacked traces)

Usage
-----
  python code/STEP5_traces/extract_segment_traces.py \
      rbp4_139_phpeb/06-12-2026/traces/run5/3dstack_clean.tif \
      rbp4_139_phpeb/06-12-2026/traces/run5/3dstack_clean_dendrite_labelmap.tif \
      --n-segments 6 --voxel 2.9 0.85 0.85
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import tifffile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", help="4D (T,Z,Y,X) stack aligned to the mask")
    ap.add_argument("labelmap", help="3D (Z,Y,X) dendrite mask")
    ap.add_argument("--n-segments", type=int, default=6,
                    help="segments for AUTO split (used only if the mask has a single label)")
    ap.add_argument("--force-split", action="store_true",
                    help="force automatic PCA split even if the mask has multiple labels")
    ap.add_argument("--voxel", nargs=3, type=float, default=[2.9, 0.85, 0.85],
                    metavar=("Z", "Y", "X"))
    ap.add_argument("--f0-pct", type=float, default=10.0, help="percentile for F0 baseline")
    ap.add_argument("--prom-frac", type=float, default=0.2,
                    help="event prominence as fraction of each segment's dF/F range")
    ap.add_argument("--min-dist", type=int, default=5, help="min frames between events")
    ap.add_argument("--proj-axis", choices=["z", "y", "x"], default="z",
                    help="projection for the segment map: z=XY, y=XZ, x=ZY")
    ap.add_argument("--mask", default=None,
                    help="full dendrite mask labelmap to draw semi-transparently under the segments")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    stack = tifffile.imread(args.stack)
    assert stack.ndim == 4, f"expected 4D, got {stack.shape}"
    T = stack.shape[0]
    lm = tifffile.imread(args.labelmap)
    assert lm.shape == stack.shape[1:], f"mask {lm.shape} != vol {stack.shape[1:]}"

    stem = Path(args.stack).with_suffix("")
    outdir = Path(args.out_dir) if args.out_dir else Path(args.stack).parent
    outdir.mkdir(parents=True, exist_ok=True)

    uniq = sorted(int(v) for v in np.unique(lm) if v > 0)
    if len(uniq) > 1 and not args.force_split:
        # --- use manually-painted labels as segments ---
        seg_vol = lm.astype(np.uint16)
        seg_labels = uniq
        print(f"using {len(seg_labels)} manually-defined segments (labels {seg_labels})")
    else:
        # --- automatic PCA split of a single-label mask ---
        N = args.n_segments
        mask = lm > 0
        coords = np.argwhere(mask)
        phys = coords * np.array(args.voxel)
        c = phys - phys.mean(0)
        axis = np.linalg.svd(c, full_matrices=False)[2][0]
        proj = c @ axis
        if proj[0] > proj[-1]:
            proj = -proj
        edges = np.linspace(proj.min(), proj.max(), N + 1)
        seg_id = np.clip(np.digitize(proj, edges) - 1, 0, N - 1)
        seg_vol = np.zeros(mask.shape, np.uint16)
        seg_vol[tuple(coords.T)] = seg_id + 1
        seg_labels = list(range(1, N + 1))
        print(f"auto PCA split into {N} segments along the dendrite axis")

    N = len(seg_labels)
    tifffile.imwrite(f"{stem}_segments{N}.tif", seg_vol)

    # --- per-segment dF/F ---
    traces = np.zeros((N, T), np.float32)
    for i, lbl in enumerate(seg_labels):
        sel = seg_vol == lbl
        F = stack[:, sel].mean(1).astype(np.float32)
        f0 = np.percentile(F, args.f0_pct)
        traces[i] = (F - f0) / (f0 if f0 else 1.0)
        print(f"  segment {lbl}: {int(sel.sum()):5d} voxels")

    # --- events per segment ---
    events = []
    for i, lbl in enumerate(seg_labels):
        tr = traces[i]
        rng = tr.max() - tr.min()
        pk, _ = find_peaks(tr, prominence=args.prom_frac * rng, distance=args.min_dist)
        for p in pk:
            events.append((lbl, int(p), float(tr[p])))
    print(f"detected {len(events)} events across {N} segments")

    # --- save CSVs ---
    for i, lbl in enumerate(seg_labels):
        with open(outdir / f"{Path(stem).name}_seg{lbl:02d}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["Slice", "Mean"])
            for t in range(T):
                w.writerow([t + 1, round(float(traces[i, t]), 6)])
    with open(f"{stem}_segment_traces.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Frame"] + [f"seg{lbl:02d}" for lbl in seg_labels])
        for t in range(T):
            w.writerow([t] + [round(float(traces[i, t]), 6) for i in range(N)])
    with open(f"{stem}_segment_events.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["segment", "frame", "dff"])
        for e in events:
            w.writerow([e[0], e[1], round(e[2], 6)])

    # --- preview: segments marked on the cell MIP + stacked traces with event marks ---
    pa = {"z": 0, "y": 1, "x": 2}[args.proj_axis]
    view = {"z": "XY", "y": "XZ", "x": "ZY"}[args.proj_axis]
    cell = stack.mean(0).max(axis=pa)                 # cell anatomy MIP along proj axis
    segmip = seg_vol.max(axis=pa)                     # segment labels projected
    clo, chi = np.percentile(cell, (2, 99.5))
    # optional full-mask underlay (context for where the whole dendrite is)
    full_mip = None
    if args.mask:
        full = tifffile.imread(args.mask) > 0
        if full.shape == seg_vol.shape:
            full_mip = full.max(axis=pa)

    vz, vy, vx = args.voxel
    aspect_map = {0: vy / vx, 1: vz / vx, 2: vz / vy}[pa]   # true physical proportions
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), gridspec_kw={"height_ratios": [1, 3.2]})
    ax[0].imshow(np.clip((cell - clo) / (chi - clo + 1e-6), 0, 1), cmap="gray", aspect=aspect_map,
                 interpolation="lanczos")             # smooth anatomy (labels stay crisp)
    if full_mip is not None:                          # semi-transparent whole mask beneath
        ax[0].imshow(np.ma.masked_where(~full_mip, full_mip), cmap="gray_r", alpha=0.25,
                     aspect=aspect_map, interpolation="nearest")
    ax[0].imshow(np.ma.masked_where(segmip == 0, segmip), cmap="turbo", alpha=0.75,
                 aspect=aspect_map, vmin=1, vmax=max(2, N), interpolation="nearest")
    ax[0].set_title(f"segments on the cell ({view} MIP), {N} segment(s)"); ax[0].axis("off")

    off = 1.1 * max([float(t.max() - t.min()) for t in traces] + [1e-6])
    cmap = plt.get_cmap("turbo")
    for i, lbl in enumerate(seg_labels):
        shift = i * off                               # seg1 at bottom, highest label on top
        ax[1].plot(traces[i] + shift, color=cmap((i + 0.5) / N), lw=0.7)
        ev = [e for e in events if e[0] == lbl]
        if ev:
            fr = [e[1] for e in ev]
            ax[1].plot(fr, [traces[i][x] + shift for x in fr], ".", color="k", ms=4)
        ax[1].text(-0.01 * T, shift + float(traces[i].mean()), f"seg{lbl}",
                   ha="right", va="center", fontsize=8, color=cmap((i + 0.5) / N))
    # dF/F scale bar in a clear band below the lowest (seg1) trace
    rng = np.median([float(t.max() - t.min()) for t in traces]) or 1.0
    cand = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    sb = float(cand[np.argmin(np.abs(cand - 0.5 * rng))])
    ax[1].set_ylim(-1.8 * sb, None)
    x0 = T * 0.015
    ax[1].plot([x0, x0], [-1.5 * sb, -0.5 * sb], color="k", lw=2.5)
    ax[1].text(x0 + T * 0.012, -1.0 * sb, f"{sb:g} \u0394F/F", va="center", ha="left", fontsize=8)
    ax[1].set_xlabel("Frame"); ax[1].set_yticks([])
    ax[1].set_title("per-segment dF/F (offset; seg1 bottom); dots = events")
    plt.tight_layout(); plt.savefig(f"{stem}_segment_traces.png", dpi=200)
    print(f"saved: {stem}_segments{N}.tif\n       {stem}_segment_traces.csv"
          f"\n       {stem}_segment_events.csv\n       {stem}_segment_traces.png"
          f"\n       per-segment seg??.csv in {outdir}")


if __name__ == "__main__":
    main()
