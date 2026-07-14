#!/usr/bin/env python3
"""
extract_dendrite_3d.py - Reconstruct ONE 3D dendrite mask from a 4D (T,Z,Y,X)
volumetric stack, using the auto_mask_m2 approach (adapted to a single mask).

Key ideas (matching apical-dendrites-2025 / auto_mask_m2):
  * the temporal MIP is only a GUIDE (for seeds/enhancement), not the mask
  * detection is 3D: spatial high-pass enhancement + Sato vesselness (tubular)
    + seed/candidate hysteresis, keep candidates connected to seeds
  * snake/resonant-scan vertical dark lines form a PERIODIC grid (~24 px here);
    those columns are inpainted (interp along X) BEFORE detection so the dendrite
    is not fragmented
  * keep the single largest seed-connected component -> one 3D dendrite

Usage
-----
  .venv/bin/python code/extra/extract_dendrite_3d.py \
      rbp4_139_phpeb/06-12-2026/traces/run5/3dstack.tif
  ... --period 24 --seed-pct 99 --cand-pct 95 --view
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, median_filter, binary_dilation, generate_binary_structure
from scipy.signal import find_peaks
from skimage.filters import sato
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt


def detect_scanline_grid(vol, period=None):
    """Return boolean X-mask of periodic dark scan-line columns."""
    prof = np.median(vol, axis=(0, 1)).astype(np.float32)     # (X,)
    X = len(prof)
    neg = median_filter(prof, size=9) - prof                  # dips are positive
    dips, _ = find_peaks(neg, distance=12, prominence=15)
    if period is None:
        # dominant period from spacings near the mode (ignore dendrite-region extras)
        if len(dips) > 2:
            sp = np.diff(dips)
            near = sp[np.abs(sp - np.median(sp)) <= 3]
            period = int(np.median(near)) if len(near) else int(np.median(sp))
        else:
            period = 24
    period = max(2, int(period))
    phase = int(dips[0]) % period if len(dips) else 0
    grid = np.arange(phase, X, period)
    dark = np.zeros(X, bool)
    for c in grid:
        dark[max(0, c - 1):min(X, c + 2)] = True              # widen +/-1
    return dark, prof, period, phase


def inpaint_columns(vol, dark):
    """Linear-interpolate dark X-columns from good neighbors, per (Z,Y)."""
    x = np.arange(vol.shape[2]); gx = x[~dark]
    out = vol.astype(np.float32).copy()
    for z in range(vol.shape[0]):
        for y in range(vol.shape[1]):
            out[z, y, dark] = np.interp(x[dark], gx, vol[z, y, ~dark])
    return out


def highpass(vol, small=(0.5, 1, 1), big=(2, 6, 6)):
    """Spatial high-pass: suppress broad background sheet, keep dendrite."""
    enh = gaussian_filter(vol, sigma=small) - gaussian_filter(vol, sigma=big)
    enh[enh < 0] = 0
    return enh


def norm(a):
    p = np.nanpercentile(a, 99.9)
    return a / p if p > 0 else a


def keep_touching_seed(cand, seed, conn):
    lbl = label(cand, connectivity=conn)
    keep = np.unique(lbl[seed & cand]); keep = keep[keep > 0]
    return np.isin(lbl, keep) if len(keep) else np.zeros_like(cand, bool)


def cofluctuation_grow(arr4d, core, guide, corr_thr, spatial_sigma, detrend_win, conn,
                       intensity_pct, keep_corr, close_x, disconnect_corr, min_voxels):
    """Define the mask from voxels co-fluctuating with the core's mean trace.

    - detrending removes shared bleaching (only real events count)
    - intensity gate removes the weakly-correlated dim halo, BUT voxels with
      correlation >= keep_corr are kept regardless of brightness -> the dim but
      strongly co-fluctuating trunk survives
    - closing along X fills the scan-line notches
    - a spatially DISCONNECTED component is still kept if its mean correlation is
      high (>= disconnect_corr) -> follows the heatmap to grab the out-of-MIP end
    """
    from scipy.ndimage import uniform_filter1d, binary_closing
    from skimage.morphology import remove_small_objects
    T = arr4d.shape[0]
    data = arr4d.astype(np.float32)
    if spatial_sigma > 0:
        data = gaussian_filter(data, sigma=(0, 0.5, spatial_sigma, spatial_sigma))
    if detrend_win and detrend_win > 1:
        data = data - uniform_filter1d(data, size=detrend_win, axis=0, mode="nearest")
    Z, Y, X = data.shape[1:]
    flat = data.reshape(T, -1)
    seed_trace = flat[:, core.reshape(-1)].mean(1)
    sc = seed_trace - seed_trace.mean()
    fc = flat - flat.mean(0, keepdims=True)
    num = fc.T @ sc
    den = np.sqrt((fc * fc).sum(0)) * np.sqrt((sc * sc).sum()) + 1e-6
    corr = (num / den).reshape(Z, Y, X)

    region = corr > corr_thr
    if intensity_pct and intensity_pct > 0:                 # remove dim + weakly-correlated halo
        strong = corr >= keep_corr                          # ...but keep dim high-corr (trunk)
        region &= (guide > np.percentile(guide, intensity_pct)) | strong
    if close_x and close_x > 1:                             # fill scan-line notches (along X)
        region = binary_closing(region, structure=np.ones((1, 1, close_x), bool))

    lbl = label(region, connectivity=conn)
    keep = set(np.unique(lbl[core & region])) - {0}        # components touching the seed
    for r in regionprops(lbl):                              # + disconnected high-corr pieces
        if r.area >= min_voxels and float(corr[lbl == r.label].mean()) >= disconnect_corr:
            keep.add(r.label)
    final = np.isin(lbl, list(keep)) if keep else (core & region)
    final = remove_small_objects(final, int(min_voxels), connectivity=conn)
    return final, corr


def detect_from_guide(g, args):
    """Structural candidate mask from a guide volume: high-pass + vesselness +
    seed/candidate hysteresis + intensity grow."""
    if args.denoise and args.denoise > 1:
        g = median_filter(g, size=(1, args.denoise, args.denoise))
    enh = highpass(g)
    vess = sato(enh, sigmas=args.sato, black_ridges=False)
    combo = norm(enh) + args.vess_weight * norm(vess)
    seed = combo > np.percentile(combo, args.seed_pct)
    cand = combo > np.percentile(combo, args.cand_pct)
    cand = keep_touching_seed(cand, seed, args.connectivity)
    if args.grow_iters > 0 and cand.any():
        st = generate_binary_structure(3, args.connectivity)
        hi = g > np.percentile(g, args.grow_pct)
        gg = cand.copy()
        for _ in range(args.grow_iters):
            gg = binary_dilation(gg, st)
        cand = keep_touching_seed(gg & hi, seed, args.connectivity)
    return cand


def cofluc_frame(arr4d, corr_thr=0.45, detrend_win=51, sato_sigmas=(1, 2, 3, 4)):
    """Use co-fluctuation ONLY as a guide for the soma/trunk backbone, then find
    the frame where that backbone is brightest (the strongest global event, when
    branches -- including independent ones -- are most likely also lit).
    Returns (best_frame, backbone_region, scores)."""
    from scipy.ndimage import uniform_filter1d
    tmax = arr4d.max(0).astype(np.float32)
    enh = highpass(tmax)
    vess = sato(enh, sigmas=list(sato_sigmas), black_ridges=False)
    combo = norm(enh) + norm(vess)
    seed = combo > np.percentile(combo, 99)
    T = arr4d.shape[0]
    data = gaussian_filter(arr4d.astype(np.float32), (0, 0.5, 0.5, 0.5))
    data -= uniform_filter1d(data, size=detrend_win, axis=0, mode="nearest")
    flat = data.reshape(T, -1)
    st = flat[:, seed.reshape(-1)].mean(1)
    sc = st - st.mean()
    fc = flat - flat.mean(0, keepdims=True)
    corr = (fc.T @ sc) / (np.sqrt((fc * fc).sum(0)) * np.sqrt((sc * sc).sum()) + 1e-6)
    backbone = corr > corr_thr                      # high corr -> reliable soma/trunk
    if backbone.sum() < 20:
        backbone = seed.reshape(-1)
    scores = arr4d.reshape(T, -1)[:, backbone].sum(1).astype(np.float64)
    return int(np.argmax(scores)), backbone.reshape(arr4d.shape[1:]), scores


def brightest_frame(arr4d, sato_sigmas=(1, 2, 3, 4)):
    """Frame where the DENDRITE is most lit (not just total brightness).

    Build a tubular 'dendrite region' from the temporal-max via high-pass + Sato
    vesselness (which responds to thin tubes, NOT the soma blob or diffuse
    background), then score each frame by the summed intensity INSIDE that region.
    Returns (best_frame_index, region_mask, scores).
    """
    tmax = arr4d.max(0).astype(np.float32)
    enh = highpass(tmax)
    vess = sato(enh, sigmas=list(sato_sigmas), black_ridges=False)
    region = vess > np.percentile(vess, 95)          # dendritic tubes; soma blob excluded
    if region.sum() < 20:
        region = enh > np.percentile(enh, 98)
    flat = arr4d.reshape(arr4d.shape[0], -1)[:, region.reshape(-1)]
    scores = flat.sum(1).astype(np.float64)
    return int(np.argmax(scores)), region, scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiff")
    ap.add_argument("--voxel", nargs=3, type=float, default=[3.9, 1.0, 1.2], metavar=("Z", "Y", "X"))
    ap.add_argument("--period", type=int, default=None, help="scan-line period in px (auto if omitted)")
    ap.add_argument("--no-fix", action="store_true", help="skip scan-line inpainting")
    ap.add_argument("--agg", choices=["max", "p99"], default="max", help="temporal aggregation (guide)")
    ap.add_argument("--guide-frame", type=int, default=None,
                    help="use this single timepoint as the structural guide/seed reference "
                         "instead of the temporal aggregate")
    ap.add_argument("--guide-window", type=int, default=0,
                    help="average +/- this many frames around the guide frame (denoises the "
                         "single-frame structural guide)")
    ap.add_argument("--brightest", action="store_true",
                    help="auto-pick the 'everything fires' frame and build a purely structural "
                         "mask from it (morphology, not correlation-biased)")
    ap.add_argument("--brightest-metric", choices=["dendrite", "cofluc"], default="cofluc",
                    help="dendrite = frame where the tubular dendrite is brightest; "
                         "cofluc = frame where the co-fluctuating soma/trunk backbone is brightest")
    ap.add_argument("--add-backbone", action="store_true",
                    help="union the co-fluctuating soma/trunk backbone into the structural mask "
                         "(recovers the dim trunk that thresholding misses)")
    ap.add_argument("--multiwindow", type=int, default=0,
                    help="detect in this many event windows and UNION them (captures branches "
                         "that fire in different events; the auto_mask approach)")
    ap.add_argument("--window-sep", type=int, default=20,
                    help="min frame separation between event windows")
    ap.add_argument("--sato", nargs="+", type=float, default=[0.5, 1, 2, 3, 4], help="vesselness scales")
    ap.add_argument("--denoise", type=int, default=0,
                    help="median-filter footprint (XY) applied to the guide before enhancement "
                         "(removes speckle; try 2-3)")
    ap.add_argument("--vess-weight", type=float, default=1.0,
                    help="weight of vesselness vs high-pass in the detection score "
                         "(>1 favors thin tubular structure, tightens around the soma)")
    ap.add_argument("--seed-pct", type=float, default=99.0)
    ap.add_argument("--cand-pct", type=float, default=95.0)
    ap.add_argument("--connectivity", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--grow-iters", type=int, default=2, help="intensity-grow dilation iterations")
    ap.add_argument("--grow-pct", type=float, default=90.0)
    ap.add_argument("--min-vol", type=float, default=1500.0)
    ap.add_argument("--all-components", action="store_true", help="keep all (default: largest only)")
    ap.add_argument("--cofluctuate", action="store_true",
                    help="grow the mask to voxels co-fluctuating with the seed (completes trunk/"
                         "branches of the same cell, drops foreign branches)")
    ap.add_argument("--corr-thr", type=float, default=0.35, help="co-fluctuation correlation threshold")
    ap.add_argument("--corr-smooth", type=float, default=0.5, help="spatial smoothing (XY) before correlation")
    ap.add_argument("--detrend-win", type=int, default=51,
                    help="temporal moving-average window removed before correlation (kills bleaching)")
    ap.add_argument("--intensity-gate", type=float, default=75.0,
                    help="keep co-fluctuating voxels above this guide percentile (tightens to the cell; 0=off)")
    ap.add_argument("--keep-corr", type=float, default=0.5,
                    help="voxels with correlation >= this survive the intensity gate (rescues the dim trunk)")
    ap.add_argument("--close-x", type=int, default=5,
                    help="horizontal closing length to fill scan-line notches in the mask")
    ap.add_argument("--disconnect-corr", type=float, default=0.5,
                    help="keep spatially-disconnected pieces whose mean correlation >= this (follows heatmap)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--preview", default=None)
    ap.add_argument("--preview-frame", type=int, default=313,
                    help="single timepoint whose Z-MIP is the preview background (less misleading than temporal-max)")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    arr = tifffile.imread(args.tiff)
    backbone = None
    cofluc_scores = None
    if arr.ndim == 4 and args.brightest:
        if args.brightest_metric == "cofluc":
            bf, backbone, scores = cofluc_frame(arr)
            cofluc_scores = scores
            label_txt = "brightest soma/trunk-backbone frame"
        else:
            bf, _region, scores = brightest_frame(arr)
            label_txt = "brightest-dendrite frame"
        top = np.argsort(scores)[::-1][:5]
        print(f"{label_txt} = {bf}  (top candidates: {top.tolist()})")
        args.guide_frame = bf
        if args.preview_frame is None or args.preview_frame == 313:
            args.preview_frame = bf
    if arr.ndim == 4:
        if args.guide_frame is not None:
            w = max(0, args.guide_window)
            lo = max(0, args.guide_frame - w); hi = min(arr.shape[0], args.guide_frame + w + 1)
            guide = arr[lo:hi].mean(0).astype(np.float32)
        elif args.agg == "max":
            guide = arr.max(0).astype(np.float32)
        else:
            guide = np.percentile(arr, 99, axis=0).astype(np.float32)
    else:
        guide = arr.astype(np.float32)
    Z, Y, X = guide.shape
    print(f"loaded {arr.shape} -> guide volume {guide.shape}")

    # --- scan-line grid + inpaint ---
    dark, prof, period, phase = detect_scanline_grid(guide, args.period)
    print(f"scan-line grid: period={period}px phase={phase} -> {int(dark.sum())} dark columns")
    gfix = guide if args.no_fix else inpaint_columns(guide, dark)
    if args.denoise and args.denoise > 1:
        gfix = median_filter(gfix, size=(1, args.denoise, args.denoise))

    # --- detection: single guide, or multi-window union ---
    if args.multiwindow and arr.ndim == 4:
        if cofluc_scores is None:
            _bf, bb, cofluc_scores = cofluc_frame(arr)
            if backbone is None:
                backbone = bb
        peaks, _ = find_peaks(cofluc_scores, distance=args.window_sep)
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(cofluc_scores))])
        order = peaks[np.argsort(cofluc_scores[peaks])[::-1]][:args.multiwindow]
        w = max(1, args.guide_window or 4)
        cand = np.zeros(arr.shape[1:], bool)
        for pk in order:
            win = arr[max(0, pk - w):pk + w + 1].mean(0).astype(np.float32)
            cand |= detect_from_guide(win, args)
        print(f"multiwindow: {len(order)} event windows at frames {sorted(int(p) for p in order)}")
    else:
        cand = detect_from_guide(gfix, args)

    lbl = label(cand, connectivity=args.connectivity)
    voxvol = float(np.prod(args.voxel))
    props = sorted(regionprops(lbl), key=lambda r: r.area, reverse=True)
    props = [r for r in props if r.area * voxvol >= args.min_vol]
    if not props:
        raise SystemExit("no component passed the volume floor; lower --cand-pct or --min-vol")

    if args.all_components:
        keep_ids = [r.label for r in props]
    else:
        keep_ids = [props[0].label]                            # single largest = one dendrite
    mask = np.isin(lbl, keep_ids)
    if args.add_backbone:
        if backbone is None and arr.ndim == 4:
            _bf, backbone, _s = cofluc_frame(arr)
        if backbone is not None:
            from scipy.ndimage import binary_closing as _bclose
            from skimage.morphology import remove_small_objects as _rso
            mask = mask | backbone
            if args.close_x > 1:                   # bridge trunk gaps along X
                mask = _bclose(mask, structure=np.ones((1, 1, args.close_x), bool))
            mask = _rso(mask, int(args.min_vol / voxvol), connectivity=args.connectivity)
    final = label(mask, connectivity=args.connectivity)

    print(f"core components: {len(keep_ids)}  (largest = {int(props[0].area)} vox "
          f"= {int(props[0].area)*voxvol:.0f} um^3, bbox={props[0].bbox})")

    corr = None
    if args.cofluctuate and arr.ndim == 4:
        core = final > 0
        print(f"co-fluctuation growth: correlating {arr.shape[1]*arr.shape[2]*arr.shape[3]} "
              f"voxels with the seed trace (detrend_win={args.detrend_win}, thr={args.corr_thr})...")
        grown, corr = cofluctuation_grow(arr, core, gfix, args.corr_thr, args.corr_smooth,
                                         args.detrend_win, args.connectivity,
                                         args.intensity_gate, args.keep_corr, args.close_x,
                                         args.disconnect_corr, int(args.min_vol / voxvol))
        final = label(grown, connectivity=args.connectivity)
        nvox = int((final > 0).sum())
        pr = sorted(regionprops(final), key=lambda r: r.area, reverse=True)
        bbox = pr[0].bbox if pr else None
        print(f"after co-fluctuation: {nvox} vox = {nvox*voxvol:.0f} um^3, "
              f"components={int(final.max())}, bbox={bbox}")

    out = args.save or str(Path(args.tiff).with_name(Path(args.tiff).stem + "_dendrite_labelmap.tif"))
    tifffile.imwrite(out, final.astype(np.uint16))
    print(f"saved labelmap -> {out}")

    # --- preview ---
    prev = args.preview or str(Path(args.tiff).with_name(Path(args.tiff).stem + "_extract_preview.png"))
    from matplotlib.colors import ListedColormap
    red = ListedColormap(["red"])
    mmip = final.max(0) > 0

    # background = Z-MIP of a single frame (less misleading than temporal-max)
    if arr.ndim == 4:
        fidx = int(np.clip(args.preview_frame, 0, arr.shape[0] - 1))
        bg = arr[fidx].max(0).astype(np.float32); bg_title = f"frame {fidx} Z-MIP"
    else:
        bg = gfix.max(0); bg_title = "structure Z-MIP"
    blo, bhi = np.percentile(bg, (2, 99.5))
    bgn = np.clip((bg - blo) / (bhi - blo + 1e-6), 0, 1)

    fig, ax = plt.subplots(4, 1, figsize=(13, 8.5))

    # 0) clean single-frame MIP, no overlay (visual reference)
    ax[0].imshow(bgn, cmap="gray", aspect="auto"); ax[0].set_title(f"{bg_title} (reference, no overlay)"); ax[0].axis("off")

    # 1) co-fluctuation correlation heatmap
    if corr is not None:
        ax[1].imshow(corr.max(0), cmap="magma", aspect="auto", vmin=0,
                     vmax=max(0.6, args.corr_thr * 1.5))
        ax[1].set_title(f"seed co-fluctuation correlation (Z-MIP), thr={args.corr_thr}")
    else:
        ax[1].imshow(bgn, cmap="gray", aspect="auto"); ax[1].set_title(bg_title)
    ax[1].axis("off")

    # 2) semitransparent red mask on the single-frame MIP
    ax[2].imshow(bgn, cmap="gray", aspect="auto")
    ax[2].imshow(np.ma.masked_where(~mmip, mmip), cmap=red, alpha=0.45, aspect="auto")
    ax[2].set_title(f"dendrite mask (red) on {bg_title}")
    ax[2].axis("off")

    # 3) scan-line correction QC
    ax[3].plot(prof); ax[3].plot(np.where(dark)[0], prof[dark], "r.", ms=4)
    ax[3].set_title(f"column profile; red = periodic scan-line grid (period {period})"); ax[3].set_xlim(0, X)

    plt.tight_layout(); plt.savefig(prev, dpi=120)
    print(f"saved preview -> {prev}")
    plt.tight_layout(); plt.savefig(prev, dpi=120)
    print(f"saved preview -> {prev}")

    if args.view:
        import napari
        v = napari.Viewer(ndisplay=3)
        v.add_image(gfix, name="structure", scale=tuple(args.voxel), rendering="attenuated_mip")
        v.add_labels(final, name="dendrite", scale=tuple(args.voxel))
        napari.run()


if __name__ == "__main__":
    main()
