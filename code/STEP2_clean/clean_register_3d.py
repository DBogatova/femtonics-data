#!/usr/bin/env python3
"""
clean_register_3d.py - Step 1-2 of the 3D dendrite pipeline.

  1. remove the periodic snake/resonant-scan dark columns in EVERY frame
     (vectorized linear inpainting along X)
  2. 3D motion correction: register each timepoint to a temporal-median
     reference by translation (phase cross-correlation + shift) so the
     dendrite stays centered

Outputs next to the input:
  *_clean_reg.tif   - cleaned + registered 4D stack (T,Z,Y,X)
  *_reg_qc.png      - drift-over-time + before/after temporal-mean QC
  *_drift.csv       - per-frame (dz,dy,dx) shifts

Usage:
  .venv/bin/python code/Preprocessing/clean_register_3d.py \
      rbp4_139_phpeb/06-12-2026/traces/run5/3dstack.tif
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import median_filter, shift as nd_shift
from scipy.signal import find_peaks
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
import sys as _psys, pathlib as _ppl
_psys.path.insert(0, str(_ppl.Path(__file__).resolve().parents[1]))
from common.progress import progress


def scanline_grid(vol, period=None):
    prof = np.median(vol, axis=(0, 1)).astype(np.float32)
    X = len(prof)
    neg = median_filter(prof, size=9) - prof
    dips, _ = find_peaks(neg, distance=12, prominence=15)
    if period is None:
        period = None
        if len(dips) > 2:
            sp = np.diff(dips)
            keep = sp[np.abs(sp - np.median(sp)) <= 3]
            # an empty selection makes np.median return NaN, which int() rejects;
            # fall through to the default rather than raising
            if keep.size:
                period = int(np.median(keep))
        if period is None or period < 2:
            period = 24
    phase = int(dips[0]) % period if len(dips) else 0
    dark = np.zeros(X, bool)
    for c in np.arange(phase, X, period):
        dark[max(0, c - 1):min(X, c + 2)] = True
    return dark, period, phase


def inpaint_all_frames(stack, dark):
    """Vectorized linear interp of dark X-columns for the whole 4D stack."""
    X = stack.shape[-1]
    good = np.where(~dark)[0]
    dark_idx = np.where(dark)[0]
    lefts, rights, wl, wr = [], [], [], []
    for xd in dark_idx:
        pos = np.searchsorted(good, xd)
        l = good[pos - 1] if pos > 0 else good[0]
        r = good[pos] if pos < len(good) else good[-1]
        if r == l:
            wl.append(1.0); wr.append(0.0)
        else:
            wl.append((r - xd) / (r - l)); wr.append((xd - l) / (r - l))
        lefts.append(l); rights.append(r)
    lefts = np.array(lefts); rights = np.array(rights)
    wl = np.array(wl, np.float32); wr = np.array(wr, np.float32)
    out = stack.astype(np.float32).copy()
    out[..., dark_idx] = wl * out[..., lefts] + wr * out[..., rights]
    return out


def frame_sharpness(clean):
    """Gradient-energy sharpness of each frame's Z-MIP (higher = sharper)."""
    from scipy.ndimage import sobel
    s = np.zeros(clean.shape[0], np.float32)
    for t in range(clean.shape[0]):
        m = clean[t].max(0)
        gx = sobel(m, axis=1); gy = sobel(m, axis=0)
        s[t] = float(np.mean(gx * gx + gy * gy))
    return s


def build_reference(clean, args):
    if getattr(args, "ref_tif", None):
        r = tifffile.imread(args.ref_tif).astype(np.float32)
        if r.ndim == 4:
            r = r.mean(0)
        assert r.shape == clean.shape[1:], f"ref {r.shape} != volume {clean.shape[1:]}"
        return r, f"external {Path(args.ref_tif).name}"
    if args.ref_frame is not None:
        return clean[args.ref_frame], f"frame {args.ref_frame}"
    if args.ref_range is not None:
        a, b = args.ref_range
        return clean[a:b + 1].mean(0), f"mean of frames {a}-{b}"
    return np.median(clean, axis=0), "temporal median"


def register_all(clean, ref, upsample, max_shift):
    T = clean.shape[0]
    reg = np.empty_like(clean)
    shifts = np.zeros((T, 3), np.float32)
    for t in range(T):
        progress(t, T, "register: motion correction")
        sh, _, _ = phase_cross_correlation(ref, clean[t], upsample_factor=upsample,
                                           normalization=None)
        sh = np.clip(sh, -max_shift, max_shift)
        shifts[t] = sh
        reg[t] = nd_shift(clean[t], sh, order=1, mode="nearest")
    return reg, shifts


def stabilized_score(reg):
    """Sharpness (gradient energy) of the temporal-mean Z-MIP; higher = better centered."""
    from scipy.ndimage import sobel
    m = reg.mean(0).max(0)
    gx = sobel(m, axis=1); gy = sobel(m, axis=0)
    return float(np.mean(gx * gx + gy * gy))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiff")
    ap.add_argument("--period", type=int, default=None)
    ap.add_argument("--upsample", type=int, default=10, help="subpixel registration factor")
    ap.add_argument("--max-shift", type=float, default=8.0,
                    help="clip per-frame shift (voxels) to reject outliers")
    ap.add_argument("--ref-frame", type=int, default=None,
                    help="use this frame index as the registration reference")
    ap.add_argument("--ref-range", nargs=2, type=int, default=None, metavar=("A", "B"),
                    help="use the mean of frames A..B as the reference")
    ap.add_argument("--ref-tif", default=None,
                    help="external reference volume (Z,Y,X) or 4D to average; e.g. an anatomical stack")
    ap.add_argument("--two-pass", action="store_true",
                    help="refine: register, rebuild reference from registered mean, re-register")
    ap.add_argument("--compare-refs", action="store_true",
                    help="clean once, try several references, report sharpness, save the best")
    ap.add_argument("--robust", action="store_true",
                    help="replace outlier per-frame shifts (failed registrations) via local median")
    ap.add_argument("--no-register", action="store_true",
                    help="only clean scan lines, skip motion correction (no interpolation/warping)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arr = tifffile.imread(args.tiff)
    assert arr.ndim == 4, f"expected 4D (T,Z,Y,X), got {arr.shape}"
    T, Z, Y, X = arr.shape
    print(f"loaded {arr.shape} {arr.dtype}")

    # --- 1) scan-line cleaning (every frame) ---
    guide = arr.max(0).astype(np.float32)
    dark, period, phase = scanline_grid(guide, args.period)
    print(f"scan-line grid: period={period} phase={phase} -> {int(dark.sum())} dark columns; inpainting all {T} frames")
    clean = inpaint_all_frames(arr, dark)
    del arr

    if args.no_register:
        stem = Path(args.tiff).with_suffix("")
        out = args.out or f"{stem}_clean.tif"
        tifffile.imwrite(out, np.clip(clean, 0, 65535).astype(np.uint16), metadata={"axes": "TZYX"})
        print(f"scan-line cleaned only (NO registration) -> {out}")
        return

    # --- 2) 3D motion correction ---
    sharp = frame_sharpness(clean)
    top = np.argsort(sharp)[::-1][:10]
    print("sharpest candidate reference frames (index):", top.tolist())
    print(f"unregistered baseline sharpness = {stabilized_score(clean):.1f}")

    if args.compare_refs:
        best_frame = int(top[0])
        candidates = {
            "median": np.median(clean, axis=0),
            f"sharpest-frame-{best_frame}": clean[best_frame],
        }
        results = {}
        for name, r in candidates.items():
            reg_c, sh_c = register_all(clean, r, args.upsample, args.max_shift)
            sc = stabilized_score(reg_c)
            results[name] = (sc, reg_c, sh_c)
            print(f"  ref={name:<20} sharpness={sc:.1f}")
        # two-pass on top of the median result
        reg_m = results["median"][1]
        reg_tp, sh_tp = register_all(clean, reg_m.mean(0), args.upsample, args.max_shift)
        sc_tp = stabilized_score(reg_tp)
        results["median+two-pass"] = (sc_tp, reg_tp, sh_tp)
        print(f"  ref={'median+two-pass':<20} sharpness={sc_tp:.1f}")
        best = max(results, key=lambda k: results[k][0])
        print(f"BEST reference = {best} (sharpness {results[best][0]:.1f})")
        _, reg, shifts = results[best]
        refdesc = best
    else:
        ref, refdesc = build_reference(clean, args)
        print(f"registering to reference = {refdesc} ...")
        reg, shifts = register_all(clean, ref, args.upsample, args.max_shift)
        if args.two_pass:
            print("two-pass: rebuilding reference from registered mean and re-registering ...")
            reg, shifts = register_all(clean, reg.mean(0), args.upsample, args.max_shift)

    if args.robust:
        from scipy.ndimage import median_filter as mf1
        sm = np.vstack([mf1(shifts[:, k], size=5) for k in range(3)]).T
        resid = shifts - sm
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid, 0)), 0) + 1e-6
        bad = (np.abs(resid) > 4 * mad).any(1) | (np.abs(shifts).max(1) >= args.max_shift - 1e-3)
        print(f"robust: replacing {int(bad.sum())} outlier frames with local-median shifts")
        for t in np.where(bad)[0]:
            shifts[t] = sm[t]
            reg[t] = nd_shift(clean[t], sm[t], order=1, mode="nearest")
        print(f"stabilized sharpness (robust) = {stabilized_score(reg):.1f}")

    print(f"stabilized sharpness ({refdesc}) = {stabilized_score(reg):.1f}")
    drift = shifts - shifts.mean(0)
    print(f"drift range (vox): z={np.ptp(drift[:,0]):.2f} y={np.ptp(drift[:,1]):.2f} x={np.ptp(drift[:,2]):.2f}")
    print(f"drift std   (vox): z={drift[:,0].std():.2f} y={drift[:,1].std():.2f} x={drift[:,2].std():.2f}")

    stem = Path(args.tiff).with_suffix("")
    out = args.out or f"{stem}_clean_reg.tif"
    tifffile.imwrite(out, np.clip(reg, 0, 65535).astype(np.uint16), metadata={"axes": "TZYX"})
    print(f"saved -> {out}")

    with open(f"{stem}_drift.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "dz", "dy", "dx"])
        for t in range(T):
            w.writerow([t, *np.round(shifts[t], 3)])

    # --- QC figure ---
    fig, ax = plt.subplots(3, 1, figsize=(12, 7))
    ax[0].plot(shifts[:, 0], label="dz"); ax[0].plot(shifts[:, 1], label="dy"); ax[0].plot(shifts[:, 2], label="dx")
    ax[0].legend(fontsize=8); ax[0].set_title("per-frame registration shift (voxels)"); ax[0].set_xlabel("frame")
    lo, hi = np.percentile(clean.mean(0).max(0), (2, 99.5))
    nrm = lambda im: np.clip((im - lo) / (hi - lo + 1e-6), 0, 1)
    ax[1].imshow(nrm(clean.mean(0).max(0)), cmap="gray", aspect="auto")
    ax[1].set_title("temporal-mean Z-MIP BEFORE registration"); ax[1].axis("off")
    ax[2].imshow(nrm(reg.mean(0).max(0)), cmap="gray", aspect="auto")
    ax[2].set_title("temporal-mean Z-MIP AFTER registration (sharper = better centered)"); ax[2].axis("off")
    plt.tight_layout(); plt.savefig(f"{stem}_reg_qc.png", dpi=120)
    print(f"saved QC -> {stem}_reg_qc.png")


if __name__ == "__main__":
    main()
