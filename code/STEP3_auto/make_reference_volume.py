#!/usr/bin/env python3
"""make_reference_volume.py - build a HIGH-QUALITY STATIC 3D volume from a 4D run.

WHY THIS EXISTS
---------------
Traces are analyzed on the RAW dynamic data (denoisers can distort amplitudes,
so they stay out of analysis). Reconstruction and masking, however, only need a
static picture of the anatomy - and a static picture can average the ENTIRE
recording: ~1400-1800 frames gives a ~37-42x noise reduction, far beyond what
any per-frame denoiser achieves, with zero risk of hallucinated structure
because it is plain arithmetic on the real data.

Drift check first: aggregation is only valid if the (already registered) stack
is stationary. Early-vs-late phase correlation must be <= --max-drift voxels,
otherwise the tool refuses (rather than silently writing a blurred volume).

OUTPUT: <stem>_ref3d.tif - a 3-channel 3D volume (C,Z,Y,X), ImageJ-compatible:
  C0 anatomy   : temporal MEAN (the sharp structural scaffold)
  C1 activity  : temporal 99.5th percentile (where activity EVER happened -
                 reveals dim branches that fire rarely; complements C0)
  C2 coherence : max neighbor temporal correlation (structure vs haze:
                 active processes co-fluctuate, scattered-light halo less so)
All channels rescaled to uint16 full range, with the raw scaling factors saved
in <stem>_ref3d.json for reproducibility.

Use it for: napari 3D rendering, guideline drawing, mask review backgrounds,
figure panels. The autoseg pipeline computes the same statistics internally;
this simply materialises them as a viewable volume.

Usage:
  python code/STEP3_auto/make_reference_volume.py <stack.tif> [more.tif ...]
  options: --tchunk 200 --max-drift 1.5 --out-dir DIR --pct 99.5
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import tifffile
from numpy.fft import fftn, ifftn
import sys as _psys, pathlib as _ppl
_psys.path.insert(0, str(_ppl.Path(__file__).resolve().parents[1]))
from common.progress import progress


def drift_voxels(store, T):
    a = np.asarray(store[: T // 3]).astype(np.float32).mean(0)
    b = np.asarray(store[-(T // 3):]).astype(np.float32).mean(0)
    F = fftn(a) * np.conj(fftn(b))
    r = np.abs(ifftn(F / (np.abs(F) + 1e-9)))
    pk = np.unravel_index(np.argmax(r), r.shape)
    return [p if p <= d // 2 else p - d for p, d in zip(pk, a.shape)]


def build(stack_path: Path, args) -> Path:
    store = tifffile.memmap(stack_path, mode="r")
    if store.ndim != 4:
        raise ValueError(f"expected 4D (T,Z,Y,X), got {store.shape}")
    T, Z, Y, X = store.shape
    print(f"[{stack_path.name}] T={T} vol={Z}x{Y}x{X}")

    d = drift_voxels(store, T)
    print(f"  drift check (z,y,x): {d} voxels")
    zshift_per_block = None
    if max(abs(int(v)) for v in d) > args.max_drift:
        if not args.register_blocks:
            raise RuntimeError(
                f"drift {d} exceeds --max-drift {args.max_drift}; register the "
                f"stack first (STEP2_clean), or pass --register-blocks for "
                f"rigid per-block drift correction here")
        # Rigid block alignment: split time into blocks, phase-align each
        # block's mean volume to the first block, then aggregate the ALIGNED
        # frames. Integer-voxel shifts only (conservative; no interpolation
        # smearing). Corrects slow axial drift, not fast motion.
        nb = args.n_blocks
        edges = np.linspace(0, T, nb + 1).astype(int)
        ref0 = np.asarray(store[edges[0]:edges[1]]).astype(np.float32).mean(0)
        zshift_per_block = []
        for i in range(nb):
            blk = np.asarray(store[edges[i]:edges[i+1]]).astype(np.float32).mean(0)
            F = fftn(ref0) * np.conj(fftn(blk))
            r = np.abs(ifftn(F / (np.abs(F) + 1e-9)))
            pk = np.unravel_index(np.argmax(r), r.shape)
            sh = [p if p <= dd // 2 else p - dd for p, dd in zip(pk, ref0.shape)]
            zshift_per_block.append([int(v) for v in sh])
        print(f"  --register-blocks: per-block shifts (z,y,x) = {zshift_per_block}")

    mean = np.zeros((Z, Y, X), np.float64)
    m2 = np.zeros((Z, Y, X), np.float64)
    amax = np.zeros((Z, Y, X), np.float32)
    def load_aligned(a, b):
        c = np.asarray(store[a:b], np.float32)
        if zshift_per_block is not None:
            nb = len(zshift_per_block)
            edges = np.linspace(0, T, nb + 1).astype(int)
            blk = min(int(np.searchsorted(edges, a, side='right')) - 1, nb - 1)
            sz, sy, sx = zshift_per_block[blk]
            c = np.roll(c, (sz, sy, sx), axis=(1, 2, 3))
        return c

    for a in range(0, T, args.tchunk):
        progress(a, 3 * T, "reference: mean / percentile")
        b = min(a + args.tchunk, T)
        c = load_aligned(a, b)
        mean += c.sum(0)
        amax = np.maximum(amax, np.percentile(c, args.pct, axis=0).astype(np.float32))
    mean /= T
    covx = np.zeros((Z, Y, X)); covy = np.zeros((Z, Y, X))
    for a in range(0, T, args.tchunk):
        progress(T + a, 3 * T, "reference: neighbour correlation")
        b = min(a + args.tchunk, T)
        c = load_aligned(a, b) - mean
        m2 += (c ** 2).sum(0)
        covx[:, :, :-1] += (c[:, :, :, :-1] * c[:, :, :, 1:]).sum(0)
        covy[:, :-1, :] += (c[:, :, :-1, :] * c[:, :, 1:, :]).sum(0)
    sd = np.sqrt(m2 / T) + 1e-6
    corr = np.zeros_like(mean)
    corr[:, :, :-1] = np.maximum(corr[:, :, :-1],
                                 (covx[:, :, :-1] / T) / (sd[:, :, :-1] * sd[:, :, 1:]))
    corr[:, :-1, :] = np.maximum(corr[:, :-1, :],
                                 (covy[:, :-1, :] / T) / (sd[:, :-1, :] * sd[:, 1:, :]))

    # co-firing channel: mean of the top-N distinct network-event frames, i.e. the
    # moments the whole tree lights up at once. Unlike the percentile channel (each
    # voxel at its own brightest instant) this is a physically consistent snapshot,
    # so branch boundaries stay crisp. Global trace = mean over structure voxels
    # (top 5% of the percentile map); peaks >= 5 frames apart.
    from scipy.signal import find_peaks
    struct = amax > np.percentile(amax, 95)
    trace = np.zeros(T, np.float32)
    for a in range(0, T, args.tchunk):
        progress(2 * T + a, 3 * T, "reference: co-firing frames")
        b = min(a + args.tchunk, T)
        trace[a:b] = load_aligned(a, b)[:, struct].mean(1)
    f0 = np.percentile(trace, 10); dff = (trace - f0) / max(f0, 1e-6)
    pk, _ = find_peaks(dff, distance=5)
    if len(pk) == 0:
        pk = np.arange(T)
    top = np.sort(pk[np.argsort(dff[pk])[::-1][:args.cofire_n]])
    cofire = np.zeros((Z, Y, X), np.float64)
    for t in top:
        cofire += load_aligned(int(t), int(t) + 1)[0]
    cofire /= len(top)
    cofire_frames = [int(t) for t in top]

    chans, scale = [], {}
    for name, v in (("anatomy_mean", mean), ("activity_p%g" % args.pct, amax),
                    ("neighbour_corr", np.clip(corr, 0, 1)), ("cofire_mean", cofire)):
        lo, hi = float(np.min(v)), float(np.max(v))
        u = ((v - lo) / (hi - lo + 1e-12) * 65535).astype(np.uint16)
        chans.append(u)
        scale[name] = {"raw_min": lo, "raw_max": hi}
    # ImageJ hyperstacks require TZCYXS axis order, so store as (Z,C,Y,X)
    ref = np.stack(chans, axis=1)  # (Z,C,Y,X)

    out_dir = Path(args.out_dir) if args.out_dir else stack_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stack_path.stem
    out = out_dir / f"{stem}_ref3d.tif"
    tifffile.imwrite(out, ref, imagej=True,
                     metadata={"axes": "ZCYX",
                               "Labels": list(scale.keys())})
    (out_dir / f"{stem}_ref3d.json").write_text(json.dumps(
        {"source": str(stack_path), "T_frames_aggregated": int(T),
         "drift_zyx": [int(v) for v in d],
         "block_shifts_zyx": zshift_per_block, "channels": scale,
         "cofire_frames": cofire_frames,
         "noise_reduction_vs_single_frame": f"~{np.sqrt(T):.0f}x (mean channel)"},
        indent=2))
    print(f"  wrote {out}  ({out.stat().st_size/1e6:.0f} MB, 3ch x {Z}x{Y}x{X})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stacks", nargs="+")
    ap.add_argument("--tchunk", type=int, default=200)
    ap.add_argument("--pct", type=float, default=99.5)
    ap.add_argument("--max-drift", type=float, default=1.5,
                    help="refuse if early-vs-late drift exceeds this (voxels)")
    ap.add_argument("--cofire-n", type=int, default=12,
                    help="number of distinct network-event frames averaged into the "
                         "cofire_mean channel (default 12)")
    ap.add_argument("--register-blocks", action="store_true",
                    help="rigid per-block drift correction before aggregating")
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    ok = 0
    for s in args.stacks:
        try:
            build(Path(s), args)
            ok += 1
        except Exception as exc:
            print(f"[FAIL] {s}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"done: {ok} ok, {len(args.stacks)-ok} failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
