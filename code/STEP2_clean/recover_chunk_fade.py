#!/usr/bin/env python3
"""Undo the brightness fade inside each snake-scan chunk (PROTOTYPE, display/masking only).

A Femtonics snake scan is acquired in chunks of `driftLength` µm along the guideline
(24 px in rbp4_140 run03, 28 px in phpebach run05). Inside every chunk the fluorescence
falls to ~15-30 % of its starting value, while the dark offset (no-light level) stays flat.
That is a multiplicative gain on the fluorescence, so it can be divided out:

    corrected = offset + (raw - offset) / gain(position in chunk [, z])

Nothing is interpolated or invented: every voxel keeps its own measurement, rescaled.
The price is noise: the chunk ends collected fewer photons, so after rescaling they are
noisier (reported as `noise_gain`). Per-voxel ΔF/F is unchanged by a pure gain, so the
analysis keeps using the raw stack; this output is for masking and display.

Usage
  python code/STEP2_clean/recover_chunk_fade.py RUN_DIR/runNN_4d.tif --planes 18 \
         [--period 24] [--per-z] [--out DIR]
Writes <stem>_defade.tif (float32, same page layout as the input) and <stem>_fade.json.
`--period` defaults to driftLength / pixelSizeL read from the run's .mesc (via
behavior_imaging_master.csv), then to the autocorrelation of the column profile.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]


def period_from_mesc(run_dir: Path):
    """driftLength / pixelSizeL from the run's .mesc unit, or None."""
    try:
        import h5py
        sys.path.insert(0, str(ROOT / "code/STEP1_extract"))
        from summarize_mesc import parse_json_attr
        session = run_dir.parents[1] if run_dir.parent.name == "preprocessed" else run_dir.parent
        sess = str(session.resolve().relative_to(ROOT))
        run_no = int(run_dir.name.replace("run", ""))
        r = next(r for r in csv.DictReader(open(ROOT / "behavior_imaging_master.csv"))
                 if r["session_dir"] == sess and r.get("behavior_run_number", "").strip() == str(run_no)
                 and r.get("munit"))
        mesc = session / "raw" / r["mesc_file"]
        with h5py.File(mesc, "r") as f:
            sk = r.get("mesc_session") or next(k for k in f if k.startswith("MSession") and r["munit"] in f[k])
            p = next(q for q in parse_json_attr(f[sk][r["munit"]], "MultiROIProtocolJSON")["scanPatterns"]["patterns"]
                     if q.get("scanMode") == 8)
        return float(p["driftLength"]) / float(p["pixelSizeL"]), f"{mesc.name}:{r['munit']}"
    except Exception:
        return None


def period_from_profile(prof: np.ndarray) -> int:
    d = prof - prof.mean()
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    return int(np.argmax(ac[10:60]) + 10)


def estimate_gain(mean_vol: np.ndarray, offset: float, period: int, per_z: bool):
    """Gain map (Z|1, X) for the within-chunk fade.

    The fade is not one fixed shape: early chunks fade to ~20-40 %, middle ones are flat,
    late ones brighten ~2x - in background and cell alike, i.e. instrument gain. So:
      1. probe = median over the tube cross-section of the time-mean, per column
         (the cell is a small fraction of the voxels, so this is the background);
      2. fit each chunk's log(probe) with a quadratic in position-in-chunk;
      3. fit a slow trend across the whole tube (cubic through the chunk means) - that is
         genuine brightness change along the dendrite/depth and is kept;
      4. gain = exp(chunk fit - trend), so only the within-chunk structure is removed.
    """
    Z, Y, X = mean_vol.shape
    sig = np.clip(mean_vol - offset, 1.0, None)
    probes = np.median(sig, axis=1) if per_z else np.median(sig.reshape(-1, X), axis=0)[None]
    xs = np.arange(X)
    out = np.ones((probes.shape[0], X))
    for i, pr in enumerate(probes):
        lp = np.log(pr)
        fit = np.empty(X)
        for c0 in range(0, X, period):
            idx = xs[c0:c0 + period]
            seg = lp[idx]
            # light smoothing inside the chunk only (3-px running mean, edges kept):
            # the probe is a median over hundreds of voxels of a 240-s mean, so it is
            # already low-noise, and the sharp drop in the last 2-3 px must survive.
            if len(seg) >= 3:
                sm = seg.copy()
                sm[1:-1] = (seg[:-2] + seg[1:-1] + seg[2:]) / 3
                seg = sm
            fit[idx] = seg
        centers = np.array([xs[c0:c0 + period].mean() for c0 in range(0, X, period)])
        means = np.array([fit[c0:c0 + period].mean() for c0 in range(0, X, period)])
        trend = np.polyval(np.polyfit(centers, means, min(3, len(centers) - 1)), xs)
        out[i] = np.exp(fit - trend)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", type=Path, help="raw runNN_4d.tif (pages = T*Z planes of Y x X)")
    ap.add_argument("--planes", type=int, required=True, help="Z planes per volume")
    ap.add_argument("--period", type=float, default=None, help="chunk length in px (default: from .mesc)")
    ap.add_argument("--per-z", action="store_true", help="estimate a separate fade per Z plane")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: next to the stack)")
    a = ap.parse_args()

    run_dir = a.stack.parent
    out = a.out or run_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = a.stack.name.replace("_4d.tif", "")
    raw = tifffile.imread(a.stack)
    if raw.ndim == 4:
        raw = raw.reshape(-1, *raw.shape[2:])
    T = raw.shape[0] // a.planes
    raw = raw[:T * a.planes].reshape(T, a.planes, *raw.shape[1:])
    mean_vol = raw.mean(0, dtype=np.float64)
    offset = float(np.median(np.percentile(raw[::max(T // 200, 1)], 10, axis=(0, 1, 2))))

    src = "cli"
    period = a.period
    if period is None:
        got = period_from_mesc(run_dir)
        if got:
            period, src = got
        else:
            period, src = period_from_profile(mean_vol.mean((0, 1))), "column-profile autocorrelation"
    if abs(period - round(period)) > 0.05:
        print(f"warning: non-integer chunk length {period:.2f} px; rounding")
    period = int(round(period))
    X = raw.shape[-1]
    gx = estimate_gain(mean_vol, offset, period, a.per_z)                # (Z|1, X)
    print(f"{stem}: T={T} Z={a.planes} X={X}; chunk {period} px ({src}); offset {offset:.0f}; "
          f"gain {gx.min():.2f}..{gx.max():.2f} -> noise up to x{1 / np.sqrt(gx.min()):.2f} (shot noise) where dimmest")

    meta = {"tool": "recover_chunk_fade.py", "version": "0.1.0-prototype", "period_px": period,
            "period_source": src, "offset": offset, "per_z": a.per_z,
            "gain": np.round(gx, 4).tolist(), "noise_gain_max": float(1 / np.sqrt(gx.min())),
            "use": "masking/display only; analysis uses the raw stack"}
    json.dump(meta, open(out / f"{stem}_fade.json", "w"), indent=1)

    corr = np.empty(raw.shape, np.float32)
    gfull = (gx[:, None, :] if a.per_z else gx[0][None, None, :])        # broadcast to (Z,Y,X)
    for t0 in range(0, T, 100):
        corr[t0:t0 + 100] = offset + (raw[t0:t0 + 100].astype(np.float32) - offset) / gfull
    tifffile.imwrite(out / f"{stem}_defade.tif", corr.reshape(-1, *raw.shape[2:]), bigtiff=True,
                     metadata={"axes": "QYX"})
    print("wrote", out / f"{stem}_defade.tif")


if __name__ == "__main__":
    main()
