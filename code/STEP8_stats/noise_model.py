#!/usr/bin/env python
"""noise_model.py - learn from the traces which runs are noisy / not analyzable.

Every processed run (a clean stack plus any cell mask: Daria's hand mask in the real
tree, else the automatic mask of the auto_pipeline mirror) gets a set of trace and stack
features computed from the RAW cleaned stack (<stem>_clean.tif). Two scores come out:

  1. supervised: probability "set aside" from a small, explainable model (L1 logistic
     regression on standardized features; a depth-2 tree as a cross-check), validated
     leave-one-MOUSE-out. Labels are weak:
        good (0)  = runs in Daria's curated cohort (stats/cohort_metrics.csv)
        bad  (1)  = runs Daria set aside in run_marks.csv (excluded / revisit), with her
                    own reason ('very noisy', 'pretty noisy', ...) or no reason given.
     Automatic marks ([auto-QC], [z-QC]) are a SEPARATE label source, modelled
     separately and never mixed into Daria's labels.
  2. unsupervised: robust z of each feature against the good set (median / MAD, the
     run's own mouse left out of the reference), combined into a 'badness' score and a
     shrinkage Mahalanobis distance. Works without any labels.

No circularity: no feature uses correlation with the soma (or any region) or behavior.
The only correlation used is the split-half reliability of the cell's OWN trace (two
interleaved halves of the same mask).

Outputs (stats/noise_model/):
  features.csv     one row per run, all features + labels
  scores.csv       probability noisy (LOMO), unsupervised scores, top reasons in words
  model_card.json  label definitions, features, coefficients, AUC, confusion
  fig_*.pdf/png    feature strips, ROC, coefficients, score map
  summary.md       plain-language summary, disagreements with the marks
  features/<key>.json  per-run feature cache (recomputed when stack or mask change)

  python code/STEP8_stats/noise_model.py                # features (cached) + models
  python code/STEP8_stats/noise_model.py --recompute    # ignore the feature cache
  python code/STEP8_stats/noise_model.py --workers 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
AUTO_ROOT = PROJECT / "auto_pipeline"
OUT = PROJECT / "stats" / "noise_model"
LOG = AUTO_ROOT / "logs" / "noise_learner_v7.jsonl"
sys.path.insert(0, str(PROJECT / "code")); sys.path.insert(0, str(PROJECT / "code/STEP7_workflow"))
sys.path.insert(0, str(PROJECT / "code/STEP1_extract")); sys.path.insert(0, str(HERE))

__version__ = "1.0.0"
FEATURE_VERSION = "1.1"


def log(what, result):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "stage": "noise_learner",
                            "what": what, "result": result}, default=str) + "\n")


# ----------------------------------------------------------------------------- runs

def _mask_kind(run_dir: Path, stem: str) -> str:
    """'hand' if Daria edited the reviewed mask (any non-auto tool), else 'auto'."""
    j = run_dir / f"{stem}_autoseg_reviewed.json"
    try:
        rev = json.loads(j.read_text()).get("reviews", [])
        tools = [r.get("tool") for r in rev]
        return "hand" if any(t and t != "auto_mask" for t in tools) else "auto"
    except Exception:
        return "unknown"


def collect_runs() -> list[dict]:
    """One entry per run (behavior_base, or mouse_munit for imaging-only runs) that has a
    clean stack and a cell mask in the real tree or the mirror. Mask preference: hand mask
    (real tree) > reviewed mask in the real tree > mirror reviewed mask > raw auto mask."""
    from femto_status import build_status
    trees = {"real": build_status(PROJECT), "mirror": build_status(AUTO_ROOT)}
    runs: dict[str, dict] = {}
    for tree, rows in trees.items():
        for r in rows:
            key = r.get("behavior_base") or f"{r['mouse']}_{r.get('munit', '?')}"
            if not r.get("run_dir") or not r.get("stem"):
                continue
            d = Path(r["run_dir"]); s = r["stem"]; stack = d / f"{s}.tif"
            if not stack.exists():
                continue
            e = runs.setdefault(key, {"key": key, "mouse": r["mouse"], "date": r["date"], "munit": r.get("munit"),
                                      "frame_rate_hz": r.get("frame_rate_hz"), "voxel_zyx_um": r.get("voxel_zyx_um"),
                                      "stack": None, "trees": [], "masks": []})
            e["trees"].append(tree)
            if e["stack"] is None or tree == "real":
                e["stack"] = str(stack); e["run_dir_" + tree] = str(d.relative_to(PROJECT))
            else:
                e["run_dir_" + tree] = str(d.relative_to(PROJECT))
            rev = d / f"{s}_autoseg_labelmap_reviewed.tif"; raw = d / f"{s}_autoseg_labelmap.tif"
            exc = d / f"{s}_exclude_labelmap.tif"
            if rev.exists():
                kind = _mask_kind(d, s)
                prio = {("real", "hand"): 0, ("mirror", "hand"): 1, ("real", "auto"): 2, ("real", "unknown"): 2,
                        ("mirror", "auto"): 3, ("mirror", "unknown"): 3}[(tree, kind)]
                e["masks"].append((prio, str(rev), kind, tree, str(exc) if exc.exists() else None))
            elif raw.exists():
                e["masks"].append((4 if tree == "real" else 5, str(raw), "auto_unreviewed", tree,
                                   str(exc) if exc.exists() else None))
    out = []
    for e in runs.values():
        if not e["masks"]:
            e["skip"] = "no mask in the real tree or the mirror"
            out.append(e); continue
        prio, m, kind, tree, exc = sorted(e["masks"])[0]
        if exc is None:                                   # exclude map from the other tree if only there
            exc = next((x[4] for x in sorted(e["masks"]) if x[4]), None)
        e.update({"mask": m, "mask_kind": kind, "mask_tree": tree, "exclude": exc})
        e.pop("masks"); out.append(e)
    return out


# ----------------------------------------------------------------------- features

def _dff(t, pct=10.0):
    f0 = np.percentile(t, pct); return (t - f0) / max(abs(f0), 1e-6)


def _robust_noise(t, w=15):
    from scipy import ndimage as ndi
    hp = t - ndi.uniform_filter1d(t, w)
    return 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-12


def _acf(x, lag):
    x = x - x.mean()
    if lag <= 0 or lag >= len(x):
        return float("nan")
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def _spectrum_feats(x, rate, prefix):
    from scipy.signal import welch
    from scipy.ndimage import median_filter
    T = len(x); out = {}
    nps = int(min(512, max(64, T // 4)))
    f, p = welch(x - x.mean(), fs=rate, nperseg=nps)
    top = min(2.2, rate / 2 * 0.98)                 # common band: lowest Nyquist in the data ~2.26 Hz
    band = (f >= 0.02) & (f <= top); hf = (f >= 1.0) & (f <= top)
    out[prefix + "hf_frac"] = float(p[hf].sum() / max(p[band].sum(), 1e-30))
    sl = (f >= 0.05) & (f <= 2.0) & (p > 0)
    out[prefix + "psd_slope"] = float(np.polyfit(np.log10(f[sl]), np.log10(p[sl]), 1)[0]) if sl.sum() > 4 else float("nan")
    # periodic artifacts: narrow peaks above the smooth spectrum (whole Nyquist range)
    lp = np.log10(np.maximum(p, 1e-30)); base = median_filter(lp, size=15, mode="nearest")
    ex = lp - base; ok = f > 0.05
    out[prefix + "comb_excess_log10"] = float(ex[ok].max()) if ok.any() else float("nan")
    out[prefix + "comb_peak_hz"] = float(f[ok][np.argmax(ex[ok])]) if ok.any() else float("nan")
    out[prefix + "comb_n_peaks"] = int(np.sum(ex[ok] > 0.5))      # > ~3.2x the local spectrum
    return out


def _events(t, prom_frac=0.2, min_dist=5):
    from scipy.signal import find_peaks
    pk, _ = find_peaks(t, prominence=prom_frac * (t.max() - t.min()), distance=min_dist)
    return pk


def _split_half(flat, idx, seed=0):
    if len(idx) < 8:
        return float("nan")
    rng = np.random.default_rng(seed); perm = rng.permutation(idx)
    a = _dff(flat[:, perm[0::2]].mean(1, dtype=np.float64)); b = _dff(flat[:, perm[1::2]].mean(1, dtype=np.float64))
    r = float(np.corrcoef(a, b)[0, 1])
    return float(2 * r / (1 + r)) if r > -0.99 else float("nan")


def _burst_robust(d, noise, rate, prefix):
    """snr_top10: median prominence of the 10 largest distinct events (>= 2 s apart) over
    noise - one end-of-run burst cannot carry it. active_block_frac: share of 30-s blocks
    containing an event > 5 noise SD (activity spread over the run vs one burst)."""
    from scipy.signal import find_peaks
    pk, pr = find_peaks(d, prominence=0, distance=max(1, int(2 * rate)))
    proms = np.sort(pr["prominences"])[::-1][:10] if len(pk) else np.array([0.0])
    big, _ = find_peaks(d, prominence=5 * noise, distance=max(1, int(rate)))
    w = max(5, int(30 * rate)); nb = max(1, len(d) // w)
    blocks = {min(int(p // w), nb - 1) for p in big}
    return {prefix + "snr_top10": float(np.median(proms) / noise), prefix + "active_block_frac": float(len(blocks) / nb)}


def _lf_frac(d, rate):
    """share of trace power below 0.05 Hz (slow drift / wander), band 0.004-2.2 Hz."""
    from scipy.signal import welch
    f, p = welch(d - d.mean(), fs=rate, nperseg=min(len(d), int(256 * rate / 5)))
    band = (f > 0) & (f <= min(2.2, rate / 2 * 0.98)); lo = band & (f < 0.05)
    return float(p[lo].sum() / max(p[band].sum(), 1e-30))


def compute_features(run: dict) -> dict:
    import tifffile
    from scipy import ndimage as ndi
    from skimage.registration import phase_cross_correlation
    from z_redundancy_qc import z_metrics
    stack = tifffile.imread(run["stack"])
    if stack.ndim != 4:
        return {"error": f"stack ndim {stack.ndim}"}
    T, Z, Y, X = stack.shape
    try:
        rate = float(run.get("frame_rate_hz") or "nan")
    except ValueError:
        rate = float("nan")
    if not np.isfinite(rate) or rate <= 0:
        return {"error": "no frame rate"}
    try:
        vox = np.array([float(v) for v in str(run.get("voxel_zyx_um")).split("/")], float)
        assert vox.size == 3
    except Exception:
        from common.voxel import resolve_voxel
        try:
            vox = np.array(resolve_voxel(Path(run["stack"]), None, quiet=True), float)
        except SystemExit:
            vox = np.array([np.nan] * 3)
    mask = tifffile.imread(run["mask"]) > 0
    excl = tifffile.imread(run["exclude"]) > 0 if run.get("exclude") else np.zeros_like(mask)
    if mask.shape != (Z, Y, X) or excl.shape != (Z, Y, X):
        return {"error": f"mask shape {mask.shape} / exclude {excl.shape} vs stack {(Z, Y, X)}"}
    cell = mask & ~excl
    if cell.sum() < 20:
        return {"error": f"mask has {int(cell.sum())} voxels"}
    near = ndi.binary_dilation(mask, iterations=2) | ndi.binary_dilation(excl, iterations=1)
    bg = ~near
    flat = stack.reshape(T, -1)
    ci = np.flatnonzero(cell.ravel()); bi = np.flatnonzero(bg.ravel())
    F = flat[:, ci].mean(1, dtype=np.float64)
    Fb = flat[:, bi].mean(1, dtype=np.float64) if bi.size else np.full(T, np.nan)
    Ft = flat.mean(1, dtype=np.float64)
    d = _dff(F); minutes = T / rate / 60.0
    f = {"T": int(T), "nz": int(Z), "ny": int(Y), "nx": int(X), "frame_rate_hz": rate, "duration_s": T / rate,
         "voxel_z_um": vox[0], "voxel_y_um": vox[1], "voxel_x_um": vox[2],
         "mask_voxels": int(cell.sum()), "mask_fill_frac": float(cell.mean()), "bg_voxels": int(bi.size)}
    # amplitude / noise (PI hypothesis quantities, on the whole-cell trace)
    nz_ = _robust_noise(d)
    f.update({"cell_F_raw": float(F.mean()), "bg_F_raw": float(np.nanmean(Fb)),
              "dff_noise_sd": float(nz_), "dff_sd": float(d.std()),
              "dff_baseline_sd": float(d[d <= np.median(d)].std()),
              "dff_p99": float(np.percentile(d, 99)), "dff_max": float(d.max()),
              "snr_p99": float(np.percentile(d, 99) / nz_),
              "noise_over_amp": float(nz_ / max(np.percentile(d, 99), 1e-9)),
              "dff_skew": float(pd.Series(d).skew()), "dff_kurt": float(pd.Series(d).kurt()),
              "reliability_split_half": _split_half(flat, ci)})
    # signal to background
    f["sig_to_bg"] = float((F.mean() - np.nanmean(Fb)) / max(np.nanmean(Fb), 1e-6))
    # rate-free and burst-robust versions (one huge burst makes p99 look great on a flat run;
    # per-frame noise grows with sqrt(volume rate) at fixed photon flux)
    f["noise_1s"] = float(nz_ / np.sqrt(rate))
    f.update(_burst_robust(d, nz_, rate, ""))
    f["lf_frac"] = _lf_frac(d, rate)
    f["baseline_wander"] = float(f["dff_baseline_sd"] / nz_)
    # pseudo-regions: the mask cut into 4 equal X chunks (what a region trace looks like;
    # each chunk's OWN trace only - no between-chunk correlation)
    xs = np.argwhere(cell)[:, 2]; edges = np.quantile(xs, [0, .25, .5, .75, 1.0])
    ch = []
    for i in range(4):
        sel_ = cell.copy(); xx = np.arange(X)
        keep = (xx >= edges[i]) & ((xx < edges[i + 1]) if i < 3 else (xx <= edges[i + 1]))
        sel_[:, :, ~keep] = False
        idx = np.flatnonzero(sel_.ravel())
        if idx.size < 10:
            continue
        dc = _dff(flat[:, idx].mean(1, dtype=np.float64)); nc_ = _robust_noise(dc)
        ch.append({"noise_1s": nc_ / np.sqrt(rate), "skew": float(pd.Series(dc).skew()),
                   **_burst_robust(dc, nc_, rate, ""), "wander": float(dc[dc <= np.median(dc)].std() / nc_)})
    if ch:
        c = pd.DataFrame(ch).median()
        f.update({"chunk_noise_1s": float(c["noise_1s"]), "chunk_skew": float(c["skew"]),
                  "chunk_snr_top10": float(c["snr_top10"]), "chunk_active_block_frac": float(c["active_block_frac"]),
                  "chunk_wander": float(c["wander"])})
    # temporal structure
    hp = d - ndi.uniform_filter1d(d, 15)
    f.update({"ac1": _acf(d, 1), "ac_0p5s": _acf(d, int(round(0.5 * rate))), "ac1_resid": _acf(hp, 1)})
    f.update(_spectrum_feats(d, rate, "cell_"))
    f.update(_spectrum_feats(_dff(Ft), rate, "tube_"))
    # events
    ev = _events(d)
    from scipy.signal import find_peaks
    pk4, _ = find_peaks(d, prominence=4 * nz_, distance=5)
    f["events_per_min"] = float(len(ev) / minutes)
    f["events4sd_per_min"] = float(len(pk4) / minutes)
    # bleaching: running 10th percentile of the cell trace in 30 s windows -> linear slope
    w = max(5, int(30 * rate)); nb = max(2, T // w)
    t_mid = np.array([(i + 0.5) * w / rate / 60 for i in range(nb)])
    base = np.array([np.percentile(F[i * w:(i + 1) * w], 10) for i in range(nb)])
    f["bleach_frac_per_min"] = float(np.polyfit(t_mid, base, 1)[0] / max(base[0], 1e-6)) if nb >= 2 else float("nan")
    baset = np.array([np.percentile(Ft[i * w:(i + 1) * w], 10) for i in range(nb)])
    f["tube_bleach_frac_per_min"] = float(np.polyfit(t_mid, baset, 1)[0] / max(baset[0], 1e-6)) if nb >= 2 else float("nan")
    # per-voxel statistics in chunks (memory): mean, temporal p99.5, diff-variance (noise), median
    rng = np.random.default_rng(0)
    nvox = flat.shape[1]
    means = np.empty(nvox); p995 = np.empty(nvox); dvar = np.empty(nvox); med = np.empty(nvox)
    for a in range(0, nvox, 8000):
        blk = flat[:, a:a + 8000].astype(np.float32)
        means[a:a + 8000] = blk.mean(0)
        p995[a:a + 8000] = np.percentile(blk, 99.5, axis=0)
        med[a:a + 8000] = np.median(blk, axis=0)
        dvar[a:a + 8000] = np.var(np.diff(blk, axis=0), axis=0) / 2.0       # white-noise variance estimate
        del blk
    transient = np.clip(p995 - means, 0, None)
    out_idx = np.flatnonzero((~mask & ~excl).ravel())
    t_in = transient[ci].mean(); t_out = transient[out_idx].mean() if out_idx.size else np.nan
    f["activity_outside_ratio"] = float(t_out / max(t_in, 1e-6))
    vz = transient / np.sqrt(dvar + 1e-6)                        # per-voxel transient in noise units
    thr = np.median(vz[ci])
    act = vz > thr
    n_act_out = int(act[out_idx].sum()); n_act_in = int(act[ci].sum())
    f["frac_active_outside"] = float(n_act_out / max(n_act_out + n_act_in, 1))
    f["voxel_snr_median"] = float(np.median(vz[ci]))
    f["bg_voxel_snr_median"] = float(np.median(vz[bi])) if bi.size else float("nan")
    # variance vs mean across voxels. Photon (shot) noise: noise variance grows LINEARLY with
    # the mean, var = g * (mean - offset); the detector offset (~1500 ADU here) makes a log-log
    # slope meaningless, so fit the line: g = gain (ADU per photon), r2 = how well shot noise
    # explains the voxel-to-voxel noise (low r2 = extra non-photon noise), offset = implied dark level.
    sel = rng.choice(nvox, size=min(6000, nvox), replace=False)
    mm, vv = means[sel], dvar[sel]
    ok = (vv > 0) & (mm < np.percentile(mm, 99.5))
    if ok.sum() > 50 and np.ptp(mm[ok]) > 0:
        sl, ic = np.polyfit(mm[ok], vv[ok], 1); res = vv[ok] - (sl * mm[ok] + ic)
        f["varmean_gain"] = float(sl); f["varmean_r2"] = float(1 - res.var() / vv[ok].var())
        f["varmean_offset"] = float(-ic / sl) if sl > 0 else float("nan")
    else:
        f["varmean_gain"] = f["varmean_r2"] = f["varmean_offset"] = float("nan")
    f["gain_var_over_mean"] = float(np.median(dvar[ci] / np.maximum(means[ci], 1)))
    # single-voxel noise in dF/F units (what a small region / a movie pixel sees)
    f["voxel_dff_noise"] = float(np.median(np.sqrt(dvar[ci]) / np.maximum(med[ci], 1)))
    # motion: block-mean volumes registered to the run mean (3D phase correlation, um)
    nblk = 8; bl = T // nblk
    ref = stack.mean(0, dtype=np.float32)
    shifts = []
    for i in range(nblk):
        v = stack[i * bl:(i + 1) * bl].mean(0, dtype=np.float32)
        try:
            s, _, _ = phase_cross_correlation(ref, v, upsample_factor=10, normalization=None)
        except TypeError:
            s, _, _ = phase_cross_correlation(ref, v, upsample_factor=10)
        shifts.append(s)
    shifts = np.array(shifts) * (vox if np.all(np.isfinite(vox)) else 1.0)
    f["motion_block_max_um"] = float(np.linalg.norm(shifts - np.median(shifts, 0), axis=1).max())
    f["motion_block_range_um"] = float(np.linalg.norm(np.ptp(shifts, 0)))
    # frame-to-frame: correlation of each volume with the run mean inside a dilated mask
    roi = np.flatnonzero(ndi.binary_dilation(cell, iterations=2).ravel())
    rr = ref.ravel()[roi]; rr = (rr - rr.mean()) / (rr.std() + 1e-9)
    fr = np.empty(T)
    for a in range(0, T, 200):
        b = flat[a:a + 200][:, roi].astype(np.float32)
        b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-9)
        fr[a:a + 200] = b @ rr / len(roi)
    f["frame_corr_median"] = float(np.median(fr)); f["frame_corr_p05"] = float(np.percentile(fr, 5))
    # ref3d drift recorded by the reference builder (this is what the [auto-QC] motion mark used)
    rj = Path(run["stack"]).with_name(Path(run["stack"]).stem + "_ref3d.json")
    try:
        j = json.loads(rj.read_text()); dz = np.array(j.get("drift_zyx") or [0, 0, 0], float)
        f["ref3d_drift_um"] = float(np.linalg.norm(dz * (vox if np.all(np.isfinite(vox)) else 1)))
        f["ref3d_block_registered"] = int(j.get("block_shifts_zyx") is not None)
    except Exception:
        f["ref3d_drift_um"] = float("nan"); f["ref3d_block_registered"] = -1
    # z structure (same measure as the [z-QC] marks: first 30 volumes)
    zm = z_metrics(stack[:min(30, T)].astype(np.float64))
    f["z_pc1_frac"] = zm["pc1_frac"]; f["z_n_comp_95"] = zm["n_comp_95"]; f["z_adj_corr"] = zm["adj_corr"]
    # cell on the tube edge (same as the [auto-QC] boundary measure)
    nb_, nm_ = 0, 0
    for xi in range(X):
        col = cell[:, :, xi]
        if col.any():
            nm_ += 1; zz, yy = np.where(col)
            if zz.mean() < 2 or zz.mean() > Z - 3 or yy.mean() < 2 or yy.mean() > Y - 3:
                nb_ += 1
    f["edge_fraction"] = float(nb_ / max(nm_, 1))
    del stack, flat
    return f


def _cache_key(run):
    return run["key"].replace("/", "_")


def features_for(run, recompute=False):
    cache = OUT / "features" / f"{_cache_key(run)}.json"
    stamp = {"stack": run["stack"], "mask": run["mask"], "exclude": run.get("exclude"),
             "mtimes": [os.path.getmtime(p) for p in (run["stack"], run["mask"], run.get("exclude")) if p],
             "feature_version": FEATURE_VERSION}
    if cache.exists() and not recompute:
        try:
            c = json.loads(cache.read_text())
            if c.get("stamp") == stamp:
                return c["features"]
        except Exception:
            pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = compute_features(run)
    except Exception as e:                      # one bad run must not stop the batch
        feats = {"error": f"{type(e).__name__}: {e}"}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"stamp": stamp, "features": feats}, default=float, indent=1))
    return feats


# ------------------------------------------------------------------------- labels

def load_labels(keys):
    marks = pd.read_csv(PROJECT / "run_marks.csv", dtype=str).fillna("")
    cohort = set(pd.read_csv(PROJECT / "stats" / "cohort_metrics.csv")["behavior_base"])
    lab = {}
    for k in keys:
        m = marks[marks.behavior_base == k]
        mark, reason = (m.iloc[0]["mark"], m.iloc[0]["reason"]) if len(m) else ("", "")
        auto = reason.startswith("[auto-QC]") or reason.startswith("[z-QC]")
        lab[k] = {"in_cohort": int(k in cohort), "mark": mark, "mark_reason": reason,
                  "mark_source": ("" if not mark else "automatic" if auto else "daria"),
                  "daria_noise_reason": int(bool(mark) and not auto and "nois" in reason.lower())}
        # Daria's weak label: 0 = curated cohort, 1 = set aside by Daria, NaN = no label
        lab[k]["y_daria"] = 0.0 if k in cohort else (1.0 if mark and not auto else np.nan)
        lab[k]["y_auto"] = 0.0 if k in cohort else (1.0 if mark and auto else np.nan)
        lab[k]["label_group"] = ("good (cohort)" if k in cohort else
                                 "Daria: noisy" if lab[k]["daria_noise_reason"] else
                                 "Daria: set aside, no reason" if mark and not auto else
                                 "automatic mark" if mark else "unlabeled")
    return lab


# -------------------------------------------------------------------------- model

# feature, direction that means "worse" (+1 higher is worse, -1 lower is worse, 0 two-sided),
# log-transform, plain-language description
NOISE_FEATURES = [
    ("noise_1s", +1, True, "cell dF/F noise in a 1-s average (rate-free)"),
    ("snr_top10", -1, True, "size of the 10 largest events over noise (one burst cannot carry it)"),
    ("active_block_frac", -1, False, "share of 30-s blocks with an event > 5 noise SD"),
    ("chunk_noise_1s", +1, True, "noise of a quarter-cell (region-sized) trace, 1-s average"),
    ("chunk_snr_top10", -1, True, "largest events over noise in quarter-cell traces"),
    ("chunk_wander", +1, False, "baseline wander of quarter-cell traces (baseline SD / fast noise)"),
    ("baseline_wander", +1, False, "baseline wander of the cell trace (baseline SD / fast noise)"),
    ("unreliability", +1, True, "1 - split-half reliability of the cell mask (noise share of the trace)"),
    ("voxel_snr_median", -1, True, "single-voxel transient over noise"),
    ("voxel_dff_noise", +1, True, "single-voxel noise in dF/F units"),
    ("sig_to_bg", -1, False, "cell brightness above tube background"),
    ("cell_hf_frac", +1, False, "share of trace power above 1 Hz"),
    ("lf_frac", +1, False, "share of trace power below 0.05 Hz (slow drift)"),
    ("tube_comb_excess_log10", +1, False, "narrow periodic peak in the whole-tube spectrum (artifact)"),
    ("ac_0p5s", -1, False, "autocorrelation of the cell trace at 0.5 s (GCaMP7s signal is slow)"),
    ("varmean_r2", -1, False, "how well shot noise (variance linear in mean) explains voxel noise"),
    ("bleach_frac_per_min", -1, False, "baseline change per minute (negative = bleaching)"),
    ("motion_block_max_um", +1, True, "largest drift of a time block vs the run mean"),
    ("frame_corr_median", -1, False, "how well single volumes match the run mean (blur / motion / noise)"),
    ("dff_skew", -1, False, "skewness of dF/F (sparse big events = high)"),
]
GEOMETRY_FEATURES = [
    ("activity_outside_ratio", +1, False, "transient activity outside the mask relative to inside"),
    ("frac_active_outside", +1, False, "share of active voxels that lie outside the mask"),
    ("z_pc1_frac", +1, False, "thin slab: share of across-plane variance in one shared image"),
    ("edge_fraction", +1, False, "share of the cell sitting on the tube edge"),
]
MODEL_FEATURES = NOISE_FEATURES + GEOMETRY_FEATURES
AUTO_SOURCE_FEATURES = {"z_pc1_frac", "edge_fraction", "activity_outside_ratio", "ref3d_drift_um"}


def _design(df, names):
    X = df[names].astype(float).copy()
    for n, _, lg, _ in MODEL_FEATURES:
        if n in X and lg:
            X[n] = np.log10(np.maximum(X[n], 1e-6))
    return X


def _fit_l1(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced",
                                         max_iter=5000))
    return m.fit(X, y)


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def lomo(df, ycol, names, Cs=(0.05, 0.1, 0.3, 1.0, 3.0), seed=0):
    """Leave-one-mouse-out with the L1 strength picked by an inner leave-one-mouse-out on
    the training mice (pooled AUC). Returns out-of-fold probabilities for EVERY run (labeled
    or not), using the fold model that never saw that run's mouse."""
    from sklearn.tree import DecisionTreeClassifier
    lab = df[df[ycol].notna()]
    mice = sorted(df.mouse.unique()); lab_mice = sorted(lab.mouse.unique())
    p_l1 = pd.Series(np.nan, index=df.index); p_tree = pd.Series(np.nan, index=df.index)
    folds = []
    Xall = _design(df, names)
    for m in mice:
        tr = lab[lab.mouse != m]
        if tr[ycol].nunique() < 2:
            continue
        # inner LOMO for C
        best, bestC = -1, Cs[len(Cs) // 2]
        for C in Cs:
            pp, yy = [], []
            for mi in sorted(tr.mouse.unique()):
                a = tr[tr.mouse != mi]; b = tr[tr.mouse == mi]
                if a[ycol].nunique() < 2:
                    continue
                mdl = _fit_l1(Xall.loc[a.index], a[ycol].values, C)
                pp += list(mdl.predict_proba(Xall.loc[b.index])[:, 1]); yy += list(b[ycol].values)
            sc = _auc(yy, pp) if pp else float("nan")
            if np.isfinite(sc) and sc > best + 1e-9:
                best, bestC = sc, C
        mdl = _fit_l1(Xall.loc[tr.index], tr[ycol].values, bestC)
        te = df.index[df.mouse == m]
        p_l1.loc[te] = mdl.predict_proba(Xall.loc[te])[:, 1]
        coef = mdl[-1].coef_[0]
        Xi = mdl[0].transform(Xall.loc[tr.index])
        tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, class_weight="balanced",
                                      random_state=seed).fit(Xi, tr[ycol].values)
        p_tree.loc[te] = tree.predict_proba(mdl[0].transform(Xall.loc[te]))[:, 1]
        folds.append({"held_out_mouse": m, "C": bestC, "inner_auc": best, "n_train": int(len(tr)),
                      "n_test_labeled": int(df.loc[te, ycol].notna().sum()),
                      "coef": dict(zip(names, map(float, coef)))})
    return p_l1, p_tree, folds


def _kbest_fit(X, y, k):
    """Pick the k features with the largest univariate |AUC - 0.5| on THESE rows only, then
    an L2 logistic regression on them (standardized). Returns (model, features)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    sc = {}
    for c in X.columns:
        ok = X[c].notna().values
        if ok.sum() > 4 and len(np.unique(y[ok])) == 2:
            sc[c] = abs(_auc(y[ok], X[c].values[ok]) - 0.5)
    feats = [c for c, _ in sorted(sc.items(), key=lambda t: -t[1])[:k]]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    return m.fit(X[feats], y), feats


def lomo_kbest(df, ycol, names, ks=(1, 2, 3, 5)):
    """Leave-one-mouse-out: in each fold the number of features k is chosen by an inner
    leave-one-mouse-out, the k features by univariate AUC on the training mice only."""
    lab = df[df[ycol].notna()]; Xall = _design(df, names)
    p = pd.Series(np.nan, index=df.index); folds = []
    for m in sorted(df.mouse.unique()):
        tr = lab[lab.mouse != m]
        if tr[ycol].nunique() < 2:
            continue
        best, bestk = -1, ks[1]
        for k in ks:
            pp, yy = [], []
            for mi in sorted(tr.mouse.unique()):
                a = tr[tr.mouse != mi]; b = tr[tr.mouse == mi]
                if a[ycol].nunique() < 2:
                    continue
                mdl, fs = _kbest_fit(Xall.loc[a.index], a[ycol].values, k)
                pp += list(mdl.predict_proba(Xall.loc[b.index, fs])[:, 1]); yy += list(b[ycol].values)
            s_ = _auc(yy, pp) if pp else float("nan")
            if np.isfinite(s_) and s_ > best + 1e-9:
                best, bestk = s_, k
        mdl, fs = _kbest_fit(Xall.loc[tr.index], tr[ycol].values, bestk)
        te = df.index[df.mouse == m]
        p.loc[te] = mdl.predict_proba(Xall.loc[te, fs])[:, 1]
        folds.append({"held_out_mouse": m, "k": bestk, "inner_auc": best, "features": fs,
                      "coef": dict(zip(fs, map(float, mdl[-1].coef_[0])))})
    return p, folds


def univariate_table(df, names, ycols):
    """AUC of each feature alone, oriented so that > 0.5 means 'worse direction = set aside'
    (direction fixed a priori, nothing learned)."""
    rows = []
    dirs = {n: d for n, d, _, _ in MODEL_FEATURES}
    for n in names:
        r = {"feature": n, "worse_direction": dirs[n]}
        for yc in ycols:
            ok = df[yc].notna() & df[n].notna()
            r[f"auc_{yc}"] = _auc(df.loc[ok, yc], df.loc[ok, n] * (dirs[n] or 1)) if ok.sum() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def final_model(df, ycol, names, C):
    from sklearn.tree import DecisionTreeClassifier, export_text
    lab = df[df[ycol].notna()]; X = _design(lab, names)
    mdl = _fit_l1(X, lab[ycol].values, C)
    Xi = mdl[0].transform(X)
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, class_weight="balanced", random_state=0).fit(
        Xi, lab[ycol].values)
    return mdl, export_text(tree, feature_names=list(names))


def summarize_perf(df, ycol, p, thr=0.5):
    lab = df[ycol].notna() & p.notna()
    y = df.loc[lab, ycol].values.astype(int); pp = p[lab].values
    auc = _auc(y, pp)
    # cluster (mouse) bootstrap CI of the pooled AUC
    rng = np.random.default_rng(0); mice = df.loc[lab, "mouse"].values; um = np.unique(mice); boots = []
    for _ in range(2000):
        pick = rng.choice(um, size=len(um), replace=True)
        idx = np.concatenate([np.flatnonzero(mice == m) for m in pick])
        if len(np.unique(y[idx])) == 2:
            boots.append(_auc(y[idx], pp[idx]))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [np.nan, np.nan]
    pred = (pp >= thr).astype(int)
    conf = {"true_bad_called_bad": int(((y == 1) & (pred == 1)).sum()), "true_bad_called_good": int(((y == 1) & (pred == 0)).sum()),
            "true_good_called_bad": int(((y == 0) & (pred == 1)).sum()), "true_good_called_good": int(((y == 0) & (pred == 0)).sum())}
    return {"auc_lomo": auc, "auc_ci95_mouse_bootstrap": ci, "threshold": thr, "confusion": conf,
            "n_runs": int(lab.sum()), "n_bad": int((y == 1).sum()), "n_good": int((y == 0).sum()),
            "n_mice": int(len(um)), "n_mice_bad": int(len(np.unique(mice[y == 1]))),
            "n_mice_good": int(len(np.unique(mice[y == 0])))}


def unsupervised(df, names):
    """Robust z of each feature vs the good (cohort) set, in the 'worse' direction; the run's
    own mouse is left out of the reference. badness = mean of the 3 largest worse-direction z.
    mahal = Ledoit-Wolf Mahalanobis distance to the good set (same leave-mouse-out)."""
    from sklearn.covariance import LedoitWolf
    X = _design(df, names); dirs = {n: d for n, d, _, _ in MODEL_FEATURES}
    good = df.in_cohort == 1
    Z = pd.DataFrame(np.nan, index=df.index, columns=names); mahal = pd.Series(np.nan, index=df.index)
    for m in df.mouse.unique():
        ref = X[good & (df.mouse != m)]; te = df.index[df.mouse == m]
        med = ref.median(); mad = 1.4826 * (ref - med).abs().median()
        mad = np.maximum(mad, 0.5 * ref.std()) + 1e-9   # floor: n=16 good runs, MAD can be tiny
        z = (X.loc[te] - med) / mad
        Z.loc[te] = z
        R = ref.fillna(med); lw = LedoitWolf().fit((R - med) / mad)
        zz = ((X.loc[te].fillna(med) - med) / mad).values
        mahal.loc[te] = np.sqrt(np.einsum("ij,jk,ik->i", zz, lw.precision_, zz) / len(names))
    W = Z.copy()
    for n in names:
        W[n] = Z[n] * dirs[n] if dirs[n] != 0 else Z[n].abs()
    bad = W.apply(lambda r: np.sort(r.dropna().values)[-3:].mean() if r.notna().any() else np.nan, axis=1)
    return Z, W, bad, mahal


def _phrase(n, z, val, refmed):
    desc = {k: d for k, _, _, d in MODEL_FEATURES}[n]
    hi = "high" if z > 0 else "low"
    return f"{desc}: {hi} ({val:.3g} vs good-set median {refmed:.3g}, robust z {z:+.1f})"


# ------------------------------------------------------------------------ figures

COLORS = {"good (cohort)": "#1b9e77", "Daria: noisy": "#d95f02", "Daria: set aside, no reason": "#e7298a",
          "automatic mark": "#7570b3", "unlabeled": "#999999"}


def figures(df, names, perf, coefs):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 8})
    Xd = _design(df, names)
    # 1. feature strips
    nc = 5; nr = int(np.ceil(len(names) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(11, 2.0 * nr))
    groups = list(COLORS)
    rng = np.random.default_rng(0)
    for ax, n in zip(axs.ravel(), names):
        for gi, g in enumerate(groups):
            v = Xd.loc[df.label_group == g, n].dropna()
            ax.scatter(gi + rng.uniform(-0.15, 0.15, len(v)), v, s=9, c=COLORS[g], alpha=0.8, lw=0)
        lg = {k: l for k, _, l, _ in MODEL_FEATURES}[n]
        ax.set_title(("log10 " if lg else "") + n, fontsize=7); ax.set_xticks([])
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[g], label=g) for g in groups]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Trace / stack features per run, by label (each dot = one run)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_features.{ext}", dpi=150)
    plt.close(fig)
    # 2. ROC
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for col, lab, c in (("p_daria_kbest", "Daria labels, k-best noise (p_noisy)", "#1f78b4"),
                        ("p_daria_l1", "Daria labels, L1 logistic", "#d95f02"),
                        ("p_daria_tree", "Daria labels, depth-2 tree", "#e6ab02"),
                        ("p_daria_all_l1", "Daria labels, noise + geometry", "#66a61e"),
                        ("p_daria_june_kbest", "Daria labels, June only, k-best", "#a6761d"),
                        ("p_auto_l1", "automatic marks, L1 logistic", "#7570b3"),
                        ("unsup_badness", "unsupervised badness (Daria labels)", "#666666")):
        y = df["y_auto" if "auto" in col else "y_daria"]
        if col.startswith("p_daria_june"):
            y = df["y_daria_june"]
        ok = y.notna() & df[col].notna()
        if ok.sum() and y[ok].nunique() == 2:
            fpr, tpr, _ = roc_curve(y[ok], df.loc[ok, col])
            ax.plot(fpr, tpr, color=c, label=f"{lab} (AUC {_auc(y[ok], df.loc[ok, col]):.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("false-alarm rate (good runs called bad)")
    ax.set_ylabel("hit rate (set-aside runs called bad)"); ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.set_title("Leave-one-mouse-out ROC")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_roc.{ext}", dpi=150)
    plt.close(fig)
    # 3. which features matter: univariate AUC (direction fixed a priori) + k-best selection frequency
    uni = pd.DataFrame(perf["univariate_auc"]).set_index("feature").reindex(names)
    freq = perf["daria_june"]["kbest_feature_frequency"]; nf = max(1, len(perf["daria_june"]["kbest_folds"]))
    o = uni["auc_y_daria_june"].sort_values().index
    fig, ax = plt.subplots(figsize=(5.4, 5.0)); yy = np.arange(len(o))
    for col, lab, c, dy in (("auc_y_daria_june", "Daria marks, June", "#1f78b4", 0.2),
                            ("auc_y_daria", "Daria marks, all runs", "#d95f02", 0.0),
                            ("auc_y_auto", "automatic marks", "#7570b3", -0.2)):
        ax.barh(yy + dy, uni.loc[o, col] - 0.5, height=0.2, left=0.5, color=c, label=lab)
    ax.set_yticks(yy); ax.set_yticklabels([f"{n}  [{freq.get(n, 0)}/{nf}]" for n in o], fontsize=6)
    ax.axvline(0.5, color="k", lw=0.6); ax.set_xlim(0, 1)
    ax.set_xlabel("AUC of the feature alone (> 0.5: worse value goes with 'set aside')")
    ax.set_title("Which features matter  [folds selecting it, June model]", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_feature_importance.{ext}", dpi=150)
    plt.close(fig)
    # 4. score map
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for g in groups:
        s = df[df.label_group == g]
        ax.scatter(s.unsup_badness, s.p_noisy, s=18, c=COLORS[g], label=g, alpha=0.85, lw=0)
    for _, r in df[(df.flag_disagree != "")].iterrows():
        ax.annotate(r.short, (r.unsup_badness, r.p_noisy), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0.5, color="k", ls=":", lw=0.7); ax.axvline(2.0, color="k", ls=":", lw=0.7)
    xmax = 12.0; out_ = df[df.unsup_badness > xmax]
    ax.set_xlim(0, xmax)
    for _, r in out_.iterrows():          # off-scale runs drawn at the edge, value in the label
        ax.scatter([xmax * 0.99], [r.p_noisy], marker=">", s=22, c=COLORS[r.label_group])
        ax.annotate(f"{r.short} ({r.unsup_badness:.0f})", (xmax * 0.99, r.p_noisy), fontsize=5, ha="right",
                    xytext=(-4, 3), textcoords="offset points")
    ax.set_xlabel("unsupervised badness (robust z vs good set, top-3 mean)")
    ax.set_ylabel("P(set aside) - leave-one-mouse-out"); ax.legend(fontsize=6, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_scores.{ext}", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main

def _short(k):
    return (k.replace("rbp4_", "").replace("_phpeb", "").replace("phpebne_", "ne").replace("_26-", " ")
            .replace("_Run", " r").replace("_MUnit", " mu"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only-features", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    todo = [r for r in runs if not r.get("skip")]
    log("collect_runs", f"{len(runs)} runs with a clean stack; {len(todo)} with a mask; "
                        f"skipped: {[r['key'] for r in runs if r.get('skip')]}")
    from concurrent.futures import ProcessPoolExecutor
    feats = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {r["key"]: ex.submit(features_for, r, a.recompute) for r in todo}
        for k, fu in futs.items():
            feats[k] = fu.result()
            print(f"  {k}: {'ERROR ' + feats[k]['error'] if 'error' in feats[k] else 'ok'}", flush=True)
    rows = []
    lab = load_labels([r["key"] for r in todo])
    for r in todo:
        f = feats[r["key"]]
        rows.append({"run": r["key"], "short": _short(r["key"]), "mouse": r["mouse"], "date": r["date"],
                     "munit": r.get("munit"), "trees": "+".join(sorted(set(r["trees"]))),
                     "run_dir_real": r.get("run_dir_real", ""), "run_dir_mirror": r.get("run_dir_mirror", ""),
                     "mask_kind": r["mask_kind"], "mask_tree": r["mask_tree"],
                     "mask_path": str(Path(r["mask"]).relative_to(PROJECT)), **lab[r["key"]], **f})
    df = pd.DataFrame(rows)
    # pull in the pi_hypothesis per-run features if they exist (descriptive only: never
    # correlation / behavior columns)
    tf = PROJECT / "stats" / "trace_features_run.csv"
    merged_cols = []
    if tf.exists():
        t = pd.read_csv(tf)
        kcol = next((c for c in ("run", "behavior_base", "key") if c in t.columns), None)
        if kcol:
            import re
            # never correlation (r_*, *_r, *corr*, pairwise), behavior or quality-rating columns
            bad = re.compile(r"(^r_|_r$|_r_|corr|pairwise|behav|coupl|pupil|whisk|lag|quality)")
            if "root" in t.columns:            # prefer the real-tree row when a run appears twice
                t = t.assign(_o=(~t["root"].astype(str).str.contains("auto")).astype(int)).sort_values(
                    "_o", ascending=False, kind="stable").drop(columns="_o")
            keep = [c for c in t.columns if c != kcol and not bad.search(c.lower())
                    and pd.api.types.is_numeric_dtype(t[c])]
            t = t[[kcol] + keep].rename(columns={c: "tf_" + c for c in keep}).rename(columns={kcol: "run"})
            df = df.merge(t.drop_duplicates("run"), on="run", how="left"); merged_cols = ["tf_" + c for c in keep]
    df.to_csv(OUT / "features.csv", index=False)
    errs = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    log("features", f"{len(df)} runs, {len(errs)} errors {list(errs.run)}; merged trace_features_run cols: {len(merged_cols)}")
    if a.only_features:
        return 0
    df = df[df.get("error").isna()].copy() if "error" in df else df
    df = df.reset_index(drop=True)
    df["unreliability"] = (1 - df.reliability_split_half).clip(lower=1e-6)
    names = [n for n, _, _, _ in MODEL_FEATURES]
    noise_names = [n for n, _, _, _ in NOISE_FEATURES]
    june = df.date.str.startswith("06-")
    df["y_daria_june"] = df.y_daria.where(june)
    # models
    res = {}
    df["y_daria_noisy_explicit"] = np.where(df.in_cohort == 1, 0.0, np.where(df.daria_noise_reason == 1, 1.0, np.nan))
    for tag, ycol, fn, sub in (("daria", "y_daria", noise_names, df.index),
                               ("daria_all", "y_daria", names, df.index),
                               ("daria_june", "y_daria_june", noise_names, df.index),
                               ("daria_noisy_explicit", "y_daria_noisy_explicit", noise_names, df.index),
                               ("auto", "y_auto", [n for n in names if n not in AUTO_SOURCE_FEATURES], df.index),
                               ("auto_with_source_features", "y_auto", names, df.index)):
        p1, pt, folds = lomo(df, ycol, fn)
        df[f"p_{tag}_l1"] = p1; df[f"p_{tag}_tree"] = pt
        Cfin = float(pd.Series([f["C"] for f in folds]).mode().iloc[0]) if folds else 0.3
        mdl, tree_txt = final_model(df, ycol, fn, Cfin)
        coef = dict(zip(fn, map(float, mdl[-1].coef_[0])))
        nzf = {n: float(np.mean([abs(f["coef"][n]) > 1e-9 for f in folds])) if folds else np.nan for n in fn}
        pk, kfolds = lomo_kbest(df, ycol, fn)
        df[f"p_{tag}_kbest"] = pk
        from collections import Counter
        kfreq = Counter(f_ for fo in kfolds for f_ in fo["features"])
        res[tag] = {"features": fn, "C_final": Cfin, "perf_kbest": summarize_perf(df, ycol, pk),
                    "kbest_folds": kfolds, "kbest_feature_frequency": dict(kfreq.most_common()), "coef": coef, "intercept": float(mdl[-1].intercept_[0]),
                    "fold_nonzero_frac": nzf, "folds": folds, "tree": tree_txt,
                    "perf_l1": summarize_perf(df, ycol, p1), "perf_tree": summarize_perf(df, ycol, pt)}
        log(f"model_{tag}", {"auc_kbest": res[tag]["perf_kbest"]["auc_lomo"], "kbest_feats": dict(kfreq.most_common(5)),
                             "auc_l1": res[tag]["perf_l1"]["auc_lomo"], "auc_tree": res[tag]["perf_tree"]["auc_lomo"],
                             "n": res[tag]["perf_l1"]["n_runs"], "mice": res[tag]["perf_l1"]["n_mice"],
                             "C": Cfin, "nonzero": [n for n, v in coef.items() if abs(v) > 1e-9]})
    # p_noisy = the small model (k univariately strongest noise features, chosen inside each
    # fold): fewer knobs than the 20-feature L1 fit, so less overfitting with 45 labeled runs
    # Decision (made after seeing that the all-era model is at chance, disclosed in the
    # summary): the score is the June-trained model. September set-aside runs are set aside
    # mostly for non-noise reasons and differ in frame rate / depth sampling, so pooling them
    # in teaches the model the era, not the noise. Every run still gets a held-out score: a
    # run's own mouse is never in its training fold (September mice are out of era).
    df["p_noisy"] = df.p_daria_june_kbest
    df["p_noisy_extrapolated"] = ~df.date.str.startswith("06-")
    uni = univariate_table(df, names, ["y_daria", "y_daria_june", "y_daria_noisy_explicit", "y_auto"])
    uni.to_csv(OUT / "univariate_auc.csv", index=False); res["univariate_auc"] = uni.to_dict(orient="records")
    Zg, Wg, bad_all, mahal_all = unsupervised(df, names)
    df["unsup_badness_all"] = bad_all; df["unsup_mahal_all"] = mahal_all
    Z, W, bad, mahal = Zg, Wg, None, None
    _, _, bad, mahal = unsupervised(df, noise_names)
    df["unsup_badness"] = bad; df["unsup_mahal"] = mahal
    for n in names:
        df[f"z_{n}"] = Z[n]
    res["unsup_auc_daria"] = summarize_perf(df, "y_daria", df.unsup_badness, thr=2.0)
    res["unsup_auc_auto"] = summarize_perf(df, "y_auto", df.unsup_badness, thr=2.0)
    res["mahal_auc_daria"] = summarize_perf(df, "y_daria", df.unsup_mahal, thr=2.0)
    res["unsup_all_auc_daria"] = summarize_perf(df, "y_daria", df.unsup_badness_all, thr=2.0)
    res["unsup_auc_noisy_explicit"] = summarize_perf(df, "y_daria_noisy_explicit", df.unsup_badness, thr=2.0)
    # reasons in words
    Xd = _design(df, names); good = df.in_cohort == 1
    reasons = []
    for i, r in df.iterrows():
        refm = Xd[good & (df.mouse != r.mouse)].median()
        w = W.loc[i].dropna().sort_values(ascending=False)
        top = [n for n in w.index[:3] if w[n] > 2.0]
        lg = {k: l for k, _, l, _ in MODEL_FEATURES}
        ph = [_phrase(n, Z.loc[i, n], (10 ** Xd.loc[i, n]) if lg[n] else Xd.loc[i, n],
                      (10 ** refm[n]) if lg[n] else refm[n]) for n in top]
        reasons.append("; ".join(ph) if ph else "no feature beyond 2 robust SD of the good set")
    df["top_reasons"] = reasons
    # disagreements
    flag = []
    for _, r in df.iterrows():
        f = ""
        if r.in_cohort == 1 and (r.p_noisy >= 0.75 or r.unsup_badness > 3):
            f = "cohort run looks noisy"
        elif r.mark == "" and r.in_cohort == 0 and (r.p_noisy >= 0.75 or r.unsup_badness > 3):
            f = "unmarked run looks noisy"
        elif r.mark_source == "daria" and r.p_noisy < 0.25 and r.unsup_badness < 2:
            f = "Daria-marked run looks clean on traces"
        elif r.mark_source == "automatic" and r.p_auto_kbest < 0.25 and r.unsup_badness_all < 2:
            f = "auto-marked run looks clean on traces"
        flag.append(f)
    df["flag_disagree"] = flag
    df["noise_tier"] = pd.cut(df.p_noisy, [-0.01, 0.25, 0.5, 0.75, 1.01],
                              labels=["looks clean", "probably fine", "possibly noisy", "likely noisy"]).astype(str)
    cols = ["run", "mouse", "date", "munit", "trees", "mask_kind", "label_group", "mark", "mark_reason",
            "p_noisy", "noise_tier", "p_noisy_extrapolated", "chunk_snr_top10", "p_daria_kbest", "p_daria_l1", "p_daria_tree", "p_daria_june_kbest", "p_daria_june_l1", "p_daria_all_l1", "p_auto_l1",
            "p_auto_with_source_features_l1", "unsup_badness", "unsup_mahal", "unsup_badness_all", "top_reasons", "flag_disagree", "noise_1s", "snr_top10", "active_block_frac",
            "chunk_snr_top10", "chunk_wander", "noise_over_amp", "snr_p99", "frame_rate_hz", "run_dir_real", "run_dir_mirror"]
    df.sort_values("p_noisy", ascending=False)[cols].to_csv(OUT / "scores.csv", index=False)
    df.to_csv(OUT / "features.csv", index=False)
    card = {"model": "noise_model", "version": __version__, "created": datetime.now(timezone.utc).isoformat(),
            "code": "code/STEP8_stats/noise_model.py",
            "labels": {"good": "run in stats/cohort_metrics.csv (Daria's curated analyzable set)",
                       "daria_bad": "run_marks.csv excluded/revisit whose reason does not start with [auto-QC]/[z-QC] "
                                    "(her 'very noisy' / 'pretty noisy' plus marks with no reason; the no-reason marks "
                                    "are assumed to be hers - GUI marks carry no reason - but this is not verified)",
                       "auto_bad": "run_marks.csv marks whose reason starts with [auto-QC] or [z-QC] (kept separate)",
                       "unlabeled": "processed runs neither in the cohort nor marked"},
            "label_counts": df.label_group.value_counts().to_dict(),
            "label_mice": df.groupby("label_group").mouse.nunique().to_dict(),
            "validation": "leave-one-MOUSE-out; L1 strength chosen by inner leave-one-mouse-out on the training mice; "
                          "AUC pooled over out-of-fold predictions, 95% CI by mouse bootstrap",
            "no_circularity": "no feature uses correlation with the soma/regions or behavior",
            "feature_definitions": {n: {"worse_direction": d, "log10": l, "meaning": m} for n, d, l, m in MODEL_FEATURES},
            "auto_source_features": sorted(AUTO_SOURCE_FEATURES),
            "models": res, "trace_features_run_merged_columns": merged_cols,
            "caveats": ["weak labels: 'set aside' is not only 'noisy' (also no behavior, other cells, ...)",
                        "Daria's set-aside runs come mostly from mice with no cohort run (September mice); the all-run "
                        "model can partly learn the recording era -> the June-only model is the cleaner test",
                        "same-session runs may be the same cell; runs within a mouse are not independent",
                        "frame rate (4.5-18.5 Hz) changes per-frame noise; frame rate itself is not a model feature"]}
    (OUT / "model_card.json").write_text(json.dumps(card, indent=1, default=float))
    figures(df, names, res, res)
    write_summary(df, res, names)
    log("done", {"scores": str(OUT / "scores.csv"), "n_scored": int(len(df)),
                 "auc_daria": res["daria"]["perf_l1"]["auc_lomo"], "auc_june": res["daria_june"]["perf_l1"]["auc_lomo"],
                 "auc_auto": res["auto"]["perf_l1"]["auc_lomo"], "unsup_auc_daria": res["unsup_auc_daria"]["auc_lomo"]})
    print(f"wrote {OUT}")
    return 0


def _fmt_perf(p):
    c = p["confusion"]
    return (f"AUC {p['auc_lomo']:.2f} [95% CI {p['auc_ci95_mouse_bootstrap'][0]:.2f}..{p['auc_ci95_mouse_bootstrap'][1]:.2f}] "
            f"(n = {p['n_runs']} runs: {p['n_bad']} set aside from {p['n_mice_bad']} mice, {p['n_good']} good from "
            f"{p['n_mice_good']} mice; {p['n_mice']} mice total). At threshold {p['threshold']}: "
            f"{c['true_bad_called_bad']}/{p['n_bad']} set-aside runs flagged, "
            f"{c['true_good_called_bad']}/{p['n_good']} good runs wrongly flagged.")


def write_summary(df, res, names):
    L = []
    j = res["daria_june"]; d = res["daria"]; da = res["daria_all"]; au = res["auto"]; ex_ = res["daria_noisy_explicit"]
    uni = pd.DataFrame(res["univariate_auc"]).set_index("feature")
    desc = {k: m for k, _, _, m in MODEL_FEATURES}
    nf = len(j["kbest_folds"]); freq = j["kbest_feature_frequency"]
    nlab = df.y_daria.notna() | df.y_auto.notna()
    L.append("# Noise model: which runs look noisy from the traces\n")
    L.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by `code/STEP8_stats/noise_model.py` v{__version__}. "
             f"Scored **{len(df)} processed runs from {df.mouse.nunique()} mice** (every run with a clean stack and a cell "
             f"mask: {int((df.mask_kind == 'hand').sum())} hand masks, {int((df.mask_kind != 'hand').sum())} automatic). "
             f"All features come from the raw cleaned stack; none uses correlation with the soma or behavior.\n")
    L.append("## Plain-language summary\n")
    top = max(freq, key=freq.get)
    L.append(f"1. **Within the June recordings, the runs Daria set aside are the ones whose region-sized traces have small "
             f"events relative to their noise.** One number does most of the work: `{top}` ({desc[top]}), picked in "
             f"{freq[top]}/{nf} leave-one-mouse-out folds. Alone it separates June set-aside runs from cohort runs with "
             f"AUC {uni.loc[top, 'auc_y_daria_june']:.2f}. The fitted model (`p_noisy`) reaches leave-one-mouse-out "
             f"{_fmt_perf(j['perf_kbest'])}")
    L.append(f"2. **Pooling all eras does not work** (AUC {d['perf_kbest']['auc_lomo']:.2f} k-best, {d['perf_l1']['auc_lomo']:.2f} "
             f"L1 on {len(d['features'])} noise features, {da['perf_l1']['auc_lomo']:.2f} with mask-geometry added). The "
             f"September mice (155, phpebne_050, phpebne_053) have no cohort run, were set aside mostly without a stated "
             f"reason, and run at other frame rates and depth sampling. There, the set-aside runs are often LESS noisy "
             f"per second than the cohort (noise_1s alone: AUC {uni.loc['noise_1s', 'auc_y_daria']:.2f}, i.e. reversed). "
             f"So 'set aside' in September is mostly not about noise, and a pooled model learns the era, not the noise.")
    L.append(f"3. **Daria's three explicit 'noisy' runs** (2 mice) are too few to train on (k-best AUC "
             f"{ex_['perf_kbest']['auc_lomo']:.2f}). The label-free score separates them from the cohort better "
             f"(AUC {res['unsup_auc_noisy_explicit']['auc_lomo']:.2f}). What sets them apart: higher noise in a 1-s average "
             f"(AUC {uni.loc['noise_1s', 'auc_y_daria_noisy_explicit']:.2f}), higher single-voxel noise "
             f"({uni.loc['voxel_dff_noise', 'auc_y_daria_noisy_explicit']:.2f}) and smaller region events "
             f"({uni.loc['chunk_snr_top10', 'auc_y_daria_noisy_explicit']:.2f}). 140 06-18 r02 is a special case: its "
             f"whole-cell numbers look excellent because ONE burst at the end of the run carries them (skew 11, but only "
             f"0.75 events/min). The burst-robust features only partly catch it: region event SNR "
             f"{df.loc[df.run == 'rbp4_140_phpeb_26-06-18_Run002', 'chunk_snr_top10'].squeeze():.1f} vs a cohort median of "
             f"{df.loc[df.in_cohort == 1, 'chunk_snr_top10'].median():.1f}.")
    L.append(f"4. **The automatic marks are about geometry, not noise.** Noise features alone do not predict them "
             f"(AUC {au['perf_l1']['auc_lomo']:.2f} L1, {au['perf_kbest']['auc_lomo']:.2f} k-best). With the thin-slab and "
             f"tube-edge measures added the AUC is {res['auto_with_source_features']['perf_l1']['auc_lomo']:.2f}. That part is "
             f"circular, because those are the quantities the marks were computed from.")
    L.append(f"5. **Large models overfit here.** The 20-feature L1 regression and the depth-2 tree are at or below chance on "
             f"June (AUC {j['perf_l1']['auc_lomo']:.2f} / {j['perf_tree']['auc_lomo']:.2f}). With 26 labeled June runs from "
             f"5 mice, only a one- or two-feature model holds up when a whole mouse is left out.")
    L.append(f"\nLabels: good = {int((df.in_cohort == 1).sum())} cohort runs ({df[df.in_cohort == 1].mouse.nunique()} mice). "
             f"Set aside by Daria = {int((df.y_daria == 1).sum())} runs ({df[df.y_daria == 1].mouse.nunique()} mice): "
             f"{int(df.daria_noise_reason.sum())} say 'noisy' and {int(((df.y_daria == 1) & (df.daria_noise_reason == 0)).sum())} "
             f"give no reason. The no-reason marks are assumed to be Daria's, since GUI marks carry no reason; this is not "
             f"verified. Automatic marks ([auto-QC]/[z-QC]) = {int((df.y_auto == 1).sum())} runs "
             f"({df[df.y_auto == 1].mouse.nunique()} mice), always kept separate. Unlabeled = "
             f"{int((df.label_group == 'unlabeled').sum())} runs.")
    L.append(f"\nHow to read `p_noisy`: P(a run like this was set aside), learned on June. September runs are scored by "
             f"the same model, outside the era it was trained on (column `p_noisy_extrapolated`), so read them with that "
             f"caveat. Tiers: <0.25 looks clean, 0.25-0.5 probably fine, 0.5-0.75 possibly noisy, >0.75 likely noisy. "
             f"`unsup_badness` = mean of the 3 worst robust z-scores against the good set, own mouse left out (>2 unusual, "
             f">3 clearly unusual). The decision to use the June model was made after seeing that the pooled model was at chance.")
    L.append("\n## Runs that look noisy but are NOT marked\n")
    s = df[df.flag_disagree.isin(["unmarked run looks noisy", "cohort run looks noisy"])].sort_values("p_noisy", ascending=False)
    L.append("Criterion: p_noisy >= 0.75 or noise badness > 3. These are candidates to look at, not verdicts.\n")
    if len(s) == 0:
        L.append("None.")
    for _, r in s.iterrows():
        L.append(f"- **{r.run}** ({r.label_group}; p_noisy {r.p_noisy:.2f}{' (extrapolated)' if r.p_noisy_extrapolated else ''}, "
                 f"badness {r.unsup_badness:.1f}, region event SNR {r.chunk_snr_top10:.1f}): {r.top_reasons}")
    L.append("\n## Marked runs the model disagrees with (look clean on the traces)\n")
    L.append("Criterion: Daria-marked with p_noisy < 0.25 and noise badness < 2, or auto-marked with p_auto < 0.25 and "
             "overall badness < 2. Daria may have had a non-noise reason; nothing here argues for un-marking.\n")
    s = df[df.flag_disagree.str.contains("looks clean")].sort_values("p_noisy")
    if len(s) == 0:
        L.append("None.")
    for _, r in s.iterrows():
        L.append(f"- **{r.run}** ({r.mark}, {r.mark_source}: '{r.mark_reason or 'no reason given'}'; p_noisy {r.p_noisy:.2f}"
                 f"{' (extrapolated)' if r.p_noisy_extrapolated else ''}, badness {r.unsup_badness:.1f}, region event SNR "
                 f"{r.chunk_snr_top10:.1f}) - {r.top_reasons}")
    L.append("\n## Daria's explicitly noisy runs\n")
    for _, r in df[df.daria_noise_reason == 1].iterrows():
        L.append(f"- {r.run} ('{r.mark_reason}'): p_noisy {r.p_noisy:.2f}, badness {r.unsup_badness:.1f}, noise_1s "
                 f"{r.noise_1s:.4f}, region event SNR {r.chunk_snr_top10:.1f} - {r.top_reasons}")
    g = df[df.in_cohort == 1]
    L.append(f"\nFor reference, cohort medians: noise_1s {g.noise_1s.median():.4f}, region event SNR "
             f"{g.chunk_snr_top10.median():.1f}.")
    L.append("\n## Caveats\n")
    L.append("- The unit of inference is the mouse, and every validation leaves a whole mouse out. CIs are mouse-bootstrap "
             "and wide (5 June mice). Runs from one session may be the same cell, so they are not independent.")
    L.append("- 'Set aside' mixes reasons: noise, other cells, no behavior, thin slab. A flag here is a candidate to "
             "review, not a decision. This stage never writes run_marks.csv.")
    L.append("- dF/F uses the pipeline's convention (F0 = 10th percentile of raw F, detector offset not subtracted), so "
             "absolute dF/F values are compressed. All comparisons are within that convention. stats/trace_features_run.csv "
             "(pi_hypothesis stage) has dark-offset-corrected values; its non-correlation columns are merged into "
             "features.csv as tf_* for reference and are not used by the model.")
    L.append("- Per-frame noise grows with frame rate (4.5-18.5 Hz here). The model uses rate-free versions (1-s "
             "average, physical frequency bands, autocorrelation at 0.5 s). Frame rate itself is not a feature.")
    L.append("\nFiles: `scores.csv` (per run: p_noisy, tier, other model scores, unsupervised scores, top reasons in words, "
             "disagreement flag), `features.csv` (all features + labels), `univariate_auc.csv`, `model_card.json`, "
             "`fig_features`, `fig_roc`, `fig_feature_importance`, `fig_scores` (.pdf/.png), `features/` (per-run cache).")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
