#!/usr/bin/env python
"""trace_features.py - per-run and per-region trace features from the RAW stack.

One reusable feature table for the noise / sensor / PI-hypothesis analyses. Everything is
computed from <stem>_clean.tif (never display-masked data) and the saved regions
<stem>_segments_final.tif, honoring <stem>_ignore.json (common/regions.py). dF/F and the
event detector are run_metrics.dff / run_metrics.events, so the numbers match the cohort.

Which runs: every <stem>_metrics.json under each root (the real project and, by default,
the automatic mirror auto_pipeline/). Each row is labelled
    set = curated            in <real>/stats/cohort_metrics.csv
          curated_other      real tree, has regions + metrics but not in the cohort
                             (e.g. runs you excluded as noisy, or stale metrics)
          screening          in auto_pipeline/stats/cohort_metrics.csv
          screening_other    auto mirror, not in its cohort
and carries your run_marks.csv mark + reason (mark_source = daria / auto-QC / z-QC) so a
noise model can use your labels. Nothing here uses correlation with the soma or behavior
for any QC decision; the correlation columns are copied only so other stages can relate
them to the features.

Per region (stats/trace_features.csv, one row per run x region):
  F0                 10th percentile of the raw mean trace (same F0 as dF/F)
  dark_offset        run-level: 1st percentile of the time-averaged volume (the tube's
                     dimmest voxels; detector offset + background)
  F0_dark            F0 - dark_offset  (cell fluorescence above the tube background)
  F0_adc             F0 - ADC_OFFSET   (photon-ish: above the detector zero). ADC_OFFSET =
                     1454 = -Channel_0_Conversion_ConversionLinearOffset of the UG channel,
                     identical in every .mesc in the project (checked 2026-10-05); the
                     extracted TIFFs store the raw uint16 values, so this is the true zero.
  noise_sd           robust dF/F noise: 1.4826 * MAD of dF/F on baseline-only frames
                     (frames outside every event window and below 3 x high-pass noise)
  noise_hp           run_metrics.robust_noise (high-pass MAD; insensitive to slow events)
  sd_total           SD of the whole dF/F trace
  p99, max           dF/F percentiles
  n_events, event_rate_per_min      run_metrics.events (prominence >= 0.2 x range)
  n_events_4sd, event_rate_4sd_per_min   events with prominence >= 4 x noise_sd (noise-gated)
  event_amp_median   median prominence (dF/F) of run_metrics events
  event_amp_4sd_median   same for the noise-gated events
  event_amp_median_darkcorr   event_amp_median * F0 / F0_dark (dF over background-corrected F)
  event_amp_median_adc        event_amp_median * F0 / F0_adc  (dF over offset-corrected F)
  event_amp_F        event_amp_median * F0 (raw ADC units: the signal size, no normalization)
  snr                event_amp_median / noise_sd
  dyn_range          (p99 - p10 of raw) / F0;  dyn_range_dark uses F0_dark
  skew, kurtosis     of dF/F (Fisher kurtosis, 0 = Gaussian)
  frac_time_in_events   fraction of frames inside event windows (-1 s .. +4 s around peaks)
  decay_tau_s        median decay time constant of isolated events (exp fit, R^2 >= 0.6)
  rise_time_s        median 10-90 % rise time of isolated events (coarse: 5-18 Hz volumes)
  noise_sd_adc       noise_sd * F0 / F0_adc (noise in offset-corrected dF/F); dyn_range_adc likewise
  noise_sd_F         noise_sd * F0 (raw units);  var_per_F = noise_sd_F^2 * n_vox / F0_dark
                     (var_per_F_adc uses F0_adc)
                     (~ detector gain if the region noise were independent shot noise)
  reliability, distance_um   copied from <stem>_metrics.json
Per run (stats/trace_features_run.csv): ref_* (reference region = soma or proximal
trunk, the same one run_metrics uses), branch_* (mean over branch regions), trunk_*,
cell_* (mean over all regions), plus correlation columns from the metrics file and
mean_pairwise_r / mean_pairwise_r_corr (all region pairs; corr = disattenuated by
split-half reliability).

Results are cached per run in stats/trace_features_cache/ (small JSON, keyed on the stack,
regions and ignore-list mtimes), so re-running is fast.

  python code/STEP8_stats/trace_features.py                  # both roots
  python code/STEP8_stats/trace_features.py --roots real     # curated tree only
  python code/STEP8_stats/trace_features.py --force          # ignore the cache
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy import stats as sst
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "code")); sys.path.insert(0, str(HERE))
from common.regions import apply_ignore, ignore_path          # noqa: E402
from common.run_marks import load_marks                       # noqa: E402
from run_metrics import dff, events, robust_noise, compartment_of, region_names  # noqa: E402

__version__ = "1.1.0"
ADC_OFFSET = 1454.0     # UG channel ConversionLinearOffset = -1454 in every project .mesc
OUT = PROJECT / "stats"
CACHE = OUT / "trace_features_cache"


# ---------------------------------------------------------------- per-trace features
def _event_mask(d, pk, rate, noise_hp):
    T = len(d); m = np.zeros(T, bool)
    pre, post = int(round(1.0 * rate)), int(round(4.0 * rate))
    for p in pk:
        m[max(0, p - pre):min(T, p + post + 1)] = True
    hi = d > np.median(d) + 3 * noise_hp
    hi = ndi.binary_dilation(hi, iterations=max(1, int(round(rate))))
    return m, m | hi


def _fit_decay(d, p, end, rate):
    y = d[p:end]; t = np.arange(len(y)) / rate
    if len(y) < 4:
        return np.nan
    f = lambda t, a, tau, c: a * np.exp(-t / tau) + c
    try:
        c0 = float(np.min(y)); popt, _ = curve_fit(f, t, y, p0=(y[0] - c0, 1.0, c0),
                                                   bounds=([0, 0.3 / rate, -np.inf], [np.inf, 30, np.inf]), maxfev=4000)
        res = y - f(t, *popt); r2 = 1 - np.sum(res ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        return float(popt[1]) if r2 >= 0.6 and popt[1] < 20 else np.nan
    except Exception:
        return np.nan


def _rise(d, p, rate):
    lo = max(0, p - int(round(3 * rate)))
    b0 = max(0, p - int(round(2 * rate))); b1 = max(b0 + 1, p - int(round(0.5 * rate)))
    if b1 > p or p - lo < 2:
        return np.nan
    b = float(np.median(d[b0:b1])); A = d[p] - b
    if A <= 0:
        return np.nan
    def cross(level):
        for i in range(p, lo, -1):
            if d[i - 1] < level <= d[i]:
                return (i - 1) + (level - d[i - 1]) / (d[i] - d[i - 1])
        return np.nan
    t10, t90 = cross(b + 0.1 * A), cross(b + 0.9 * A)
    return float((t90 - t10) / rate) if np.isfinite(t10) and np.isfinite(t90) and t90 >= t10 else np.nan


def trace_features(raw: np.ndarray, rate: float, dark: float, n_vox: int, prom_frac: float = 0.2) -> dict:
    raw = raw.astype(np.float64); T = len(raw)
    F0 = float(np.percentile(raw, 10)); d = dff(raw)
    nhp = robust_noise(d)
    pk = events(d, prom_frac)
    prom = find_peaks(d, prominence=prom_frac * (d.max() - d.min()), distance=5)[1]["prominences"] if len(pk) else np.array([])
    ev_win, excl = _event_mask(d, pk, rate, nhp)
    base = d[~excl] if (~excl).sum() >= 20 else d
    noise_sd = float(1.4826 * np.median(np.abs(base - np.median(base))) + 1e-9)
    pk4, pr4 = find_peaks(d, prominence=4 * noise_sd, distance=5)
    pr4 = pr4["prominences"]
    minutes = T / rate / 60.0
    F0d = F0 - dark
    amp = float(np.median(prom)) if len(prom) else np.nan
    # isolated events: no other event 2 s before or 6 s after
    taus, rises = [], []
    pre, post = int(round(2 * rate)), int(round(6 * rate))
    for i, p in enumerate(pk):
        if (i > 0 and p - pk[i - 1] < pre) or (i + 1 < len(pk) and pk[i + 1] - p < post) or p + 4 > T:
            continue
        taus.append(_fit_decay(d, p, min(T, p + post), rate)); rises.append(_rise(d, p, rate))
    taus = np.array(taus, float); rises = np.array(rises, float)
    p10, p99 = np.percentile(raw, [10, 99])
    out = {
        "F0": F0, "F_mean": float(raw.mean()), "dark_offset": float(dark), "F0_dark": float(F0d),
        "dark_frac_of_F0": float(dark / F0) if F0 > 0 else np.nan,
        "noise_sd": noise_sd, "noise_hp": float(nhp), "sd_total": float(d.std()),
        "p99": float(np.percentile(d, 99)), "max": float(d.max()),
        "n_events": int(len(pk)), "event_rate_per_min": float(len(pk) / minutes),
        "n_events_4sd": int(len(pk4)), "event_rate_4sd_per_min": float(len(pk4) / minutes),
        "event_amp_median": amp, "event_amp_p90": float(np.percentile(prom, 90)) if len(prom) else np.nan,
        "event_amp_4sd_median": float(np.median(pr4)) if len(pr4) else np.nan,
        "event_amp_median_darkcorr": float(amp * F0 / F0d) if F0d > 0 and np.isfinite(amp) else np.nan,
        "snr": float(amp / noise_sd) if np.isfinite(amp) else np.nan,
        "snr_p99": float(np.percentile(d, 99) / noise_sd),
        "dyn_range": float((p99 - p10) / F0) if F0 > 0 else np.nan,
        "dyn_range_dark": float((p99 - p10) / F0d) if F0d > 0 else np.nan,
        "F0_adc": float(F0 - ADC_OFFSET), "dark_adc": float(dark - ADC_OFFSET),
        "event_amp_median_adc": float(amp * F0 / (F0 - ADC_OFFSET)) if F0 > ADC_OFFSET and np.isfinite(amp) else np.nan,
        "event_amp_F": float(amp * F0) if np.isfinite(amp) else np.nan,
        "noise_sd_adc": float(noise_sd * F0 / (F0 - ADC_OFFSET)) if F0 > ADC_OFFSET else np.nan,
        "dyn_range_adc": float((p99 - p10) / (F0 - ADC_OFFSET)) if F0 > ADC_OFFSET else np.nan,
        "skew": float(sst.skew(d)), "kurtosis": float(sst.kurtosis(d)),
        "frac_time_in_events": float(ev_win.mean()), "frac_baseline_frames": float((~excl).mean()),
        "n_isolated": int(len(taus)), "n_decay_fit": int(np.isfinite(taus).sum()),
        "decay_tau_s": float(np.nanmedian(taus)) if np.isfinite(taus).any() else np.nan,
        "rise_time_s": float(np.nanmedian(rises)) if np.isfinite(rises).any() else np.nan,
        "noise_sd_F": float(noise_sd * F0), "n_vox": int(n_vox),
    }
    out["var_per_F"] = float(out["noise_sd_F"] ** 2 * n_vox / F0d) if F0d > 0 else np.nan
    out["var_per_F_adc"] = float(out["noise_sd_F"] ** 2 * n_vox / (F0 - ADC_OFFSET)) if F0 > ADC_OFFSET else np.nan
    return out, d


# ---------------------------------------------------------------- run enumeration
def _cohort_bases(root: Path) -> set:
    p = root / "stats" / "cohort_metrics.csv"
    return set(pd.read_csv(p)["behavior_base"]) if p.exists() else set()


def enumerate_runs(roots: dict) -> list[dict]:
    marks = load_marks()
    runs = []
    for tag, root in roots.items():
        coh = _cohort_bases(root)
        for f in sorted(glob.glob(str(root / "rbp4_*/**/*_metrics.json"), recursive=True)):
            if "/old/" in f:
                continue
            mp = Path(f); stem = mp.name[:-len("_metrics.json")]
            stack, seg = mp.with_name(stem + ".tif"), mp.with_name(stem + "_segments_final.tif")
            if not (stack.exists() and seg.exists()):
                continue
            try:
                m = json.loads(mp.read_text())
            except Exception:
                continue
            bb = m.get("behavior_base")
            mk = marks.get(bb) or {}
            reason = mk.get("reason", "") or ""
            src = "" if not mk else ("auto-QC" if reason.startswith("[auto-QC]") else "z-QC" if reason.startswith("[z-QC]") else "daria")
            stale = mp.stat().st_mtime < max(seg.stat().st_mtime, ignore_path(seg).stat().st_mtime if ignore_path(seg).exists() else 0)
            if tag == "real":
                s = "curated" if bb in coh else "curated_other"
            else:
                s = "screening" if bb in coh else "screening_other"
            runs.append({"root": tag, "root_path": str(root), "set": s, "behavior_base": bb, "mouse": m.get("mouse"),
                         "date": m.get("date"), "run_dir": str(mp.parent.relative_to(root)), "stem": stem,
                         "metrics_file": str(mp), "metrics_stale": bool(stale), "mark": mk.get("mark", ""),
                         "mark_reason": reason, "mark_source": src})
    return runs


# ---------------------------------------------------------------- one run
def _cache_key(stack: Path, seg: Path, mp: Path) -> str:
    ig = ignore_path(seg)
    return "|".join(f"{p.name}:{p.stat().st_mtime:.0f}" for p in (stack, seg, mp) + ((ig,) if ig.exists() else ())) + f"|v{__version__}"


def features_for_run(run: dict, force: bool = False):
    root = Path(run["root_path"]); rd = root / run["run_dir"]; stem = run["stem"]
    stack_p, seg_p, mp = rd / f"{stem}.tif", rd / f"{stem}_segments_final.tif", Path(run["metrics_file"])
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{run['root']}__{run['behavior_base']}__{stem}.json"
    key = _cache_key(stack_p, seg_p, mp)
    if cp.exists() and not force:
        c = json.loads(cp.read_text())
        if c.get("key") == key:
            return c["regions"], c["run"]
    m = json.loads(mp.read_text())
    rate = float(m.get("frame_rate_hz") or np.nan)
    if not np.isfinite(rate) or rate <= 0:
        return None, {"error": "no frame rate"}
    stack = tifffile.imread(stack_p); seg = tifffile.imread(seg_p)
    T = stack.shape[0]; flat = stack.reshape(T, -1)
    labels = sorted(int(v) for v in np.unique(seg) if v > 0)
    names = region_names(seg_p.with_suffix(".json"), labels)
    seg, ignored, ign = apply_ignore(seg, names, seg_p)
    labels = [l for l in labels if l not in ign]
    mean_img = stack.mean(0, dtype=np.float64)
    dark = float(np.percentile(mean_img, 1))
    # reference region: same rule as run_metrics (soma, else the named proximal trunk)
    comp = {l: compartment_of(names[l]) for l in labels}
    soma = [l for l in labels if comp[l] == "soma"]
    ref = soma[0] if soma else next((l for l in labels if names[l] == m.get("reference_region")), None)
    mreg = {v.get("name"): v for v in (m.get("regions") or {}).values()}
    rows, dtr = [], {}
    for l in labels:
        idx = np.flatnonzero((seg == l).ravel())
        raw = flat[:, idx].mean(1)
        f, d = trace_features(raw, rate, dark, len(idx))
        dtr[l] = d
        info = mreg.get(names[l], {})
        rows.append({"label": l, "region": names[l], "compartment": comp[l], "is_reference": l == ref,
                     "reliability": info.get("reliability"), "distance_um": info.get("distance_um"), **f})
    # pairwise correlation over all regions (and disattenuated by split-half reliability)
    rs, rcs = [], []
    rel = {r["label"]: r["reliability"] for r in rows}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            r = float(np.corrcoef(dtr[a], dtr[b])[0, 1]); rs.append(r)
            if rel[a] and rel[b] and rel[a] > 0 and rel[b] > 0:
                rcs.append(min(1.0, r / np.sqrt(rel[a] * rel[b])))
    runrow = {"frame_rate_hz": rate, "T": int(T), "duration_s": T / rate, "n_regions": len(labels),
              "reference": m.get("reference"), "reference_region": names.get(ref) if ref else None,
              "ignored_regions": ";".join(ignored), "dark_offset": dark,
              "mean_img_p50": float(np.percentile(mean_img, 50)), "mean_img_p99": float(np.percentile(mean_img, 99)),
              "mean_pairwise_r": float(np.mean(rs)) if rs else np.nan,
              "mean_pairwise_r_corr": float(np.mean(rcs)) if rcs else np.nan}
    for k in ("r_soma_branch", "r_soma_branch_corr", "r_soma_branch_core", "r_soma_trunk", "r_soma_trunk_corr",
              "r_trunk_branch", "soma_snr", "imaging_quality"):
        runrow[k] = m.get(k)
    del stack, flat
    cp.write_text(json.dumps({"key": key, "regions": rows, "run": runrow}, default=float))
    return rows, runrow


FEATS = ["F0", "F_mean", "F0_dark", "dark_frac_of_F0", "noise_sd", "noise_hp", "sd_total", "p99", "max",
         "n_events", "event_rate_per_min", "n_events_4sd", "event_rate_4sd_per_min", "event_amp_median",
         "event_amp_p90", "event_amp_4sd_median", "event_amp_median_darkcorr", "snr", "snr_p99", "dyn_range",
         "dyn_range_dark", "skew", "kurtosis", "frac_time_in_events", "frac_baseline_frames", "decay_tau_s",
         "rise_time_s", "noise_sd_F", "var_per_F", "n_vox", "reliability", "n_decay_fit",
         "F0_adc", "dark_adc", "event_amp_median_adc", "event_amp_F", "noise_sd_adc", "dyn_range_adc", "var_per_F_adc"]


def run_level(run: dict, rows: list[dict], runrow: dict) -> dict:
    df = pd.DataFrame(rows)
    out = {k: run[k] for k in ("set", "root", "behavior_base", "mouse", "date", "run_dir", "stem", "mark",
                               "mark_reason", "mark_source", "metrics_stale")}
    out["session"] = f"{run['mouse']}_{run['date']}"
    out.update(runrow)
    groups = {"ref": df[df.is_reference], "branch": df[df.compartment == "branch"],
              "trunk": df[(df.compartment == "trunk") & ~df.is_reference], "cell": df}
    for g, sub in groups.items():
        for k in FEATS:
            out[f"{g}_{k}"] = float(pd.to_numeric(sub[k], errors="coerce").mean()) if len(sub) else np.nan
    out["n_branch_regions"] = int(len(groups["branch"]))
    return out


def log(what, result):
    p = PROJECT / "auto_pipeline" / "logs" / "pi_hypothesis_v7.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps({"time": pd.Timestamp.utcnow().isoformat(), "stage": "pi_hypothesis",
                            "what": what, "result": result}) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["real", "auto"], choices=["real", "auto"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-prefix", default="trace_features", help="stats/<prefix>.csv and stats/<prefix>_run.csv")
    a = ap.parse_args(argv)
    roots = {"real": PROJECT, "auto": PROJECT / "auto_pipeline"}
    roots = {k: v for k, v in roots.items() if k in a.roots}
    runs = enumerate_runs(roots)
    print(f"{len(runs)} run(s) with stack + regions + metrics")
    reg_rows, run_rows = [], []
    t0 = time.time()
    for i, r in enumerate(runs):
        try:
            rows, rr = features_for_run(r, a.force)
        except Exception as e:
            print(f"  [{i+1}/{len(runs)}] {r['root']} {r['behavior_base']}: FAILED {e}"); continue
        if rows is None:
            print(f"  [{i+1}/{len(runs)}] {r['root']} {r['behavior_base']}: skipped ({rr.get('error')})"); continue
        for x in rows:
            reg_rows.append({**{k: r[k] for k in ("set", "root", "behavior_base", "mouse", "date", "run_dir", "stem",
                                                   "mark", "mark_source")}, **x})
        run_rows.append(run_level(r, rows, rr))
        print(f"  [{i+1}/{len(runs)}] {r['root']:4s} {r['set']:15s} {r['behavior_base']}: {len(rows)} regions "
              f"({time.time()-t0:.0f} s)", flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(reg_rows).to_csv(OUT / f"{a.out_prefix}.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUT / f"{a.out_prefix}_run.csv", index=False)
    print(f"wrote stats/{a.out_prefix}.csv ({len(reg_rows)} region rows) and stats/{a.out_prefix}_run.csv ({len(run_rows)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
