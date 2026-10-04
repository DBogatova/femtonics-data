#!/usr/bin/env python
"""behavior_coupling.py - how each region of a cell relates to behavior.

For every run with regions, on the REAL volume times (taken from the AndorXyla trigger
pulses, not frame/rate), for pupil, whisking and accelerometer:

  cross-correlation   per region (reference, trunk, branches) and the whole cell:
                      r at lag 0 and the strongest r within +-MAX_LAG s with its lag
                      (positive lag = behavior follows the cell). Significance from a
                      circular-shift null (shifts > 20 s, 500 draws): p and the 95 % band.
  coherence           magnitude-squared coherence (Welch) averaged in bands
                      slow 0.01-0.1 Hz, mid 0.1-0.5 Hz, fast 0.5-1.5 Hz.
  state dependence    rate of branch-only events (no reference event within +-2 frames)
                      and of global events in ACTIVE vs QUIET frames (active = pupil or
                      whisking above its median); ratio and a binomial test.

Outputs: <stem>_behavior_coupling.json and <stem>_behavior_coupling.png per run;
the cohort is collected by cohort_stats.py.

  python code/STEP8_stats/behavior_coupling.py --all
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy.signal import coherence
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
_CODE_ROOT = HERE.parents[1]
ROOT = Path(os.environ["FEMTO_ROOT"]).resolve() if os.environ.get("FEMTO_ROOT") else _CODE_ROOT
sys.path.insert(0, str(_CODE_ROOT / "code")); sys.path.insert(0, str(_CODE_ROOT / "code/STEP7_workflow")); sys.path.insert(0, str(HERE))
from run_metrics import dff, region_names, compartment_of, events, guideline_deep_end_first   # noqa: E402
from common.regions import apply_ignore, newest_input_mtime   # noqa: E402

__version__ = "0.1.0"
MAX_LAG_S = 5.0
BANDS = {"slow 0.01-0.1 Hz": (0.01, 0.1), "mid 0.1-0.5 Hz": (0.1, 0.5), "fast 0.5-1.5 Hz": (0.5, 1.5)}


def session_dir(run_dir: Path) -> Path:
    return run_dir.parents[1] if run_dir.parent.name == "preprocessed" else run_dir.parent


def volume_times(sdir: Path, run_id: str, T: int, rate: float):
    """Mid-time of every imaging volume (s, aligned to the first imaging pulse)."""
    f = sorted(glob.glob(str(sdir / "trigger" / f"{run_id}*_trigger.csv")))
    if not f:
        return np.arange(T) / rate, "frame/rate (no trigger file)"
    tr = pd.read_csv(f[0], usecols=["aligned_time_s", "AndorXylaTrigger"])
    e = np.flatnonzero(np.diff((tr.AndorXylaTrigger.values > 0.5).astype(int)) == 1) + 1
    nz = max(1, int(round(len(e) / T)))
    if len(e) < nz * T:
        return np.arange(T) / rate, "frame/rate (trigger count mismatch)"
    t = tr.aligned_time_s.values[e]
    return (t[::nz][:T] + t[nz - 1::nz][:T]) / 2, f"trigger pulses ({nz} planes/volume)"


def behavior_on_volumes(sdir: Path, base: str, run_id: str, tv: np.ndarray, rate: float):
    out = {}
    bf = sdir / "behavior" / f"{base}_behavior.csv"
    if bf.exists():
        b = pd.read_csv(bf)
        if "in_imaging_window" in b:
            b = b[b.in_imaging_window == 1]
        for name, col in (("pupil", "pupil_smooth"), ("whisking", "whisker_smooth_pad")):
            if col in b and b[col].notna().sum() > 10:
                ok = b[col].notna().values
                out[name] = np.interp(tv, b.aligned_time_s.values[ok], b[col].values[ok])
    af = sorted(glob.glob(str(sdir / "trigger" / f"{run_id}*_accel.csv")))
    if af:
        a = pd.read_csv(af[0], usecols=["aligned_time_s", "accel_mag"])
        ta, va = a.aligned_time_s.values, a.accel_mag.values
        half = 0.5 / rate; lo = np.searchsorted(ta, tv - half); hi = np.searchsorted(ta, tv + half)
        cs = np.concatenate([[0], np.cumsum(np.nan_to_num(va))])
        n = np.maximum(hi - lo, 1); out["accelerometer"] = (cs[hi] - cs[lo]) / n
    return out


def xcorr(a, b, max_lag):
    a = (a - a.mean()) / (a.std() + 1e-12); b = (b - b.mean()) / (b.std() + 1e-12); T = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    return lags, np.array([np.mean(a[max(0, -l):T - max(0, l)] * b[max(0, l):T - max(0, -l)]) for l in lags])


def shift_null(a, b, max_lag, rate, n=500, seed=0):
    rng = np.random.default_rng(seed); T = len(a); minshift = int(20 * rate)
    if T <= 2 * minshift:
        return None
    peaks = []
    for _ in range(n):
        sh = int(rng.integers(minshift, T - minshift)); _, x = xcorr(a, np.roll(b, sh), max_lag); peaks.append(np.max(np.abs(x)))
    return np.array(peaks)


def analyze_run(run: dict, root: Path):
    run_dir = root / run["run_dir"]; stem = run["stem"]; sdir = session_dir(run_dir)
    seg_p = run_dir / f"{stem}_segments_final.tif"
    if not seg_p.exists():
        return None
    base = run.get("behavior_base") or f"{run['mouse']}_{run.get('munit', '?')}"
    run_id = base.split("_")[-1] if "_" in base else ""
    stack = tifffile.imread(run_dir / f"{stem}.tif"); seg = tifffile.imread(seg_p); T = stack.shape[0]
    rate = float(run.get("frame_rate_hz") or 0) or T / 240.0
    tv, tsrc = volume_times(sdir, run_id, T, rate)
    beh = behavior_on_volumes(sdir, base, run_id, tv, rate)
    if not beh:
        return {"behavior_base": base, "note": "no behavior data"}
    labels = sorted(int(v) for v in np.unique(seg) if v > 0)
    names = region_names(seg_p.with_suffix(".json"), labels)
    seg, _ignored, _ign = apply_ignore(seg, names, seg_p)          # <stem>_ignore.json
    labels = [l for l in labels if l not in _ign]
    comp = {l: compartment_of(names[l]) for l in labels}
    flat = stack.reshape(T, -1)
    tr = {names[l]: dff(flat[:, np.flatnonzero((seg == l).ravel())].mean(1).astype(np.float64)) for l in labels}
    tr["whole cell"] = dff(flat[:, np.flatnonzero((seg > 0).ravel())].mean(1).astype(np.float64))
    L = int(round(MAX_LAG_S * rate))
    res = {"behavior_base": base, "mouse": run["mouse"], "date": run["date"], "version": __version__,
           "ignored_regions": _ignored,
           "time_source": tsrc, "max_lag_s": MAX_LAG_S, "regions": {}, "behaviors": list(beh)}
    for rn, t in tr.items():
        row = {"compartment": comp.get(next((l for l in labels if names[l] == rn), -1), "cell") if rn != "whole cell" else "cell"}
        for bn, bv in beh.items():
            lags, x = xcorr(t, bv, L); k = int(np.argmax(np.abs(x)))
            null = shift_null(t, bv, L, rate) if rn == "whole cell" or row["compartment"] in ("soma", "branch") else None
            p = float((np.sum(null >= abs(x[k])) + 1) / (len(null) + 1)) if null is not None else None
            f, C = coherence(t, bv, fs=rate, nperseg=min(256, T // 4))
            coh = {bnd: float(np.nanmean(C[(f >= lo) & (f < hi)])) for bnd, (lo, hi) in BANDS.items()}
            row[bn] = {"r_lag0": float(x[L]), "r_peak": float(x[k]), "lag_peak_s": float(lags[k] / rate), "p_peak": p,
                       "null95": float(np.percentile(null, 95)) if null is not None else None, "coherence": coh}
        res["regions"][rn] = row
    # state dependence of branch-only vs global events
    ref = next((l for l in labels if comp[l] == "soma"), None)
    if ref is None:
        tk = [l for l in labels if comp[l] == "trunk"]
        if tk:
            xs = {l: np.argwhere(seg == l)[:, 2].mean() for l in tk}
            ref = min(xs, key=xs.get) if guideline_deep_end_first(run, root) else max(xs, key=xs.get)
    brs = [l for l in labels if comp[l] == "branch" and l != ref]
    state = None
    if ref is not None and brs and ("pupil" in beh or "whisking" in beh):
        active = np.zeros(T, bool)
        for bn in ("pupil", "whisking"):
            if bn in beh: active |= beh[bn] > np.median(beh[bn])
        rev = events(tr[names[ref]]); bev = np.unique(np.concatenate([events(tr[names[l]]) for l in brs]))
        near = lambda a, b: np.array([np.any(np.abs(b - x) <= 2) for x in a], bool) if len(a) and len(b) else np.zeros(len(a), bool)
        only = bev[~near(bev, rev)]; glob_ = rev[near(rev, bev)]
        fa = active.mean(); n_on_act = int(active[only].sum()) if len(only) else 0
        state = {"frac_frames_active": float(fa),
                 "branch_only_events": int(len(only)), "branch_only_in_active": n_on_act,
                 "branch_only_rate_ratio_active_vs_quiet": (float((n_on_act / fa) / ((len(only) - n_on_act) / (1 - fa)))
                                                            if len(only) >= 5 and 0 < n_on_act < len(only) else None),
                 "p_branch_only_state": float(stats.binomtest(n_on_act, len(only), fa).pvalue) if len(only) else None,
                 "global_events": int(len(glob_)), "global_in_active": int(active[glob_].sum()) if len(glob_) else 0,
                 "p_global_state": float(stats.binomtest(int(active[glob_].sum()), len(glob_), fa).pvalue) if len(glob_) else None,
                 "reference": names[ref]}
    res["state_dependence"] = state
    # figure: cross-correlograms (whole cell + each region) for each behavior
    bnames = list(beh); fig, ax = plt.subplots(1, len(bnames), figsize=(4.6 * len(bnames), 3.6), squeeze=False)
    cols = plt.cm.tab10.colors
    for j, bn in enumerate(bnames):
        a = ax[0, j]
        for i, (rn, t) in enumerate(tr.items()):
            lags, x = xcorr(t, beh[bn], L)
            a.plot(lags / rate, x, color="k" if rn == "whole cell" else cols[i % 10], lw=2 if rn == "whole cell" else 1, label=rn)
        n95 = res["regions"]["whole cell"][bn]["null95"]
        if n95:
            a.axhspan(-n95, n95, color="0.85", zorder=0, label="95% shuffle band")
        a.axvline(0, color="k", lw=0.5); a.set_xlabel("lag (s)  [+ = behavior follows cell]"); a.set_title(bn, loc="left", fontsize=10)
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    ax[0, 0].set_ylabel("correlation"); ax[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{base}: activity vs behavior (time from the imaging triggers)", fontsize=10)
    fig.tight_layout(); fig.savefig(run_dir / f"{stem}_behavior_coupling.png", dpi=150); plt.close(fig)
    (run_dir / f"{stem}_behavior_coupling.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--run"); g.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    from femto_status import build_status
    runs = [r for r in build_status(ROOT) if r.get("run_dir") and r.get("stem")]
    for r in [r for r in runs if r.get("mark")]:
        bb = r.get("behavior_base") or f"{r['mouse']}_{r.get('munit', '?')}"
        print(f"  {bb}: skipped ({r['mark']}{': ' + r['mark_reason'] if r.get('mark_reason') else ''})")
    runs = [r for r in runs if not r.get("mark")]                 # run_marks.csv: excluded / revisit
    if args.run:
        runs = [r for r in runs if r.get("behavior_base") == args.run
                or (r.get("_imaging_only") and f"{r['mouse']}_{r.get('munit', '')}" == args.run)]
    n = 0
    for r in runs:
        seg_p = ROOT / r["run_dir"] / f"{r['stem']}_segments_final.tif"; out_p = ROOT / r["run_dir"] / f"{r['stem']}_behavior_coupling.json"
        if not seg_p.exists():
            continue
        if out_p.exists() and out_p.stat().st_mtime >= newest_input_mtime(seg_p) and not args.force:
            n += 1; continue
        res = analyze_run(r, ROOT); n += 1
        bb = r.get("behavior_base") or f"{r['mouse']}_{r.get('munit', '?')}"
        if res and "note" not in res:
            wc = res["regions"]["whole cell"]
            print(f"  {bb}: " + "; ".join(
                f"{b} r0 {wc[b]['r_lag0']:+.2f} peak {wc[b]['r_peak']:+.2f}@{wc[b]['lag_peak_s']:+.1f}s p={wc[b]['p_peak']:.3f}" for b in res["behaviors"]))
        elif res and "note" in res:
            print(f"  {bb}: {res['note']}")
    print(f"done: {n} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
