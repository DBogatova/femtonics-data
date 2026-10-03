#!/usr/bin/env python
"""run_metrics.py - one row of soma-dendrite coupling metrics per run.

Computed from the RAW registered stack and the saved regions (never from display-masked
data), for every run that has <stem>_segments_final.tif. Written to
<run_dir>/<stem>_metrics.json and collected by cohort_stats.py.

Regions are grouped into compartments by their saved names: soma* -> soma, trunk* ->
trunk, everything else -> branch. Metrics:

  coupling
    r_soma_branch          Pearson r of dF/F, soma vs each branch (mean over branches)
    r_soma_branch_core     same, from the eroded core of each region (halo control: if
                           the halo drove the coupling, core r is clearly lower)
    r_soma_trunk, r_trunk_branch
  events (per compartment, prominence-based peaks on dF/F)
    rate_<comp>_per_min
    frac_branch_independent   branch events with NO soma event within +-window frames
    frac_soma_independent     soma events with no branch event within +-window
    frac_global               events present in soma AND >=1 branch within window
    branch_first_frac         of paired soma+branch events, fraction where the branch
                              peaks first (ties excluded)
    lag_soma_branch_frames    cross-correlation lag of soma vs mean-branch dF/F
  behavior state (from the published behavior CSV, on the imaging frame axis)
    r_soma_branch_quiet / _active   coupling in low / high arousal frames
                                    (active = pupil OR whisking above their median)
  geometry and noise
    regions[*].distance_um   distance from the soma along the dendrite (geodesic through the
                             cell mask, median over the region's voxels); 'euclidean' flag
                             if the region is not connected to the soma through the mask
    regions[*].reliability   split-half reliability of the region's dF/F (voxels split into
                             two interleaved halves, Spearman-Brown corrected): how much of
                             the trace is signal rather than noise. Grows with region size.
    r_soma_branch_corr       coupling corrected for each region's noise (disattenuated:
                             r / sqrt(rel_soma * rel_branch), capped at 1). Removes the
                             region-size effect so cells with different ROI sizes compare
                             fairly. Shared noise (motion, scattered light) is not removed,
                             so this is a conservative (still slightly high) estimate.
    coupling_by_distance     per region: distance_um, r with soma (raw and corrected)
  expression proxy
    soma_f_raw               soma mean raw fluorescence (F, not dF/F): brighter = more
                             sensor. Comparable within a mouse at fixed laser power.
    soma_snr                 soma dF/F p99 / robust noise
  covariates: mouse, date, dpi (days post injection, from mice.csv), frame_rate_hz,
    imaging_quality, n_regions, region names.

  python code/STEP8_stats/run_metrics.py --all          # every run with regions
  python code/STEP8_stats/run_metrics.py --run <behavior_base>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "code/STEP7_workflow"))
from common.voxel import resolve_voxel                        # noqa: E402

__version__ = "0.2.0"


def dff(t, f0_pct=10.0):
    f0 = np.percentile(t, f0_pct); return (t - f0) / max(f0, 1e-6)


def robust_noise(t):
    hp = t - ndi.uniform_filter1d(t, 15)
    return 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-9


def events(t, prom_frac=0.2, min_dist=5):
    pk, _ = find_peaks(t, prominence=prom_frac * (t.max() - t.min()), distance=min_dist)
    return pk


def split_half_reliability(flat, idx, seed=0):
    """Spearman-Brown corrected correlation between the mean traces of two interleaved
    random halves of a region's voxels."""
    if len(idx) < 8:
        return float("nan")
    rng = np.random.default_rng(seed); perm = rng.permutation(idx)
    a = dff(flat[:, perm[0::2]].mean(1).astype(np.float64)); b = dff(flat[:, perm[1::2]].mean(1).astype(np.float64))
    r = float(np.corrcoef(a, b)[0, 1])
    return float(2 * r / (1 + r)) if r > -0.99 else float("nan")


def distances_from_soma(seg, soma_label, mask, voxel):
    """Median geodesic distance (um, through mask | regions) from the soma region to every
    region; falls back to centroid Euclidean distance for regions not connected."""
    from skimage.graph import MCP_Geometric
    allowed = (np.asarray(mask) > 0) | (seg > 0)
    cost = np.where(allowed, 1.0, np.inf)
    mcp = MCP_Geometric(cost, sampling=tuple(voxel))
    src = [tuple(p) for p in np.argwhere(seg == soma_label)]
    dist, _ = mcp.find_costs(src)
    out = {}
    w = np.asarray(voxel, float)
    sc = np.argwhere(seg == soma_label).mean(0)
    for l in (int(v) for v in np.unique(seg) if v > 0):
        d = dist[seg == l]
        if np.isfinite(d).mean() > 0.5:
            out[l] = (float(np.median(d[np.isfinite(d)])), "geodesic")
        else:
            out[l] = (float(np.linalg.norm((np.argwhere(seg == l).mean(0) - sc) * w)), "euclidean")
    return out


def compartment_of(name: str) -> str:
    n = name.lower()
    if n.startswith("soma"):
        return "soma"
    if n.startswith("trunk"):
        return "trunk"
    return "branch"


def region_names(seg_json: Path, labels):
    """Same rule as the figure: explicit segment_names, else reconstructed from the wrap
    clicks (sorted surviving labels -> 1..N)."""
    names = {}
    if seg_json.exists():
        try:
            j = json.loads(seg_json.read_text())
            if isinstance(j.get("segment_names"), dict):
                names = {int(k): v for k, v in j["segment_names"].items()}
            else:
                last = {}
                for w in j.get("wrap_clicks", []):
                    last[int(w["label"])] = w.get("name") or f"seg{w['label']}"
                pre = sorted(last)
                if len(pre) == len(labels):
                    names = {f: last[p_] for f, p_ in zip(labels, pre)}
        except Exception:
            pass
    return {l: names.get(l, f"seg{l}") for l in labels}


def behavior_state(run_dir: Path, base: str, T: int, rate_hz: float):
    """Boolean 'active' per imaging frame from the published behavior CSV, or None."""
    session = run_dir.parents[1] if run_dir.parent.name == "preprocessed" else run_dir.parent
    csv = session / "behavior" / f"{base}_behavior.csv"
    if not csv.exists():
        return None, None
    b = pd.read_csv(csv)
    if "in_imaging_window" in b.columns:
        b = b[b["in_imaging_window"] == 1]
    tcol = "aligned_time_s" if "aligned_time_s" in b.columns else None
    if tcol is None or len(b) == 0:
        return None, None
    pup = "pupil_smooth" if "pupil_smooth" in b.columns else next((c for c in b.columns if "pupil" in c.lower()), None)
    whi = "whisker_smooth_pad" if "whisker_smooth_pad" in b.columns else next((c for c in b.columns if "whisk" in c.lower()), None)
    if pup is None and whi is None:
        return None, None
    frames = np.clip(np.round(b[tcol].values * rate_hz).astype(int), 0, T - 1)
    def per_frame(col):
        v = pd.to_numeric(b[col], errors="coerce").values
        out = np.full(T, np.nan); s = pd.Series(v).groupby(frames).mean()
        out[s.index.values] = s.values
        return pd.Series(out).interpolate(limit_direction="both").values
    p = per_frame(pup) if pup else None; w = per_frame(whi) if whi else None
    active = np.zeros(T, bool)
    if p is not None: active |= p > np.nanmedian(p)
    if w is not None: active |= w > np.nanmedian(w)
    return active, {"pupil_col": pup, "whisk_col": whi, "frac_active": float(active.mean())}


def metrics_for_run(run: dict, root: Path, window: int = 2, prom_frac: float = 0.2) -> dict | None:
    run_dir = root / run["run_dir"]; stem = run["stem"]
    stack_p = run_dir / f"{stem}.tif"; seg_p = run_dir / f"{stem}_segments_final.tif"
    if not (stack_p.exists() and seg_p.exists()):
        return None
    voxel = resolve_voxel(stack_p, None, quiet=True)
    stack = tifffile.imread(stack_p); seg = tifffile.imread(seg_p)
    T = stack.shape[0]; rate = float(run.get("frame_rate_hz") or 0) or T / 240.0
    labels = sorted(int(v) for v in np.unique(seg) if v > 0)
    names = region_names(seg_p.with_suffix(".json"), labels)
    comp = {l: compartment_of(names[l]) for l in labels}
    flat = stack.reshape(T, -1)
    def trace(mask):
        idx = np.flatnonzero(mask.ravel()); return flat[:, idx].mean(1).astype(np.float64)
    raw = {l: trace(seg == l) for l in labels}
    tr = {l: dff(raw[l]) for l in labels}
    core = {}
    for l in labels:
        er = ndi.binary_erosion(seg == l, structure=np.ones((3, 3, 3)))
        core[l] = dff(trace(er)) if er.sum() >= 20 else tr[l]
    by = {c: [l for l in labels if comp[l] == c] for c in ("soma", "trunk", "branch")}
    out = {"behavior_base": run["behavior_base"], "mouse": run["mouse"], "date": run["date"],
           "rank": int(run.get("rank", 0)), "frame_rate_hz": rate, "T": int(T),
           "imaging_quality": run.get("quality"), "n_regions": len(labels),
           "regions": {str(l): {"name": names[l], "compartment": comp[l], "voxels": int((seg == l).sum())} for l in labels},
           "params": {"window_frames": window, "prom_frac": prom_frac, "version": __version__}}
    if not by["soma"] or not by["branch"]:
        out["note"] = "needs a region named soma* and at least one branch region"
        return out
    s = by["soma"][0]; soma = tr[s]; soma_core = core[s]
    # geometry and noise per region
    mask_p = run_dir / f"{stem}_autoseg_labelmap_reviewed.tif"
    cell_mask = tifffile.imread(mask_p) if mask_p.exists() else (seg > 0)
    dist = distances_from_soma(seg, s, cell_mask, voxel)
    rel = {l: split_half_reliability(flat, np.flatnonzero((seg == l).ravel())) for l in labels}
    for l in labels:
        out["regions"][str(l)].update({"distance_um": round(dist[l][0], 2), "distance_kind": dist[l][1],
                                       "reliability": round(rel[l], 4) if rel[l] == rel[l] else None})
    def r_corr(a_lab, b_lab, r_raw):
        ra, rb = rel[a_lab], rel[b_lab]
        if not (ra == ra and rb == rb) or ra <= 0 or rb <= 0:
            return float("nan")
        return float(min(1.0, r_raw / np.sqrt(ra * rb)))
    branches = by["branch"]; br_mean = np.mean([tr[l] for l in branches], axis=0)
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    out["r_soma_branch"] = float(np.mean([r(soma, tr[l]) for l in branches]))
    out["r_soma_branch_core"] = float(np.mean([r(soma_core, core[l]) for l in branches]))
    out["r_soma_branch_per_region"] = {names[l]: r(soma, tr[l]) for l in branches}
    out["r_soma_branch_corr"] = float(np.nanmean([r_corr(s, l, r(soma, tr[l])) for l in branches]))
    out["coupling_by_distance"] = [
        {"region": names[l], "compartment": comp[l], "distance_um": round(dist[l][0], 2),
         "r_with_soma": round(r(soma, tr[l]), 4), "r_with_soma_corr": round(r_corr(s, l, r(soma, tr[l])), 4),
         "reliability": round(rel[l], 4), "voxels": int((seg == l).sum())}
        for l in labels if l != s]
    if by["trunk"]:
        tk = np.mean([tr[l] for l in by["trunk"]], axis=0)
        out["r_soma_trunk"] = r(soma, tk); out["r_trunk_branch"] = r(tk, br_mean)
        out["r_soma_trunk_corr"] = float(np.nanmean([r_corr(s, l, r(soma, tr[l])) for l in by["trunk"]]))
    # events
    ev = {l: events(tr[l], prom_frac) for l in labels}
    minutes = T / rate / 60.0
    for c in ("soma", "trunk", "branch"):
        if by[c]:
            out[f"rate_{c}_per_min"] = float(np.mean([len(ev[l]) for l in by[c]]) / minutes)
    soma_ev = ev[s]; br_ev = np.unique(np.concatenate([ev[l] for l in branches])) if branches else np.array([], int)
    def near(a, b):
        return np.array([np.any(np.abs(b - x) <= window) for x in a], bool) if len(a) and len(b) else np.zeros(len(a), bool)
    b_near_s = near(br_ev, soma_ev); s_near_b = near(soma_ev, br_ev)
    out["n_branch_events"] = int(len(br_ev)); out["n_soma_events"] = int(len(soma_ev))
    out["frac_branch_independent"] = float(1 - b_near_s.mean()) if len(br_ev) else np.nan
    out["frac_soma_independent"] = float(1 - s_near_b.mean()) if len(soma_ev) else np.nan
    out["frac_global"] = float(s_near_b.mean()) if len(soma_ev) else np.nan
    # who leads, among paired events
    firsts = []
    for x in soma_ev:
        d = br_ev[np.abs(br_ev - x) <= window]
        if len(d):
            dd = d[np.argmin(np.abs(d - x))] - x
            if dd != 0: firsts.append(dd < 0)
    out["n_paired_events"] = int(len(firsts))
    out["branch_first_frac"] = float(np.mean(firsts)) if firsts else np.nan
    # lag
    a = soma - soma.mean(); b = br_mean - br_mean.mean(); L = max(1, min(30, T - 1))
    xc = [np.dot(a[l:], b[:T - l]) if l >= 0 else np.dot(a[:T + l], b[-l:]) for l in range(-L, L + 1)]
    out["lag_soma_branch_frames"] = int(np.arange(-L, L + 1)[int(np.argmax(xc))])
    # behavior state
    active, binfo = behavior_state(run_dir, run["behavior_base"], T, rate)
    if active is not None and 0.1 < active.mean() < 0.9:
        out["r_soma_branch_active"] = r(soma[active], br_mean[active])
        out["r_soma_branch_quiet"] = r(soma[~active], br_mean[~active])
        out["behavior"] = binfo
    # expression proxy
    out["soma_f_raw"] = float(raw[s].mean())
    out["soma_snr"] = float(np.percentile(soma, 99) / robust_noise(soma))
    out["branch_f_raw"] = float(np.mean([raw[l].mean() for l in branches]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run"); g.add_argument("--all", action="store_true")
    ap.add_argument("--window", type=int, default=2, help="frames: soma and branch events this close count as the same event")
    ap.add_argument("--prom-frac", type=float, default=0.2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    from femto_status import build_status
    runs = [r for r in build_status(ROOT) if r.get("run_dir") and r.get("stem")]
    if args.run:
        runs = [r for r in runs if r["behavior_base"] == args.run]
    n = 0
    for r in runs:
        run_dir = ROOT / r["run_dir"]; out_p = run_dir / f"{r['stem']}_metrics.json"
        seg_p = run_dir / f"{r['stem']}_segments_final.tif"
        if not seg_p.exists():
            continue
        if out_p.exists() and out_p.stat().st_mtime >= seg_p.stat().st_mtime and not args.force:
            print(f"  {r['behavior_base']}: metrics up to date"); n += 1; continue
        m = metrics_for_run(r, ROOT, args.window, args.prom_frac)
        if m is None:
            continue
        out_p.write_text(json.dumps(m, indent=2, default=float))
        msg = m.get("note") or (f"r(soma,branch)={m['r_soma_branch']:.2f} core={m['r_soma_branch_core']:.2f} "
                                f"branch-independent={m['frac_branch_independent']:.2f} "
                                f"branch-first={m.get('branch_first_frac', float('nan')):.2f}")
        print(f"  {r['behavior_base']}: {msg}"); n += 1
    print(f"done: {n} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
