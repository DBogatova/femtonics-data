#!/usr/bin/env python
"""pi_hypothesis.py - do highly correlated cells have low baseline and big peaks?

The PI's idea: in runs where the reference (soma / proximal trunk) and the branches are
highly correlated, the cell buffers Ca2+ at a LOW baseline and fires BIG, all-or-none
events once threshold is reached; runs with low correlation have a HIGHER noise profile
and SMALLER peaks. Tested on the per-run trace features of trace_features.py (raw stack,
regions honoring the ignore list, run_metrics dF/F + events):

  primary family (BH-FDR over these tests), reference-region features vs r_soma_branch:
      F0_adc (baseline above the detector zero, predicted -), noise_sd (-), sd_total (+/-),
      event_amp_median (+), p99 (+), snr (+), kurtosis (+, all-or-none)
  secondary family: same features in offset-corrected dF/F and raw ADC units, branch-region
      features, and the other correlation measures (r_soma_branch_corr, r_soma_trunk,
      mean_pairwise_r, mean_pairwise_r_corr) - BH within the family
  per test: Spearman across runs (descriptive), mouse-cluster bootstrap 95% CI of rho,
      within-mouse Spearman (both variables centered on the mouse mean), MixedLM
      z(y) ~ z(x) + (1|mouse) (standardized slope, Wald CI and p; positive features are
      log10-transformed first), leave-one-mouse-out range of rho, partial Spearman and
      mixed slope controlling for log SNR (the key confound: noise lowers r)
  region level: each non-reference region's r with the reference vs its own SNR / noise /
      amplitude, MixedLM with a random intercept per run (cell).

Sets: curated = stats/cohort_metrics.csv (main test); screening = auto_pipeline cohort
(replication; it re-uses some curated runs with automatic masks/regions, so the
independent part - runs not in the curated set - is reported separately). Nothing here
feeds back into QC: no run is selected or rejected on correlation.

  python code/STEP8_stats/pi_hypothesis.py
Outputs: stats/assumptions/pi_hypothesis/{results.json, tests.csv, region_tests.csv,
         fig_pi_sd_amplitude.pdf/png, fig_baseline_vs_peak.pdf/png, fig_snr_confound.pdf/png,
         fig_region_level.pdf/png, numbers.md}
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = PROJECT / "stats" / "assumptions" / "pi_hypothesis"
__version__ = "1.1.0"   # 1.1.0: + secondary set screening_all (all auto-mirror runs, marks ignored)
N_BOOT = 2000

PRIMARY_Y = [("F0_adc", "-"), ("noise_sd", "-"), ("sd_total", "?"), ("event_amp_median", "+"),
             ("p99", "+"), ("snr", "+"), ("kurtosis", "+")]
SECONDARY_Y = ["F0", "F0_dark", "noise_sd_adc", "noise_hp", "event_amp_median_adc", "event_amp_F",
               "event_amp_4sd_median", "max", "event_rate_per_min", "event_rate_4sd_per_min", "dyn_range",
               "dyn_range_adc", "skew", "frac_time_in_events", "decay_tau_s", "rise_time_s"]
X_MAIN = "r_soma_branch"
X_OTHER = ["r_soma_branch_corr", "r_soma_trunk", "mean_pairwise_r", "mean_pairwise_r_corr"]
POSITIVE = {"F0_adc", "F0", "noise_sd", "sd_total", "event_amp_median", "p99", "snr", "noise_sd_adc", "noise_hp",
            "event_amp_median_adc", "event_amp_F", "event_amp_4sd_median", "max", "dyn_range", "dyn_range_adc",
            "decay_tau_s", "rise_time_s", "event_rate_per_min", "event_rate_4sd_per_min"}
LABEL = {"F0_adc": "baseline F0 above detector zero (ADC)", "noise_sd": "noise SD (dF/F, baseline frames)",
         "sd_total": "SD of whole trace (dF/F)", "event_amp_median": "median event amplitude (dF/F)",
         "p99": "99th percentile (dF/F)", "snr": "SNR = event amplitude / noise SD", "kurtosis": "kurtosis",
         "r_soma_branch": "r(reference, branch)"}


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]; o = np.argsort(pv); n = len(pv)
        qq = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        tmp = np.empty(n); tmp[o] = np.minimum(qq, 1); q[ok] = tmp
    return q


def _tr(df, col):
    v = pd.to_numeric(df[col], errors="coerce").astype(float)
    if col.split("_", 1)[-1] in POSITIVE or col in POSITIVE:
        v = np.log10(v.where(v > 0))
    return v


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, np.nan
    r, p = sst.spearmanr(x[ok], y[ok]); return float(r), float(p)


def _rho_fast(x, y):
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(sst.rankdata(x), sst.rankdata(y))[0, 1])


def cluster_boot_rho(df, xc, yc, B=N_BOOT, seed=0):
    """Two-stage bootstrap: resample mice, then runs within each drawn mouse."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero((df.mouse == m).values) for m in df.mouse.unique()]
    X, Y = df[xc].values, df[yc].values; vals = []
    for _ in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in (groups[i] for i in rng.integers(0, len(groups), len(groups)))])
        r = _rho_fast(X[idx], Y[idx])
        if np.isfinite(r): vals.append(r)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 100 else (np.nan, np.nan)


def within_mouse_perm(df, xc, yc, B=5000, seed=0):
    """p for the within-mouse rho: shuffle x among runs of the same mouse (tests only the
    within-mouse association; between-mouse differences cannot be tested with 5-8 mice)."""
    s = df[[xc, yc, "mouse"]].dropna()
    s = s[s.groupby("mouse")[xc].transform("size") > 1]
    if len(s) < 5:
        return np.nan
    gi = [np.flatnonzero((s.mouse == m).values) for m in s.mouse.unique()]
    rx = s.groupby("mouse")[xc].rank().values; ry = s.groupby("mouse")[yc].rank().values
    for g in gi:
        rx[g] -= rx[g].mean(); ry[g] -= ry[g].mean()
    obs = abs(np.dot(rx, ry)); rng = np.random.default_rng(seed); cnt = 0
    for _ in range(B):
        px = rx.copy()
        for g in gi:
            px[g] = rx[rng.permutation(g)]
        cnt += abs(np.dot(px, ry)) >= obs - 1e-12
    return float((cnt + 1) / (B + 1))


def within_mouse_rho(df, xc, yc):
    s = df[[xc, yc, "mouse"]].dropna()
    s = s[s.groupby("mouse")[xc].transform("size") > 1]
    if len(s) < 5:
        return np.nan, np.nan, 0
    rx = s.groupby("mouse")[xc].rank(); ry = s.groupby("mouse")[yc].rank()
    dx = rx - rx.groupby(s.mouse).transform("mean"); dy = ry - ry.groupby(s.mouse).transform("mean")
    r = float(np.corrcoef(dx, dy)[0, 1]); n = len(s); k = s.mouse.nunique()
    dfree = n - k - 1
    t = r * np.sqrt(dfree / max(1e-12, 1 - r * r)) if dfree > 0 else np.nan
    return r, float(2 * sst.t.sf(abs(t), dfree)) if dfree > 0 else np.nan, int(k)


def mixed(df, yc, xcs, group="mouse"):
    """z(y) ~ sum z(x) + (1|group). Returns dict for the first x."""
    import statsmodels.formula.api as smf
    s = df[[yc, *xcs, group]].dropna().copy()
    if len(s) < 6 or s[group].nunique() < 2:
        return {}
    for c in [yc, *xcs]:
        sd = s[c].std()
        if not sd > 0:
            return {}
        s[c] = (s[c] - s[c].mean()) / sd
    s = s.rename(columns={yc: "y", **{c: f"x{i}" for i, c in enumerate(xcs)}})
    form = "y ~ " + " + ".join(f"x{i}" for i in range(len(xcs)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fit = smf.mixedlm(form, s, groups=s[group]).fit(reml=True, method=["lbfgs", "powell"])
        except Exception as e:
            return {"error": str(e)}
        # likelihood-ratio test (ML fits with / without x0): Wald SEs are unreliable with
        # 5-8 groups and a mouse variance at the boundary
        try:
            red = ("y ~ " + " + ".join(f"x{i}" for i in range(1, len(xcs)))) if len(xcs) > 1 else "y ~ 1"
            f1 = smf.mixedlm(form, s, groups=s[group]).fit(reml=False, method=["lbfgs", "powell"])
            f0 = smf.mixedlm(red, s, groups=s[group]).fit(reml=False, method=["lbfgs", "powell"])
            lr = max(0.0, 2 * (f1.llf - f0.llf)); p_lrt = float(sst.chi2.sf(lr, 1))
        except Exception:
            p_lrt = np.nan
    ci = fit.conf_int().loc["x0"]
    return {"beta": float(fit.params["x0"]), "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "p_wald": float(fit.pvalues["x0"]),
            "p": p_lrt, "group_var": float(fit.cov_re.iloc[0, 0]), "converged": bool(fit.converged),
            "warnings": sorted({type(x.message).__name__ for x in w})}


def partial_spearman(x, y, z):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 6:
        return np.nan, np.nan
    rx, ry, rz = (sst.rankdata(v[ok]) for v in (x, y, z))
    A = np.c_[np.ones(ok.sum()), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]; ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1]); n = ok.sum()
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r))
    return r, float(2 * sst.t.sf(abs(t), n - 3))


def one_test(df, xc, yc, family, setname, expected="?"):
    s = df.copy(); s["_x"] = _tr(s, xc); s["_y"] = _tr(s, yc); s["_snr"] = _tr(s, "ref_snr")
    s = s[np.isfinite(s._x) & np.isfinite(s._y)]
    rho, p = spearman(s._x.values, s._y.values)
    lo, hi = cluster_boot_rho(s, "_x", "_y")
    rw, pw, kw = within_mouse_rho(s, "_x", "_y")
    pw = within_mouse_perm(s, "_x", "_y")
    mm = mixed(s, "_y", ["_x"])
    lomo = []
    for m in s.mouse.unique():
        r_, _ = spearman(s[s.mouse != m]._x.values, s[s.mouse != m]._y.values); lomo.append(r_)
    out = {"family": family, "set": setname, "x": xc, "y": yc, "expected_sign": expected,
           "n_runs": int(len(s)), "n_mice": int(s.mouse.nunique()), "n_sessions": int(s.session.nunique()),
           "spearman_rho": rho, "spearman_p_runs": p, "rho_ci_lo": lo, "rho_ci_hi": hi,
           "rho_within_mouse": rw, "p_within_mouse": pw, "n_mice_within": kw,
           "lomo_rho_min": float(np.nanmin(lomo)) if lomo else np.nan, "lomo_rho_max": float(np.nanmax(lomo)) if lomo else np.nan,
           "lomo_same_sign": int(np.sum(np.sign(lomo) == np.sign(rho))) if lomo and np.isfinite(rho) else 0,
           "mixed_beta": mm.get("beta"), "mixed_ci_lo": mm.get("ci_lo"), "mixed_ci_hi": mm.get("ci_hi"),
           "mixed_p": mm.get("p"), "mixed_p_wald": mm.get("p_wald"), "mixed_mouse_var": mm.get("group_var"), "mixed_converged": mm.get("converged"),
           "mixed_warnings": ";".join(mm.get("warnings", [])) or mm.get("error", "")}
    if yc != "ref_snr":
        pr, pp = partial_spearman(s._x.values, s._y.values, s._snr.values)
        mm2 = mixed(s, "_y", ["_x", "_snr"])
        out.update({"partial_rho_given_snr": pr, "partial_p_given_snr": pp, "mixed_beta_given_snr": mm2.get("beta"),
                    "mixed_p_given_snr": mm2.get("p"), "mixed_ci_lo_given_snr": mm2.get("ci_lo"),
                    "mixed_ci_hi_given_snr": mm2.get("ci_hi")})
    return out


def region_level(reg, runs_ok, setname):
    """r of each non-reference region with the reference vs that region's own features."""
    import statsmodels.formula.api as smf
    rows = []
    for _, run in runs_ok.iterrows():
        mp = Path(run["root_path"]) / run["run_dir"] / f"{run['stem']}_metrics.json"
        try:
            cbd = {c["region"]: c for c in json.loads(mp.read_text()).get("coupling_by_distance", [])}
        except Exception:
            continue
        sub = reg[(reg.root == run["root"]) & (reg.behavior_base == run["behavior_base"]) & (reg.stem == run["stem"])]
        for _, rg in sub[~sub.is_reference.astype(bool)].iterrows():
            c = cbd.get(rg["region"])
            if c is None:
                continue
            rows.append({"set": setname, "behavior_base": run["behavior_base"], "mouse": run["mouse"],
                         "region": rg["region"], "compartment": rg["compartment"], "r_with_ref": c["r_with_soma"],
                         "distance_um": c.get("distance_um"), **{k: rg[k] for k in ("snr", "noise_sd", "event_amp_median",
                                                                                     "F0_adc", "n_vox", "reliability")}})
    R = pd.DataFrame(rows); res = []
    if R.empty:
        return R, res
    for f in ("snr", "noise_sd", "event_amp_median", "F0_adc"):
        s = R.copy(); s["_y"] = s["r_with_ref"]; s["_x"] = np.log10(s[f].where(s[f] > 0)); s["_d"] = s["distance_um"]
        s = s.dropna(subset=["_x", "_y", "_d"])
        z = lambda v: (v - v.mean()) / v.std()
        s["_x"], s["_y"], s["_d"] = z(s._x), z(s._y), z(s._d)
        out = {"set": setname, "feature": f, "n_regions": int(len(s)), "n_runs": int(s.behavior_base.nunique()),
               "n_mice": int(s.mouse.nunique()), "spearman_rho": spearman(s._x.values, s._y.values)[0]}
        for lab, form in (("", "_y ~ _x"), ("_given_distance", "_y ~ _x + _d")):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.mixedlm(form, s, groups=s["behavior_base"]).fit(reml=True)
                ci = fit.conf_int().loc["_x"]
                out.update({f"beta{lab}": float(fit.params["_x"]), f"ci_lo{lab}": float(ci[0]),
                            f"ci_hi{lab}": float(ci[1]), f"p{lab}": float(fit.pvalues["_x"])})
            except Exception as e:
                out[f"error{lab}"] = str(e)
        res.append(out)
    return R, res


# ------------------------------------------------------------------- figures
def _palette(mice):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("tab10"); return {m: cm(i % 10) for i, m in enumerate(sorted(mice))}


def _scatter(ax, sets, xc, yc, pal, logy=False):
    for name, df, mk, filled in sets:
        for m, g in df.groupby("mouse"):
            ax.scatter(g[xc], g[yc], s=34 if filled else 26, marker=mk, facecolors=pal[m] if filled else "none",
                       edgecolors=pal[m], linewidths=1.1, alpha=0.9, zorder=3 if filled else 2)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)


def fig_pi(cur, scr, tests, pal, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.2)); axs = axs.ravel()
    sets = [("screening", scr, "^", False), ("curated", cur, "o", True)]
    for ax, (y, exp) in zip(axs, PRIMARY_Y):
        yc = f"ref_{y}"
        _scatter(ax, sets, X_MAIN, yc, pal, logy=y in POSITIVE)
        ax.set_xlabel("r(reference, branch)"); ax.set_ylabel(LABEL.get(y, y))
        txt = []
        for sn in ("curated", "screening"):
            t = tests[(tests.set == sn) & (tests.y == yc) & (tests.x == X_MAIN)]
            if len(t):
                t = t.iloc[0]
                txt.append(f"{sn[:4]}: rho {t.spearman_rho:+.2f} [{t.rho_ci_lo:+.2f},{t.rho_ci_hi:+.2f}]"
                           f"\n   mixed b {t.mixed_beta:+.2f} q={t.get('q', np.nan):.2g} ({t.n_runs}r/{t.n_mice}m)")
        ax.set_title(f"{LABEL.get(y, y)}\nPI predicts: {dict(zip('+-?', ['higher', 'lower', 'n/a']))[exp]} when r is high",
                     fontsize=9)
        ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=7, family="monospace",
                bbox=dict(fc="white", ec="0.8", alpha=0.85))
    ax = axs[-1]; ax.axis("off")
    h = [Line2D([], [], marker="o", ls="", mfc=pal[m], mec=pal[m], label=m.replace("rbp4_", "")) for m in sorted(pal)]
    h += [Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="curated (hand masks)"),
          Line2D([], [], marker="^", ls="", mfc="none", mec="k", label="screening (automatic)")]
    ax.legend(handles=h, loc="center", fontsize=9, frameon=False, title="mouse / set")
    fig.suptitle("PI hypothesis: high-correlation runs = low baseline + big peaks?  (reference region; one dot per run; "
                 "rho = Spearman with mouse-bootstrap CI; b = standardized mixed-model slope, mouse random intercept)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_pi_sd_amplitude.{ext}", dpi=150)
    plt.close(fig)


def fig_baseline_peak(cur, scr, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    allr = pd.concat([cur, scr])[X_MAIN]
    norm = plt.Normalize(np.nanmin(allr), np.nanmax(allr))
    for ax, yc, yl in zip(axs, ["ref_event_amp_median", "ref_event_amp_median_adc", "ref_event_amp_F"],
                          ["event amplitude, dF/F (F0 incl. detector offset)",
                           "event amplitude, dF/(F - detector zero)", "event amplitude, raw dF (ADC units)"]):
        for df, mk, lw in ((scr, "^", 0.6), (cur, "o", 1.2)):
            sc = ax.scatter(df["ref_F0_adc"], df[yc], c=df[X_MAIN], cmap="viridis", norm=norm, marker=mk, s=40,
                            edgecolors="k", linewidths=lw)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.25)
        ax.set_xlabel("baseline F0 above detector zero (ADC, log)"); ax.set_ylabel(yl)
        for i, (nm, df) in enumerate((("curated", cur), ("screening", scr))):
            r, p = spearman(np.log10(df["ref_F0_adc"].values), np.log10(df[yc].where(df[yc] > 0).values))
            ax.text(0.02, 0.98 - 0.07 * i, f"{nm}: rho {r:+.2f} (p_runs {p:.2g}, {len(df)} runs)", transform=ax.transAxes,
                    va="top", fontsize=8)
    fig.colorbar(sc, ax=axs, label="r(reference, branch)", shrink=0.85)
    fig.suptitle("Baseline vs peak (reference region). o curated, ^ screening. Left: normalizing by an F0 that "
                 "includes the detector offset; middle: offset removed; right: no normalization", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_baseline_vs_peak.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_confound(cur, scr, pal, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    sets = [("screening", scr, "^", False), ("curated", cur, "o", True)]
    _scatter(axs[0], sets, "ref_snr", X_MAIN, pal); axs[0].set_xscale("log")
    axs[0].set_xlabel("reference SNR (event amplitude / noise SD)"); axs[0].set_ylabel("r(reference, branch)")
    axs[0].set_title("Is high r just high SNR?", fontsize=10)
    for d, mk in ((scr, "^"), (cur, "o")):
        axs[1].scatter(d[X_MAIN], d["r_soma_branch_corr"] - d[X_MAIN], marker=mk, c="k" if mk == "o" else "0.5", s=25)
    axs[1].axhline(0, c="0.6", lw=0.8); axs[1].set_xlabel("r(reference, branch)")
    axs[1].set_ylabel("noise-corrected r - raw r"); axs[1].set_title(
        "Attenuation by independent (voxel-level) noise\n(split-half reliability correction)", fontsize=10)
    _scatter(axs[2], sets, "branch_snr", X_MAIN, pal); axs[2].set_xscale("log")
    axs[2].set_xlabel("branch SNR (mean over branch regions)"); axs[2].set_ylabel("r(reference, branch)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_snr_confound.{ext}", dpi=150)
    plt.close(fig)


def fig_region(R, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if R.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, nm in zip(axs, ("curated", "screening")):
        s = R[R.set == nm]
        sc = ax.scatter(s["snr"], s["r_with_ref"], c=s["distance_um"], cmap="plasma", s=18)
        ax.set_xscale("log"); ax.set_xlabel("region SNR"); ax.set_title(f"{nm}: {len(s)} regions, "
                                                                          f"{s.behavior_base.nunique()} runs", fontsize=10)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("region r with reference"); fig.colorbar(sc, ax=axs, label="distance from soma (um)")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_region_level.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def log(what, result):
    p = PROJECT / "auto_pipeline" / "logs" / "pi_hypothesis_v7.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps({"time": pd.Timestamp.utcnow().isoformat(), "stage": "pi_hypothesis", "what": what,
                            "result": result}, default=str) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(PROJECT / "stats" / "trace_features_run.csv")
    reg = pd.read_csv(PROJECT / "stats" / "trace_features.csv")
    d["root_path"] = d["root"].map({"real": str(PROJECT), "auto": str(PROJECT / "auto_pipeline")})
    cur = d[d.set == "curated"].copy(); scr = d[d.set == "screening"].copy()
    scr_ind = scr[~scr.behavior_base.isin(cur.behavior_base)].copy()
    scr_unm = scr[scr["mark"].fillna("") == ""].copy()
    noisy = d[(d.set == "curated_other") & d.mark_reason.fillna("").str.contains("noisy")].copy()
    # screening_all (v1.1): every automatic-mirror run with metrics, whether or not it is in the
    # auto cohort (set-aside runs included). Secondary/descriptive: once run_marks are applied the
    # auto cohort shrinks to mostly the curated recordings, so this keeps an independent-ish replication.
    # Marks are never based on correlation, so including set-aside runs is not circular.
    scr_all = d[d.set.isin(["screening", "screening_other"])].copy()
    sets = {"curated": cur, "screening": scr, "screening_independent": scr_ind, "screening_unmarked": scr_unm,
            "screening_all": scr_all,
            "screening_all_independent": scr_all[~scr_all.behavior_base.isin(cur.behavior_base)].copy()}
    rows = []
    for sn, df in sets.items():
        for y, exp in PRIMARY_Y:
            rows.append(one_test(df, X_MAIN, f"ref_{y}", "primary", sn, exp))
        for y in SECONDARY_Y:
            rows.append(one_test(df, X_MAIN, f"ref_{y}", "secondary", sn))
        for y, exp in PRIMARY_Y:
            rows.append(one_test(df, X_MAIN, f"branch_{y}", "secondary", sn, exp))
            for xo in X_OTHER:
                rows.append(one_test(df, xo, f"ref_{y}", "secondary", sn, exp))
    T = pd.DataFrame(rows)
    T["q"] = np.nan; T["q_within"] = np.nan
    for (fam, sn), idx in T.groupby(["family", "set"]).groups.items():
        T.loc[idx, "q"] = bh(T.loc[idx, "mixed_p"].values)
        T.loc[idx, "q_within"] = bh(T.loc[idx, "p_within_mouse"].values)
    T.to_csv(OUT / "tests.csv", index=False)
    # region level
    Rs, rres = [], []
    for sn in ("curated", "screening"):
        R, rr = region_level(reg, sets[sn], sn); Rs.append(R); rres += rr
    R = pd.concat(Rs) if Rs else pd.DataFrame()
    RT = pd.DataFrame(rres)
    if len(RT):
        for sn, idx in RT.groupby("set").groups.items():
            RT.loc[idx, "q"] = bh(RT.loc[idx, "p"].values)
    RT.to_csv(OUT / "region_tests.csv", index=False); R.to_csv(OUT / "region_table.csv", index=False)
    # extra descriptives
    desc = {}
    for sn, df in (("curated", cur), ("screening", scr), ("screening_all", scr_all)):
        att = (df["r_soma_branch_corr"] - df[X_MAIN]).dropna()
        desc[sn] = {"n_runs": int(len(df)), "n_mice": int(df.mouse.nunique()), "n_sessions": int(df.session.nunique()),
                    "mice": sorted(df.mouse.unique()), "sessions_with_multiple_runs": int((df.groupby("session").size() > 1).sum()),
                    "noise_attenuation_r_corr_minus_r_median": float(att.median()), "noise_attenuation_max": float(att.max()),
                    "ref_reliability_median": float(df["ref_reliability"].median()),
                    "ref_dark_frac_of_F0_median": float(df["ref_dark_frac_of_F0"].median()),
                    "ref_F0_adc_median": float(df["ref_F0_adc"].median()),
                    "spearman_baseline_vs_amp_dff": spearman(np.log10(df.ref_F0_adc.values), np.log10(df.ref_event_amp_median.values))[0],
                    "spearman_baseline_vs_amp_adc": spearman(np.log10(df.ref_F0_adc.values), np.log10(df.ref_event_amp_median_adc.values))[0],
                    "spearman_baseline_vs_amp_rawF": spearman(np.log10(df.ref_F0_adc.values), np.log10(df.ref_event_amp_F.values))[0]}
    desc["daria_noisy_runs"] = noisy[["behavior_base", "mark_reason", X_MAIN, "mean_pairwise_r", "ref_snr", "ref_noise_sd",
                                      "ref_event_amp_median", "ref_F0_adc", "metrics_stale"]].to_dict("records")
    desc["daria_noisy_vs_curated_percentile"] = {
        r["behavior_base"]: {c: float((cur[c] < r[c]).mean()) for c in (X_MAIN, "ref_snr", "ref_noise_sd", "ref_event_amp_median")}
        for _, r in noisy.iterrows()}
    pal = _palette(set(d.mouse))
    fig_pi(cur, scr, T[T.family == "primary"], pal, OUT)
    fig_baseline_peak(cur, scr, OUT); fig_confound(cur, scr, pal, OUT); fig_region(R, OUT)
    res = {"version": __version__, "inputs": ["stats/trace_features_run.csv", "stats/trace_features.csv"],
           "x_main": X_MAIN, "primary_features": PRIMARY_Y, "descriptives": desc,
           "primary": T[T.family == "primary"].to_dict("records"), "region_level": rres,
           "notes": ["MixedLM: z-scored log10(y) for positive features ~ z(x) + (1|mouse); beta and Wald CI from REML, "
                     "p from an ML likelihood-ratio test (Wald SEs are unreliable with 5-8 mice; NaN CI = singular fit)",
                     "q = BH-FDR of the LRT p within (family, set); q_within = BH-FDR of the within-mouse permutation p",
                     "within-mouse rho: Spearman of mouse-centered ranks (mice with >=2 runs); permutation shuffles x within mouse", "rho CI = mouse-then-run cluster bootstrap, 2000 resamples",
                     "same-session runs may be the same cell"]}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: None if o is None else (
        float(o) if isinstance(o, (np.floating, np.integer)) else str(o))))
    # numbers table
    P = T[(T.family == "primary")]
    lines = ["| set | feature | n runs / mice | rho [mouse-boot CI] | within-mouse rho (perm q) | mixed beta [Wald CI] | LRT q | beta given SNR [CI] | partial rho given SNR | LOMO rho range |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, t in P.iterrows():
        lines.append(f"| {t.set} | {t.y} | {t.n_runs} / {t.n_mice} | {t.spearman_rho:+.2f} [{t.rho_ci_lo:+.2f}, {t.rho_ci_hi:+.2f}] | "
                     f"{t.rho_within_mouse:+.2f} ({t.q_within:.2g}) | {t.mixed_beta:+.2f} [{t.mixed_ci_lo:+.2f}, {t.mixed_ci_hi:+.2f}] | {t.q:.3g} | "
                     + (f"{t.mixed_beta_given_snr:+.2f} [{t.mixed_ci_lo_given_snr:+.2f}, {t.mixed_ci_hi_given_snr:+.2f}] | {t.partial_rho_given_snr:+.2f} | "
                        if pd.notna(t.get("mixed_beta_given_snr")) else "- | - | ")
                     + f"{t.lomo_rho_min:+.2f}..{t.lomo_rho_max:+.2f} |")
    (OUT / "numbers.md").write_text("\n".join(lines) + "\n")
    log("pi_hypothesis_run", f"tests.csv {len(T)} tests; primary curated q<0.05: "
        f"{int(((P.set=='curated') & (P.q<0.05)).sum())}/{int((P.set=='curated').sum())}; figures + results.json written")
    print((OUT / "numbers.md").read_text())
    print(json.dumps({k: v for k, v in desc.items() if k not in ("daria_noisy_runs",)}, indent=1, default=str))
    print(pd.DataFrame(desc["daria_noisy_runs"]).to_string())
    print(RT.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
