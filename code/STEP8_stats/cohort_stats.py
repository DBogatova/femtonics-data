#!/usr/bin/env python
"""cohort_stats.py - collect every run's metrics into one table and make the cohort figures.

Inputs : <run_dir>/<stem>_metrics.json (from run_metrics.py), mice.csv (injection dates)
Outputs: stats/cohort_metrics.csv                         one row per run
         stats/cohort_summary.txt                         tests, in words
         stats/fig_independence.{png,pdf}                 per run: soma-branch r, branch-independent
                                                          fraction, who leads, quiet vs active
         stats/fig_within_mouse.{png,pdf}                 each metric vs days post-injection,
                                                          one line per mouse
         stats/fig_expression.{png,pdf}                   coupling vs expression proxy (soma F)

Tests (scipy only; unit of inference = mouse):
  Every p-value is an exact one-sided Wilcoxon on per-mouse means (runs are nested in
  mice, so run-level tests are printed as descriptive only); every headline effect has a
  two-stage (mouse, then run) bootstrap 95% CI; the primary family is BH-FDR corrected
  together and written to stats/cohort_tests.csv. With k mice the smallest attainable p is
  0.5^k, and tests that cannot reach 0.05 are labelled UNDERPOWERED rather than 'n.s.'.
  Metrics files whose regions are gone or newer than them are excluded and listed.
  (The earlier statsmodels mixed model never ran - statsmodels is not in the venv.)
  Original hypotheses:
  H1 independence : per run, is r(soma,branch) < r(soma,trunk)? is frac_branch_independent > 0?
                    Sign test / Wilcoxon across runs; mixed model with mouse as random effect.
  H2 time/expression : r(soma,branch) ~ dpi (+ soma_f_raw), random intercept per mouse.
                    Reported with the honest caveat that dpi and mouse are partly confounded.
  Halo control : r_core vs r_full (paired). If coupling were halo, core << full.
"""
from __future__ import annotations
import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["FEMTO_ROOT"]).resolve() if os.environ.get("FEMTO_ROOT") else HERE.parents[1]
OUT = ROOT / "stats"
sys.path.insert(0, str(HERE.parents[1] / "code"))
from common.run_marks import is_set_aside, load_marks   # noqa: E402  (run_marks.csv: excluded / revisit)

METRICS = ["r_soma_branch", "r_soma_branch_corr", "r_soma_trunk_corr", "r_soma_branch_core", "r_soma_trunk", "r_trunk_branch",
           "frac_branch_independent", "frac_soma_independent", "frac_global", "branch_first_frac",
           "frac_branch_independent_null", "p_branch_coupled_vs_chance",
           "frac_soma_independent_null", "p_soma_coupled_vs_chance",
           "n_paired_events", "lag_soma_branch_frames", "lag_soma_branch_s", "r_soma_branch_quiet", "r_soma_branch_active",
           "soma_f_raw", "branch_f_raw", "soma_snr", "rate_soma_per_min", "rate_branch_per_min",
           "n_soma_events", "n_branch_events", "duration_s"]
MIN_EVENTS = 5          # per-run event fractions from fewer events than this are not tested
N_BOOT = 5000
_STALE: list[str] = []  # metrics files whose regions no longer exist (filled by collect)


def _is_current(f: str) -> bool:
    """A metrics file counts only if the regions it was computed from still exist and are
    not newer than it. Orphaned files (regions deleted / regenerated) used to be counted."""
    seg = Path(f.replace("_metrics.json", "_segments_final.tif"))
    if not seg.exists():
        return False
    return Path(f).stat().st_mtime >= seg.stat().st_mtime


# ---------------------------------------------------------------- inference helpers
def bh(p):
    """Benjamini-Hochberg q-values (NaN-safe)."""
    p = np.asarray(p, float); q = np.full(p.shape, np.nan); ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]; o = np.argsort(pv); m = len(pv)
        qq = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]
        r = np.empty(m); r[o] = np.minimum(qq, 1.0); q[ok] = r
    return q


def cluster_boot(df: pd.DataFrame, stat, B: int = N_BOOT, seed: int = 0):
    """95% percentile CI by two-stage bootstrap: resample mice, then runs within each mouse.
    stat(df) -> float. With few mice the CI is wide by construction - that is the point."""
    rng = np.random.default_rng(seed); groups = [g for _, g in df.groupby("mouse")]
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(B):
        parts = []
        for gi in rng.integers(0, len(groups), len(groups)):
            g = groups[gi]; parts.append(g.iloc[rng.integers(0, len(g), len(g))])
        v = stat(pd.concat(parts))
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def mouse_mean(df: pd.DataFrame, col: str) -> float:
    """Each mouse weighted equally (mean of per-mouse means)."""
    return float(df.groupby("mouse")[col].mean().mean()) if len(df) else float("nan")


def mouse_test(df: pd.DataFrame, col: str, alternative: str = "greater"):
    """Exact one-sample Wilcoxon signed-rank on the per-mouse means of `col` (vs 0).
    Returns (p, n_mice, min_attainable_p)."""
    from scipy import stats
    mm = df.groupby("mouse")[col].mean().dropna(); k = len(mm)
    if k < 2:
        return float("nan"), k, float("nan")
    try:
        p = float(stats.wilcoxon(mm.values, alternative=alternative, method="exact").pvalue)
    except (TypeError, ValueError):
        p = float(stats.wilcoxon(mm.values, alternative=alternative).pvalue)
    return p, k, 0.5 ** k


def ns(df: pd.DataFrame) -> str:
    return f"n = {len(df)} runs, {df.mouse.nunique()} mice"


def collect() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(ROOT / "rbp4_*/**/*_metrics.json"), recursive=True):
        if "/old/" in f:
            continue
        m = json.load(open(f))
        if "note" in m or is_set_aside(m.get("behavior_base", "")):
            continue
        if not _is_current(f):
            _STALE.append(str(Path(f).relative_to(ROOT)))
            continue
        row = {k: m.get(k) for k in ["behavior_base", "mouse", "date", "rank", "frame_rate_hz", "T",
                                     "imaging_quality", "n_regions", "reference", "reference_region"] + METRICS}
        row["reference"] = row.get("reference") or "soma"
        row["n_branch_regions"] = sum(1 for v in m["regions"].values() if v["compartment"] == "branch")
        row["metrics_file"] = str(Path(f).relative_to(ROOT))
        # Mark imaging-only: no behavior_state keys present
        row["has_behavior"] = m.get("r_soma_branch_quiet") is not None or m.get("r_soma_branch_active") is not None
        rows.append(row)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    mice = pd.read_csv(ROOT / "mice.csv")
    # Parse injection_date carefully — some rows may be blank
    mice["injection_date"] = pd.to_datetime(mice["injection_date"], errors="coerce")
    d = d.merge(mice[["mouse", "injection_date", "line", "virus"]], on="mouse", how="left")
    d["date_dt"] = pd.to_datetime(d["date"], format="%m-%d-%Y", errors="coerce")
    d["dpi"] = (d["date_dt"] - d["injection_date"]).dt.days
    # Build short label — handle missing behavior_base for imaging-only runs
    run_num = d["behavior_base"].str.extract(r"Run(\d+)")[0]
    run_num = run_num.fillna("?")
    d["short"] = d["mouse"].str.replace("rbp4_", "", regex=False) + " " + d["date"].str[:5] + " r" + \
        run_num.astype(str) + \
        np.where(d["reference"] == "proximal_trunk", " (no soma)", "") + \
        np.where(~d["has_behavior"], " [no beh]", "")
    return d.sort_values(["mouse", "date_dt", "rank"]).reset_index(drop=True)


def collect_distance() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(ROOT / "rbp4_*/**/*_metrics.json"), recursive=True):
        if "/old/" in f:
            continue
        m = json.load(open(f))
        if is_set_aside(m.get("behavior_base", "")) or not _is_current(f):
            continue
        for c in m.get("coupling_by_distance", []):
            rows.append({"behavior_base": m["behavior_base"], "mouse": m["mouse"], "reference": m.get("reference", "soma"), **c})
    return pd.DataFrame(rows)


def fig_distance(dd: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    mk = {"trunk": "o", "branch": "^"}
    _bases = sorted(dd.behavior_base.unique())
    _cmap = plt.cm.tab20 if len(_bases) > 10 else plt.cm.tab10
    colors = {b: _cmap(i / max(1, len(_bases) - 1)) for i, b in enumerate(_bases)}
    for b, g in dd.groupby("behavior_base"):
        g = g.sort_values("distance_um")
        ax.plot(g.distance_um, g.r_with_soma_corr, "-", color=colors[b], lw=1, alpha=0.6)
        hollow = g.reference.iloc[0] == "proximal_trunk"
        for _, r in g.iterrows():
            ax.scatter(r.distance_um, r.r_with_soma_corr, marker=mk.get(r.compartment, "s"), s=40, zorder=3,
                       facecolor="none" if hollow else colors[b], edgecolor=colors[b], linewidth=1.4)
        ax.plot([], [], "--" if hollow else "-", color=colors[b],
                label=b.replace("rbp4_", "").replace("_phpeb", "") + (" (vs proximal trunk)" if hollow else ""))
    ax.scatter([], [], marker="o", color="0.4", label="trunk"); ax.scatter([], [], marker="^", color="0.4", label="branch")
    ax.set_xlabel("distance from the reference region along the dendrite (um)"); ax.set_ylabel("r with reference (noise-corrected)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Coupling vs distance (filled: soma reference; hollow: proximal trunk, no soma in scan)", fontsize=9, loc="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=170); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def tests(d: pd.DataFrame, dd: pd.DataFrame | None = None):
    """Cohort tests. Unit of inference is the MOUSE: runs are nested in mice, so every
    p-value below comes from per-mouse means (exact Wilcoxon), every CI from a two-stage
    (mouse, then run) bootstrap, and the primary family is BH-FDR corrected together.
    Run-level and pooled-event numbers are printed as descriptive only."""
    from scipy import stats
    L = []; fam = []          # fam: rows of the primary test family
    n = len(d); n_beh = int(d["has_behavior"].sum()) if "has_behavior" in d.columns else n
    n_io = n - n_beh
    L.append(f"{n} run(s), {d.mouse.nunique()} mouse/mice: " + ", ".join(f"{m} ({k})" for m, k in d.mouse.value_counts().items()))
    if n_io:
        L.append(f"  ({n_beh} with behavior, {n_io} imaging-only — imaging-only runs included in imaging analyses, excluded from behavior analyses)")
    if _STALE:
        L.append(f"  NOT counted: {len(_STALE)} metrics file(s) whose regions (_segments_final.tif) are missing or newer "
                 f"than the metrics: " + ", ".join(_STALE))
    nsr = int((d.reference == "soma").sum()); nt = n - nsr
    L.append(f"reference region: soma in {nsr} run(s), most proximal trunk in {nt} (soma below the scanned tube)."
             + ("  'soma' below means 'reference region'." if nt else ""))
    fr = d.frame_rate_hz.dropna()
    L.append(f"volume rate per run (1000/TStepInMs): {fr.min():.2f}..{fr.max():.2f} Hz; event window is in seconds, "
             f"converted per run" + (f"; {int(d.frame_rate_hz.isna().sum())} run(s) with NO rate -> rate metrics NaN" if d.frame_rate_hz.isna().any() else ""))
    L.append("Inference: unit = mouse (per-mouse means, exact one-sided Wilcoxon; with k mice the smallest attainable "
             "p is 0.5^k). CI = 95% two-stage bootstrap (mice, then runs). q = Benjamini-Hochberg over the primary family.")
    L.append("Run-level and pooled-event numbers are descriptive (runs within a mouse are not independent).")
    L.append("")

    def add(key, label, df, col, alternative, est_fmt="{:+.3f}"):
        p, k, pmin = mouse_test(df, col, alternative)
        est = mouse_mean(df, col); ci = cluster_boot(df, lambda x: mouse_mean(x, col))
        fam.append({"test": key, "label": label, "estimate": est, "ci_lo": ci[0], "ci_hi": ci[1], "p_mouse": p,
                    "n_runs": len(df), "n_mice": k, "min_attainable_p": pmin})

    L.append("H1  Soma-dendrite independence")
    x = d["r_soma_branch"].dropna(); y = d["r_soma_trunk"].dropna()
    L.append(f"  r(soma,branch) mean {x.mean():.2f} [{x.min():.2f}..{x.max():.2f}] ({ns(d.dropna(subset=['r_soma_branch']))})   "
             f"r(soma,trunk) mean {y.mean():.2f} ({ns(d.dropna(subset=['r_soma_trunk']))})")
    xc = d["r_soma_branch_corr"].dropna()
    if len(xc):
        L.append(f"  noise-corrected r(soma,branch) mean {xc.mean():.2f} [{xc.min():.2f}..{xc.max():.2f}] (removes region-size effects)")
    both = d.dropna(subset=["r_soma_branch", "r_soma_trunk"]).copy()
    if len(both) >= 2:
        both["d_trunk_minus_branch"] = both.r_soma_trunk - both.r_soma_branch
        w = stats.wilcoxon(both["r_soma_trunk"], both["r_soma_branch"], alternative="greater")
        L.append(f"  [descriptive] branch less coupled than trunk in {int((both.d_trunk_minus_branch > 0).sum())}/{len(both)} runs, "
                 f"run-level Wilcoxon p={w.pvalue:.3g} ({ns(both)}; treats runs as independent)")
        add("H1a_trunk_gt_branch", "r(soma,trunk) - r(soma,branch) > 0", both, "d_trunk_minus_branch", "greater")
    if nt and nsr:
        xs_ = d[d.reference == "soma"]["r_soma_branch"].dropna(); xt_ = d[d.reference != "soma"]["r_soma_branch"].dropna()
        L.append(f"  soma-referenced runs: r {xs_.mean():.2f} (n={len(xs_)});  proximal-trunk-referenced: r {xt_.mean():.2f} (n={len(xt_)})")
    ev = d[(d.n_branch_events >= MIN_EVENTS) & (d.n_soma_events >= MIN_EVENTS)].dropna(
        subset=["frac_branch_independent", "frac_branch_independent_null"]).copy()
    n_drop = int(d.frac_branch_independent.notna().sum() - len(ev))
    if len(ev):
        ev["coupling_vs_chance"] = ev.frac_branch_independent_null - ev.frac_branch_independent
        ci = cluster_boot(ev, lambda x: mouse_mean(x, "frac_branch_independent"))
        L.append(f"  branch events with no soma event within the window: {100*mouse_mean(ev, 'frac_branch_independent'):.0f}% "
                 f"[95% CI {100*ci[0]:.0f}..{100*ci[1]:.0f}%] vs {100*mouse_mean(ev, 'frac_branch_independent_null'):.0f}% expected "
                 f"by chance (circular-shift null) ({ns(ev)}; {n_drop} run(s) with <{MIN_EVENTS} soma or branch events not tested)")
        sig = int((ev.p_branch_coupled_vs_chance < 0.05).sum())
        L.append(f"  [descriptive] per run, branch events coupled to soma beyond chance (p<0.05, uncorrected) in {sig}/{len(ev)} runs")
        add("H1b_branch_coupled_beyond_chance", "chance - observed branch-independent fraction > 0", ev, "coupling_vs_chance", "greater")
    fs = d["frac_soma_independent"].dropna()
    L.append(f"  soma events with no branch event: mean {100*fs.mean():.0f}% ({ns(d.dropna(subset=['frac_soma_independent']))}, descriptive)")
    bf = d.dropna(subset=["branch_first_frac"])
    bf = bf[bf.n_paired_events >= 1].copy()
    if len(bf):
        tot = int(bf.n_paired_events.sum()); first = int(round((bf.branch_first_frac * bf.n_paired_events).sum()))
        b = stats.binomtest(first, tot, 0.5, alternative="greater") if tot else None
        L.append(f"  [descriptive] of paired soma+branch events, branch peaks first in {first}/{tot}"
                 + (f" (pooled binomial p={b.pvalue:.3g}; treats {tot} events from {len(bf)} runs / {bf.mouse.nunique()} mice as independent)" if b else ""))
        bf["first_n"] = bf.branch_first_frac * bf.n_paired_events
        mf = bf.groupby("mouse").apply(lambda g: g.first_n.sum() / g.n_paired_events.sum(), include_groups=False)
        L.append("  per mouse, fraction branch-first (events pooled within mouse): "
                 + ", ".join(f"{m.replace('rbp4_', '')} {v:.2f} (n={int(bf[bf.mouse == m].n_paired_events.sum())})" for m, v in mf.items()))
        bm = pd.DataFrame({"mouse": mf.index, "d_branch_first": mf.values - 0.5})
        p, k, pmin = mouse_test(bm, "d_branch_first", "greater")
        ci = cluster_boot(bf, lambda x: x.groupby("mouse").apply(
            lambda g: g.first_n.sum() / g.n_paired_events.sum(), include_groups=False).mean())
        fam.append({"test": "H1c_branch_leads", "label": "fraction branch-first > 0.5", "estimate": float(mf.mean()),
                    "ci_lo": ci[0], "ci_hi": ci[1], "p_mouse": p, "n_runs": len(bf), "n_mice": k, "min_attainable_p": pmin})
    L.append("")
    L.append("Halo control")
    hc = d.dropna(subset=["r_soma_branch", "r_soma_branch_core"]).copy()
    if len(hc):
        hc["halo_drop"] = hc.r_soma_branch - hc.r_soma_branch_core
        ci = cluster_boot(hc, lambda x: mouse_mean(x, "halo_drop"))
        L.append(f"  r full {hc.r_soma_branch.mean():.2f} vs core {hc.r_soma_branch_core.mean():.2f} "
                 f"(mean drop {mouse_mean(hc, 'halo_drop'):+.3f}, 95% CI {ci[0]:+.3f}..{ci[1]:+.3f}; {ns(hc)}); "
                 "a large drop would mean the coupling was scattered light")
    L.append("")
    L.append("Behavior state")
    bs = d.dropna(subset=["r_soma_branch_quiet", "r_soma_branch_active"]).copy()
    if len(bs):
        bs["d_active_minus_quiet"] = bs.r_soma_branch_active - bs.r_soma_branch_quiet
        L.append(f"  r(soma,branch) quiet {bs.r_soma_branch_quiet.mean():.2f} vs active {bs.r_soma_branch_active.mean():.2f} "
                 f"({int((bs.d_active_minus_quiet > 0).sum())}/{len(bs)} runs more coupled when active; {ns(bs)})")
        add("BS_active_vs_quiet", "r active - r quiet != 0", bs, "d_active_minus_quiet", "two-sided")
    L.append("")

    L.append("H2  Coupling vs time since injection / expression")
    dp = d.dropna(subset=["dpi", "r_soma_branch"])
    L.append(f"  days post-injection known for {ns(dp)} (mice.csv); range "
             f"{int(dp.dpi.min()) if len(dp) else '?'}..{int(dp.dpi.max()) if len(dp) else '?'}; "
             f"mice with >1 session: {[m for m, g in dp.groupby('mouse') if g.dpi.nunique() > 1]}")
    if len(dp) >= 4 and dp.dpi.nunique() >= 3:
        rho = stats.spearmanr(dp.dpi, dp.r_soma_branch)
        L.append(f"  [descriptive] pooled Spearman over runs: rho={rho.statistic:+.2f} p={rho.pvalue:.3g} ({ns(dp)}; runs not independent)")
        ses = dp.groupby(["mouse", "dpi"], as_index=False).r_soma_branch.mean()
        rs = stats.spearmanr(ses.dpi, ses.r_soma_branch) if len(ses) >= 4 else None
        ci = cluster_boot(dp, lambda x: stats.spearmanr(x.dpi, x.r_soma_branch).statistic if x.dpi.nunique() > 2 else np.nan)
        # within-mouse slope: removes the between-mouse confound; only mice with >1 session contribute
        wm = [np.polyfit(g.dpi, g.r_soma_branch, 1)[0] for _, g in dp.groupby("mouse") if g.dpi.nunique() > 1]
        if rs is not None:
            L.append(f"  session-level Spearman (one point per mouse x session): rho={rs.statistic:+.2f} "
                     f"(n = {len(ses)} sessions, {ses.mouse.nunique()} mice), mouse-bootstrap 95% CI of run-level rho {ci[0]:+.2f}..{ci[1]:+.2f}")
            fam.append({"test": "H2_dpi", "label": "Spearman r(soma,branch) ~ dpi over sessions (sessions nested in mice; p is session-level)", "estimate": float(rs.statistic),
                        "ci_lo": ci[0], "ci_hi": ci[1], "p_mouse": float(rs.pvalue), "n_runs": len(dp),
                        "n_mice": int(ses.mouse.nunique()), "min_attainable_p": float("nan")})
        L.append(f"  within-mouse slope r per 10 days: " + (", ".join(f"{10*s:+.3f}" for s in wm) if wm else "none")
                 + f" ({len(wm)} mice with >1 session) - the only part of the trend not confounded with mouse identity")
    else:
        L.append("  too few runs / sessions yet for a time trend")
    L.append("")

    if dd is not None and not dd.empty:
        L.append("Coupling vs distance from soma (noise-corrected, all regions)")
        rho = stats.spearmanr(dd.distance_um, dd.r_with_soma_corr) if len(dd) >= 4 else None
        sl = []
        for b, g in dd.dropna(subset=["distance_um", "r_with_soma_corr"]).groupby("behavior_base"):
            if len(g) >= 2 and g.distance_um.nunique() >= 2:
                sl.append({"behavior_base": b, "mouse": g.mouse.iloc[0], "slope": np.polyfit(g.distance_um, g.r_with_soma_corr, 1)[0] * 100})
        sl = pd.DataFrame(sl)
        if rho:
            L.append(f"  [descriptive] pooled Spearman rho={rho.statistic:+.2f} p={rho.pvalue:.3g} over {len(dd)} regions "
                     f"from {dd.behavior_base.nunique()} runs / {dd.mouse.nunique()} mice (regions within a cell are not independent)")
        if len(sl):
            L.append(f"  per-cell slope: {', '.join(f'{x:+.2f}' for x in sl.slope)} r per 100 um ({ns(sl)}; "
                     f"{int((sl.slope < 0).sum())}/{len(sl)} negative)")
            add("DIST_slope_negative", "per-cell slope of r vs distance < 0", sl, "slope", "less")
        L.append("")

    # ---- primary family, FDR-corrected together
    tab = pd.DataFrame(fam)
    if len(tab):
        tab["q_bh"] = bh(tab.p_mouse.values)
        def verdict(r):
            if not np.isfinite(r.p_mouse):
                return "not testable"
            if r.q_bh < 0.05:
                return "significant (q<0.05)"
            if np.isfinite(r.min_attainable_p) and r.min_attainable_p > 0.05:
                return f"UNDERPOWERED: {int(r.n_mice)} mice cannot reach p<0.05 at all"
            return "not significant after FDR"
        tab["verdict"] = tab.apply(verdict, axis=1)
        L.append(f"PRIMARY TEST FAMILY ({len(tab)} tests, BH-FDR across all of them)")
        for _, r in tab.iterrows():
            L.append(f"  {r.test:34s} est {r.estimate:+.3f} [95% CI {r.ci_lo:+.3f}..{r.ci_hi:+.3f}]  p={r.p_mouse:.3g} q={r.q_bh:.3g}  "
                     f"(n = {int(r.n_runs)} runs, {int(r.n_mice)} mice" + (f", min p {r.min_attainable_p:.3g}" if np.isfinite(r.min_attainable_p) else "")
                     + f")  -> {r.verdict}")
            L.append(f"  {'':34s} {r.label}")
        L.append("")

    mk = load_marks()
    if mk:
        L.append(f"Runs set aside by you (run_marks.csv), not in any statistic: {len(mk)}")
        for k, r in sorted(mk.items()):
            L.append(f"  {r['mark']:8s} {k}" + (f"  - {r.get('reason')}" if r.get("reason") else ""))
        L.append("")
    L.append("Behavior (whole cell, cross-correlation within +-5 s, circular-shift null; q = BH across all runs x behaviors)")
    brow = []
    for f in sorted(glob.glob(str(ROOT / "rbp4_*/**/*_behavior_coupling.json"), recursive=True)):
        if "/old/" in f:
            continue
        j = json.load(open(f)); wc = j.get("regions", {}).get("whole cell")
        if is_set_aside(j.get("behavior_base", "")) or not wc:
            continue
        brow.append((j, wc))
    if brow:
        allp = [wc[b]["p_peak"] for j, wc in brow for b in j["behaviors"] if b in wc]
        allq = iter(bh(allp))
        n_sig_raw = sum(p < 0.05 for p in allp)
        for j, wc in brow:
            parts = []
            for b in j["behaviors"]:
                if b in wc:
                    q = next(allq)
                    parts.append(f"{b} r0 {wc[b]['r_lag0']:+.2f}, peak {wc[b]['r_peak']:+.2f} at {wc[b]['lag_peak_s']:+.1f} s "
                                 f"(p={wc[b]['p_peak']:.3f}, q={q:.3f})")
            L.append(f"  {j['behavior_base']}: " + "; ".join(parts))
            sd = j.get("state_dependence")
            if sd and sd.get("branch_only_events", 0) >= 5:
                L.append(f"      branch-only events in active frames: {sd['branch_only_in_active']}/{sd['branch_only_events']} "
                         f"(expected {100 * sd['frac_frames_active']:.0f}% by chance, p={sd['p_branch_only_state']:.2f}, uncorrected)")
        allq_arr = bh(allp)
        L.append(f"  {n_sig_raw}/{len(allp)} run x behavior pairs p<0.05 uncorrected; {int(np.sum(allq_arr < 0.05))} survive BH q<0.05 "
                 f"({len(brow)} runs, {len({j['behavior_base'].rsplit('_', 2)[0] for j, _ in brow})} mouse ids). "
                 "Null p floor is set by the number of shifts (500 -> 0.002).")
    else:
        L.append("  (run behavior_coupling.py --all)")
    L.append("")
    L.append("Caveats: expression is proxied by soma raw F, which also depends on laser power and depth; "
             "compare within a mouse at fixed settings. A rise in coupling with expression is consistent with "
             "sensor filling / buffering but does not by itself demonstrate the mechanism. 'runs' are counted, not "
             "cells: whether two runs of one session image the same cell is not recorded in the metrics.")
    return "\n".join(L), (tab if len(tab) else None)


def fig_independence(d: pd.DataFrame, out: Path):
    n = len(d); fig, ax = plt.subplots(2, 2, figsize=(max(8, 1.1 * n + 4), 7.5))
    xs = np.arange(n); colors = {m: c for m, c in zip(sorted(d.mouse.unique()), plt.cm.tab10.colors)}
    col = [colors[m] for m in d.mouse]
    a = ax[0, 0]; a.bar(xs - 0.2, d.r_soma_trunk.fillna(0), 0.4, color="0.6", label="soma-trunk")
    a.bar(xs + 0.2, d.r_soma_branch, 0.4, color=col, label="soma-branch (color = mouse)")
    a.scatter(xs + 0.2, d.r_soma_branch_core, marker="_", color="k", s=120, label="soma-branch, core voxels only", zorder=3)
    a.set_ylim(0, 1); a.set_ylabel("dF/F correlation"); a.set_title("Coupling: branch vs trunk", loc="left"); a.legend(fontsize=7, frameon=False)
    a = ax[0, 1]; a.bar(xs - 0.2, 100 * d.frac_branch_independent, 0.4, color=col, label="branch events without soma")
    a.bar(xs + 0.2, 100 * d.frac_soma_independent, 0.4, color="0.6", label="soma events without branch")
    a.set_ylabel("% of events"); a.set_ylim(0, 100); a.set_title("Independent events", loc="left"); a.legend(fontsize=7, frameon=False)
    a = ax[1, 0]; a.bar(xs, 100 * d.branch_first_frac, color=col); a.axhline(50, color="k", lw=0.8, ls="--")
    for x, (f, k) in enumerate(zip(d.branch_first_frac, d.n_paired_events)):
        a.text(x, 100 * (f if f == f else 0) + 2, f"n={int(k)}", ha="center", fontsize=7)
    a.set_ylim(0, 110); a.set_ylabel("% paired events branch first"); a.set_title("Who leads (paired events)", loc="left")
    a = ax[1, 1]
    for x, (q, ac, c) in enumerate(zip(d.r_soma_branch_quiet, d.r_soma_branch_active, col)):
        if q == q and ac == ac:
            a.plot([x - 0.15, x + 0.15], [q, ac], "-o", color=c, ms=4)
    a.set_xticks([]); a.set_ylim(0, 1); a.set_ylabel("r(soma, branch)"); a.set_title("Quiet -> active (per run)", loc="left")
    for a in ax.ravel()[:3]:
        a.set_xticks(xs); a.set_xticklabels(d.short, rotation=60, ha="right", fontsize=7)
    for a in ax.ravel():
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.suptitle(f"Soma-dendrite independence, {n} runs, {d.mouse.nunique()} mice", fontsize=12)
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=170); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def fig_within_mouse(d: pd.DataFrame, out: Path):
    metrics = [("r_soma_branch", "r(soma, branch)"), ("frac_branch_independent", "branch events without soma"),
               ("branch_first_frac", "fraction branch-first"), ("soma_f_raw", "soma raw F (expression proxy)")]
    fig, ax = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8))
    colors = {m: c for m, c in zip(sorted(d.mouse.unique()), plt.cm.tab10.colors)}
    for a, (k, lab) in zip(ax, metrics):
        for m, g in d.dropna(subset=["dpi", k]).groupby("mouse"):
            g = g.sort_values("dpi"); a.plot(g.dpi, g[k], "o-", color=colors[m], label=m.replace("rbp4_", ""), ms=5, alpha=0.85)
        a.set_xlabel("days post-injection"); a.set_title(lab, loc="left", fontsize=10)
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    ax[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Within-mouse trends (one line per mouse; one point per run)", fontsize=11)
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=170); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def fig_expression(d: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    colors = {m: c for m, c in zip(sorted(d.mouse.unique()), plt.cm.tab10.colors)}
    for m, g in d.groupby("mouse"):
        ax[0].scatter(g.soma_f_raw, g.r_soma_branch, color=colors[m], label=m.replace("rbp4_", ""), s=30)
        ax[1].scatter(g.soma_f_raw, g.frac_branch_independent, color=colors[m], s=30)
    ax[0].set_xlabel("soma raw F"); ax[0].set_ylabel("r(soma, branch)"); ax[0].legend(fontsize=7, frameon=False)
    ax[1].set_xlabel("soma raw F"); ax[1].set_ylabel("branch events without soma")
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.suptitle("Coupling vs expression proxy (brighter soma = more sensor; compare within a mouse)", fontsize=10)
    fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=170); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    d = collect()
    OUT.mkdir(exist_ok=True)
    if d.empty:
        print("no metrics yet - run run_metrics.py --all"); return 1
    d.to_csv(OUT / "cohort_metrics.csv", index=False)
    dd = collect_distance()
    if not dd.empty:
        dd.to_csv(OUT / "coupling_by_distance.csv", index=False)
        fig_distance(dd, OUT / "fig_coupling_vs_distance")
    txt, tab = tests(d, dd)
    if tab is not None:
        tab.to_csv(OUT / "cohort_tests.csv", index=False)
    (OUT / "cohort_summary.txt").write_text(txt + "\n"); print(txt)
    fig_independence(d, OUT / "fig_independence"); fig_within_mouse(d, OUT / "fig_within_mouse"); fig_expression(d, OUT / "fig_expression")
    print(f"\nwrote stats/cohort_metrics.csv ({len(d)} runs), cohort_summary.txt, fig_independence, fig_within_mouse, fig_expression, fig_coupling_vs_distance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
