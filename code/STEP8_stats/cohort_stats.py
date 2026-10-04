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

Tests (statsmodels if installed, else plain):
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
           "n_paired_events", "lag_soma_branch_frames", "r_soma_branch_quiet", "r_soma_branch_active",
           "soma_f_raw", "branch_f_raw", "soma_snr", "rate_soma_per_min", "rate_branch_per_min",
           "n_soma_events", "n_branch_events"]


def collect() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(ROOT / "rbp4_*/**/*_metrics.json"), recursive=True):
        if "/old/" in f:
            continue
        m = json.load(open(f))
        if "note" in m or is_set_aside(m.get("behavior_base", "")):
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
        if is_set_aside(m.get("behavior_base", "")):
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


def tests(d: pd.DataFrame) -> str:
    from scipy import stats
    L = []
    n = len(d); n_beh = int(d["has_behavior"].sum()) if "has_behavior" in d.columns else n
    n_io = n - n_beh
    L.append(f"{n} run(s), {d.mouse.nunique()} mouse/mice: " + ", ".join(f"{m} ({k})" for m, k in d.mouse.value_counts().items()))
    if n_io:
        L.append(f"  ({n_beh} with behavior, {n_io} imaging-only — imaging-only runs included in imaging analyses, excluded from behavior analyses)")
    ns = int((d.reference == "soma").sum()); nt = n - ns
    L.append(f"reference region: soma in {ns} run(s), most proximal trunk in {nt} (soma below the scanned tube)."
             + ("  'soma' below means 'reference region'; tests are reported for all runs and for soma-referenced runs alone." if nt else ""))
    L.append("")
    L.append("H1  Soma-dendrite independence")
    x = d["r_soma_branch"].dropna(); y = d["r_soma_trunk"].dropna()
    L.append(f"  r(soma,branch) mean {x.mean():.2f} [{x.min():.2f}..{x.max():.2f}]   r(soma,trunk) mean {y.mean():.2f}")
    xc = d["r_soma_branch_corr"].dropna()
    if len(xc):
        L.append(f"  noise-corrected r(soma,branch) mean {xc.mean():.2f} [{xc.min():.2f}..{xc.max():.2f}] (removes region-size effects)")
    both = d.dropna(subset=["r_soma_branch", "r_soma_trunk"])
    if len(both) >= 2:
        w = stats.wilcoxon(both["r_soma_trunk"], both["r_soma_branch"], alternative="greater")
        L.append(f"  branch less coupled than trunk: {int((both.r_soma_trunk > both.r_soma_branch).sum())}/{len(both)} runs, "
                 f"Wilcoxon p={w.pvalue:.3g}" + ("  (n small - descriptive)" if len(both) < 6 else ""))
    if nt and ns:
        xs_ = d[d.reference == "soma"]["r_soma_branch"].dropna(); xt_ = d[d.reference != "soma"]["r_soma_branch"].dropna()
        L.append(f"  soma-referenced runs: r {xs_.mean():.2f} (n={len(xs_)});  proximal-trunk-referenced: r {xt_.mean():.2f} (n={len(xt_)})")
    fi = d["frac_branch_independent"].dropna()
    L.append(f"  branch events with no soma event within the window: mean {100*fi.mean():.0f}% [{100*fi.min():.0f}..{100*fi.max():.0f}%]")
    fs = d["frac_soma_independent"].dropna()
    L.append(f"  soma events with no branch event: mean {100*fs.mean():.0f}%")
    bf = d.dropna(subset=["branch_first_frac"])
    if len(bf):
        tot = int(bf.n_paired_events.sum()); first = int((bf.branch_first_frac * bf.n_paired_events).sum())
        b = stats.binomtest(first, tot, 0.5, alternative="greater") if tot else None
        L.append(f"  of paired soma+branch events, branch peaks first in {first}/{tot}" + (f" (binomial p={b.pvalue:.3g})" if b else ""))
    L.append("")
    L.append("Halo control")
    hc = d.dropna(subset=["r_soma_branch", "r_soma_branch_core"])
    if len(hc):
        L.append(f"  r full {hc.r_soma_branch.mean():.2f} vs core {hc.r_soma_branch_core.mean():.2f} "
                 f"(mean drop {(hc.r_soma_branch - hc.r_soma_branch_core).mean():+.2f}); a large drop would mean the coupling was scattered light")
    L.append("")
    L.append("Behavior state")
    bs = d.dropna(subset=["r_soma_branch_quiet", "r_soma_branch_active"])
    if len(bs):
        L.append(f"  r(soma,branch) quiet {bs.r_soma_branch_quiet.mean():.2f} vs active {bs.r_soma_branch_active.mean():.2f} "
                 f"({int((bs.r_soma_branch_active > bs.r_soma_branch_quiet).sum())}/{len(bs)} runs more coupled when active)")
    L.append("")
    mk = load_marks()
    if mk:
        L.append(f"Runs set aside by you (run_marks.csv), not in any statistic: {len(mk)}")
        for k, r in sorted(mk.items()):
            L.append(f"  {r['mark']:8s} {k}" + (f"  - {r.get('reason')}" if r.get("reason") else ""))
        L.append("")
    L.append("Behavior (whole cell, cross-correlation within +-5 s, circular-shift null)")
    nb = 0
    for f in sorted(glob.glob(str(ROOT / "rbp4_*/**/*_behavior_coupling.json"), recursive=True)):
        if "/old/" in f:
            continue
        j = json.load(open(f)); wc = j.get("regions", {}).get("whole cell")
        if is_set_aside(j.get("behavior_base", "")):
            continue
        if not wc:
            continue
        nb += 1
        L.append(f"  {j['behavior_base']}: " + "; ".join(
            f"{b} r0 {wc[b]['r_lag0']:+.2f}, peak {wc[b]['r_peak']:+.2f} at {wc[b]['lag_peak_s']:+.1f} s (p={wc[b]['p_peak']:.3f})"
            for b in j["behaviors"] if b in wc))
        sd = j.get("state_dependence")
        if sd and sd.get("branch_only_events", 0) >= 5:
            L.append(f"      branch-only events in active frames: {sd['branch_only_in_active']}/{sd['branch_only_events']} "
                     f"(expected {100 * sd['frac_frames_active']:.0f}% by chance, p={sd['p_branch_only_state']:.2f})")
    if not nb:
        L.append("  (run behavior_coupling.py --all)")
    L.append("")
    L.append("H2  Coupling vs time since injection / expression")
    dd = d.dropna(subset=["dpi", "r_soma_branch"])
    L.append(f"  days post-injection: {int(dd.dpi.min()) if len(dd) else '?'}..{int(dd.dpi.max()) if len(dd) else '?'}; "
             f"mice with >1 session: {[m for m, g in dd.groupby('mouse') if g.dpi.nunique() > 1]}")
    if len(dd) >= 4 and dd.dpi.nunique() >= 3:
        rho = stats.spearmanr(dd.dpi, dd.r_soma_branch)
        L.append(f"  Spearman r(soma,branch) vs dpi: rho={rho.statistic:+.2f} p={rho.pvalue:.3g} (pooled; mouse and dpi partly confounded)")
        try:
            import statsmodels.formula.api as smf
            md = smf.mixedlm("r_soma_branch ~ dpi", dd, groups=dd["mouse"]).fit(reml=False)
            L.append(f"  mixed model r ~ dpi + (1|mouse): slope {md.params['dpi']:+.4f}/day, p={md.pvalues['dpi']:.3g}")
            if dd.soma_f_raw.notna().sum() >= 4:
                me = smf.mixedlm("r_soma_branch ~ soma_f_raw", dd, groups=dd["mouse"]).fit(reml=False)
                L.append(f"  mixed model r ~ soma brightness + (1|mouse): slope {1000*me.params['soma_f_raw']:+.3f}/1000 F, p={me.pvalues['soma_f_raw']:.3g}")
        except Exception as e:
            L.append(f"  (mixed model skipped: {type(e).__name__}: {str(e)[:60]})")
    else:
        L.append("  too few runs / sessions yet for a time trend")
    L.append("")
    L.append("Caveats: expression is proxied by soma raw F, which also depends on laser power and depth; "
             "compare within a mouse at fixed settings. A rise in coupling with expression is consistent with "
             "sensor filling / buffering but does not by itself demonstrate the mechanism.")
    return "\n".join(L)


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
    txt = tests(d)
    dd = collect_distance()
    if not dd.empty:
        from scipy import stats as _st
        dd.to_csv(OUT / "coupling_by_distance.csv", index=False)
        fig_distance(dd, OUT / "fig_coupling_vs_distance")
        rho = _st.spearmanr(dd.distance_um, dd.r_with_soma_corr) if len(dd) >= 4 else None
        slopes = [np.polyfit(g.distance_um, g.r_with_soma_corr, 1)[0] * 100 for _, g in dd.groupby("behavior_base") if len(g) >= 2]
        txt += ("\n\nCoupling vs distance from soma (noise-corrected, all regions)"
                + (f"\n  pooled Spearman rho={rho.statistic:+.2f} p={rho.pvalue:.3g} over {len(dd)} regions" if rho else "")
                + (f"\n  per-cell slope: {', '.join(f'{x:+.2f}' for x in slopes)} r per 100 um" if slopes else ""))
    (OUT / "cohort_summary.txt").write_text(txt + "\n"); print(txt)
    fig_independence(d, OUT / "fig_independence"); fig_within_mouse(d, OUT / "fig_within_mouse"); fig_expression(d, OUT / "fig_expression")
    print(f"\nwrote stats/cohort_metrics.csv ({len(d)} runs), cohort_summary.txt, fig_independence, fig_within_mouse, fig_expression, fig_coupling_vs_distance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
