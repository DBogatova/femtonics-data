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
import argparse, json, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "stats"

METRICS = ["r_soma_branch", "r_soma_branch_core", "r_soma_trunk", "r_trunk_branch",
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
        if "note" in m:
            continue
        row = {k: m.get(k) for k in ["behavior_base", "mouse", "date", "rank", "frame_rate_hz", "T",
                                     "imaging_quality", "n_regions"] + METRICS}
        row["n_branch_regions"] = sum(1 for v in m["regions"].values() if v["compartment"] == "branch")
        row["metrics_file"] = str(Path(f).relative_to(ROOT))
        rows.append(row)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    mice = pd.read_csv(ROOT / "mice.csv", parse_dates=["injection_date"])
    d = d.merge(mice[["mouse", "injection_date", "line", "virus"]], on="mouse", how="left")
    d["date_dt"] = pd.to_datetime(d["date"], format="%m-%d-%Y")
    d["dpi"] = (d["date_dt"] - d["injection_date"]).dt.days
    d["short"] = d["mouse"].str.replace("rbp4_", "", regex=False) + " " + d["date"].str[:5] + " r" + \
        d["behavior_base"].str.extract(r"Run(\d+)")[0].astype(int).astype(str)
    return d.sort_values(["mouse", "date_dt", "rank"]).reset_index(drop=True)


def tests(d: pd.DataFrame) -> str:
    from scipy import stats
    L = []
    n = len(d); L.append(f"{n} run(s), {d.mouse.nunique()} mouse/mice: " + ", ".join(f"{m} ({k})" for m, k in d.mouse.value_counts().items()))
    L.append("")
    L.append("H1  Soma-dendrite independence")
    x = d["r_soma_branch"].dropna(); y = d["r_soma_trunk"].dropna()
    L.append(f"  r(soma,branch) mean {x.mean():.2f} [{x.min():.2f}..{x.max():.2f}]   r(soma,trunk) mean {y.mean():.2f}")
    both = d.dropna(subset=["r_soma_branch", "r_soma_trunk"])
    if len(both) >= 2:
        w = stats.wilcoxon(both["r_soma_trunk"], both["r_soma_branch"], alternative="greater")
        L.append(f"  branch less coupled than trunk: {int((both.r_soma_trunk > both.r_soma_branch).sum())}/{len(both)} runs, "
                 f"Wilcoxon p={w.pvalue:.3g}" + ("  (n small - descriptive)" if len(both) < 6 else ""))
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
    txt = tests(d); (OUT / "cohort_summary.txt").write_text(txt + "\n"); print(txt)
    fig_independence(d, OUT / "fig_independence"); fig_within_mouse(d, OUT / "fig_within_mouse"); fig_expression(d, OUT / "fig_expression")
    print(f"\nwrote stats/cohort_metrics.csv ({len(d)} runs), cohort_summary.txt, fig_independence, fig_within_mouse, fig_expression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
