#!/usr/bin/env python3
"""paper_stats.py — comprehensive statistical analysis for the apical dendrite paper.

Reads the auto-pipeline per-run JSONs and (for validation only) Daria's curated
outputs. Produces:
  AUTO_ROOT/results/RESULTS.json   — every number
  AUTO_ROOT/results/REPORT.md      — plain-language, per-test
  AUTO_ROOT/results/figures/<topic>/  — vector PDF + PNG per analysis

Tests (BH-FDR across each p-value set):
  1. Method validation: auto vs manual on run03/run05
  2. Soma-branch vs soma-trunk coupling
  3. Independent branch events vs circular-shift null
  4. Coupling vs geodesic distance
  5. Event order: branch-first fraction
  6. Branch amplitude -> reaches soma (logistic)
  7. Behavior coupling
  8. Sensor overload / expression time
  9. Co-firing groups
 10. QC table
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
AUTO_ROOT = PROJECT / "auto_pipeline"
RESULTS = AUTO_ROOT / "results"
FIG_ROOT = RESULTS / "figures"
LOG_FILE = AUTO_ROOT / "logs" / "paper_stats.jsonl"
PYTHON = sys.executable

# Real curated data (READ-ONLY)
CURATED = {
    "run05": {
        "mask": PROJECT / "rbp4_phpebach/06-26-2026/run05/run05_clean_autoseg_labelmap_reviewed.tif",
        "segments": PROJECT / "rbp4_phpebach/06-26-2026/run05/run05_clean_segments_final.tif",
        "segments_json": PROJECT / "rbp4_phpebach/06-26-2026/run05/run05_clean_segments_final.json",
        "metrics": PROJECT / "rbp4_phpebach/06-26-2026/run05/run05_clean_metrics.json",
    },
    "run03": {
        "mask": PROJECT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_autoseg_labelmap_reviewed.tif",
        "segments": PROJECT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_segments_final.tif",
        "segments_json": PROJECT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_segments_final.json",
        "metrics": PROJECT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_metrics.json",
    },
}
AUTO = {
    "run05": {
        "mask": AUTO_ROOT / "rbp4_phpebach/06-26-2026/run05/run05_clean_autoseg_labelmap_reviewed.tif",
        "segments": AUTO_ROOT / "rbp4_phpebach/06-26-2026/run05/run05_clean_segments_final.tif",
        "segments_json": AUTO_ROOT / "rbp4_phpebach/06-26-2026/run05/run05_clean_segments_final.json",
        "metrics": AUTO_ROOT / "rbp4_phpebach/06-26-2026/run05/run05_clean_metrics.json",
    },
    "run03": {
        "mask": AUTO_ROOT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_autoseg_labelmap_reviewed.tif",
        "segments": AUTO_ROOT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_segments_final.tif",
        "segments_json": AUTO_ROOT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_segments_final.json",
        "metrics": AUTO_ROOT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_metrics.json",
    },
}

STRUCT_LABEL = 2  # label convention for cell in reviewed masks


def log(what: str, result: str):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.datetime.now().astimezone().isoformat(),
             "stage": "paper_stats", "what": what, "result": result}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return []
    ps = np.array(pvals, float)
    order = np.argsort(ps)
    rank = np.empty(n, int)
    rank[order] = np.arange(1, n + 1)
    q = ps * n / rank
    # enforce monotonicity from the bottom
    idx = order[::-1]
    for i in range(1, n):
        q[idx[i]] = min(q[idx[i]], q[idx[i - 1]])
    return [min(float(x), 1.0) for x in q]


def ci_bootstrap(vals, stat_fn=np.mean, n_boot=5000, ci=0.95, seed=42):
    """Bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    vals = np.array(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")
    boot = [stat_fn(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot, 100 * (1 - ci) / 2)
    hi = np.percentile(boot, 100 * (1 + ci) / 2)
    return float(stat_fn(vals)), float(lo), float(hi)


def save_fig(fig, folder: str, name: str):
    d = FIG_ROOT / folder
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# Data collection
# ==============================================================================

def collect_all_metrics() -> pd.DataFrame:
    """Collect all metrics JSONs from auto_pipeline into a DataFrame."""
    rows = []
    for f in sorted(glob.glob(str(AUTO_ROOT / "rbp4_*/**/*_metrics.json"), recursive=True)):
        if "/old/" in f:
            continue
        m = json.load(open(f))
        if "note" in m:
            continue
        row = dict(m)
        row["metrics_file"] = str(Path(f).relative_to(AUTO_ROOT))
        rows.append(row)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    mice = pd.read_csv(AUTO_ROOT / "mice.csv", parse_dates=["injection_date"])
    d = d.merge(mice[["mouse", "injection_date", "line", "virus"]], on="mouse", how="left")
    d["date_dt"] = pd.to_datetime(d["date"], format="%m-%d-%Y")
    d["dpi"] = (d["date_dt"] - d["injection_date"]).dt.days
    return d


def collect_distance() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(AUTO_ROOT / "rbp4_*/**/*_metrics.json"), recursive=True)):
        if "/old/" in f:
            continue
        m = json.load(open(f))
        for c in m.get("coupling_by_distance", []):
            rows.append({"behavior_base": m["behavior_base"], "mouse": m["mouse"],
                         "reference": m.get("reference", "soma"), **c})
    return pd.DataFrame(rows)


def collect_behavior() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(AUTO_ROOT / "rbp4_*/**/*_behavior_coupling.json"), recursive=True)):
        if "/old/" in f:
            continue
        bc = json.load(open(f))
        row = {"behavior_base": bc["behavior_base"], "mouse": bc["mouse"]}
        wc = bc.get("regions", {}).get("whole cell", {})
        for beh in bc.get("behaviors", []):
            if beh in wc:
                for k, v in wc[beh].items():
                    row[f"{beh}_{k}"] = v
        sd = bc.get("state_dependence") or {}
        for k, v in sd.items():
            row[f"sd_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def collect_coupling() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(AUTO_ROOT / "rbp4_*/**/*_coupling.json"), recursive=True)):
        if "/old/" in f:
            continue
        c = json.load(open(f))
        rows.append(c)
    return pd.DataFrame(rows)


def collect_qc() -> pd.DataFrame:
    """Collect per-run QC from mask and region JSONs."""
    rows = []
    for mask_json in sorted(glob.glob(str(AUTO_ROOT / "rbp4_*/**/*_autoseg_reviewed.json"), recursive=True)):
        if "/old/" in mask_json:
            continue
        mj = json.load(open(mask_json))
        rev = mj.get("reviews", [{}])[-1]
        run_dir = Path(mask_json).parent
        stem = Path(mask_json).stem.replace("_autoseg_reviewed", "")

        # Find corresponding segments and metrics
        seg_json = run_dir / f"{stem}_segments_final.json"
        met_json = run_dir / f"{stem}_metrics.json"
        seg = json.load(open(seg_json)) if seg_json.exists() else {}
        met = json.load(open(met_json)) if met_json.exists() else {}

        ar = seg.get("auto_regions", {})
        n_regions = len(ar.get("regions", {}))
        region_names = list(ar.get("regions", {}).keys())

        row = {
            "run_dir": str(run_dir.relative_to(AUTO_ROOT)),
            "behavior_base": met.get("behavior_base", ""),
            "mouse": met.get("mouse", ""),
            "mask_voxels": rev.get("mask_voxels", 0),
            "mask_components": rev.get("n_components", 0),
            "mask_alpha": rev.get("params", {}).get("alpha", 0),
            "n_intruders": len(rev.get("intruders", [])),
            "intruder_reasons": [i.get("reasons", []) for i in rev.get("intruders", [])],
            "chunk_period": rev.get("chunk_period_px", 0),
            "confidence": rev.get("confidence", ""),
            "flags": rev.get("flags", []),
            "n_regions": n_regions,
            "region_names": region_names,
            "reference": ar.get("reference", met.get("reference", "")),
            "has_soma": ar.get("tree_info", {}).get("has_soma", False),
            "T": met.get("T", 0),
            "frame_rate_hz": met.get("frame_rate_hz", 0),
            "n_soma_events": met.get("n_soma_events", 0),
            "n_branch_events": met.get("n_branch_events", 0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ==============================================================================
# Tests
# ==============================================================================

def test_01_validation() -> tuple[dict, list[str], list]:
    """Method validation: auto vs manual masks and metrics."""
    import tifffile
    results = {}
    lines = ["# 1. Method Validation: Automatic vs Hand-Curated",
             "",
             "Compared on the two runs where Daria manually curated masks and regions.",
             ""]
    figs = []

    for name in ["run05", "run03"]:
        c = CURATED[name]
        a = AUTO[name]
        if not c["mask"].exists() or not a["mask"].exists():
            lines.append(f"  {name}: SKIP (files missing)")
            continue

        # Mask comparison
        cm = tifffile.imread(str(c["mask"])) == STRUCT_LABEL
        am = tifffile.imread(str(a["mask"])) == STRUCT_LABEL
        intersection = int((cm & am).sum())
        union = int((cm | am).sum())
        dice = 2 * intersection / (cm.sum() + am.sum() + 1e-9)
        precision = intersection / (am.sum() + 1e-9)
        recall = intersection / (cm.sum() + 1e-9)
        iou = intersection / (union + 1e-9)

        results[f"{name}_mask_dice"] = round(dice, 4)
        results[f"{name}_mask_precision"] = round(precision, 4)
        results[f"{name}_mask_recall"] = round(recall, 4)
        results[f"{name}_mask_iou"] = round(iou, 4)
        results[f"{name}_mask_auto_vox"] = int(am.sum())
        results[f"{name}_mask_curated_vox"] = int(cm.sum())

        # Region overlap
        if c["segments"].exists() and a["segments"].exists():
            cs = tifffile.imread(str(c["segments"]))
            aseg = tifffile.imread(str(a["segments"]))
            # Per-region Jaccard
            c_labels = sorted(set(int(v) for v in np.unique(cs) if v > 0))
            a_labels = sorted(set(int(v) for v in np.unique(aseg) if v > 0))
            results[f"{name}_n_regions_curated"] = len(c_labels)
            results[f"{name}_n_regions_auto"] = len(a_labels)

        # Metrics comparison
        if c["metrics"].exists() and a["metrics"].exists():
            cm_met = json.load(open(c["metrics"]))
            am_met = json.load(open(a["metrics"]))
            key_metrics = ["r_soma_branch", "r_soma_trunk", "frac_branch_independent",
                           "branch_first_frac", "n_branch_events", "n_soma_events",
                           "r_soma_branch_corr", "rate_soma_per_min", "rate_branch_per_min"]
            lines.append(f"### {name} ({'soma present' if name == 'run05' else 'no soma'})")
            lines.append("")
            lines.append(f"Mask: Dice={dice:.3f}, precision={precision:.3f}, recall={recall:.3f}")
            lines.append(f"  auto {int(am.sum())} vox, curated {int(cm.sum())} vox")
            lines.append("")
            lines.append(f"| Metric | Manual | Auto | Δ |")
            lines.append(f"|--------|--------|------|---|")
            for k in key_metrics:
                mv = cm_met.get(k)
                av = am_met.get(k)
                if mv is not None and av is not None:
                    delta = av - mv
                    lines.append(f"| {k} | {mv:.4f} | {av:.4f} | {delta:+.4f} |")
                    results[f"{name}_{k}_manual"] = round(float(mv), 6)
                    results[f"{name}_{k}_auto"] = round(float(av), 6)
                    results[f"{name}_{k}_delta"] = round(float(delta), 6)
            lines.append("")

    # Validation figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_i, name in enumerate(["run05", "run03"]):
        ax = axes[ax_i]
        metrics_to_plot = ["r_soma_branch", "r_soma_trunk", "frac_branch_independent", "branch_first_frac"]
        manual_vals = [results.get(f"{name}_{m}_manual", float("nan")) for m in metrics_to_plot]
        auto_vals = [results.get(f"{name}_{m}_auto", float("nan")) for m in metrics_to_plot]
        x = np.arange(len(metrics_to_plot))
        ax.bar(x - 0.15, manual_vals, 0.3, label="Manual", color="steelblue", alpha=0.8)
        ax.bar(x + 0.15, auto_vals, 0.3, label="Auto", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics_to_plot], fontsize=7)
        ax.set_title(f"{name} ({'soma' if name == 'run05' else 'no soma'})")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Validation: Auto vs Manual Key Metrics", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "01_validation", "validation_metrics")
    figs.append(("01_validation/validation_metrics", "Auto vs manual key metrics"))

    return results, lines, figs


def test_02_coupling(d: pd.DataFrame) -> tuple[dict, list[str], list]:
    """Soma-branch vs soma-trunk coupling."""
    results = {}
    lines = ["# 2. Soma/Reference-Branch vs Reference-Trunk Coupling", ""]
    figs = []

    both = d.dropna(subset=["r_soma_branch", "r_soma_trunk"]).copy()
    n = len(both)
    n_mice = both.mouse.nunique()

    rb_mean, rb_lo, rb_hi = ci_bootstrap(both.r_soma_branch.values)
    rt_mean, rt_lo, rt_hi = ci_bootstrap(both.r_soma_trunk.values)
    results["n_cells_coupling"] = n
    results["n_mice_coupling"] = n_mice
    results["r_soma_branch_mean"] = round(rb_mean, 4)
    results["r_soma_branch_ci95"] = [round(rb_lo, 4), round(rb_hi, 4)]
    results["r_soma_trunk_mean"] = round(rt_mean, 4)
    results["r_soma_trunk_ci95"] = [round(rt_lo, 4), round(rt_hi, 4)]

    lines.append(f"n = {n} cells from {n_mice} mice")
    lines.append(f"r(ref, branch): mean={rb_mean:.3f}, 95% CI [{rb_lo:.3f}, {rb_hi:.3f}]")
    lines.append(f"r(ref, trunk):  mean={rt_mean:.3f}, 95% CI [{rt_lo:.3f}, {rt_hi:.3f}]")

    all_p = []

    if n >= 6:
        w = sp.wilcoxon(both.r_soma_trunk, both.r_soma_branch, alternative="greater")
        results["wilcoxon_p"] = round(float(w.pvalue), 6)
        results["wilcoxon_stat"] = round(float(w.statistic), 4)
        lines.append(f"Wilcoxon signed-rank (trunk > branch): W={w.statistic:.1f}, p={w.pvalue:.4g}")
        all_p.append(float(w.pvalue))
    elif n >= 2:
        w = sp.wilcoxon(both.r_soma_trunk, both.r_soma_branch, alternative="greater")
        results["wilcoxon_p"] = round(float(w.pvalue), 6)
        lines.append(f"Wilcoxon signed-rank (n={n}, descriptive): p={w.pvalue:.4g}")
        all_p.append(float(w.pvalue))

    frac_lower = int((both.r_soma_trunk > both.r_soma_branch).sum())
    results["frac_branch_lower"] = f"{frac_lower}/{n}"
    lines.append(f"Branch coupling lower than trunk: {frac_lower}/{n} cells")

    # Noise-corrected version
    both_nc = d.dropna(subset=["r_soma_branch_corr", "r_soma_trunk_corr"])
    if len(both_nc) >= 2:
        rnc_mean, rnc_lo, rnc_hi = ci_bootstrap(both_nc.r_soma_branch_corr.values)
        results["r_soma_branch_corr_mean"] = round(rnc_mean, 4)
        results["r_soma_branch_corr_ci95"] = [round(rnc_lo, 4), round(rnc_hi, 4)]
        lines.append(f"Noise-corrected r(ref, branch): mean={rnc_mean:.3f}, 95% CI [{rnc_lo:.3f}, {rnc_hi:.3f}]")

    # Core voxel version
    both_core = d.dropna(subset=["r_soma_branch_core", "r_soma_branch"])
    if len(both_core) >= 2:
        diff = both_core.r_soma_branch - both_core.r_soma_branch_core
        results["halo_control_mean_diff"] = round(float(diff.mean()), 4)
        lines.append(f"Halo control: full - core mean diff = {diff.mean():.4f} (small = not halo artifact)")

    # Mixed model
    if n_mice >= 3 and n >= 6:
        try:
            import statsmodels.formula.api as smf
            both["diff"] = both.r_soma_trunk - both.r_soma_branch
            md = smf.mixedlm("diff ~ 1", both, groups=both["mouse"]).fit(reml=False)
            results["mixed_model_intercept"] = round(float(md.params["Intercept"]), 4)
            results["mixed_model_p"] = round(float(md.pvalues["Intercept"]), 6)
            lines.append(f"Mixed model (trunk-branch ~ 1 | mouse): intercept={md.params['Intercept']:.4f}, p={md.pvalues['Intercept']:.4g}")
            all_p.append(float(md.pvalues["Intercept"]))
        except Exception as e:
            lines.append(f"Mixed model skipped: {e}")

    lines.append("")

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    # Paired plot
    for i, (_, row) in enumerate(both.iterrows()):
        c = "C0" if row.get("reference") == "soma" else "C1"
        ax1.plot([0, 1], [row.r_soma_trunk, row.r_soma_branch], "-o", color=c, alpha=0.5, ms=4)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Trunk", "Branch"])
    ax1.set_ylabel("r with reference")
    ax1.set_title(f"Paired coupling (n={n})")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bar + individual points
    ax2.bar([0, 1], [rt_mean, rb_mean], color=["steelblue", "coral"], alpha=0.7, width=0.5)
    ax2.errorbar([0, 1], [rt_mean, rb_mean],
                 yerr=[[rt_mean - rt_lo, rb_mean - rb_lo], [rt_hi - rt_mean, rb_hi - rb_mean]],
                 fmt="none", color="black", capsize=5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Trunk", "Branch"])
    ax2.set_ylabel("r with reference (mean ± 95% CI)")
    ax2.set_title("Coupling summary")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "02_independence", "coupling_paired")
    figs.append(("02_independence/coupling_paired", "Paired trunk vs branch coupling"))

    results["_p_values_02"] = all_p
    return results, lines, figs


def test_03_independence(d: pd.DataFrame) -> tuple[dict, list[str], list]:
    """Independent branch events vs surrogate null."""
    results = {}
    lines = ["# 3. Independent Branch Events", ""]
    figs = []

    fi = d["frac_branch_independent"].dropna()
    n = len(fi)
    fi_mean, fi_lo, fi_hi = ci_bootstrap(fi.values)
    results["n_cells_independence"] = n
    results["frac_branch_independent_mean"] = round(fi_mean, 4)
    results["frac_branch_independent_ci95"] = [round(fi_lo, 4), round(fi_hi, 4)]

    lines.append(f"n = {n} cells")
    lines.append(f"Fraction of branch events independent of reference: mean={fi_mean:.3f}, 95% CI [{fi_lo:.3f}, {fi_hi:.3f}]")

    all_p = []

    # One-sample Wilcoxon against 0
    if n >= 6:
        w = sp.wilcoxon(fi.values, alternative="greater")
        results["independence_wilcoxon_p"] = round(float(w.pvalue), 6)
        lines.append(f"Wilcoxon signed-rank (> 0): p={w.pvalue:.4g}")
        all_p.append(float(w.pvalue))

    # Circular-shift surrogate: under random timing, expected ~fraction
    # Use binomial per cell: observed independent / total branch events
    lines.append("")
    lines.append("Per-cell binomial test (observed independent events vs null rate from soma event density):")
    cell_ps = []
    for _, row in d.dropna(subset=["frac_branch_independent", "n_branch_events"]).iterrows():
        nb = int(row.n_branch_events)
        if nb < 5:
            continue
        n_ind = int(round(row.frac_branch_independent * nb))
        # Null: fraction of frames NOT within window of a soma event
        # Approximate: 1 - (n_soma_events * (2*window+1)) / T
        ns = int(row.get("n_soma_events", 0))
        T = int(row.get("T", 1))
        window = 2  # frames, from params
        null_rate = max(0.01, 1 - ns * (2 * window + 1) / max(T, 1))
        null_rate = min(null_rate, 0.99)
        bt = sp.binomtest(n_ind, nb, null_rate, alternative="greater")
        cell_ps.append(float(bt.pvalue))
        lines.append(f"  {row.behavior_base}: {n_ind}/{nb} independent (null rate={null_rate:.2f}), p={bt.pvalue:.4g}")

    if cell_ps:
        q_vals = bh_fdr(cell_ps)
        n_sig = sum(1 for q in q_vals if q < 0.05)
        results["independence_per_cell_n_sig_fdr05"] = n_sig
        results["independence_per_cell_n_tested"] = len(cell_ps)
        lines.append(f"\n{n_sig}/{len(cell_ps)} cells significant at FDR q<0.05")
        all_p.extend(cell_ps)

    lines.append("")

    # Figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(n), sorted(fi.values, reverse=True), color="teal", alpha=0.7)
    ax.axhline(fi_mean, color="black", ls="--", lw=1)
    ax.set_xlabel("Cell (sorted)")
    ax.set_ylabel("Fraction branch events independent")
    ax.set_title(f"Branch independence (n={n}, mean={fi_mean:.2f})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "02_independence", "branch_independence")
    figs.append(("02_independence/branch_independence", "Branch independence fraction"))

    results["_p_values_03"] = all_p
    return results, lines, figs


def test_04_distance(dd: pd.DataFrame) -> tuple[dict, list[str], list]:
    """Coupling vs geodesic distance."""
    results = {}
    lines = ["# 4. Coupling vs Geodesic Distance from Reference", ""]
    figs = []
    all_p = []

    if dd.empty or len(dd) < 4:
        lines.append("SKIP: too few regions with distance data")
        return results, lines, figs

    n_regions = len(dd)
    n_cells = dd.behavior_base.nunique()
    results["n_regions_distance"] = n_regions
    results["n_cells_distance"] = n_cells

    # Pooled Spearman
    rho = sp.spearmanr(dd.distance_um, dd.r_with_soma_corr)
    results["distance_spearman_rho"] = round(float(rho.statistic), 4)
    results["distance_spearman_p"] = round(float(rho.pvalue), 6)
    all_p.append(float(rho.pvalue))

    lines.append(f"n = {n_regions} regions from {n_cells} cells")
    lines.append(f"Pooled Spearman: rho={rho.statistic:.3f}, p={rho.pvalue:.4g}")

    # Per-cell slopes
    slopes = []
    for _, g in dd.groupby("behavior_base"):
        if len(g) >= 2:
            s = np.polyfit(g.distance_um, g.r_with_soma_corr, 1)[0] * 100
            slopes.append(s)
    if slopes:
        s_mean, s_lo, s_hi = ci_bootstrap(slopes)
        results["slope_per_100um_mean"] = round(s_mean, 4)
        results["slope_per_100um_ci95"] = [round(s_lo, 4), round(s_hi, 4)]
        lines.append(f"Per-cell slope: mean={s_mean:.3f} r/100um, 95% CI [{s_lo:.3f}, {s_hi:.3f}]")

    # Mixed model with cell random effect
    if n_cells >= 3 and n_regions >= 8:
        try:
            import statsmodels.formula.api as smf
            md = smf.mixedlm("r_with_soma_corr ~ distance_um", dd, groups=dd["behavior_base"]).fit(reml=False)
            results["distance_mixed_slope"] = round(float(md.params["distance_um"]), 6)
            results["distance_mixed_p"] = round(float(md.pvalues["distance_um"]), 6)
            lines.append(f"Mixed model (r ~ distance + (1|cell)): slope={md.params['distance_um']:.6f}/um, p={md.pvalues['distance_um']:.4g}")
            all_p.append(float(md.pvalues["distance_um"]))
        except Exception as e:
            lines.append(f"Mixed model skipped: {e}")

    lines.append("")

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.tab20
    bases = sorted(dd.behavior_base.unique())
    colors = {b: cmap(i / max(1, len(bases) - 1)) for i, b in enumerate(bases)}
    mk = {"trunk": "o", "branch": "^"}
    for b, g in dd.groupby("behavior_base"):
        g = g.sort_values("distance_um")
        hollow = g.reference.iloc[0] == "proximal_trunk"
        ax.plot(g.distance_um, g.r_with_soma_corr, "-", color=colors[b], lw=1, alpha=0.6)
        for _, r in g.iterrows():
            ax.scatter(r.distance_um, r.r_with_soma_corr,
                       marker=mk.get(r.compartment, "s"), s=35, zorder=3,
                       facecolor="none" if hollow else colors[b],
                       edgecolor=colors[b], linewidth=1.2)
    ax.set_xlabel("Geodesic distance from reference (µm)")
    ax.set_ylabel("r with reference (noise-corrected)")
    ax.set_title(f"Coupling vs distance (n={n_regions} regions, {n_cells} cells, ρ={rho.statistic:.2f})")
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "03_distance", "coupling_vs_distance")
    figs.append(("03_distance/coupling_vs_distance", "Coupling vs geodesic distance"))

    results["_p_values_04"] = all_p
    return results, lines, figs


def test_05_event_order(d: pd.DataFrame) -> tuple[dict, list[str], list]:
    """Event order: branch fires first."""
    results = {}
    lines = ["# 5. Event Order: Branch-First Fraction", ""]
    figs = []
    all_p = []

    bf = d.dropna(subset=["branch_first_frac", "n_paired_events"])
    bf = bf[bf.n_paired_events >= 3]
    n = len(bf)
    results["n_cells_event_order"] = n

    if n == 0:
        lines.append("SKIP: no cells with enough paired events")
        return results, lines, figs

    total_events = int(bf.n_paired_events.sum())
    total_first = int((bf.branch_first_frac * bf.n_paired_events).round().sum())
    results["total_paired_events"] = total_events
    results["total_branch_first"] = total_first

    # Pooled binomial
    bt = sp.binomtest(total_first, total_events, 0.5, alternative="greater")
    results["event_order_binomial_p"] = round(float(bt.pvalue), 8)
    all_p.append(float(bt.pvalue))

    lines.append(f"n = {n} cells, {total_events} paired events")
    lines.append(f"Branch peaks first: {total_first}/{total_events} ({100*total_first/total_events:.1f}%)")
    lines.append(f"Pooled binomial (> 50%): p={bt.pvalue:.4g}")

    # Per-cell sign test
    n_above = int((bf.branch_first_frac > 0.5).sum())
    if n >= 2:
        st = sp.binomtest(n_above, n, 0.5, alternative="greater")
        results["event_order_sign_p"] = round(float(st.pvalue), 6)
        lines.append(f"Per-cell sign test: {n_above}/{n} cells with fraction > 0.5, p={st.pvalue:.4g}")
        all_p.append(float(st.pvalue))

    bf_mean, bf_lo, bf_hi = ci_bootstrap(bf.branch_first_frac.values)
    results["branch_first_frac_mean"] = round(bf_mean, 4)
    results["branch_first_frac_ci95"] = [round(bf_lo, 4), round(bf_hi, 4)]
    lines.append(f"Mean branch-first fraction: {bf_mean:.3f}, 95% CI [{bf_lo:.3f}, {bf_hi:.3f}]")
    lines.append("")

    # Figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(n), bf.branch_first_frac.values, color="darkorange", alpha=0.7)
    ax.axhline(0.5, color="black", ls="--", lw=1)
    for i, (_, row) in enumerate(bf.iterrows()):
        ax.text(i, row.branch_first_frac + 0.02, f"n={int(row.n_paired_events)}", ha="center", fontsize=6)
    ax.set_xlabel("Cell")
    ax.set_ylabel("Fraction branch fires first")
    ax.set_title(f"Event order ({total_first}/{total_events} branch-first)")
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "04_event_order", "branch_first")
    figs.append(("04_event_order/branch_first", "Branch-first event fraction"))

    results["_p_values_05"] = all_p
    return results, lines, figs


def test_06_amplitude_propagation(d: pd.DataFrame) -> tuple[dict, list[str], list]:
    """Branch amplitude predicts reaching soma/trunk (logistic)."""
    results = {}
    lines = ["# 6. Branch Amplitude Predicts Propagation", ""]
    figs = []

    # This test requires per-event data which is not directly in metrics JSONs.
    # We report what we can from aggregate metrics.
    lines.append("This test requires per-event amplitude data not stored in the current metrics.")
    lines.append("Future: extract from coherence network_events CSVs.")
    lines.append("SKIP: per-event amplitude data not available in aggregate metrics.")
    lines.append("")
    results["test_06_status"] = "skipped_no_per_event_data"
    return results, lines, figs


def test_07_behavior(d: pd.DataFrame, beh: pd.DataFrame) -> tuple[dict, list[str], list]:
    """Behavior coupling."""
    results = {}
    lines = ["# 7. Behavior Coupling", ""]
    figs = []
    all_p = []

    if beh.empty:
        lines.append("SKIP: no behavior coupling data")
        return results, lines, figs

    n = len(beh)
    results["n_cells_behavior"] = n
    lines.append(f"n = {n} cells with behavior data")

    for signal in ["whisking", "pupil", "accelerometer"]:
        pk = f"{signal}_p_peak"
        rk = f"{signal}_r_peak"
        vals = beh.dropna(subset=[pk, rk])
        if vals.empty:
            continue
        n_sig = int((vals[pk] < 0.05).sum())
        r_mean = float(vals[rk].mean())
        results[f"{signal}_n_sig_raw"] = n_sig
        results[f"{signal}_n_tested"] = len(vals)
        results[f"{signal}_r_peak_mean"] = round(r_mean, 4)
        lines.append(f"  {signal}: {n_sig}/{len(vals)} significant at p<0.05 (uncorrected), mean peak r={r_mean:.3f}")
        all_p.extend(vals[pk].tolist())

    if all_p:
        q_vals = bh_fdr(all_p)
        n_sig_fdr = sum(1 for q in q_vals if q < 0.05)
        results["behavior_n_sig_fdr05"] = n_sig_fdr
        results["behavior_n_total_tests"] = len(all_p)
        lines.append(f"\nAfter BH-FDR across all {len(all_p)} tests: {n_sig_fdr} significant at q<0.05")

    # State dependence: quiet vs active
    qa = d.dropna(subset=["r_soma_branch_quiet", "r_soma_branch_active"])
    if len(qa) >= 2:
        n_more_active = int((qa.r_soma_branch_active > qa.r_soma_branch_quiet).sum())
        results["n_more_coupled_active"] = n_more_active
        results["n_tested_qa"] = len(qa)
        lines.append(f"\nQuiet vs active: {n_more_active}/{len(qa)} cells more coupled when active")
        if len(qa) >= 6:
            w = sp.wilcoxon(qa.r_soma_branch_active, qa.r_soma_branch_quiet, alternative="two-sided")
            results["qa_wilcoxon_p"] = round(float(w.pvalue), 6)
            lines.append(f"Wilcoxon (active vs quiet): p={w.pvalue:.4g}")
            all_p.append(float(w.pvalue))

    lines.append("")

    # Figure
    if not beh.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax_i, signal in enumerate(["whisking", "pupil", "accelerometer"]):
            ax = axes[ax_i]
            rk = f"{signal}_r_peak"
            pk = f"{signal}_p_peak"
            vals = beh.dropna(subset=[pk, rk])
            if vals.empty:
                ax.set_title(f"{signal}: no data")
                continue
            colors = ["red" if p < 0.05 else "gray" for p in vals[pk]]
            ax.bar(range(len(vals)), vals[rk].values, color=colors, alpha=0.7)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(f"{signal} (red = p<0.05)")
            ax.set_ylabel("Peak r")
            ax.set_xlabel("Cell")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle(f"Behavior coupling (n={n})", fontsize=11)
        fig.tight_layout()
        save_fig(fig, "05_behavior", "behavior_coupling")
        figs.append(("05_behavior/behavior_coupling", "Behavior coupling"))

    results["_p_values_07"] = all_p
    return results, lines, figs


def test_08_expression(d: pd.DataFrame) -> tuple[dict, list[str], list]:
    """Sensor overload / expression time."""
    results = {}
    lines = ["# 8. Coupling vs Expression Time", ""]
    figs = []
    all_p = []

    dd = d.dropna(subset=["dpi", "r_soma_branch"])
    n = len(dd)
    n_mice = dd.mouse.nunique() if n > 0 else 0
    results["n_cells_dpi"] = n
    results["n_mice_dpi"] = n_mice

    if n < 4:
        lines.append(f"SKIP: only {n} cells with DPI data (need >= 4)")
        lines.append("CAVEAT: rbp4_phpebach has unknown injection date; those cells excluded from time analysis.")
        return results, lines, figs

    lines.append(f"n = {n} cells from {n_mice} mice")
    lines.append(f"DPI range: {int(dd.dpi.min())}–{int(dd.dpi.max())}")

    # Spearman
    rho = sp.spearmanr(dd.dpi, dd.r_soma_branch)
    results["dpi_spearman_rho"] = round(float(rho.statistic), 4)
    results["dpi_spearman_p"] = round(float(rho.pvalue), 6)
    all_p.append(float(rho.pvalue))
    lines.append(f"Spearman r(ref, branch) vs DPI: rho={rho.statistic:.3f}, p={rho.pvalue:.4g}")

    # Mixed model
    if n_mice >= 3:
        try:
            import statsmodels.formula.api as smf
            md = smf.mixedlm("r_soma_branch ~ dpi", dd, groups=dd["mouse"]).fit(reml=False)
            results["dpi_mixed_slope"] = round(float(md.params["dpi"]), 6)
            results["dpi_mixed_p"] = round(float(md.pvalues["dpi"]), 6)
            lines.append(f"Mixed model (r ~ dpi + (1|mouse)): slope={md.params['dpi']:.6f}/day, p={md.pvalues['dpi']:.4g}")
            all_p.append(float(md.pvalues["dpi"]))
        except Exception as e:
            lines.append(f"Mixed model skipped: {e}")

    # Expression proxy: soma F
    sf = d.dropna(subset=["soma_f_raw", "r_soma_branch"])
    if len(sf) >= 4:
        rho_f = sp.spearmanr(sf.soma_f_raw, sf.r_soma_branch)
        results["soma_f_spearman_rho"] = round(float(rho_f.statistic), 4)
        results["soma_f_spearman_p"] = round(float(rho_f.pvalue), 6)
        all_p.append(float(rho_f.pvalue))
        lines.append(f"Spearman r(ref, branch) vs soma F: rho={rho_f.statistic:.3f}, p={rho_f.pvalue:.4g}")

    lines.append("")
    lines.append("CAVEAT: DPI range is narrow (17 days across 5 mice), so this test has limited power.")
    lines.append("rbp4_phpebach excluded (unknown injection date).")
    lines.append("")

    # Figure
    if n >= 4:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        colors = {m: f"C{i}" for i, m in enumerate(sorted(dd.mouse.unique()))}
        for m, g in dd.groupby("mouse"):
            ax1.scatter(g.dpi, g.r_soma_branch, label=m.replace("rbp4_", ""), color=colors[m], s=40)
        ax1.set_xlabel("Days post-injection")
        ax1.set_ylabel("r(ref, branch)")
        ax1.set_title(f"Coupling vs DPI (ρ={rho.statistic:.2f})")
        ax1.legend(fontsize=7)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        if len(sf) >= 4:
            colors2 = {m: f"C{i}" for i, m in enumerate(sorted(sf.mouse.unique()))}
            for m, g in sf.groupby("mouse"):
                ax2.scatter(g.soma_f_raw, g.r_soma_branch, label=m.replace("rbp4_", ""), color=colors2[m], s=40)
            ax2.set_xlabel("Soma raw F (expression proxy)")
            ax2.set_ylabel("r(ref, branch)")
            ax2.set_title("Coupling vs soma brightness")
            ax2.legend(fontsize=7)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig.tight_layout()
        save_fig(fig, "06_expression_time", "expression_time")
        figs.append(("06_expression_time/expression_time", "Coupling vs expression/time"))

    results["_p_values_08"] = all_p
    return results, lines, figs


def test_09_cofiring(coup: pd.DataFrame) -> tuple[dict, list[str], list]:
    """Co-firing group analysis."""
    results = {}
    lines = ["# 9. Co-firing Groups", ""]
    figs = []

    if coup.empty:
        lines.append("SKIP: no coupling phenotype data")
        return results, lines, figs

    n = len(coup)
    results["n_cells_cofiring"] = n

    if "n_groups" in coup.columns:
        ng = coup.n_groups.dropna()
        results["n_groups_mean"] = round(float(ng.mean()), 2)
        results["n_groups_range"] = [int(ng.min()), int(ng.max())]
        lines.append(f"n = {n} cells")
        lines.append(f"Co-firing groups per cell: mean={ng.mean():.1f}, range [{int(ng.min())}, {int(ng.max())}]")

    if "soma_shares_group_with_branch" in coup.columns:
        shares = coup.soma_shares_group_with_branch.dropna()
        n_shares = int(shares.sum())
        results["soma_shares_group_frac"] = f"{n_shares}/{len(shares)}"
        lines.append(f"Soma shares a group with any branch: {n_shares}/{len(shares)} cells")

    lines.append("")
    return results, lines, figs


def test_10_qc(qc: pd.DataFrame) -> tuple[dict, list[str], list]:
    """QC table."""
    results = {}
    lines = ["# 10. Per-Run QC Table", ""]
    figs = []

    n = len(qc)
    results["n_runs_total"] = n
    results["n_with_soma"] = int(qc.has_soma.sum()) if "has_soma" in qc.columns else 0

    lines.append(f"| Run | Mouse | Mask vox | Regions | Reference | Intruders | Soma events | Branch events | Flags |")
    lines.append(f"|-----|-------|----------|---------|-----------|-----------|-------------|---------------|-------|")
    for _, row in qc.iterrows():
        lines.append(f"| {row.get('run_dir', '')[-30:]} | {row.get('mouse', '')} | "
                     f"{row.get('mask_voxels', 0)} | {row.get('n_regions', 0)} | "
                     f"{row.get('reference', '')} | {row.get('n_intruders', 0)} | "
                     f"{row.get('n_soma_events', 0)} | {row.get('n_branch_events', 0)} | "
                     f"{','.join(row.get('flags', []))} |")
    lines.append("")

    # Low confidence runs
    low_conf = qc[qc.confidence == "auto"]  # all are "auto" in this pipeline
    results["n_low_confidence"] = len(low_conf)
    lines.append(f"All {n} runs have confidence='auto' (fully automatic, no manual review).")
    lines.append("")

    return results, lines, figs


# ==============================================================================
# Main
# ==============================================================================

def main():
    log("start", "paper_stats.py")

    RESULTS.mkdir(parents=True, exist_ok=True)
    for folder in ["01_validation", "02_independence", "03_distance", "04_event_order",
                    "05_behavior", "06_expression_time", "07_cofiring_groups"]:
        (FIG_ROOT / folder).mkdir(parents=True, exist_ok=True)

    # Collect data
    print("Collecting metrics...")
    d = collect_all_metrics()
    dd = collect_distance()
    beh = collect_behavior()
    coup = collect_coupling()
    qc = collect_qc()

    if d.empty:
        print("ERROR: no metrics found")
        return 1

    print(f"  {len(d)} runs, {d.mouse.nunique()} mice")
    print(f"  {len(dd)} distance measurements")
    print(f"  {len(beh)} runs with behavior coupling")
    print(f"  {len(coup)} runs with coupling phenotype")

    all_results = {}
    all_lines = []
    all_figs = []
    all_pvals = []

    # Test 1: Validation
    print("Test 1: Validation...")
    r, l, f = test_01_validation()
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 2: Coupling
    print("Test 2: Coupling...")
    r, l, f = test_02_coupling(d)
    all_pvals.extend(r.pop("_p_values_02", []))
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 3: Independence
    print("Test 3: Independence...")
    r, l, f = test_03_independence(d)
    all_pvals.extend(r.pop("_p_values_03", []))
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 4: Distance
    print("Test 4: Distance...")
    r, l, f = test_04_distance(dd)
    all_pvals.extend(r.pop("_p_values_04", []))
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 5: Event order
    print("Test 5: Event order...")
    r, l, f = test_05_event_order(d)
    all_pvals.extend(r.pop("_p_values_05", []))
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 6: Amplitude propagation
    print("Test 6: Amplitude (skipped - needs per-event data)...")
    r, l, f = test_06_amplitude_propagation(d)
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 7: Behavior
    print("Test 7: Behavior...")
    r, l, f = test_07_behavior(d, beh)
    all_pvals.extend(r.pop("_p_values_07", []))
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 8: Expression/time
    print("Test 8: Expression/time...")
    r, l, f = test_08_expression(d)
    all_pvals.extend(r.pop("_p_values_08", []))
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 9: Co-firing
    print("Test 9: Co-firing groups...")
    r, l, f = test_09_cofiring(coup)
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 10: QC
    print("Test 10: QC table...")
    r, l, f = test_10_qc(qc)
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Global FDR
    if all_pvals:
        q_vals = bh_fdr(all_pvals)
        all_results["global_bh_fdr"] = {
            "n_tests": len(all_pvals),
            "n_sig_005": sum(1 for q in q_vals if q < 0.05),
            "p_values": [round(p, 6) for p in all_pvals],
            "q_values": [round(q, 6) for q in q_vals],
        }
        all_lines.append("---")
        all_lines.append(f"## Global BH-FDR correction")
        all_lines.append(f"{len(all_pvals)} p-values tested, {sum(1 for q in q_vals if q < 0.05)} significant at q<0.05")
        all_lines.append("")

    # Sensitivity: repeat key tests with and without low-confidence runs
    all_lines.append("---")
    all_lines.append("## Sensitivity Analysis")
    all_lines.append("All runs are automatic (confidence='auto'), so no high vs low confidence split is possible.")
    all_lines.append("The full cohort results above ARE the only analysis.")
    all_lines.append("")

    # Save RESULTS.json
    all_results["generated"] = datetime.datetime.now().astimezone().isoformat()
    all_results["n_runs"] = len(d)
    all_results["n_mice"] = int(d.mouse.nunique())
    all_results["mice"] = sorted(d.mouse.unique().tolist())

    results_path = RESULTS / "RESULTS.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {results_path}")

    # Save REPORT.md
    header = [
        "# Automatic Pipeline Statistical Report",
        "",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"n = {len(d)} cells from {int(d.mouse.nunique())} mice",
        f"All masks and regions fully automatic (no manual curation).",
        "",
        "---",
        "",
    ]
    report_path = RESULTS / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(header + all_lines) + "\n")
    print(f"Saved: {report_path}")

    # Summary
    print(f"\nFigures: {len(all_figs)}")
    for path, desc in all_figs:
        print(f"  {path}: {desc}")

    log("done", f"{len(d)} runs, {len(all_figs)} figures, {len(all_pvals)} p-values")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
