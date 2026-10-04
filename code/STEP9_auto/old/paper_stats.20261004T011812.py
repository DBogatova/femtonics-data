#!/usr/bin/env python3
"""paper_stats.py — comprehensive statistical analysis for the apical dendrite paper.

Reads the auto-pipeline per-run JSONs and (for validation only) Daria's curated
outputs. Produces:
  AUTO_ROOT/results/RESULTS.json   — every number
  AUTO_ROOT/results/REPORT.md      — plain-language, per-test
  AUTO_ROOT/results/figures/<topic>/  — vector PDF + PNG per analysis

Tests (BH-FDR across cohort-level p-values):
  1. Method validation: auto vs manual on run03/run05
  2. Soma-branch vs soma-trunk coupling (paired Wilcoxon + mixed model)
  3. Independent branch events: dual-null surrogate test
     (A) COUPLED-NOISE NULL: how many 'independent' events does detection noise
         produce from a perfectly coupled branch? Synthetic branch = gain*reference
         + phase-randomized residual; same event detector.
     (B) INDEPENDENT NULL: circular shifts (>20 s) of the branch trace — what
         fraction of events look independent if coupling is destroyed?
     Per-cell: observed frac_independent bracketed between nulls (A) and (B).
     Pooled: Wilcoxon signed-rank on (observed - null_A) > 0, and Fisher/Stouffer
     across per-cell p-values.
  4. Coupling vs geodesic distance (mixed model, primary; Spearman, descriptive)
  5. Event order: branch-first fraction (sign test, primary; pooled binomial, descriptive)
  6. Branch amplitude -> reaches soma (logistic) [skipped: needs per-event data]
  7. Behavior coupling
  8. Sensor overload / expression time
  9. Co-firing groups
 10. QC table

Changes from pass 1 (stats2):
  - test_03: replaced meaningless one-sample Wilcoxon vs 0 and wrong-direction
    per-cell binomial with proper dual-null surrogate test.
  - test_04: Spearman noted as descriptive (pseudo-replication); mixed model is primary.
  - test_05: pooled binomial noted as descriptive (pseudo-replication); sign test is primary.
  - Global FDR: only cohort-level p-values (one per test), not per-cell surrogates.
  - Ground-truth validation: independence test run on Daria's two curated cells.
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
from scipy.ndimage import uniform_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
AUTO_ROOT = PROJECT / "auto_pipeline"
sys.path.insert(0, str(PROJECT / "code"))
from common.run_marks import is_set_aside, load_marks   # noqa: E402  (run_marks.csv)
RESULTS = AUTO_ROOT / "results"
FIG_ROOT = RESULTS / "figures"
LOG_FILE = AUTO_ROOT / "logs" / "paper_stats.jsonl"
PYTHON = sys.executable

# Add project code to sys.path for imports
sys.path.insert(0, str(PROJECT / "code"))
sys.path.insert(0, str(PROJECT / "code" / "STEP7_workflow"))
sys.path.insert(0, str(PROJECT / "code" / "STEP8_stats"))

# Real curated data (READ-ONLY)
CURATED = {
    "run05": {
        "run_dir": PROJECT / "rbp4_phpebach/06-26-2026/run05",
        "stem": "run05_clean",
        "mask": PROJECT / "rbp4_phpebach/06-26-2026/run05/run05_clean_autoseg_labelmap_reviewed.tif",
        "segments": PROJECT / "rbp4_phpebach/06-26-2026/run05/run05_clean_segments_final.tif",
        "segments_json": PROJECT / "rbp4_phpebach/06-26-2026/run05/run05_clean_segments_final.json",
        "metrics": PROJECT / "rbp4_phpebach/06-26-2026/run05/run05_clean_metrics.json",
        "stack": PROJECT / "rbp4_phpebach/06-26-2026/run05/run05_clean.tif",
    },
    "run03": {
        "run_dir": PROJECT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03",
        "stem": "run03_clean",
        "mask": PROJECT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_autoseg_labelmap_reviewed.tif",
        "segments": PROJECT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_segments_final.tif",
        "segments_json": PROJECT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_segments_final.json",
        "metrics": PROJECT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_metrics.json",
        "stack": PROJECT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean.tif",
    },
}
AUTO = {
    "run05": {
        "run_dir": AUTO_ROOT / "rbp4_phpebach/06-26-2026/run05",
        "stem": "run05_clean",
        "mask": AUTO_ROOT / "rbp4_phpebach/06-26-2026/run05/run05_clean_autoseg_labelmap_reviewed.tif",
        "segments": AUTO_ROOT / "rbp4_phpebach/06-26-2026/run05/run05_clean_segments_final.tif",
        "segments_json": AUTO_ROOT / "rbp4_phpebach/06-26-2026/run05/run05_clean_segments_final.json",
        "metrics": AUTO_ROOT / "rbp4_phpebach/06-26-2026/run05/run05_clean_metrics.json",
        "stack": AUTO_ROOT / "rbp4_phpebach/06-26-2026/run05/run05_clean.tif",
    },
    "run03": {
        "run_dir": AUTO_ROOT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03",
        "stem": "run03_clean",
        "mask": AUTO_ROOT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_autoseg_labelmap_reviewed.tif",
        "segments": AUTO_ROOT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_segments_final.tif",
        "segments_json": AUTO_ROOT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_segments_final.json",
        "metrics": AUTO_ROOT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean_metrics.json",
        "stack": AUTO_ROOT / "rbp4_140_phpeb/06-18-2026/preprocessed/run03/run03_clean.tif",
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
# Event detection (same as run_metrics.py — import-compatible)
# ==============================================================================

def dff(t, f0_pct=10.0):
    """ΔF/F from raw trace."""
    f0 = np.percentile(t, f0_pct)
    return (t - f0) / max(f0, 1e-6)


def events(t, prom_frac=0.2, min_dist=5):
    """Prominence-based peak detection on dF/F, matching run_metrics.py."""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(t, prominence=prom_frac * (t.max() - t.min()), distance=min_dist)
    return pk


def near(a, b, window=2):
    """For each event in `a`, is any event in `b` within ±window frames?"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    return np.array([np.any(np.abs(b - x) <= window) for x in a], bool)


def frac_independent(ref_ev, branch_ev, window=2):
    """Fraction of branch events with no reference event within ±window."""
    if len(branch_ev) == 0:
        return float("nan")
    return float(1 - near(branch_ev, ref_ev, window).mean())


# ==============================================================================
# Surrogate null generators for the independence test
# ==============================================================================

def phase_randomize(x, rng):
    """Phase-randomize a real-valued time series, preserving power spectrum and
    autocorrelation structure. Returns a real array of the same length."""
    n = len(x)
    ft = np.fft.rfft(x)
    phases = rng.uniform(0, 2 * np.pi, len(ft))
    # keep DC and (if n even) Nyquist real
    phases[0] = 0
    if n % 2 == 0:
        phases[-1] = 0
    ft_rand = ft * np.exp(1j * phases)
    return np.fft.irfft(ft_rand, n=n)


def coupled_noise_null(ref_dff, branch_dff, window=2, prom_frac=0.2,
                       n_draws=500, seed=42):
    """NULL A: What fraction of 'branch-independent' events would you see from a
    perfectly coupled branch (same signal as the reference) due to detection noise
    alone?

    Model: synthetic_branch = gain * ref + phase_randomized(residual)
    where gain = least-squares fit, residual = branch - gain*ref, and the
    phase-randomized residual preserves the branch's noise level + autocorrelation.

    Returns: array of null frac_independent values (length n_draws).
    """
    rng = np.random.default_rng(seed)
    # Least-squares gain: branch ≈ gain * ref + offset
    # Use centered traces (subtract mean) so the gain is a pure scaling
    ref_c = ref_dff - ref_dff.mean()
    br_c = branch_dff - branch_dff.mean()
    gain = np.dot(ref_c, br_c) / (np.dot(ref_c, ref_c) + 1e-12)
    residual = br_c - gain * ref_c

    ref_ev = events(ref_dff, prom_frac)
    null_fracs = np.empty(n_draws)

    for i in range(n_draws):
        noise = phase_randomize(residual, rng)
        synth = gain * ref_c + noise + branch_dff.mean()
        synth_dff = synth  # already in dF/F space
        synth_ev = events(synth_dff, prom_frac)
        null_fracs[i] = frac_independent(ref_ev, synth_ev, window)

    return null_fracs


def independent_null(ref_dff, branch_dff, window=2, prom_frac=0.2,
                     n_draws=500, min_shift_s=20.0, frame_rate_hz=5.0,
                     seed=43):
    """NULL B: What fraction of 'branch-independent' events would you see if
    the branch were completely unrelated to the reference?

    Method: circular shifts of the branch trace by >min_shift_s seconds.
    The branch's own event statistics (rate, shape, noise) are perfectly preserved;
    only its temporal relationship to the reference is destroyed.

    Returns: array of null frac_independent values (length n_draws).
    """
    rng = np.random.default_rng(seed)
    T = len(branch_dff)
    min_shift = max(1, int(min_shift_s * frame_rate_hz))
    max_shift = T - min_shift
    if max_shift <= min_shift:
        # Recording too short for meaningful shifts
        return np.full(n_draws, np.nan)

    ref_ev = events(ref_dff, prom_frac)
    null_fracs = np.empty(n_draws)

    for i in range(n_draws):
        shift = rng.integers(min_shift, max_shift)
        shifted = np.roll(branch_dff, shift)
        shifted_ev = events(shifted, prom_frac)
        null_fracs[i] = frac_independent(ref_ev, shifted_ev, window)

    return null_fracs


def _pooled_events(traces, prom_frac):
    ev = []
    for tr in traces:
        ev.extend(events(tr, prom_frac).tolist())
    return np.unique(np.array(ev, int))


def coupled_noise_null_pooled(ref_dff, branch_dffs, branch_noises, window=2, prom_frac=0.2,
                              n_draws=500, seed=42):
    """NULL A (v3): every branch perfectly coupled to the reference.

    synthetic_branch_i = gain_i * ref + phase_randomized(noise_i), where noise_i is the
    branch's own MEASUREMENT noise from a within-region voxel split, (half1 - half2)/2:
    the signal cancels and only noise (with its real level and autocorrelation) remains.
    v2 used residual = branch - gain*ref, which still contains the branch's genuine
    independent events, so the null re-created them and was inflated (0.64-0.77).
    Events are detected per branch and pooled exactly as the observed statistic."""
    rng = np.random.default_rng(seed)
    ref_c = ref_dff - ref_dff.mean()
    gains = [float(np.dot(ref_c, b - b.mean()) / (np.dot(ref_c, ref_c) + 1e-12)) for b in branch_dffs]
    ref_ev = events(ref_dff, prom_frac)
    out = np.empty(n_draws)
    for i in range(n_draws):
        synth = [g * ref_c + phase_randomize(nz - nz.mean(), rng) + b.mean()
                 for g, nz, b in zip(gains, branch_noises, branch_dffs)]
        out[i] = frac_independent(ref_ev, _pooled_events(synth, prom_frac), window)
    return out


def independent_null_pooled(ref_dff, branch_dffs, window=2, prom_frac=0.2, n_draws=500,
                            min_shift_s=20.0, frame_rate_hz=5.0, seed=43):
    """NULL B (v3): all branches circularly shifted together by >min_shift_s (keeps their
    mutual relation and own statistics, destroys the relation to the reference)."""
    rng = np.random.default_rng(seed)
    T = len(ref_dff); lo = max(1, int(min_shift_s * frame_rate_hz)); hi = T - lo
    if hi <= lo:
        return np.full(n_draws, np.nan)
    ref_ev = events(ref_dff, prom_frac)
    out = np.empty(n_draws)
    for i in range(n_draws):
        k = int(rng.integers(lo, hi))
        out[i] = frac_independent(ref_ev, _pooled_events([np.roll(b, k) for b in branch_dffs], prom_frac), window)
    return out


def independence_test_one_cell(ref_dff, branch_dffs, window=2, prom_frac=0.2,
                               n_draws=500, frame_rate_hz=5.0, seed=42, branch_noises=None):
    """Run the dual-null independence test for one cell.

    Args:
        ref_dff: reference (soma/proximal trunk) dF/F trace, shape (T,)
        branch_dffs: list of branch dF/F traces, each shape (T,)
        window, prom_frac: event detector params (same as run_metrics.py)
        n_draws: surrogate draws per null
        frame_rate_hz: for min_shift_s conversion

    Returns dict with:
        observed: observed frac_independent (averaged over branches)
        null_coupled_mean/std: mean/std of coupled-noise null
        null_independent_mean/std: mean/std of independent null
        p_vs_coupled: one-sided p (observed > coupled null)
        p_vs_independent: one-sided p (observed < independent null)
        n_branch_events: total branch events
        n_ref_events: reference events
    """
    ref_ev = events(ref_dff, prom_frac)

    # Pool branch events across branches (same as run_metrics.py)
    all_br_ev = []
    for br in branch_dffs:
        all_br_ev.extend(events(br, prom_frac).tolist())
    all_br_ev = np.unique(np.array(all_br_ev, int))

    observed = frac_independent(ref_ev, all_br_ev, window)
    n_branch = len(all_br_ev)
    n_ref = len(ref_ev)

    if n_branch < 3:
        return {"observed": observed, "n_branch_events": n_branch, "n_ref_events": n_ref,
                "null_coupled_mean": float("nan"), "null_coupled_std": float("nan"),
                "null_independent_mean": float("nan"), "null_independent_std": float("nan"),
                "p_vs_coupled": float("nan"), "p_vs_independent": float("nan"),
                "skip_reason": "too few branch events"}

    if branch_noises is None:
        raise ValueError("branch_noises (within-region split-half noise) required since v3")
    null_A = coupled_noise_null_pooled(ref_dff, branch_dffs, branch_noises, window, prom_frac, n_draws, seed)
    null_B = independent_null_pooled(ref_dff, branch_dffs, window, prom_frac, n_draws,
                                     min_shift_s=20.0, frame_rate_hz=frame_rate_hz, seed=seed + 1)

    # p-values: how extreme is the observed value relative to each null?
    # vs coupled: observed should be > coupled null (real independent events beyond noise)
    # permutation p-values with the +1 correction (never exactly 0 or 1)
    _a = null_A[np.isfinite(null_A)]
    p_coupled = float((np.sum(_a >= observed) + 1) / (len(_a) + 1)) if len(_a) else float("nan")
    # vs independent: observed should be < independent null (real coupling)
    _b = null_B[np.isfinite(null_B)]
    p_indep = float((np.sum(_b <= observed) + 1) / (len(_b) + 1)) if len(_b) else float("nan")

    return {
        "observed": float(observed),
        "n_branch_events": int(n_branch),
        "n_ref_events": int(n_ref),
        "null_coupled_mean": float(np.nanmean(null_A)),
        "null_coupled_std": float(np.nanstd(null_A)),
        "null_coupled_dist": null_A,  # for figure
        "null_independent_mean": float(np.nanmean(null_B)),
        "null_independent_std": float(np.nanstd(null_B)),
        "null_independent_dist": null_B,  # for figure
        "p_vs_coupled": p_coupled,
        "p_vs_independent": p_indep,
    }


def independence_test_from_files(run_dir, stem, seg_json_path, window=2,
                                  prom_frac=0.2, n_draws=500):
    """Run the independence test for one run, loading data from disk.

    Args:
        run_dir: path to the run directory
        stem: e.g. 'run05_clean'
        seg_json_path: path to _segments_final.json
        window, prom_frac: event detection params
        n_draws: surrogate draws

    Returns: dict from independence_test_one_cell, or None if missing data.
    """
    import tifffile
    stack_p = Path(run_dir) / f"{stem}.tif"
    seg_p = Path(run_dir) / f"{stem}_segments_final.tif"
    if not stack_p.exists() or not seg_p.exists():
        return None

    stack = tifffile.imread(str(stack_p))
    seg = tifffile.imread(str(seg_p))
    T = stack.shape[0]
    flat = stack.reshape(T, -1)

    # Load segment names
    seg_json = json.loads(Path(seg_json_path).read_text()) if Path(seg_json_path).exists() else {}
    segment_names = seg_json.get("segment_names", {})

    labels = sorted(int(v) for v in np.unique(seg) if v > 0)
    sys.path.insert(0, str(PROJECT / "code"))
    from common.regions import apply_ignore
    _nm = {l: segment_names.get(str(l), f"seg{l}") for l in labels}
    seg, _ignored, _ign = apply_ignore(seg, _nm, seg_p)          # <stem>_ignore.json
    labels = [l for l in labels if l not in _ign]
    if not labels:
        return None

    # Classify compartments
    def compartment_of(name):
        n = name.lower()
        if n.startswith("soma"): return "soma"
        if n.startswith("trunk"): return "trunk"
        return "branch"

    comps = {}
    for l in labels:
        name = segment_names.get(str(l), f"seg{l}")
        comps[l] = compartment_of(name)

    # Traces
    def trace(mask):
        idx = np.flatnonzero(mask.ravel())
        return dff(flat[:, idx].mean(1).astype(np.float64))

    tr = {l: trace(seg == l) for l in labels}

    # Find reference and branches
    by = {c: [l for l in labels if comps[l] == c] for c in ("soma", "trunk", "branch")}
    if not by["branch"]:
        return None

    if by["soma"]:
        ref_label = by["soma"][0]
    elif by["trunk"]:
        ref_label = by["trunk"][0]  # first trunk = most proximal
    else:
        return None

    ref_dff = tr[ref_label]
    branch_dffs = [tr[l] for l in by["branch"]]

    def split_noise(mask):
        idx = np.flatnonzero(mask.ravel())
        h1, h2 = idx[0::2], idx[1::2]
        if len(h2) == 0:
            return np.zeros(T)
        return (dff(flat[:, h1].mean(1).astype(np.float64)) - dff(flat[:, h2].mean(1).astype(np.float64))) / 2.0
    branch_noises = [split_noise(seg == l) for l in by["branch"]]

    # Get frame rate from metrics if available
    met_p = Path(run_dir) / f"{stem}_metrics.json"
    rate = 5.0
    if met_p.exists():
        met = json.loads(met_p.read_text())
        rate = float(met.get("frame_rate_hz", 5.0))

    result = independence_test_one_cell(ref_dff, branch_dffs, window, prom_frac,
                                         n_draws, rate, seed=42, branch_noises=branch_noises)
    result["reference"] = segment_names.get(str(ref_label), f"seg{ref_label}")
    result["branches"] = [segment_names.get(str(l), f"seg{l}") for l in by["branch"]]
    return result


def stouffer_combine(pvals):
    """Stouffer's method to combine p-values (weighted equally).
    Returns combined z and two-sided p."""
    pvals = np.clip(np.array([p for p in pvals if np.isfinite(p)], float), 1.0 / 501, 1 - 1.0 / 501)  # keep every cell; permutation resolution (500 draws)
    if len(pvals) == 0:
        return float("nan"), float("nan")
    # Clip to avoid inf
    pvals = np.clip(pvals, 1e-15, 1 - 1e-15)
    zs = sp.norm.ppf(1 - pvals)  # one-sided p -> z
    z_combined = np.sum(zs) / np.sqrt(len(zs))
    p_combined = float(1 - sp.norm.cdf(z_combined))
    return float(z_combined), p_combined


def fisher_combine(pvals):
    """Fisher's method to combine p-values."""
    pvals = np.array([p for p in pvals if np.isfinite(p) and 0 < p <= 1], float)
    if len(pvals) == 0:
        return float("nan"), float("nan")
    pvals = np.clip(pvals, 1e-300, 1.0)
    chi2 = -2 * np.sum(np.log(pvals))
    p_combined = float(sp.chi2.sf(chi2, 2 * len(pvals)))
    return float(chi2), p_combined


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
        if "note" in m or is_set_aside(m.get("behavior_base", "")):
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
        if is_set_aside(m.get("behavior_base", "")):
            continue
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
        if is_set_aside(bc.get("behavior_base", "")):
            continue
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
        if "/old/" in f or f.endswith("_behavior_coupling.json"):   # different file kind, same suffix
            continue
        c = json.load(open(f))
        if is_set_aside(c.get("behavior_base", "")):
            continue
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
        _mf = Path(mask_json).with_name(Path(mask_json).name.replace("_autoseg_reviewed.json", "_metrics.json"))
        if _mf.exists() and is_set_aside(json.load(open(_mf)).get("behavior_base", "")):
            continue
        run_dir = Path(mask_json).parent
        stem = Path(mask_json).stem.replace("_autoseg_reviewed", "")

        seg_json = run_dir / f"{stem}_segments_final.json"
        met_json = run_dir / f"{stem}_metrics.json"
        seg = json.load(open(seg_json)) if seg_json.exists() else {}
        met = json.load(open(met_json)) if met_json.exists() else {}

        ar = seg.get("auto_regions", {})
        n_regions = len(ar.get("regions", {}))
        region_names = list(ar.get("regions", {}).keys())

        has_soma = ar.get("tree_info", {}).get("has_soma", False)
        has_bif = any("bifurcation" in str(n).lower() for n in region_names)
        # Determine branch regions from region names (branch/branch1/branch2/etc.)
        # Also check metrics JSON compartment info
        n_branch_regions = sum(1 for n in region_names if "branch" in n.lower())
        if n_branch_regions == 0:
            # Fallback to metrics regions compartment
            for rk, rv in met.get("regions", {}).items():
                if rv.get("compartment") == "branch":
                    n_branch_regions += 1
        n_intruders = len(rev.get("intruders", []))

        # Check behavior data availability
        session = run_dir
        while session.name.startswith("run") or session.name in ("preprocessed", "traces"):
            session = session.parent
        has_behavior = (session / "behavior").exists()
        bc_json = run_dir / f"{stem}_behavior_coupling.json"
        has_bc = bc_json.exists()

        # Reliability check
        low_rel = False
        for rname, rdata in ar.get("regions", {}).items():
            if isinstance(rdata, dict) and rdata.get("reliability", 1.0) < 0.90:
                low_rel = True

        # Short recording
        T = met.get("T", 0)

        # Build flags
        flags = list(rev.get("flags", []))
        if not has_soma:
            flags.append("no_soma")
        if not has_bif and n_branch_regions > 0:
            flags.append("no_bifurcation")
        if n_intruders > 0:
            flags.append(f"{n_intruders}_suspects")
        if low_rel:
            flags.append("low_reliability")
        if T > 0 and T < 500:
            flags.append("short_recording")
        if not has_behavior:
            flags.append("no_behavior")
        if n_branch_regions == 0:
            flags.append("no_branches")

        # Ground-truth marker
        is_gt = False
        rd_str = str(run_dir)
        if "rbp4_phpebach/06-26-2026/run05" in rd_str or "rbp4_140_phpeb/06-18-2026/preprocessed/run03" in rd_str:
            is_gt = True

        row = {
            "run_dir": str(run_dir.relative_to(AUTO_ROOT)),
            "behavior_base": met.get("behavior_base", ""),
            "mouse": met.get("mouse", ""),
            "mask_voxels": rev.get("mask_voxels", 0),
            "mask_components": rev.get("n_components", 0),
            "mask_alpha": rev.get("params", {}).get("alpha", 0),
            "n_intruders": n_intruders,
            "intruder_reasons": [i.get("reasons", []) for i in rev.get("intruders", [])],
            "chunk_period": rev.get("chunk_period_px", 0),
            "confidence": rev.get("confidence", ""),
            "flags": flags,
            "n_regions": n_regions,
            "region_names": region_names,
            "reference": ar.get("reference", met.get("reference", "")),
            "has_soma": has_soma,
            "has_bifurcation": has_bif,
            "n_branch_regions": n_branch_regions,
            "has_behavior": has_behavior,
            "T": T,
            "frame_rate_hz": met.get("frame_rate_hz", 0),
            "n_soma_events": met.get("n_soma_events", 0),
            "n_branch_events": met.get("n_branch_events", 0),
            "is_ground_truth": is_gt,
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
            lines.append("| Metric | Manual | Auto | Δ |")
            lines.append("|--------|--------|------|---|")
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
    for i, (_, row) in enumerate(both.iterrows()):
        c = "C0" if row.get("reference") == "soma" else "C1"
        ax1.plot([0, 1], [row.r_soma_trunk, row.r_soma_branch], "-o", color=c, alpha=0.5, ms=4)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Trunk", "Branch"])
    ax1.set_ylabel("r with reference")
    ax1.set_title(f"Paired coupling (n={n})")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

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
    """Independent branch events: dual-null surrogate test.

    For each cell, we bracket the observed frac_branch_independent between:
      (A) coupled-noise null — what a perfectly coupled branch gives
      (B) independent null — what a completely unrelated branch gives

    The claim 'branches have their own events' requires:
      observed > coupled-noise null (real independent events beyond detection noise)
      observed < independent null (real coupling remains)
    """
    results = {}
    lines = ["# 3. Independent Branch Events: Dual-Null Surrogate Test", ""]
    figs = []
    all_p = []

    lines.append("For each cell, 500 surrogates per null are generated from the raw traces:")
    lines.append("  (A) COUPLED-NOISE NULL: each branch = gain × reference + phase-randomized measurement noise from a within-region voxel split (half1-half2)/2.")
    lines.append("      Tests whether observed independence exceeds what noise alone produces.")
    lines.append("  (B) INDEPENDENT NULL: all branch traces circularly shifted together (>20 s).")
    lines.append("      Tests whether observed coupling is above chance (i.e., independence < full independence).")
    lines.append("")

    # Find all runs that have segments + stacks
    cell_results = []
    for _, row in d.iterrows():
        met_file = AUTO_ROOT / row["metrics_file"]
        run_dir = met_file.parent
        stem = met_file.stem.replace("_metrics", "")
        seg_json_p = run_dir / f"{stem}_segments_final.json"

        if not seg_json_p.exists():
            continue
        if row.get("n_branch_events", 0) < 3:
            continue

        print(f"    Independence test: {row.get('behavior_base', '')}...")
        try:
            cr = independence_test_from_files(
                run_dir, stem, seg_json_p,
                window=2, prom_frac=0.2, n_draws=500
            )
        except Exception as e:
            print(f"      ERROR: {e}")
            continue

        if cr is None or "skip_reason" in cr:
            continue

        cr["behavior_base"] = row.get("behavior_base", "")
        cr["mouse"] = row.get("mouse", "")
        cell_results.append(cr)

    n = len(cell_results)
    results["n_cells_independence"] = n
    lines.append(f"n = {n} cells with sufficient events")
    lines.append("")

    if n == 0:
        lines.append("SKIP: no cells with enough branch events for surrogate test.")
        return results, lines, figs

    # Collect per-cell numbers
    obs = np.array([c["observed"] for c in cell_results])
    null_A_means = np.array([c["null_coupled_mean"] for c in cell_results])
    null_B_means = np.array([c["null_independent_mean"] for c in cell_results])
    p_coupled = np.array([c["p_vs_coupled"] for c in cell_results])
    p_indep = np.array([c["p_vs_independent"] for c in cell_results])

    # Descriptive stats
    obs_mean, obs_lo, obs_hi = ci_bootstrap(obs)
    results["frac_branch_independent_mean"] = round(obs_mean, 4)
    results["frac_branch_independent_ci95"] = [round(obs_lo, 4), round(obs_hi, 4)]
    results["null_coupled_mean"] = round(float(np.nanmean(null_A_means)), 4)
    results["null_independent_mean"] = round(float(np.nanmean(null_B_means)), 4)

    lines.append(f"Observed frac_independent: mean={obs_mean:.3f}, 95% CI [{obs_lo:.3f}, {obs_hi:.3f}]")
    lines.append(f"Coupled-noise null mean:   {np.nanmean(null_A_means):.3f}")
    lines.append(f"Independent null mean:     {np.nanmean(null_B_means):.3f}")
    lines.append("")

    # Test A: observed > coupled null (one per cell)
    lines.append("## Test A: Observed > Coupled-Noise Null")
    lines.append("  (Real independent events beyond what detection noise produces)")
    excess_A = obs - null_A_means  # should be > 0 if branches have real independent events
    valid_A = np.isfinite(excess_A)
    if valid_A.sum() >= 6:
        w_A = sp.wilcoxon(excess_A[valid_A], alternative="greater")
        results["test_A_wilcoxon_p"] = round(float(w_A.pvalue), 6)
        results["test_A_wilcoxon_stat"] = round(float(w_A.statistic), 4)
        lines.append(f"  Wilcoxon signed-rank (observed - null_A > 0): W={w_A.statistic:.1f}, p={w_A.pvalue:.4g}")
        all_p.append(float(w_A.pvalue))
    elif valid_A.sum() >= 2:
        w_A = sp.wilcoxon(excess_A[valid_A], alternative="greater")
        results["test_A_wilcoxon_p"] = round(float(w_A.pvalue), 6)
        lines.append(f"  Wilcoxon (n={valid_A.sum()}, descriptive): p={w_A.pvalue:.4g}")
        all_p.append(float(w_A.pvalue))

    n_above_A = int((excess_A > 0).sum())
    results["n_above_coupled_null"] = n_above_A
    lines.append(f"  Observed > coupled null: {n_above_A}/{n} cells")

    # Stouffer combination of per-cell p_vs_coupled
    valid_pA = [float(p) for p in p_coupled if np.isfinite(p)]
    if valid_pA:
        z_stouffer, p_stouffer = stouffer_combine(valid_pA)
        _, p_fisher = fisher_combine(valid_pA)
        results["test_A_stouffer_p"] = round(p_stouffer, 6)
        results["test_A_fisher_p"] = round(p_fisher, 6)
        lines.append(f"  Stouffer combined p: {p_stouffer:.4g} (z={z_stouffer:.2f})")
        lines.append(f"  Fisher combined p:   {p_fisher:.4g}")

    lines.append("")

    # Test B: observed < independent null (coupling exists)
    lines.append("## Test B: Observed < Independent Null")
    lines.append("  (Real coupling: independence is less than what fully unrelated traces give)")
    deficit_B = null_B_means - obs  # should be > 0 if real coupling exists
    valid_B = np.isfinite(deficit_B)
    if valid_B.sum() >= 6:
        w_B = sp.wilcoxon(deficit_B[valid_B], alternative="greater")
        results["test_B_wilcoxon_p"] = round(float(w_B.pvalue), 6)
        results["test_B_wilcoxon_stat"] = round(float(w_B.statistic), 4)
        lines.append(f"  Wilcoxon signed-rank (null_B - observed > 0): W={w_B.statistic:.1f}, p={w_B.pvalue:.4g}")
        all_p.append(float(w_B.pvalue))
    elif valid_B.sum() >= 2:
        w_B = sp.wilcoxon(deficit_B[valid_B], alternative="greater")
        results["test_B_wilcoxon_p"] = round(float(w_B.pvalue), 6)
        lines.append(f"  Wilcoxon (n={valid_B.sum()}, descriptive): p={w_B.pvalue:.4g}")
        all_p.append(float(w_B.pvalue))

    n_below_B = int((obs < null_B_means).sum())
    results["n_below_independent_null"] = n_below_B
    lines.append(f"  Observed < independent null: {n_below_B}/{n} cells")

    valid_pB = [float(p) for p in p_indep if np.isfinite(p)]
    if valid_pB:
        z_stouffer_B, p_stouffer_B = stouffer_combine(valid_pB)
        _, p_fisher_B = fisher_combine(valid_pB)
        results["test_B_stouffer_p"] = round(p_stouffer_B, 6)
        results["test_B_fisher_p"] = round(p_fisher_B, 6)
        lines.append(f"  Stouffer combined p: {p_stouffer_B:.4g} (z={z_stouffer_B:.2f})")
        lines.append(f"  Fisher combined p:   {p_fisher_B:.4g}")

    lines.append("")

    # Per-cell table
    lines.append("### Per-Cell Results")
    lines.append("")
    lines.append("| Cell | Observed | Null_A (coupled) | Null_B (independent) | p(>A) | p(<B) | n_branch | n_ref |")
    lines.append("|------|----------|------------------|---------------------|-------|-------|----------|-------|")
    for cr in cell_results:
        lines.append(f"| {cr['behavior_base'][:40]} | {cr['observed']:.3f} | "
                     f"{cr['null_coupled_mean']:.3f}±{cr['null_coupled_std']:.3f} | "
                     f"{cr['null_independent_mean']:.3f}±{cr['null_independent_std']:.3f} | "
                     f"{cr['p_vs_coupled']:.3f} | {cr['p_vs_independent']:.3f} | "
                     f"{cr['n_branch_events']} | {cr['n_ref_events']} |")
    lines.append("")

    # Per-cell FDR
    if valid_pA:
        q_A = bh_fdr(valid_pA)
        n_sig_A = sum(1 for q in q_A if q < 0.05)
        results["test_A_n_sig_fdr05"] = n_sig_A
        lines.append(f"Per-cell FDR (test A, obs > coupled): {n_sig_A}/{len(valid_pA)} significant at q<0.05")
    if valid_pB:
        q_B = bh_fdr(valid_pB)
        n_sig_B = sum(1 for q in q_B if q < 0.05)
        results["test_B_n_sig_fdr05"] = n_sig_B
        lines.append(f"Per-cell FDR (test B, obs < independent): {n_sig_B}/{len(valid_pB)} significant at q<0.05")
    lines.append("")

    # Interpretation
    lines.append("### Interpretation")
    if n_above_A > n / 2:
        lines.append(f"  Majority of cells ({n_above_A}/{n}) show more independent branch events than")
        lines.append("  the coupled-noise null: real independent events exist beyond detection noise.")
    else:
        lines.append(f"  Only {n_above_A}/{n} cells exceed the coupled-noise null: the claim of")
        lines.append("  independent branch events is not well supported at the individual cell level.")
    if n_below_B > n / 2:
        lines.append(f"  Majority of cells ({n_below_B}/{n}) show less independence than the")
        lines.append("  independent null: real coupling between branch and reference exists.")
    lines.append("")

    # Overfitting caveat (n=2 ground truth, same params for all)
    lines.append("CAVEAT: Same event detection parameters (window=2, prom_frac=0.2) are used for all cells.")
    lines.append("Ground truth for the claim is limited to n=2 curated cells. Report overfitting risk honestly.")
    lines.append("")

    # Store per-cell results (without large distributions) for RESULTS.json
    results["per_cell"] = [
        {k: v for k, v in cr.items() if k not in ("null_coupled_dist", "null_independent_dist")}
        for cr in cell_results
    ]

    # Figure: observed vs both nulls per cell
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Per-cell observed vs null means
    ax = axes[0]
    x = np.arange(n)
    # Sort by observed
    order = np.argsort(obs)[::-1]
    ax.bar(x - 0.25, obs[order], 0.25, label="Observed", color="teal", alpha=0.8)
    ax.bar(x, null_A_means[order], 0.25, label="Coupled null (A)", color="salmon", alpha=0.7)
    ax.bar(x + 0.25, null_B_means[order], 0.25, label="Independent null (B)", color="lightblue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([cell_results[i]["behavior_base"][-12:] for i in order],
                       rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Fraction branch events independent")
    ax.set_title(f"Per-cell independence (n={n})")
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: Summary violin/box comparing observed and both nulls
    ax = axes[1]
    # Collect all null distributions for a combined violin
    all_null_A = np.concatenate([cr.get("null_coupled_dist", []) for cr in cell_results
                                 if "null_coupled_dist" in cr and cr["null_coupled_dist"] is not None])
    all_null_B = np.concatenate([cr.get("null_independent_dist", []) for cr in cell_results
                                 if "null_independent_dist" in cr and cr["null_independent_dist"] is not None])

    data_to_plot = []
    labels_plot = []
    if len(all_null_A) > 0:
        data_to_plot.append(all_null_A[np.isfinite(all_null_A)])
        labels_plot.append(f"Coupled null\n(n={n}×500)")
    data_to_plot.append(obs)
    labels_plot.append(f"Observed\n(n={n})")
    if len(all_null_B) > 0:
        data_to_plot.append(all_null_B[np.isfinite(all_null_B)])
        labels_plot.append(f"Independent null\n(n={n}×500)")

    parts = ax.violinplot(data_to_plot, showmedians=True, showextrema=False)
    colors = ["salmon", "teal", "lightblue"]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[min(i, len(colors) - 1)])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels_plot) + 1))
    ax.set_xticklabels(labels_plot, fontsize=8)
    ax.set_ylabel("Fraction branch events independent")
    ax.set_title("Observed bracketed between nulls")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Branch Independence: Dual-Null Surrogate Test", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "02_independence", "branch_independence")
    figs.append(("02_independence/branch_independence", "Dual-null independence test"))

    results["_p_values_03"] = all_p
    return results, lines, figs


def test_03_ground_truth() -> tuple[dict, list[str], list]:
    """Run the independence test on Daria's two curated cells (read-only, in memory)."""
    results = {}
    lines = ["# 3b. Ground-Truth Independence Test (Daria's Curated Cells)", ""]
    figs = []

    lines.append("Running the same dual-null independence test on Daria's hand-curated")
    lines.append("segments+stacks. These are the ground truth for the claim 'branches have")
    lines.append("their own events'. All computation in memory; no writes to real tree.")
    lines.append("")

    for name, info in CURATED.items():
        if not info["stack"].exists() or not info["segments"].exists():
            lines.append(f"  {name}: SKIP (files missing)")
            continue

        print(f"    Ground-truth independence: {name}...")
        cr = independence_test_from_files(
            info["run_dir"], info["stem"], info["segments_json"],
            window=2, prom_frac=0.2, n_draws=500
        )
        if cr is None:
            lines.append(f"  {name}: SKIP (no branches or reference)")
            continue

        lines.append(f"### {name} ({'soma present' if name == 'run05' else 'no soma, proximal trunk ref'})")
        lines.append(f"  Reference: {cr.get('reference', '?')}")
        lines.append(f"  Branches: {cr.get('branches', [])}")
        lines.append(f"  n_branch_events = {cr['n_branch_events']}, n_ref_events = {cr['n_ref_events']}")
        lines.append(f"  Observed frac_independent = {cr['observed']:.3f}")
        lines.append(f"  Coupled-noise null:   {cr['null_coupled_mean']:.3f} ± {cr['null_coupled_std']:.3f}")
        lines.append(f"  Independent null:     {cr['null_independent_mean']:.3f} ± {cr['null_independent_std']:.3f}")
        lines.append(f"  p(observed > coupled null) = {cr['p_vs_coupled']:.4g}")
        lines.append(f"  p(observed < independent null) = {cr['p_vs_independent']:.4g}")

        bracketed = cr["observed"] > cr["null_coupled_mean"] and cr["observed"] < cr["null_independent_mean"]
        lines.append(f"  Bracketed: {'YES' if bracketed else 'NO'} "
                     f"(coupled < observed < independent: "
                     f"{cr['null_coupled_mean']:.3f} < {cr['observed']:.3f} < {cr['null_independent_mean']:.3f})")
        lines.append("")

        key = f"ground_truth_{name}"
        results[f"{key}_observed"] = round(cr["observed"], 4)
        results[f"{key}_null_coupled_mean"] = round(cr["null_coupled_mean"], 4)
        results[f"{key}_null_independent_mean"] = round(cr["null_independent_mean"], 4)
        results[f"{key}_p_vs_coupled"] = round(cr["p_vs_coupled"], 4)
        results[f"{key}_p_vs_independent"] = round(cr["p_vs_independent"], 4)
        results[f"{key}_bracketed"] = bracketed
        results[f"{key}_n_branch_events"] = cr["n_branch_events"]

        # Compare with Daria's metrics
        if info["metrics"].exists():
            met = json.loads(info["metrics"].read_text())
            lines.append(f"  Daria's frac_branch_independent (from _metrics.json): {met.get('frac_branch_independent', '?')}")
            lines.append(f"  → Our surrogate test confirms this is {'above' if cr['p_vs_coupled'] < 0.05 else 'not above'} "
                         f"detection noise (p={cr['p_vs_coupled']:.4g}) and "
                         f"{'below' if cr['p_vs_independent'] < 0.05 else 'not below'} full independence "
                         f"(p={cr['p_vs_independent']:.4g}).")
            lines.append("")

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

    # Pooled Spearman (DESCRIPTIVE — pseudo-replication: regions within a cell
    # are not independent observations)
    rho = sp.spearmanr(dd.distance_um, dd.r_with_soma_corr)
    results["distance_spearman_rho"] = round(float(rho.statistic), 4)
    results["distance_spearman_p"] = round(float(rho.pvalue), 6)
    # NOTE: not added to all_p because pseudo-replicated

    lines.append(f"n = {n_regions} regions from {n_cells} cells")
    lines.append(f"Pooled Spearman (descriptive, pseudo-replicated): rho={rho.statistic:.3f}, p={rho.pvalue:.4g}")
    lines.append("  NOTE: regions from the same cell are not independent; the mixed model below is the primary test.")

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

    # Mixed model with cell random effect (PRIMARY TEST)
    if n_cells >= 3 and n_regions >= 8:
        try:
            import statsmodels.formula.api as smf
            md = smf.mixedlm("r_with_soma_corr ~ distance_um", dd, groups=dd["behavior_base"]).fit(reml=False)
            results["distance_mixed_slope"] = round(float(md.params["distance_um"]), 6)
            results["distance_mixed_p"] = round(float(md.pvalues["distance_um"]), 6)
            lines.append(f"Mixed model (r ~ distance + (1|cell), PRIMARY): slope={md.params['distance_um']:.6f}/um, p={md.pvalues['distance_um']:.4g}")
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

    # Pooled binomial (DESCRIPTIVE — events within a cell are not independent)
    bt = sp.binomtest(total_first, total_events, 0.5, alternative="greater")
    results["event_order_binomial_p"] = round(float(bt.pvalue), 8)
    lines.append(f"n = {n} cells, {total_events} paired events")
    lines.append(f"Branch peaks first: {total_first}/{total_events} ({100*total_first/total_events:.1f}%)")
    lines.append(f"Pooled binomial (descriptive, pseudo-replicated): p={bt.pvalue:.4g}")
    lines.append("  NOTE: events from the same cell are not independent; the sign test below is the primary test.")
    # NOT added to all_p because pseudo-replicated

    # Per-cell sign test (PRIMARY TEST — one observation per cell)
    n_above = int((bf.branch_first_frac > 0.5).sum())
    if n >= 2:
        st = sp.binomtest(n_above, n, 0.5, alternative="greater")
        results["event_order_sign_p"] = round(float(st.pvalue), 6)
        lines.append(f"Per-cell sign test (PRIMARY): {n_above}/{n} cells with fraction > 0.5, p={st.pvalue:.4g}")
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

    per_signal_ps = []
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
        per_signal_ps.extend(vals[pk].tolist())

    if per_signal_ps:
        q_vals = bh_fdr(per_signal_ps)
        n_sig_fdr = sum(1 for q in q_vals if q < 0.05)
        results["behavior_n_sig_fdr05"] = n_sig_fdr
        results["behavior_n_total_tests"] = len(per_signal_ps)
        lines.append(f"\nAfter BH-FDR across all {len(per_signal_ps)} behavior tests: {n_sig_fdr} significant at q<0.05")

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

    lines.append("| Run | Mouse | Mask vox | Regions | Reference | Intruders | Soma events | Branch events | Flags |")
    lines.append("|-----|-------|----------|---------|-----------|-----------|-------------|---------------|-------|")
    for _, row in qc.iterrows():
        flags_str = ", ".join(row.get("flags", []))
        gt = " **[GT]**" if row.get("is_ground_truth", False) else ""
        lines.append(f"| {row.get('run_dir', '')[-35:]}{gt} | {row.get('mouse', '')} | "
                     f"{row.get('mask_voxels', 0)} | {row.get('n_regions', 0)} | "
                     f"{row.get('reference', '')} | {row.get('n_intruders', 0)} | "
                     f"{int(row.get('n_soma_events', 0))} | {int(row.get('n_branch_events', 0))} | "
                     f"{flags_str} |")
    lines.append("")

    low_conf = qc[qc.confidence == "auto"]
    results["n_low_confidence"] = len(low_conf)
    lines.append(f"All {n} runs have confidence='auto' (fully automatic, no manual review).")
    lines.append("")

    return results, lines, figs


# ==============================================================================
# Main
# ==============================================================================

def main():
    log("start", "paper_stats.py v2 (stats2: fixed independence test)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    for folder in ["01_validation", "02_independence", "03_distance", "04_event_order",
                    "05_behavior", "06_expression_time", "07_cofiring_groups"]:
        (FIG_ROOT / folder).mkdir(parents=True, exist_ok=True)

    # Archive old results
    for f in [RESULTS / "RESULTS.json", RESULTS / "REPORT.md"]:
        if f.exists():
            old_dir = RESULTS / "old"
            old_dir.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            f.rename(old_dir / f"{f.stem}_{ts}{f.suffix}")

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
    # Collect only COHORT-LEVEL p-values for global FDR (one per test, no per-cell)
    cohort_pvals = []
    cohort_pval_labels = []

    # Test 1: Validation
    print("Test 1: Validation...")
    r, l, f = test_01_validation()
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 2: Coupling
    print("Test 2: Coupling...")
    r, l, f = test_02_coupling(d)
    test2_ps = r.pop("_p_values_02", [])
    for p in test2_ps:
        cohort_pvals.append(p)
        cohort_pval_labels.append("test02_coupling")
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 3: Independence (dual-null surrogate)
    print("Test 3: Independence (dual-null surrogate)...")
    r, l, f = test_03_independence(d)
    test3_ps = r.pop("_p_values_03", [])
    for p in test3_ps:
        cohort_pvals.append(p)
        cohort_pval_labels.append("test03_independence")
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 3b: Ground truth independence
    print("Test 3b: Ground-truth independence on curated cells...")
    r, l, f = test_03_ground_truth()
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 4: Distance
    print("Test 4: Distance...")
    r, l, f = test_04_distance(dd)
    test4_ps = r.pop("_p_values_04", [])
    for p in test4_ps:
        cohort_pvals.append(p)
        cohort_pval_labels.append("test04_distance")
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 5: Event order
    print("Test 5: Event order...")
    r, l, f = test_05_event_order(d)
    test5_ps = r.pop("_p_values_05", [])
    for p in test5_ps:
        cohort_pvals.append(p)
        cohort_pval_labels.append("test05_event_order")
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
    test7_ps = r.pop("_p_values_07", [])
    for p in test7_ps:
        cohort_pvals.append(p)
        cohort_pval_labels.append("test07_behavior")
    all_results.update(r)
    all_lines.extend(l)
    all_figs.extend(f)

    # Test 8: Expression/time
    print("Test 8: Expression/time...")
    r, l, f = test_08_expression(d)
    test8_ps = r.pop("_p_values_08", [])
    for p in test8_ps:
        cohort_pvals.append(p)
        cohort_pval_labels.append("test08_expression")
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

    # Global FDR on cohort-level p-values only
    all_lines.append("---")
    all_lines.append("## Global BH-FDR Correction (Cohort-Level Tests Only)")
    all_lines.append("")
    all_lines.append("Only cohort-level p-values (one per test) are included in the global FDR.")
    all_lines.append("Per-cell surrogate p-values and pseudo-replicated (pooled) p-values are excluded.")
    all_lines.append("")
    if cohort_pvals:
        q_vals = bh_fdr(cohort_pvals)
        all_results["global_bh_fdr"] = {
            "n_tests": len(cohort_pvals),
            "n_sig_005": sum(1 for q in q_vals if q < 0.05),
            "tests": [
                {"label": lab, "p": round(p, 6), "q": round(q, 6)}
                for lab, p, q in zip(cohort_pval_labels, cohort_pvals, q_vals)
            ],
        }
        all_lines.append("| Test | p | q (BH) |")
        all_lines.append("|------|---|--------|")
        for lab, p, q in zip(cohort_pval_labels, cohort_pvals, q_vals):
            sig = "**" if q < 0.05 else ""
            all_lines.append(f"| {lab} | {p:.4g} | {q:.4g} {sig}|")
        all_lines.append("")
        all_lines.append(f"{len(cohort_pvals)} tests, {sum(1 for q in q_vals if q < 0.05)} significant at q<0.05")
    all_lines.append("")

    # Sensitivity analysis
    all_lines.append("---")
    all_lines.append("## Sensitivity Analysis")
    all_lines.append("All runs are automatic (confidence='auto'), so no high vs low confidence split is possible.")
    all_lines.append("The full cohort results above ARE the only analysis.")
    all_lines.append("")

    # Statistical notes
    all_lines.append("---")
    all_lines.append("## Statistical Notes")
    all_lines.append("")
    all_lines.append("### Changes from pass 1")
    all_lines.append("1. **Independence test (test 3)**: Replaced meaningless one-sample Wilcoxon vs 0")
    all_lines.append("   (trivially significant: any nonzero frac_independent passes) and wrong-direction")
    all_lines.append("   per-cell binomial (null rate = 1 - soma_coverage, which tested whether observed")
    all_lines.append("   independence was ABOVE chance, but the claim needs it above coupled noise and")
    all_lines.append("   below full independence). Now uses dual-null surrogates on raw traces.")
    all_lines.append("2. **Pseudo-replication**: Pooled Spearman (test 4) and pooled binomial (test 5)")
    all_lines.append("   treat observations from the same cell as independent. Both are now flagged as")
    all_lines.append("   descriptive; the mixed model (test 4) and sign test (test 5) are the primary tests.")
    all_lines.append("3. **Global FDR**: Now includes only cohort-level p-values, not per-cell surrogates.")
    all_lines.append("4. **Ground truth**: Independence test validated on Daria's two curated cells (test 3b).")
    all_lines.append("")
    all_lines.append("### Overfitting risk")
    all_lines.append("Same parameters (window=2 frames, prom_frac=0.2, n_draws=500) for all cells.")
    all_lines.append("Ground truth is n=2 curated cells. The surrogate test does not use ground truth")
    all_lines.append("to set parameters; parameters come from the event detector which was developed")
    all_lines.append("independently. However, the small ground-truth sample limits validation power.")
    all_lines.append("")

    # Save RESULTS.json
    all_results["generated"] = datetime.datetime.now().astimezone().isoformat()
    all_results["n_runs"] = len(d)
    all_results["n_mice"] = int(d.mouse.nunique())
    all_results["mice"] = sorted(d.mouse.unique().tolist())
    all_results["stats_version"] = "2.0 (stats2: dual-null surrogate independence test)"

    results_path = RESULTS / "RESULTS.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {results_path}")

    # Save REPORT.md
    header = [
        "# Automatic Pipeline Statistical Report (v3)",
        "",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"n = {len(d)} cells from {int(d.mouse.nunique())} mice",
        "All masks and regions fully automatic (no manual curation).",
        *([f"Set aside by Daria (run_marks.csv), not in any statistic: "
           + "; ".join(f"{k} ({v['mark']}{': ' + v['reason'] if v.get('reason') else ''})" for k, v in sorted(load_marks().items()))]
          if load_marks() else []),
        "",
        "**Key change from v1**: Independence test now uses dual-null surrogates",
        "(coupled-noise null and circular-shift independent null) instead of the",
        "trivially-significant Wilcoxon-vs-0 and wrong-direction binomial.",
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

    log("done", f"{len(d)} runs, {len(all_figs)} figures, {len(cohort_pvals)} cohort p-values (FDR)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
