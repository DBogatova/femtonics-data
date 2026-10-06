#!/usr/bin/env python
"""sensor_assumptions.py - test known GCaMP7s / L5 pyramidal-neuron assumptions on our data.

Every test below is PRE-SPECIFIED (hypothesis, statistic and predicted direction are fixed
in TESTS before any number is computed) and gets a plain-language verdict.

Data: the raw cleaned stack <stem>_clean.tif and the saved regions (<stem>_segments_final.tif,
names from .json, <stem>_ignore.json honored), exactly like run_metrics.py. Nothing is ever
written next to the data: all outputs go to stats/assumptions/sensor/ (curated set) and
stats/assumptions/sensor/screening/ (automatic mirror, secondary).

Run sets
  curated    the runs listed in stats/cohort_metrics.csv (cohort_stats.py output)
  screening  the runs listed in auto_pipeline/stats/cohort_metrics.csv (automatic masks and
             regions), minus anything set aside in run_marks.csv
Inference unit = mouse: per-run statistics are averaged within a mouse, then an exact
Wilcoxon signed-rank test runs on the per-mouse means (5 mice -> smallest one-sided p is
0.031, two-sided 0.0625). 95% CI = two-stage bootstrap (mice, then runs). BH-FDR is applied
within each test family. Leave-one-mouse-out (LOMO): every mouse-level estimate is
recomputed with each mouse left out, and the sign must survive for 'sign stable'.
No circularity: noise / quality quantities never use correlation with the reference or
with behavior.

Events: same detector as run_metrics.events (prominence >= 0.2 x trace range, >= 5 frames
apart) unless a test says '3-sigma events' (prominence >= 3 x robust noise; used where the
range-relative threshold would truncate the size distribution).

  python code/STEP8_stats/sensor_assumptions.py              # extract (cached) + analyze both sets
  python code/STEP8_stats/sensor_assumptions.py --set curated --force-extract
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi, stats
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
AUTO = PROJECT / "auto_pipeline"
sys.path.insert(0, str(PROJECT / "code")); sys.path.insert(0, str(PROJECT / "code/STEP7_workflow")); sys.path.insert(0, str(HERE))
from run_metrics import dff, robust_noise, events, region_names, compartment_of   # noqa: E402
from behavior_coupling import session_dir, volume_times, behavior_on_volumes     # noqa: E402
from cohort_stats import bh, cluster_boot                                         # noqa: E402
from common.regions import apply_ignore                                           # noqa: E402
from common.run_marks import is_set_aside                                         # noqa: E402

__version__ = "1.0.0"
OUT = PROJECT / "stats" / "assumptions" / "sensor"
LOG = AUTO / "logs" / "sensor_assumptions_v7.jsonl"
WINDOW_S = 0.4            # same coincidence window as run_metrics
N_VOX_SAMPLE = 3000       # voxels sampled per run for the photon-noise check
# Detector offset. Every two-photon UG unit in every .mesc of this project stores
# Channel_0_Conversion_ConversionLinearOffset = -1454 (scale 1.0), i.e. true signal = raw - 1454.
# The extracted TIFFs keep the raw uint16 values (tissue background ~1570, so only ~115 counts
# are light). run_metrics.dff divides by the raw F0 (~1600), which compresses every dF/F by
# F0_raw / (F0_raw - 1454), a factor that differs between regions and cells (about 3-15x).
# Correlations, event timing and event detection are invariant to it; amplitudes, F0 and the
# photon-noise / bleaching tests are not, so this script works on offset-corrected F.
DETECTOR_OFFSET = 1454.0
# The stored offset is not the real zero: voxels in the tube outside the cell sit at raw ~1566-1590
# with almost no variance (they receive ~no photons in this sparse labeling), so each run's
# EMPIRICAL DARK LEVEL (5th percentile of the sampled non-cell voxel means) is the light zero used
# for 'light above dark' quantities (F0 as an expression proxy, photon noise, bleaching) and as a
# second sensitivity version of the amplitude-ratio test.
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})
warnings.filterwarnings("ignore", category=RuntimeWarning)


def log(what, result):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"time": datetime.now().astimezone().isoformat(), "stage": "sensor_assumptions",
                            "what": what, "result": result}, default=str) + "\n")


# ============================================================================ pre-specified tests
# Written before any analysis was run. 'pred' is the direction predicted by the assumption
# ('greater' = effect > null value). Families are the BH-FDR units.
LIT = {
    "dana2019": "Dana et al. 2019 Nat Methods 16:649 (jGCaMP7; bioRxiv 434589): jGCaMP7s Kd 68 nM vs 147 nM GCaMP6s; "
                "slowest decay of the jGCaMP7 family in mouse V1; 1-AP half-decay in culture 455 ms for GCaMP6s "
                "(jGCaMP7s slower); 'tightening the indicator affinity caused saturation at lower numbers of spikes'; "
                "at the fly NMJ jGCaMP7s responses were SMALLER than GCaMP6s for 40-160 Hz trains (saturation).",
    "chen2013": "Chen et al. 2013 Nature 499:295 (GCaMP6): slow sensors integrate bursts; long expression -> nuclear filling.",
    "larkum1999": "Larkum, Zhu & Sakmann 1999 Nature 398:338 / Larkum 1999 PNAS 96:14600: bAPs above a critical frequency "
                  "trigger distal Ca2+ plateaus (BAC firing); Ca2+ spike initiation zone ~550-900 um from soma (rat).",
    "francioni2019": "Francioni, Padamsey & Rochefort 2019 eLife 8:e49145: GCaMP6s in V1 L5 soma/trunk/tuft - high "
                     "somato-dendritic coupling, transient FREQUENCY decreases with distance from soma, coupling "
                     "unchanged by locomotion.",
    "beaulieu2019": "Beaulieu-Laroche et al. 2019 Neuron 103:235: L5 somato-dendritic activity highly correlated, "
                    "unchanged by visual stimuli and locomotion.",
    "takahashi2016": "Takahashi et al. 2016 Science 354:1587: apical tuft Ca2+ of L5 in S1 tracks perceptual detection.",
    "williams2019": "Williams & Fletcher 2019 Neuron 101:486: cholinergic (arousal) input enhances L5 apical dendritic "
                    "excitability / Ca2+ electrogenesis.",
    "cichon2015": "Cichon & Gan 2015 Nature 520:180: branch-specific dendritic Ca2+ spikes in vivo (L5 tuft).",
    "stuart1997": "Stuart, Spruston, Sakmann & Hausser 1997 TINS 20:125 / Waters et al. 2003: bAP amplitude attenuates "
                  "along the apical trunk; bAP Ca2+ influx decreases distally.",
    "poisson": "Photon (shot) noise: variance of a PMT signal grows linearly with its mean (var/mean slope ~1 on log-log).",
    "nucleus": "stats/nucleus/README.md (this project): nuclear state cannot be scored for the analyzed mice.",
}

DEVIATIONS = [
    "After the first run (archived in stats/assumptions/sensor/old/20261005_233247/) the .mesc files were found to store "
    "a detector offset of -1454 counts for every 2P unit. All amplitudes, F0, photon-noise and bleaching quantities "
    "were then recomputed on offset-corrected F. Correlations, event detection and timing are mathematically "
    "unchanged. T3 is reported both ways (T3_amp_ratio_slope = corrected, T3_amp_ratio_slope_rawF0 = pipeline dF/F).",
    "T1/T2 descriptive amplitude and tau summaries and the dF/F compression factor (T0) were added after the first look.",
    "The stored offset turned out not to be the light zero (non-cell voxels sit ~115 counts above it with almost no "
    "variance). F0 as an expression proxy (T6), photon noise (T8, T11b) and bleaching (T10) therefore use light above "
    "each run's empirical dark level; T3 is also reported on that scale (T3_amp_ratio_slope_dark).",
]

TESTS = {
    "T1": {"family": "T1_kinetics", "assumption": "jGCaMP7s is a slow sensor: event decay is slow (half-decay of "
           "order 0.5-2 s) and similar across compartments (same sensor everywhere)",
           "test": "half-decay t1/2 of isolated events per compartment; branch minus reference per run (mouse-level, two-sided)",
           "pred": "two-sided", "ref": "dana2019"},
    "T2": {"family": "T2_saturation", "assumption": "the high-affinity sensor saturates for the largest events: decay "
           "slows and the peak flattens as events get bigger",
           "test": "per region Spearman(t1/2, peak) > 0 and log-log slope of integral on peak > 1 (mouse-level, one-sided)",
           "pred": "greater", "ref": "dana2019"},
    "T3": {"family": "T3_amplitude_distance", "assumption": "bAP attenuation: the amplitude of reference-led events "
           "falls with distance from the reference",
           "test": "per run Theil-Sen slope of log2(region amplitude / reference amplitude) at reference events vs "
                   "distance (per 100 um) < 0", "pred": "less", "ref": "stuart1997"},
    "T4": {"family": "T4_branch_only", "assumption": "branch-only events are dendritic spikes: larger / longer than "
           "branch events that coincide with a reference event (alternative: small local events)",
           "test": "per run log2 ratio (branch-only / coupled) of amplitude, FWHM and t1/2 (mouse-level, two-sided)",
           "pred": "two-sided", "ref": "cichon2015"},
    "T5": {"family": "T5_bimodality", "assumption": "trunk/reference events are all-or-none (bimodal size: small "
           "bAP-like vs large plateau / BAC events)",
           "test": "3-sigma events, log amplitude, 2- vs 1-component Gaussian mixture, parametric-bootstrap LRT per region",
           "pred": "greater", "ref": "larkum1999"},
    "T6": {"family": "T6_expression", "assumption": "sensor overload: more sensor (higher F0) -> more buffering -> "
           "slower decay and higher coupling (within mouse)",
           "test": "within-mouse Spearman (z-scored per mouse, within-mouse permutation) of F0 vs reference t1/2 and "
                   "vs r(ref,branch)", "pred": "greater", "ref": "chen2013"},
    "T7": {"family": "T7_frame_rate", "assumption": "confound check: volume rate (4.9-8.1 Hz) does not drive measured "
           "coupling or kinetics", "test": "within-mouse Spearman of rate vs r, t1/2, rise; decimation by 2 "
           "(sub-sample / average pairs) paired change in r and t1/2", "pred": "two-sided", "ref": "dana2019"},
    "T8": {"family": "T8_position_noise", "assumption": "noise is photon-limited (dF/F noise ~ 1/sqrt(F0 x voxels)); "
           "regions near the tube edge are noisier beyond that", "test": "per run slope of log noise on "
           "log(F0 x voxels) = -0.5; Spearman(excess noise, fraction of region voxels on the tube boundary) > 0",
           "pred": "greater", "ref": "poisson"},
    "T9": {"family": "T9_behavior", "assumption": "arousal / movement raises apical dendritic event rate and "
           "amplitude, more in branches than at the reference",
           "test": "per run log ratio active/quiet of event rate and amplitude per compartment, for pupil, whisking, "
                   "locomotion (accelerometer); branch/reference rate ratio active vs quiet (mouse-level, two-sided)",
           "pred": "two-sided", "ref": "williams2019"},
    "T10": {"family": "T10_bleaching", "assumption": "photobleaching lowers baseline F over the run, and a shared "
            "bleaching trend inflates soma-branch correlation", "test": "baseline slope (%/min) < 0; "
            "r(ref,branch) with sliding-baseline dF/F minus r with global F0 (two-sided)", "pred": "less",
            "ref": "dana2019"},
    "T11": {"family": "T11_other", "assumption": "(a) event frequency falls with distance from the soma; (b) dF/F "
            "noise is photon noise at the voxel level; (c) saturation / slow decay goes with higher coupling",
            "test": "(a) Theil-Sen slope of 3-sigma event rate vs distance < 0; (b) voxel log var vs log mean slope "
                    "~1; (c) within-mouse Spearman r(ref,branch) vs saturation index", "pred": "mixed",
            "ref": "francioni2019"},
}


# ============================================================================ run sets
def load_runset(name: str) -> tuple[Path, pd.DataFrame]:
    root = PROJECT if name == "curated" else AUTO
    d = pd.read_csv(root / "stats" / "cohort_metrics.csv")
    d = d[~d["behavior_base"].fillna("").map(is_set_aside)].copy()
    d["root"] = str(root)
    ok = []
    for _, r in d.iterrows():
        mf = root / r["metrics_file"]; stem = mf.name.replace("_metrics.json", "")
        ok.append((mf.parent / f"{stem}.tif").exists() and (mf.parent / f"{stem}_segments_final.tif").exists())
    return root, d[np.array(ok, bool)].reset_index(drop=True)


def acquisition_meta(root: Path, mouse: str, date: str, base: str) -> dict:
    """Laser power, PMT gain (UG) and stage Z from the session .summary.csv (via ranked_runs.csv)."""
    try:
        rr = pd.read_csv(root / "ranked_runs.csv")
        run_id = base.split("_")[-1]
        row = rr[(rr.mouse == mouse) & (rr.date == date) & (rr.behavior_run == run_id)]
        if row.empty:
            row = rr[(rr.mouse == mouse) & (rr.date == date) & (rr.munit.astype(str).str.replace("_", "") == run_id)]
        if row.empty:
            return {}
        row = row.iloc[0]; mesc = PROJECT / str(row.mesc_path)
        sm = sorted(mesc.parent.glob("*.summary.csv"))
        if not sm:
            return {"munit": row.munit}
        s = pd.read_csv(sm[0]); s = s[(s.unit == row.munit) & (s.scan_type == "snake")]
        if s.empty:
            return {"munit": row.munit}
        s = s.iloc[0]
        return {"munit": row.munit, "power_LP1": float(s.power_LP1), "UG": float(s.UG), "stage_z_um": float(s.stage_z_um),
                "wavelength_LP1": float(s.wavelength_LP1) if "wavelength_LP1" in s else None}
    except Exception as e:                                             # metadata is optional
        return {"meta_error": str(e)}


# ============================================================================ extraction (cached)
def extract_run(root: Path, row: pd.Series, cache_dir: Path, force=False) -> Path | None:
    base = row["behavior_base"]; out = cache_dir / f"{base}.npz"
    mf = root / row["metrics_file"]
    if out.exists() and not force and out.stat().st_mtime >= mf.stat().st_mtime:
        return out
    m = json.loads(mf.read_text()); run_dir = mf.parent; stem = mf.name.replace("_metrics.json", "")
    stack = tifffile.imread(run_dir / f"{stem}.tif"); seg_p = run_dir / f"{stem}_segments_final.tif"
    seg = tifffile.imread(seg_p); T, Z, Y, X = stack.shape
    labels = sorted(int(v) for v in np.unique(seg) if v > 0)
    names = region_names(seg_p.with_suffix(".json"), labels)
    seg, _ign, ign_labels = apply_ignore(seg, names, seg_p)
    labels = [l for l in labels if l not in ign_labels]
    comp = [compartment_of(names[l]) for l in labels]
    # reference exactly as run_metrics: first soma region, else its saved proximal trunk
    if m.get("reference") == "proximal_trunk":
        ref = next(i for i, l in enumerate(labels) if names[l] == m["reference_region"])
    else:
        ref = next(i for i, l in enumerate(labels) if comp[i] == "soma")
    flat = stack.reshape(T, -1)
    F = np.stack([flat[:, np.flatnonzero((seg == l).ravel())].mean(1) for l in labels]).astype(np.float64)
    dist = np.array([m["regions"][str(l)].get("distance_um", np.nan) for l in labels], float)
    vox = np.array([(seg == l).sum() for l in labels], int)
    geo = []
    for l in labels:
        zz, yy, xx = np.nonzero(seg == l)
        on_edge = (zz == 0) | (zz == Z - 1) | (yy == 0) | (yy == Y - 1)
        geo.append([zz.mean(), yy.mean(), xx.mean(), on_edge.mean(),
                    np.minimum(zz, Z - 1 - zz).mean(), np.minimum(yy, Y - 1 - yy).mean()])
    geo = np.array(geo, float)
    # photon-noise check: per voxel mean and high-pass variance (first difference, robust)
    rng = np.random.default_rng(0); cell = np.flatnonzero((seg > 0).ravel())
    bgm = np.flatnonzero((seg == 0).ravel())
    pick = np.concatenate([rng.choice(cell, min(N_VOX_SAMPLE, len(cell)), replace=False),
                           rng.choice(bgm, min(N_VOX_SAMPLE, len(bgm)), replace=False)])
    v = flat[:, pick].astype(np.float64); dv = np.diff(v, axis=0)
    vox_mean = v.mean(0); vox_var = (1.4826 * np.median(np.abs(dv - np.median(dv, 0)), 0)) ** 2 / 2
    vox_is_cell = np.r_[np.ones(min(N_VOX_SAMPLE, len(cell)), bool), np.zeros(min(N_VOX_SAMPLE, len(bgm)), bool)]
    del stack, flat, v, dv
    rate = float(row["frame_rate_hz"])
    sdir = session_dir(run_dir); run_id = base.split("_")[-1]
    beh = {}; tsrc = "none"
    if bool(row.get("has_behavior", False)):
        tv, tsrc = volume_times(sdir, run_id, T, rate)
        beh = behavior_on_volumes(sdir, base, run_id, tv, rate)
    meta = acquisition_meta(root, row["mouse"], row["date"], base)
    np.savez_compressed(out, F=F.astype(np.float32), dist=dist, vox=vox, geo=geo, ref=ref,
                        names=np.array([names[l] for l in labels]), comp=np.array(comp),
                        vox_mean=vox_mean.astype(np.float32), vox_var=vox_var.astype(np.float32), vox_is_cell=vox_is_cell,
                        **{f"beh_{k}": np.asarray(b, np.float32) for k, b in beh.items()},
                        meta=json.dumps({"rate": rate, "T": T, "shape": [T, Z, Y, X], "time_source": tsrc,
                                         "r_soma_branch_metrics": m.get("r_soma_branch"), **meta}))
    return out


def load_cache(p: Path) -> dict:
    z = np.load(p, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    d["beh"] = {k[4:]: d.pop(k).astype(float) for k in list(d) if k.startswith("beh_")}
    d["F"] = d["F"].astype(np.float64)
    return d


# ============================================================================ event features
def smooth3(x):
    return ndi.uniform_filter1d(x, 3, mode="nearest")


def events_3sigma(x, min_dist=5):
    pk, _ = find_peaks(x, prominence=3 * robust_noise(x), distance=min_dist)
    return pk


def event_features(x: np.ndarray, peaks: np.ndarray, rate: float) -> pd.DataFrame:
    """Per event: amplitude above the local pre-event baseline, 10-90 % rise time, half-decay
    time, single-exponential tau (log-linear on the 100-20 % part of the decay), integral
    (dF/F x s from onset to return to 10 %), FWHM, and isolation."""
    xs = smooth3(x); T = len(x); pre = max(2, int(round(2.0 * rate))); post_max = int(round(10.0 * rate))
    rows = []
    for i, p in enumerate(peaks):
        lo = max(0, p - pre); on = lo + int(np.argmin(xs[lo:p + 1])); b = xs[on]; a = x[p] - b
        nxt = peaks[i + 1] if i + 1 < len(peaks) else T
        prv = peaks[i - 1] if i > 0 else -10 ** 9
        end = min(T - 1, p + post_max, nxt - 1) if nxt < T else min(T - 1, p + post_max)
        r = {"peak": int(p), "amp": float(a), "base": float(b)}
        if a <= 0:
            rows.append(r); continue
        seg = x[on:p + 1] - b
        def cross_up(level):
            k = np.flatnonzero(seg >= level)
            if not len(k): return np.nan
            k = k[0]
            if k == 0: return 0.0
            return (k - 1) + (level - seg[k - 1]) / max(seg[k] - seg[k - 1], 1e-12)
        r["rise_s"] = float((cross_up(0.9 * a) - cross_up(0.1 * a)) / rate)
        dec = x[p:end + 1] - b
        def cross_down(level):
            k = np.flatnonzero(dec <= level)
            if not len(k) or k[0] == 0: return np.nan
            k = k[0]; return (k - 1) + (dec[k - 1] - level) / max(dec[k - 1] - dec[k], 1e-12)
        r["t_half_s"] = float(cross_down(0.5 * a) / rate)
        k20 = np.flatnonzero(dec <= 0.2 * a); stop = k20[0] if len(k20) else len(dec)
        y = dec[:stop]
        if len(y) >= 3 and np.all(y > 0):
            sl = np.polyfit(np.arange(len(y)) / rate, np.log(y), 1)[0]
            r["tau_s"] = float(-1 / sl) if sl < 0 else np.nan
        k10 = np.flatnonzero(dec <= 0.1 * a); ret = p + (k10[0] if len(k10) else len(dec) - 1)
        r["integral"] = float(np.clip(x[on:ret + 1] - b, 0, None).sum() / rate)
        above = x - b >= 0.5 * a; l_, r_ = p, p
        while l_ > 0 and above[l_ - 1]: l_ -= 1
        while r_ < T - 1 and above[r_ + 1]: r_ += 1
        r["fwhm_s"] = float((r_ - l_ + 1) / rate)
        r["isolated"] = bool((p - prv) > 2.0 * rate and (nxt - p) > 3.0 * rate)
        r["truncated"] = bool(nxt < T and not len(k10))
        rows.append(r)
    return pd.DataFrame(rows)


def near(a, b, w):
    a = np.asarray(a); b = np.asarray(b)
    return np.array([np.any(np.abs(b - x) <= w) for x in a], bool) if len(a) and len(b) else np.zeros(len(a), bool)


def r_ref_branch(D: np.ndarray, ref: int, comp: np.ndarray) -> float:
    br = [i for i in range(len(comp)) if comp[i] == "branch" and i != ref]
    if not br:
        return np.nan
    return float(np.mean([np.corrcoef(D[ref], D[i])[0, 1] for i in br]))


# ============================================================================ per-run analysis
def analyze_run(base: str, d: dict, row: pd.Series) -> dict:
    rate = d["meta"]["rate"]; Fraw = d["F"]; F = Fraw - DETECTOR_OFFSET; T = F.shape[1]; R = F.shape[0]; ref = int(d["ref"])
    comp = d["comp"].astype(str); names = d["names"].astype(str); w = max(1, int(round(WINDOW_S * rate)))
    comp_eff = comp.copy(); comp_eff[ref] = "ref"            # reference vs other trunk vs branch
    D = np.stack([dff(F[i]) for i in range(R)])                 # offset-corrected dF/F (r identical to pipeline)
    Draw = np.stack([dff(Fraw[i]) for i in range(R)])           # pipeline dF/F (raw F0), for comparison only
    dark = float(np.percentile(d["vox_mean"][~d["vox_is_cell"]], 5))
    F0raw = np.array([np.percentile(Fraw[i], 10) for i in range(R)]); F0light = F0raw - dark
    Dlight = np.stack([(Fraw[i] - F0raw[i]) / max(F0light[i], 1.0) for i in range(R)])   # dF/F above the dark level
    noise_light = np.array([robust_noise(Dlight[i]) for i in range(R)])
    noise = np.array([robust_noise(D[i]) for i in range(R)])
    F0 = np.array([np.percentile(F[i], 10) for i in range(R)])
    ev = [events(D[i]) for i in range(R)]
    ev3 = [events_3sigma(D[i]) for i in range(R)]
    feats = []
    for i in range(R):
        fe = event_features(D[i], ev[i], rate); fe["region"] = names[i]; fe["comp"] = comp_eff[i]; fe["ri"] = i
        fe["dist"] = d["dist"][i]
        fe["coupled"] = near(ev[i], ev[ref], w) if i != ref else True
        feats.append(fe)
    E = pd.concat(feats, ignore_index=True)
    E["amp_sigma"] = E["amp"] / noise[E["ri"].values]
    out = {"behavior_base": base, "mouse": row["mouse"], "date": row["date"], "rate": rate, "T": T,
           "reference": row.get("reference"), "n_regions": R,
           "r_ref_branch": r_ref_branch(D, ref, comp), "r_ref_branch_metrics": d["meta"].get("r_soma_branch_metrics"),
           "F0_ref": float(F0[ref]), "F0_ref_raw": float(np.percentile(Fraw[ref], 10)), "noise_ref": float(noise[ref]),
           "dark_level": dark, "F0_light_ref": float(F0light[ref]),
           "F0_light_median_regions": float(np.median(F0light)),
           "n_regions_F0light_below_10": int((F0light < 10).sum()),
           "dff_compression_ref": float(np.percentile(Fraw[ref], 10) / F0[ref]),
           "dff_compression_dark_ref": float(F0raw[ref] / max(F0light[ref], 1.0)),
           "dff_compression_branch": float(np.mean([np.percentile(Fraw[i], 10) / F0[i] for i in range(R) if comp[i] == "branch"]))
           if any(c == "branch" for c in comp) else np.nan,
           "power": d["meta"].get("power_LP1"), "UG": d["meta"].get("UG"), "stage_z_um": d["meta"].get("stage_z_um")}
    iso = E[E.isolated & (E.amp > 0)]
    for c in ("ref", "trunk", "branch"):
        s = iso[iso.comp == c]
        out[f"t_half_{c}"] = float(s.t_half_s.median()) if s.t_half_s.notna().sum() >= 3 else np.nan
        out[f"tau_{c}"] = float(s.tau_s.median()) if s.tau_s.notna().sum() >= 3 else np.nan
        out[f"rise_{c}"] = float(s.rise_s.median()) if s.rise_s.notna().sum() >= 3 else np.nan
        out[f"n_iso_{c}"] = int(len(s))
        a = E[(E.comp == c) & (E.amp > 0)]
        out[f"amp_p90_{c}"] = float(a.amp.quantile(0.9)) if len(a) >= 3 else np.nan
    out["frame_s"] = 1 / rate
    # ---- T2 saturation: per region Spearman(t1/2, amp), log-log slope integral ~ amp
    sp, sl = [], []
    for i in range(R):
        s = E[(E.ri == i) & (E.amp > 0) & E.t_half_s.notna() & ~E.truncated]
        if len(s) >= 6:
            sp.append(stats.spearmanr(s.amp, s.t_half_s)[0])
        s2 = E[(E.ri == i) & (E.amp > 0) & (E.integral > 0) & ~E.truncated]
        if len(s2) >= 6:
            sl.append(stats.theilslopes(np.log(s2.integral), np.log(s2.amp))[0])
    out["sat_spearman_thalf_amp"] = float(np.nanmean(sp)) if sp else np.nan
    out["sat_loglog_slope_int_amp"] = float(np.nanmean(sl)) if sl else np.nan
    out["sat_index"] = out["sat_spearman_thalf_amp"]          # pre-specified saturation index for T11c
    # ---- T3 amplitude vs distance at reference events
    pre = max(2, int(round(2.0 * rate)))
    others = [i for i in range(R) if i != ref and np.isfinite(d["dist"][i])]
    for tag, DD in (("", D), ("_rawF0", Draw), ("_dark", Dlight)):
        xs = [smooth3(DD[i]) for i in range(R)]
        amps = np.full((len(ev[ref]), R), np.nan)
        for k, p in enumerate(ev[ref]):
            for i in range(R):
                lo = max(0, p - pre); b = xs[i][lo:p + 1].min()
                amps[k, i] = DD[i][max(0, p - w):min(T, p + w + 1)].max() - b
        if len(ev[ref]) >= 3 and len(others) >= 2:
            ratio = np.log2(np.clip(np.nanmedian(amps[:, others], 0), 1e-6, None) / max(np.nanmedian(amps[:, ref]), 1e-6))
            out[f"amp_dist_slope_log2_per100um{tag}"] = float(stats.theilslopes(ratio, d["dist"][others] / 100.0)[0])
            out[f"amp_dist_points{tag}"] = list(zip((d["dist"][others]).tolist(), ratio.tolist()))
    # ---- T4 branch-only vs coupled branch events
    b = E[(E.comp == "branch") & (E.amp > 0)]
    for key, col in (("amp", "amp"), ("fwhm", "fwhm_s"), ("thalf", "t_half_s")):
        o = b[~b.coupled][col].dropna(); c = b[b.coupled][col].dropna()
        out[f"bo_vs_cpl_{key}_log2"] = float(np.log2(o.median() / c.median())) if len(o) >= 3 and len(c) >= 3 and c.median() > 0 and o.median() > 0 else np.nan
    out["n_branch_only"] = int((~b.coupled).sum()); out["n_branch_coupled"] = int(b.coupled.sum())
    o = b[~b.coupled]; out["bo_amp_sigma_median"] = float(o.amp_sigma.median()) if len(o) else np.nan
    # ---- T5 bimodality (3-sigma events, reference + trunk regions)
    bim = []
    for i in range(R):
        if comp_eff[i] not in ("ref", "trunk"):
            continue
        fe = event_features(D[i], ev3[i], rate); a = fe.amp[fe.amp > 0].values
        if len(a) >= 20:
            bim.append({"region": names[i], "comp": comp_eff[i], **gmm_bimodality(np.log(a)), "n": int(len(a))})
    out["bimodality"] = bim
    # ---- T7 decimation by 2
    if R >= 2:
        Dsub = D[:, ::2]; Davg = D[:, : (T // 2) * 2].reshape(R, -1, 2).mean(2)
        out["r_dec_sub"] = r_ref_branch(Dsub, ref, comp); out["r_dec_avg"] = r_ref_branch(Davg, ref, comp)
        fs = event_features(Dsub[ref], events(Dsub[ref]), rate / 2)
        fs = fs[fs.isolated.fillna(False).astype(bool) & (fs.amp > 0)] if "isolated" in fs else fs.iloc[0:0]
        out["t_half_ref_dec"] = float(fs.t_half_s.median()) if len(fs) and fs.t_half_s.notna().sum() >= 3 else np.nan
    # ---- T8 noise vs photon expectation and tube position
    lp = np.log(np.maximum(F0light, 1.0) * d["vox"]); ln = np.log(noise_light)
    if R >= 3:
        out["noise_slope_on_logF0vox"] = float(np.polyfit(lp, ln, 1)[0])
        excess = ln - (-0.5 * lp); excess -= excess.mean()
        out["excess_noise_vs_edge_rho"] = float(stats.spearmanr(excess, d["geo"][:, 3])[0]) if np.ptp(d["geo"][:, 3]) > 0 else np.nan
        out["excess_noise_vs_zedge_rho"] = float(stats.spearmanr(excess, -d["geo"][:, 4])[0]) if np.ptp(d["geo"][:, 4]) > 0 else np.nan
    out["regions"] = [{"name": names[i], "comp": comp_eff[i], "dist": float(d["dist"][i]), "vox": int(d["vox"][i]),
                       "F0": float(F0light[i]), "noise": float(noise_light[i]), "F0_raw": float(F0raw[i]),
                       "noise_offset_corrected": float(noise[i]), "edge_frac": float(d["geo"][i, 3]),
                       "z_edge_dist": float(d["geo"][i, 4]), "y_edge_dist": float(d["geo"][i, 5]),
                       "rate3_per_min": float(len(ev3[i]) / (T / rate / 60)),
                       "rate_per_min": float(len(ev[i]) / (T / rate / 60))} for i in range(R)]
    # ---- T9 behavior states
    out["behavior"] = behavior_tests(d, D, ev, ref, comp_eff, rate, E)
    # ---- T10 bleaching
    win = max(5, int(round(20 * rate))); tt = np.arange(T) / rate
    bl = ndi.percentile_filter(Fraw[ref] - dark, 10, size=win, mode="nearest")
    sl = np.polyfit(tt / 60, bl, 1)[0]; out["bleach_pct_per_min"] = float(100 * sl / bl[: win].mean())
    Ddet = np.stack([(F[i] - ndi.percentile_filter(F[i], 10, size=max(5, int(round(30 * rate))), mode="nearest"))
                     / np.maximum(ndi.percentile_filter(F[i], 10, size=max(5, int(round(30 * rate))), mode="nearest"), 1e-6)
                     for i in range(R)])
    out["r_ref_branch_detrended"] = r_ref_branch(Ddet, ref, comp)
    out["F0_first_last_ratio"] = float(np.percentile(Fraw[ref][-win:] - dark, 10) / max(np.percentile(Fraw[ref][:win] - dark, 10), 1.0))
    # ---- T11a event rate vs distance (3-sigma events)
    oth = [i for i in range(R) if np.isfinite(d["dist"][i])]
    if len(oth) >= 3:
        rr = np.array([len(ev3[i]) for i in oth]) / (T / rate / 60)
        out["rate_dist_slope_per100um"] = float(stats.theilslopes(rr, d["dist"][oth] / 100.0)[0])
        out["rate_dist_rel_slope_per100um"] = float(stats.theilslopes(rr / max(rr[np.argmin(d['dist'][oth])], 1e-6),
                                                                       d["dist"][oth] / 100.0)[0])
    # ---- T11b voxel photon noise
    vm, vv, cell = d["vox_mean"].astype(float), d["vox_var"].astype(float), d["vox_is_cell"]
    off = dark                                         # empirical dark level (see DETECTOR_OFFSET note)
    ok = (vm - off > 5) & (vv > 0)
    if ok.sum() > 100:
        sl_, ic = np.polyfit(np.log(vm[ok] - off), np.log(vv[ok]), 1)
        out["vox_loglog_slope"] = float(sl_)
        g = np.polyfit(vm[ok], vv[ok], 1); out["vox_gain_var_per_count"] = float(g[0])
        out["vox_linear_r2"] = float(np.corrcoef(vm[ok], vv[ok])[0, 1] ** 2)
    out["_events"] = E
    return out


def gmm_bimodality(x: np.ndarray, B: int = 200, seed: int = 0) -> dict:
    from sklearn.mixture import GaussianMixture
    x = x.reshape(-1, 1)
    def lr(v):
        g1 = GaussianMixture(1, random_state=0).fit(v); g2 = GaussianMixture(2, n_init=3, random_state=0).fit(v)
        return 2 * (g2.score(v) - g1.score(v)) * len(v), g1, g2
    obs, g1, g2 = lr(x)
    rng = np.random.default_rng(seed); mu, sd = float(g1.means_[0, 0]), float(np.sqrt(g1.covariances_.ravel()[0]))
    null = np.array([lr(rng.normal(mu, sd, size=x.shape))[0] for _ in range(B)])
    w = g2.weights_; m2 = g2.means_.ravel(); s2 = np.sqrt(g2.covariances_.ravel())
    sep = abs(m2[0] - m2[1]) / np.sqrt((s2[0] ** 2 + s2[1] ** 2) / 2)          # Ashman D
    return {"lr": float(obs), "p": float((1 + np.sum(null >= obs)) / (1 + B)), "ashman_D": float(sep),
            "min_weight": float(w.min()), "dBIC": float(g1.bic(x) - g2.bic(x))}


def behavior_tests(d, D, ev, ref, comp_eff, rate, E) -> dict:
    if not d["beh"]:
        return {}
    T = D.shape[1]; res = {}
    states = {}
    for bn, v in d["beh"].items():
        v = np.asarray(v, float)[:T]
        if len(v) < T or not np.isfinite(v).any():
            continue
        thr = np.nanpercentile(v, 75 if bn == "accelerometer" else 50)   # pre-specified thresholds
        a = v > thr
        if 0.1 < a.mean() < 0.9:
            states[bn] = a
    for bn, a in states.items():
        r = {"frac_active": float(a.mean())}
        mins_a = a.sum() / rate / 60; mins_q = (~a).sum() / rate / 60
        rate_c = {}
        for c in ("ref", "trunk", "branch"):
            idx = [i for i in range(len(comp_eff)) if comp_eff[i] == c]
            if not idx:
                continue
            na = np.mean([a[ev[i]].sum() for i in idx]); nq = np.mean([(~a)[ev[i]].sum() for i in idx])
            r[f"rate_log2_{c}"] = float(np.log2(((na + 0.5) / mins_a) / ((nq + 0.5) / mins_q)))
            rate_c[c] = ((na + 0.5) / mins_a, (nq + 0.5) / mins_q)
            s = E[(E.comp == c) & (E.amp > 0)]
            sa = s[a[s.peak.values]].amp; sq = s[~a[s.peak.values]].amp
            r[f"amp_log2_{c}"] = float(np.log2(sa.median() / sq.median())) if len(sa) >= 3 and len(sq) >= 3 else np.nan
        if "branch" in rate_c and "ref" in rate_c:
            r["branch_over_ref_rate_log2"] = float(np.log2((rate_c["branch"][0] / rate_c["ref"][0]) /
                                                           (rate_c["branch"][1] / rate_c["ref"][1])))
        res[bn] = r
    return res


# ============================================================================ cohort inference
def mouse_level(df: pd.DataFrame, col: str, null: float = 0.0, alternative: str = "two-sided") -> dict:
    s = df[["mouse", col]].dropna()
    if s.empty:
        return {"n_runs": 0, "n_mice": 0}
    mm = s.groupby("mouse")[col].mean() - null; k = len(mm)
    res = {"est": float(mm.mean() + null), "null": float(null), "alt": alternative, "n_runs": int(len(s)), "n_mice": int(k),
           "n_mice_pos": int((mm > 0).sum()), "per_mouse": {m: float(v + null) for m, v in mm.items()}}
    if k >= 2 and np.ptp(mm.values) + abs(mm.values).sum() > 0:
        try:
            res["p"] = float(stats.wilcoxon(mm.values, alternative=alternative, method="exact").pvalue)
        except ValueError:
            res["p"] = float("nan")
        res["min_p"] = 0.5 ** k * (2 if alternative == "two-sided" else 1)
    else:
        res["p"] = float("nan")
    tmp = s.copy(); tmp["v"] = tmp[col]
    lo, hi = cluster_boot(tmp, lambda g: float(g.groupby("mouse")["v"].mean().mean()), B=2000)
    res["ci"] = [lo, hi]
    lomo = [float((mm.drop(m)).mean() + null) for m in mm.index] if k >= 3 else []
    res["lomo_range"] = [min(lomo), max(lomo)] if lomo else None
    res["lomo_sign_stable"] = bool(lomo and (np.sign(np.array(lomo) - null) == np.sign(mm.mean())).all())
    return res


def within_mouse_spearman(df: pd.DataFrame, x: str, y: str, n_perm: int = 10000, alternative="greater") -> dict:
    s = df[["mouse", x, y]].dropna()
    s = s[s.groupby("mouse")[x].transform("count") >= 2]
    if len(s) < 4:
        return {"n_runs": int(len(s)), "n_mice": int(s.mouse.nunique())}
    def z(v): return (v - v.mean()) / (v.std(ddof=0) + 1e-12)
    rx = s.groupby("mouse")[x].transform(lambda v: z(v.rank())); ry = s.groupby("mouse")[y].transform(lambda v: z(v.rank()))
    obs = float(np.mean(rx * ry))
    rng = np.random.default_rng(0); groups = [np.flatnonzero((s.mouse == m).values) for m in s.mouse.unique()]
    ryv = ry.values.copy(); null = np.empty(n_perm)
    for b in range(n_perm):
        perm = ryv.copy()
        for g in groups: perm[g] = ryv[rng.permutation(g)]
        null[b] = np.mean(rx.values * perm)
    if alternative == "greater":
        p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    elif alternative == "less":
        p = (1 + np.sum(null <= obs)) / (1 + n_perm)
    else:
        p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm)
    per = {m: float(stats.spearmanr(g[x], g[y])[0]) if len(g) >= 3 else None for m, g in s.groupby("mouse")}
    lomo = []
    for m in s.mouse.unique():
        k = s.mouse != m
        lomo.append(float(np.mean(rx[k] * ry[k])))
    k_pos = sum(1 for v in per.values() if v is not None and v > 0); k_n = sum(1 for v in per.values() if v is not None)
    return {"est": obs, "null": 0.0, "alt": alternative, "p": float(p), "n_runs": int(len(s)), "n_mice": int(s.mouse.nunique()),
            "n_mice_pos": k_pos, "n_mice_with_rho": k_n, "per_mouse_rho": per,
            "lomo_range": [min(lomo), max(lomo)], "lomo_sign_stable": bool((np.sign(lomo) == np.sign(obs)).all())}


def verdict(r: dict, pred: str) -> str:
    p, q, est = r.get("p"), r.get("q"), r.get("est")
    if est is None or p is None or not np.isfinite(p):
        return "not testable (too few runs/events)"
    null = r.get("null", 0.0); up = est > null
    if pred in ("greater", "less"):
        right = up if pred == "greater" else not up
        if q is not None and q < 0.05:
            return "SUPPORTED" if right else "CONTRADICTED (significant, opposite direction)"
        k, n = r.get("n_mice_pos"), r.get("n_mice_with_rho", r.get("n_mice"))
        if k is not None and n and ((pred == "greater" and k == n) or (pred == "less" and k == 0)):
            return "consistent in every mouse but not significant (few mice)"
        return ("trend in the predicted direction, not significant" if right
                else "not supported (estimate in the opposite direction)")
    if q is not None and q < 0.05:
        return "difference detected (" + ("higher" if up else "lower") + ")"
    k, n = r.get("n_mice_pos"), r.get("n_mice_with_rho", r.get("n_mice"))
    if k is not None and n and n >= 4 and k in (0, n):
        return (f"same direction in all {n} mice ({'higher' if up else 'lower'}), but the smallest attainable "
                f"p with {n} mice is {0.5 ** n * 2:.3f}: not significant")
    return "no detectable difference"


# ============================================================================ figures
def savefig(fig, name, outdir):
    fig.tight_layout(); fig.savefig(outdir / f"{name}.png", dpi=160); fig.savefig(outdir / f"{name}.pdf"); plt.close(fig)


def mouse_colors(mice):
    cm = plt.cm.tab10.colors; return {m: cm[i % 10] for i, m in enumerate(sorted(mice))}


def strip(ax, df, cols, labels, mc, null=0.0, ylabel=""):
    for j, c in enumerate(cols):
        s = df[["mouse", c]].dropna()
        for m, g in s.groupby("mouse"):
            ax.scatter(np.full(len(g), j) + np.random.default_rng(1).uniform(-0.12, 0.12, len(g)), g[c], s=10,
                       color=mc[m], alpha=0.6, lw=0)
            ax.scatter([j + 0.25], [g[c].mean()], marker="D", s=22, color=mc[m], edgecolor="k", lw=0.4)
    ax.axhline(null, color="0.5", lw=0.6, ls="--"); ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)


# ============================================================================ main analysis
def analyze_set(name: str, force_extract=False) -> dict:
    root, runs = load_runset(name)
    outdir = OUT if name == "curated" else OUT / "screening"; outdir.mkdir(parents=True, exist_ok=True)
    cache = OUT / "cache" / name; cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); per = []; allE = []
    for _, row in runs.iterrows():
        try:
            p = extract_run(root, row, cache, force_extract)
        except Exception as e:
            log(f"{name}: extract {row['behavior_base']}", f"FAILED {e!r}"); print(f"  {row['behavior_base']}: extract failed {e!r}"); continue
        d = load_cache(p); r = analyze_run(row["behavior_base"], d, row)
        E = r.pop("_events"); E["behavior_base"] = r["behavior_base"]; E["mouse"] = r["mouse"]; allE.append(E)
        per.append(r); print(f"  {r['behavior_base']}: r={r['r_ref_branch']:.3f} (metrics {r['r_ref_branch_metrics']})  "
                             f"t1/2 ref={r['t_half_ref']:.2f}s")
    log(f"{name}: extraction+per-run", f"{len(per)} runs in {time.time() - t0:.0f} s")
    P = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in per])
    E = pd.concat(allE, ignore_index=True)
    # sanity: our reference-branch r must reproduce run_metrics exactly (same traces, same dF/F)
    chk = P[["r_ref_branch", "r_ref_branch_metrics"]].dropna().astype(float)
    max_dev = float(np.abs(chk.r_ref_branch - chk.r_ref_branch_metrics).max()) if len(chk) else float("nan")
    log(f"{name}: sanity r(ref,branch) vs run_metrics", f"max |diff| = {max_dev:.2e} over {len(chk)} runs")
    mc = mouse_colors(P.mouse.unique()); R = {}

    # ---------------- T1 kinetics
    for c in ("ref", "trunk", "branch"):
        R[f"T1_t_half_{c}"] = {**mouse_level(P, f"t_half_{c}", null=0.0), "kind": "descriptive"}
        R[f"T1_tau_{c}"] = {**mouse_level(P, f"tau_{c}", null=0.0), "kind": "descriptive"}
        R[f"T2_amp_p90_{c}"] = {**mouse_level(P, f"amp_p90_{c}", null=0.0), "kind": "descriptive",
                                "note": "90th percentile event amplitude, offset-corrected dF/F"}
    R["T0_dff_compression_ref"] = {**mouse_level(P, "dff_compression_ref", null=1.0), "kind": "descriptive",
                                   "note": "pipeline dF/F x this factor = offset-corrected dF/F (reference region)"}
    R["T0_dff_compression_branch"] = {**mouse_level(P, "dff_compression_branch", null=1.0), "kind": "descriptive"}
    R["T0_dff_compression_dark_ref"] = {**mouse_level(P, "dff_compression_dark_ref", null=1.0), "kind": "descriptive",
                                        "note": "pipeline dF/F x this factor = dF/F above the empirical dark level"}
    R["T0_F0_light_ref_counts"] = {**mouse_level(P, "F0_light_ref", null=0.0), "kind": "descriptive"}
    R["T0_dark_level"] = {**mouse_level(P, "dark_level", null=DETECTOR_OFFSET), "kind": "descriptive",
                          "note": "empirical dark level vs the stored offset 1454"}
    P["thalf_branch_minus_ref"] = P.t_half_branch - P.t_half_ref
    R["T1_branch_minus_ref"] = mouse_level(P, "thalf_branch_minus_ref", alternative="two-sided")
    P["rise_over_frame_ref"] = P.rise_ref / P.frame_s
    R["T1_rise_ref_in_frames"] = {**mouse_level(P, "rise_over_frame_ref", null=0.0), "kind": "descriptive"}
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    strip(ax[0], P, ["t_half_ref", "t_half_trunk", "t_half_branch"], ["ref", "trunk", "branch"], mc, ylabel="half-decay (s)")
    ax[0].axhspan(0.455, 2.0, color="g", alpha=0.08); ax[0].set_title("decay of isolated events", loc="left")
    strip(ax[1], P, ["rise_ref", "rise_branch"], ["ref", "branch"], mc, ylabel="10-90% rise (s)")
    ax[1].scatter(np.zeros(len(P)) - 0.35, P.frame_s, marker="_", color="k", s=40); ax[1].set_title("rise vs frame interval (black)", loc="left")
    iso = E[E.isolated.fillna(False).astype(bool) & (E.amp > 0)]
    for c, col in (("ref", "k"), ("branch", "tab:orange")):
        s = iso[iso.comp == c].t_half_s.dropna(); ax[2].hist(s.clip(0, 6), bins=30, histtype="step", color=col, label=c, density=True)
    ax[2].set_xlabel("half-decay (s), pooled events"); ax[2].legend(frameon=False)
    savefig(fig, "T1_kinetics", outdir)

    # ---------------- T2 saturation
    R["T2_spearman_thalf_amp"] = mouse_level(P, "sat_spearman_thalf_amp", alternative="greater")
    R["T2_loglog_slope_integral_amp"] = {**mouse_level(P, "sat_loglog_slope_int_amp", null=1.0, alternative="greater"), "null": 1.0}
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    s = E[(E.amp > 0) & E.t_half_s.notna() & ~E.truncated.fillna(False).astype(bool)]
    for m, g in s.groupby("mouse"):
        ax[0].scatter(g.amp, g.t_half_s, s=3, color=mc[m], alpha=0.4, lw=0, label=m.replace("rbp4_", ""))
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("event amplitude (dF/F)"); ax[0].set_ylabel("half-decay (s)")
    ax[0].legend(fontsize=5, frameon=False, markerscale=3)
    s2 = E[(E.amp > 0) & (E.integral > 0)]
    for m, g in s2.groupby("mouse"):
        ax[1].scatter(g.amp, g.integral, s=3, color=mc[m], alpha=0.4, lw=0)
    xx = np.logspace(np.log10(s2.amp.min()), np.log10(s2.amp.max()), 10)
    ax[1].plot(xx, xx * np.median(s2.integral / s2.amp), "k--", lw=0.8, label="slope 1 (linear)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("amplitude"); ax[1].set_ylabel("integral (dF/F s)"); ax[1].legend(frameon=False)
    strip(ax[2], P, ["sat_spearman_thalf_amp", "sat_loglog_slope_int_amp"], ["rho(t1/2,amp)", "slope int~amp"], mc)
    ax[2].axhline(1, color="r", lw=0.5, ls=":")
    savefig(fig, "T2_saturation", outdir)

    # ---------------- T3 amplitude vs distance
    R["T3_amp_ratio_slope"] = mouse_level(P, "amp_dist_slope_log2_per100um", alternative="less")
    R["T3_amp_ratio_slope_rawF0"] = {**mouse_level(P, "amp_dist_slope_log2_per100um_rawF0", alternative="less"),
                                     "note": "same test on the pipeline dF/F (raw F0 incl. detector offset)"}
    R["T3_amp_ratio_slope_dark"] = {**mouse_level(P, "amp_dist_slope_log2_per100um_dark", alternative="less"),
                                    "note": "same test with dF/F above the empirical dark level"}
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    for r in per:
        pts = r.get("amp_dist_points")
        if pts:
            pts = np.array(pts); o = np.argsort(pts[:, 0])
            ax[0].plot(pts[o, 0], pts[o, 1], "-o", ms=2, lw=0.7, color=mc[r["mouse"]])
    ax[0].axhline(0, color="0.5", lw=0.6, ls="--"); ax[0].set_xlabel("distance from reference (um)")
    ax[0].set_ylabel("log2 amplitude / reference\n(at reference events)")
    strip(ax[1], P, ["amp_dist_slope_log2_per100um"], ["slope per 100 um"], mc)
    savefig(fig, "T3_amplitude_vs_distance", outdir)

    # ---------------- T4 branch-only vs coupled
    for k in ("amp", "fwhm", "thalf"):
        R[f"T4_{k}_log2"] = mouse_level(P, f"bo_vs_cpl_{k}_log2", alternative="two-sided")
    R["T4_branch_only_amp_in_noise_sd"] = {**mouse_level(P, "bo_amp_sigma_median", null=3.0, alternative="greater"), "null": 3.0}
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    strip(ax[0], P, ["bo_vs_cpl_amp_log2", "bo_vs_cpl_fwhm_log2", "bo_vs_cpl_thalf_log2"], ["amplitude", "FWHM", "t1/2"], mc,
          ylabel="log2 (branch-only / coupled)")
    b = E[(E.comp == "branch") & (E.amp > 0)]
    ax[1].hist(np.log10(b[b.coupled.astype(bool)].amp), bins=30, histtype="step", color="k", density=True, label="coupled")
    ax[1].hist(np.log10(b[~b.coupled.astype(bool)].amp), bins=30, histtype="step", color="tab:orange", density=True, label="branch-only")
    ax[1].set_xlabel("log10 amplitude (dF/F)"); ax[1].legend(frameon=False)
    savefig(fig, "T4_branch_only_events", outdir)

    # ---------------- T5 bimodality
    B = pd.DataFrame([{"behavior_base": r["behavior_base"], "mouse": r["mouse"], **x} for r in per for x in r["bimodality"]])
    if len(B):
        B["q"] = bh(B.p.values)
        B["bimodal"] = (B.q < 0.05) & (B.ashman_D > 2)
        Bm = B.groupby("behavior_base").agg(mouse=("mouse", "first"), any_bimodal=("bimodal", "max")).reset_index()
        Bm["any_bimodal"] = Bm.any_bimodal.astype(float)
        R["T5_frac_runs_bimodal"] = {**mouse_level(Bm, "any_bimodal", alternative="greater"), "kind": "descriptive",
                                     "n_regions": int(len(B)), "n_regions_bimodal": int(B.bimodal.sum()),
                                     "by_compartment": {c: f"{int(g.bimodal.sum())}/{len(g)} regions, "
                                                        f"{g[g.bimodal].behavior_base.nunique()} runs, {g[g.bimodal].mouse.nunique()} mice"
                                                        for c, g in B.groupby("comp")}}
        B.to_csv(outdir / "T5_bimodality_regions.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    if len(B):
        ax[0].scatter(B.ashman_D, -np.log10(B.p), c=[mc[m] for m in B.mouse], s=14)
        ax[0].axvline(2, color="0.5", ls="--", lw=0.6); ax[0].set_xlabel("Ashman D (mode separation)"); ax[0].set_ylabel("-log10 p (LRT)")
    s = E[(E.comp == "ref") & (E.amp > 0)]
    for m, g in s.groupby("mouse"):
        ax[1].hist(np.log10(g.amp), bins=25, histtype="step", color=mc[m], density=True)
    ax[1].set_xlabel("log10 reference event amplitude (pipeline events)")
    savefig(fig, "T5_bimodality", outdir)

    # ---------------- T6 expression proxy
    R["T6_F0_vs_thalf"] = within_mouse_spearman(P, "F0_light_ref", "t_half_ref", alternative="greater")
    R["T6_F0_vs_r"] = within_mouse_spearman(P, "F0_light_ref", "r_ref_branch", alternative="greater")
    R["T6_F0_vs_noise"] = {**within_mouse_spearman(P, "F0_light_ref", "noise_ref", alternative="less"), "kind": "descriptive"}
    Pg = P.assign(mouse=P.mouse + "|" + P.reference.astype(str))     # same mouse AND same reference type
    R["T6_F0_vs_thalf_same_reftype"] = {**within_mouse_spearman(Pg, "F0_light_ref", "t_half_ref", alternative="greater"),
                                        "kind": "descriptive", "note": "within mouse x reference type (soma vs proximal "
                                        "trunk differ in both brightness and decay, which could fake the T6 relation)"}
    R["T6_F0raw_vs_thalf"] = {**within_mouse_spearman(P, "F0_ref_raw", "t_half_ref", alternative="greater"), "kind": "descriptive",
                              "note": "same with raw F0 (includes the detector offset)"}
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for j, y in enumerate(["t_half_ref", "r_ref_branch", "noise_ref"]):
        for m, g in P.groupby("mouse"):
            ax[j].plot(g.F0_light_ref, g[y], "o-", ms=3, lw=0.6, color=mc[m], label=m.replace("rbp4_", ""))
        ax[j].set_xlabel("reference F0 above dark (counts)"); ax[j].set_ylabel(y)
    ax[0].legend(fontsize=5, frameon=False)
    savefig(fig, "T6_expression_proxy", outdir)

    # ---------------- T7 frame rate
    R["T7_rate_vs_r"] = within_mouse_spearman(P, "rate", "r_ref_branch", alternative="two-sided")
    R["T7_rate_vs_thalf"] = within_mouse_spearman(P, "rate", "t_half_ref", alternative="two-sided")
    R["T7_rate_vs_rise"] = within_mouse_spearman(P, "rate", "rise_ref", alternative="two-sided")
    P["dr_dec_sub"] = P.r_dec_sub - P.r_ref_branch; P["dr_dec_avg"] = P.r_dec_avg - P.r_ref_branch
    P["dthalf_dec"] = P.t_half_ref_dec - P.t_half_ref
    R["T7_decimate_sub_dr"] = mouse_level(P, "dr_dec_sub", alternative="two-sided")
    R["T7_decimate_avg_dr"] = mouse_level(P, "dr_dec_avg", alternative="two-sided")
    R["T7_decimate_dthalf"] = mouse_level(P, "dthalf_dec", alternative="two-sided")
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for j, y in enumerate(["r_ref_branch", "t_half_ref"]):
        for m, g in P.groupby("mouse"):
            ax[j].plot(g.rate, g[y], "o", ms=4, color=mc[m])
        ax[j].set_xlabel("volume rate (Hz)"); ax[j].set_ylabel(y)
    strip(ax[2], P, ["dr_dec_sub", "dr_dec_avg"], ["sub-sample /2", "average pairs"], mc, ylabel="change in r(ref,branch)")
    savefig(fig, "T7_frame_rate", outdir)

    # ---------------- T8 noise vs photon expectation and position
    R["T8_noise_slope"] = {**mouse_level(P, "noise_slope_on_logF0vox", null=-0.5, alternative="two-sided"), "null": -0.5}
    R["T8_excess_noise_vs_edge"] = mouse_level(P, "excess_noise_vs_edge_rho", alternative="greater")
    R["T8_excess_noise_vs_zedge"] = mouse_level(P, "excess_noise_vs_zedge_rho", alternative="greater")
    RG = pd.DataFrame([{"behavior_base": r["behavior_base"], "mouse": r["mouse"], **g} for r in per for g in r["regions"]])
    RG.to_csv(outdir / "regions_noise_geometry.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    RG["F0v"] = np.maximum(RG.F0, 1.0) * RG.vox                     # same clipping as the per-run test
    for m, g in RG.groupby("mouse"):
        ax[0].scatter(g.F0v, g.noise, s=8, color=mc[m], label=m.replace("rbp4_", ""))
    xx = np.logspace(np.log10(RG.F0v.min()), np.log10(RG.F0v.max()), 10)
    ax[0].plot(xx, np.exp(np.nanmedian(np.log(RG.noise) + 0.5 * np.log(RG.F0v))) * xx ** -0.5, "k--", lw=0.8, label="photon noise (-1/2)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("F0 above dark x voxels"); ax[0].set_ylabel("dF/F noise above dark (robust SD)")
    ax[0].legend(fontsize=5, frameon=False)
    RG["excess"] = np.log(RG.noise) + 0.5 * np.log(RG.F0v)
    RG["excess"] -= RG.groupby("behavior_base").excess.transform("mean")
    ax[1].scatter(RG.edge_frac, RG.excess, s=8, c=[mc[m] for m in RG.mouse]); ax[1].set_xlabel("fraction of region on tube boundary")
    ax[1].set_ylabel("excess log noise (within run)")
    savefig(fig, "T8_noise_position", outdir)

    # ---------------- T9 behavior
    BH_rows = []
    for r in per:
        for bn, v in r.get("behavior", {}).items():
            BH_rows.append({"behavior_base": r["behavior_base"], "mouse": r["mouse"], "behavior": bn, **v})
    BHd = pd.DataFrame(BH_rows)
    if len(BHd):
        BHd.to_csv(outdir / "T9_behavior_runs.csv", index=False)
        for bn, g in BHd.groupby("behavior"):
            for c in ("ref", "trunk", "branch"):
                for k in ("rate", "amp"):
                    col = f"{k}_log2_{c}"
                    if col in g:
                        R[f"T9_{bn}_{k}_{c}"] = mouse_level(g, col, alternative="two-sided")
            if "branch_over_ref_rate_log2" in g:
                R[f"T9_{bn}_branch_over_ref_rate"] = mouse_level(g, "branch_over_ref_rate_log2", alternative="two-sided")
        fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
        for j, bn in enumerate(["pupil", "whisking", "accelerometer"]):
            g = BHd[BHd.behavior == bn]
            if len(g):
                cols = [c for c in ["rate_log2_ref", "rate_log2_branch", "amp_log2_ref", "amp_log2_branch", "branch_over_ref_rate_log2"] if c in g]
                strip(ax[j], g, cols, [c.replace("_log2", "").replace("branch_over_ref_rate", "br/ref rate") for c in cols], mc,
                      ylabel="log2 active / quiet")
                ax[j].tick_params(axis="x", rotation=40)
            ax[j].set_title(bn, loc="left")
        savefig(fig, "T9_behavior_state", outdir)

    # ---------------- T10 bleaching
    R["T10_bleach_pct_per_min"] = mouse_level(P, "bleach_pct_per_min", alternative="less")
    P["dr_detrend"] = P.r_ref_branch_detrended - P.r_ref_branch
    R["T10_detrend_dr"] = mouse_level(P, "dr_detrend", alternative="less")
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    strip(ax[0], P, ["bleach_pct_per_min"], ["baseline change"], mc, ylabel="% per minute")
    strip(ax[1], P, ["dr_detrend"], ["sliding-baseline r - global r"], mc, ylabel="change in r(ref,branch)")
    savefig(fig, "T10_bleaching", outdir)

    # ---------------- T11 other
    R["T11a_rate_vs_distance"] = mouse_level(P, "rate_dist_rel_slope_per100um", alternative="less")
    R["T11b_voxel_photon_slope"] = {**mouse_level(P, "vox_loglog_slope", null=1.0, alternative="two-sided"), "null": 1.0}
    R["T11c_sat_vs_r"] = within_mouse_spearman(P, "sat_index", "r_ref_branch", alternative="greater")
    R["T11c_thalf_vs_r"] = within_mouse_spearman(P, "t_half_ref", "r_ref_branch", alternative="greater")
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for m, g in RG.groupby("mouse"):
        ax[0].scatter(g.dist, g.rate3_per_min, s=8, color=mc[m])
    ax[0].set_xlabel("distance from reference (um)"); ax[0].set_ylabel("3-sigma events / min")
    strip(ax[1], P, ["vox_loglog_slope"], ["voxel var~mean slope"], mc, null=1.0)
    for m, g in P.groupby("mouse"):
        ax[2].plot(g.sat_index, g.r_ref_branch, "o", ms=4, color=mc[m])
    ax[2].set_xlabel("saturation index rho(t1/2, amp)"); ax[2].set_ylabel("r(ref,branch)")
    savefig(fig, "T11_other", outdir)

    # ---------------- FDR within families + verdicts
    for fam in sorted({TESTS[k]["family"] for k in TESTS}):
        import re
        tid = fam.split("_")[0]
        keys = [k for k in R if re.match(rf"{tid}(?!\d)", k) and R[k].get("kind") != "descriptive" and np.isfinite(R[k].get("p", np.nan))]
        qs = bh([R[k]["p"] for k in keys])
        for k, q in zip(keys, qs):
            R[k]["q"] = float(q)
    for k, r in R.items():
        pred = r.get("alt", "two-sided")              # the alternative fixed when the test was called
        r["pred"] = pred
        r["verdict"] = "descriptive" if r.get("kind") == "descriptive" else verdict(r, pred)
    res = {"set": name, "version": __version__, "deviations_from_prespecification": DEVIATIONS, "root": str(root), "n_runs": int(len(P)), "n_mice": int(P.mouse.nunique()),
           "runs": P.behavior_base.tolist(), "mice": sorted(P.mouse.unique()),
           "sanity_max_abs_diff_r_vs_run_metrics": max_dev,
           "same_session_runs": {f"{m} {dt}": g.behavior_base.tolist() for (m, dt), g in P.groupby(["mouse", "date"]) if len(g) > 1},
           "tests": TESTS, "literature": LIT, "results": R}
    P.to_csv(outdir / "per_run_features.csv", index=False)
    E.drop(columns=[c for c in ["ri"] if c in E]).to_csv(outdir / "events_features.csv.gz", index=False, compression="gzip")
    (outdir / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: None if o is None else float(o) if np.isscalar(o) else str(o)))
    log(f"{name}: analysis done", {"n_runs": len(P), "n_mice": int(P.mouse.nunique()), "n_results": len(R), "outdir": str(outdir)})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["curated", "screening", "both"], default="both")
    ap.add_argument("--force-extract", action="store_true")
    a = ap.parse_args(argv)
    sets = ["curated", "screening"] if a.set == "both" else [a.set]
    for s in sets:
        print(f"== {s}"); analyze_set(s, a.force_extract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
