#!/usr/bin/env python
"""normalized_plots.py - coherence figures and cohort trace sheets at ONE amplitude scale.

Why: each run's own coherence figure scales its traces to that run's range, so a cell with
small signals looks as big as one with large signals. Here every curated run is drawn at
the same dF/F per inch (common scale) or the same noise-SD per inch (z-scored), so
amplitudes (or signal-to-noise) can be compared between cells by eye.

Curated runs = the rows of stats/cohort_metrics.csv (built by cohort_stats.py: unmarked
runs with current regions). Traces are computed exactly like the coherence figure and
run_metrics.py: mean raw fluorescence of the region's voxels in <stem>_clean.tif,
dF/F with F0 = 10th percentile, ignored regions (<stem>_ignore.json) left out. Nothing
in the run folders is written; everything goes to stats/normalized/ (+ stats/plot_scale.json).

Steps
  1. traces       per region dF/F + noise SD -> stats/normalized/normalized_traces.npz,
                  stats/normalized/trace_amplitudes.csv
  2. scale        stats/plot_scale.json, chosen ONCE for the cohort (kept if it exists;
                  --write-scale recomputes it):
                    dff_per_lane = 99th percentile over all curated traces of the trace's
                                   peak (max) dF/F, rounded up to a readable value
                    z_per_lane   = the same for the peak of trace / noise SD
                    inch_per_lane, scale bars
  3. figures      stats/normalized/<behavior_base>_coherence_common.png/.pdf and
                  _coherence_zscore.png/.pdf (segment_event_coherence.py --scale ...)
  4. sheets       stats/normalized/sheet_common.png/.pdf and sheet_zscore.png/.pdf: every
                  curated run's reference + branch traces on one page, same scale, same
                  seconds per inch, sorted by r(reference, branch) high -> low.

  PY=/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/.venv311/bin/python
  $PY code/STEP8_stats/normalized_plots.py                 # all steps
  $PY code/STEP8_stats/normalized_plots.py --sheets-only   # re-render sheets from the cache
  $PY code/STEP8_stats/normalized_plots.py --write-scale   # re-choose the cohort scale
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_CODE_ROOT = HERE.parents[1]
ROOT = Path(os.environ["FEMTO_ROOT"]).resolve() if os.environ.get("FEMTO_ROOT") else _CODE_ROOT
sys.path.insert(0, str(_CODE_ROOT / "code")); sys.path.insert(0, str(_CODE_ROOT / "code/STEP7_workflow"))
from common.plot_scale import robust_noise, nice_ceil, nice_bar, default_scale_file, load_scale  # noqa: E402
from common.voxel import resolve_voxel                                                          # noqa: E402

__version__ = "1.0.0"
OUT = ROOT / "stats" / "normalized"
COHERENCE_TOOL = _CODE_ROOT / "code/extra/segment_event_coherence.py"
LOG = ROOT / "auto_pipeline" / "logs" / "normalized_plots_v7.jsonl"
INCH_PER_LANE = 0.55          # per-run coherence figure: one region lane, in inches


def log(what, result):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"time": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "stage": "normalized_plots", "what": what, "result": result}, default=str) + "\n")


def dff(t, f0_pct=10.0):
    f0 = np.percentile(t, f0_pct); return (t - f0) / max(f0, 1e-6)


def curated_runs() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "stats" / "cohort_metrics.csv")
    df["metrics_path"] = [ROOT / m for m in df["metrics_file"]]
    df["run_dir"] = [p.parent for p in df["metrics_path"]]
    df["stem"] = [p.name[: -len("_metrics.json")] for p in df["metrics_path"]]
    return df


# ----------------------------------------------------------------------------- 1. traces
def compute_traces(df: pd.DataFrame) -> dict:
    import tifffile
    from common.regions import apply_ignore
    from run_metrics import region_names                      # same naming rule as the stats
    out = {}
    for _, r in df.iterrows():
        bb = r["behavior_base"]; rd = r["run_dir"]; stem = r["stem"]
        m = json.loads(Path(r["metrics_path"]).read_text())
        seg_p = rd / f"{stem}_segments_final.tif"
        stack = tifffile.imread(rd / f"{stem}.tif"); seg = tifffile.imread(seg_p)
        T = stack.shape[0]; flat = stack.reshape(T, -1)
        labels = sorted(int(v) for v in np.unique(seg) if v > 0)
        names = region_names(seg_p.with_suffix(".json"), labels)
        seg, _ign, ign_labels = apply_ignore(seg, names, seg_p)
        labels = [l for l in labels if l not in ign_labels]
        regs = m["regions"]
        assert {int(k) for k in regs} == set(labels), f"{bb}: metrics regions != current regions"
        if m.get("reference") == "soma":
            ref = next(int(k) for k, v in regs.items() if v["compartment"] == "soma")
        else:
            ref = next(int(k) for k, v in regs.items() if v["name"] == m.get("reference_region"))
        tr = {l: dff(flat[:, np.flatnonzero((seg == l).ravel())].mean(1).astype(np.float64)) for l in labels}
        out[bb] = {"T": T, "rate": float(r["frame_rate_hz"]), "ref": ref,
                   "labels": labels, "names": {l: names[l] for l in labels},
                   "comp": {l: regs[str(l)]["compartment"] for l in labels},
                   "dist": {l: regs[str(l)].get("distance_um") for l in labels},
                   "traces": tr, "noise": {l: robust_noise(tr[l]) for l in labels}}
        print(f"  traces {bb}: {len(labels)} regions, T={T}")
        del stack, flat
    return out


def save_cache(data: dict, df: pd.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {}
    rows = []
    for bb, d in data.items():
        meta[bb] = {k: d[k] for k in ("T", "rate", "ref", "labels")}
        meta[bb].update({"names": {str(k): v for k, v in d["names"].items()},
                         "comp": {str(k): v for k, v in d["comp"].items()},
                         "dist": {str(k): v for k, v in d["dist"].items()},
                         "noise": {str(k): v for k, v in d["noise"].items()}})
        for l in d["labels"]:
            t = d["traces"][l]; arrays[f"{bb}__{l}"] = t.astype(np.float32)
            sd = d["noise"][l]
            rows.append({"behavior_base": bb, "mouse": df.set_index("behavior_base").at[bb, "mouse"],
                         "region": d["names"][l], "label": l, "compartment": d["comp"][l],
                         "is_reference": l == d["ref"], "distance_um": d["dist"][l],
                         "peak_dff_max": float(t.max()), "peak_dff_p99": float(np.percentile(t, 99)),
                         "noise_sd_dff": sd, "peak_z_max": float(t.max() / sd),
                         "peak_z_p99": float(np.percentile(t, 99) / sd)})
    np.savez_compressed(OUT / "normalized_traces.npz", meta=json.dumps(meta), **arrays)
    pd.DataFrame(rows).to_csv(OUT / "trace_amplitudes.csv", index=False)


def load_cache() -> dict:
    z = np.load(OUT / "normalized_traces.npz")
    meta = json.loads(str(z["meta"]))
    data = {}
    for bb, m in meta.items():
        labels = [int(l) for l in m["labels"]]
        data[bb] = {"T": m["T"], "rate": m["rate"], "ref": int(m["ref"]), "labels": labels,
                    "names": {int(k): v for k, v in m["names"].items()},
                    "comp": {int(k): v for k, v in m["comp"].items()},
                    "dist": {int(k): v for k, v in m["dist"].items()},
                    "noise": {int(k): v for k, v in m["noise"].items()},
                    "traces": {l: z[f"{bb}__{l}"].astype(np.float64) for l in labels}}
    return data


# ----------------------------------------------------------------------------- 2. scale
def choose_scale(data: dict, df: pd.DataFrame) -> dict:
    peaks = np.array([d["traces"][l].max() for d in data.values() for l in d["labels"]])
    zpk = np.array([d["traces"][l].max() / d["noise"][l] for d in data.values() for l in d["labels"]])
    p_dff, p_z = float(np.percentile(peaks, 99)), float(np.percentile(zpk, 99))
    lane, zlane = nice_ceil(p_dff), nice_ceil(p_z)
    return {
        "version": __version__, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "created_by": "code/STEP8_stats/normalized_plots.py",
        "method": "lane = 99th percentile, over every curated region trace, of that trace's peak (max) "
                  "value; rounded up to a readable number. Chosen once for the cohort; figures read it.",
        "dff_definition": "mean raw F of the region voxels in <stem>_clean.tif; dF/F with F0 = 10th percentile",
        "noise_definition": "1.4826 x MAD of (trace - 15-frame running mean) of the dF/F trace "
                            "(= run_metrics.robust_noise, the noise behind soma_snr)",
        "dff_per_lane": lane, "dff_peak_p99_raw": round(p_dff, 4),
        "dff_peak_median": round(float(np.median(peaks)), 4), "dff_peak_max": round(float(peaks.max()), 4),
        "z_per_lane": zlane, "z_peak_p99_raw": round(p_z, 3),
        "z_peak_median": round(float(np.median(zpk)), 3), "z_peak_max": round(float(zpk.max()), 3),
        "inch_per_lane": INCH_PER_LANE,
        "dff_scale_bar": 0.5 if 1.0 <= lane <= 3.0 else nice_bar(lane),
        "z_scale_bar": nice_bar(zlane),
        "n_traces": int(len(peaks)), "n_runs": int(len(data)), "n_mice": int(df["mouse"].nunique()),
        "runs": sorted(data),
    }


# ----------------------------------------------------------------------------- 3. per-run figures
def build_run_figures(df: pd.DataFrame, scale_file: Path, modes=("common", "zscore")):
    from coherence_with_behavior import region_names_and_order
    from common.regions import ignored_names
    py = sys.executable
    done = []
    for _, r in df.iterrows():
        bb, rd, stem = r["behavior_base"], r["run_dir"], r["stem"]
        stack = rd / f"{stem}.tif"; lm = rd / f"{stem}_segments_final.tif"
        vx = resolve_voxel(stack, None, quiet=True)
        disp = {"mask": False, "edge_um": 2.0, "hide_other": True}
        side = rd / f"{stem}_coherence.png.display.json"         # match the run's own figure
        if side.exists():
            try:
                disp.update(json.loads(side.read_text()))
            except Exception:
                pass
        cmd = [py, str(COHERENCE_TOOL), str(stack), str(lm), "--voxel", *[f"{v:g}" for v in vx],
               "--frame-ms", f"{1000.0 / float(r['frame_rate_hz']):.3f}"]
        order, names = region_names_and_order(lm)
        if order:
            ign = {n.lower() for n in ignored_names(lm)}
            drop = [o for o, nm in zip(order, names) if nm.lower() in ign]
            kept = [(o, nm) for o, nm in zip(order, names) if nm.lower() not in ign]
            if drop:
                cmd += ["--exclude", *map(str, drop)]
            cmd += ["--order", *[str(o) for o, _ in kept], "--names", *[nm for _, nm in kept]]
        cmd += ["--mask"] if disp.get("mask") else ["--no-mask"]
        cmd += ["--hide-other"] if disp.get("hide_other") else ["--show-other"]
        cmd += ["--edge-um", f"{float(disp.get('edge_um', 2.0)):g}"]
        for mode in modes:
            pre = OUT / f"{bb}_coherence_{mode}"
            c = cmd + ["--scale", mode, "--scale-file", str(scale_file), "--out-prefix", str(pre)]
            t0 = time.time()
            p = subprocess.run(c, cwd=str(ROOT), capture_output=True, text=True)
            ok = p.returncode == 0 and pre.with_suffix(".png").exists()
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("trace panel")), "")
            print(f"  figure {pre.name}: {'ok' if ok else 'FAILED'} ({time.time() - t0:.0f} s) {line}")
            if not ok:
                print(p.stderr[-2000:])
            done.append({"run": bb, "mode": mode, "ok": ok, "panel": line})
    return done


# ----------------------------------------------------------------------------- 4. sheets
def same_session_groups(df):
    g = df.groupby(["mouse", "date"])["short"].apply(list)
    return [v for v in g if len(v) > 1]


def draw_sheet(data: dict, df: pd.DataFrame, cfg: dict, mode: str, stem: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "pdf.fonttype": 42})
    d2 = df.set_index("behavior_base")
    order = sorted(data, key=lambda b: -float(d2.at[b, "r_soma_branch"]))
    n = len(order)
    lane = cfg["dff_per_lane"] if mode == "common" else cfg["z_per_lane"]
    bar = cfg["dff_scale_bar"] if mode == "common" else cfg["z_scale_bar"]
    unit = "ΔF/F" if mode == "common" else "SD"
    lane_in = 0.75                      # sheet: one lane per run, reference + branches overlaid
    head_in, foot_in = 1.25, 1.35
    W = 11.0; L_in, R_in = 1.9, 2.3
    H = head_in + n * 1.1 * lane_in + foot_in
    tmax = max(d["T"] / d["rate"] for d in data.values())
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L_in / W, foot_in / H, (W - L_in - R_in) / W, (H - head_in - foot_in) / H])
    per_run = 1.1 * lane                # vertical units per run (1 lane + 10% gap)
    ytop = n * per_run
    bcols = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]
    for k, bb in enumerate(order):
        d = data[bb]; t = np.arange(d["T"]) / d["rate"]
        yb0 = ytop - (k + 1) * per_run
        y0 = yb0 + 0.12 * lane                                # trace baseline
        ref = d["ref"]
        sc = (lambda l: d["traces"][l]) if mode == "common" else (lambda l: d["traces"][l] / d["noise"][l])
        if k % 2 == 0:
            ax.axhspan(yb0, yb0 + per_run, color="0.94", lw=0, zorder=0)
        brs = [l for l in d["labels"] if d["comp"][l] == "branch"]
        for j, l in enumerate(brs):
            ax.plot(t, sc(l) + y0, color=bcols[j % len(bcols)], lw=0.55, alpha=0.8, zorder=3)
        ax.plot(t, sc(ref) + y0, color="k", lw=0.65, zorder=4)
        row = d2.loc[bb]
        refkind = "soma" if row["reference"] == "soma" else "proximal trunk"
        ax.text(-0.012 * tmax, yb0 + 0.55 * per_run,
                f"{row['short'].split(' (')[0]}\nr(ref,branch) = {float(row['r_soma_branch']):.2f}\n"
                f"ref: {refkind}; {len(brs)} branch{'es' if len(brs) != 1 else ''}",
                ha="right", va="center", fontsize=8, linespacing=1.2)
        def stats(ls):
            if not ls:
                return np.nan, np.nan
            pk = np.mean([np.percentile(d["traces"][l], 99) for l in ls])
            sd = np.mean([d["noise"][l] for l in ls])
            return pk, sd
        rp, rs = stats([ref]); bp, bs = stats(brs)
        ax.text(tmax * 1.045, yb0 + 0.55 * per_run,
                (f"ref    peak {rp:.2f}  SD {rs:.3f}\nbranch peak {bp:.2f}  SD {bs:.3f}" if mode == "common" else
                 f"ref    peak/SD {rp / rs:5.0f}\nbranch peak/SD {bp / bs:5.0f}"),
                fontsize=7.5, va="center", family="monospace", linespacing=1.4)
    ax.set_xlim(0, tmax); ax.set_ylim(0, ytop)
    ax.set_yticks([]); ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_xlabel("time (s)", fontsize=10); ax.tick_params(axis="x", labelsize=9)
    ax.text(tmax * 1.045, ytop + 0.08 * lane,
            ("peak = 99th pct ΔF/F; SD = noise SD (ΔF/F)" if mode == "common"
             else "peak/SD = 99th pct ΔF/F / noise SD") + "\nbranch: mean over branches",
            fontsize=7, va="bottom", color="0.3")
    # scale bar (vertical) at the right edge of the first row
    xb = tmax * 1.008; yb = 0.12 * lane                       # last row, right edge
    ax.plot([xb, xb], [yb, yb + bar], color="k", lw=2, clip_on=False, solid_capstyle="butt")
    ax.text(xb + 0.006 * tmax, yb + bar / 2, f"{bar:g} {unit}", va="center", ha="left",
            fontsize=8, clip_on=False)
    n_m = df["mouse"].nunique()
    title = ("All curated runs at one amplitude scale (ΔF/F)" if mode == "common"
             else "All curated runs, z-scored (each trace / its own noise SD)")
    sub = (f"{n} runs, {n_m} mice. Each row = one run: black = reference region (soma, or proximal trunk if the soma "
           f"is below the scan), colors = branch regions (one line each), overlaid.\n"
           f"Same scale on every row: one lane = {lane:g} {unit} = {lane_in:.2f} in, same seconds per inch. "
           f"Sorted by r(reference, branch), highest first. No per-trace rescaling.")
    if mode == "zscore":
        sub += "\nSD = robust frame-to-frame noise of that trace (1.4826 x MAD of the high-pass residual); " \
               "the height shows signal-to-noise, not amplitude."
    fig.text(0.02, 1 - 0.18 / H, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.50 / H, sub, fontsize=8.5, va="top", linespacing=1.3)
    groups = same_session_groups(df)
    foot = ("Caveats: runs of one session may image the same cell (not recorded): "
            + "; ".join(" / ".join(s.split(" (")[0] for s in g) for g in groups)
            + ".\nThe sort is descriptive (runs within a mouse are not independent). Traces from "
              "<stem>_clean.tif, dF/F with F0 = 10th percentile, ignored regions left out. "
              f"Lane value from stats/plot_scale.json (99th percentile of {cfg['n_traces']} curated "
              f"traces' peaks).")
    import textwrap
    foot = "\n".join(textwrap.fill(par, 175) for par in foot.split("\n"))
    fig.text(0.02, 0.12 / H, foot, fontsize=7, va="bottom", color="0.25")
    fig.savefig(stem.with_suffix(".png"), dpi=150); fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return order


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-scale", action="store_true", help="re-choose stats/plot_scale.json")
    ap.add_argument("--sheets-only", action="store_true", help="use the trace cache, no figures")
    ap.add_argument("--no-figures", action="store_true", help="skip the per-run coherence figures")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    df = curated_runs()
    print(f"curated runs: {len(df)} from {df['mouse'].nunique()} mice")
    if a.sheets_only and (OUT / "normalized_traces.npz").exists():
        data = load_cache()
    else:
        data = compute_traces(df); save_cache(data, df)
        log("traces", f"{sum(len(d['labels']) for d in data.values())} region traces from {len(data)} runs "
                      f"/ {df['mouse'].nunique()} mice -> stats/normalized/normalized_traces.npz, trace_amplitudes.csv")
    sf = default_scale_file(ROOT)
    cfg = load_scale(sf)
    if cfg is None or a.write_scale:
        cfg = choose_scale(data, df)
        sf.write_text(json.dumps(cfg, indent=2))
        log("scale", {k: cfg[k] for k in ("dff_per_lane", "dff_peak_p99_raw", "z_per_lane", "z_peak_p99_raw",
                                          "inch_per_lane", "dff_scale_bar", "z_scale_bar", "n_traces", "n_runs", "n_mice")})
    print(f"scale: {cfg['dff_per_lane']} dF/F per lane, {cfg['z_per_lane']} SD per lane ({sf})")
    if not (a.sheets_only or a.no_figures):
        res = build_run_figures(df, sf)
        log("run_figures", f"{sum(x['ok'] for x in res)}/{len(res)} ok -> stats/normalized/<behavior_base>_coherence_{{common,zscore}}.png/.pdf")
    for mode in ("common", "zscore"):
        order = draw_sheet(data, df, cfg, mode, OUT / f"sheet_{mode}")
        print(f"sheet_{mode}: {len(order)} runs")
    log("sheets", f"stats/normalized/sheet_common.png/.pdf, sheet_zscore.png/.pdf ({len(data)} runs, sorted by r_soma_branch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
