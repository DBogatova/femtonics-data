#!/usr/bin/env python3
"""Combined behavior + calcium plots for FEMTONICS runs.

Adapted from apical-dendrites-2025/code/Behavior-Analysis/behavior_plots.py,
which was written for the SCAPE/Andor rig. That original is left untouched; this
is the Femtonics version. Everything it needs per run is looked up in
`behavior_imaging_master.csv` (from STEP1_extract/match_behavior_imaging.py), so
there are no hardcoded run constants.

WHAT WAS WRONG FOR FEMTONICS IN THE ORIGINAL, AND WHAT THIS DOES INSTEAD
-----------------------------------------------------------------------
1. FRAME_RATE was a module constant (6 Hz).
   The Femtonics volume rate varies PER RUN: across the 44 behavior-matched runs
   it spans 4.523-10.836 Hz (median 5.871). Assuming 6 Hz puts the end of the Ca
   time axis off by a median of 23.3 s and up to 193.5 s; 37 of 44 runs would be
   off by more than 5 s. Here the rate comes from `frame_rate_hz` in the master
   CSV, derived from .mesc metadata (1000 / TStepInMs), so it is authoritative.
   --frame-rate overrides it.

2. PATHS assumed <root>/<YYYY-MM-DD>/<mouse>/<runN>/{behavior,trigger}.
   Femtonics layout is <root>/<mouse>/<MM-DD-YYYY>/{behavior,trigger}, with
   imaging under <root>/<mouse>/<MM-DD-YYYY>/preprocessed/<runN>/. Note the date
   is MM-DD-YYYY in folder names but yy-mm-dd inside filenames.

3. THE CAMERA->IMAGING OFFSET was computed by globbing "*_trigger.csv" and taking
   the first match. In this layout every run of a session shares one trigger/
   folder, so that glob can return ANOTHER run's file; the within-session offset
   spread reaches 6.8 s. Here it is read from the run's own behavior .mat
   (`settings.aligned_time_s`), which is exact and per-run:
       offset = -aligned_time_s[0]
   The offset IS required: pupil/whisker come off the .mat on the CAMERA clock
   (frame index / 10 Hz), while the accelerometer is already on aligned_time_s.

4. CA/ACh came from raw "-reslice-green.tif" / "-reslice-red.tif" stacks.
   These runs are single-channel snake scans, so there is no ACh channel. Ca is
   taken from the per-segment dF/F CSVs written by STEP5_traces when present
   (already computed, costs milliseconds, and matches what the coherence and
   branchprop figures show); reading the 4D tif is available but opt-in.

5. CROP_START_SECONDS defaulted to 12 s.
   Here the imaging window already starts at aligned t=0 and the 2P laser on/off
   transitions sit OUTSIDE it, so there is no startup artefact to crop. Default
   is 0; --crop-start still exists.

Retained deliberately: accelerometer y-limit 0.25, `accel_mag` as the signal,
Arial with pdf.fonttype 42 so PDF text stays editable.

Usage
-----
    python code/STEP7_behavior/behavior_plots.py --list
    python code/STEP7_behavior/behavior_plots.py --run rbp4_141_phpeb_26-06-25_Run007
    python code/STEP7_behavior/behavior_plots.py --all --out-dir figures/behavior
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe; this script never calls plt.show()
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42

# ---- project-wide conventions (do not change casually) ----------------------
BEHAVIOR_HZ = 10.0     # Basler behavior camera
ACCEL_YMAX = 0.25       # accelerometer y-limit, same in every plot script
C_CA, C_PUPIL, C_WHISK, C_ACCEL = "green", "blue", "orange", "purple"
MASTER_CSV = "behavior_imaging_master.csv"


# =============================================================================
# run resolution
# =============================================================================
def load_master(root: Path) -> pd.DataFrame:
    p = root / MASTER_CSV
    if not p.exists():
        raise SystemExit(
            f"ERROR: {p} not found.\n"
            f"       Generate it:  python code/STEP1_extract/match_behavior_imaging.py")
    d = pd.read_csv(p)
    d = d[d["behavior_base"].notna()].copy()
    if d.empty:
        raise SystemExit(f"ERROR: {p} has no rows with a behavior_base "
                         f"(no behavior<->imaging matches).")
    return d


def run_folder_name(run_number) -> str:
    """behavior run 7 -> imaging folder 'run7' (lowercase, unpadded)."""
    return f"run{int(float(run_number))}"


def resolve_paths(root: Path, row: pd.Series) -> dict:
    """Turn a master-CSV row into concrete paths. The CSV stores BARE filenames
    for the behavior artefacts and a root-relative path for extracted_tif."""
    sess = root / str(row["mouse"]) / str(row["date"])
    return {
        "session": sess,
        "behavior_mat": sess / "behavior" / str(row["behavior_mat"]),
        "behavior_csv": sess / "behavior" / str(row["behavior_csv"]),
        "accel_csv": (sess / "trigger" / str(row["accel_csv"]))
                     if pd.notna(row.get("accel_csv")) else None,
        "run_dir": sess / "preprocessed" / run_folder_name(row["behavior_run_number"]),
        "extracted_tif": (root / str(row["extracted_tif"]))
                         if pd.notna(row.get("extracted_tif")) else None,
    }


# =============================================================================
# calcium
# =============================================================================
def _finish_ca(ca: np.ndarray, fps: float, crop_start: float, src: str):
    ca = np.asarray(ca, dtype=float)
    t = np.arange(len(ca)) / fps
    m = t >= crop_start
    print(f"  Ca: {len(ca)} frames from {src}; {t[-1]:.1f}s at {fps:.3f} Hz")
    return t[m] - crop_start, ca[m]


def load_calcium(paths: dict, fps: float, ca_source: str, crop_start: float):
    """Return (time, ca) or (None, None).

    'segments' (default): mean of the per-segment dF/F CSVs from STEP5_traces.
    'tif': mean over live voxels of the 4D stack (reads GBs).
    """
    if ca_source == "none":
        return None, None

    if ca_source in ("segments", "auto"):
        rd = paths["run_dir"]
        if rd.is_dir():
            combined = sorted(rd.glob("*_segment_traces.csv"))
            if combined:
                df = pd.read_csv(combined[0])
                num = [c for c in df.columns if c.lower() != "frame"]
                if num:
                    return _finish_ca(df[num].mean(axis=1).to_numpy(), fps,
                                      crop_start, combined[0].name)
            segs = sorted(rd.glob("*_seg[0-9][0-9].csv"))
            if segs:
                cols = []
                for s in segs:
                    df = pd.read_csv(s)
                    num = [c for c in df.columns if c.lower() != "frame"]
                    if num:
                        cols.append(df[num].mean(axis=1).to_numpy())
                if cols:
                    n = min(len(c) for c in cols)
                    ca = np.mean([c[:n] for c in cols], axis=0)
                    return _finish_ca(ca, fps, crop_start, f"{len(cols)} seg CSVs")
        if ca_source == "segments":
            print(f"  Ca: no per-segment CSV under {rd}")
            return None, None

    tif = paths["extracted_tif"]
    if tif is None or not tif.exists():
        print("  Ca: no extracted_tif available")
        return None, None
    try:
        import tifffile
    except ImportError:
        print("  Ca: tifffile not installed; cannot read the stack")
        return None, None
    print(f"  Ca: reading {tif.name} (whole stack)")
    store = tifffile.memmap(str(tif), mode="r")
    T = store.shape[0]
    sample = np.asarray(store[: min(100, T)]).astype(np.float32)
    tmean = sample.mean(axis=0)
    live = np.flatnonzero(tmean > np.percentile(tmean, 5))
    del sample, tmean
    f0_vals = [np.asarray(store[t]).astype(np.float32).ravel()[live].mean()
               for t in range(0, min(500, T))]
    f0 = float(np.percentile(f0_vals, 10))
    dff = np.empty(T, dtype=np.float32)
    for a in range(0, T, 50):
        b = min(a + 50, T)
        fr = np.asarray(store[a:b]).astype(np.float32)
        for i in range(fr.shape[0]):
            dff[a + i] = (fr[i].ravel()[live].mean() - f0) / (f0 + 1e-6) * 100
    del store
    return _finish_ca(dff, fps, crop_start, tif.name)


# =============================================================================
# behavior
# =============================================================================
def load_behavior(paths: dict, crop_start: float, apply_offset: bool):
    mat = paths["behavior_mat"]
    if not mat.exists():
        print(f"  behavior MAT not found: {mat}")
        return None, None, None
    md = loadmat(mat)
    pupil = gaussian_filter1d(md["pupil"]["pupil_raw"][0][0].flatten(), sigma=2)
    whisk = gaussian_filter1d(md["whisker"]["whisker_smooth_long"][0][0].flatten(), sigma=3)
    n = len(pupil)
    time = np.arange(n) / BEHAVIOR_HZ     # CAMERA clock -> hence the offset below

    offset = 0.0
    if apply_offset:
        try:
            at = md["settings"]["aligned_time_s"][0][0].ravel()
            if at.size:
                offset = float(-at[0])
                print(f"  camera->imaging offset = {offset:.3f}s (settings.aligned_time_s)")
        except Exception:
            print("  WARNING: settings.aligned_time_s missing; offset left at 0, so "
                  "pupil/whisker may be misaligned against Ca and accelerometer.")
    else:
        print("  camera->imaging offset DISABLED by request")

    total = offset + crop_start
    pre = (time >= offset) & (time < total) if total > offset else np.zeros(n, bool)
    pmax = np.max(pupil[pre]) if pre.any() else np.max(pupil)
    wmax = np.max(np.abs(whisk[pre])) if pre.any() else np.max(np.abs(whisk))
    pupil = pupil / (pmax + 1e-6)
    whisk = whisk / (wmax + 1e-6)

    m = time >= total
    print(f"  behavior: {n} samples at {BEHAVIOR_HZ:.0f} Hz; {int(m.sum())} kept")
    return time[m] - total, pupil[m], whisk[m]


def load_accel(paths: dict, crop_start: float):
    p = paths["accel_csv"]
    if p is None or not p.exists():
        print("  accelerometer CSV not found; panel omitted")
        return None, None
    df = pd.read_csv(p)
    acc = df["accel_mag"].to_numpy() if "accel_mag" in df.columns else df.iloc[:, 1].to_numpy()
    t = (df["aligned_time_s"].to_numpy() if "aligned_time_s" in df.columns
         else df["sample"].to_numpy() / 1000.0)
    acc = gaussian_filter1d(np.abs(acc), sigma=10)
    m = t >= crop_start
    print(f"  accelerometer: {int(m.sum())} samples")
    return t[m] - crop_start, acc[m]


# =============================================================================
# figure
# =============================================================================
def plot_run(root: Path, row: pd.Series, args):
    base = str(row["behavior_base"])
    paths = resolve_paths(root, row)
    fps = args.frame_rate if args.frame_rate else float(row["frame_rate_hz"])
    src = "--frame-rate" if args.frame_rate else "master CSV (.mesc metadata)"
    print(f"\n=== {base} ===")
    print(f"  imaging {row.get('munit')} {row.get('scan_type')}  "
          f"n_t={row.get('n_t')} slices={row.get('n_slices')}")
    print(f"  frame rate = {fps:.3f} Hz  [{src}]")

    t_ca, ca = load_calcium(paths, fps, args.ca_source, args.crop_start)
    t_b, pupil, whisk = load_behavior(paths, args.crop_start, not args.no_offset)
    t_a, acc = load_accel(paths, args.crop_start)

    panels = []
    if ca is not None:
        panels.append(("Global Ca", t_ca, ca, C_CA, None))
    if pupil is not None:
        panels.append(("Pupil", t_b, pupil, C_PUPIL, None))
        panels.append(("Whisking", t_b, whisk, C_WHISK, None))
    if acc is not None:
        panels.append(("Accelerometer", t_a, acc, C_ACCEL, (0, ACCEL_YMAX)))
    panels = [p for p in panels if len(p[1])]
    if not panels:
        print("  nothing to plot")
        return None

    t_end = min(p[1][-1] for p in panels)
    cropped = []
    for lab, t, v, color, yl in panels:
        k = (t >= 0) & (t <= t_end)
        cropped.append((lab, t[k], v[k], color, yl))
    panels = cropped

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 1.6 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (lab, t, v, color, yl) in zip(axes, panels):
        ax.plot(t, v, color=color, lw=1.8)
        if yl:
            ax.set_ylim(*yl)
        ax.text(0.008, 0.94, lab, transform=ax.transAxes, ha="left", va="top",
                color=color, fontweight="bold", fontsize=20)
        ax.set_ylabel("")
        ax.grid(False)
        ax.tick_params(labelsize=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axes[-1].set_xlabel("Time (s)", fontsize=18)

    # Surface quality facts the master CSV already knows, so no reader is unaware
    # that a run lost frames or carried a warning.
    notes = [f"{fps:.3f} Hz"]
    loss = row.get("behavior_frame_loss_pct")
    if pd.notna(loss) and float(loss) > 0:
        notes.append(f"frame loss {float(loss):.2f}%")
    if pd.notna(row.get("match_confidence")):
        notes.append(str(row["match_confidence"]))
    fig.suptitle(f"{base}   ({'; '.join(notes)})", fontsize=11, y=1.005)

    plt.tight_layout(h_pad=0.4)
    out_dir = Path(args.out_dir) if args.out_dir else paths["run_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{base}_behavior_combined"
    last = None
    for f in args.formats:
        p = stem.with_suffix(f".{f}")
        plt.savefig(p, format=f, dpi=args.dpi, bbox_inches="tight")
        print(f"  wrote {p}")
        last = p
    plt.close(fig)
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description="Behavior + Ca plots for Femtonics runs.")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]),
                    help="femtonics-data root (default: two levels above this file)")
    ap.add_argument("--run", help="behavior_base, e.g. rbp4_141_phpeb_26-06-25_Run007")
    ap.add_argument("--all", action="store_true", help="plot every matched run")
    ap.add_argument("--list", action="store_true", help="list matched runs and exit")
    ap.add_argument("--frame-rate", type=float, default=None,
                    help="override the per-run imaging rate (Hz)")
    ap.add_argument("--ca-source", choices=["segments", "tif", "none", "auto"],
                    default="segments", help="Ca source (default: per-segment CSVs)")
    ap.add_argument("--crop-start", type=float, default=0.0,
                    help="seconds cut from the start (default 0: the imaging window "
                         "already excludes the laser on/off transitions)")
    ap.add_argument("--no-offset", action="store_true",
                    help="do NOT apply the camera->imaging offset (usually wrong)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: the run's preprocessed dir)")
    ap.add_argument("--formats", nargs="+", choices=["pdf", "png"], default=["pdf", "png"])
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.root)
    d = load_master(root)

    if args.list:
        cols = [c for c in ["behavior_base", "mouse", "date", "frame_rate_hz", "n_t",
                            "behavior_frame_loss_pct"] if c in d.columns]
        print(d[cols].sort_values("behavior_base").to_string(index=False))
        print(f"\n{len(d)} matched run(s). frame_rate_hz "
              f"{d.frame_rate_hz.min():.3f}-{d.frame_rate_hz.max():.3f} Hz "
              f"-- this is why the rate must not be hardcoded.")
        return 0

    if args.run:
        sel = d[d["behavior_base"] == args.run]
        if sel.empty:
            print(f"ERROR: no run '{args.run}' in {MASTER_CSV}. Try --list.", file=sys.stderr)
            return 2
    elif args.all:
        sel = d
    else:
        print("ERROR: give --run BASE, or --all, or --list.", file=sys.stderr)
        return 2

    ok = 0
    for _, row in sel.iterrows():
        try:
            if plot_run(root, row, args):
                ok += 1
        except Exception as exc:          # one bad run must not stop a batch
            print(f"  FAILED {row['behavior_base']}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    print(f"\n{ok} figure(s) written of {len(sel)} run(s).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
