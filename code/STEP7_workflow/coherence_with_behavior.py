#!/usr/bin/env python3
"""
coherence_with_behavior.py - ONE command per run to produce the target output:
the segment-event COHERENCE figure (correlation matrix + cell MIP + per-segment
dF/F with network events) stacked over the ALIGNED BEHAVIOUR panel (pupil /
whisking / accelerometer on the same imaging-frame axis), under one title block.

It orchestrates the two existing, unmodified tools and then composes their PNGs:

  (a) code/extra/segment_event_coherence.py   -> the coherence figure + the
      <stem>_coherence_network_events.csv, run on the stack + the BEST available
      labelmap (priority: *_segments_final.tif > newest hand *_segments*.tif >
      autoseg reviewed > autoseg). Skipped if the coherence outputs are already
      newer than that labelmap (and --force was not given).
  (b) behavior-tracking-daria/batch/coherence_behavior.py -> the behaviour panel
      on the coherence frame axis, identity fields derived from the master CSV.
  (c) compose -> <stem>_coherence_full.png + .pdf (coherence on top, behaviour
      below, same width, one title block with mouse/date/run, frame rate and the
      quality flags e.g. behaviour frame loss).

SAFETY (hard rule): it NEVER overwrites an existing coherence or behaviour figure
or the hand-made labelmaps. If a component figure already exists and is up to
date it is reused as-is. If a rebuild is genuinely needed but a figure with the
canonical name already exists, the rebuild is rendered into a private temp dir,
byte-compared, and used for the composite WITHOUT touching the on-disk original
(a warning is printed). Only when the canonical figure is ABSENT is it written
into the run directory. The composite <stem>_coherence_full.* is this tool's own
output and is the only thing normally written into the run dir.

CLI
---
  PY=/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/.venv311/bin/python
  $PY code/STEP7_workflow/coherence_with_behavior.py --run rbp4_141_phpeb_26-06-25_Run007
  $PY code/STEP7_workflow/coherence_with_behavior.py --run-dir <dir> [--force]
  $PY code/STEP7_workflow/coherence_with_behavior.py --all-ready   # every run at >= segments_located
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# import the shared status/resolution logic (same directory, unmodified)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import femto_status as fs  # noqa: E402

COHERENCE_TOOL = "code/extra/segment_event_coherence.py"          # relative to root
BEHAVIOR_TOOL = "/Users/daria/Desktop/behavior-tracking-daria/batch/coherence_behavior.py"

STAGE_READY_IDX = fs.STAGE_IDX["segments_located"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_behavior_base(base: str) -> tuple[str, str, str]:
    """rbp4_141_phpeb_26-06-25_Run007 -> (mouse, mat_date, run_id)."""
    m = re.search(r"^(.*)_(\d{2}-\d{2}-\d{2})_(Run\d+)$", base)
    if not m:
        raise ValueError(f"cannot parse behavior_base '{base}'")
    return m.group(1), m.group(2), m.group(3)


def run_subprocess(cmd, cwd, label):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if r.stdout:
        print("    " + r.stdout.strip().replace("\n", "\n    "))
    if r.returncode != 0:
        print("    STDERR:\n    " + (r.stderr or "").strip().replace("\n", "\n    "))
        raise RuntimeError(f"{label} failed (exit {r.returncode})")
    return r


# ---------------------------------------------------------------------------
# component (a): coherence figure  ->  (coherence_png, events_csv, regenerated?)
# ---------------------------------------------------------------------------
def ensure_coherence(run, root, scratch: Path, force: bool):
    run_dir, stem = run["run_dir"], run["stem"]
    stack = run["stack"]
    lm = fs.best_labelmap(run_dir, stem)
    if lm is None:
        raise RuntimeError(f"no labelmap for {run['behavior_base']} (need segments_located)")

    coh_png = run_dir / f"{stem}_coherence.png"
    coh_pdf = run_dir / f"{stem}_coherence.pdf"
    events = run_dir / f"{stem}_coherence_network_events.csv"

    up_to_date = (coh_png.exists() and events.exists()
                  and coh_png.stat().st_mtime >= lm.stat().st_mtime)
    if up_to_date and not force:
        print(f"  coherence: reuse existing (newer than labelmap {lm.name})")
        return coh_png, events, run_dir, False

    vx = fs.voxel_args(run)
    fr = run["frame_rate_hz"]

    def build(out_prefix: Path):
        cmd = [sys.executable, COHERENCE_TOOL, str(stack), str(lm)]
        if vx:
            cmd += ["--voxel", *vx]
        if fr:
            try:
                cmd += ["--frame-ms", f"{1000.0/float(fr):.3f}"]
            except ValueError:
                pass
        cmd += ["--out-prefix", str(out_prefix)]
        run_subprocess(cmd, root, "segment_event_coherence")

    if not coh_png.exists():
        # fresh: safe to write the canonical files into the run dir
        print(f"  coherence: building fresh (labelmap {lm.name})")
        build(run_dir / f"{stem}_coherence")
        return coh_png, events, run_dir, True

    # canonical figure EXISTS but is stale/forced -> never overwrite it
    print(f"  coherence: rebuild needed but '{coh_png.name}' exists -> rendering to temp, "
          f"will NOT overwrite")
    tmp_prefix = scratch / f"{stem}_coherence"
    build(tmp_prefix)
    tmp_png = scratch / f"{stem}_coherence.png"
    tmp_events = scratch / f"{stem}_coherence_network_events.csv"
    if sha256(tmp_png) == sha256(coh_png):
        print("  coherence: rebuild is byte-identical to existing -> reuse existing")
        return coh_png, events, run_dir, False
    print("  coherence: WARNING rebuild DIFFERS from existing; existing figure PRESERVED, "
          "composite uses the temp rebuild")
    return tmp_png, tmp_events, scratch, True


# ---------------------------------------------------------------------------
# component (b): behaviour panel  ->  behaviour_png
# ---------------------------------------------------------------------------
def ensure_behavior(run, root, scratch: Path, coh_source_dir: Path, events_csv: Path, force: bool):
    run_dir, stem = run["run_dir"], run["stem"]
    mouse, mat_date, run_id = parse_behavior_base(run["behavior_base"])
    folder_date = run["date"]
    fr = run["frame_rate_hz"]

    beh_png = run_dir / f"{stem}_coherence_behavior.png"
    up_to_date = beh_png.exists() and beh_png.stat().st_mtime >= events_csv.stat().st_mtime

    # If the coherence used is the (private) rebuild, behaviour MUST be rendered
    # against that rebuilt events CSV, so it cannot reuse the on-disk companion.
    coh_is_scratch = (coh_source_dir != run_dir)

    if up_to_date and not force and not coh_is_scratch:
        print(f"  behaviour: reuse existing (newer than events CSV)")
        return beh_png, "reused"

    # choose the directory coherence_behavior will read (needs events + a
    # per-segment trace CSV to count imaging frames)
    if coh_is_scratch:
        # bring the per-segment trace CSVs next to the rebuilt events CSV
        for pat in ("*_segment_traces.csv", "*_seg*.csv"):
            for f in run_dir.glob(pat):
                shutil.copy2(f, coh_source_dir / f.name)
        beh_run_dir = coh_source_dir
        out_stem = coh_source_dir / f"{stem}_coherence_behavior"
        status = "temp"
    elif not beh_png.exists():
        beh_run_dir = run_dir            # fresh: write canonical companion in run dir
        out_stem = run_dir / f"{stem}_coherence_behavior"
        status = "written"
    else:
        # canonical companion exists but a rebuild is forced -> render to temp,
        # never overwrite the on-disk figure
        beh_run_dir = run_dir
        out_stem = scratch / f"{stem}_coherence_behavior"
        status = "temp"
        print(f"  behaviour: rebuild forced but '{beh_png.name}' exists -> temp, will NOT overwrite")

    cmd = [sys.executable, BEHAVIOR_TOOL,
           "--run-dir", str(beh_run_dir),
           "--mouse", mouse, "--folder-date", folder_date, "--mat-date", mat_date,
           "--run-id", run_id, "--project-root", str(root),
           "--out", str(out_stem), "--formats", "png", "pdf"]
    if fr:
        cmd += ["--imaging-rate", str(fr)]
    print(f"  behaviour: rendering ({out_stem.name})")
    run_subprocess(cmd, root, "coherence_behavior")
    return out_stem.with_suffix(".png"), status


# ---------------------------------------------------------------------------
# component (c): compose
# ---------------------------------------------------------------------------
def title_lines(run) -> list[str]:
    mouse, mat_date, run_id = parse_behavior_base(run["behavior_base"])
    fr = run["frame_rate_hz"]
    flags = []
    fl = run.get("behavior_frame_loss_pct", "")
    if fl and fl not in ("0", "0.0"):
        flags.append(f"behaviour frame loss {fl}%")
    if run.get("suspect_nz"):
        flags.append("suspect nz (re-extract)")
    warn = run.get("behavior_warnings", "")
    l1 = f"{mouse}   {run['date']}   {run_id}   ({run['munit']})"
    l2 = f"imaging {fr} Hz    quality: {run['imaging_quality']}  [{run['priority']}, score {run['quality_score']}]"
    l3 = ("flags: " + "; ".join(flags)) if flags else "flags: none"
    lines = [l1, l2, l3]
    if warn:
        lines.append("warn: " + (warn if len(warn) < 120 else warn[:117] + "..."))
    return lines


def compose(coh_png: Path, beh_png: Path, out_png: Path, out_pdf: Path,
            titles: list[str], dpi: int) -> tuple[int, int, int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    coh = mpimg.imread(coh_png)
    beh = mpimg.imread(beh_png)
    ch, cw = coh.shape[:2]
    bh, bw = beh.shape[:2]

    fig_w = 12.0
    title_in = 0.28 * len(titles) + 0.35
    coh_in = fig_w * ch / cw
    beh_in = fig_w * bh / bw
    fig_h = title_in + coh_in + beh_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(3, 1, height_ratios=[title_in, coh_in, beh_in], hspace=0.015)

    axt = fig.add_subplot(gs[0]); axt.axis("off")
    axt.text(0.008, 0.92, titles[0], transform=axt.transAxes, ha="left", va="top",
             fontsize=13, fontweight="bold")
    axt.text(0.008, 0.40, "\n".join(titles[1:]), transform=axt.transAxes, ha="left",
             va="top", fontsize=9.5, family="monospace")

    axc = fig.add_subplot(gs[1]); axc.imshow(coh, aspect="auto"); axc.axis("off")
    axb = fig.add_subplot(gs[2]); axb.imshow(beh, aspect="auto"); axb.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    plt.close(fig)

    out_h = mpimg.imread(out_png).shape[0]
    return out_h, ch, bh


# ---------------------------------------------------------------------------
# per-run driver
# ---------------------------------------------------------------------------
def process_run(run, root, force: bool, dpi: int) -> dict:
    run_dir, stem = run["run_dir"], run["stem"]
    print(f"\n=== {run['behavior_base']}  (rank {run['rank']}, stage {run['stage']}) ===")
    print(f"  run dir : {run_dir}")

    written, reused = [], []
    scratch = Path(tempfile.mkdtemp(prefix="cohbeh_"))
    try:
        coh_png, events_csv, coh_dir, coh_regen = ensure_coherence(run, root, scratch, force)
        coh_written_here = coh_regen and coh_dir == run_dir
        (written if coh_written_here else reused).append(coh_png)

        beh_png, beh_status = ensure_behavior(run, root, scratch, coh_dir, events_csv, force)
        (written if beh_status == "written" else reused).append(beh_png)

        out_png = run_dir / f"{stem}_coherence_full.png"
        out_pdf = run_dir / f"{stem}_coherence_full.pdf"
        out_h, ch, bh = compose(coh_png, beh_png, out_png, out_pdf, title_lines(run), dpi)
        written += [out_png, out_pdf]

        print(f"  composed: {out_png.name}  (height {out_h}px vs coherence {ch}px, "
              f"behaviour {bh}px)  {'OK taller' if out_h > ch else 'WARNING not taller'}")
        return {"ok": True, "composite": out_png, "composite_h": out_h,
                "coherence_h": ch, "behavior_h": bh, "written": written, "reused": reused}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def find_run(runs, key: str, by_dir: bool):
    if by_dir:
        target = Path(key).resolve()
        for r in runs:
            if r["run_dir"] and Path(r["run_dir"]).resolve() == target:
                return r
        return None
    for r in runs:
        if r["behavior_base"] == key:
            return r
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="behavior_base, e.g. rbp4_141_phpeb_26-06-25_Run007")
    g.add_argument("--run-dir", help="explicit run directory (must contain a *_clean.tif)")
    g.add_argument("--all-ready", action="store_true",
                   help="process every run whose stage >= segments_located")
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs are current")
    ap.add_argument("--root", default=None, help="femtonics-data root (default: inferred)")
    ap.add_argument("--dpi", type=int, default=200, help="composite raster dpi (default 200)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else fs.project_root()
    runs = fs.build_status(root)

    if args.all_ready:
        targets = [r for r in runs
                   if r["stack"] is not None
                   and fs.STAGE_IDX[r["stage"]] >= STAGE_READY_IDX
                   and (args.force or r["stage"] != "complete")]
        if not targets:
            print("no runs at stage >= segments_located to process")
            return 0
        print(f"--all-ready: {len(targets)} run(s): " +
              ", ".join(f"{r['rank']}:{r['behavior_base']}" for r in targets))
    else:
        key = args.run or args.run_dir
        r = find_run(runs, key, by_dir=bool(args.run_dir))
        if r is None:
            ap.error(f"no run matches {key!r}")
        if r["stack"] is None:
            ap.error(f"{r['behavior_base']} has no local stack (stage {r['stage']})")
        if fs.STAGE_IDX[r["stage"]] < STAGE_READY_IDX:
            ap.error(f"{r['behavior_base']} is at '{r['stage']}' (< segments_located); "
                     f"locate segments first")
        targets = [r]

    rc = 0
    results = []
    for r in targets:
        try:
            results.append(process_run(r, root, args.force, args.dpi))
        except Exception as e:  # keep going in batch mode
            print(f"  ERROR: {e}")
            rc = 1
            if not args.all_ready:
                raise

    print("\n=== summary ===")
    for r, res in zip(targets, results):
        if res.get("ok"):
            print(f"  rank {r['rank']:>2} {r['behavior_base']}: {res['composite'].name} "
                  f"(+{len(res['written'])} written, {len(res['reused'])} reused/temp)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
