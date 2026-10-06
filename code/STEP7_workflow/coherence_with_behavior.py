#!/usr/bin/env python3
"""
coherence_with_behavior.py - ONE command per run to produce the target output:
the segment-event COHERENCE figure (correlation matrix + cell MIP + per-segment
dF/F with network events) stacked over the ALIGNED BEHAVIOR panel (pupil /
whisking / accelerometer on the same imaging-frame axis), under one title block.

It orchestrates the two existing, unmodified tools and then composes their PNGs:

  (a) code/extra/segment_event_coherence.py   -> the coherence figure + the
      <stem>_coherence_network_events.csv, run on the stack + the BEST available
      labelmap (priority: *_segments_final.tif > newest hand *_segments*.tif >
      autoseg reviewed > autoseg). Skipped if the coherence outputs are already
      newer than that labelmap (and --force was not given).
  (b) behavior-tracking-daria/batch/coherence_behavior.py -> the behavior panel
      on the coherence frame axis, identity fields derived from the master CSV.
  (c) compose -> <stem>_coherence_full.png + .pdf (coherence on top, behavior
      below, same width, one title block with mouse/date/run, frame rate and the
      quality flags e.g. behavior frame loss).

SAFETY (hard rule): nothing is ever deleted or silently overwritten. Up-to-date
components are reused as-is. When a component is stale (the regions changed) or
--force is given, the existing files are MOVED to <run_dir>/old/ with a timestamp
(<name>.YYYYmmdd-HHMMSS.<ext>) and the component is rebuilt in place, so the run
folder always holds one consistent set derived from the current regions. Hand-made
labelmaps are only ever read. Region names from <stem>_segments_final.json label the
coherence figure, ordered proximal -> distal from a region named soma*.

CLI
---
  PY=/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/.venv311/bin/python
  $PY code/STEP7_workflow/coherence_with_behavior.py --run rbp4_141_phpeb_26-06-25_Run007
  $PY code/STEP7_workflow/coherence_with_behavior.py --run-dir <dir> [--force]
  $PY code/STEP7_workflow/coherence_with_behavior.py --all-ready   # every run at >= segments_located
"""
from __future__ import annotations

import argparse
import json
import time
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import femto_status as fs  # noqa: E402
from common.regions import ignore_path, ignored_names  # noqa: E402

DISPLAY = {"mask": False, "edge_um": 2.0, "hide_other": True,          # cell-picture display mask (set from CLI)
           "scale": "auto"}                                             # trace amplitude scale (auto|common|zscore)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.display_mask import options_match, write_options   # noqa: E402
from common.plot_scale import default_scale_file, load_scale     # noqa: E402


def _scale_sidecar(png: Path) -> Path:
    return Path(str(png) + ".scale.json")


def scale_matches(png: Path, scale: str, scale_file: Path) -> bool:
    """The figure was drawn with this amplitude scale (missing sidecar = 'auto', the
    historical look) and, for common/zscore, after the cohort scale file last changed."""
    side = _scale_sidecar(png)
    try:
        rec = json.loads(side.read_text()) if side.exists() else {"scale": "auto"}
    except Exception:
        return False
    if rec.get("scale", "auto") != scale:
        return False
    return scale == "auto" or (scale_file.exists() and png.stat().st_mtime >= scale_file.stat().st_mtime)


def write_scale(png: Path, scale: str, scale_file: Path):
    rec = {"scale": scale}
    if scale != "auto":
        cfg = load_scale(scale_file) or {}
        rec.update({"scale_file": str(scale_file), "dff_per_lane": cfg.get("dff_per_lane"),
                    "z_per_lane": cfg.get("z_per_lane"), "inch_per_lane": cfg.get("inch_per_lane")})
    _scale_sidecar(png).write_text(json.dumps(rec))
_CODE_ROOT = Path(__file__).resolve().parents[2]   # real project root (for scripts)
COHERENCE_TOOL = str(_CODE_ROOT / "code/extra/segment_event_coherence.py")
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

    deps = [Path(lm), ignore_path(Path(lm))] + [Path(run_dir) / f"{stem}{suf}" for suf in
                         ("_autoseg_labelmap_reviewed.tif", "_exclude_labelmap.tif")]
    newest = max(d.stat().st_mtime for d in deps if d.exists())
    scale_file = default_scale_file(root)
    scale = DISPLAY["scale"]
    if scale != "auto" and load_scale(scale_file) is None:
        print(f"  coherence: WARNING --scale {scale} needs {scale_file} "
              f"(code/STEP8_stats/normalized_plots.py --write-scale); using the auto scale")
        scale = "auto"
    up_to_date = (coh_png.exists() and events.exists() and coh_png.stat().st_mtime >= newest
                  and options_match(coh_png, DISPLAY["mask"], DISPLAY["edge_um"], DISPLAY["hide_other"])
                  and scale_matches(coh_png, scale, scale_file))
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
        order, names = region_names_and_order(Path(lm))
        if order:
            ign = {n.lower() for n in ignored_names(Path(lm))}
            drop = [o for o, nm in zip(order, names) if nm.lower() in ign]
            if drop:
                kept = [(o, nm) for o, nm in zip(order, names) if nm.lower() not in ign]
                order, names = [o for o, _ in kept], [nm for _, nm in kept]
                cmd += ["--exclude", *[str(o) for o in drop]]
                print(f"  coherence: ignoring {', '.join(n for n in ignored_names(Path(lm)))} (<stem>_ignore.json)")
            cmd += ["--order", *[str(o) for o in order], "--names", *names]
            print(f"  coherence: region names (proximal->distal): {', '.join(names)}")
        cmd += ["--mask"] if DISPLAY["mask"] else ["--no-mask"]
        cmd += ["--hide-other"] if DISPLAY["hide_other"] else ["--show-other"]
        cmd += ["--edge-um", f"{DISPLAY['edge_um']:g}"]
        if scale != "auto":
            cmd += ["--scale", scale, "--scale-file", str(scale_file)]
        cmd += ["--out-prefix", str(out_prefix)]
        run_subprocess(cmd, root, "segment_event_coherence")

    if not coh_png.exists():
        # fresh: safe to write the canonical files into the run dir
        print(f"  coherence: building fresh (labelmap {lm.name})")
        build(run_dir / f"{stem}_coherence")
        write_options(coh_png, DISPLAY["mask"], DISPLAY["edge_um"], DISPLAY["hide_other"])
        write_scale(coh_png, scale, scale_file)
        return coh_png, events, run_dir, True

    # canonical figure exists but is stale (regions changed) or forced: archive the old
    # outputs into old/ (timestamped, nothing is deleted) and rebuild in place, so the
    # run folder always holds one consistent set derived from the current regions.
    print(f"  coherence: rebuilding (inputs, display options changed, or --force)")
    archive([coh_png, coh_pdf, events, Path(str(coh_png) + ".display.json"), _scale_sidecar(coh_png)], run_dir)
    build(run_dir / f"{stem}_coherence")
    write_options(coh_png, DISPLAY["mask"], DISPLAY["edge_um"], DISPLAY["hide_other"])
    write_scale(coh_png, scale, scale_file)
    return coh_png, events, run_dir, True


# ---------------------------------------------------------------------------
# helpers: region names, archiving superseded outputs
# ---------------------------------------------------------------------------
def region_names_and_order(lm_path: Path):
    """(order, names) for the coherence tool, or (None, None).

    Names come from <stem>_segments_final.json: 'segment_names' {final_id: name} when
    present, else reconstructed from 'wrap_clicks' (sorted surviving labels -> 1..N, the
    same renumbering the region tool applies). Order: proximal -> distal. If a region is
    named soma*, regions are ordered by distance along the tube from it (so the soma is
    always first, whichever end of the scan it sits at); otherwise by mean X."""
    import numpy as np, tifffile
    js = lm_path.with_suffix(".json")
    if not lm_path.name.endswith("_segments_final.tif") or not js.exists():
        return None, None
    try:
        j = json.loads(js.read_text())
    except Exception:
        return None, None
    lm = tifffile.imread(str(lm_path))
    final = sorted(int(v) for v in np.unique(lm) if v > 0)
    names = {}
    if isinstance(j.get("segment_names"), dict):
        names = {int(k): str(v) for k, v in j["segment_names"].items()}
    else:
        clicks = j.get("wrap_clicks", [])
        last = {}
        for w in clicks:
            last[int(w["label"])] = w.get("name") or f"seg{w['label']}"
        pre = sorted(last)
        if len(pre) == len(final):
            names = {f: last[p_] for f, p_ in zip(final, pre)}
    if set(names) != set(final):
        return None, None
    xm = {l: float(np.argwhere(lm == l)[:, 2].mean()) for l in final}
    soma = [l for l in final if names[l].lower().startswith("soma")]
    if soma:
        s0 = xm[soma[0]]
        order = sorted(final, key=lambda l: (abs(xm[l] - s0), l))
    else:
        order = sorted(final, key=lambda l: xm[l])
    return order, [names[l] for l in order]


def archive(paths, run_dir: Path):
    """Move existing generated outputs into run_dir/old/ with a timestamp (never delete)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    old = run_dir / "old"
    moved = []
    for pth in paths:
        pth = Path(pth)
        if pth.exists():
            old.mkdir(exist_ok=True)
            dest = old / f"{pth.stem}.{stamp}{pth.suffix}"
            shutil.move(str(pth), str(dest)); moved.append(dest.name)
    if moved:
        print(f"  archived superseded outputs -> old/: {', '.join(moved)}")


# ---------------------------------------------------------------------------
# component (b): behavior panel  ->  behavior_png
# ---------------------------------------------------------------------------
def ensure_behavior(run, root, scratch: Path, coh_source_dir: Path, events_csv: Path, force: bool):
    run_dir, stem = run["run_dir"], run["stem"]
    mouse, mat_date, run_id = parse_behavior_base(run["behavior_base"])
    folder_date = run["date"]
    fr = run["frame_rate_hz"]

    beh_png = run_dir / f"{stem}_coherence_behavior.png"
    up_to_date = beh_png.exists() and beh_png.stat().st_mtime >= events_csv.stat().st_mtime

    # If the coherence used is the (private) rebuild, behavior MUST be rendered
    # against that rebuilt events CSV, so it cannot reuse the on-disk companion.
    coh_is_scratch = (coh_source_dir != run_dir)

    if up_to_date and not force and not coh_is_scratch:
        print(f"  behavior: reuse existing (newer than events CSV)")
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
        # stale or forced: archive the existing companion into old/, rebuild in place
        archive([beh_png, beh_png.with_suffix(".pdf")], run_dir)
        beh_run_dir = run_dir
        out_stem = run_dir / f"{stem}_coherence_behavior"
        status = "written"

    cmd = [sys.executable, BEHAVIOR_TOOL,
           "--run-dir", str(beh_run_dir),
           "--mouse", mouse, "--folder-date", folder_date, "--mat-date", mat_date,
           "--run-id", run_id, "--project-root", str(root),
           "--out", str(out_stem), "--formats", "png", "pdf"]
    if fr:
        cmd += ["--imaging-rate", str(fr)]
    print(f"  behavior: rendering ({out_stem.name})")
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
        flags.append(f"behavior frame loss {fl}%")
    if run.get("suspect_nz"):
        flags.append("suspect nz (re-extract)")
    warn = run.get("behavior_warnings", "")
    l1 = f"{mouse}   {run['date']}   {run_id}   ({run['munit']})"
    l2 = f"imaging {fr} Hz"
    l3 = ("flags: " + "; ".join(flags)) if flags else None      # shown only when something is flagged
    lines = [l1, l2] + ([l3] if l3 else [])
    if warn:
        lines.append("warn: " + (warn if len(warn) < 120 else warn[:117] + "..."))
    return lines


def _title_lines_imaging_only(run) -> list[str]:
    """Title lines for imaging-only runs (no behavior_base to parse)."""
    mouse = run.get("mouse", "?")
    fr = run.get("frame_rate_hz", "?")
    munit = run.get("munit", "?")
    date = run.get("date", "?")
    l1 = f"{mouse}   {date}   ({munit})   [no behavior]"
    l2 = f"imaging {fr} Hz"
    bstatus = run.get("_behavior_status", "missing")
    l3 = f"behavior status: {bstatus}"
    return [l1, l2, l3]


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
    plt.close(fig)

    # PDF: stack the component PDFs themselves (vector), under a vector title block,
    # instead of embedding the PNG renders. Falls back to a raster PDF only if a
    # component PDF is missing.
    coh_pdf, beh_pdf = coh_png.with_suffix(".pdf"), beh_png.with_suffix(".pdf")
    if coh_pdf.exists() and beh_pdf.exists():
        vector_stack_pdf([coh_pdf, beh_pdf], titles, out_pdf)
    else:
        print("  compose: component PDF missing -> raster PDF fallback")
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(mpimg.imread(out_png)); ax.axis("off")
        fig.savefig(out_pdf); plt.close(fig)

    out_h = mpimg.imread(out_png).shape[0]
    return out_h, ch, bh


def vector_stack_pdf(parts, titles, out_pdf: Path, width_pt: float = 864.0):
    """One-page PDF: a vector title block, then each part's first page scaled to a common
    width and stacked top-to-bottom. Text, lines and traces stay vector; only genuine
    images inside the parts (e.g. the cell MIP) remain images."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pypdf import PdfReader, PdfWriter, Transformation

    title_h = 72.0 * (0.28 * len(titles) + 0.35)
    fig = plt.figure(figsize=(width_pt / 72.0, title_h / 72.0))
    fig.text(0.008, 0.92, titles[0], ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.008, 0.40, "\n".join(titles[1:]), ha="left", va="top", fontsize=9.5, family="monospace")
    buf = io.BytesIO(); fig.savefig(buf, format="pdf"); plt.close(fig); buf.seek(0)

    pages = [PdfReader(buf).pages[0]] + [PdfReader(str(pp)).pages[0] for pp in parts]
    scales = [width_pt / float(pg.mediabox.width) for pg in pages]
    heights = [float(pg.mediabox.height) * sc for pg, sc in zip(pages, scales)]
    total = sum(heights)
    w = PdfWriter()
    page = w.add_blank_page(width=width_pt, height=total)
    y = total
    for pg, sc, h in zip(pages, scales, heights):
        y -= h
        x0, y0 = float(pg.mediabox.left), float(pg.mediabox.bottom)
        page.merge_transformed_page(pg, Transformation().translate(-x0, -y0).scale(sc, sc).translate(0, y))
    with open(out_pdf, "wb") as fh:
        w.write(fh)


# ---------------------------------------------------------------------------
# per-run driver
# ---------------------------------------------------------------------------
def process_run(run, root, force: bool, dpi: int) -> dict:
    run_dir, stem = run["run_dir"], run["stem"]
    print(f"\n=== {run.get('behavior_base') or run.get('mouse', '?')}  (rank {run['rank']}, stage {run['stage']}) ===")
    print(f"  run dir : {run_dir}")

    written, reused = [], []
    scratch = Path(tempfile.mkdtemp(prefix="cohbeh_"))

    # Check if this run has behavior data
    has_behavior_base = bool(run.get("behavior_base", "").strip())
    is_imaging_only = run.get("_imaging_only", False)
    behavior_status = run.get("_behavior_status", "paired")
    skip_behavior = (not has_behavior_base) or (is_imaging_only and behavior_status != "paired")
    if not skip_behavior:
        # ranked run, but is its behavior CSV actually on disk? (e.g. rbp4_139: not yet)
        sess = Path(run_dir)
        while sess.parent != sess and not re.fullmatch(r"\d\d-\d\d-\d{4}", sess.name):
            sess = sess.parent
        run_id = run["behavior_base"].rsplit("_", 1)[-1]
        if not any((sess / "behavior").rglob(f"*{run_id}*.csv")) if (sess / "behavior").exists() else True:
            skip_behavior, behavior_status = True, "behavior CSV not on disk"

    try:
        coh_png, events_csv, coh_dir, coh_regen = ensure_coherence(run, root, scratch, force)
        coh_written_here = coh_regen and coh_dir == run_dir
        (written if coh_written_here else reused).append(coh_png)

        if skip_behavior:
            print(f"  behavior: SKIPPED (imaging-only run, behavior_status={behavior_status})")
            # Produce the composite without the behavior panel:
            # coherence figure alone is the full output
            out_png = run_dir / f"{stem}_coherence_full.png"
            out_pdf = run_dir / f"{stem}_coherence_full.pdf"
            archive([out_png, out_pdf], run_dir)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            coh_img = mpimg.imread(coh_png)
            ch, cw = coh_img.shape[:2]
            titles = _title_lines_imaging_only(run)
            fig_w = 12.0
            title_in = 0.28 * len(titles) + 0.35
            coh_in = fig_w * ch / cw
            fig_h = title_in + coh_in

            fig = plt.figure(figsize=(fig_w, fig_h))
            gs = fig.add_gridspec(2, 1, height_ratios=[title_in, coh_in], hspace=0.015)
            axt = fig.add_subplot(gs[0]); axt.axis("off")
            axt.text(0.008, 0.92, titles[0], transform=axt.transAxes, ha="left", va="top",
                     fontsize=13, fontweight="bold")
            axt.text(0.008, 0.40, "\n".join(titles[1:]), transform=axt.transAxes, ha="left",
                     va="top", fontsize=9.5, family="monospace")
            axc = fig.add_subplot(gs[1]); axc.imshow(coh_img, aspect="auto"); axc.axis("off")
            fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
            fig.savefig(out_png, dpi=dpi); plt.close(fig)

            # Simple raster PDF
            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(mpimg.imread(out_png)); ax.axis("off")
            fig.savefig(out_pdf); plt.close(fig)
            written += [out_png, out_pdf]

            print(f"  composed (no behavior): {out_png.name}  ({ch}px coherence)")
            return {"ok": True, "composite": out_png, "composite_h": ch,
                    "coherence_h": ch, "behavior_h": 0,
                    "written": written, "reused": reused, "no_behavior": True}

        beh_png, beh_status = ensure_behavior(run, root, scratch, coh_dir, events_csv, force)
        (written if beh_status == "written" else reused).append(beh_png)

        out_png = run_dir / f"{stem}_coherence_full.png"
        out_pdf = run_dir / f"{stem}_coherence_full.pdf"
        archive([out_png, out_pdf], run_dir)
        out_h, ch, bh = compose(coh_png, beh_png, out_png, out_pdf, title_lines(run), dpi)
        written += [out_png, out_pdf]

        print(f"  composed: {out_png.name}  (height {out_h}px vs coherence {ch}px, "
              f"behavior {bh}px)  {'OK taller' if out_h > ch else 'WARNING not taller'}")
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
        if r.get("behavior_base") == key:
            return r
        # Also match by mouse+munit for imaging-only runs without a behavior_base
        if r.get("_imaging_only") and not r.get("behavior_base") and key == f"{r['mouse']}_{r['munit']}":
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
    ap.add_argument("--mask", dest="mask", action="store_true", default=False,
                    help="black background outside the cell and other cells blacked out "
                         "(display only; default: the original, unmasked recording)")
    ap.add_argument("--no-mask", dest="mask", action="store_false", help="original look (default)")
    ap.add_argument("--hide-other", dest="hide_other", action="store_true", default=True,
                    help="fill cells marked 'other cell' with nearby background flicker (default)")
    ap.add_argument("--show-other", dest="hide_other", action="store_false",
                    help="show other cells as recorded")
    ap.add_argument("--edge-um", type=float, default=2.0, help="soft edge of the display mask (um)")
    ap.add_argument("--scale", choices=["auto", "common", "zscore"], default="auto",
                    help="trace amplitude scale: auto (default, each run's own range), common (one "
                         "cohort dF/F per inch from stats/plot_scale.json, amplitudes compare across "
                         "runs) or zscore (traces / noise SD). Display only.")
    args = ap.parse_args(argv)
    DISPLAY["mask"], DISPLAY["edge_um"], DISPLAY["hide_other"] = args.mask, args.edge_um, args.hide_other
    DISPLAY["scale"] = args.scale

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
