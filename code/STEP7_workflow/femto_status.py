#!/usr/bin/env python3
"""
femto_status.py - STATUS TRACKER + DRIVER for the Femtonics 2P dendrite workflow.

WHAT IT IS
----------
One command that answers "where is every run, and what do I do next?" for the
44 ranked runs in ranked_runs.csv. Every stage is detected BY DISK PRESENCE of
the artifact a stage produces, so the table can never drift from reality: if the
file is there the stage is done, if it is not it is not.

THE WORKFLOW LADDER (per run)
-----------------------------
  not_local          no local 4D stack (<stem>_clean.tif absent)   -> fetch it
  stack              <stem>_clean.tif present                        -> make_reference_volume.py
  reference          <stem>_clean_ref3d.tif                          -> auto_segment.py
  auto_segmented     <stem>_clean_autoseg_labelmap.tif               -> trace_mask_napari.py (GUI; review_autoseg_napari.py = cell-toggle alternative)
  mask_reviewed      <stem>_clean_autoseg_labelmap_reviewed.tif      -> wrap_segments_napari.py (GUI)
  segments_located   <stem>_clean_segments_final.tif  OR hand-made
                     <stem>_clean_segments*.tif                      -> coherence_with_behavior.py
  coherence_built    <stem>_clean_coherence.png + _network_events.csv-> coherence_with_behavior.py
  behavior_added     <stem>_clean_coherence_behavior.png             -> coherence_with_behavior.py
  complete           <stem>_clean_coherence_full.png (the composite) -> done

The reported stage is the FURTHEST one whose artifact is present (a later
artifact implies the earlier work happened - the old hand-tracing route skips
the autoseg 'reviewed' box, so a strict "stop at first gap" would mis-rank those
runs). The full per-artifact checklist is always shown / written so any gap is
explicit.

NOTHING IS HARDCODED: the run list, the run directories, the stems and the
stages are all derived from ranked_runs.csv + behavior_imaging_master.csv + what
is on disk. No run is ever written to, moved, or deleted.

CLI
---
  PY=/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/.venv311/bin/python
  $PY code/STEP7_workflow/femto_status.py                    # full 44-row table + write processing_status.csv
  $PY code/STEP7_workflow/femto_status.py --filter stage=segments_located
  $PY code/STEP7_workflow/femto_status.py --mouse rbp4_141_phpeb
  $PY code/STEP7_workflow/femto_status.py --min-quality 2
  $PY code/STEP7_workflow/femto_status.py --next            # the single next run to act on + exact command
  $PY code/STEP7_workflow/femto_status.py --next --run-it   # run that command (non-GUI stages only)
"""
from __future__ import annotations

import argparse
import os
import re
import csv
import subprocess
import sys
from pathlib import Path

# The interpreter the printed commands should be run with (the project venv).
VENV_PY = "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/.venv311/bin/python"

# Stage ladder, low -> high. Index in this list is the stage's rank on the ladder.
STAGES = [
    "not_local",
    "stack",
    "reference",
    "auto_segmented",
    "mask_reviewed",
    "segments_located",
    "coherence_built",
    "behavior_added",
    "complete",
]
STAGE_IDX = {s: i for i, s in enumerate(STAGES)}


# ---------------------------------------------------------------------------
# repo layout
# ---------------------------------------------------------------------------
def project_root() -> Path:
    """femtonics-data/ (this file lives in femtonics-data/code/STEP7_workflow/).

    Honors the FEMTO_ROOT environment variable when set (absolute path to a mirror
    tree that has its own root CSVs).  Without it, returns the real project root
    derived from this file's location.
    """
    env = os.environ.get("FEMTO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CSV loading + join
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_runs(root: Path) -> list[dict]:
    """Join ranked_runs.csv with behavior_imaging_master.csv on integer rank.

    Returns one dict per ranked run (1..44), rank-sorted, carrying the identity,
    quality and file-location fields the rest of the tool needs.
    """
    ranked = _read_csv(root / "ranked_runs.csv")
    master_rows = _read_csv(root / "behavior_imaging_master.csv")

    # master has ~48 extra unranked rows (blank rank); index only the ranked ones
    master_by_rank: dict[int, dict] = {}
    for m in master_rows:
        r = (m.get("rank") or "").strip()
        if r.isdigit():
            master_by_rank[int(r)] = m

    runs: list[dict] = []
    for row in ranked:
        r = (row.get("rank") or "").strip()
        if not r.isdigit():
            continue
        rank = int(r)
        m = master_by_rank.get(rank, {})
        runs.append(
            {
                "rank": rank,
                "priority": row.get("priority", "") or m.get("priority", ""),
                "mouse": row.get("mouse", "") or m.get("mouse", ""),
                "date": row.get("date", "") or m.get("date", ""),
                "munit": row.get("munit", "") or m.get("munit", ""),
                "behavior_run": row.get("behavior_run", "") or m.get("behavior_run_number", ""),
                "behavior_base": (m.get("behavior_base") or "").strip(),
                "frame_rate_hz": (m.get("frame_rate_hz") or row.get("volume_rate_hz") or "").strip(),
                "voxel_zyx_um": (row.get("voxel_zyx_um") or "").strip(),
                "imaging_quality": (row.get("imaging_quality") or m.get("imaging_quality") or "").strip(),
                "quality_score": (m.get("quality_score") or "").strip(),
                "behavior_frame_loss_pct": (row.get("behavior_frame_loss_pct")
                                            or m.get("behavior_frame_loss_pct") or "").strip(),
                "behavior_warnings": (m.get("behavior_warnings") or "").strip(),
                "quality_notes": (m.get("quality_notes") or "").strip(),
                # location fields (from master; fall back to ranked's extracted_4d_tif)
                "session_dir": (m.get("session_dir") or "").strip(),
                "extracted_tif": (m.get("extracted_tif") or row.get("extracted_4d_tif") or "").strip(),
                "extracted_tif_other_candidates": (m.get("extracted_tif_other_candidates") or "").strip(),
                # broken/suspect extractions record their (present) clean-stack path here
                "extracted_tif_suspect_nz": (m.get("extracted_tif_suspect_nz") or "").strip(),
            }
        )
    runs.sort(key=lambda d: d["rank"])

    # Append imaging-only runs from <root>/imaging_only_runs.csv (if present).
    # These are runs without paired behavior data; they get ranks continuing after
    # the max ranked run, priority P5, and behavior fields left empty.
    io_csv = root / "imaging_only_runs.csv"
    if io_csv.exists():
        io_rows = _read_csv(io_csv)
        max_rank = max((r["rank"] for r in runs), default=0)
        for i, row in enumerate(io_rows, start=1):
            bstatus = (row.get("behavior_status") or "missing").strip()
            runs.append(
                {
                    "rank": max_rank + i,
                    "priority": "P5",
                    "mouse": (row.get("mouse") or "").strip(),
                    "date": (row.get("date") or "").strip(),
                    "munit": (row.get("munit") or "").strip(),
                    "behavior_run": (row.get("behavior_run_number") or row.get("behavior_run") or "").strip(),
                    "behavior_base": (row.get("behavior_base") or "").strip(),
                    "frame_rate_hz": (row.get("frame_rate_hz") or "").strip(),
                    "voxel_zyx_um": (row.get("voxel_zyx_um") or "").strip(),
                    "imaging_quality": (row.get("imaging_quality") or "").strip(),
                    "quality_score": (row.get("quality_score") or "").strip(),
                    "behavior_frame_loss_pct": "",
                    "behavior_warnings": "",
                    "quality_notes": (row.get("quality_notes") or "").strip(),
                    "session_dir": (row.get("session_dir") or "").strip(),
                    "extracted_tif": (row.get("extracted_tif") or "").strip(),
                    "extracted_tif_other_candidates": "",
                    "extracted_tif_suspect_nz": "",
                    # Marker fields for imaging-only runs
                    "_imaging_only": True,
                    "_behavior_status": bstatus,
                    "_run_dir_hint": (row.get("run_dir") or "").strip(),
                    "_stem_hint": (row.get("stem") or "").strip(),
                }
            )
    return runs


# ---------------------------------------------------------------------------
# run-directory + stem resolution (disk-driven, no hardcoded run lists)
# ---------------------------------------------------------------------------
def _session_root(root: Path, run: dict) -> Path:
    sd = run["session_dir"]
    if sd:
        return root / sd
    # fall back: <root>/<mouse>/<date>
    return root / run["mouse"] / run["date"]


def _clean_in(d: Path) -> Path | None:
    """The canonical cleaned 4D stack (*_clean.tif) directly inside directory d."""
    if not d.is_dir():
        return None
    hits = sorted(p for p in d.glob("*_clean.tif") if p.is_file())
    return hits[0] if hits else None


def _walk_up_for_clean(file_rel: str, sroot: Path) -> tuple[Path, Path] | None:
    """From a CSV file path, walk up its ancestors (bounded to the session dir)
    looking for a directory that holds a *_clean.tif. Handles nesting such as
    run6/raw/run63d.tif where the clean stack lives one level up in run6/."""
    p = (sroot / file_rel)
    d = p.parent
    while True:
        clean = _clean_in(d)
        if clean is not None:
            return clean, d
        if d == sroot:
            break
        if sroot not in d.parents:
            break
        d = d.parent
    return None


def _candidate_clean_dirs(other: str, sroot: Path) -> list[tuple[Path, Path]]:
    """(clean_file, dir) for every *_clean.tif listed in other_candidates that
    actually exists on disk."""
    out = []
    for c in (x.strip() for x in other.split(";")):
        if not c or not c.endswith("_clean.tif"):
            continue
        f = sroot / c
        if f.is_file():
            out.append((f, f.parent))
    return out


def resolve_run_dirs(root: Path, runs: list[dict]) -> None:
    """Populate run['run_dir'], run['stem'], run['stack'] in place.

    Two-phase, disk-driven, contamination-safe assignment of the (exactly 13)
    local *_clean.tif stacks to the runs that own them:

      phase 1  direct ownership - a run whose OWN extracted_tif path leads (via
               ancestor walk) to a directory holding a *_clean.tif claims it.
      phase 2  indirect ownership - runs whose extracted_tif does not lead to a
               clean stack (their extraction was named by MUnit, or the record
               is a broken/empty extraction) claim an as-yet-unclaimed clean
               dir listed in their other_candidates, best rank first. Ambiguous
               siblings that merely *list* the same dir are left not-local.
    """
    claimed: dict[Path, int] = {}

    # phase 1
    for run in sorted(runs, key=lambda d: d["rank"]):
        run["run_dir"] = None
        run["stem"] = None
        run["stack"] = None
        ex = run["extracted_tif"]
        if not ex:
            continue
        sroot = _session_root(root, run)
        found = _walk_up_for_clean(ex, sroot)
        if found:
            clean, d = found
            if d not in claimed:
                claimed[d] = run["rank"]
                run["run_dir"], run["stack"], run["stem"] = d, clean, clean.stem

    # phase 2
    for run in sorted(runs, key=lambda d: d["rank"]):
        if run["stack"] is not None:
            continue
        sroot = _session_root(root, run)
        # other_candidates first, then suspect-nz (broken-but-present extractions)
        cands = (_candidate_clean_dirs(run["extracted_tif_other_candidates"], sroot)
                 + _candidate_clean_dirs(run["extracted_tif_suspect_nz"], sroot))
        for clean, d in cands:
            if d not in claimed:
                claimed[d] = run["rank"]
                run["run_dir"], run["stack"], run["stem"] = d, clean, clean.stem
                run["suspect_nz"] = bool(run["extracted_tif_suspect_nz"])
                break

    # phase 2b: CONVENTION PATH - a freshly extracted+cleaned run has no
    # (useful) extracted_tif record yet, but the pipeline always lands its
    # stack at <session>/preprocessed/run<N>/run<N>_clean.tif (or the same
    # without the preprocessed level, as in rbp4_phpebach). Claim that dir if
    # it holds a clean stack and nobody with a record owns it. Without this,
    # every future extraction stays invisible until someone hand-edits the
    # master CSV.
    for run in sorted(runs, key=lambda d: d["rank"]):
        if run["stack"] is not None:
            continue
        n = str(run.get("behavior_run") or "")
        # accept both numeric ('3', '3.0') and RunNNN forms ('Run003')
        m_ = re.search(r"(\d+)", n)
        if not m_:
            continue
        rn_i = int(m_.group(1))
        # accept both the zero-padded convention (run05) and the older
        # unpadded folders (run5); padded first since that is canonical now
        rns = [f"run{rn_i:02d}", f"run{rn_i}"]
        sroot = _session_root(root, run)
        cands = [sroot / parent / rn for rn in rns
                 for parent in ("preprocessed", ".", "traces")]
        for d in cands:
            clean = _clean_in(d)
            if clean and d not in claimed:
                claimed[d] = run["rank"]
                run["run_dir"], run["stack"], run["stem"] = d, clean, clean.stem
                break

    # phase 3: still not local -> record a best-effort run_dir for display only
    for run in runs:
        if run["run_dir"] is None:
            # imaging-only runs: check the explicit run_dir/stem hint from CSV
            hint = run.get("_run_dir_hint", "")
            stem_hint = run.get("_stem_hint", "")
            if hint:
                d = root / hint
                clean = _clean_in(d)
                if clean and d not in claimed:
                    claimed[d] = run["rank"]
                    run["run_dir"], run["stack"], run["stem"] = d, clean, clean.stem
                    continue
                elif clean:
                    # dir exists + clean.tif but already claimed — try stem hint directly
                    if stem_hint and (d / f"{stem_hint}.tif").exists():
                        run["run_dir"], run["stack"], run["stem"] = d, d / f"{stem_hint}.tif", stem_hint
                        continue
            ex = run["extracted_tif"]
            sroot = _session_root(root, run)
            run["run_dir"] = (sroot / ex).parent if ex else sroot


# ---------------------------------------------------------------------------
# stage detection (pure disk presence)
# ---------------------------------------------------------------------------
def _exists_any(paths) -> bool:
    return any(p.exists() for p in paths)


def best_labelmap(run_dir: Path, stem: str) -> Path | None:
    """Priority: *_segments_final.tif > newest hand *_segments*.tif >
    autoseg reviewed > autoseg. (stem ends in _clean, so *_segments* == the
    hand-made *_clean_segments* convention.)"""
    final = run_dir / f"{stem}_segments_final.tif"
    if final.exists():
        return final
    seg = [p for p in run_dir.glob(f"{stem}_segments*.tif") if p.is_file()]
    if seg:
        return max(seg, key=lambda p: p.stat().st_mtime)
    rev = run_dir / f"{stem}_autoseg_labelmap_reviewed.tif"
    if rev.exists():
        return rev
    auto = run_dir / f"{stem}_autoseg_labelmap.tif"
    if auto.exists():
        return auto
    return None


def detect_artifacts(run: dict) -> dict:
    """Boolean per-artifact checklist for a run (all keys always present)."""
    d, stem = run["run_dir"], run["stem"]
    a = {k: False for k in
         ("stack", "reference", "auto_segmented", "mask_reviewed",
          "segments_located", "coherence_built", "behavior_added", "complete")}
    if not stem or run["stack"] is None:
        return a
    a["stack"] = True
    a["reference"] = (d / f"{stem}_ref3d.tif").exists()
    a["auto_segmented"] = (d / f"{stem}_autoseg_labelmap.tif").exists()
    a["mask_reviewed"] = (d / f"{stem}_autoseg_labelmap_reviewed.tif").exists()
    a["segments_located"] = (
        (d / f"{stem}_segments_final.tif").exists()
        or any(d.glob(f"{stem}_segments*.tif"))
    )
    a["coherence_built"] = (
        _exists_any([d / f"{stem}_coherence.png", d / f"{stem}_coherence.pdf"])
        and (d / f"{stem}_coherence_network_events.csv").exists()
    )
    a["behavior_added"] = _exists_any(
        [d / f"{stem}_coherence_behavior.png", d / f"{stem}_coherence_behavior.pdf"])
    a["complete"] = _exists_any(
        [d / f"{stem}_coherence_full.png", d / f"{stem}_coherence_full.pdf"])
    return a


def stage_from_artifacts(a: dict) -> str:
    """Furthest stage whose artifact is present (later implies earlier)."""
    if not a["stack"]:
        return "not_local"
    idx = STAGE_IDX["stack"]
    for name in ("reference", "auto_segmented", "mask_reviewed",
                 "segments_located", "coherence_built", "behavior_added", "complete"):
        if a[name]:
            idx = max(idx, STAGE_IDX[name])
    return STAGES[idx]


def checklist_str(a: dict) -> str:
    flag = lambda b: "+" if b else "-"
    return (f"stk{flag(a['stack'])} ref{flag(a['reference'])} "
            f"auto{flag(a['auto_segmented'])} rev{flag(a['mask_reviewed'])} "
            f"seg{flag(a['segments_located'])} coh{flag(a['coherence_built'])} "
            f"beh{flag(a['behavior_added'])} full{flag(a['complete'])}")


# ---------------------------------------------------------------------------
# next action / exact command per stage
# ---------------------------------------------------------------------------
def voxel_args(run: dict) -> list[str]:
    """'0.8/0.9/0.9' -> ['0.8','0.9','0.9'] (Z Y X); [] if unknown."""
    v = run["voxel_zyx_um"]
    parts = [p for p in v.split("/") if p.strip()]
    return parts if len(parts) == 3 else []


def next_action(run: dict, stage: str) -> dict:
    """Return {'label', 'cmd' (list|None), 'gui' (bool), 'runnable' (bool)}."""
    stack = run["stack"]
    stack_s = str(stack) if stack else ""
    vx = voxel_args(run)
    base = run["behavior_base"]

    # Imaging-only behavior suffix: automatic imaging steps stay runnable
    is_io = run.get("_imaging_only", False)
    bstatus = run.get("_behavior_status", "paired")
    need_beh = is_io and bstatus != "paired"
    beh_suffix = " — needs behavior (camera tracking)" if need_beh else ""

    if stage == "not_local":
        src = run["extracted_tif"] or "(no recorded path)"
        # removed on purpose to save space? RECOVERY_4D.csv knows how to rebuild it
        man = project_root() / "RECOVERY_4D.csv"
        if man.exists() and run.get("session_dir") and run.get("extracted_tif"):
            import csv as _csv
            want = f"{run['session_dir']}/{run['extracted_tif']}"
            for r in _csv.DictReader(open(man, newline="")):
                if r.get("removed") and r["path"] == want:
                    return {"label": f"4D stack removed to save space - restore with: femto restore {want}{beh_suffix}",
                            "cmd": None, "gui": False, "runnable": False}
        return {"label": f"fetch 4D stack ({src}){beh_suffix}", "cmd": None, "gui": False, "runnable": False}
    if stage == "stack":
        return {"label": f"build reference volume{beh_suffix}", "gui": False, "runnable": True,
                "cmd": [VENV_PY, "code/STEP3_auto/make_reference_volume.py", stack_s]}
    if stage == "reference":
        return {"label": f"auto-segment cells{beh_suffix}", "gui": False, "runnable": True,
                "cmd": [VENV_PY, "code/STEP3_auto/auto_segment.py", stack_s]}
    if stage == "auto_segmented":
        cmd = [VENV_PY, "code/STEP3_auto/trace_mask_napari.py", stack_s]
        if vx:
            cmd += ["--voxel", *vx]
        return {"label": f"trace + grow mask (napari GUI){beh_suffix}", "cmd": cmd, "gui": True, "runnable": False}
    if stage == "mask_reviewed":
        cmd = [VENV_PY, "code/STEP7_workflow/wrap_segments_napari.py", stack_s]
        if vx:
            cmd += ["--voxel", *vx]
        return {"label": f"one-click anatomy wrap (napari GUI){beh_suffix}", "cmd": cmd, "gui": True, "runnable": False}
    if stage in ("segments_located", "coherence_built", "behavior_added"):
        label = {"segments_located": "build coherence + behavior composite",
                 "coherence_built": "add behavior + composite",
                 "behavior_added": "build the composite"}[stage]
        return {"label": f"{label}{beh_suffix}", "gui": False, "runnable": True,
                "cmd": [VENV_PY, "code/STEP7_workflow/coherence_with_behavior.py", "--run", base] if base else None}
    return {"label": f"complete - nothing to do{beh_suffix}", "cmd": None, "gui": False, "runnable": False}


def cmd_display(cmd: list[str] | None) -> str:
    if not cmd:
        return ""
    shown = ["$PY" if c == VENV_PY else c for c in cmd]
    return "cd <femtonics-data> && PY=" + VENV_PY + "\n  " + " ".join(shown)


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def build_status(root: Path) -> list[dict]:
    runs = load_runs(root)
    resolve_run_dirs(root, runs)
    for run in runs:
        a = detect_artifacts(run)
        run["artifacts"] = a
        run["stage"] = stage_from_artifacts(a)
        run["checklist"] = checklist_str(a)
        run["next"] = next_action(run, run["stage"])
    # your per-run decision (run_marks.csv in the real project root): excluded / revisit
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.run_marks import load_marks
    marks = load_marks()
    for run in runs:
        m = marks.get(run.get("behavior_base", ""))
        run["mark"] = m["mark"] if m else None
        run["mark_reason"] = m.get("reason", "") if m else ""
        if run["mark"]:
            what = "EXCLUDED" if run["mark"] == "excluded" else "REVISIT LATER"
            run["next"] = {"label": f"{what}" + (f": {run['mark_reason']}" if run["mark_reason"] else ""),
                           "cmd": None, "gui": False, "runnable": False}
    return runs


def _qscore(run: dict) -> float:
    s = run["quality_score"]
    try:
        return float(s)
    except ValueError:
        return float("-inf")


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def _rel(root: Path, p: Path | None) -> str:
    if not p:
        return ""
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def write_status_csv(root: Path, runs: list[dict]) -> Path:
    out = root / "processing_status.csv"
    cols = ["rank", "priority", "mouse", "date", "munit", "behavior_run",
            "behavior_base", "quality", "quality_score", "stage", "checklist",
            "stack_local", "run_dir", "stem", "next_action", "next_command"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            cmd = r["next"]["cmd"]
            w.writerow([
                r["rank"], r["priority"], r["mouse"], r["date"], r["munit"], r["behavior_run"],
                r["behavior_base"], r["imaging_quality"], r["quality_score"], r["stage"],
                r["checklist"], "yes" if r["stack"] else "no",
                _rel(root, r["run_dir"]), r["stem"] or "",
                r["next"]["label"],
                " ".join(cmd) if cmd else "",
            ])
    return out


def print_table(runs: list[dict]) -> None:
    hdr = f"{'rk':>2}  {'pri':<3} {'quality':<8} {'run (behavior_base)':<34} {'stage':<16} next_action"
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        q = r["imaging_quality"] or "?"
        qs = r["quality_score"]
        qcol = f"{q}({qs})" if qs else q
        base = r["behavior_base"] or f"{r['mouse']} {r['munit']}"
        print(f"{r['rank']:>2}  {r['priority']:<3} {qcol:<8} {base:<34} {r['stage']:<16} {r['next']['label']}")


def cmd_next(runs: list[dict], root: Path, run_it: bool) -> int:
    cand = [r for r in runs if r["stack"] is not None and r["stage"] != "complete" and not r.get("mark")]
    if not cand:
        print("All local runs are complete (or no local stacks). Nothing to do.")
        return 0
    r = min(cand, key=lambda d: d["rank"])   # highest-ranked (best) actionable local run
    nx = r["next"]
    print(f"NEXT: rank {r['rank']}  {r['behavior_base']}")
    print(f"  mouse/date/munit : {r['mouse']} {r['date']} {r['munit']}  (behavior {r['behavior_run']})")
    print(f"  quality          : {r['imaging_quality']} (score {r['quality_score']}, {r['priority']})")
    print(f"  stage            : {r['stage']}   [{r['checklist']}]")
    print(f"  run dir          : {r['run_dir']}")
    print(f"  do next          : {nx['label']}")
    print(f"  command          :\n  {cmd_display(nx['cmd'])}\n")

    if not run_it:
        return 0
    if nx["cmd"] is None:
        print("  --run-it: nothing runnable for this stage.")
        return 0
    if nx["gui"]:
        print("  --run-it: this is an interactive napari GUI step; launch it yourself (printed above).")
        return 0
    print(f"  --run-it: executing (cwd={root}) ...\n")
    sys.stdout.flush()
    return subprocess.call(nx["cmd"], cwd=str(root))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="femtonics-data root (default: inferred)")
    ap.add_argument("--filter", default=None, metavar="stage=NAME",
                    help="show only runs at a stage, e.g. --filter stage=segments_located")
    ap.add_argument("--mouse", default=None, help="show only this mouse")
    ap.add_argument("--min-quality", type=float, default=None,
                    help="show only runs with quality_score >= this")
    ap.add_argument("--next", action="store_true", help="print the single next run to act on")
    ap.add_argument("--run-it", action="store_true",
                    help="with --next: execute the command (non-GUI stages only)")
    ap.add_argument("--no-csv", action="store_true", help="do not (re)write processing_status.csv")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else project_root()
    runs = build_status(root)

    if args.next:
        return cmd_next(runs, root, args.run_it)

    view = runs
    if args.filter:
        if not args.filter.startswith("stage="):
            ap.error("--filter must look like stage=NAME")
        want = args.filter.split("=", 1)[1]
        if want not in STAGE_IDX:
            ap.error(f"unknown stage '{want}'; valid: {', '.join(STAGES)}")
        view = [r for r in view if r["stage"] == want]
    if args.mouse:
        view = [r for r in view if r["mouse"] == args.mouse]
    if args.min_quality is not None:
        view = [r for r in view if _qscore(r) >= args.min_quality]

    if not args.no_csv:
        out = write_status_csv(root, runs)   # CSV always covers ALL runs
        print(f"wrote {out.relative_to(root)}  ({len(runs)} runs)\n")

    print_table(view)

    # compact stage tally over all runs
    tally: dict[str, int] = {}
    for r in runs:
        tally[r["stage"]] = tally.get(r["stage"], 0) + 1
    print("\nstage tally (all 44): " +
          "  ".join(f"{s}={tally[s]}" for s in STAGES if s in tally))
    if view is not runs:
        print(f"(showing {len(view)} of {len(runs)} runs after filters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
