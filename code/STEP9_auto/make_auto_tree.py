#!/usr/bin/env python3
"""make_auto_tree.py - build / refresh the automatic-pipeline mirror tree.

Builds AUTO_ROOT/<mouse>/<date>/... mirroring the real relative paths for every
local run that femto_status reports as having a cleaned stack on disk.

Mirror layout
-------------
  Large read-only inputs  -> HARD LINKS (same inode, zero extra space)
    runNN_clean.tif, runNN_4d.tif (if present), _ref3d.tif/.json,
    original _autoseg_labelmap.tif/.json and _autoseg.json

  Session sub-folders     -> DIRECTORY SYMLINKS (read-only use)
    behavior/, trigger/, raw/, snapshots/

  Root CSVs               -> COPIES (scripts writing to a hard-linked CSV would modify
    behavior_imaging_master.csv, ranked_runs.csv, mice.csv, run_identity.csv,
    and any other root-level .csv the tools read)

  Daria's curated files   -> NOT copied (mirror holds automatic results only)
    _reviewed, _exclude, _trace_session, _segments_final, figures, metrics

Safety
------
  - Refuses to run if the destination is inside a real rbp4_* folder.
  - Never overwrites a real data file.
  - Verifies hard links share the inode after creation.
  - Checks that AUTO_ROOT and the data root are on the same filesystem; falls back
    to symlinks with a warning if not.

Idempotent: re-running skips files whose hard links already share the inode and
whose symlinks already point to the right target.

Usage
-----
  python code/STEP9_auto/make_auto_tree.py                   # all local runs -> auto_pipeline/
  python code/STEP9_auto/make_auto_tree.py --run <run_dir>   # one run only
  python code/STEP9_auto/make_auto_tree.py --dest /tmp/x     # custom destination
  python code/STEP9_auto/make_auto_tree.py --include-curated # also hard-link curated files (testing only)
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PROJECT = CODE.parent

sys.path.insert(0, str(CODE / "STEP7_workflow"))
import femto_status as fs

AUTO_ROOT_DEFAULT = PROJECT / "auto_pipeline"

# Files to hard-link into each mirrored run directory (globs relative to the run dir).
HARDLINK_GLOBS = [
    "*_clean.tif",
    "*_4d.tif",
    "*_ref3d.tif",
    "*_ref3d.json",
    "*_autoseg_labelmap.tif",
    "*_autoseg_labelmap.json",
    "*_autoseg.json",
]

# Extra curated files to hard-link when --include-curated is set (testing only).
CURATED_GLOBS = [
    "*_autoseg_labelmap_reviewed.tif",
    "*_autoseg_reviewed.json",
    "*_exclude_labelmap.tif",
    "*_trace_session.npz",
    "*_segments_final.tif",
    "*_segments_final.json",
]

# Session sub-folders to symlink (directory-level, read-only).
SESSION_SYMLINK_DIRS = ["behavior", "trigger", "raw", "snapshots"]

# Root CSVs to copy (not hard-link: tools may write to them).
ROOT_CSVS = [
    "behavior_imaging_master.csv",
    "ranked_runs.csv",
    "mice.csv",
    "run_identity.csv",
    "imaging_only_runs.csv",
]

LOG_FILE = "logs/infra.jsonl"


def log_entry(dest: Path, stage: str, what: str, result: str):
    logdir = dest / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.datetime.now().isoformat(),
        "stage": stage,
        "what": what,
        "result": result,
    }
    with open(logdir / "infra.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def same_device(a: Path, b: Path) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def safe_hardlink(src: Path, dst: Path, *, fallback_symlink: bool = False) -> str:
    """Create a hard link dst -> src.  Returns 'hardlink', 'symlink', or 'exists'."""
    if dst.exists():
        if dst.is_file() and os.stat(dst).st_ino == os.stat(src).st_ino:
            return "exists"
        # Destination exists but is a different file: remove and re-link.
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if fallback_symlink:
        dst.symlink_to(src)
        return "symlink"
    os.link(src, dst)
    # Verify inode match.
    assert os.stat(dst).st_ino == os.stat(src).st_ino, \
        f"inode mismatch after hard-link: {src} -> {dst}"
    return "hardlink"


def safe_dirlink(src: Path, dst: Path) -> str:
    """Create a directory symlink dst -> src.  Returns 'symlink' or 'exists'."""
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return "exists"
        dst.unlink()
    elif dst.exists():
        # Something is here but it is not a symlink; leave it.
        return "skip_not_symlink"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)
    return "symlink"


def session_dir_of(run_dir: Path) -> Path:
    """<mouse>/<date>/ from <mouse>/<date>/preprocessed/runNN/ or <mouse>/<date>/runNN/."""
    if run_dir.parent.name == "preprocessed":
        return run_dir.parents[1]
    return run_dir.parent


def rel_run_dir(root: Path, run_dir: Path) -> Path:
    """Relative path of the run dir from the project root."""
    return run_dir.relative_to(root)


def mirror_run(root: Path, dest: Path, run_dir: Path, *,
               include_curated: bool = False, fallback_symlink: bool = False) -> dict:
    """Mirror a single run directory into dest. Returns a summary dict."""
    rel = rel_run_dir(root, run_dir)
    mirror_dir = dest / rel

    globs = list(HARDLINK_GLOBS)
    if include_curated:
        globs += CURATED_GLOBS

    linked = []
    for pat in globs:
        for src in sorted(run_dir.glob(pat)):
            if not src.is_file():
                continue
            dst = mirror_dir / src.name
            kind = safe_hardlink(src, dst, fallback_symlink=fallback_symlink)
            linked.append({"file": src.name, "kind": kind})

    # Session-level directory symlinks.
    session = session_dir_of(run_dir)
    mirror_session = dest / session.relative_to(root)
    syms = []
    for dname in SESSION_SYMLINK_DIRS:
        src_d = session / dname
        if src_d.is_dir():
            dst_d = mirror_session / dname
            kind = safe_dirlink(src_d, dst_d)
            syms.append({"dir": dname, "kind": kind})

    return {"run_dir": str(rel), "mirror_dir": str(mirror_dir),
            "hardlinks": linked, "symlinks": syms}


def copy_root_csvs(root: Path, dest: Path) -> list[str]:
    """Copy root-level CSVs.  Returns list of copied filenames."""
    copied = []
    for name in ROOT_CSVS:
        src = root / name
        if not src.exists():
            continue
        dst = dest / name
        # Only overwrite if the source is newer.
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(name)
    # Also copy any other root-level .csv that is not already handled.
    for f in sorted(root.glob("*.csv")):
        if f.name in ROOT_CSVS:
            continue
        dst = dest / f.name
        if dst.exists() and dst.stat().st_mtime >= f.stat().st_mtime:
            continue
        shutil.copy2(f, dst)
        copied.append(f.name)
    return copied


def validate_dest(dest: Path, root: Path):
    """Refuse if dest is inside a real rbp4_* folder."""
    try:
        rel = dest.relative_to(root)
        parts = rel.parts
        if parts and parts[0].startswith("rbp4_"):
            sys.exit(f"ABORT: destination {dest} is inside real data tree ({parts[0]}). "
                     f"Use a path under auto_pipeline/ or /tmp/.")
    except ValueError:
        pass  # dest is outside root entirely - fine


def get_local_runs(root: Path, single_run_dir: str | None = None) -> list[dict]:
    """Return femto_status runs that are local (have a stack on disk)."""
    runs = fs.build_status(root)
    local = [r for r in runs if r.get("stack") is not None and r.get("run_dir") is not None]
    if single_run_dir:
        target = Path(single_run_dir).resolve()
        local = [r for r in local if Path(r["run_dir"]).resolve() == target
                 or (root / r["run_dir"]).resolve() == target]
        if not local:
            sys.exit(f"no local run found matching {single_run_dir}")
    return local


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=None,
                    help="mirror destination (default: auto_pipeline/ in project root)")
    ap.add_argument("--run", default=None, help="mirror only this run directory")
    ap.add_argument("--include-curated", action="store_true",
                    help="also hard-link curated files (for testing only)")
    ap.add_argument("--root", default=None, help="project root (default: inferred)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else PROJECT
    dest = Path(args.dest).resolve() if args.dest else AUTO_ROOT_DEFAULT

    validate_dest(dest, root)
    dest.mkdir(parents=True, exist_ok=True)

    # Check if dest is on the same filesystem as root.
    fallback = False
    if not same_device(root, dest):
        print(f"WARNING: {dest} is on a different filesystem than {root}. "
              f"Hard links will be replaced with symlinks.")
        fallback = True

    local_runs = get_local_runs(root, args.run)
    print(f"Mirroring {len(local_runs)} local run(s) into {dest}")

    # Copy root CSVs first.
    copied_csvs = copy_root_csvs(root, dest)
    if copied_csvs:
        print(f"  root CSVs copied/updated: {', '.join(copied_csvs)}")

    summaries = []
    for r in local_runs:
        rd = r["run_dir"]
        rd_path = rd if isinstance(rd, Path) else Path(rd)
        if not rd_path.is_absolute():
            rd_path = root / rd_path
        s = mirror_run(root, dest, rd_path,
                       include_curated=args.include_curated,
                       fallback_symlink=fallback)
        n_new = sum(1 for h in s["hardlinks"] if h["kind"] != "exists")
        n_exist = sum(1 for h in s["hardlinks"] if h["kind"] == "exists")
        n_sym = sum(1 for sl in s["symlinks"] if sl["kind"] != "exists")
        label = f"{r.get('behavior_base', '?'):40s}"
        print(f"  {label}  {n_new} new links, {n_exist} existing, {n_sym} dir symlinks -> {s['run_dir']}")
        summaries.append(s)

    log_entry(dest, "infra", "make_auto_tree",
              f"mirrored {len(summaries)} runs, copied {len(copied_csvs)} root CSVs, "
              f"dest={dest}, include_curated={args.include_curated}")

    print(f"\nDone. Mirror at {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
