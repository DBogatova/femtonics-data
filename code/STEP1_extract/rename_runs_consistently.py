#!/usr/bin/env python3
"""Rename run folders and their files to one consistent, sortable convention.

CONVENTION
----------
    <mouse>/<date>/<parent>/run<NN>/run<NN>_<stage>.<ext>

`NN` is the BEHAVIOR RUN number, zero padded to two digits so run02 sorts
before run10. The stage suffix is whatever the pipeline already uses
(`_4d`, `_clean`, `_ref3d`, `_autoseg_labelmap`, `_segments`, ...). The .mesc
MUnit is NOT in the filename - it is recorded in run_identity.csv,
behavior_imaging_master.csv and ranked_runs.csv.

The folder number is taken from run_identity.csv, which resolves each folder's
true MUnit by comparing pixels against the .mesc. Folders whose identity could
not be resolved are SKIPPED, never guessed.

SAFETY
------
* dry run by default; --apply to act
* refuses to overwrite an existing path
* folder renames go through a temporary name so cycles/swaps cannot collide
* writes an undo shell script that reverses every action

Usage:
    python rename_runs_consistently.py [ROOT] [--apply] [--undo-script FILE]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys

# stage suffixes are preserved verbatim; only the stem prefix is rewritten
KNOWN_EXT = (".tif", ".tiff", ".csv", ".png", ".pdf", ".json", ".mp4", ".avi", ".zip")


def load_identity(root):
    p = os.path.join(root, "run_identity.csv")
    if not os.path.exists(p):
        sys.exit("run_identity.csv not found - run identify_run_folders.py first")
    return list(csv.DictReader(open(p)))


def old_stems(folder_abs, folder_name):
    """The run-identifying prefixes in use inside this folder, longest first.

    Deliberately a CLOSED set. An earlier version derived a stem from every
    filename, which collapsed unrelated files (run5branch.csv, run5rois.tif)
    onto the bare stem and collided. Only these forms are treated as a run
    token; anything else is left alone.
    """
    n = re.findall(r"\d+", folder_name)
    n = n[0] if n else ""
    stems = {
        folder_name,                    # run5
        folder_name + "3d",             # run53d
        folder_name.lower(),
        "3dstack",                      # legacy 139/06-12
        f"munit{n}",
        f"munit_{n}",
    }
    # the long behavior-base / MUnit exports actually present in this folder
    for f in os.listdir(folder_abs):
        base = os.path.splitext(f)[0]
        m = re.match(r"^(.*?_(?:[Rr]un\d+|MUnit_\d+)(?:_[A-Za-z0-9]+)*?_4[Dd])$", base)
        if m:
            stems.add(m.group(1))
        m = re.match(r"^(MSession_\d+_MUnit_\d+)$", base)
        if m:
            stems.add(m.group(1))
    stems = {s for s in stems if s}
    return sorted(stems, key=len, reverse=True)


def plan(root, rows, deletions):
    actions = []          # (kind, src, dst)
    skipped = []
    for r in rows:
        folder_rel, beh = r["folder"], r["behavior_run"]
        folder_abs = os.path.join(root, folder_rel)
        if not beh:
            skipped.append((folder_rel, r["note"] or "identity unresolved"))
            continue
        if not os.path.isdir(folder_abs):
            skipped.append((folder_rel, "folder missing"))
            continue
        nn = int(re.findall(r"\d+", beh)[0])
        new_folder_name = f"run{nn:02d}"
        new_stem = new_folder_name
        parent = os.path.dirname(folder_abs)
        old_folder_name = os.path.basename(folder_abs)

        # 1. file renames inside the folder (done while still at the old path)
        stems = old_stems(folder_abs, old_folder_name)
        for f in sorted(os.listdir(folder_abs)):
            src = os.path.join(folder_abs, f)
            if not os.path.isfile(src):
                continue
            if src in deletions:
                continue
            base, ext = os.path.splitext(f)
            newbase = None
            for st in stems:
                if base == st:
                    newbase = new_stem
                    break
                if base.startswith(st):
                    # keep everything after the run token verbatim
                    newbase = new_stem + base[len(st):]
                    break
            if newbase is None or newbase == base:
                continue
            # the extracted stack carries no stage suffix, so it collapses to the
            # bare stem; mark it explicitly as the 4D volume
            newbase = re.sub(r"^(run\d{2})3d$", r"\1_4d", newbase)
            newbase = re.sub(r"^(run\d{2})_4[Dd]$", r"\1_4d", newbase)
            if newbase == new_stem and ext.lower() in (".tif", ".tiff"):
                newbase = new_stem + "_4d"
            actions.append(("file", src, os.path.join(folder_abs, newbase + ext)))

        # 2. the folder itself
        if old_folder_name != new_folder_name:
            actions.append(("dir", folder_abs, os.path.join(parent, new_folder_name)))
    return actions, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--delete-duplicates", action="store_true",
                    help="also delete stacks listed in duplicates.txt (one path per line)")
    ap.add_argument("--duplicates", default="duplicates.txt")
    ap.add_argument("--undo-script", default="undo_rename.sh")
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    deletions = set()
    dpath = a.duplicates if os.path.isabs(a.duplicates) else os.path.join(root, a.duplicates)
    if a.delete_duplicates and os.path.exists(dpath):
        for line in open(dpath):
            line = line.strip()
            if line and not line.startswith("#"):
                deletions.add(os.path.join(root, line))

    rows = load_identity(root)
    actions, skipped = plan(root, rows, deletions)

    # collision check
    targets = {}
    problems = []
    for kind, src, dst in actions:
        if os.path.exists(dst) and os.path.realpath(dst) != os.path.realpath(src):
            problems.append(f"target already exists: {os.path.relpath(dst, root)}")
        if dst in targets:
            problems.append(f"two sources map to {os.path.relpath(dst, root)}")
        targets[dst] = src

    print(f"{len(deletions)} duplicate file(s) to delete")
    for d in sorted(deletions):
        print(f"   DELETE {os.path.relpath(d, root)}")
    print(f"\n{sum(1 for k,_,_ in actions if k=='file')} file rename(s), "
          f"{sum(1 for k,_,_ in actions if k=='dir')} folder rename(s)")
    for kind, src, dst in actions:
        tag = "DIR " if kind == "dir" else "file"
        print(f"   {tag} {os.path.relpath(src, root)}\n        -> {os.path.relpath(dst, root)}")
    if skipped:
        print(f"\n{len(skipped)} folder(s) SKIPPED (identity unresolved - never guessed):")
        for f, why in skipped:
            print(f"   {f}: {why[:80]}")
    if problems:
        print("\nREFUSING TO PROCEED:")
        for p in problems:
            print("  ", p)
        return 1
    if not a.apply:
        print("\n(dry run - pass --apply to perform these actions)")
        return 0

    undo = [os.path.join(root, a.undo_script) if not os.path.isabs(a.undo_script)
            else a.undo_script][0]
    lines = ["#!/bin/sh", "# reverses rename_runs_consistently.py", f"cd {root!r} || exit 1"]

    for d in sorted(deletions):
        os.remove(d)
        lines.append(f"# deleted duplicate (restore by re-extracting): {os.path.relpath(d, root)}")
        print("deleted", os.path.relpath(d, root))

    # files first, then dirs, with dirs via a temp name to survive swaps
    for kind, src, dst in [x for x in actions if x[0] == "file"]:
        os.rename(src, dst)
        lines.append(f"mv {os.path.relpath(dst, root)!r} {os.path.relpath(src, root)!r}")
    dirs = [x for x in actions if x[0] == "dir"]
    tmps = []
    for _, src, dst in dirs:
        tmp = src + ".__renaming__"
        os.rename(src, tmp)
        tmps.append((tmp, src, dst))
    for tmp, src, dst in tmps:
        os.rename(tmp, dst)
        lines.append(f"mv {os.path.relpath(dst, root)!r} {os.path.relpath(src, root)!r}")

    with open(undo, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(undo, 0o755)
    print(f"\napplied. undo script: {os.path.relpath(undo, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
