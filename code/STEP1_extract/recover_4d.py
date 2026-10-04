#!/usr/bin/env python3
"""Make 4D stacks (*_4d.tif, raw/*MUnit_N*_4D.tif) safely removable and restorable.

A 4D stack is only an intermediate: it is extracted from the .mesc with
extract_mesc.py --unit MUnit_N --nz Z, and the analysis uses the cleaned copy
(*_clean.tif). This tool records, per stack, exactly how to rebuild it and a checksum
of its pixel data, so a removed stack can be re-extracted and proven identical.

    python code/STEP1_extract/recover_4d.py manifest            # write RECOVERY_4D.csv
    python code/STEP1_extract/recover_4d.py verify PATH [PATH]  # re-extract to /tmp, compare checksums
    python code/STEP1_extract/recover_4d.py remove              # remove stacks whose recipe is verified-class
    python code/STEP1_extract/recover_4d.py restore PATH|all    # re-extract removed stacks in place

RECOVERY_4D.csv (project root): path, mesc, munit, nz, shape, pixel_md5, recipe_ok, removed, note.
recipe_ok = the .mesc is on disk, the unit + nz are known and the metadata shape matches.
If the .mesc itself is later removed, `restore` tells you where it came from (SCC path
in <mouse>/SOURCE_scc_ayla.md or the session's raw/ folder).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT / "RECOVERY_4D.csv"
PY = sys.executable
FIELDS = ["path", "mesc", "munit", "nz", "shape", "pixel_md5", "recipe_ok", "removed", "note"]


def pixel_md5(path: Path) -> str:
    h = hashlib.md5()
    with tifffile.TiffFile(path) as t:
        for p in t.pages:                       # page by page: constant memory
            h.update(np.ascontiguousarray(p.asarray()).tobytes())
    return h.hexdigest()


def tif_shape(path: Path):
    with tifffile.TiffFile(path) as t:
        s = t.series[0].shape
        n_pages = len(t.pages)
    return tuple(int(v) for v in s), n_pages


def read_csv(p):
    return list(csv.DictReader(open(p, newline=""))) if p.exists() else []


def session_of(p: Path) -> Path:
    s = p
    while s.parent != s and not re.fullmatch(r"\d\d-\d\d-\d{4}", s.name):
        s = s.parent
    return s


def recipe_for(p: Path, master, imaging_only):
    """(mesc_path, munit, nz, expected (n_t, nz, ny, nx), note) or None."""
    rel = str(p.relative_to(PROJECT))
    sess = session_of(p); sess_rel = str(sess.relative_to(PROJECT))
    for r in imaging_only:
        if r.get("run_dir") and rel.startswith(r["run_dir"].rstrip("/") + "/"):
            m = sess / "raw" / r["mesc_file"]
            return m, r["munit"], int(r["n_slices"]), (int(r["n_t"]), int(r["n_slices"]), int(r["n_y"]), int(r["n_x"])), "imaging_only_runs.csv"
    rows = [r for r in master if r.get("session_dir") == sess_rel and r.get("munit")]
    mu = re.search(r"MUnit_(\d+)", p.name)
    if mu:                                              # raw/<...>_MUnit_N_..._4D.tif
        hit = [r for r in rows if r["munit"] == f"MUnit_{mu.group(1)}"]
        note = "master by MUnit in file name"
    else:
        fm = re.fullmatch(r"run(\d+)", p.parent.name)
        if not fm:
            return None
        n = int(fm.group(1))
        hit = [r for r in rows if (r.get("behavior_run_number") or "").strip().lstrip("0") == str(n)]
        note = "master by behavior run number (folder = behavior run)"
    if len(hit) != 1 or not (hit[0].get("n_slices") or "").strip():
        return None
    r = hit[0]
    m = sess / "raw" / r["mesc_file"]
    nz = int(float(r["n_slices"]))
    return m, r["munit"], nz, (int(float(r["n_t"])), nz, int(float(r["n_y"])), int(float(r["n_x"]))), note


def stacks():
    out = []
    for p in sorted(PROJECT.glob("rbp4_*/**/*.tif")):
        if "old" in p.parts or p.is_symlink():
            continue
        if p.name.endswith("_4d.tif") or re.search(r"_4D\.tif$", p.name):
            out.append(p)
    return out


def cmd_manifest(_a):
    master = read_csv(PROJECT / "behavior_imaging_master.csv")
    imaging_only = read_csv(PROJECT / "imaging_only_runs.csv")
    old = {r["path"]: r for r in read_csv(MANIFEST)}
    rows = []
    for p in stacks():
        rel = str(p.relative_to(PROJECT))
        rec = recipe_for(p, master, imaging_only)
        shape, n_pages = tif_shape(p)
        row = {"path": rel, "shape": "x".join(map(str, shape)), "removed": "", "note": ""}
        if rec is None:
            row.update(recipe_ok="no", note="unit/nz not identifiable - keep this stack")
        else:
            m, unit, nz, exp, note = rec
            same = (n_pages == exp[0] * exp[1]) and (shape[-2:] == exp[2:])
            row.update(mesc=str(m.relative_to(PROJECT)), munit=unit, nz=nz,
                       recipe_ok="yes" if (m.exists() and same) else "no",
                       note=note + ("" if m.exists() else "; .mesc not on disk") + ("" if same else f"; shape {shape} vs metadata {exp}"))
        prev = old.get(rel)
        row["pixel_md5"] = prev["pixel_md5"] if prev and prev.get("shape") == row["shape"] and prev.get("pixel_md5") else pixel_md5(p)
        rows.append(row); print(f"{row['recipe_ok']:3s} {rel}  {row.get('munit','')} nz={row.get('nz','')}  {row['note']}")
    for rel, prev in old.items():                      # keep records of already-removed stacks
        if prev.get("removed") and rel not in {r["path"] for r in rows}:
            rows.append(prev)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"wrote {MANIFEST.name}: {sum(r['recipe_ok']=='yes' for r in rows)} restorable / {len(rows)}")


def extract(row, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(PROJECT / "code/STEP1_extract/extract_mesc.py"), str(PROJECT / row["mesc"]), "--unit", row["munit"],
           "--nz", str(row["nz"]), "--out", str(out_dir), "--out-name", name]
    subprocess.run(cmd, check=True, cwd=str(PROJECT), stdout=subprocess.DEVNULL)
    return out_dir / f"{name}.tif"


def cmd_verify(a):
    rows = {r["path"]: r for r in read_csv(MANIFEST)}
    for rel in a.paths:
        r = rows[rel]
        got = extract(r, Path("/tmp/kiro_recover_verify"), "verify")
        ok = pixel_md5(got) == r["pixel_md5"]
        print(f"{'IDENTICAL' if ok else 'DIFFERENT'}  {rel}")
        got.unlink()
        if not ok:
            sys.exit(1)


def cmd_remove(a):
    rows = read_csv(MANIFEST); n = 0; freed = 0
    for r in rows:
        p = PROJECT / r["path"]
        if r["recipe_ok"] != "yes" or r.get("removed") or not p.exists():
            continue
        if not a.yes:
            print("would remove", r["path"]); continue
        freed += p.stat().st_size; p.unlink(); r["removed"] = "yes"; n += 1
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"removed {n} stacks, {freed / 1e9:.1f} GB" if a.yes else "dry run (add --yes)")


def cmd_restore(a):
    rows = read_csv(MANIFEST)
    for r in rows:
        if a.path not in ("all", r["path"]) or not r.get("removed"):
            continue
        p = PROJECT / r["path"]
        if not (PROJECT / r["mesc"]).exists():
            print(f"MISSING .mesc {r['mesc']} - copy it back first (see SOURCE_scc_ayla.md / the session raw/)"); continue
        got = extract(r, p.parent, p.stem)
        ok = pixel_md5(got) == r["pixel_md5"]
        r["removed"] = "" if ok else "yes"
        print(f"{'restored' if ok else 'CHECKSUM MISMATCH'}  {r['path']}")
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("manifest").set_defaults(f=cmd_manifest)
    v = sp.add_parser("verify"); v.add_argument("paths", nargs="+"); v.set_defaults(f=cmd_verify)
    r = sp.add_parser("remove"); r.add_argument("--yes", action="store_true"); r.set_defaults(f=cmd_remove)
    s = sp.add_parser("restore"); s.add_argument("path"); s.set_defaults(f=cmd_restore)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
