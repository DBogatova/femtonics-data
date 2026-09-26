#!/usr/bin/env python3
"""Identify which .mesc MUnit each on-disk run folder actually contains.

WHY
---
Run folders are named `run<N>`, but N has meant different things in different
sessions (MUnit index in some, behavior run number in others). Shape alone
cannot disambiguate: runs acquired back-to-back with identical settings have
identical (T,Z,Y,X). e.g. rbp4_phpebach/06-26 run5/run6/run7 could each be any
of MUnit_6/7/8/9.

HOW
---
Compare actual PIXELS against the .mesc. For every candidate MUnit (same shape),
read a few volumes straight out of the container and compare with the same
volumes of the on-disk stack. The true unit matches exactly; a different
recording of the same shape does not. For a cleaned/processed stack an exact
match is not expected, so the candidate with the (dramatically) lowest mean
absolute difference wins, and the margin over the runner-up is reported so a
weak call is visible rather than silent.

Writes run_identity.csv: one row per run folder, with the resolved MUnit, the
behavior run it therefore belongs to, and the evidence.

Usage:
    python identify_run_folders.py [ROOT] [-o run_identity.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import h5py
import numpy as np
import tifffile

CONTAINER_EXTS = (".mesc", ".hdf", ".hdf5")
N_PROBE = 4                      # volumes sampled per candidate
BIG = 50_000_000                 # a "stack" is at least this many bytes


def containers(directory):
    out = []
    for ext in CONTAINER_EXTS:
        out += [os.path.join(directory, f) for f in os.listdir(directory)
                if f.endswith(ext)] if os.path.isdir(directory) else []
    return sorted(set(out))


def load_master(root):
    p = os.path.join(root, "behavior_imaging_master.csv")
    rows = [r for r in csv.DictReader(open(p)) if r["munit"]]
    return rows


def stack_shape(p):
    try:
        with tifffile.TiffFile(p) as t:
            return tuple(t.series[0].shape)
    except Exception:
        return None


def unit_volumes(container, session, unit, n_t, nz, idx):
    """Read volumes `idx` of session/unit straight from the container."""
    with h5py.File(container, "r") as f:
        ds = f[f"{session}/{unit}/Channel_0"]
        out = []
        for i in idx:
            a = ds[i * nz:(i + 1) * nz]        # (nz, y, x)
            out.append(np.asarray(a))
        return out


def stack_volumes(p, idx, nz):
    arr = tifffile.imread(p)
    out = []
    for i in idx:
        v = arr[i]
        if v.ndim == 2:                        # ribbon: (y,x) per timepoint
            v = v[None, ...]
        out.append(np.asarray(v))
    del arr
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("-o", "--out", default="run_identity.csv")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    master = load_master(root)

    by_session = {}
    for r in master:
        by_session.setdefault((r["mouse"], r["date"]), []).append(r)

    rows = []
    for (mouse, date), rs in sorted(by_session.items()):
        sroot = os.path.join(root, mouse, date)
        if not os.path.isdir(sroot):
            continue
        # every run<N> folder in this session
        folders = []
        for dirpath, dirnames, filenames in os.walk(sroot):
            for d in dirnames:
                if re.fullmatch(r"[Rr]un\d+", d):
                    folders.append(os.path.join(dirpath, d))
        folders = sorted(set(folders))
        if not folders:
            continue

        cont = containers(os.path.join(sroot, "raw")) or containers(sroot)
        cont = cont[0] if cont else None

        for d in folders:
            stacks = [os.path.join(d, f) for f in sorted(os.listdir(d))
                      if f.lower().endswith((".tif", ".tiff"))
                      and os.path.getsize(os.path.join(d, f)) > BIG]
            if not stacks:
                continue
            # prefer a raw (non-clean, non-denoised) stack for an exact test
            raws = [p for p in stacks if not re.search(
                r"_clean|_denoised|_ref3d|_autoseg|_segments|labelmap", os.path.basename(p))]
            probe = raws[0] if raws else stacks[0]
            is_raw = bool(raws)
            sh = stack_shape(probe)
            if sh is None:
                continue

            cands = []
            for r in rs:
                nz = int(float(r["n_slices"] or 1))
                want = (int(r["n_t"]), nz, int(r["n_y"]), int(r["n_x"]))
                if tuple(sh) == want or tuple(sh) == tuple(x for x in want if x != 1):
                    cands.append(r)

            row = {
                "mouse": mouse, "date": date,
                "folder": os.path.relpath(d, root),
                "folder_number": int(re.findall(r"\d+", os.path.basename(d))[0]),
                "probe_file": os.path.basename(probe),
                "probe_is_raw": "yes" if is_raw else "no (processed)",
                "shape": "x".join(map(str, sh)),
                "n_shape_candidates": len(cands),
                "resolved_munit": "", "behavior_run": "", "method": "",
                "margin": "", "note": "",
            }

            if not cands:
                row["note"] = "shape matches no unit in this session's metadata"
                rows.append(row)
                continue
            if len(cands) == 1:
                r = cands[0]
                row.update(resolved_munit=r["munit"], method="unique shape",
                           behavior_run=(f"Run{int(float(r['behavior_run_number'])):03d}"
                                         if r["behavior_run_number"] else ""))
                rows.append(row)
                continue
            if cont is None:
                row["note"] = ("ambiguous and no local container to compare pixels: "
                               + ",".join(c["munit"] for c in cands))
                rows.append(row)
                continue

            nz = int(float(cands[0]["n_slices"] or 1))
            n_t = sh[0]
            idx = np.unique(np.linspace(0, n_t - 1, N_PROBE).astype(int))
            try:
                disk = stack_volumes(probe, idx, nz)
            except Exception as e:
                row["note"] = f"could not read stack: {e}"
                rows.append(row)
                continue

            scores = []
            for c in cands:
                try:
                    vols = unit_volumes(cont, c["mesc_session"], c["munit"],
                                        int(c["n_t"]), nz, idx)
                except Exception as e:
                    scores.append((float("inf"), c, f"read error {e}"))
                    continue
                diffs = []
                for v_disk, v_mesc in zip(disk, vols):
                    vd = np.squeeze(v_disk).astype(np.float64)
                    vm = np.squeeze(v_mesc).astype(np.float64)
                    if vd.shape != vm.shape:
                        diffs.append(float("inf"))
                        continue
                    diffs.append(float(np.abs(vd - vm).mean()))
                scores.append((float(np.mean(diffs)), c, ""))
            scores.sort(key=lambda s: s[0])
            best, runner = scores[0], (scores[1] if len(scores) > 1 else None)
            r = best[1]
            exact = best[0] == 0.0
            row.update(
                resolved_munit=r["munit"],
                behavior_run=(f"Run{int(float(r['behavior_run_number'])):03d}"
                              if r["behavior_run_number"] else ""),
                method=("exact pixel match vs .mesc" if exact
                        else "lowest pixel difference vs .mesc"),
                margin=("inf" if runner is None else
                        (f"{runner[0]:.3f} vs {best[0]:.3f}")),
            )
            if not exact and runner is not None and runner[0] < 5 * max(best[0], 1e-9):
                row["note"] = ("WEAK: runner-up nearly as close, do not trust "
                               "without another check")
            rows.append(row)

    # flag folder-number vs behavior-run disagreement
    for row in rows:
        if row["behavior_run"]:
            bn = int(re.findall(r"\d+", row["behavior_run"])[0])
            if bn != row["folder_number"]:
                row["note"] = (f"MISNAMED: folder says run{row['folder_number']} but this "
                               f"is behavior {row['behavior_run']}. "
                               + row["note"]).strip()

    out = a.out if os.path.isabs(a.out) else os.path.join(root, a.out)
    cols = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} run folders)\n")
    for r in rows:
        print(f"  {r['folder']:52}{r['resolved_munit']:9}{r['behavior_run']:8}"
              f"{r['method']:34}{r['note'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
