#!/usr/bin/env python3
"""Extract the 4D snake volume for ranked_runs.csv rows that have none yet.

Each row of ranked_runs.csv already carries everything needed: mesc_path, munit,
nz (from the .mesc protocol, never guessed), behavior_run, and the expected
(n_t, nz, n_y, n_x). This drives extract_mesc.py per run and then VERIFIES the
written stack against that expected shape, because a wrong --nz is otherwise
silent: the frame count still divides and you get a plausible-looking stack with
the planes interleaved wrongly.

Output follows the project convention so find_extracted()/the panel pick it up:
    <mouse>/<date>/preprocessed/run<NN>/run<NN>_4d.tif     (NN = behavior run, 2-digit)

Idempotent: a run whose output already exists with the correct shape is skipped.

Usage:
    python extract_ranked_snakes.py --ranks 12-15,19,22-26 [--dry-run]
    python extract_ranked_snakes.py --all-missing [--max-gb 20]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR = ROOT / "code/STEP1_extract/extract_mesc.py"
RANKED = ROOT / "ranked_runs.csv"


def parse_ranks(spec: str) -> set[int]:
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def expected_shape(r: dict) -> tuple[int, int, int, int]:
    return (int(r["n_t"]), int(r["nz"]), int(r["n_y"]), int(r["n_x"]))


def out_path_for(r: dict) -> Path:
    run = r["behavior_run"]                      # Run007
    nn = f"{int(run.replace('Run', '')):02d}"    # 07
    return ROOT / r["mouse"] / r["date"] / "preprocessed" / f"run{nn}" / f"run{nn}_4d.tif"


def actual_shape(path: Path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        if t.series:
            return tuple(t.series[0].shape)
        return (len(t.pages), *t.pages[0].shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", help="e.g. 12-15,19,22-26,28-32,35,36,38-44")
    ap.add_argument("--all-missing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-gb", type=float, default=0,
                    help="refuse to start if the total write exceeds this")
    ap.add_argument("--keep-free-gb", type=float, default=10.0,
                    help="refuse if it would leave less than this much free")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(RANKED)))
    if a.ranks:
        want = parse_ranks(a.ranks)
        rows = [r for r in rows if int(r["rank"]) in want]
    elif not a.all_missing:
        sys.exit("give --ranks or --all-missing")

    jobs, skipped = [], []
    for r in rows:
        if r["scan_type"] != "snake":
            skipped.append((r, "not a snake"))
            continue
        if not r["mesc_path"] or not (ROOT / r["mesc_path"]).exists():
            skipped.append((r, f"mesc not local: {r['mesc_path']}"))
            continue
        out = out_path_for(r)
        exp = expected_shape(r)
        if out.exists():
            try:
                if tuple(actual_shape(out)) == exp:
                    skipped.append((r, f"already extracted, shape OK {exp}"))
                    continue
                skipped.append((r, f"EXISTS with WRONG shape {actual_shape(out)} != {exp}"))
                continue
            except Exception as exc:
                skipped.append((r, f"exists but unreadable ({exc})"))
                continue
        jobs.append((r, out, exp))

    gb = sum(e[0] * e[1] * e[2] * e[3] * 2 / 1e9 for _r, _o, e in jobs)
    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"{len(jobs)} to extract, {len(skipped)} skipped; "
          f"{gb:.1f} GB to write, {free:.1f} GB free -> {free - gb:.1f} GB after")
    for r, why in skipped:
        print(f"  skip rank {r['rank']:>2s} {r['munit']:9s} {why}")
    if a.max_gb and gb > a.max_gb:
        sys.exit(f"refusing: {gb:.1f} GB exceeds --max-gb {a.max_gb}")
    if free - gb < a.keep_free_gb:
        sys.exit(f"refusing: would leave {free - gb:.1f} GB (< --keep-free-gb {a.keep_free_gb})")
    if a.dry_run:
        for r, out, exp in jobs:
            print(f"  would extract rank {r['rank']:>2s} {r['munit']:9s} nz={r['nz']:>2s} -> {out.relative_to(ROOT)}")
        return

    ok = bad = 0
    for i, (r, out, exp) in enumerate(jobs, 1):
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(EXTRACTOR), str(ROOT / r["mesc_path"]),
               "--unit", r["munit"], "--nz", r["nz"],
               "--out", str(out.parent), "--out-name", out.stem]
        print(f"\n[{i}/{len(jobs)}] rank {r['rank']} {r['mouse']} {r['date']} "
              f"{r['munit']} nz={r['nz']} -> {out.name}")
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"   FAILED: {p.stdout.strip()} {p.stderr.strip()[:200]}")
            bad += 1
            continue
        if not out.exists():
            print(f"   FAILED: no output at {out}")
            bad += 1
            continue
        got = tuple(actual_shape(out))
        if got != exp:
            print(f"   SHAPE MISMATCH: got {got}, expected {exp} -- WRONG nz, removing")
            out.unlink()
            bad += 1
            continue
        print(f"   ok shape {got}  {out.stat().st_size / 1e9:.2f} GB")
        ok += 1

    print(f"\nextracted {ok}, failed {bad}")


if __name__ == "__main__":
    main()
