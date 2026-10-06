#!/usr/bin/env python3
"""STEP2-clean every newly extracted run<NN>_4d.tif into run<NN>_clean.tif.

The panel greys a run out until `<stem>_clean.tif` exists (femto_status: the
"not_local" stage is literally "no <stem>_clean.tif"), so extraction alone does
not make a run appear -- this is the step that does.

The scan-line period is taken from the .mesc via auto_mask.period_from_mesc()
rather than from clean_register_3d's own auto-detection, which fails outright on
some runs (NaN period) and otherwise falls back to a hardcoded 24. The .mesc
value is the authoritative one and is recorded per run in the log.

Runs are independent, so they are cleaned in parallel.

Usage:
    python clean_new_extractions.py [--workers 4] [--dry-run] [--pattern run*_4d.tif]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEANER = ROOT / "code/STEP2_clean/clean_register_3d.py"
sys.path.insert(0, str(ROOT / "code/STEP9_auto"))
sys.path.insert(0, str(ROOT / "code/common"))


def period_for(run_dir: Path):
    try:
        import auto_mask
        got = auto_mask.period_from_mesc(run_dir)
        if got:
            return int(got[0]), got[1]
    except Exception as exc:
        return None, f"lookup failed: {exc}"
    return None, "not found in .mesc"


def shape_of(path: Path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        return tuple(t.series[0].shape) if t.series else (len(t.pages), *t.pages[0].shape)


def clean_one(src: Path):
    out = src.with_name(src.name.replace("_4d.tif", "_clean.tif"))
    if out.exists():
        try:
            if shape_of(out) == shape_of(src):
                return (src, "skip", "already clean, shape matches", None)
        except Exception:
            pass
    period, src_note = period_for(src.parent)
    cmd = [sys.executable, str(CLEANER), str(src), "--no-register", "--out", str(out)]
    if period:
        cmd += ["--period", str(period)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()
        return (src, "fail", (tail[-1] if tail else "unknown error"), period)
    if not out.exists():
        return (src, "fail", "no output written", period)
    try:
        if shape_of(out) != shape_of(src):
            bad = shape_of(out)
            out.unlink()
            return (src, "fail", f"shape changed {bad} != {shape_of(src)}", period)
    except Exception as exc:
        return (src, "fail", f"unreadable output ({exc})", period)
    return (src, "ok", f"period={period} from {src_note}", period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pattern", default="*/*/preprocessed/run*/run*_4d.tif")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-free-gb", type=float, default=10.0)
    a = ap.parse_args()

    srcs = sorted(ROOT.glob(a.pattern))
    todo = [s for s in srcs if not s.with_name(s.name.replace("_4d.tif", "_clean.tif")).exists()]
    gb = sum(s.stat().st_size for s in todo) / 1e9
    free = shutil.disk_usage(ROOT).free / 1e9
    print(f"{len(srcs)} extracted stacks, {len(todo)} need cleaning; "
          f"{gb:.1f} GB to write, {free:.1f} GB free -> {free - gb:.1f} GB after")
    if free - gb < a.keep_free_gb:
        sys.exit(f"refusing: would leave {free - gb:.1f} GB")
    if a.dry_run:
        for s in todo:
            per, note = period_for(s.parent)
            print(f"  would clean {s.relative_to(ROOT)}  period={per} ({note})")
        return

    ok = fail = skip = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for src, status, note, _per in pool.map(clean_one, todo):
            rel = src.relative_to(ROOT)
            print(f"  [{status:4s}] {rel}  {note}")
            ok += status == "ok"; fail += status == "fail"; skip += status == "skip"
    print(f"\ncleaned {ok}, skipped {skip}, failed {fail}")


if __name__ == "__main__":
    main()
