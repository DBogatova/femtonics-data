#!/usr/bin/env python3
"""mask_eval.py — score auto_mask.py against hand-curated masks.

Ground truth = <run>_clean_autoseg_labelmap_reviewed.tif whose sibling
<run>_clean_autoseg_reviewed.json was written by the manual tool
(trace_mask_napari).  Files written by auto_mask itself are NOT ground truth and
are skipped (auto_mask writes to the same filename when --out-dir is the run dir).

auto_mask is re-run on every GT run into a scratch directory (never the run
folder), so the hand masks cannot be overwritten.

    .venv/bin/python code/STEP9_auto/mask_eval.py --out results.tsv
    .venv/bin/python code/STEP9_auto/mask_eval.py --mode rule    # no ML model (no train/test leakage)

Columns: dice, precision, recall, ratio (=auto/GT voxels), plus failure-mode
diagnostics in voxels:
  fp_halo  false positives within 2 voxels (YZ) of the GT  -> boundary/width error
  fp_far   false positives further away                    -> intruders / wrong branches
  fn_cols  false negatives in X columns where auto has no voxel at all -> truncation / missed branch
  fn_thin  remaining false negatives (auto present in column but too thin)
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import auto_mask as am  # noqa: E402

STRUCT = 2


def find_gt_runs():
    runs = []
    for f in sorted(glob.glob(str(ROOT / "*/*/preprocessed/*/*_clean_autoseg_labelmap_reviewed.tif"))):
        j = f.replace("_autoseg_labelmap_reviewed.tif", "_autoseg_reviewed.json")
        stack = f.replace("_autoseg_labelmap_reviewed.tif", ".tif")
        try:
            tools = {r.get("tool") for r in json.load(open(j)).get("reviews", [])}
        except Exception:
            tools = set()
        rel = str(Path(f).relative_to(ROOT))
        if "auto_mask" in tools or "trace_mask_napari" not in tools:
            print(f"[skip] {rel}: written by {sorted(t for t in tools if t)} - not a hand mask")
            continue
        if not Path(stack).exists() or not Path(stack.replace(".tif", "_ref3d.tif")).exists():
            print(f"[skip] {rel}: missing _clean.tif or _ref3d.tif")
            continue
        runs.append((stack, f))
    return runs


def training_cells():
    try:
        return set(json.load(open(am._MODEL_CARD)).get("training_cells", []))
    except Exception:
        return set()


def score(auto: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((auto & gt).sum())
    na, ng = int(auto.sum()), int(gt.sum())
    fp = auto & ~gt
    fn = gt & ~auto
    near_gt = ndi.binary_dilation(gt, structure=np.ones((5, 5, 1), bool))  # +-2 vox in Z,Y
    auto_cols = auto.any(axis=(0, 1))
    return {
        "dice": 2 * tp / (na + ng + 1e-9), "precision": tp / (na + 1e-9),
        "recall": tp / (ng + 1e-9), "ratio": na / (ng + 1e-9),
        "auto_vox": na, "gt_vox": ng,
        "fp_halo": int((fp & near_gt).sum()), "fp_far": int((fp & ~near_gt).sum()),
        "fn_cols": int(fn[:, :, ~auto_cols].sum()), "fn_thin": int(fn[:, :, auto_cols].sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["default", "rule"], default="default",
                    help="default = auto_mask as shipped; rule = ML model disabled (v0.4 fallback)")
    ap.add_argument("--out", help="write the table (TSV) here")
    ap.add_argument("--verbose", action="store_true", help="show auto_mask output")
    args = ap.parse_args()

    if args.mode == "rule":
        am._load_ml_model = lambda: None
    train = training_cells()
    rows = []
    with tempfile.TemporaryDirectory(prefix="mask_eval_") as tmp:
        for stack, gt_path in find_gt_runs():
            rel = str(Path(stack).relative_to(ROOT))[:-4]
            sub = Path(tmp) / rel.replace("/", "__")
            t0 = time.time()
            buf = io.StringIO()
            with contextlib.redirect_stdout(sys.stdout if args.verbose else buf):
                res = am.auto_mask(stack, str(sub))
            dt = time.time() - t0
            out = sub / Path(gt_path).name
            if res is None or not out.exists():
                print(f"[fail] {rel}\n{buf.getvalue()[-2000:]}")
                continue
            r = score(tifffile.imread(out) == STRUCT, tifffile.imread(gt_path) == STRUCT)
            r.update(run=rel, sec=dt, in_train=rel in train)
            rows.append(r)

    cols = ["dice", "precision", "recall", "ratio", "auto_vox", "gt_vox",
            "fp_halo", "fp_far", "fn_cols", "fn_thin", "sec"]
    hdr = "run\ttrain\t" + "\t".join(cols)
    lines = [hdr]
    for r in rows:
        lines.append(f"{r['run']}\t{'Y' if r['in_train'] else 'n'}\t" + "\t".join(
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    for name, sel in (("MEDIAN all", rows), ("MEDIAN held-out", [r for r in rows if not r["in_train"]])):
        if sel:
            lines.append(f"{name} (n={len(sel)})\t-\t" + "\t".join(
                f"{np.median([r[c] for r in sel]):.3f}" for c in cols))
    text = f"# mask_eval mode={args.mode} auto_mask v{am.__version__} {time.strftime('%Y-%m-%d %H:%M')}\n" \
           + "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
