#!/usr/bin/env python3
"""run_all.py — run the full automatic pipeline on all 20 mirrored runs.

Steps:
  1. Ensure mirror is built (make_auto_tree.py)
  2. auto_mask.py on every run (6 parallel)
  3. auto_regions.py on every run (6 parallel)
  4. coherence_with_behavior.py per run
  5. run_metrics.py --all, coupling_phenotype.py --all, behavior_coupling.py --all, cohort_stats.py

All outputs land in AUTO_ROOT.  The real data tree is never modified.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
AUTO_ROOT = PROJECT / "auto_pipeline"
PYTHON = "/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/.venv311/bin/python"
LOG_DIR = AUTO_ROOT / "logs"
LOG_FILE = LOG_DIR / "runner.jsonl"
MAX_WORKERS = 6


def log(stage: str, what: str, result: str):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "what": what,
        "result": result,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def find_runs() -> list[dict]:
    """Find all 20 mirror run directories with *_clean.tif + *_ref3d.tif."""
    runs = []
    for clean in sorted(AUTO_ROOT.rglob("*_clean.tif")):
        if "old" in clean.parts or any(part.startswith("_") for part in clean.relative_to(AUTO_ROOT).parts):
            continue                      # archives and scratch copies (_audit_*, _selftest_*)
        run_dir = clean.parent
        stem = clean.stem
        ref3d = run_dir / f"{stem}_ref3d.tif"
        if not ref3d.exists():
            continue
        runs.append({
            "run_dir": run_dir,
            "stack": clean,
            "stem": stem,
            "label": str(run_dir.relative_to(AUTO_ROOT)),
        })
    return runs


def run_one_mask(run: dict) -> dict:
    """Run auto_mask.py on one run. Returns result dict."""
    t0 = time.time()
    stack = run["stack"]
    out_dir = run["run_dir"]
    cmd = [
        PYTHON, str(PROJECT / "code/STEP9_auto/auto_mask.py"),
        str(stack), "--out-dir", str(out_dir),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           cwd=str(PROJECT))
        elapsed = time.time() - t0
        success = r.returncode == 0
        # Check output files
        stem = run["stem"]
        reviewed = out_dir / f"{stem}_autoseg_labelmap_reviewed.tif"
        has_mask = reviewed.exists()
        return {
            "label": run["label"],
            "success": success and has_mask,
            "elapsed": elapsed,
            "returncode": r.returncode,
            "stdout_tail": r.stdout[-500:] if r.stdout else "",
            "stderr_tail": r.stderr[-500:] if r.stderr else "",
        }
    except Exception as e:
        return {
            "label": run["label"],
            "success": False,
            "elapsed": time.time() - t0,
            "error": str(e),
        }


def run_one_regions(run: dict) -> dict:
    """Run auto_regions.py on one run. Returns result dict."""
    t0 = time.time()
    stack = run["stack"]
    out_dir = run["run_dir"]
    stem = run["stem"]
    mask = out_dir / f"{stem}_autoseg_labelmap_reviewed.tif"
    exclude = out_dir / f"{stem}_exclude_labelmap.tif"
    if not mask.exists():
        return {"label": run["label"], "success": False, "error": "no mask"}
    cmd = [
        PYTHON, str(PROJECT / "code/STEP9_auto/auto_regions.py"),
        str(stack), "--mask", str(mask), "--out-dir", str(out_dir),
    ]
    if exclude.exists():
        cmd += ["--exclude", str(exclude)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200,
                           cwd=str(PROJECT))
        elapsed = time.time() - t0
        success = r.returncode == 0
        seg = out_dir / f"{stem}_segments_final.tif"
        has_seg = seg.exists()
        return {
            "label": run["label"],
            "success": success and has_seg,
            "elapsed": elapsed,
            "returncode": r.returncode,
            "stdout_tail": r.stdout[-500:] if r.stdout else "",
            "stderr_tail": r.stderr[-500:] if r.stderr else "",
        }
    except Exception as e:
        return {
            "label": run["label"],
            "success": False,
            "elapsed": time.time() - t0,
            "error": str(e),
        }


def run_coherence(run: dict) -> dict:
    """Run coherence_with_behavior.py on one run."""
    t0 = time.time()
    run_dir = run["run_dir"]
    cmd = [
        PYTHON, str(PROJECT / "code/STEP7_workflow/coherence_with_behavior.py"),
        "--run-dir", str(run_dir),
        "--root", str(AUTO_ROOT),
        "--show-other", "--force",
    ]
    env = os.environ.copy()
    env["FEMTO_ROOT"] = str(AUTO_ROOT)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           cwd=str(PROJECT), env=env)
        elapsed = time.time() - t0
        return {
            "label": run["label"],
            "success": r.returncode == 0,
            "elapsed": elapsed,
            "returncode": r.returncode,
            "stdout_tail": r.stdout[-500:] if r.stdout else "",
            "stderr_tail": r.stderr[-500:] if r.stderr else "",
        }
    except Exception as e:
        return {
            "label": run["label"],
            "success": False,
            "elapsed": time.time() - t0,
            "error": str(e),
        }


def run_stats_tool(name: str, script: str, args: list[str]) -> dict:
    """Run a stats tool with FEMTO_ROOT set."""
    t0 = time.time()
    cmd = [PYTHON, str(PROJECT / script)] + args
    env = os.environ.copy()
    env["FEMTO_ROOT"] = str(AUTO_ROOT)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200,
                           cwd=str(PROJECT), env=env)
        elapsed = time.time() - t0
        return {
            "name": name,
            "success": r.returncode == 0,
            "elapsed": elapsed,
            "returncode": r.returncode,
            "stdout_tail": r.stdout[-500:] if r.stdout else "",
            "stderr_tail": r.stderr[-500:] if r.stderr else "",
        }
    except Exception as e:
        return {
            "name": name,
            "success": False,
            "elapsed": time.time() - t0,
            "error": str(e),
        }


def parallel_run(func, runs, desc: str, max_workers=MAX_WORKERS):
    """Run func on each run in parallel, return results list."""
    results = []
    n = len(runs)
    print(f"\n{'='*60}")
    print(f"  {desc} ({n} runs, {max_workers} workers)")
    print(f"{'='*60}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(func, r): r["label"] for r in runs}
        for future in as_completed(futures):
            label = futures[future]
            try:
                res = future.result()
                ok = "OK" if res.get("success") else "FAIL"
                elapsed = res.get("elapsed", 0)
                print(f"  [{ok}] {label} ({elapsed:.1f}s)")
                if not res.get("success"):
                    err = res.get("error", res.get("stderr_tail", ""))
                    if err:
                        print(f"        {err[:200]}")
                results.append(res)
                log("runner", f"{desc}:{label}", f"{ok} {elapsed:.1f}s")
            except Exception as e:
                print(f"  [ERR] {label}: {e}")
                results.append({"label": label, "success": False, "error": str(e)})
    total = time.time() - t0
    ok_count = sum(1 for r in results if r.get("success"))
    print(f"  Done: {ok_count}/{n} succeeded in {total:.1f}s total")
    return results


def main():
    t_start = time.time()
    log("runner", "start", f"pid={os.getpid()}")
    print(f"AUTO_ROOT: {AUTO_ROOT}")
    print(f"PYTHON:    {PYTHON}")

    # ── Step 0: Verify mirror ─────────────────────────────────────────────
    runs = find_runs()
    print(f"\nFound {len(runs)} mirror runs")
    for r in runs:
        print(f"  {r['label']}: {r['stem']}")
    log("runner", "runs_found", f"{len(runs)} runs")

    if len(runs) != 20:
        print(f"WARNING: expected 20 runs, found {len(runs)}")

    # ── Step 1: Masks (parallel) ──────────────────────────────────────────
    mask_results = parallel_run(run_one_mask, runs, "auto_mask")

    # ── Step 2: Regions (parallel) ────────────────────────────────────────
    region_results = parallel_run(run_one_regions, runs, "auto_regions")

    # ── Step 3: Coherence figures (parallel) ──────────────────────────────
    # Only for runs that got regions
    runs_with_regions = []
    for r in runs:
        seg = r["run_dir"] / f"{r['stem']}_segments_final.tif"
        if seg.exists():
            runs_with_regions.append(r)
    print(f"\n{len(runs_with_regions)} runs have regions, building coherence figures")
    coh_results = parallel_run(run_coherence, runs_with_regions, "coherence_figures")

    # ── Step 4: Stats ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Running statistics tools")
    print(f"{'='*60}")

    stats_tools = [
        ("run_metrics", "code/STEP8_stats/run_metrics.py", ["--all", "--force"]),
        ("coupling_phenotype", "code/STEP8_stats/coupling_phenotype.py", ["--all", "--force"]),
        ("behavior_coupling", "code/STEP8_stats/behavior_coupling.py", ["--all", "--force"]),
        ("cohort_stats", "code/STEP8_stats/cohort_stats.py", []),
        ("paper_stats", "code/STEP9_auto/paper_stats.py", []),
    ]
    stats_results = []
    for name, script, args in stats_tools:
        res = run_stats_tool(name, script, args)
        ok = "OK" if res["success"] else "FAIL"
        print(f"  [{ok}] {name} ({res['elapsed']:.1f}s)")
        if not res["success"]:
            err = res.get("error", res.get("stderr_tail", ""))
            print(f"        {err[:300]}")
        stats_results.append(res)
        log("runner", f"stats:{name}", f"{ok} {res['elapsed']:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  SUMMARY  (total: {total_time:.0f}s = {total_time/60:.1f}min)")
    print(f"{'='*60}")

    mask_ok = sum(1 for r in mask_results if r.get("success"))
    region_ok = sum(1 for r in region_results if r.get("success"))
    coh_ok = sum(1 for r in coh_results if r.get("success"))
    stats_ok = sum(1 for r in stats_results if r.get("success"))

    print(f"  Masks:      {mask_ok}/{len(runs)}")
    print(f"  Regions:    {region_ok}/{len(runs)}")
    print(f"  Coherence:  {coh_ok}/{len(runs_with_regions)}")
    print(f"  Stats:      {stats_ok}/{len(stats_tools)}")

    # Per-run table
    print(f"\n{'label':<55} {'mask':>5} {'reg':>5} {'coh':>5}")
    print("-" * 75)
    for r in runs:
        label = r["label"]
        m = next((x for x in mask_results if x.get("label") == label), {})
        rg = next((x for x in region_results if x.get("label") == label), {})
        c = next((x for x in coh_results if x.get("label") == label), {})
        print(f"  {label:<53} {'OK' if m.get('success') else 'FAIL':>5} "
              f"{'OK' if rg.get('success') else 'FAIL':>5} "
              f"{'OK' if c.get('success') else 'FAIL':>5}")

    # Save summary
    summary = {
        "time": datetime.now(timezone.utc).isoformat(),
        "total_seconds": total_time,
        "mask_ok": mask_ok, "mask_total": len(runs),
        "region_ok": region_ok, "region_total": len(runs),
        "coherence_ok": coh_ok, "coherence_total": len(runs_with_regions),
        "stats_ok": stats_ok, "stats_total": len(stats_tools),
        "mask_results": mask_results,
        "region_results": region_results,
        "coherence_results": coh_results,
        "stats_results": stats_results,
    }
    summary_path = AUTO_ROOT / "logs" / "runner_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")
    log("runner", "done", f"{mask_ok}/{len(runs)} masks, {region_ok}/{len(runs)} regions, "
        f"{coh_ok}/{len(runs_with_regions)} coherence, {stats_ok}/{len(stats_tools)} stats, "
        f"{total_time:.0f}s total")


if __name__ == "__main__":
    main()
