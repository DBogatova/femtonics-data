#!/usr/bin/env python3
"""
compartment_coupling.py - Quantify soma / trunk / branch dependencies across runs.

For each run folder containing soma.csv, trunk.csv and branch*.csv:
  * convert each ROI-MIP trace to dF/F (F0 = 10th percentile)
  * Pearson correlation between compartments
  * cross-correlation lag (who leads whom, in frames)
  * event detection (robust MAD threshold) and event-by-event coincidence:
      - how many branch events are accompanied by a soma event within +/- W frames
      - how many branch events are SOMA-INDEPENDENT (the key metric for
        independent dendritic computation)

Usage:  .venv/bin/python code/extra/compartment_coupling.py ROOT [ROOT ...]
"""
import sys
import csv
import glob
import os
import numpy as np
from scipy.signal import find_peaks

W = 3            # coincidence window (frames)
MAXLAG = 20      # cross-correlation search (+/- frames)
K = 3.5          # event threshold = median + K * robust-sigma
MIN_DIST = 5     # min frames between events


def load(path):
    x, y = [], []
    with open(path, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) >= 2 and row[0].strip():
                x.append(float(row[0])); y.append(float(row[1]))
    return np.array(y)


def dff(v, pct=10):
    f0 = np.percentile(v, pct)
    return (v - f0) / (f0 if f0 else 1.0)


def events(tr):
    med = np.median(tr)
    sigma = 1.4826 * np.median(np.abs(tr - med)) or (tr.std() or 1.0)
    thr = med + K * sigma
    pk, _ = find_peaks(tr, height=thr, distance=MIN_DIST)
    return pk


def xcorr_lag(a, b, maxlag=MAXLAG):
    a = (a - a.mean()) / (a.std() or 1); b = (b - b.mean()) / (b.std() or 1)
    n = len(a); best_lag, best = 0, -np.inf
    for lag in range(-maxlag, maxlag + 1):
        if lag < 0:
            c = np.corrcoef(a[:n + lag], b[-lag:])[0, 1]
        elif lag > 0:
            c = np.corrcoef(a[lag:], b[:n - lag])[0, 1]
        else:
            c = np.corrcoef(a, b)[0, 1]
        if np.isfinite(c) and c > best:
            best, best_lag = c, lag
    # lag>0 => b delayed relative to a => a (soma) leads b (branch) [backprop]
    return best_lag, best


def find_runs(roots):
    runs = []
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "soma.csv"), recursive=True):
            runs.append(os.path.dirname(p))
    return sorted(set(runs))


def main():
    roots = sys.argv[1:] or ["rbp4_141_phpeb", "rbp4ach"]
    runs = find_runs(roots)
    print(f"found {len(runs)} runs\n")

    pooled = {"r_soma_branch": [], "r_soma_trunk": [], "lag_soma_branch": [],
              "branch_events": 0, "branch_indep": 0, "soma_events": 0, "soma_indep": 0}

    for d in runs:
        soma = dff(load(os.path.join(d, "soma.csv")))
        trunk_p = os.path.join(d, "trunk.csv")
        trunk = dff(load(trunk_p)) if os.path.exists(trunk_p) else None
        branch_files = sorted(f for f in glob.glob(os.path.join(d, "*.csv"))
                              if "branch" in os.path.basename(f).lower())
        rel = os.path.relpath(d)
        print(f"=== {rel} ===")
        se = events(soma)
        pooled["soma_events"] += len(se)
        if trunk is not None:
            r = np.corrcoef(soma, trunk)[0, 1]
            lag, c = xcorr_lag(soma, trunk)
            pooled["r_soma_trunk"].append(r)
            print(f"  soma-trunk : r={r:+.2f}  bestlag={lag:+d}f (r={c:.2f})")
        for bf in branch_files:
            b = dff(load(bf))
            name = os.path.splitext(os.path.basename(bf))[0]
            r = np.corrcoef(soma, b)[0, 1]
            lag, c = xcorr_lag(soma, b)
            be = events(b)
            # branch events with no soma event within +/-W
            indep = sum(1 for p in be if not np.any(np.abs(se - p) <= W))
            somaindep = sum(1 for p in se if not np.any(np.abs(be - p) <= W))
            pooled["r_soma_branch"].append(r)
            pooled["lag_soma_branch"].append(lag)
            pooled["branch_events"] += len(be); pooled["branch_indep"] += indep
            pct = 100 * indep / len(be) if len(be) else 0
            lead = ("soma leads" if lag > 0 else "branch leads" if lag < 0 else "simultaneous")
            print(f"  soma-{name:<12}: r={r:+.2f}  lag={lag:+d}f ({lead})  "
                  f"events={len(be):3d}  soma-independent={indep:3d} ({pct:.0f}%)")
        print()

    print("===== POOLED =====")
    def m(x): return f"{np.mean(x):+.2f} ± {np.std(x):.2f}" if x else "n/a"
    print(f"  mean r(soma,branch) = {m(pooled['r_soma_branch'])}")
    print(f"  mean r(soma,trunk)  = {m(pooled['r_soma_trunk'])}")
    print(f"  median lag(soma->branch) = {np.median(pooled['lag_soma_branch']):+.0f} frames "
          f"(>0 = soma leads / backprop)")
    tot, ind = pooled["branch_events"], pooled["branch_indep"]
    print(f"  branch events total = {tot}, soma-independent = {ind} "
          f"({100*ind/tot if tot else 0:.0f}%)")


if __name__ == "__main__":
    main()
