#!/usr/bin/env python3
"""
segment_branch_propagation.py - For every branch-initiated event, how far toward
the soma does it propagate? Classifies each branch event as:
    branch-local   : only distal segment(s) respond
    reached-trunk  : spreads into the proximal half but NOT the soma
    reached-soma   : the soma-end segment responds

A segment "responds" if, in a window around the branch peak, its ΔF/F excursion
(peak minus local pre-baseline) exceeds an EMPIRICAL per-segment threshold set by
that segment's own null distribution (sliding random windows) at false-positive
rate p. This avoids calling slow baseline wobble a "response".

Outputs a figure (propagation profile + per-event reach raster), a CSV, and a
printed summary. Segments are ordered soma->branch by mean X.

Usage
-----
  python code/extra/segment_branch_propagation.py run_clean.tif segments_labelmap.tif \
      --voxel 0.8 0.9 0.9 --frame-ms 167.8 --p 0.05
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def dff(F, pct=10.0):
    f0 = np.percentile(F, pct)
    return (F - f0) / (f0 if f0 else 1.0)


def event_stat(tr, f, pre=(2, 16), post=8):
    """peak in [f-2, f+post] minus median of [f-16, f-2]."""
    a0, a1 = max(0, f - pre[0]), min(len(tr), f + post + 1)
    b0, b1 = max(0, f - pre[1]), max(1, f - pre[0])
    return float(tr[a0:a1].max() - np.median(tr[b0:b1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack"); ap.add_argument("labelmap")
    ap.add_argument("--voxel", nargs=3, type=float, default=[0.8, 0.9, 0.9], metavar=("Z", "Y", "X"))
    ap.add_argument("--f0-pct", type=float, default=10.0)
    ap.add_argument("--prom-frac", type=float, default=0.2, help="branch event prominence (frac of range)")
    ap.add_argument("--min-dist", type=int, default=5)
    ap.add_argument("--post", type=int, default=8, help="frames after branch peak to look for a proximal response")
    ap.add_argument("--p", type=float, default=0.05, help="per-segment false-positive rate for 'responded'")
    ap.add_argument("--frame-ms", type=float, default=None)
    ap.add_argument("--exclude", nargs="*", type=int, default=[], metavar="LABEL")
    ap.add_argument("--order", nargs="*", type=int, default=None, metavar="LABEL",
                    help="explicit soma->branch segment order (overrides X-sort; plotted evenly spaced)")
    ap.add_argument("--names", nargs="*", default=None, metavar="NAME",
                    help="display names in soma->branch order (e.g. soma trunk branch1 branch2 branch2-far)")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    stack = tifffile.imread(args.stack); T = stack.shape[0]
    lm = tifffile.imread(args.labelmap)
    if args.exclude:
        lm = lm.copy(); lm[np.isin(lm, args.exclude)] = 0
    labels = [int(v) for v in np.unique(lm) if v > 0]
    xc = {l: np.argwhere(lm == l)[:, 2].mean() for l in labels}
    if args.order:
        order = [l for l in args.order if l in labels]
        order += [l for l in sorted(labels, key=lambda l: xc[l]) if l not in order]
        n = len(order)
        X = np.arange(n, dtype=float)                      # ordinal positions (explicit order)
        Xn = X / (n - 1) if n > 1 else np.zeros(n)
    else:
        order = sorted(labels, key=lambda l: xc[l])        # soma -> branch by X
        n = len(order)
        X = np.array([xc[l] for l in order]); Xn = (X - X.min()) / (X.max() - X.min() + 1e-9)
    tr = np.array([dff(stack[:, lm == l].mean(1).astype(np.float32), args.f0_pct) for l in order])
    branch = n - 1                                          # branch-end index
    disp = args.names if (args.names and len(args.names) == n) else [f"seg{l}" for l in order]

    # per-segment empirical response threshold at false-positive rate p
    rng = np.random.default_rng(0); Nnull = 3000
    gs = rng.integers(20, T - args.post - 2, Nnull)
    thr = np.array([np.quantile([event_stat(tr[i], g, post=args.post) for g in gs], 1 - args.p)
                    for i in range(n)])

    # branch-initiated events = peaks in the branch-end segment
    btr = tr[branch]; rngamp = btr.max() - btr.min()
    bpk, _ = find_peaks(btr, prominence=args.prom_frac * rngamp, distance=args.min_dist)
    bpk = [int(p) for p in bpk if 20 <= p < T - args.post - 2]

    # classify each branch event by how far proximally it responds
    resp_matrix = np.zeros((len(bpk), n), bool)
    classes = []; rows = []
    prox_half = Xn < 0.5                                   # proximal half (toward soma) by X
    for k, f0 in enumerate(bpk):
        responded = np.array([event_stat(tr[i], f0, post=args.post) >= thr[i] for i in range(n)])
        resp_matrix[k] = responded
        soma_resp = responded[0]
        prox_resp = responded[prox_half].any()
        if soma_resp:
            c = "reached-soma"
        elif prox_resp:
            c = "reached-trunk"
        else:
            c = "branch-local"
        classes.append(c)
        most_prox_X = X[responded].min() if responded.any() else np.nan
        rows.append((f0, c, int(soma_resp),
                     "|".join(disp[i] for i in range(n) if responded[i]),
                     round(float(most_prox_X), 1)))

    classes = np.array(classes)
    nB = len(bpk)
    n_soma = int((classes == "reached-soma").sum())
    n_trunk = int((classes == "reached-trunk").sum())
    n_local = int((classes == "branch-local").sum())
    p_resp = resp_matrix.mean(0)                           # P(respond | branch event) per segment

    tag = args.out_prefix or str(Path(args.stack).with_suffix("")) + "_branchprop"
    name = f"{Path(args.stack).parent.name}/{Path(args.stack).name}"
    print(f"\n=== branch-event propagation: {name}  ({n} segments soma->branch) ===")
    print("segment order (soma->branch):", ", ".join(f"{disp[i]}(x{X[i]:.0f})" for i in range(n)))
    print(f"branch-initiated events: {nB}")
    print(f"  branch-local : {n_local}  ({100*n_local/max(1,nB):.0f}%)")
    print(f"  reached-trunk: {n_trunk}  ({100*n_trunk/max(1,nB):.0f}%)")
    print(f"  reached-soma : {n_soma}  ({100*n_soma/max(1,nB):.0f}%)")
    print("P(respond | branch event) per segment (soma->branch): " +
          ", ".join(f"{disp[i]}={p_resp[i]:.2f}" for i in range(n)))

    with open(f"{tag}_events.csv", "w") as f:
        f.write("branch_frame,reach_class,soma_responded,responders,most_proximal_X\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # ---- figure ----
    cls_col = {"branch-local": "#2c7fb8", "reached-trunk": "#f0a202", "reached-soma": "#d7191c"}
    fig = plt.figure(figsize=(11, 4.6))
    gs2 = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])

    axP = fig.add_subplot(gs2[0, 0])
    axP.plot(Xn, p_resp, "-o", color="k", lw=1.5)
    axP.set_xticks(Xn); axP.set_xticklabels(disp, fontsize=7, rotation=30, ha="right")
    axP.set_ylim(0, 1.05); axP.set_xlabel("position  (soma → branch)")
    axP.set_ylabel("P(respond | branch event)")
    axP.set_title(f"propagation profile\n{nB} branch events", fontsize=9)
    axP.axhline(args.p, color="0.6", ls=":", lw=1)
    axP.text(0.0, args.p + 0.02, f"chance p={args.p}", fontsize=7, color="0.5")
    axP.text(0.02, 0.06, "soma", fontsize=8, color="0.3")
    axP.text(0.78, 0.06, "branch", fontsize=8, color="0.3")

    axR = fig.add_subplot(gs2[0, 1])
    ordk = np.argsort([{"branch-local": 0, "reached-trunk": 1, "reached-soma": 2}[c] for c in classes])
    M = resp_matrix[ordk]
    axR.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    for j, k in enumerate(ordk):
        axR.add_patch(plt.Rectangle((-0.7, j - 0.5), 0.5, 1, color=cls_col[classes[k]], lw=0))
    axR.set_xticks(range(n)); axR.set_xticklabels(disp, fontsize=7, rotation=30, ha="right")
    axR.set_xlabel("segment (soma → branch)"); axR.set_ylabel("branch event (sorted by reach)")
    ms = f"  |  frame={args.frame_ms:.0f} ms" if args.frame_ms else ""
    axR.set_title(f"reach per event  (blue=local {n_local}, orange=trunk {n_trunk}, "
                  f"red=soma {n_soma}){ms}", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{tag}.png", dpi=180); plt.savefig(f"{tag}.pdf")
    print(f"saved: {tag}.png / .pdf  and  {tag}_events.csv\n")


if __name__ == "__main__":
    main()
