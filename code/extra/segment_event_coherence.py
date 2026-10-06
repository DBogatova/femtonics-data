#!/usr/bin/env python3
"""
segment_event_coherence.py - How coherent are events across dendrite segments?

Orders segments by position along X (leftmost = soma end, rightmost = branch end),
then quantifies, per run:
  * pairwise ΔF/F correlation between segments (coherence matrix)
  * event co-participation: cluster per-segment events in time into "network
    events"; report how many segments participate (global vs local/isolated)
  * lead-lag: within each multi-segment network event, which segment peaks first
    -> is the soma end or the branch end driving the spike?
  * soma-end <-> branch-end cross-correlation lag (who leads on average)

Outputs a figure (PNG + PDF), a per-network-event CSV, and a printed summary.

Usage
-----
  python code/extra/segment_event_coherence.py run_clean.tif segments_labelmap.tif \
      --voxel 0.8 0.9 0.9 --window 3 [--exclude 3] [--frame-ms 184]
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def dff(F, f0_pct):
    f0 = np.percentile(F, f0_pct)
    return (F - f0) / (f0 if f0 else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack")
    ap.add_argument("labelmap")
    ap.add_argument("--voxel", nargs=3, type=float, default=[0.8, 0.9, 0.9], metavar=("Z", "Y", "X"))
    ap.add_argument("--f0-pct", type=float, default=10.0)
    ap.add_argument("--prom-frac", type=float, default=0.2, help="event prominence as frac of each seg's range")
    ap.add_argument("--min-dist", type=int, default=5, help="min frames between a segment's own events")
    ap.add_argument("--window", type=int, default=3, help="frames within which events across segments = one network event")
    ap.add_argument("--exclude", nargs="*", type=int, default=[], metavar="LABEL")
    ap.add_argument("--order", nargs="*", type=int, default=None, metavar="LABEL",
                    help="explicit soma->branch segment order (overrides X-sort)")
    ap.add_argument("--names", nargs="*", default=None, metavar="NAME",
                    help="display names in soma->branch order (e.g. soma trunk branch1 branch2 branch2-far)")
    ap.add_argument("--proj-axis", choices=["z", "y", "x"], default="z",
                    help="projection for the segment MIP panel: z=XY, y=XZ, x=ZY")
    ap.add_argument("--frame-ms", type=float, default=None, help="frame period (ms) to report lags in real time")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--mask", dest="mask", action="store_true", default=False,
                    help="black background outside the cell and other cells blacked out "
                         "(display only; default: the original, unmasked recording)")
    ap.add_argument("--no-mask", dest="mask", action="store_false", help="original look (default)")
    ap.add_argument("--hide-other", dest="hide_other", action="store_true", default=True,
                    help="replace voxels of cells marked 'other cell' with nearby background "
                         "flicker (default; display only)")
    ap.add_argument("--show-other", dest="hide_other", action="store_false",
                    help="show other cells as recorded")
    ap.add_argument("--edge-um", type=float, default=2.0, help="soft edge of the display mask (um)")
    ap.add_argument("--scale", choices=["auto", "common", "zscore"], default="auto",
                    help="trace panel amplitude scale. auto (default, historical look): spacing from this "
                         "run's own range. common: one cohort-wide dF/F per inch from --scale-file, so "
                         "amplitudes compare across figures. zscore: traces / their noise SD (SNR), "
                         "cohort-wide SD per inch. Traces, events and statistics never change.")
    ap.add_argument("--scale-file", default=None,
                    help="plot_scale.json (default: <project>/stats/plot_scale.json)")
    args = ap.parse_args()
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    from common.plot_scale import load_scale, robust_noise, default_scale_file
    scale_cfg = None
    if args.scale != "auto":
        sf = args.scale_file or default_scale_file()
        scale_cfg = load_scale(sf)
        if scale_cfg is None:
            print(f"WARNING: --scale {args.scale} needs {sf} (build it with "
                  f"code/STEP8_stats/normalized_plots.py --write-scale); using --scale auto")
            args.scale = "auto"

    stack = tifffile.imread(args.stack)
    assert stack.ndim == 4, f"expected 4D, got {stack.shape}"
    T = stack.shape[0]
    lm = tifffile.imread(args.labelmap)
    assert lm.shape == stack.shape[1:], f"mask {lm.shape} != vol {stack.shape[1:]}"
    if args.exclude:
        lm = lm.copy(); lm[np.isin(lm, args.exclude)] = 0

    labels = [int(v) for v in np.unique(lm) if v > 0]
    # order by mean X (leftmost = soma end, rightmost = branch end), or explicit --order
    xmean = {l: np.argwhere(lm == l)[:, 2].mean() for l in labels}
    if args.order:
        order = [l for l in args.order if l in labels]
        order += [l for l in sorted(labels, key=lambda l: xmean[l]) if l not in order]
    else:
        order = sorted(labels, key=lambda l: xmean[l])
    n = len(order)
    names = args.names if (args.names and len(args.names) == n) else [f"seg{l}" for l in order]
    role = ["soma-end" if i == 0 else "branch-end" if i == n - 1 else "mid" for i in range(n)]

    # per-segment ΔF/F (ordered soma->branch)
    traces = np.zeros((n, T), np.float32)
    for i, l in enumerate(order):
        traces[i] = dff(stack[:, lm == l].mean(1).astype(np.float32), args.f0_pct)

    # correlation matrix
    C = np.corrcoef(traces)
    mean_pair = (C.sum() - n) / (n * (n - 1))
    soma_branch_r = C[0, -1]

    # per-segment events
    seg_events = []          # list of (seg_index, frame, amp)
    per_seg_peaks = []
    for i in range(n):
        tr = traces[i]; rng = tr.max() - tr.min()
        pk, _ = find_peaks(tr, prominence=args.prom_frac * rng, distance=args.min_dist)
        per_seg_peaks.append(pk)
        for p in pk:
            seg_events.append((i, int(p), float(tr[p])))

    # cluster events across segments into network events (single-linkage in time)
    seg_events.sort(key=lambda e: e[1])
    net = []                 # each: dict frame->list of (seg_index, frame)
    cur = []
    last = None
    for si, fr, amp in seg_events:
        if last is None or fr - last <= args.window:
            cur.append((si, fr))
        else:
            net.append(cur); cur = [(si, fr)]
        last = fr
    if cur:
        net.append(cur)

    # summarize each network event
    rows = []
    participation = []
    lead_pos = []            # anatomical index of the leading segment (multi-seg events)
    isolated_by_seg = np.zeros(n, int)
    for ev in net:
        # earliest peak per participating segment
        first = {}
        for si, fr in ev:
            if si not in first or fr < first[si]:
                first[si] = fr
        parts = sorted(first)                       # participating seg indices
        participation.append(len(parts))
        lead = min(parts, key=lambda s: first[s])
        onset = min(first.values())
        rows.append((onset, len(parts), order[lead], role[lead],
                     "|".join(names[s] for s in parts)))
        if len(parts) == 1:
            isolated_by_seg[parts[0]] += 1
        else:
            lead_pos.append(lead)

    participation = np.array(participation)
    n_events = len(net)
    n_global = int((participation >= max(2, int(np.ceil(0.8 * n)))).sum())
    n_iso = int((participation == 1).sum())
    n_partial = n_events - n_global - n_iso
    lead_pos = np.array(lead_pos)
    soma_leads = int((lead_pos == 0).sum())
    branch_leads = int((lead_pos == n - 1).sum())
    mid_leads = int(len(lead_pos) - soma_leads - branch_leads)

    # soma-end vs branch-end onset delay over events where BOTH ends fire
    # (delta = branch_peak - soma_peak ; >0 => soma end precedes => soma leads)
    deltas = []
    for ev in net:
        first = {}
        for si, fr in ev:
            if si not in first or fr < first[si]:
                first[si] = fr
        if 0 in first and (n - 1) in first:
            deltas.append(first[n - 1] - first[0])
    deltas = np.array(deltas, float)
    n_both = len(deltas)
    soma_first = int((deltas > 0).sum())
    branch_first = int((deltas < 0).sum())
    simul = int((deltas == 0).sum())
    mean_delta = float(deltas.mean()) if n_both else 0.0
    fm = args.frame_ms or 0.0
    delay_txt = (f"soma→branch onset delay (n={n_both}): "
                 f"mean {mean_delta:+.2f} f" + (f" ({mean_delta*fm:+.0f} ms)" if fm else "") +
                 f"  | soma-first {soma_first}, branch-first {branch_first}, tie {simul}")

    # soma-end <-> branch-end cross-correlation lag. With xcorr(lag)=Σ a[t+lag]·b[t] and
    # a=soma, b=branch, the peak lag L means soma[t+L] matches branch[t]: soma trails branch
    # by L, i.e. positive lag => soma-end LAGS branch-end => branch-end leads (and vice versa).
    a = traces[0] - traces[0].mean(); b = traces[-1] - traces[-1].mean()
    maxlag = max(1, min(30, T - 1))                      # short recordings: keep lags < T
    xcorr = np.array([np.dot(a[lag:], b[:T - lag]) if lag >= 0 else np.dot(a[:T + lag], b[-lag:])
                      for lag in range(-maxlag, maxlag + 1)])
    xcorr /= (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    best_lag = np.arange(-maxlag, maxlag + 1)[np.argmax(xcorr)]
    lead_txt = ("branch-end leads" if best_lag > 0 else "soma-end leads" if best_lag < 0 else "simultaneous")
    ms = f" ({abs(best_lag)*args.frame_ms:.0f} ms)" if args.frame_ms else ""

    tag = args.out_prefix or str(Path(args.stack).with_suffix("")) + "_coherence"

    # ---- console summary ----
    print(f"\n=== {Path(args.stack).parent.name}/{Path(args.stack).name}  ({n} segments, soma->branch) ===")
    print("order (soma->branch):", ", ".join(f"{names[i]}[{role[i]}]" for i in range(n)))
    print(f"mean pairwise ΔF/F correlation: {mean_pair:.2f}   soma-end vs branch-end r = {soma_branch_r:.2f}")
    print(f"network events: {n_events}  |  global(>=80% segs): {n_global}   partial: {n_partial}   isolated(1 seg): {n_iso}")
    print(f"isolated events per segment: " +
          ", ".join(f"{names[i]}={isolated_by_seg[i]}" for i in range(n)))
    print(f"multi-seg event leader: soma-end={soma_leads}  mid={mid_leads}  branch-end={branch_leads}")
    print(f"cross-corr soma<->branch: best lag = {best_lag} frames -> {lead_txt}{ms}")
    print(delay_txt)

    # ---- per-event CSV ----
    with open(f"{tag}_network_events.csv", "w") as f:
        f.write("onset_frame,n_participants,lead_seg,lead_role,participants\n")
        for r in sorted(rows):
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")

    # ---- figure ----
    # auto: the historical 12 x 8 in figure. common / zscore: the trace panel gets a fixed
    # height per region (inch_per_lane), so one lane is the same dF/F (or SD) per inch on
    # every run; the top row keeps its auto size (8 in x 1/2.2).
    if args.scale == "auto":
        fig_h, hr = 8.0, [1, 1.2]
    else:
        top_in = 8.0 / 2.2
        lane_in = float(scale_cfg["inch_per_lane"])
        trace_in = (n + 1.0) * lane_in + 0.9      # room for the top trace's peaks
        fig_h, hr = top_in + trace_in, [top_in, trace_in]
    fig = plt.figure(figsize=(12, fig_h))
    gs = fig.add_gridspec(2, 2, height_ratios=hr, width_ratios=[1, 1.1])
    cmap = plt.get_cmap("turbo")
    col = [cmap((i + 0.5) / n) for i in range(n)]

    # A: correlation heatmap
    axC = fig.add_subplot(gs[0, 0])
    # pcolormesh (not imshow) so the matrix is vector cells in the PDF
    im = axC.pcolormesh(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5, C,
                        cmap="magma", vmin=-0.2, vmax=1, shading="flat", edgecolors="none")
    axC.set_xlim(-0.5, n - 0.5); axC.set_ylim(n - 0.5, -0.5); axC.set_aspect("equal")
    axC.set_xticks(range(n)); axC.set_yticks(range(n))
    axC.set_xticklabels(names, rotation=90, fontsize=7); axC.set_yticklabels(names, fontsize=7)
    axC.set_title(f"ΔF/F correlation (soma→branch)\nmean pair r={mean_pair:.2f}", fontsize=9)
    for i in range(n):
        for j in range(n):
            axC.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                     fontsize=6, color="w" if C[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=axC, fraction=0.046)

    # B: two stacked MIPs — plain anatomy (top) and with segments (bottom)
    pa = {"z": 0, "y": 1, "x": 2}[args.proj_axis]
    view = {"z": "XY", "y": "XZ", "x": "ZY"}[args.proj_axis]
    vz, vy, vx = args.voxel
    aspect_map = {0: vy / vx, 1: vz / vx, 2: vz / vy}[pa]
    vol = stack.mean(0).astype(np.float32)             # a COPY: traces above used the raw stack
    if args.hide_other:
        import sys as _sys, pathlib as _pl
        _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
        from common.display_mask import load_other_cell_fill, fill_other_cells
        fill = load_other_cell_fill(args.stack, args.labelmap)
        if fill is not None:
            fill_other_cells(vol, *fill); print(f"cell picture: other cells hidden ({len(fill[0])} voxels)")
    wgt = None
    if args.mask:
        import sys as _sys, pathlib as _pl
        _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
        from common.display_mask import load_display_weight
        try:
            wgt, desc = load_display_weight(args.stack, args.labelmap, voxel=args.voxel, edge_um=args.edge_um)
        except SystemExit as e:
            wgt, desc = None, f"skipped ({e})"
        if wgt is not None and wgt.shape == vol.shape:
            print(f"cell picture: {desc}"); vol = vol * wgt
        else:
            wgt = None
    cell = vol.max(axis=pa)
    if wgt is not None:
        inside = (wgt > 0.5).max(axis=pa)
        clo, chi = 0.0, float(np.percentile(cell[inside], 99.5))
    else:
        clo, chi = np.percentile(cell, (2, 99.5))
    g = np.clip((cell - clo) / (chi - clo + 1e-6), 0, 1)
    ov = np.zeros((*cell.shape, 4))
    for i, l in enumerate(order):
        ov[(lm == l).max(axis=pa)] = (*col[i][:3], 0.8)

    gsB = gs[0, 1].subgridspec(2, 1, hspace=0.3)
    axM0 = fig.add_subplot(gsB[0])
    axM0.imshow(g, cmap="gray", aspect=aspect_map, interpolation="lanczos")
    axM0.set_title(f"cell ({view} MIP)", fontsize=9); axM0.axis("off")
    axM1 = fig.add_subplot(gsB[1])
    axM1.imshow(g, cmap="gray", aspect=aspect_map, interpolation="lanczos")
    axM1.imshow(ov, aspect=aspect_map, interpolation="nearest")
    for i, l in enumerate(order):
        ys, xs = np.where((lm == l).max(axis=pa))
        if len(xs):
            axM1.text(xs.mean(), ys.mean(), names[i], color="w", fontsize=6.5, ha="center", va="center",
                      bbox=dict(fc="black", ec="none", alpha=0.4, pad=0.5))
    axM1.set_title("+ segments", fontsize=9); axM1.axis("off")

    # C: event raster with traces (soma bottom -> branch top)
    axR = fig.add_subplot(gs[1, :])
    # what is drawn: dF/F (auto, common) or dF/F / noise SD (zscore); events, correlations
    # and the CSV above are always computed on the dF/F traces
    if args.scale == "zscore":
        noise_sd = np.array([robust_noise(t) for t in traces])
        shown = traces / noise_sd[:, None]
        off = float(scale_cfg["z_per_lane"])
        SB, sb_unit = float(scale_cfg["z_scale_bar"]), "SD"
        scale_note = (f"z-scored: each trace / its noise SD (signal-to-noise); "
                      f"lane = {off:g} SD, same on every run")
    elif args.scale == "common":
        shown = traces
        off = float(scale_cfg["dff_per_lane"])
        SB, sb_unit = float(scale_cfg["dff_scale_bar"]), "ΔF/F"
        scale_note = f"common amplitude scale: lane = {off:g} ΔF/F, same on every run"
    else:
        shown = traces
        off = 1.15 * max(float(t.max() - t.min()) for t in traces)
        SB, sb_unit = 0.5, "ΔF/F"     # fixed-size bar (same 0.5 dF/F on every run)
        scale_note = None
    # network-event guide lines colored by leader position
    for r in rows:
        onset, npart, leadseg, leadrole, _ = r
        c = "red" if leadrole == "soma-end" else "blue" if leadrole == "branch-end" else "0.8"
        if npart >= 2:
            axR.axvline(onset, color=c, lw=0.5, alpha=0.35)
    for i in range(n):
        axR.plot(shown[i] + i * off, color=col[i], lw=0.6)
        pk = per_seg_peaks[i]
        axR.plot(pk, shown[i][pk] + i * off, ".", color="k", ms=4)
        ylab = i * off + (shown[i].mean() if args.scale == "auto" else 0.25 * off)
        axR.text(-0.01 * T, ylab, f"{names[i]}\n{role[i]}",
                 ha="right", va="center", fontsize=7, color=col[i])
    axR.set_xlabel("frame", fontsize=11); axR.set_yticks([])
    axR.tick_params(axis="x", labelsize=11)            # same as the behavior panel
    xb = T - 1 + 0.012 * T
    axR.plot([xb, xb], [0, SB], color="k", lw=1.6, clip_on=False, solid_capstyle="butt")
    axR.text(xb + 0.006 * T, SB / 2, f"{SB:g} {sb_unit}", rotation=90, ha="left", va="center",
             fontsize=8, clip_on=False)
    ttl = ("per-segment ΔF/F (soma bottom → branch top); dots=events; "
           "vlines=multi-seg events (red=soma-led, blue=branch-led)")
    axR.set_title(ttl if scale_note is None else ttl + "\n" + scale_note, fontsize=9)
    plt.tight_layout()
    # Pin the trace axis to a fixed horizontal span of the figure and to the exact frame
    # range, so the behavior panel (same span, same range, same figure width) lines up
    # frame-for-frame when the two are stacked. Keep in sync with coherence_behavior.py.
    TRACE_LEFT, TRACE_RIGHT = 0.10, 0.955
    pos = axR.get_position()
    axR.set_position([TRACE_LEFT, pos.y0, TRACE_RIGHT - TRACE_LEFT, pos.height])
    axR.set_xlim(0, T - 1)
    if args.scale != "auto":
        # exact units per inch: the y range is set from the axis' physical height
        h_in = pos.height * fig_h
        per_in = off / float(scale_cfg["inch_per_lane"])
        y0 = -0.35 * off
        axR.set_ylim(y0, y0 + h_in * per_in)
        print(f"trace panel: --scale {args.scale}, lane {off:g} {sb_unit} = "
              f"{scale_cfg['inch_per_lane']:g} in ({per_in:.3g} {sb_unit}/in)")
    plt.savefig(f"{tag}.png", dpi=180); plt.savefig(f"{tag}.pdf")
    print(f"saved: {tag}.png / .pdf  and  {tag}_network_events.csv\n")


if __name__ == "__main__":
    main()
