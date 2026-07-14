#!/usr/bin/env python3
"""
plot_traces_tool.py - Stacked trace plots with peak (frame-number) annotations.

Reads a folder of CSV files (each with columns Slice,Mean - or the first two
columns) and plots each as its own subplot stacked vertically, sharing the
x-axis. Optionally detects prominent peaks and labels them with their frame
number.

EXAMPLES
--------
# Plot every CSV in a folder, auto title from folder name:
python plot_traces_tool.py preprocessed/run12

# Custom title / labels, explicit top-to-bottom order:
python plot_traces_tool.py preprocessed/run12 \
    --title "rbp4_132 05-20-2026 run12" --xlabel Frames --ylabel dff \
    --order dendrites2 dendrites branch-2 branch-1 trunk soma

# Tune peak sensitivity globally and per-trace (lower = more peaks):
python plot_traces_tool.py preprocessed/run12 \
    --prom-frac 0.25 --min-dist 10 --prom-override branch-2=0.08

# Turn peak labels off:
python plot_traces_tool.py preprocessed/run12 --no-peaks

NOTES
-----
* Run with the project venv:  .venv/bin/python plot_traces_tool.py ...
* A common filename prefix shared by all files (e.g. "run12-") and a trailing
  "-trace" are stripped to make subplot labels. Use --order to control the
  top-to-bottom sequence (also acts as a filter: only listed traces are shown).
"""
import argparse
import csv
import os
import re
import glob as globmod

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Color palette cycled across traces (distinct, colorblind-friendly-ish).
PALETTE = [
    "tab:red", "tab:purple", "magenta", "tab:green",
    "tab:orange", "tab:blue", "tab:brown", "tab:cyan",
    "tab:olive", "tab:pink", "tab:gray",
]

# Anatomical ordering hint (distal/superficial -> proximal), most superficial on top.
BRANCH_COLORS = ["tab:red", "tab:green", "magenta", "tab:purple", "tab:brown", "tab:cyan"]


def read_trace(path):
    """Return (frames, values) from a CSV with Slice,Mean or first two columns."""
    frames, values = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # locate columns by name, else fall back to first two
        try:
            xi = header.index("Slice")
            yi = header.index("Mean")
        except ValueError:
            xi, yi = 0, 1
        for row in reader:
            if len(row) <= max(xi, yi) or not row[xi].strip():
                continue
            frames.append(float(row[xi]))
            values.append(float(row[yi]))
    return frames, values


def to_dff(values, f0_pct):
    """Convert raw fluorescence to dF/F using F0 = given percentile of the trace."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    f0 = np.percentile(v, f0_pct)
    if f0 == 0:
        f0 = np.nanmean(v) or 1.0
    return ((v - f0) / f0).tolist()


def common_prefix(names):
    """Longest shared leading substring across names (used to clean labels)."""
    if not names:
        return ""
    pre = os.path.commonprefix(names)
    return pre


def label_from_filename(fname, strip_prefix):
    base = os.path.splitext(os.path.basename(fname))[0]
    if strip_prefix and base.startswith(strip_prefix):
        base = base[len(strip_prefix):]
    if base.endswith("-trace"):
        base = base[: -len("-trace")]
    return base


def anatomy_key(label):
    # top -> bottom: dendrites(0), branches(1), nexus/bifurcation(2), trunk(3), soma(4).
    # within branches: higher number on top, so branch1 sits lowest (just above the
    # junction/trunk); a "-far" variant is more distal so it sits ABOVE its base number.
    low = label.lower()
    if "dend" in low:
        g = 0
    elif "branch" in low:
        g = 1
    elif "nexus" in low or "bifurc" in low:
        g = 2
    elif "trunk" in low:
        g = 3
    elif "soma" in low:
        g = 4
    else:
        g = 2
    if g == 1:
        m = re.search(r"(\d+)", low)
        num = int(m.group(1)) if m else 0
        far = 1 if "far" in low else 0
        return (g, -num, -far, low)   # higher number on top; -far sits above its base
    return (g, 0, 0, low)


def role_color(label):
    """Consistent color per anatomical role, independent of ordering."""
    low = label.lower()
    if "soma" in low:
        return "tab:blue"
    if "trunk" in low:
        return "tab:orange"
    if "nexus" in low or "bifurc" in low:
        return "gray"
    if "dend" in low:
        return "tab:red"
    if "branch" in low:
        m = re.search(r"(\d+)", low)
        n = int(m.group(1)) if m else 1
        return BRANCH_COLORS[(n - 1) % len(BRANCH_COLORS)]
    return "black"


def parse_overrides(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--prom-override expects label=value, got: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Stacked trace plots with peak frame-number annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("folder", help="folder containing trace CSV files")
    ap.add_argument("--pattern", default="*.csv", help="glob for files (default *.csv)")
    ap.add_argument("--title", default=None, help="figure title (default: folder name)")
    ap.add_argument("--no-title", action="store_true", help="omit the figure title")
    ap.add_argument("--xlabel", default="Frames")
    ap.add_argument("--ylabel", default="\u0394F/F")
    ap.add_argument("--font", default=None,
                    help="font family for all text, e.g. Arial")
    ap.add_argument("--dff", action="store_true",
                    help="convert raw fluorescence to dF/F (F0 = --f0-pct percentile)")
    ap.add_argument("--f0-pct", type=float, default=10.0,
                    help="percentile used as F0 baseline for --dff (default 10)")
    ap.add_argument("--order", nargs="+", default=None,
                    help="explicit top-to-bottom labels; also filters to these")
    ap.add_argument("--prom-frac", type=float, default=0.25,
                    help="peak prominence as fraction of each trace's range")
    ap.add_argument("--min-dist", type=int, default=10,
                    help="min spacing (frames) between labeled peaks")
    ap.add_argument("--prom-override", action="append", default=None,
                    metavar="LABEL=FRAC", help="per-trace prominence fraction")
    ap.add_argument("--no-peaks", action="store_true", help="disable peak labels")
    ap.add_argument("--overlay", action="store_true",
                    help="plot all traces on one shared axes (with legend)")
    ap.add_argument("--offset", type=float, default=None,
                    help="vertical offset between overlaid traces "
                         "(0 = true overlap; omitted = auto spacing)")
    ap.add_argument("--colors", nargs="+", default=None,
                    help="explicit colors per trace, top-to-bottom (overrides palette)")
    ap.add_argument("--row-height", type=float, default=2.3,
                    help="height in inches per subplot (lower = more compact)")
    ap.add_argument("--width", type=float, default=12.0, help="figure width in inches")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    if args.font:
        plt.rcParams["font.family"] = args.font

    files = sorted(globmod.glob(os.path.join(args.folder, args.pattern)))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {args.folder}")

    prefix = common_prefix([os.path.basename(f) for f in files])
    items = [(label_from_filename(f, prefix), f) for f in files]
    label_to_file = dict(items)

    # ordering
    if args.order:
        missing = [l for l in args.order if l not in label_to_file]
        if missing:
            raise SystemExit(f"--order labels not found: {missing}\n"
                             f"available: {sorted(label_to_file)}")
        ordered = [(l, label_to_file[l]) for l in args.order]
    else:
        ordered = sorted(items, key=lambda it: anatomy_key(it[0]))

    overrides = parse_overrides(args.prom_override)
    title = args.title if args.title is not None else os.path.basename(os.path.normpath(args.folder))

    n = len(ordered)
    if args.colors:
        palette = (args.colors * (n // len(args.colors) + 1))
    else:
        palette = [role_color(lbl) for (lbl, _f) in ordered]

    if args.overlay:
        # load all selected traces first (needed to auto-compute offset)
        loaded = [(label, *(read_trace(fname))) for (label, fname) in ordered]
        if args.dff:
            loaded = [(lab, fr, to_dff(v, args.f0_pct)) for (lab, fr, v) in loaded]

        if args.offset is None:
            # auto: largest single-trace span + 20% gap -> guaranteed separation
            spans = [max(v) - min(v) for _, _, v in loaded if v]
            step = 1.2 * max(spans) if spans else 0.0
        else:
            step = args.offset

        fig, ax = plt.subplots(figsize=(args.width, max(args.row_height * 2.2, 3.0)))
        # first in --order goes on top: give it the largest offset
        for i, ((label, frames, values), color) in enumerate(zip(loaded, palette)):
            shift = (n - 1 - i) * step
            ax.plot(frames, [v + shift for v in values],
                    color=color, linewidth=0.8, label=label)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)   # spacing still equals true dF/F despite offset
        ax.legend(frameon=False, loc="upper right")
        if not args.no_title:
            ax.set_title(title)
        plt.tight_layout()
        out = args.out or os.path.join(args.folder, f"{title.replace(' ', '_')}_overlay.png")
        plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"saved {out}")
        return

    fig, axes = plt.subplots(n, 1, figsize=(args.width, max(args.row_height * n, 2.5)),
                             sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (label, fname), color in zip(axes, ordered, palette):
        frames, values = read_trace(fname)
        if args.dff:
            values = to_dff(values, args.f0_pct)
        ax.plot(frames, values, color=color, linewidth=0.8)

        if not args.no_peaks and values:
            rng = max(values) - min(values)
            frac = overrides.get(label, args.prom_frac)
            if rng > 0:
                peaks, _ = find_peaks(values, prominence=frac * rng, distance=args.min_dist)
                for p in peaks:
                    ax.annotate(str(int(frames[p])), (frames[p], values[p]),
                                textcoords="offset points", xytext=(0, 4),
                                ha="center", va="bottom", fontsize=6,
                                color="black", rotation=90)
                ax.scatter([frames[p] for p in peaks], [values[p] for p in peaks],
                           s=10, color="black", zorder=5)
            ax.margins(y=0.18)

        ax.set_title(label)
        ax.set_ylabel(args.ylabel)

    axes[-1].set_xlabel(args.xlabel)
    if not args.no_title:
        fig.suptitle(title, y=1.0)
    plt.tight_layout()

    out = args.out or os.path.join(args.folder, f"{title.replace(' ', '_')}_traces.png")
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
