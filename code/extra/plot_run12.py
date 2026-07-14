import csv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

folder = "preprocessed/run12"

# Peak-detection tuning:
#   prom_frac  -> default peak prominence as a fraction of each trace's range
#   min_dist   -> minimum spacing (in frames) between labeled peaks
#   prom_override -> per-trace prominence fraction (lower = catches smaller peaks)
prom_frac = 0.25
min_dist = 10
prom_override = {"branch-2": 0.08, "branch-1": 0.10}

# top -> bottom: dendrites2 on top, soma at bottom (distal -> proximal)
traces = [
    ("dendrites2 (top-most / most superficial)", "run12-dendrites2.csv", "tab:red"),
    ("dendrites",     "run12-dendrites.csv",     "tab:purple"),
    ("branch-2",      "run12-branch-2.csv",      "magenta"),
    ("branch-1",      "run12-branch-1.csv",      "tab:green"),
    ("trunk",         "run12-trunk.csv",         "tab:orange"),
    ("soma",          "run12-soma.csv",          "tab:blue"),
]

fig, axes = plt.subplots(len(traces), 1, figsize=(12, 14), sharex=True)
for ax, (name, fname, color) in zip(axes, traces):
    slices, means = [], []
    with open(f"{folder}/{fname}") as f:
        for row in csv.DictReader(f):
            slices.append(int(row["Slice"]))
            means.append(float(row["Mean"]))
    ax.plot(slices, means, color=color, linewidth=0.8)

    # detect large peaks and label them with their frame number
    rng = max(means) - min(means)
    frac = prom_override.get(name, prom_frac)
    peaks, _ = find_peaks(means, prominence=frac * rng, distance=min_dist)
    for p in peaks:
        ax.annotate(str(slices[p]), (slices[p], means[p]),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=6, color="black",
                    rotation=90)
    ax.scatter([slices[p] for p in peaks], [means[p] for p in peaks],
               s=10, color="black", zorder=5)

    ax.set_title(name)
    ax.set_ylabel(r"$\Delta$F/F")
    ax.margins(y=0.18)  # headroom so labels are not clipped

axes[-1].set_xlabel("Frames")
fig.suptitle("rbp4_132 05-20-2026 run12", y=1.0)
plt.tight_layout()
out = f"{folder}/run12_traces_separate.png"
plt.savefig(out, dpi=150)
out_pdf = f"{folder}/run12_traces_separate.pdf"
plt.savefig(out_pdf)  # vector
print(f"saved {out}")
print(f"saved {out_pdf}")
