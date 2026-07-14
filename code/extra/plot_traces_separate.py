import csv
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("prefix", nargs="?", default="",
                    help='dataset prefix, e.g. "dend2" -> dend2-soma-trace.csv')
parser.add_argument("--xlabel", default="Slice")
parser.add_argument("--title", default=None)
args = parser.parse_args()

tag = f"{args.prefix}-" if args.prefix else ""
title = args.title if args.title is not None else (args.prefix or "trace")

traces = ["dendrites", "branch", "trunk", "soma"]
colors = ["tab:red", "tab:green", "tab:orange", "tab:blue"]

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for ax, name, color in zip(axes, traces, colors):
    slices, means = [], []
    with open(f"preprocessed/{tag}{name}-trace.csv") as f:
        for row in csv.DictReader(f):
            slices.append(int(row["Slice"]))
            means.append(float(row["Mean"]))
    ax.plot(slices, means, color=color)
    ax.set_title(name)
    ax.set_ylabel("Mean")

axes[-1].set_xlabel(args.xlabel)
fig.suptitle(title, y=1.0)
plt.tight_layout()
out = f"preprocessed/{tag}traces_separate.png"
plt.savefig(out, dpi=150)
print(f"saved {out}")
