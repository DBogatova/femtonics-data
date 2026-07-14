import csv
import matplotlib.pyplot as plt

traces = ["soma", "trunk", "branch", "dendrites"]
offset = 0.3

plt.figure(figsize=(10, 6))
for i, name in enumerate(traces):
    slices, means = [], []
    with open(f"preprocessed/{name}-trace.csv") as f:
        for row in csv.DictReader(f):
            slices.append(int(row["Slice"]))
            means.append(float(row["Mean"]) + i * offset)
    plt.plot(slices, means, label=name)

plt.xlabel("Slice")
plt.ylabel("Mean (stacked, offset=0.3)")
plt.legend()
plt.tight_layout()
plt.savefig("preprocessed/traces_plot.png", dpi=150)
