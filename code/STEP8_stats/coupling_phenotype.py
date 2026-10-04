#!/usr/bin/env python
"""coupling_phenotype.py - group segments within a cell and cells across the cohort by
how they fire together. Needs nothing but the recordings and the saved regions: no
covariate (nucleus, time, expression) is used, so cell types found here can later be
tested against those covariates without circularity.

WITHIN A CELL  (per run, written to <stem>_coupling.json + <stem>_coupling.png)
  1. dF/F per region (raw stack, saved regions).
  2. Similarity = Pearson correlation of dF/F; distance = 1 - r.
  3. Average-linkage hierarchical clustering; cut at distance 1 - R_CUT (R_CUT = 0.75,
     i.e. segments that share >= 75 % of their variance pattern are one co-firing group).
  4. n_groups, which regions belong to which group, whether the soma shares a group with
     any branch, the mean within-group vs between-group r (separation), and the
     'event-level' version: for every network event, which groups participated; the
     fraction of events confined to one group ("local events").
  Dendrogram + ordered correlation matrix + group-colored traces.

ACROSS CELLS (stats/phenotype/)
  Profile per cell: r_soma_branch, n_groups, separation, frac_local_events,
  frac_branch_independent, branch_first_frac. Standardized; hierarchical clustering with
  a silhouette-chosen k (2..4) once >= 6 cells exist; before that, the profile table and
  a scatter are written and the clustering is marked as 'not yet'.

  python code/STEP8_stats/coupling_phenotype.py --all
  python code/STEP8_stats/coupling_phenotype.py --run <behavior_base>
"""
from __future__ import annotations
import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
_CODE_ROOT = HERE.parents[1]
ROOT = Path(os.environ["FEMTO_ROOT"]).resolve() if os.environ.get("FEMTO_ROOT") else _CODE_ROOT
sys.path.insert(0, str(_CODE_ROOT / "code")); sys.path.insert(0, str(_CODE_ROOT / "code/STEP7_workflow")); sys.path.insert(0, str(HERE))
from run_metrics import dff, region_names, compartment_of, events       # noqa: E402
from common.regions import apply_ignore, newest_input_mtime              # noqa: E402
from common.run_marks import is_set_aside   # noqa: E402

R_CUT = 0.75
__version__ = "0.2.0"


def analyze_run(run: dict, root: Path, r_cut: float = R_CUT, window: int = 2):
    run_dir = root / run["run_dir"]; stem = run["stem"]
    stack_p, seg_p = run_dir / f"{stem}.tif", run_dir / f"{stem}_segments_final.tif"
    if not (stack_p.exists() and seg_p.exists()):
        return None
    stack = tifffile.imread(stack_p); seg = tifffile.imread(seg_p); T = stack.shape[0]
    labels = sorted(int(v) for v in np.unique(seg) if v > 0)
    names = region_names(seg_p.with_suffix(".json"), labels)       # from the FULL label set
    seg, _ignored, _ign = apply_ignore(seg, names, seg_p)          # <stem>_ignore.json
    labels = [l for l in labels if l not in _ign]
    if len(labels) < 3:
        return {"behavior_base": run["behavior_base"], "note": "fewer than 3 regions", "ignored_regions": _ignored}
    comp = {l: compartment_of(names[l]) for l in labels}
    flat = stack.reshape(T, -1)
    tr = np.array([dff(flat[:, np.flatnonzero((seg == l).ravel())].mean(1).astype(np.float64)) for l in labels])
    C = np.corrcoef(tr); n = len(labels)
    xm = np.array([np.argwhere(seg == l)[:, 2].mean() for l in labels])
    soma_i = next((i for i, l in enumerate(labels) if comp[l] == "soma"), None)
    order = np.argsort(np.abs(xm - xm[soma_i])) if soma_i is not None else np.argsort(xm)
    D = 1 - C; iu = np.triu_indices(n, 1)
    Z = linkage(D[iu], method="average")
    groups = fcluster(Z, t=1 - r_cut, criterion="distance")
    k = int(groups.max())
    within = [C[i, j] for i in range(n) for j in range(i + 1, n) if groups[i] == groups[j]]
    between = [C[i, j] for i in range(n) for j in range(i + 1, n) if groups[i] != groups[j]]
    sep = (np.mean(within) if within else 1.0) - (np.mean(between) if between else np.nan)
    soma_group = int(groups[soma_i]) if soma_i is not None else None
    has_branch = any(comp[l] == "branch" for l in labels)
    # None = not applicable (no soma region in the scan, or no branch region)
    soma_with_branch = (bool(any(groups[i] == groups[soma_i] and comp[labels[i]] == "branch" for i in range(n)))
                        if soma_i is not None and has_branch else None)
    ev = [(i, int(p)) for i in range(n) for p in events(tr[i])]
    ev.sort(key=lambda e: e[1]); nets, cur, last = [], [], None
    for i, f in ev:
        if last is None or f - last <= window:
            cur.append((i, f))
        else:
            nets.append(cur); cur = [(i, f)]
        last = f
    if cur:
        nets.append(cur)
    multi = [set(groups[i] for i, _ in e) for e in nets if len({i for i, _ in e}) >= 2]
    frac_local = float(np.mean([len(g) == 1 for g in multi])) if multi else float("nan")
    frac_all = float(np.mean([len(g) == k for g in multi])) if multi else float("nan")
    # stability: does the grouping depend on the exact cut? Re-cut the same tree from
    # r = 0.60 to 0.90 and compare each partition with the chosen one (adjusted Rand).
    from sklearn.metrics import adjusted_rand_score
    sweep = {}
    for rc in np.round(np.arange(0.60, 0.901, 0.05), 2):
        g2 = fcluster(Z, t=1 - rc, criterion="distance")
        sweep[f"{rc:.2f}"] = {"n_groups": int(g2.max()), "ari_vs_chosen": round(float(adjusted_rand_score(groups, g2)), 3)}
    stable = [k_ for k_, v_ in sweep.items() if v_["ari_vs_chosen"] >= 0.999]
    stable_range = (min(stable), max(stable)) if stable else None
    out = {"behavior_base": run["behavior_base"], "mouse": run["mouse"], "date": run["date"], "version": __version__,
           "r_cut_sweep": sweep, "grouping_stable_from_to": stable_range,
           "r_cut": r_cut, "n_regions": n, "ignored_regions": _ignored, "n_groups": k,
           "groups": {names[labels[i]]: int(groups[i]) for i in range(n)},
           "compartments": {names[labels[i]]: comp[labels[i]] for i in range(n)},
           "mean_within_r": float(np.mean(within)) if within else None,
           "mean_between_r": float(np.mean(between)) if between else None,
           "separation": float(sep) if sep == sep else None,
           "soma_group": soma_group, "soma_shares_group_with_branch": soma_with_branch,
           "n_multi_region_events": len(multi), "frac_events_within_one_group": frac_local,
           "frac_events_all_groups": frac_all,
           "corr_matrix": {"names": [names[l] for l in labels], "r": np.round(C, 3).tolist()}}
    # figure
    fig = plt.figure(figsize=(13, 4.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 2.2])
    a0 = fig.add_subplot(gs[0])
    dn = dendrogram(Z, labels=[names[l] for l in labels], ax=a0, color_threshold=1 - r_cut, orientation="left")
    a0.axvline(1 - r_cut, color="k", ls="--", lw=0.8); a0.set_xlabel("1 - r")
    a0.set_title(f"{k} co-firing group(s) at r >= {r_cut}", fontsize=9, loc="left")
    leaves = dn["leaves"]
    a1 = fig.add_subplot(gs[1]); im = a1.imshow(C[np.ix_(leaves, leaves)], cmap="magma", vmin=-0.2, vmax=1)
    a1.set_xticks(range(n)); a1.set_yticks(range(n))
    a1.set_xticklabels([names[labels[i]] for i in leaves], rotation=90, fontsize=7); a1.set_yticklabels([names[labels[i]] for i in leaves], fontsize=7)
    for i in range(n):
        for j in range(n):
            v = C[leaves[i], leaves[j]]
            a1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w" if v < 0.6 else "k")
    a1.set_title("correlation, clustered order", fontsize=9, loc="left"); fig.colorbar(im, ax=a1, fraction=0.046)
    a2 = fig.add_subplot(gs[2]); cols = plt.cm.Set1.colors
    off = 1.1 * max(float(t.max() - t.min()) for t in tr)
    for row, i in enumerate(order):
        a2.plot(tr[i] + row * off, color=cols[(groups[i] - 1) % len(cols)], lw=0.6)
        a2.text(-0.01 * T, row * off + tr[i].mean(), f"{names[labels[i]]} (g{groups[i]})", ha="right", va="center", fontsize=7)
    a2.set_yticks([]); a2.set_xlabel("frame"); a2.set_title("dF/F colored by co-firing group (soma at bottom)", fontsize=9, loc="left")
    for a in (a0, a1, a2):
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fl = f"{100 * frac_local:.0f}" if frac_local == frac_local else "n/a"
    fig.suptitle(f"{run['behavior_base']}: {k} group(s), separation {sep:+.2f}, "
                 f"soma {'shares a group with a branch' if soma_with_branch else 'apart from branches'}, "
                 f"{fl}% of multi-region events stay within one group", fontsize=9)
    fig.tight_layout(); fig.savefig(run_dir / f"{stem}_coupling.png", dpi=150); fig.savefig(run_dir / f"{stem}_coupling.pdf"); plt.close(fig)
    (run_dir / f"{stem}_coupling.json").write_text(json.dumps(out, indent=2))
    return out


def cohort(root: Path):
    out_dir = root / "stats" / "phenotype"; out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in glob.glob(str(root / "rbp4_*/**/*_coupling.json"), recursive=True):
        if "/old/" in f or f.endswith("_behavior_coupling.json"):   # different file kind, same suffix
            continue
        j = json.load(open(f))
        if "note" in j or is_set_aside(j.get("behavior_base", "")):
            continue
        mf = Path(f).with_name(Path(f).name.replace("_coupling.json", "_metrics.json"))
        m = json.load(open(mf)) if mf.exists() else {}
        rows.append({"behavior_base": j["behavior_base"], "mouse": j["mouse"], "date": j["date"], "n_regions": j["n_regions"],
                     "n_groups": j["n_groups"], "separation": j["separation"],
                     "soma_shares_group_with_branch": j["soma_shares_group_with_branch"],
                     "frac_events_within_one_group": j["frac_events_within_one_group"],
                     "frac_events_all_groups": j["frac_events_all_groups"],
                     "r_soma_branch": m.get("r_soma_branch"), "frac_branch_independent": m.get("frac_branch_independent"),
                     "branch_first_frac": m.get("branch_first_frac"),
                     "r_soma_branch_quiet": m.get("r_soma_branch_quiet"), "r_soma_branch_active": m.get("r_soma_branch_active"),
                     "reference": m.get("reference", "soma"),
                     "grouping_stable_from": (j.get("grouping_stable_from_to") or [None, None])[0],
                     "grouping_stable_to": (j.get("grouping_stable_from_to") or [None, None])[1],
                     "n_groups_by_cut": " ".join(f"{k}:{v['n_groups']}" for k, v in (j.get("r_cut_sweep") or {}).items())})
    d = pd.DataFrame(rows)
    if d.empty:
        print("no coupling results yet"); return
    d.to_csv(out_dir / "cell_profiles.csv", index=False)
    feats = ["r_soma_branch", "n_groups", "separation", "frac_events_within_one_group", "frac_branch_independent", "branch_first_frac"]
    X = d[feats].astype(float)
    msg = [f"{len(d)} cell(s) profiled -> stats/phenotype/cell_profiles.csv"]
    for _, r in d.iterrows():
        msg.append(f"  {r.behavior_base}: grouping identical for cuts r={r.grouping_stable_from}..{r.grouping_stable_to}; "
                   f"groups by cut {r.n_groups_by_cut}")
        sw = r.soma_shares_group_with_branch
        sw_txt = ("soma + branch together" if sw is True else "soma apart from branches" if sw is False
                  else "soma/branch grouping n/a (no soma or no branch region)")
        msg.append(f"  {r.behavior_base}: {r.n_groups} group(s), separation {r.separation:+.2f}, "
                   f"{sw_txt}, "
                   f"{100 * r.frac_events_within_one_group:.0f}% local events, r(soma,branch) {r.r_soma_branch:.2f}")
    if len(d) >= 6 and bool(X.notna().all(axis=None)):
        from sklearn.metrics import silhouette_score
        Xs = (X - X.mean()) / (X.std() + 1e-9); Z = linkage(Xs.values, "ward")
        best = max(range(2, min(5, len(d))), key=lambda k: silhouette_score(Xs, fcluster(Z, k, "maxclust")))
        d["cell_type"] = fcluster(Z, best, "maxclust"); d.to_csv(out_dir / "cell_profiles.csv", index=False)
        msg.append(f"cell types: k={best} by silhouette; counts {d.cell_type.value_counts().to_dict()}")
        for t_, g in d.groupby("cell_type"):
            msg.append(f"  type {t_}: " + ", ".join(f"{f}={g[f].mean():.2f}" for f in feats))
    else:
        msg.append("cell-type clustering: not yet (needs >= 6 cells with complete profiles)")
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    colors = {m: c for m, c in zip(sorted(d.mouse.unique()), plt.cm.tab10.colors)}
    for _, r in d.iterrows():
        ax.scatter(r.r_soma_branch, r.frac_events_within_one_group, s=40 + 30 * r.n_groups, color=colors[r.mouse],
                   edgecolor="k" if r.soma_shares_group_with_branch is True else "none", linewidth=1.2)
        ax.annotate(r.behavior_base.replace("rbp4_", "").replace("_phpeb", ""), (r.r_soma_branch, r.frac_events_within_one_group),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("r(soma, branch)"); ax.set_ylabel("fraction of multi-region events within one group")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
    ax.set_title("Cell coupling phenotypes (size = n groups; black edge = soma co-fires with a branch)", fontsize=9, loc="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / "fig_cell_phenotypes.png", dpi=170); fig.savefig(out_dir / "fig_cell_phenotypes.pdf"); plt.close(fig)
    (out_dir / "summary.txt").write_text("\n".join(msg) + "\n"); print("\n".join(msg))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--run"); g.add_argument("--all", action="store_true")
    ap.add_argument("--r-cut", type=float, default=R_CUT, help="segments correlated at or above this are one co-firing group")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    from femto_status import build_status
    runs = [r for r in build_status(ROOT) if r.get("run_dir") and r.get("stem")]
    for r in [r for r in runs if r.get("mark")]:
        bb = r.get("behavior_base") or f"{r['mouse']}_{r.get('munit', '?')}"
        print(f"  {bb}: skipped ({r['mark']}{': ' + r['mark_reason'] if r.get('mark_reason') else ''})")
    runs = [r for r in runs if not r.get("mark")]                 # run_marks.csv: excluded / revisit
    if args.run:
        runs = [r for r in runs if r.get("behavior_base") == args.run
                or (r.get("_imaging_only") and f"{r['mouse']}_{r.get('munit', '')}" == args.run)]
    for r in runs:
        seg_p = ROOT / r["run_dir"] / f"{r['stem']}_segments_final.tif"; out_p = ROOT / r["run_dir"] / f"{r['stem']}_coupling.json"
        if not seg_p.exists():
            continue
        if out_p.exists() and out_p.stat().st_mtime >= newest_input_mtime(seg_p) and not args.force:
            continue
        res = analyze_run(r, ROOT, args.r_cut)
        if res and "note" not in res:
            print(f"  {r['behavior_base']}: {res['n_groups']} group(s) {res['groups']}")
    cohort(ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
