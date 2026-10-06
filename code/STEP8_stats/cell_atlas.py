#!/usr/bin/env python3
"""cell_atlas.py - every processed cell on one page: picture, 3D view, statistics.

For each cell (all automatic runs in auto_pipeline/, with Daria's hand curation used
instead wherever it exists) it writes:
  stats/cell_atlas/cells/<cell>.png    MIPs (top and side view) with regions, a static
                                       3D render, and the cell's key numbers; every view at
                                       true aspect in um (long tubes are cut into pieces along X)
  stats/cell_atlas/cells/<cell>_views.png  the top + side views alone (for slides)
  stats/cell_atlas/cells/<cell>_3d.html  interactive 3D (rotate / zoom in a browser):
                                       mask, each region, other cells
  stats/cell_atlas/index.html          one table row per cell, thumbnails, links
  stats/cell_atlas/cells.csv           the same numbers as a table
  stats/cell_atlas/atlas.pdf           all cell pages in one PDF

Coordinates are the snake tube's own (X along the scanned path, Y/Z across it), in um
from the voxel size - not re-embedded in real brain space.

    python code/STEP8_stats/cell_atlas.py [--jobs 6] [--only SUBSTRING]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

PROJECT = Path(__file__).resolve().parents[2]
AUTO = PROJECT / "auto_pipeline"
OUT = PROJECT / "stats" / "cell_atlas"
sys.path.insert(0, str(PROJECT / "code"))
from common.voxel import resolve_voxel          # noqa: E402
from common.regions import _names_from_json, ignored_names   # noqa: E402
from common.run_marks import load_marks         # noqa: E402
from common.run_marks import load_quality, rating_label   # noqa: E402  (run_quality.csv)

# no red in the region palette: red is reserved for other cells
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#e377c2", "#bcbd22",
          "#8c564b", "#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5"]


def cells():
    """One entry per mirror run; prefer Daria's curated files in the real tree."""
    out = []
    for seg in sorted(AUTO.glob("rbp4_*/**/*_segments_final.tif")):
        if "old" in seg.parts or any(p.startswith("_") for p in seg.relative_to(AUTO).parts):
            continue
        rel_dir = seg.parent.relative_to(AUTO)
        stem = seg.name.replace("_segments_final.tif", "")
        real_dir = PROJECT / rel_dir
        curated = (real_dir / f"{stem}_segments_final.tif").exists() and \
                  (real_dir / f"{stem}_autoseg_labelmap_reviewed.tif").exists()
        d = real_dir if curated else seg.parent
        out.append({"rel": str(rel_dir), "stem": stem, "dir": str(d), "source": "Daria" if curated else "automatic",
                    "stack": str((real_dir if (real_dir / f"{stem}.tif").exists() else seg.parent) / f"{stem}.tif")})
    return out


def base_of(c):
    for f in ("_metrics.json", "_behavior_coupling.json", "_coupling.json"):
        p = Path(c["dir"]) / f"{c['stem']}{f}"
        if p.exists():
            b = json.loads(p.read_text()).get("behavior_base")
            if b:
                return b
    for tbl in ("imaging_only_runs.csv", "behavior_imaging_master.csv"):   # no outputs yet (e.g. marked runs)
        if (PROJECT / tbl).exists():
            for r in csv.DictReader(open(PROJECT / tbl, newline="")):
                if r.get("run_dir") and r["run_dir"].rstrip("/") == c["rel"] and r.get("behavior_base"):
                    return r["behavior_base"]
    return c["rel"].replace("/", "_")


def load(c):
    d, st = Path(c["dir"]), c["stem"]
    ref = tifffile.imread(d / f"{st}_ref3d.tif") if (d / f"{st}_ref3d.tif").exists() else \
        tifffile.imread(AUTO / c["rel"] / f"{st}_ref3d.tif")
    anat = ref[:, 0].astype(np.float32)
    seg_p = d / f"{st}_segments_final.tif"
    seg = tifffile.imread(seg_p)
    mask = tifffile.imread(d / f"{st}_autoseg_labelmap_reviewed.tif") > 0
    ex_p = d / f"{st}_exclude_labelmap.tif"
    excl = tifffile.imread(ex_p) > 0 if ex_p.exists() else np.zeros_like(mask)
    names = _names_from_json(seg_p)
    ign = {n.lower() for n in ignored_names(seg_p)}
    try:
        vox = tuple(float(v) for v in resolve_voxel(Path(c["stack"]), None, quiet=True))
    except Exception:
        vox = (1.0, 1.0, 1.0)
    jl = lambda f: json.loads((d / f"{st}{f}").read_text()) if (d / f"{st}{f}").exists() else {}
    return anat, seg, mask, excl, names, ign, vox, jl("_metrics.json"), jl("_behavior_coupling.json"), jl("_coupling.json")


def numbers(c, m, bc, cp, names, ign, mark):
    f = lambda v, k=2: (f"{v:.{k}f}" if isinstance(v, (int, float)) and v == v else "")
    slope = ""
    cbd = [x for x in m.get("coupling_by_distance", []) if x.get("r_with_soma") is not None]
    if len(cbd) >= 3:
        dist = np.array([x["distance_um"] for x in cbd]); rr = np.array([x["r_with_soma"] for x in cbd])
        if np.ptp(dist) > 0:
            slope = f"{100 * np.polyfit(dist, rr, 1)[0]:+.2f}"
    beh = {}
    wc = (bc.get("regions") or {}).get("whole cell") or {}
    for b in ("pupil", "whisking", "accelerometer"):
        if b in wc:
            beh[b] = f"{wc[b]['r_peak']:+.2f} ({wc[b]['lag_peak_s']:+.1f} s, p={wc[b]['p_peak']:.3f})"
    row = {
        "cell": c["base"], "source": c["source"], "folder": c["rel"],
        "mark": (mark["mark"] + (": " + mark.get("reason", "") if mark.get("reason") else "")) if mark else "",
        "rating": rating_label(c.get("rating")),
        "reference": m.get("reference_region") or m.get("reference") or "",
        "regions": ", ".join(n + (" (ignored)" if n.lower() in ign else "") for _, n in sorted(names.items())),
        "r_ref_branch": f(m.get("r_soma_branch")), "r_ref_trunk": f(m.get("r_soma_trunk")),
        "branch_only_frac": f(m.get("frac_branch_independent")),
        "branch_first": (f"{f(m.get('branch_first_frac'))} of {m.get('n_paired_events')}" if m.get("n_paired_events") else ""),
        "events_ref_branch": (f"{m.get('n_soma_events', '')} / {m.get('n_branch_events', '')}" if "n_soma_events" in m else ""),
        "r_per_100um": slope, "co_firing_groups": str(cp.get("n_groups", "")) if cp else "",
        "pupil": beh.get("pupil", ""), "whisking": beh.get("whisking", ""), "accelerometer": beh.get("accelerometer", ""),
        "note": m.get("note", "") if m else "no metrics yet",
    }
    return row


def render(c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import matplotlib.transforms
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes
    import plotly.graph_objects as go

    anat, seg, mask, excl, names, ign, vox, m, bc, cp = load(c)
    labels = [l for l in sorted(names) if (seg == l).any()]
    col = {l: COLORS[i % len(COLORS)] for i, l in enumerate(labels)}
    Z, Y, X = anat.shape
    vz, vy, vx = vox
    row = numbers(c, m, bc, cp, names, ign, c["mark"])

    # ---- picture: top (XY) and side (XZ) MIPs with regions, static 3D, numbers.
    # Everything is drawn at TRUE aspect in um (1 um is the same length on every axis): the tube is
    # ~20x longer than it is wide, so instead of stretching Y/Z it is cut into n consecutive pieces
    # along X, stacked (MIPs) or side by side (3D), all at the same scale.
    xum, yum, zum = X * vx, Y * vy, Z * vz
    LW = 10.6                                            # inches available for one strip
    n = 1
    while n < 3 and yum * LW / (xum / n) < 0.75:         # cut until the strip is >= 0.75 in tall
        n += 1
    L = xum / n                                          # um of X per piece (same for every piece)
    s = LW / L                                           # inches per um
    hy, hz = yum * s, zum * s
    lab_h, tick_h, title_h, gap3 = 0.32, 0.36, 0.30, 0.25
    h_top = title_h + n * (lab_h + hy + tick_h)
    h_side = title_h + n * (hz + tick_h)
    h3 = 2.2
    FW, FH = 16.0, 0.55 + h_top + h_side + gap3 + h3 + 0.25
    fig = plt.figure(figsize=(FW, FH))
    x0_in = 0.75
    rect = lambda x, ytop, w, h: [x / FW, (FH - ytop - h) / FH, w / FW, h / FH]   # ytop from the page top
    seg_lab = {l: (np.nonzero(seg == l)) for l in labels}
    y_cur = 0.55
    for proj, lab, hh in ((0, "top view (max over Z), true aspect", hy), (1, "side view (max over Y), true aspect", hz)):
        fig.text(x0_in / FW, 1 - y_cur / FH, lab + (f", cut into {n} pieces along X" if n > 1 else ""),
                 fontsize=10, va="top", ha="left")
        y_cur += title_h
        im = anat.max(proj); lo, hi = np.percentile(im, [1, 99.7])
        hum = yum if proj == 0 else zum
        ext = [0, xum, hum, 0]
        rgba = np.zeros(im.shape + (4,))
        for l in labels:
            rgba[(seg == l).any(proj)] = (*to_rgb(col[l]), 0.18 if names[l].lower() in ign else 0.55)
        ex2 = excl.any(proj) & ~mask.any(proj)
        xs_c = (np.arange(X) + 0.5) * vx; ys_c = (np.arange(im.shape[0]) + 0.5) * (vy if proj == 0 else vz)
        for i in range(n):
            if proj == 0:
                y_cur += lab_h
            ax = fig.add_axes(rect(x0_in, y_cur, LW, hh))
            ax.imshow(np.clip((im - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", aspect="equal", extent=ext,
                      interpolation="nearest")
            ax.imshow(rgba, aspect="equal", extent=ext, interpolation="nearest")
            if ex2.any():                              # other cells: red outline only
                ax.contour(xs_c, ys_c, ex2.astype(float), [0.5], colors="red", linewidths=0.7)
            ax.contour(xs_c, ys_c, mask.any(proj).astype(float), [0.5], colors="yellow", linewidths=0.6)
            ax.set_xlim(i * L, (i + 1) * L); ax.set_ylim(hum, 0)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
            ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
            ax.set_ylabel("Y (um)" if proj == 0 else "Z (um)", fontsize=8, labelpad=2)
            if i == n - 1 and proj == 1:
                ax.set_xlabel("X along the scanned path (um)", fontsize=8.5, labelpad=1)
            if proj == 0:                              # region names above the piece that holds them
                prev = -1e9; row_ = 0
                for l in sorted(labels, key=lambda l: seg_lab[l][2].mean()):
                    xm = (seg_lab[l][2].mean() + 0.5) * vx
                    if not (i * L <= xm < (i + 1) * L or (i == n - 1 and xm >= xum)):
                        continue
                    row_ = 1 - row_ if (xm - prev) * s < 1.1 else 0
                    prev = xm
                    ax.text(xm, 0, names[l] + (" (ignored)" if names[l].lower() in ign else ""), color=col[l],
                            fontsize=8, ha="center", va="bottom", fontweight="bold", clip_on=False,
                            transform=matplotlib.transforms.offset_copy(ax.transData, fig=fig, y=2 + 10 * row_,
                                                                        units="points"))
            y_cur += hh + tick_h

    # static 3D, true aspect: the same n pieces side by side, each its own 3D axes at the same scale
    y_cur += gap3
    n3 = max(n, 2)
    L3 = xum / n3
    fig.text(x0_in / FW, 1 - (y_cur - 0.05) / FH,
             f"3D surface, true aspect (no axis stretched), {n3} consecutive pieces along X; "
             "grey = rest of the cell, red = other cells", fontsize=10, va="bottom", ha="left")

    def mesh(ax, vol, color, alpha, xoff):
        if vol.sum() < 8:
            return
        v, f, _, _ = marching_cubes(np.pad(vol, 1).astype(np.float32), 0.5, spacing=(vz, vy, vx), step_size=1)
        v = v - np.array([vz, vy, vx])
        pts = v[:, [2, 1, 0]] + np.array([xoff, 0, 0])
        pc = Poly3DCollection(pts[f], facecolor=color, alpha=alpha, linewidth=0)
        pc.set_clip_on(False)                          # zoomed box may extend past the axes patch
        ax.add_collection3d(pc)
    w3 = (FW - 0.4) / n3
    for i in range(n3):
        xa, xb = int(np.floor(i * L3 / vx)), int(np.ceil((i + 1) * L3 / vx))
        xa, xb = max(0, xa), min(X, xb)
        ax3 = fig.add_axes(rect(0.2 + i * w3, y_cur, w3, h3), projection="3d")
        sl = (slice(None), slice(None), slice(xa, xb))
        mesh(ax3, (mask & ~(seg > 0))[sl], "#bbbbbb", 0.15, xa * vx)
        for l in labels:
            mesh(ax3, (seg == l)[sl], col[l], 0.25 if names[l].lower() in ign else 0.85, xa * vx)
        mesh(ax3, (excl & ~mask)[sl], "#ff4040", 0.12, xa * vx)
        ax3.set_xlim(i * L3, (i + 1) * L3); ax3.set_ylim(0, yum); ax3.set_zlim(0, zum)
        ax3.set_box_aspect((L3, yum, zum), zoom=2.0)             # real um extents, no multiplier
        ax3.view_init(elev=24, azim=-80)
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.pane.set_alpha(0.0)
        ax3.set_xlabel("X (um)", fontsize=7, labelpad=2); ax3.set_ylabel("Y", fontsize=7, labelpad=-8)
        ax3.set_zlabel("Z", fontsize=7, labelpad=-8)
        ax3.tick_params(labelsize=6, pad=-2)
        ax3.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
        ax3.zaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    # the numbers column, to the right of the strips
    axt = fig.add_axes(rect(x0_in + LW + 0.45, 0.55, FW - (x0_in + LW + 0.45) - 0.1, h_top + h_side))
    axt.axis("off")
    lines = [("cell", row["cell"]), ("masks/regions by", row["source"]), ("folder", row["folder"]),
             ("mark", row["mark"] or "-"), ("reference", row["reference"] or "-"), ("regions", row["regions"]),
             ("", ""), ("r(reference, branch)", row["r_ref_branch"] or "-"), ("r(reference, trunk)", row["r_ref_trunk"] or "-"),
             ("coupling change / 100 um", row["r_per_100um"] or "-"),
             ("branch events without reference", row["branch_only_frac"] or "-"),
             ("branch fires first", row["branch_first"] or "-"), ("events ref / branch", row["events_ref_branch"] or "-"),
             ("co-firing groups", row["co_firing_groups"] or "-"), ("", ""),
             ("pupil  (peak r, lag, p)", row["pupil"] or "no behavior"), ("whisking", row["whisking"] or "-"),
             ("accelerometer", row["accelerometer"] or "-")]
    if row["note"]:
        lines.append(("note", row["note"]))
    y = 0.98
    for k, v in lines:
        if not k:
            y -= 0.025; continue
        axt.text(0.02, y, k, fontsize=8.5, color="0.35", va="top", transform=axt.transAxes)
        v = str(v); v = v if len(v) <= 46 else "\n".join(v[i:i + 46] for i in range(0, len(v), 46))
        axt.text(0.55, y, v, fontsize=8.5, va="top", transform=axt.transAxes)
        y -= 0.052 * (1 + v.count("\n"))
    fig.suptitle(f"{row['cell']}   ({row['source']})" + (f"   rating: {row['rating']}" if row.get("rating") else ""),
                 fontsize=12, x=0.01, ha="left")
    png = OUT / "cells" / f"{c['base']}.png"
    fig.savefig(png, dpi=90, bbox_inches="tight")
    # the top + side views alone (for slides), cut from the same figure in inches
    from matplotlib.transforms import Bbox
    fig.savefig(OUT / "cells" / f"{c['base']}_views.png", dpi=150,
                bbox_inches=Bbox([[0.05, FH - (0.55 + h_top + h_side)], [x0_in + LW + 0.12, FH - 0.5]]))
    plt.close(fig)

    # ---- interactive 3D (true aspect, um)
    traces = []
    def gomesh(vol, color, name, opacity):
        if vol.sum() < 8:
            return
        v, f, _, _ = marching_cubes(np.pad(vol, 1).astype(np.float32), 0.5, spacing=(vz, vy, vx), step_size=1)
        v = v - np.array([vz, vy, vx])
        traces.append(go.Mesh3d(x=v[:, 2], y=v[:, 1], z=v[:, 0], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=color, opacity=opacity, name=name, showlegend=True, flatshading=False))
    gomesh(mask & ~(seg > 0), "#bbbbbb", "rest of the cell", 0.2)
    for l in labels:
        gomesh(seg == l, col[l], names[l] + (" (ignored)" if names[l].lower() in ign else ""),
               0.3 if names[l].lower() in ign else 0.9)
    gomesh(excl & ~mask, "#ff4040", "other cells", 0.25)
    f3 = go.Figure(traces)
    f3.update_layout(title=f"{row['cell']} ({row['source']}) - drag to rotate, scroll to zoom; X along the scanned path, um",
                     scene=dict(aspectmode="data", xaxis_title="X (um)", yaxis_title="Y (um)", zaxis_title="Z (um)"),
                     margin=dict(l=0, r=0, t=40, b=0))
    f3.write_html(OUT / "cells" / f"{c['base']}_3d.html", include_plotlyjs="../plotly.min.js", full_html=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", default=None, help="process cells whose folder contains this text")
    a = ap.parse_args()
    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    import plotly
    js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    (OUT / "plotly.min.js").write_bytes(js.read_bytes())          # offline, one copy for all pages
    marks = load_marks()
    ratings = load_quality()
    cs = cells()
    for c in cs:
        c["base"] = base_of(c); c["mark"] = marks.get(c["base"])
        c["rating"] = (ratings.get(c["base"]) or {}).get("rating")
    if a.only:
        cs = [c for c in cs if a.only in c["rel"]]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for c, r in zip(cs, ex.map(render, cs)):
            rows.append(r); print(f"  {r['cell']:40s} {r['source']:9s} {r['mark'][:30]}")
    rows.sort(key=lambda r: (r["mark"] != "", r["cell"]))
    with open(OUT / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # index.html
    cols = ["source", "mark", "reference", "regions", "r_ref_branch", "r_ref_trunk", "r_per_100um",
            "branch_only_frac", "branch_first", "co_firing_groups", "pupil", "whisking", "accelerometer"]
    hdr = "".join(f"<th>{html.escape(c)}</th>" for c in ["cell", "picture"] + cols)
    body = []
    for r in rows:
        b = html.escape(r["cell"])
        tds = "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols)
        style = ' style="background:#fff3e0"' if r["mark"] else (' style="background:#e8f5e9"' if r["source"] == "Daria" else "")
        body.append(f'<tr{style}><td><b>{b}</b><br><a href="cells/{b}_3d.html">3D view</a> | <a href="cells/{b}.png">page</a></td>'
                    f'<td><a href="cells/{b}.png"><img src="cells/{b}.png" width="420"></a></td>{tds}</tr>')
    (OUT / "index.html").write_text(
        "<html><head><meta charset='utf-8'><title>Cell atlas</title><style>body{font-family:Arial;font-size:12px}"
        "td,th{border:1px solid #ccc;padding:4px;vertical-align:top}table{border-collapse:collapse}</style></head><body>"
        f"<h2>Cell atlas - {len(rows)} cells</h2><p>Green rows: masks and regions by Daria. Orange rows: marked "
        "(revisit/excluded; left out of statistics). Others: automatic. r = correlation of dF/F traces (raw data); "
        "'r_per_100um' = change of a region's correlation with the reference per 100 um of path. "
        "3D views open in the browser (rotate / zoom).</p>"
        f"<table><tr>{hdr}</tr>{''.join(body)}</table></body></html>")
    # one PDF with every page
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(OUT / "atlas.pdf") as pdf:
        for r in rows:
            img = plt.imread(OUT / "cells" / f"{r['cell']}.png")
            fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100)); fig.add_axes([0, 0, 1, 1]).imshow(img)
            plt.axis("off"); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {OUT / 'index.html'}, cells.csv, atlas.pdf ({len(rows)} cells)")


if __name__ == "__main__":
    main()
