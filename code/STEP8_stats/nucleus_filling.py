#!/usr/bin/env python
"""nucleus_filling.py - is GCaMP excluded from the nucleus (ring) or has it filled it (disc)?

STATUS: the automatic soma scorer below is EXPERIMENTAL and was found unreliable on
single noisy snapshots (see stats/nucleus/README.md). Use the contact sheets
(--sheets, the default) and score by eye. The automatic path is kept for a future
averaged soma-layer frame, where it may become usable.

Retrospective estimate from the 2P snapshots taken while finding each cell. For every
session, somata are detected in every snapshot, each soma gets a CENTER/RIM ratio
(mean F in the inner 35 % of the radius divided by mean F in the outer rim, both
background-subtracted): clearly below 1 = dark nucleus (healthy exclusion), ~1 or above
= filled. Per session: gallery sheet (stats/nucleus/<mouse>_<date>_somata.png), per-soma
table, and a filled fraction with a Wilson 95 % interval. All sessions: trend figure vs
days post-injection and stats/nucleus/nucleus_sessions.csv.

Honest limits: single frames at depth are noisy; detection is automatic (a bright
dendrite cross-section can masquerade as a soma); 0.4-0.75 um pixels under-resolve a
6-8 um nucleus. Treat the numbers as a ranking with wide error bars, not a measurement.
A dedicated averaged soma-layer frame per session (planned) will replace this.

  python code/STEP8_stats/nucleus_filling.py            # all sessions
  python code/STEP8_stats/nucleus_filling.py --session rbp4_141_phpeb/06-25-2026
  --filled-threshold 0.8   ratio at/above which a soma counts as filled (default 0.8)
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.feature import blob_log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "stats" / "nucleus"
SOMA_R_UM = (3.5, 9.0)              # plausible L5 soma radius range


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def score_image(im, px_um, min_contrast_sd=2.5):
    """Detect round bright bodies of soma size, return per-soma rows."""
    sm = ndi.gaussian_filter(im.astype(float), 1.0)
    bg = np.percentile(sm, 50); sd = 1.4826 * np.median(np.abs(sm - bg)) + 1e-9
    smin, smax = SOMA_R_UM[0] / px_um / 1.414, SOMA_R_UM[1] / px_um / 1.414
    blobs = blob_log((sm - bg).clip(0) / sd, min_sigma=max(1.5, smin), max_sigma=max(2.5, smax), num_sigma=7, threshold=1.2)
    rows = []
    sm2 = ndi.gaussian_filter(im.astype(float), max(1.0, 0.8 / px_um))   # ~0.8 um smoothing for the profile
    for y, x, s0 in blobs:
        y, x = int(y), int(x)
        Rmax = int(round(SOMA_R_UM[1] * 1.4 / px_um))
        if y - Rmax < 0 or x - Rmax < 0 or y + Rmax + 1 > im.shape[0] or x + Rmax + 1 > im.shape[1]:
            continue
        yy, xx = np.ogrid[-Rmax:Rmax + 1, -Rmax:Rmax + 1]; rr = np.hypot(yy, xx)
        patch = sm2[y - Rmax:y + Rmax + 1, x - Rmax:x + Rmax + 1]
        # recentre on the body's centroid (the blob detector may sit on the rim)
        body = (patch - np.percentile(patch, 20)).clip(0) * (rr <= SOMA_R_UM[1] / px_um)
        if body.sum() <= 0:
            continue
        cy, cx = int(round((body * yy).sum() / body.sum())), int(round((body * xx).sum() / body.sum()))
        y2, x2 = y + cy, x + cx
        if y2 - Rmax < 0 or x2 - Rmax < 0 or y2 + Rmax + 1 > im.shape[0] or x2 + Rmax + 1 > im.shape[1]:
            continue
        patch = sm2[y2 - Rmax:y2 + Rmax + 1, x2 - Rmax:x2 + Rmax + 1]
        # radial profile; the soma edge/rim is the outermost strong maximum in the plausible range
        step = max(1.0, 0.5 / px_um); edges = np.arange(0, Rmax + step, step)
        prof = np.array([patch[(rr >= a) & (rr < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
        mid = (edges[:-1] + edges[1:]) / 2 * px_um
        ok = (mid >= SOMA_R_UM[0] * 0.6) & (mid <= SOMA_R_UM[1])
        if not ok.any():
            continue
        out = np.median(prof[mid > SOMA_R_UM[1] * 1.1]) if (mid > SOMA_R_UM[1] * 1.1).any() else prof[-1]
        pk = np.flatnonzero(ok)[np.argmax(prof[ok])]
        r_um = mid[pk]; rim = prof[pk]; r = r_um / px_um
        center = patch[rr <= 0.4 * r].mean()
        if rim - out < min_contrast_sd * sd:
            continue
        # roundness of the body
        body = (patch - out).clip(0) * (rr <= 1.3 * r)
        syy = (body * yy ** 2).sum() / body.sum(); sxx = (body * xx ** 2).sum() / body.sum(); sxy = (body * yy * xx).sum() / body.sum()
        ev = np.linalg.eigvalsh([[syy, sxy], [sxy, sxx]]); aspect = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
        if aspect > 1.6:
            continue
        if any(abs(r0["y"] - y2) < r and abs(r0["x"] - x2) < r for r0 in rows):   # same soma found twice
            continue
        rows.append({"y": y2, "x": x2, "r_px": int(round(r)), "r_um": float(r_um), "center": float(center), "rim": float(rim),
                     "background": float(out), "ratio": float((center - out) / (rim - out)), "contrast_sd": float((rim - out) / sd),
                     "aspect": aspect})
    return rows


L5_DEPTH_UM = (450, 750)            # only frames in the L5 soma layer are scored


def viewport_tilt_deg(mesc_path, session_key, unit_key):
    """0 = top-down XY frame; anything else is a tilted side view (dendrites look like
    vertical bars there and must never be scored as somata)."""
    import h5py, json
    with h5py.File(mesc_path, "r") as f:
        u = f[session_key][unit_key]
        v = u.attrs.get("ReferenceViewportJSON")
        if v is None:
            return np.nan
        v = v if isinstance(v, str) else (v[0].decode() if isinstance(v[0], bytes) else str(v[0]))
        q = np.array(json.loads(v)["viewports"][0]["geomTransRot"], float)
        return float(np.degrees(2 * np.arccos(np.clip(abs(q[3]), 0, 1))))


def session_sheet(session_dir: Path, thr: float):
    idx_f = glob.glob(str(session_dir / "snapshots" / "*index.csv"))
    if not idx_f:
        return None
    idx = pd.read_csv(idx_f[0]); idx = idx[idx.channel_name != "Camera"]
    mesc = sorted(glob.glob(str(session_dir / "raw" / "*.mesc")))
    if mesc:
        idx = idx.copy()
        idx["tilt"] = [viewport_tilt_deg(mesc[0], r.session, r.unit) for _, r in idx.iterrows()]
        idx = idx[(idx.tilt < 5) & (idx.depth_rel_um.abs() >= L5_DEPTH_UM[0]) & (idx.depth_rel_um.abs() <= L5_DEPTH_UM[1])]
    if idx.empty:
        return {"mouse": session_dir.parts[-2], "date": session_dir.parts[-1], "n_somata": 0, "n_filled": 0,
                "frac_filled": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "median_ratio": np.nan, "n_snapshots": 0,
                "note": "no top-down frame in the L5 soma layer"}
    mouse, date = session_dir.parts[-2], session_dir.parts[-1]
    allrows = []
    for _, r in idx.iterrows():
        tif = session_dir / "snapshots" / Path(str(r.tif)).name
        if not tif.exists():
            continue
        im = tifffile.imread(tif)
        if im.ndim != 2:
            continue
        for row in score_image(im, float(r.pixel_x_um)):
            row.update({"snapshot": tif.name, "unit": r.unit, "depth_um": float(r.depth_rel_um), "px_um": float(r.pixel_x_um)})
            allrows.append(row)
    d = pd.DataFrame(allrows)
    n = len(d); k = int((d.ratio >= thr).sum()) if n else 0; lo, hi = wilson(k, n)
    # gallery: up to 48 somata, sorted by contrast, each with its ratio
    show = d.sort_values("contrast_sd", ascending=False).head(48) if n else d
    cols = 8; rws = max(1, int(np.ceil(len(show) / cols)))
    fig, ax = plt.subplots(rws, cols, figsize=(cols * 2.0, rws * 2.1 + 0.8))
    ax = np.atleast_2d(ax)
    cache = {}
    for a, (_, s) in zip(ax.ravel(), show.iterrows()):
        if s.snapshot not in cache:
            cache[s.snapshot] = tifffile.imread(session_dir / "snapshots" / s.snapshot).astype(float)
        im = cache[s.snapshot]; R = int(s.r_px) + 6; y, x = int(s.y), int(s.x)
        p = im[max(0, y - R):y + R + 1, max(0, x - R):x + R + 1]
        lo_, hi_ = s.background, s.rim * 1.5
        a.imshow(np.clip((p - lo_) / (hi_ - lo_ + 1e-9), 0, 1), cmap="gray", interpolation="nearest")
        filled = s.ratio >= thr
        a.set_title(f"{'FILLED' if filled else 'ring'} {s.ratio:.2f}  r={s.r_um:.0f}um  {abs(s.depth_um):.0f}um", fontsize=7,
                    color="orangered" if filled else "seagreen")
        a.set_xticks([]); a.set_yticks([])
    for a in ax.ravel()[len(show):]:
        a.axis("off")
    fig.suptitle(f"{mouse}  {date}   somata in 2P snapshots: {n} detected, filled (ratio >= {thr}): {k}/{n} = "
                 f"{100 * k / max(n, 1):.0f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)", fontsize=10)
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{mouse}_{date}_somata.png", dpi=110); plt.close(fig)
    if n:
        d.insert(0, "date", date); d.insert(0, "mouse", mouse)
        d.to_csv(OUT / f"{mouse}_{date}_somata.csv", index=False)
    return {"mouse": mouse, "date": date, "n_somata": n, "n_filled": k, "frac_filled": k / n if n else np.nan,
            "ci_lo": lo, "ci_hi": hi, "median_ratio": float(d.ratio.median()) if n else np.nan,
            "n_snapshots": int(idx.shape[0])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="<mouse>/<MM-DD-YYYY> (default: every session with snapshots)")
    ap.add_argument("--filled-threshold", type=float, default=0.8)
    args = ap.parse_args(argv)
    sessions = [ROOT / args.session] if args.session else sorted(p.parent for p in ROOT.glob("rbp4_*/*/snapshots"))
    res = []
    for s in sessions:
        r = session_sheet(s, args.filled_threshold)
        if r:
            res.append(r)
            if r.get("note"):
                print(f"  {r['mouse']:18s} {r['date']}  {r['note']}"); continue
            print(f"  {r['mouse']:18s} {r['date']}  somata {r['n_somata']:3d}  filled {100 * r['frac_filled']:5.1f}%  "
                                 f"(CI {100 * r['ci_lo']:.0f}-{100 * r['ci_hi']:.0f})  median ratio {r['median_ratio']:.2f}")
    if not res:
        print("no sessions with snapshots"); return 1
    t = pd.DataFrame(res)
    mice = pd.read_csv(ROOT / "mice.csv", parse_dates=["injection_date"])
    t = t.merge(mice[["mouse", "injection_date"]], on="mouse", how="left")
    t["dpi"] = (pd.to_datetime(t.date, format="%m-%d-%Y") - t.injection_date).dt.days
    t.to_csv(OUT / "nucleus_sessions.csv", index=False)
    # trend figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {m: c for m, c in zip(sorted(t.mouse.unique()), plt.cm.tab10.colors)}
    for m, g in t.dropna(subset=["dpi", "frac_filled"]).groupby("mouse"):
        g = g.sort_values("dpi")
        ax[0].errorbar(g.dpi, 100 * g.frac_filled, yerr=[(100 * (g.frac_filled - g.ci_lo)).clip(lower=0), (100 * (g.ci_hi - g.frac_filled)).clip(lower=0)],
                       fmt="o-", color=colors[m], label=m.replace("rbp4_", ""), capsize=3, ms=5)
        ax[1].plot(g.dpi, g.median_ratio, "o-", color=colors[m], ms=5)
    ax[0].set_ylabel("% somata filled"); ax[0].set_xlabel("days post-injection"); ax[0].set_ylim(0, 105); ax[0].legend(fontsize=7, frameon=False)
    ax[1].set_ylabel("median center/rim ratio"); ax[1].set_xlabel("days post-injection"); ax[1].axhline(args.filled_threshold, color="k", lw=0.7, ls="--")
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.suptitle("Nuclear filling from 2P snapshots (automatic, noisy: ranking only, see galleries)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_nucleus_trend.png", dpi=170); fig.savefig(OUT / "fig_nucleus_trend.pdf"); plt.close(fig)
    print(f"wrote stats/nucleus/ ({len(t)} sessions): galleries, per-soma CSVs, nucleus_sessions.csv, fig_nucleus_trend")
    return 0


if __name__ == "__main__":
    sys.exit(main())
