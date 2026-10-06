#!/usr/bin/env python3
"""Z-redundancy QC: how much independent depth information does a snake run have?

A snake volume can be commanded over a real Z range and still carry almost no
independent depth information, because ~1 um Z steps oversample an axial PSF of
~2-4 um. The planes are then near-copies of one shared image: the 3D machinery
downstream (skeleton, geodesic region splitting, distance-along-dendrite) runs,
but its depth axis is decorative, and branches that cross in XY within a few um
in Z cannot be separated.

This measures it on the pixels rather than guessing from the metadata:

    pc1_frac   fraction of across-plane variance explained by the single shared
               image (first SVD component of the Z x pixels matrix)
    n_comp_95  how many components are needed for 95% of that variance
               -- the effective number of independent planes
    adj_corr   mean correlation between adjacent planes
    dup_planes byte-identical adjacent planes (a real extraction bug, not thin-slab)

Reference points measured on this project's data:
    June rbp4_141 06-17 run01 (usable volume):  pc1 51.8%, n_comp_95 = 6
    Ayla 053 09-15 Run012 (thin slab):          pc1 88.9%, n_comp_95 = 2

Default verdict thresholds (--pc1-max / --min-comp):
    pc1_frac >= 0.80 OR n_comp_95 <= 2  ->  thin_slab (3D analysis not meaningful)

Usage:
    python z_redundancy_qc.py --tif path/to/runNN_clean.tif --nz 10
    python z_redundancy_qc.py --mesc file.mesc --unit MUnit_34 --nz 10
    python z_redundancy_qc.py --batch batch.csv -o z_quality.csv
      (batch CSV columns: label,source,unit,nz  -- source is a .tif or .mesc path)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_volumes(source: str, nz: int, unit: str | None = None, nvol: int = 30):
    """Return (nvol, nz, Y, X) float64 from a 4D/flat TIFF or straight from a .mesc."""
    src = Path(source)
    if src.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(src) as t:
            npages = len(t.pages)
            take = min(nvol * nz, (npages // nz) * nz)
            if take < nz:
                raise ValueError(f"{src.name}: only {npages} pages, need >= {nz}")
            a = t.asarray(key=range(take))
        return a.reshape(take // nz, nz, *a.shape[-2:]).astype(np.float64)

    import h5py
    with h5py.File(src, "r") as f:
        if unit is None:
            raise ValueError("--unit is required for a .mesc source")
        path = None
        for sess in f:
            if unit in f[sess] and "Channel_0" in f[sess][unit]:
                path = f"{sess}/{unit}/Channel_0"
                break
        if path is None:
            raise ValueError(f"{unit} not found in {src.name}")
        ds = f[path]
        take = min(nvol * nz, (ds.shape[0] // nz) * nz)
        a = ds[:take]
    return a.reshape(take // nz, nz, *a.shape[-2:]).astype(np.float64)


def z_metrics(vol: np.ndarray) -> dict:
    """vol: (T, Z, Y, X). Metrics on the time-averaged volume."""
    T, Z = vol.shape[0], vol.shape[1]
    m = vol.mean(axis=0)                       # (Z, Y, X)
    X = m.reshape(Z, -1)

    C = np.corrcoef(X)
    adj = [C[i, i + 1] for i in range(Z - 1)] or [np.nan]
    Xc = X - X.mean()
    s = np.linalg.svd(Xc, compute_uv=False)
    var = s ** 2 / np.sum(s ** 2)
    n95 = int(np.searchsorted(np.cumsum(var), 0.95) + 1)

    v0 = vol[0].reshape(Z, -1)
    dup = sum(1 for i in range(Z - 1) if np.array_equal(v0[i], v0[i + 1]))

    return {
        "n_vol_used": T, "nz": Z,
        "pc1_frac": round(float(var[0]), 4),
        "n_comp_95": n95,
        "adj_corr": round(float(np.mean(adj)), 4),
        "first_last_corr": round(float(C[0, -1]), 4),
        "dup_planes": dup,
        "plane_mean_range": round(float(np.ptp(m.mean(axis=(1, 2)))), 2),
    }


def verdict(m: dict, pc1_max: float, min_comp: int) -> str:
    if m["dup_planes"] > 0:
        return "duplicate_planes"          # extraction bug, not optics
    if m["pc1_frac"] >= pc1_max or m["n_comp_95"] <= min_comp:
        return "thin_slab"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif"), ap.add_argument("--mesc"), ap.add_argument("--unit")
    ap.add_argument("--nz", type=int)
    ap.add_argument("--batch", help="CSV with label,source,unit,nz")
    ap.add_argument("-o", "--out")
    ap.add_argument("--nvol", type=int, default=30)
    ap.add_argument("--pc1-max", type=float, default=0.80)
    ap.add_argument("--min-comp", type=int, default=2)
    a = ap.parse_args()

    jobs = []
    if a.batch:
        for r in csv.DictReader(open(a.batch)):
            jobs.append((r["label"], r["source"], r.get("unit") or None, int(r["nz"])))
    elif a.tif or a.mesc:
        if not a.nz:
            sys.exit("--nz is required")
        jobs.append((Path(a.tif or a.mesc).name, a.tif or a.mesc, a.unit, a.nz))
    else:
        sys.exit("give --tif/--mesc or --batch")

    rows = []
    hdr = (f"{'run':26s}{'nz':>3s}{'pc1%':>7s}{'n95':>5s}{'adj_r':>7s}"
           f"{'1st-last':>9s}{'dup':>4s}  verdict")
    print(hdr)
    for label, source, unit, nz in jobs:
        try:
            vol = load_volumes(source, nz, unit, a.nvol)
            m = z_metrics(vol)
            v = verdict(m, a.pc1_max, a.min_comp)
            m.update(label=label, source=source, unit=unit or "", verdict=v)
            rows.append(m)
            print(f"{label:26s}{nz:>3d}{m['pc1_frac']*100:>7.1f}{m['n_comp_95']:>5d}"
                  f"{m['adj_corr']:>7.3f}{m['first_last_corr']:>9.3f}{m['dup_planes']:>4d}  {v}")
        except Exception as exc:
            print(f"{label:26s}{nz:>3d}  ERROR: {exc}")
            rows.append({"label": label, "source": source, "unit": unit or "",
                         "nz": nz, "verdict": "not_measured", "error": str(exc)})

    if a.out and rows:
        cols = ["label", "source", "unit", "nz", "n_vol_used", "pc1_frac", "n_comp_95",
                "adj_corr", "first_last_corr", "dup_planes", "plane_mean_range",
                "verdict", "error"]
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.out}")
        n_thin = sum(1 for r in rows if r.get("verdict") == "thin_slab")
        print(f"thin_slab: {n_thin}/{len(rows)}")


if __name__ == "__main__":
    main()
