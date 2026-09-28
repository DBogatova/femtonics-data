#!/usr/bin/env python3
"""Count rising edges per channel in raw behavior trigger `RunNNN_t1.mat` files.

These are MATLAB v7.3 (HDF5) files holding a timetable per acquisition board.
The channel sample vectors live in `#refs#` as the `data` of a group that also
carries `varNames`; `device/di/Name` and `device/ai/Name` give the hardware
channel names in the same order.

The `AndorXylaTrigger` rising-edge count is the imaging fingerprint: one pulse
per scanned plane, i.e. `n_t * snake_n_slices` in the .mesc (`n_t` for a ribbon
scan, where slices == 1). That makes behavior<->imaging pairing exact.

Usage:
    python trigger_counts_from_mat.py DIR_OR_FILE [...]  [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import h5py
import numpy as np


def _mstr(f, ref):
    """Decode a MATLAB char array referenced by `ref`."""
    try:
        a = np.array(f[ref]).ravel()
    except Exception:
        return ""
    return "".join(chr(c) for c in a if c)


def channel_names(f):
    """Hardware channel names, digital first then analog, as stored."""
    out = {}
    for kind in ("di", "ai"):
        key = f"device/{kind}/Name"
        if key not in f:
            continue
        names = []
        arr = np.array(f[key]).ravel()
        for r in arr:
            n = _mstr(f, r)
            if n:
                names.append(n)
        out[kind] = names
    return out


def timetables(f):
    """[(varNames, [sample_arrays])] for every timetable found in #refs#."""
    found = []
    if "#refs#" not in f:
        return found
    for k in f["#refs#"]:
        g = f["#refs#"][k]
        if not isinstance(g, h5py.Group):
            continue
        if "varNames" not in g or "data" not in g:
            continue
        names = [_mstr(f, r) for r in np.array(g["varNames"]).ravel()]
        arrays = []
        for r in np.array(g["data"]).ravel():
            try:
                arrays.append(np.array(f[r]).ravel())
            except Exception:
                arrays.append(np.array([]))
        found.append((names, arrays, g.name))
    return found


def rising_edges(x, thresh=None):
    """Number of low->high transitions in a sampled trace."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0
    if thresh is None:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if hi - lo < 1e-9:
            return 0
        thresh = lo + 0.5 * (hi - lo)
    b = x > thresh
    return int(np.count_nonzero(b[1:] & ~b[:-1]))


def analyze(path):
    """-> dict of channel_name -> (rising_edges, n_samples, rate_hz)."""
    with h5py.File(path, "r") as f:
        names_by_kind = channel_names(f)
        rate = None
        for key in ("device/rate", "device/actualRate"):
            if key in f:
                rate = float(np.array(f[key]).ravel()[0])
                break
        tts = timetables(f)
        # the digital board's timetable is the one whose variable count
        # matches the digital channel list
        di_names = names_by_kind.get("di", [])
        ai_names = names_by_kind.get("ai", [])
        result = {}
        for names, arrays, where in tts:
            n = len(arrays)
            hw = None
            if n == len(di_names):
                hw = di_names
            elif n == len(ai_names):
                hw = ai_names
            for i, arr in enumerate(arrays):
                label = None
                if hw and i < len(hw):
                    label = hw[i]
                elif i < len(names) and names[i]:
                    label = names[i]
                if not label:
                    label = f"{os.path.basename(where)}_var{i}"
                if arr.size <= 1:
                    continue
                e = rising_edges(arr)
                prev = result.get(label)
                if prev is None or arr.size > prev[1]:
                    result[label] = (e, int(arr.size), rate)
        return result, rate


def run_number(path):
    m = re.search(r"Run(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="directories or RunNNN_t1.mat files")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.mat")))
        else:
            files.append(p)
    if not files:
        sys.exit("no .mat files found")

    rows = []
    for p in files:
        try:
            res, rate = analyze(p)
        except Exception as e:
            print(f"{os.path.basename(p)}: ERROR {e}")
            continue
        andor = res.get("AndorXylaTrigger", (None, None, None))
        basler = res.get("baslerExposureTrigger", (None, None, None))
        istart = res.get("imagingStart", (None, None, None))
        nsamp = max((v[1] for v in res.values()), default=0)
        rows.append({
            "file": os.path.abspath(p),
            "run_number": run_number(p),
            "rate_hz": rate,
            "n_samples": nsamp,
            "record_s": (nsamp / rate) if rate else "",
            "AndorXylaTrigger_edges": andor[0],
            "baslerExposureTrigger_edges": basler[0],
            "imagingStart_edges": istart[0],
            "all_channels": "; ".join(f"{k}={v[0]}" for k, v in sorted(res.items())),
        })
        print(f"{os.path.basename(p):18} rate={rate}  samples={nsamp}  "
              f"Andor={andor[0]}  basler={basler[0]}  imagingStart={istart[0]}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nwrote", args.csv)


if __name__ == "__main__":
    main()
