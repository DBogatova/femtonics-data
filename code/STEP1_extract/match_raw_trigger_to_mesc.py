#!/usr/bin/env python3
"""Match RAW behavior trigger files to .mesc imaging runs.

Companion to `match_behavior_imaging.py`, for sessions where the behavior has
NOT yet been through the processing pipeline (no `*_info.txt` / `*_behavior.csv`
yet) -- only the raw `RunNNN_t1.mat` trigger files and camera frame folders.
This is the layout on the SCC under another lab member's tree.

Same exact key: `AndorXylaTrigger` rising edges (counted from the .mat by
`trigger_counts_from_mat.py`) == `n_t * snake_n_slices` in the .mesc, which is
just `n_t` for a ribbon scan since slices == 1.

Usage:
    python match_raw_trigger_to_mesc.py SUMMARY.csv TRIGGER_COUNTS.csv \
        [--camera-dir DIR] [--date 08-13-2026] [--mouse rbp4_phpebne_053] \
        [-o out.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

REAL = ("snake", "ribbon", "ribbon_transverse", "ribbon_longitudinal")
COMMENT_RUN_RE = re.compile(r"\brun\s*#?\s*(\d{1,2})\b", re.I)

COLUMNS = [
    "mouse", "date", "mesc_file", "mesc_session", "munit", "scan_type",
    "imaging_timestamp_utc", "duration_s", "n_t", "n_slices", "n_y", "n_x",
    "frame_rate_hz", "pixel_x_um", "pixel_y_um", "voxel_z_um",
    "span_x_um", "span_y_um",
    "imaging_comment", "comment_run_number",
    "planes_expected", "trigger_andor_edges", "trigger_mat",
    "behavior_run", "camera_frames_expected", "camera_dir", "camera_frames_found",
    "camera_state",
    "match_confidence", "notes",
]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def comment_run(c):
    m = COMMENT_RUN_RE.search(c or "")
    return int(m.group(1)) if m else None


def load_units(path):
    units, seen = [], set()
    for r in csv.DictReader(open(path)):
        if r.get("scan_type") not in REAL:
            continue
        key = (r["session"], r["unit"])
        if key in seen:
            continue
        seen.add(key)
        n_t = int(f(r.get("n_t")) or 0)
        sl = int(f(r.get("snake_n_slices")) or 0) or 1
        units.append({
            "row": r, "session": r["session"], "unit": r["unit"],
            "n_t": n_t, "slices": sl, "fp": n_t * sl,
            "comment": (r.get("comment") or "").strip(),
            "ts": r.get("timestamp", ""),
        })
    units.sort(key=lambda u: (u["ts"], u["unit"]))
    return units


def load_triggers(path):
    trig = []
    for r in csv.DictReader(open(path)):
        e = f(r.get("AndorXylaTrigger_edges"))
        trig.append({
            "run": int(f(r.get("run_number")) or 0),
            "edges": int(e) if e is not None else None,
            "basler": int(f(r.get("baslerExposureTrigger_edges")) or 0),
            "mat": os.path.basename(r.get("file", "")),
            "path": r.get("file", ""),
        })
    trig.sort(key=lambda t: t["run"])
    return trig


def match(units, trig):
    by_u, by_t = defaultdict(list), defaultdict(list)
    for u in units:
        by_u[u["fp"]].append(u)
    for t in trig:
        by_t[t["edges"]].append(t)

    pairs, used_u, used_t = [], set(), set()
    for fp, us in by_u.items():
        ts = list(by_t.get(fp, []))
        if not ts:
            continue
        us = sorted(us, key=lambda u: u["ts"])
        unique = len(us) == 1 and len(ts) == 1
        free_u, free_t, bound = list(us), list(ts), []
        if not unique:
            for u in list(free_u):
                n = comment_run(u["comment"])
                hit = next((t for t in free_t if t["run"] == n), None) if n else None
                if hit:
                    bound.append((u, hit, "fingerprint+comment_run"))
                    free_u.remove(u)
                    free_t.remove(hit)
        for u, t in zip(free_u, free_t):
            bound.append((u, t, "fingerprint_exact" if unique else "fingerprint+order"))
        for u, t, conf in bound:
            pairs.append((u, t, conf))
            used_u.add(id(u))
            used_t.add(id(t))

    lone_u = [u for u in units if id(u) not in used_u]
    lone_t = [t for t in trig if id(t) not in used_t]

    # leftovers: if exactly one each remains, the comment run number or plain
    # elimination identifies it, but the counts disagree -> report loudly
    if len(lone_u) == 1 and len(lone_t) == 1:
        pairs.append((lone_u[0], lone_t[0], "ELIMINATION_count_mismatch"))
        lone_u, lone_t = [], []

    pairs.sort(key=lambda p: p[0]["ts"])
    return pairs, lone_u, lone_t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary")
    ap.add_argument("triggers")
    ap.add_argument("--camera-dir", default=None,
                    help="local folder holding per-run camera frame dirs (RunNNN or runN)")
    ap.add_argument("--camera-index", default=None,
                    help="text file of '<folder> <n_frames>' per line listing the camera "
                         "folders that exist at the SOURCE, so runs that simply were not "
                         "downloaded are not reported as missing recordings")
    ap.add_argument("--mouse", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    units = load_units(a.summary)
    trig = load_triggers(a.triggers)
    pairs, lone_u, lone_t = match(units, trig)

    index = {}
    if a.camera_index and os.path.exists(a.camera_index):
        for line in open(a.camera_index):
            parts = line.split()
            if len(parts) >= 2:
                index[parts[0]] = int(parts[1])

    def camera_for(run):
        """-> (folder_name, n_frames_local, source_state)."""
        names = (f"Run{run:03d}", f"run{run}", f"Run{run}", f"run{run:03d}")
        if a.camera_dir and os.path.isdir(a.camera_dir):
            for name in names:
                p = os.path.join(a.camera_dir, name)
                if os.path.isdir(p):
                    n = len([x for x in os.listdir(p)
                             if x.lower().endswith((".tif", ".tiff"))])
                    return name, n, "local"
        for name in names:
            if name in index:
                return name, "", ("source_empty" if index[name] == 0
                                  else "on_source_not_downloaded")
        return "", "", "no_recording"

    rows = []
    for u, t, conf in pairs:
        r = u["row"]
        cam, nfr, cstate = camera_for(t["run"])
        notes = []
        if conf == "fingerprint+order":
            notes.append("several runs share this trigger count; paired by order")
        if conf == "fingerprint+comment_run":
            notes.append("tie broken by the run number in the .mesc comment")
        if conf == "ELIMINATION_count_mismatch":
            notes.append(f"COUNT MISMATCH: imaging expects {u['fp']} pulses, "
                         f"trigger file has {t['edges']}; paired only by elimination")
        cr = comment_run(u["comment"])
        if cr is not None and cr != t["run"]:
            notes.append(f"CHECK: comment says run {cr}, paired with Run{t['run']:03d}")
        if cstate == "local" and t["basler"] and nfr != t["basler"]:
            notes.append(f"camera frames on disk ({nfr}) != basler pulses ({t['basler']})")
        elif cstate == "on_source_not_downloaded":
            notes.append(f"camera folder '{cam}' exists on SCC, not downloaded")
        elif cstate == "source_empty":
            notes.append(f"camera folder '{cam}' exists but is EMPTY at the source")
        elif cstate == "no_recording":
            notes.append("no camera recording for this run")
        px, py = f(r.get("pixel_x_um")), f(r.get("pixel_y_um"))
        rows.append({
            "mouse": a.mouse, "date": a.date,
            "mesc_file": r.get("file", ""), "mesc_session": u["session"],
            "munit": u["unit"], "scan_type": r.get("scan_type", ""),
            "imaging_timestamp_utc": r.get("timestamp", ""),
            "duration_s": r.get("duration_calc_s", ""),
            "n_t": u["n_t"], "n_slices": (u["slices"] if u["slices"] > 1 else ""),
            "n_y": r.get("n_y", ""), "n_x": r.get("n_x", ""),
            "frame_rate_hz": r.get("frame_rate_hz", ""),
            "pixel_x_um": r.get("pixel_x_um", ""), "pixel_y_um": r.get("pixel_y_um", ""),
            "voxel_z_um": r.get("voxel_z_um", ""),
            "span_x_um": round(px * int(r["n_x"]), 1) if px and r.get("n_x") else "",
            "span_y_um": round(py * int(r["n_y"]), 1) if py and r.get("n_y") else "",
            "imaging_comment": u["comment"],
            "comment_run_number": "" if cr is None else cr,
            "planes_expected": u["fp"], "trigger_andor_edges": t["edges"],
            "trigger_mat": t["mat"],
            "behavior_run": "Run%03d" % t["run"],
            "camera_frames_expected": t["basler"] or "",
            "camera_dir": cam, "camera_frames_found": nfr,
            "camera_state": cstate,
            "match_confidence": conf, "notes": "; ".join(notes),
        })
    for u in lone_u:
        r = u["row"]
        rows.append({c: "" for c in COLUMNS} | {
            "mouse": a.mouse, "date": a.date, "mesc_file": r.get("file", ""),
            "mesc_session": u["session"], "munit": u["unit"],
            "scan_type": r.get("scan_type", ""),
            "imaging_timestamp_utc": r.get("timestamp", ""),
            "duration_s": r.get("duration_calc_s", ""), "n_t": u["n_t"],
            "imaging_comment": u["comment"], "planes_expected": u["fp"],
            "match_confidence": "no_behavior",
            "notes": "NO trigger file for this imaging run",
        })
    for t in lone_t:
        rows.append({c: "" for c in COLUMNS} | {
            "mouse": a.mouse, "date": a.date,
            "behavior_run": "Run%03d" % t["run"], "trigger_mat": t["mat"],
            "trigger_andor_edges": t["edges"],
            "match_confidence": "no_imaging",
            "notes": "NO imaging unit matches this trigger count",
        })

    w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            ww = csv.DictWriter(fh, fieldnames=COLUMNS)
            ww.writeheader()
            ww.writerows(rows)
        print(f"wrote {a.out} ({len(rows)} rows)")
    else:
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
