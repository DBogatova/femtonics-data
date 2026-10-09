#!/usr/bin/env python3
"""Match behavior recordings to .mesc imaging runs and write one master CSV.

The matching key is a hard fingerprint, not a guess:

    imaging  :  n_t * snake_n_slices   (total scanned planes stored in the .mesc)
    behavior :  AndorXylaTrigger rising edges  (from the behavior *_info.txt)

Those two counts are the *same physical event* (one trigger pulse per scanned
plane), so an equal count is an exact pairing. Duplicates within a session
(identical acquisition settings repeated) are resolved by acquisition order:
behavior RunNNN ascending  <->  MUnit timestamp ascending. Those rows are
flagged as `order_within_group` so you can eyeball them.

Secondary sanity check: the behavior `imaging window` length vs the .mesc
`duration_calc_s` (n_t * t_step_ms).

Sessions whose raw triggers recorded no Andor edges (October 2026) cannot use
the fingerprint; their units are paired to behavior_raw/ trigger .mat files by
save time instead (match_confidence = raw_time_paired, see pair_raw_by_time).
Those rows stay unranked until the behavior pipeline has processed them.

Usage
-----
    python code/STEP1_extract/match_behavior_imaging.py [ROOT] [-o OUT.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

# --- sessions whose behavior folder and .mesc folder live under different
#     mouse names, e.g. {"rbp4_phpebach": "rbp4ach"}. Verify any entry you add
#     by checking that the trigger counts still line up (they are exact).
#     Empty = every session is already filed under one mouse folder.
FOLDER_ALIASES = {}

REAL_SCAN_TYPES = ("snake", "ribbon_transverse", "ribbon_longitudinal",
                   "ribbon")

# Femtonics writes the same HDF5 container under either extension: MESc saves
# `.mesc`, but an exported / re-saved file can land as `.hdf`. Globbing only
# `*.mesc` silently loses whole sessions, so always look for both.
CONTAINER_EXTS = (".mesc", ".hdf", ".hdf5")


def find_containers(directory, recursive=False):
    """Every Femtonics raw container in `directory`, any accepted extension."""
    hits = []
    for ext in CONTAINER_EXTS:
        pat = os.path.join(directory, "**", "*" + ext) if recursive \
            else os.path.join(directory, "*" + ext)
        hits += glob.glob(pat, recursive=recursive)
    return sorted(set(hits))
MIN_TIMESERIES_S = 20.0  # raster timeSeries shorter than this = reference snapshot

NEG_STRONG = [
    "bad", "crap", "not good", "subpar", "mediocre", "not the best",
    "not great", "no soma", "missed camera", "too many cells",
]
NEG_MILD = [
    "multiple cells", "2 cells", "two cells", "neighbor cells", "misalignment",
    "no pupil", "cropped pupil", "motion", "not active", "water ran out",
    "meh", "another cell close",
]
POS_STRONG = ["very good"]
POS = ["good", "ok"]
POS_WEAK = ["maybe", "might be useful", "first time"]


# ---------------------------------------------------------------- imaging side
# A session lives at <root>/<mouse>/<MM-DD-YYYY>/... . Anything else at the top
# level (code/, stats/, auto_pipeline/ and other working dirs) is NOT mouse data.
# auto_pipeline/ in particular holds a MIRROR of the data tree, so without this
# guard every mirrored session is scanned a second time as mouse="auto_pipeline",
# date="<real mouse>", and every run in it is paired twice -- 24 duplicate rows
# on this project, which silently broke the one-run-one-partner invariant.
DATE_DIR_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")


def is_session_path(parts):
    """True when parts = (mouse, MM-DD-YYYY, ...) -- a real session location."""
    return len(parts) >= 2 and bool(DATE_DIR_RE.match(parts[1]))


def load_summaries(root):
    """{(mouse, date): {'summary': path, 'mesc': path|None, 'units': [...]}}"""
    sessions = {}
    for summary in glob.glob(os.path.join(root, "**", "*.summary.csv"), recursive=True):
        rel = os.path.relpath(summary, root)
        parts = rel.split(os.sep)
        if not is_session_path(parts):
            continue
        mouse, date = parts[0], parts[1]
        rows = list(csv.DictReader(open(summary)))
        units = []
        seen = set()
        for r in rows:
            st = r.get("scan_type") or ""
            dur = _f(r.get("duration_calc_s")) or _f(r.get("duration_s")) or 0.0
            if st in REAL_SCAN_TYPES:
                pass
            elif st == "timeSeries" and dur >= MIN_TIMESERIES_S:
                pass
            else:
                continue
            key = (r["session"], r["unit"])
            if key in seen:          # dual-detector: one row per channel
                continue
            seen.add(key)
            n_t = int(_f(r.get("n_t")) or 0)
            slices = int(_f(r.get("snake_n_slices")) or 0) or 1
            units.append({
                "session": r["session"],
                "unit": r["unit"],
                "scan_type": st,
                "timestamp": r.get("timestamp", ""),
                "duration_s": dur,
                "comment": (r.get("comment") or "").strip(),
                "n_t": n_t,
                "n_slices": slices if st == "snake" else "",
                "n_y": r.get("n_y", ""),
                "n_x": r.get("n_x", ""),
                "frame_rate_hz": r.get("frame_rate_hz", ""),
                "pixel_x_um": r.get("pixel_x_um", ""),
                "voxel_z_um": r.get("voxel_z_um", ""),
                "fingerprint": n_t * slices,
                "file": r.get("file", ""),
            })
        units.sort(key=lambda u: (u["timestamp"], u["unit"]))
        mescs = find_containers(os.path.dirname(summary))
        sessions[(mouse, date)] = {
            "summary": summary,
            "summary_real": os.path.realpath(summary),
            "mesc": mescs[0] if mescs else None,
            "mesc_real": os.path.realpath(mescs[0]) if mescs else None,
            "owns_mesc": bool(mescs) and not os.path.islink(mescs[0]),
            "raw_dir": os.path.dirname(summary),
            "units": units,
        }
    # sessions with a raw container but no summary at all
    for mesc in find_containers(root, recursive=True):
        rel = os.path.relpath(mesc, root).split(os.sep)
        if len(rel) < 2:
            continue
        key = (rel[0], rel[1])
        if key not in sessions:
            sessions[key] = {"summary": None, "summary_real": None, "mesc": mesc,
                             "mesc_real": os.path.realpath(mesc),
                             "owns_mesc": not os.path.islink(mesc),
                             "raw_dir": os.path.dirname(mesc), "units": []}
    return sessions


# --------------------------------------------------------------- behavior side
INFO_PATTERNS = {
    "imaging_window_s": re.compile(r"imaging window\s*:\s*\[\s*[\d.]+\s+([\d.]+)\]"),
    "t0_s": re.compile(r"t0 time\s*:\s*([\d.]+)"),
    "camera_edges": re.compile(r"camera edges\s*:\s*(\d+)"),
    "tiff_frames": re.compile(r"TIFF frames\s*:\s*(\d+)"),
    "frames_used": re.compile(r"frames used\s*:\s*(\d+)"),
    "behavior_rows": re.compile(r"behavior rows\s*:\s*(\d+)"),
    "alignment_applied": re.compile(r"applied\s*:\s*(\d+)"),
}
ANDOR_RE = re.compile(r"^\s*AndorXylaTrigger\s*:\s*(\d+)\s*:\s*([\d.]+)", re.M)
WARN_HDR = re.compile(r"^Warnings \((\d+)\)", re.M)


def parse_info(path):
    txt = open(path, errors="replace").read()
    out = {"info_path": path}
    for k, rx in INFO_PATTERNS.items():
        m = rx.search(txt)
        out[k] = float(m.group(1)) if m else None
    m = ANDOR_RE.search(txt)
    out["andor_edges"] = int(m.group(1)) if m else None
    m = WARN_HDR.search(txt)
    n_warn = int(m.group(1)) if m else 0
    warns = []
    if n_warn:
        block = txt.split("Warnings (")[1].split("Files written")[0]
        for line in block.splitlines():
            line = line.strip(" -\t")
            if line and not line.startswith(")") and "(none)" not in line:
                if line not in warns and not line[0].isdigit():
                    warns.append(line)
    out["n_warnings"] = n_warn
    out["warnings"] = " | ".join(warns[:n_warn]) if warns else ""
    base = os.path.basename(path)[: -len("_info.txt")]
    out["behavior_base"] = base
    m = re.search(r"Run(\d+)$", base)
    out["run_number"] = int(m.group(1)) if m else None
    d = os.path.dirname(path)
    out["behavior_csv"] = _exists(os.path.join(d, base + "_behavior.csv"))
    out["behavior_mat"] = _exists(os.path.join(d, base + "_behavior.mat"))
    return out


def load_behavior(root):
    sessions = defaultdict(list)
    for info in glob.glob(os.path.join(root, "**", "behavior", "*_info.txt"), recursive=True):
        rel = os.path.relpath(info, root).split(os.sep)
        if len(rel) < 3 or not is_session_path(rel):
            continue
        sessions[(rel[0], rel[1])].append(parse_info(info))
    for v in sessions.values():
        v.sort(key=lambda b: (b["run_number"] is None, b["run_number"], b["behavior_base"]))
    return sessions


def trigger_files(root, mouse, date, base):
    """Run001_t1_trigger.csv / _accel.csv next to the session."""
    m = re.search(r"Run(\d+)$", base)
    if not m:
        return "", ""
    d = os.path.join(root, mouse, date, "trigger")
    return (_exists(os.path.join(d, f"Run{m.group(1)}_t1_trigger.csv")),
            _exists(os.path.join(d, f"Run{m.group(1)}_t1_accel.csv")))


# ------------------------------------------------------------------- matching
COMMENT_RUN_RE = re.compile(r"\brun\s*#?\s*(\d{1,2})\b", re.I)


def comment_run_number(comment):
    m = COMMENT_RUN_RE.search(comment or "")
    return int(m.group(1)) if m else None


def match_session(units, behavior):
    """Pair imaging units to behavior runs.

    Primary key: trigger-pulse count (n_t*slices == AndorXylaTrigger edges).
    Inside a group of runs that share the same count, the tie is broken by
    (a) an explicit run number written in the .mesc comment, else
    (b) acquisition order (behavior RunNNN ascending vs MUnit time ascending).

    Returns (pairs, unmatched_units, unmatched_behavior) where each pair is
    (unit, beh, confidence, group_size, order_alternative_or_"").
    """
    by_fp_u = defaultdict(list)
    for u in units:
        by_fp_u[u["fingerprint"]].append(u)
    by_fp_b = defaultdict(list)
    for b in behavior:
        by_fp_b[b["andor_edges"]].append(b)

    pairs, used_u, used_b = [], set(), set()
    for fp, us in by_fp_u.items():
        bs = list(by_fp_b.get(fp, []))
        if not bs:
            continue
        us = sorted(us, key=lambda u: u["timestamp"])
        unique = len(us) == 1 and len(bs) == 1

        # what pure ordering would have said, for the audit trail
        order_map = {id(u): b["behavior_base"]
                     for u, b in zip(us, bs)}

        free_u, free_b = list(us), list(bs)
        bound = []
        if not unique:
            # (a) honour explicit run numbers written in the comments
            for u in list(free_u):
                n = comment_run_number(u["comment"])
                if n is None:
                    continue
                hit = next((b for b in free_b if b["run_number"] == n), None)
                if hit is not None:
                    bound.append((u, hit, "fingerprint+comment_run"))
                    free_u.remove(u)
                    free_b.remove(hit)
        # (b) the rest by acquisition order
        for u, b in zip(free_u, free_b):
            bound.append((u, b, "fingerprint_exact" if unique
                          else "fingerprint+order"))

        for u, b, conf in bound:
            alt = order_map.get(id(u), "")
            alt = "" if alt == b["behavior_base"] else alt
            pairs.append((u, b, conf, max(len(us), len(bs)), alt))
            used_u.add(id(u))
            used_b.add(id(b))

    rest_u = [u for u in units if id(u) not in used_u]
    rest_b = [b for b in behavior if id(b) not in used_b]

    # fallback: same count left over -> pair by order if durations agree
    if rest_u and rest_b and len(rest_u) == len(rest_b):
        ok = all(abs((b["imaging_window_s"] or -1) - u["duration_s"]) < 1.5
                 for u, b in zip(rest_u, rest_b))
        if ok:
            for u, b in zip(rest_u, rest_b):
                pairs.append((u, b, "duration_order_fallback", len(rest_u), ""))
            rest_u, rest_b = [], []

    pairs.sort(key=lambda p: p[0]["timestamp"])
    return pairs, rest_u, rest_b


# ------------------------------------------- raw trigger pairing (no Andor edges)
# Some sessions (rbp4_155 10-06 / 10-08, rbp4Flp0_008 10-08) recorded ZERO
# AndorXylaTrigger edges, so the plane-count fingerprint above cannot work. For
# those, and only those, a raw trigger .mat is paired to the imaging unit that
# ended just before the .mat was saved: the acquisition script writes the .mat
# RAW_SAVE_LAG_S after the imaging stops (14-15 s on 10-08). A pair also needs the
# trigger record to outlast the imaging by 0..RAW_RECORD_PAD_S. Anything outside
# those windows (e.g. file mtimes lost in a copy) is left unpaired, never guessed.
RAW_SAVE_LAG_S = (0.0, 60.0)
RAW_RECORD_PAD_S = (0.0, 30.0)


def _utc(ts):
    from datetime import datetime, timezone
    try:
        return datetime.strptime(ts.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc).timestamp()
    except (AttributeError, ValueError):
        return None


def load_raw_triggers(session_dir, mouse_date):
    """Raw trigger .mat files of one session, with their trig_*.csv summary.

    -> list of dicts (mouse_date, run_number, mat, saved_utc, record_s, andor_edges,
       camera_edges). Only files listed in a trig_*.csv are returned: that CSV is
       where the Andor edge count comes from."""
    raw = os.path.join(session_dir, "behavior_raw")
    out = []
    for tcsv in sorted(glob.glob(os.path.join(raw, "trig_*.csv"))):
        for r in csv.DictReader(open(tcsv)):
            mat = os.path.join(raw, "trigger", os.path.basename(r.get("file", "")))
            if not os.path.isfile(mat):
                continue
            out.append({
                "mouse_date": mouse_date,
                "run_number": int(_f(r.get("run_number")) or 0) or None,
                "mat": mat,
                "saved_utc": os.path.getmtime(mat),
                "record_s": _f(r.get("record_s")),
                "andor_edges": int(_f(r.get("AndorXylaTrigger_edges")) or 0),
                "camera_edges": _i(_f(r.get("baslerExposureTrigger_edges"))),
            })
    return out


def pair_raw_by_time(units, triggers):
    """{id(unit): trigger} for triggers with zero Andor edges, one-to-one.

    Each trigger takes the unit whose end (timestamp + duration) is closest
    before its save time, inside RAW_SAVE_LAG_S and RAW_RECORD_PAD_S."""
    out, taken = {}, set()
    for t in sorted((t for t in triggers if t["andor_edges"] == 0),
                    key=lambda t: t["saved_utc"]):
        best = None
        for u in units:
            if id(u) in taken:
                continue
            t0 = _utc(u["timestamp"])
            if t0 is None or not u["duration_s"]:
                continue
            lag = t["saved_utc"] - (t0 + u["duration_s"])
            if not (RAW_SAVE_LAG_S[0] <= lag <= RAW_SAVE_LAG_S[1]):
                continue
            if t["record_s"] is not None:
                pad = t["record_s"] - u["duration_s"]
                if not (RAW_RECORD_PAD_S[0] <= pad <= RAW_RECORD_PAD_S[1]):
                    continue
            if best is None or lag < best[1]:
                best = (u, lag)
        if best:
            u, lag = best
            taken.add(id(u))
            out[id(u)] = dict(t, lag_s=round(lag, 1))
    return out


# -------------------------------------------------------------------- quality
def comment_quality(comment):
    """-> (quality, score, notes).  quality in good/ok/caution/bad/unknown."""
    c = (comment or "").lower()
    if not c:
        return "unknown", 0, ""
    score, notes = 0, []
    hedged = any(w in c for w in ("maybe", "might be", "possible", "possibly"))
    for w in POS_STRONG:
        if w in c:
            score += 3
            notes.append("+" + w)
    if not any(w in c for w in POS_STRONG):
        for w in POS:
            if re.search(r"\b" + re.escape(w), c):
                score += 2
                notes.append("+" + w)
                break
    for w in POS_WEAK:
        if w in c:
            score += 1
            notes.append("?" + w)
            break
    neg_strong = [w for w in NEG_STRONG if w in c]
    neg_mild = [w for w in NEG_MILD if w in c]
    score -= 3 * len(neg_strong) + len(neg_mild)
    notes += ["-" + w for w in neg_strong] + ["~" + w for w in neg_mild]
    if hedged:
        notes.append("?hedged")

    if neg_strong:
        q = "bad"
    elif score >= 2 and not neg_mild and not hedged:
        q = "good"
    elif score >= 1:
        q = "ok"
    elif score <= -1:
        q = "caution"
    else:
        q = "unknown"
    return q, score, ";".join(notes)


SALVAGE = ("might be useful", "maybe useful", "but might")


def priority(row):
    """P1 best ... P4 deprioritise. Returns (priority, reason)."""
    if row["mesc_present"] != "yes":
        return "P4", "imaging .mesc file missing"
    if not row["behavior_base"]:
        if row["match_confidence"] in ("raw_behavior_unprocessed", "raw_time_paired"):
            return "P4", "behavior recorded but not yet processed"
        return "P4", "no matching behavior recording"
    if not row["munit"]:
        return "P4", "no matching imaging run"
    if row["imaging_quality"] == "bad":
        if any(s in (row["imaging_comment"] or "").lower() for s in SALVAGE):
            return "P3", "comment marks run as bad but possibly salvageable"
        return "P4", "comment marks run as bad"

    probs = []
    conf = row["match_confidence"]
    if conf == "fingerprint+order":
        probs.append("match by order inside a same-fingerprint group")
    elif conf == "duration_order_fallback":
        probs.append("no trigger-count match; paired by duration+order")
    if row["comment_run_conflict"]:
        probs.append("comment run number disagrees with the pairing")
    if str(row["behavior_n_warnings"]) not in ("", "0"):
        probs.append("behavior warning")
    if row["behavior_frame_loss_pct"] not in ("", None) and float(row["behavior_frame_loss_pct"]) >= 5.0:
        probs.append("large behavior frame loss")
    if row["duration_mismatch_s"] not in ("", None) and abs(float(row["duration_mismatch_s"])) > 1.0:
        probs.append("duration mismatch")
    if row["imaging_quality"] == "caution":
        probs.append("comment caveat")
    if "~" in (row["quality_notes"] or ""):
        probs.append("comment caveat")
    probs = list(dict.fromkeys(probs))

    if not probs:
        if row["imaging_quality"] in ("good", "ok"):
            return "P1", "clean unique match, comment " + row["imaging_quality"]
        return "P2", "clean unique match, no quality comment"
    if row["imaging_quality"] == "good" and len(probs) == 1:
        return "P2", "; ".join(probs)
    if len(probs) == 1 and conf in ("fingerprint_exact", "fingerprint+comment_run"):
        return "P2", "; ".join(probs)
    return "P3", "; ".join(probs)


# ---------------------------------------------------------------------- utils
def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _exists(p):
    return os.path.basename(p) if os.path.exists(p) else ""


_TIF_SHAPE_CACHE = {}


def _tif_shape(path):
    if path not in _TIF_SHAPE_CACHE:
        try:
            import tifffile
            with tifffile.TiffFile(path) as t:
                _TIF_SHAPE_CACHE[path] = tuple(t.series[0].shape)
        except Exception:
            _TIF_SHAPE_CACHE[path] = None
    return _TIF_SHAPE_CACHE[path]


def find_extracted(raw_dir, session_dir, unit, run_number=None, expect=None):
    """Already-extracted 4D TIFF for this unit.

    A candidate must (a) have exactly the unit's (T,Z,Y,X) metadata shape and
    (b) be positively identifiable as this unit -- either the MUnit number is
    in the filename, or it sits in a `run<N>` folder whose N is this run's
    behavior run number. Folder numbering alone is not trusted: some sessions
    number `run<N>` by MUnit index and others by behavior run number.

    Returns (path, other_same_shape_files, suspect_wrong_nz_files).
    """
    n = unit.split("_")[-1]
    cands = []
    for ext in ("tif", "tiff"):
        for pat in (f"*.{ext}", os.path.join("*", f"*.{ext}"),
                    os.path.join("*", "*", f"*.{ext}")):
            for d in (raw_dir, session_dir, os.path.join(session_dir, "preprocessed")):
                cands += glob.glob(os.path.join(d, pat))
    cands = sorted(set(cands))

    def named(p):
        b = os.path.basename(p).lower()
        return (f"munit_{n}_" in b or f"munit{n}_" in b
                or b.endswith(f"munit_{n}.tif") or b.endswith(f"munit_{n}.tiff")
                or b.endswith(f"munit{n}.tif") or b.endswith(f"munit{n}.tiff"))

    def in_run_folder(p):
        if run_number is None:
            return False
        segs = os.path.relpath(p, session_dir).split(os.sep)
        # canonical zero-padded folders (run05) plus legacy unpadded (run5)
        return (f"run{int(run_number):02d}" in segs
                or f"run{int(run_number)}" in segs)

    if expect is None:
        hits = [c for c in cands if named(c)]
        return (os.path.relpath(hits[0], session_dir) if hits else ""), "", ""

    expect = tuple(expect)
    # a ribbon scan has no Z axis, so its stack is stored 3D (T,Y,X)
    expect_sq = tuple(d for d in expect if d != 1)
    total = 1
    for d in expect:
        total *= d
    fits, suspect = [], []
    for c in cands:
        sh = _tif_shape(c)
        if sh is None:
            continue
        if tuple(sh) == expect or tuple(sh) == expect_sq:
            fits.append(c)
        elif len(sh) == 4 and sh[0] * sh[1] * sh[2] * sh[3] == total \
                and sh[2:] == expect[2:] and (named(c) or in_run_folder(c)):
            suspect.append(c)          # right raw frames, wrong --nz reshape

    identified = [c for c in fits if named(c)] or [c for c in fits if in_run_folder(c)]
    identified.sort(key=lambda p: ("_clean" in p, len(p)))
    others = [os.path.relpath(p, session_dir) for p in fits
              if not identified or p != identified[0]]
    best = os.path.relpath(identified[0], session_dir) if identified else ""
    return (best, "; ".join(others),
            "; ".join(os.path.relpath(p, session_dir) for p in suspect))


COLUMNS = [
    "rank", "priority", "priority_reason",
    "mouse", "date", "session_dir",
    "mesc_file", "mesc_present", "mesc_session", "munit", "scan_type",
    "imaging_timestamp_utc", "imaging_duration_s", "n_t", "n_slices", "n_y", "n_x",
    "frame_rate_hz", "pixel_x_um", "voxel_z_um",
    "imaging_comment", "imaging_quality", "quality_notes", "comment_run_number",
    "comment_run_conflict",
    "planes_expected_n_t_x_slices",
    "behavior_base", "behavior_run_number", "behavior_andor_edges",
    "behavior_imaging_window_s", "behavior_t0_s", "behavior_camera_edges",
    "behavior_tiff_frames", "behavior_frames_used", "behavior_frame_loss_pct",
    "behavior_n_warnings",
    "behavior_warnings", "behavior_csv", "behavior_mat",
    "trigger_csv", "accel_csv",
    "match_confidence", "match_group_size", "order_based_alternative",
    "duration_mismatch_s",
    "extracted_tif", "extracted_tif_other_candidates", "extracted_tif_suspect_nz",
    "quality_score", "flags",
    # raw-trigger pairing for sessions with no Andor edges (see pair_raw_by_time)
    "raw_trigger_mat", "raw_trigger_lag_s",
]


# ------------------------------------------------------------ ranked shortlist
RANKED_COLUMNS = [
    "rank", "priority", "mouse", "date", "scan_type",
    "munit", "behavior_run", "imaging_timestamp_utc", "duration_s",
    "n_t", "nz", "n_y", "n_x", "voxel_zyx_um", "volume_rate_hz",
    "imaging_quality", "imaging_comment",
    "match_confidence", "behavior_frame_loss_pct",
    "mesc_path", "extracted_4d_tif", "behavior_csv_path", "trigger_csv_path",
    "notes",
]


def write_ranked(rows, root, out):
    """One row per usable run (imaging AND behavior), ranked best -> worst."""
    path = out if os.path.isabs(out) else os.path.join(root, out)
    ranked = sorted((r for r in rows if r["rank"]), key=lambda r: int(r["rank"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RANKED_COLUMNS)
        w.writeheader()
        for r in ranked:
            sd = r["session_dir"]
            beh_dir = os.path.join(sd, "behavior")
            trg_dir = os.path.join(sd, "trigger")
            vox = "/".join(str(r[k] or "?") for k in
                           ("voxel_z_um", "pixel_x_um", "pixel_x_um"))
            w.writerow({
                "rank": r["rank"],
                "priority": r["priority"],
                "mouse": r["mouse"],
                "date": r["date"],
                "scan_type": r["scan_type"],
                "munit": r["munit"],
                "behavior_run": "Run%03d" % int(r["behavior_run_number"]),
                "imaging_timestamp_utc": r["imaging_timestamp_utc"],
                "duration_s": r["imaging_duration_s"],
                "n_t": r["n_t"], "nz": r["n_slices"],
                "n_y": r["n_y"], "n_x": r["n_x"],
                "voxel_zyx_um": vox,
                "volume_rate_hz": r["frame_rate_hz"],
                "imaging_quality": r["imaging_quality"],
                "imaging_comment": r["imaging_comment"],
                "match_confidence": r["match_confidence"],
                "behavior_frame_loss_pct": r["behavior_frame_loss_pct"],
                "mesc_path": os.path.join(sd, "raw", r["mesc_file"]),
                "extracted_4d_tif": (os.path.join(sd, r["extracted_tif"])
                                     if r["extracted_tif"] else ""),
                "behavior_csv_path": (os.path.join(beh_dir, r["behavior_csv"])
                                      if r["behavior_csv"] else ""),
                "trigger_csv_path": (os.path.join(trg_dir, r["trigger_csv"])
                                     if r["trigger_csv"] else ""),
                "notes": "; ".join(x for x in (r["priority_reason"], r["flags"]) if x),
            })
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("-o", "--out", default="behavior_imaging_master.csv")
    ap.add_argument("--ranked-out", default="ranked_runs.csv",
                    help="short ranked shortlist of the usable runs "
                         "(imaging + behavior), best first")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    imaging = load_summaries(root)
    behavior = load_behavior(root)

    # resolve folder aliases: behavior under mouse A, imaging under mouse B
    alias_used = []
    for (bm, bd) in list(behavior):
        if (bm, bd) in imaging and imaging[(bm, bd)]["units"]:
            continue
        tgt = FOLDER_ALIASES.get(bm)
        if tgt and (tgt, bd) in imaging:
            imaging[(bm, bd)] = imaging[(tgt, bd)]
            alias_used.append((bm, bd, tgt))

    keys = sorted(set(imaging) | set(behavior))

    # ---- pass 1: match every session, so we know which units of a .mesc that
    #      is shared by two mice (one file, two animals) belong to which mouse.
    matched_by_session = {}
    claimed = {}                      # (mesc_real, session, unit) -> "mouse/date"
    shared_real = defaultdict(set)
    for key in keys:
        img = imaging.get(key)
        if img and img.get("mesc_real"):
            shared_real[img["mesc_real"]].add(key)
    shared_real = {p: s for p, s in shared_real.items() if len(s) > 1}

    for key in keys:
        img = imaging.get(key, {"units": []})
        pairs, lone_u, lone_b = match_session(img["units"], behavior.get(key, []))
        matched_by_session[key] = (pairs, lone_u, lone_b)
        real = img.get("mesc_real")
        for u, b, *_ in pairs:
            claimed[(real, u["session"], u["unit"])] = "%s/%s" % key

    # ---- pass 1b: sessions whose raw triggers have no Andor edges. Pair the
    #      leftover units to behavior_raw/ trigger .mat files by save time, and
    #      for a two-mouse .mesc give every leftover unit to the right animal
    #      (paired -> the trigger's mouse; unpaired -> mouse of the nearest
    #      time-paired unit). Units with a processed-behavior match are untouched.
    raw_pair, raw_owner_note = {}, {}
    groups = defaultdict(list)
    for key in keys:
        img = imaging.get(key)
        if img and img.get("mesc_real"):
            groups[img["mesc_real"]].append(key)
    for real, gkeys in groups.items():
        # one copy of the unit list: the folder that owns the .mesc (others hold
        # symlinks and, when they have a summary, a duplicate unit list)
        owner_key = next((k for k in gkeys if imaging[k].get("owns_mesc")), gkeys[0])
        trig = {}
        for key in gkeys:
            if behavior.get(key):
                continue
            for t in load_raw_triggers(os.path.join(root, *key), "%s/%s" % key):
                sig = (os.path.basename(t["mat"]), os.path.getsize(t["mat"]), t["saved_utc"])
                if sig in trig:
                    # same file copied into two mouse folders: animal unknown
                    trig[sig]["mouse_date"] = None
                else:
                    trig[sig] = t
        if not trig:
            continue
        lone = list(matched_by_session[owner_key][1])
        hits = pair_raw_by_time(lone, list(trig.values()))
        if not hits:
            continue
        raw_pair.update(hits)
        if len(gkeys) < 2 or any(h["mouse_date"] is None for h in hits.values()):
            continue                   # single mouse, or trigger folders not split by animal
        paired = [(_utc(u["timestamp"]), hits[id(u)]["mouse_date"])
                  for u in lone if id(u) in hits]
        moved = defaultdict(list)
        for u in lone:
            if id(u) in hits:
                target = hits[id(u)]["mouse_date"]
            else:
                t0 = _utc(u["timestamp"]) or 0
                target = min(paired, key=lambda p: abs(p[0] - t0))[1]
                raw_owner_note[id(u)] = target
            claimed[(real, u["session"], u["unit"])] = target
            tkey = tuple(target.split("/"))
            if tkey != owner_key and tkey in matched_by_session:
                matched_by_session[owner_key][1].remove(u)
                moved[tkey].append(u)
        for key in gkeys:
            if key != owner_key:       # drop the duplicate lists, keep only moved units
                matched_by_session[key][1][:] = moved.get(key, [])
    for key in keys:
        matched_by_session[key][1].sort(key=lambda u: (u["timestamp"], u["unit"]))

    rows, notes = [], []
    for (mouse, date) in keys:
        img = imaging.get((mouse, date), {"units": [], "mesc": None, "mesc_real": None,
                                          "owns_mesc": True, "summary": None,
                                          "raw_dir": ""})
        if mouse in FOLDER_ALIASES.values() and any(a[2] == mouse for a in alias_used):
            continue  # already reported under the behavior folder name
        pairs, lone_u, lone_b = matched_by_session[(mouse, date)]
        bhv = behavior.get((mouse, date), [])
        units = img["units"]
        session_dir = os.path.join(root, mouse, date)
        mesc_present = bool(img["mesc"])
        mesc_name = os.path.basename(img["mesc"]) if img["mesc"] else (
            "MISSING (summary.csv present)" if img["summary"] else "MISSING")
        # raw, not-yet-processed behavior (trigger .mat + camera frames only)
        raw_beh = sorted(glob.glob(os.path.join(session_dir, "behavior_raw",
                                                "trigger", "*.mat")))
        raw_beh_note = ""
        if raw_beh and not bhv:
            raw_beh_note = (f"raw behavior present but NOT processed: "
                            f"{len(raw_beh)} trigger .mat in behavior_raw/ "
                            f"- see behavior_imaging_match.csv for the pairing")
        is_shared = img.get("mesc_real") in shared_real
        others = sorted("%s/%s" % k for k in shared_real.get(img.get("mesc_real"), set())
                        if k != (mouse, date))

        def base_row(u=None, b=None, conf="", gsize="", alt=""):
            q, qscore, qn = comment_quality(u["comment"] if u else "")
            trig, accel = trigger_files(root, mouse, date, b["behavior_base"]) if b else ("", "")
            mism = ""
            if u and b and b["imaging_window_s"] is not None:
                mism = round(b["imaging_window_s"] - u["duration_s"], 3)
            crn = comment_run_number(u["comment"]) if u else None
            conflict = ""
            if crn is not None and b and b["run_number"] is not None:
                conflict = "yes" if crn != b["run_number"] else ""
            r = {c: "" for c in COLUMNS}
            r.update(
                mouse=mouse, date=date,
                session_dir=os.path.relpath(session_dir, root),
                mesc_file=mesc_name, mesc_present="yes" if mesc_present else "no",
                imaging_quality=q, quality_notes=qn, quality_score=qscore,
                comment_run_number="" if crn is None else crn,
                comment_run_conflict=conflict,
                match_confidence=conf, match_group_size=gsize,
                order_based_alternative=alt,
                duration_mismatch_s=mism,
            )
            if u:
                shape = None
                if u["n_t"] and u["n_slices"] and u["n_y"] and u["n_x"]:
                    shape = (u["n_t"], int(float(u["n_slices"])),
                             int(u["n_y"]), int(u["n_x"]))
                tif, tif_alts, tif_bad = find_extracted(
                    img["raw_dir"], session_dir, u["unit"],
                    b["run_number"] if b else None, shape)
                r.update(
                    mesc_session=u["session"], munit=u["unit"],
                    scan_type=u["scan_type"],
                    imaging_timestamp_utc=u["timestamp"],
                    imaging_duration_s=round(u["duration_s"], 2),
                    n_t=u["n_t"], n_slices=u["n_slices"],
                    n_y=u["n_y"], n_x=u["n_x"],
                    frame_rate_hz=u["frame_rate_hz"],
                    pixel_x_um=u["pixel_x_um"], voxel_z_um=u["voxel_z_um"],
                    imaging_comment=u["comment"],
                    planes_expected_n_t_x_slices=u["fingerprint"],
                    extracted_tif=tif,
                    extracted_tif_other_candidates=tif_alts,
                    extracted_tif_suspect_nz=tif_bad,
                )
            if b:
                loss = ""
                if b["camera_edges"] and b["frames_used"] is not None:
                    loss = round(100.0 * (b["camera_edges"] - b["frames_used"])
                                 / b["camera_edges"], 2)
                r.update(
                    behavior_base=b["behavior_base"],
                    behavior_run_number=b["run_number"],
                    behavior_andor_edges=b["andor_edges"],
                    behavior_imaging_window_s=b["imaging_window_s"],
                    behavior_t0_s=b["t0_s"],
                    behavior_camera_edges=_i(b["camera_edges"]),
                    behavior_tiff_frames=_i(b["tiff_frames"]),
                    behavior_frames_used=_i(b["frames_used"]),
                    behavior_frame_loss_pct=loss,
                    behavior_n_warnings=b["n_warnings"],
                    behavior_warnings=b["warnings"],
                    behavior_csv=b["behavior_csv"], behavior_mat=b["behavior_mat"],
                    trigger_csv=trig, accel_csv=accel,
                )
            return r

        for u, b, conf, gsize, alt in pairs:
            r = base_row(u, b, conf, gsize, alt)
            flags = []
            if conf == "fingerprint+order":
                flags.append(f"AMBIGUOUS: {gsize} runs in this session share trigger "
                             f"count {u['fingerprint']}; paired by acquisition order")
            elif conf == "fingerprint+comment_run":
                flags.append(f"{gsize} runs share trigger count {u['fingerprint']}; "
                             f"pairing taken from the run number in the .mesc comment")
            if conf == "duration_order_fallback":
                flags.append("AMBIGUOUS: no trigger-count match, paired by duration+order")
            if alt:
                flags.append(f"pure acquisition order would instead give {alt}")
            if r["comment_run_conflict"]:
                flags.append(f"CHECK: comment says run {r['comment_run_number']} but "
                             f"paired with behavior Run{b['run_number']:03d}")
            if b["n_warnings"]:
                flags.append("behavior warning: " + b["warnings"])
            if r["behavior_frame_loss_pct"] not in ("", None) and float(r["behavior_frame_loss_pct"]) >= 5.0:
                flags.append(f"SEVERE: {r['behavior_frame_loss_pct']}% of camera frames "
                             "missing from the behavior video")
            if r["duration_mismatch_s"] != "" and abs(float(r["duration_mismatch_s"])) > 1.0:
                flags.append("duration mismatch "
                             f"{r['duration_mismatch_s']}s (behavior vs imaging)")
            if "missed camera" in (u["comment"] or "").lower():
                flags.append("CHECK: comment says camera/accelerometer was missed, "
                             "yet a behavior file matches this run's trigger count")
            if r["extracted_tif_suspect_nz"]:
                flags.append("BROKEN EXTRACTION: " + r["extracted_tif_suspect_nz"]
                             + f" has the right frame count but was reshaped with the "
                             f"wrong --nz (expected {r['n_slices']} slices); re-extract it")
            if is_shared:
                flags.append("this .mesc holds two mice; shared with " + ", ".join(others))
            if not mesc_present:
                flags.append("raw .mesc file not on disk")
            r["flags"] = " | ".join(flags)
            r["priority"], r["priority_reason"] = priority(r)
            rows.append(r)

        for u in lone_u:
            # a unit of a two-mouse .mesc that belongs to the *other* mouse
            owner = claimed.get((img.get("mesc_real"), u["session"], u["unit"]))
            if is_shared and owner and owner != "%s/%s" % (mouse, date):
                continue
            if is_shared and not owner and not img.get("owns_mesc"):
                continue
            r = base_row(u, None,
                         "raw_behavior_unprocessed" if raw_beh_note else "no_behavior", "")
            r["flags"] = (raw_beh_note if raw_beh_note
                          else "NO BEHAVIOR FILE for this imaging run")
            rp = raw_pair.get(id(u))
            if rp:
                r["match_confidence"] = "raw_time_paired"
                r["behavior_run_number"] = rp["run_number"]
                r["behavior_andor_edges"] = rp["andor_edges"]
                r["behavior_camera_edges"] = rp["camera_edges"]
                r["raw_trigger_mat"] = os.path.relpath(rp["mat"], root)
                r["raw_trigger_lag_s"] = rp["lag_s"]
                r["flags"] = (f"paired by time to behavior_raw Run{rp['run_number']:03d} "
                              f"(.mat saved {rp['lag_s']} s after imaging ended; the "
                              f"trigger has no Andor edges, so no trigger-count check) "
                              f"- behavior not yet processed")
                crn = r["comment_run_number"]
                if crn != "" and rp["run_number"] and int(crn) != rp["run_number"]:
                    r["comment_run_conflict"] = "yes"
                    r["flags"] += (f" | CHECK: comment says run {crn} but the trigger "
                                   f"saved right after it is Run{rp['run_number']:03d}")
            if is_shared:
                if rp and rp["mouse_date"]:
                    r["flags"] += f" | this .mesc holds two mice; unit belongs to {mouse}"
                elif id(u) in raw_owner_note:
                    r["flags"] += (f" | this .mesc holds two mice; assigned to {mouse} "
                                   "by the nearest time-paired unit (no trigger of its own)")
                else:
                    r["flags"] += (" | this .mesc holds two mice ("
                                   + ", ".join(others) + "); which animal this unit "
                                   "belongs to is unresolved without a behavior match")
            r["priority"], r["priority_reason"] = priority(r)
            rows.append(r)

        for b in lone_b:
            r = base_row(None, b, "no_imaging", "")
            r["flags"] = ("NO IMAGING UNIT for this behavior run"
                          + ("" if mesc_present or img["summary"] else
                             " (whole session .mesc missing)"))
            r["priority"], r["priority_reason"] = "P4", (
                "behavior without imaging" if (mesc_present or img["summary"])
                else "session .mesc missing entirely")
            rows.append(r)

        if raw_beh_note:
            notes.append(f"{mouse}/{date}: {len(raw_beh)} raw trigger .mat in "
                         f"behavior_raw/ awaiting the behavior pipeline.")
        if bhv and not units and not img["summary"]:
            notes.append(f"{mouse}/{date}: {len(bhv)} behavior run(s) but NO .mesc "
                         f"and no summary.csv -> imaging data missing.")
        if units and not bhv and not raw_beh:
            notes.append(f"{mouse}/{date}: {len(units)} imaging run(s) but NO behavior folder.")
        if img["summary"] and not mesc_present:
            notes.append(f"{mouse}/{date}: summary.csv exists but the .mesc itself is missing.")

    order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}

    def sort_key(r):
        paired = bool(r["munit"] and r["behavior_base"])
        nflags = len([f for f in (r["flags"] or "").split(" | ") if f])
        return (not paired, order[r["priority"]], -int(r["quality_score"] or 0),
                nflags, r["mouse"], r["date"],
                r["imaging_timestamp_utc"] or "zzz",
                str(r["behavior_run_number"]))

    rows.sort(key=sort_key)
    n = 0
    for r in rows:
        if r["munit"] and r["behavior_base"]:
            n += 1
            r["rank"] = n

    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    ranked_path = write_ranked(rows, root, args.ranked_out)

    for bm, bd, tgt in alias_used:
        print(f"note: behavior '{bm}/{bd}' matched to imaging in '{tgt}/{bd}' "
              f"(different folder name, same date)")
    for n in notes:
        print("note:", n)
    print(f"\nWritten: {out}  ({len(rows)} rows, every run incl. unmatched)")
    n_ranked = sum(1 for r in rows if r["rank"])
    print(f"Written: {ranked_path}  ({n_ranked} usable runs, best first)")
    counts = defaultdict(int)
    for r in rows:
        counts[r["priority"]] += 1
    for p in ("P1", "P2", "P3", "P4"):
        print(f"  {p}: {counts[p]}")


def _i(v):
    return "" if v is None else int(v)


if __name__ == "__main__":
    sys.exit(main())
