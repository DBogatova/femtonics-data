#!/usr/bin/env python3
"""Ignore named regions in figures and statistics without touching the mask or regions.

The list lives in its own small file next to the regions, <stem>_ignore.json:
    {"ignore": ["branch2"], "reason": "...", "updated": "..."}
so re-saving masks or regions never loses it, and deleting nothing is ever needed:
clearing the list brings the region back everywhere.

Honored by: coherence_with_behavior.py (figures), run_metrics.py, coupling_phenotype.py,
behavior_coupling.py and STEP9_auto/paper_stats.py. Changing the list makes their
outputs stale, so the next "Build figure" / "Statistics" rebuilds them.

Command line (also: `femto ignore ...`):
    python code/common/regions.py RUN_DIR_OR_STACK                 # show names + ignore list
    python code/common/regions.py RUN_DIR_OR_STACK branch2 [more] [--reason TEXT]
    python code/common/regions.py RUN_DIR_OR_STACK --clear         # ignore nothing
    python code/common/regions.py RUN_DIR_OR_STACK --keep branch2  # un-ignore one
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEG_SUFFIX = "_segments_final.tif"
IGNORE_SUFFIX = "_ignore.json"


def ignore_path(seg_p: Path) -> Path:
    """<run_dir>/<stem>_ignore.json for <run_dir>/<stem>_segments_final.tif."""
    seg_p = Path(seg_p)
    return seg_p.with_name(seg_p.name.replace(SEG_SUFFIX, IGNORE_SUFFIX))


def ignored_names(seg_p: Path) -> list[str]:
    p = ignore_path(seg_p)
    if not p.exists():
        return []
    try:
        return [str(n) for n in json.loads(p.read_text()).get("ignore", [])]
    except Exception as e:                                   # unreadable file must not be silent
        print(f"WARNING: could not read {p.name} ({e}); ignoring nothing", file=sys.stderr)
        return []


def apply_ignore(seg: np.ndarray, names: dict, seg_p: Path):
    """Zero the labels whose name is in the ignore list.
    Returns (seg_copy_or_same, ignored_names_found, ignored_labels)."""
    ign = set(n.lower() for n in ignored_names(seg_p))
    if not ign:
        return seg, [], []
    labs = [int(l) for l, n in names.items() if str(n).lower() in ign]
    unknown = ign - {str(names[l]).lower() for l in labs}
    if unknown:
        print(f"WARNING: {ignore_path(seg_p).name} lists {sorted(unknown)} but this run has no such region",
              file=sys.stderr)
    if not labs:
        return seg, [], []
    seg = seg.copy()
    seg[np.isin(seg, labs)] = 0
    return seg, [names[l] for l in labs], labs


def newest_input_mtime(seg_p: Path) -> float:
    """Staleness: outputs must be newer than the regions AND the ignore list."""
    t = Path(seg_p).stat().st_mtime
    p = ignore_path(seg_p)
    return max(t, p.stat().st_mtime) if p.exists() else t


def _names_from_json(seg_p: Path) -> dict:
    js = Path(seg_p).with_suffix(".json")
    if not js.exists():
        return {}
    j = json.loads(js.read_text())
    if isinstance(j.get("segment_names"), dict):
        return {int(k): str(v) for k, v in j["segment_names"].items()}
    last = {}
    for w in j.get("wrap_clicks", []):
        last[int(w["label"])] = w.get("name") or f"seg{w['label']}"
    return {i + 1: last[k] for i, k in enumerate(sorted(last))}


def _seg_path(target: str) -> Path:
    t = Path(target)
    if t.is_dir():
        hits = sorted(p for p in t.glob(f"*{SEG_SUFFIX}"))
        if len(hits) != 1:
            sys.exit(f"expected one *{SEG_SUFFIX} in {t}, found {len(hits)}")
        return hits[0]
    if t.name.endswith(SEG_SUFFIX):
        return t
    if t.name.endswith(".tif"):
        return t.with_name(t.name[:-4] + SEG_SUFFIX)
    sys.exit(f"not a run directory or stack: {t}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="run directory, runNN_clean.tif, or its _segments_final.tif")
    ap.add_argument("names", nargs="*", help="region names to ignore (added to the list)")
    ap.add_argument("--keep", nargs="+", default=[], metavar="NAME", help="remove these from the list")
    ap.add_argument("--clear", action="store_true", help="ignore nothing")
    ap.add_argument("--reason", default=None, help="why (stored in the file and shown in reports)")
    a = ap.parse_args(argv)

    seg_p = _seg_path(a.target)
    if not seg_p.exists():
        sys.exit(f"no regions yet: {seg_p}")
    names = _names_from_json(seg_p)
    have = {n.lower(): n for n in names.values()}
    cur = ignored_names(seg_p)
    p = ignore_path(seg_p)

    if a.names or a.keep or a.clear:
        bad = [n for n in a.names + a.keep if n.lower() not in have]
        if bad:
            sys.exit(f"unknown region name(s) {bad}; this run has {sorted(names.values())}")
        new = [] if a.clear else list(cur)
        for n in a.names:
            if have[n.lower()] not in new:
                new.append(have[n.lower()])
        new = [n for n in new if n.lower() not in {k.lower() for k in a.keep}]
        doc = json.loads(p.read_text()) if p.exists() else {}
        doc.update({"ignore": new, "updated": datetime.now(timezone.utc).isoformat(),
                    "regions_file": seg_p.name})
        if a.reason is not None:
            doc["reason"] = a.reason
        p.write_text(json.dumps(doc, indent=2))
        cur = new
        print(f"saved {p.name}")
    print(f"regions: {', '.join(names[l] for l in sorted(names))}")
    print(f"ignored in figures and statistics: {', '.join(cur) if cur else '(none)'}")
    if cur:
        print("rebuild with 'Build figure + movies' / 'Statistics' (outputs are now out of date)")


if __name__ == "__main__":
    main()
