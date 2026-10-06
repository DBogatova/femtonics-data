#!/usr/bin/env python3
"""Mark a whole run as EXCLUDED (never analyze) or REVISIT (set aside, re-analyze later).

One table for the whole project, <project root>/run_marks.csv
    behavior_base,mark,reason,updated
It always lives in the real project root (also when FEMTO_ROOT points at the automatic
mirror), so one decision applies to your analysis and to the automatic pipeline alike.

A marked run keeps all its files. It is shown greyed in the control panel with the
reason, its automatic steps are not run, and it is left out of every statistic
(run_metrics, coupling_phenotype, behavior_coupling, cohort_stats, paper_stats).
Clearing the mark brings it back.

Command line (also: `femto mark ...`):
    python code/common/run_marks.py                                   # list marked runs
    python code/common/run_marks.py RUN exclude --reason "multiple cells, out of frame"
    python code/common/run_marks.py RUN revisit --reason "try again with a tighter mask"
    python code/common/run_marks.py RUN clear
RUN = behavior_base (e.g. rbp4_141_phpeb_26-06-17_Run001) or the run folder.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
MARKS_CSV = PROJECT / "run_marks.csv"
FIELDS = ["behavior_base", "mark", "reason", "updated"]
MARKS = ("excluded", "revisit")
ALIASES = {"exclude": "excluded", "excluded": "excluded", "revisit": "revisit", "later": "revisit",
           "clear": None, "ok": None, "none": None}


def load_marks() -> dict:
    if not MARKS_CSV.exists():
        return {}
    with open(MARKS_CSV, newline="") as f:
        return {r["behavior_base"]: r for r in csv.DictReader(f) if r.get("mark") in MARKS}


def mark_of(behavior_base: str):
    """(mark, reason) or (None, '')."""
    r = load_marks().get(behavior_base or "")
    return (r["mark"], r.get("reason", "")) if r else (None, "")


def is_set_aside(behavior_base: str) -> bool:
    return mark_of(behavior_base)[0] is not None


def set_mark(behavior_base: str, mark, reason: str = ""):
    marks = load_marks()
    if mark is None:
        marks.pop(behavior_base, None)
    else:
        marks[behavior_base] = {"behavior_base": behavior_base, "mark": mark, "reason": reason,
                                "updated": datetime.now().isoformat(timespec="seconds")}
    with open(MARKS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(marks):
            w.writerow({c: marks[k].get(c, "") for c in FIELDS})


def resolve_base(target: str) -> str:
    """behavior_base from a behavior_base or a run folder (via femto_status)."""
    if "_Run" in target and not Path(target).exists():
        return target
    sys.path.insert(0, str(PROJECT / "code/STEP7_workflow"))
    import femto_status as fs
    want = Path(target).resolve()
    for r in fs.build_status(PROJECT):
        if r.get("behavior_base") == target:
            return target
        rd = r.get("run_dir")
        if rd and (PROJECT / rd).resolve() == want:
            return r["behavior_base"]
    sys.exit(f"could not find a run for {target!r} (use the behavior_base, e.g. rbp4_141_phpeb_26-06-17_Run001)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="behavior_base or run folder")
    ap.add_argument("mark", nargs="?", choices=sorted(ALIASES), help="exclude | revisit | clear")
    ap.add_argument("--reason", default="")
    a = ap.parse_args(argv)
    if a.run and a.mark:
        base = resolve_base(a.run)
        set_mark(base, ALIASES[a.mark], a.reason)
        print(f"{base}: {ALIASES[a.mark] or 'cleared (analyzed normally)'}" + (f" - {a.reason}" if a.reason else ""))
    elif a.run:
        base = resolve_base(a.run); m, why = mark_of(base)
        print(f"{base}: {m or 'not marked'}" + (f" - {why}" if why else ""))
        return
    marks = load_marks()
    print(f"{len(marks)} marked run(s) in {MARKS_CSV.name}" + (":" if marks else ""))
    for k, r in sorted(marks.items()):
        print(f"  {r['mark']:8s} {k}  {r.get('reason', '')}")


if __name__ == "__main__":
    main()
