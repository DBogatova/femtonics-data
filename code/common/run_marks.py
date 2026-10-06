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

QUALITY RATING (separate from the marks above; also `femto rate ...`)
A processed, analyzable run can be rated 'very good', 'good' or 'questionable'. A rating
never sets a run aside: rated runs stay in every statistic. cohort_stats / paper_stats
print the rating per run and repeat their main tests (a) without 'questionable' runs and
(b) on 'very good' runs only, next to the main result. Stored in its own table,
<project root>/run_quality.csv  (behavior_base,rating,reason,updated), so run_marks.csv
and everything that reads it are unchanged.
    python code/common/run_marks.py rate                               # list rated runs
    python code/common/run_marks.py rate RUN very_good --reason "clean, big events"
    python code/common/run_marks.py rate RUN questionable --reason "drift at the end"
    python code/common/run_marks.py rate RUN clear
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


# ---------------------------------------------------------------- quality rating
QUALITY_CSV = PROJECT / "run_quality.csv"
QUALITY_FIELDS = ["behavior_base", "rating", "reason", "updated"]
RATINGS = ("very_good", "good", "questionable")             # best -> worst
RATING_LABELS = {"very_good": "very good", "good": "good", "questionable": "questionable"}
RATING_ALIASES = {"very_good": "very_good", "very-good": "very_good", "verygood": "very_good",
                  "very good": "very_good", "vg": "very_good", "good": "good",
                  "questionable": "questionable", "q": "questionable",
                  "clear": None, "none": None, "": None}


def normalize_rating(rating):
    """'very good' / 'very-good' / 'VG' -> 'very_good'; 'clear'/'none'/None -> None.
    Raises ValueError for anything else."""
    if rating is None:
        return None
    key = str(rating).strip().lower()
    if key not in RATING_ALIASES:
        raise ValueError(f"unknown rating {rating!r}; use one of {', '.join(RATINGS)} or clear")
    return RATING_ALIASES[key]


def load_quality() -> dict:
    """{behavior_base: row} for every rated run (rows with an unknown rating are ignored)."""
    p = QUALITY_CSV
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {r["behavior_base"]: r for r in csv.DictReader(f) if r.get("rating") in RATINGS}


def quality_of(behavior_base: str):
    """(rating, reason) or (None, '')."""
    r = load_quality().get(behavior_base or "")
    return (r["rating"], r.get("reason", "")) if r else (None, "")


def get_quality(behavior_base: str):
    """'very_good' | 'good' | 'questionable' | None. Never affects is_set_aside()."""
    return quality_of(behavior_base)[0]


def rating_label(rating) -> str:
    """'very_good' -> 'very good'; None -> ''."""
    return RATING_LABELS.get(rating or "", "")


def set_quality(behavior_base: str, rating, reason: str = ""):
    """Rate a run (very_good | good | questionable) or clear its rating (None / 'clear')."""
    rating = normalize_rating(rating)
    if not behavior_base:
        raise ValueError("behavior_base is required")
    rows = load_quality()
    if rating is None:
        rows.pop(behavior_base, None)
    else:
        rows[behavior_base] = {"behavior_base": behavior_base, "rating": rating, "reason": reason,
                               "updated": datetime.now().isoformat(timespec="seconds")}
    p = QUALITY_CSV
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUALITY_FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in QUALITY_FIELDS})
    tmp.replace(p)                                       # atomic: a reader never sees half a file
    return rating


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


def main_rate(argv=None):
    """`run_marks.py rate [RUN RATING] [--reason TEXT]` (also `femto rate ...`)."""
    ap = argparse.ArgumentParser(prog="femto rate", description="Rate a processed run: "
                                 "very_good | good | questionable | clear. A rating never sets a run aside.")
    ap.add_argument("run", nargs="?", help="behavior_base or run folder")
    ap.add_argument("rating", nargs="?", help="very_good | good | questionable | clear")
    ap.add_argument("--reason", default="")
    a = ap.parse_args(argv)
    if a.run and a.rating:
        try:
            rating = normalize_rating(a.rating)
        except ValueError as e:
            ap.error(str(e))
        base = resolve_base(a.run)
        set_quality(base, rating, a.reason)
        m, _ = mark_of(base)
        print(f"{base}: " + (f"rated {rating_label(rating)}" if rating else "rating cleared")
              + (f" - {a.reason}" if a.reason and rating else "")
              + (f"  (note: this run is also marked '{m}', so it stays out of the statistics)" if m else ""))
        return
    if a.run:
        base = resolve_base(a.run); q, why = quality_of(base)
        print(f"{base}: {rating_label(q) or 'not rated'}" + (f" - {why}" if why else ""))
        return
    rows = load_quality()
    print(f"{len(rows)} rated run(s) in {QUALITY_CSV.name}" + (":" if rows else ""))
    for k, r in sorted(rows.items(), key=lambda kv: (RATINGS.index(kv[1]["rating"]), kv[0])):
        print(f"  {rating_label(r['rating']):12s} {k}  {r.get('reason', '')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rate":
        main_rate(sys.argv[2:])
    else:
        main()
