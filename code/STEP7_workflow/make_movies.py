#!/usr/bin/env python
"""make_movies.py - build the 3D rotation movies for a run from its final regions.

Three kinds (any subset, default all):
  structure   static turntable: the cell rotating once around its long axis
  time        activity over time in a tilted 3D view (one slow turn)
  dual        stacked: structure on top, live activity below, spinning in sync

Input : <run_dir>/<stem>.tif (cleaned 4D stack) + <stem>_segments_final.tif
Output: <run_dir>/<runNN>_final_3d_{structure,time,dual}.mp4

A movie is rebuilt only when it is missing or older than the regions it shows
(re-picking regions makes the old movies stale); --force rebuilds regardless. Only
these generated *_final_3d_* files are ever replaced - nothing else is touched.
The three kinds render in parallel.

  python code/STEP7_workflow/make_movies.py --run rbp4_140_phpeb_26-06-18_Run003
  python code/STEP7_workflow/make_movies.py --run <base> --kinds dual
  python code/STEP7_workflow/make_movies.py --stack <dir>/run03_clean.tif --kinds structure time
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
from common.display_mask import options_match, write_options   # noqa: E402
ROOT = HERE.parents[1]
MOVIE = ROOT / "code/STEP6_movie/segment_3d_movie.py"
KINDS = ("structure", "time", "dual")

# rendering settings (the run05 movies Daria approved on 2026-09-28)
ARGS = {
    "structure": ["--fps", "10", "--frames", "120", "--rotations", "1", "--alpha", "0.55", "--scale", "3"],
    "time":      ["--time", "--angle", "35", "--fps", "12", "--time-subsample", "2", "--alpha", "0.45", "--scale", "3"],
    "dual":      ["--dual", "--rotations", "4", "--fps", "12", "--time-subsample", "2", "--alpha", "0.5", "--scale", "3"],
}


def resolve_stack(run: str | None, stack: str | None) -> Path:
    if stack:
        return Path(stack)
    sys.path.insert(0, str(HERE))
    from femto_status import build_status
    for r in build_status(ROOT):
        if r.get("behavior_base") == run:
            if not r.get("run_dir") or not r.get("stem"):
                sys.exit(f"{run}: no local stack (stage {r.get('stage')})")
            return ROOT / r["run_dir"] / f"{r['stem']}.tif"
    sys.exit(f"unknown run {run!r} (use the behavior_base, e.g. rbp4_140_phpeb_26-06-18_Run003)")


def outputs_for(stack: Path) -> dict:
    stem = stack.stem                                     # runNN_clean
    short = stem[:-len("_clean")] if stem.endswith("_clean") else stem
    return {k: stack.parent / f"{short}_final_3d_{k}.mp4" for k in KINDS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="behavior_base, e.g. rbp4_140_phpeb_26-06-18_Run003")
    g.add_argument("--stack", help="path to <runNN>_clean.tif")
    ap.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    ap.add_argument("--mask", dest="mask", action="store_true", default=False,
                    help="black background outside the cell and other cells blacked out "
                         "(display only; default: the original, unmasked recording)")
    ap.add_argument("--no-mask", dest="mask", action="store_false", help="original look (default)")
    ap.add_argument("--hide-other", dest="hide_other", action="store_true", default=True,
                    help="fill cells marked 'other cell' with nearby background flicker (default)")
    ap.add_argument("--show-other", dest="hide_other", action="store_false",
                    help="show other cells as recorded")
    ap.add_argument("--edge-um", type=float, default=2.0, help="soft edge of the display mask (um)")
    args = ap.parse_args(argv)

    stack = resolve_stack(args.run, args.stack)
    labels = stack.parent / f"{stack.stem}_segments_final.tif"
    if not stack.exists():
        sys.exit(f"missing stack {stack}")
    if not labels.exists():
        sys.exit(f"no regions yet: {labels.name} - pick regions first (wrap_segments_napari.py)")

    outs = outputs_for(stack)
    # a movie is stale if the regions, the reviewed mask or the exclusion changed after it
    deps = [labels] + [stack.parent / f"{stack.stem}{suf}" for suf in
                       ("_autoseg_labelmap_reviewed.tif", "_exclude_labelmap.tif")]
    newest = max(d.stat().st_mtime for d in deps if d.exists())
    mask_args = ((["--mask"] if args.mask else ["--no-mask"]) + ["--edge-um", f"{args.edge_um:g}"]
                 + (["--hide-other"] if args.hide_other else ["--show-other"]))
    todo, procs = [], []
    for k in args.kinds:
        o = outs[k]
        if (o.exists() and o.stat().st_mtime >= newest and not args.force
                and options_match(o, args.mask, args.edge_um, args.hide_other)):
            print(f"  {k:9s} up to date: {o.name}")
            continue
        todo.append(k)
    py = sys.executable
    for k in todo:
        cmd = [py, str(MOVIE), str(stack), str(labels), *ARGS[k], *mask_args, "--out", str(outs[k])]
        print(f"  {k:9s} rendering -> {outs[k].name}", flush=True)
        procs.append((k, subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True)))
    ok = True
    for k, p in procs:
        out, _ = p.communicate()
        last = [l for l in out.splitlines() if l.startswith("saved")]
        if p.returncode == 0 and outs[k].exists():
            write_options(outs[k], args.mask, args.edge_um, args.hide_other)
            print(f"  {k:9s} OK  {outs[k].name}  ({outs[k].stat().st_size / 1e6:.1f} MB)  {last[-1] if last else ''}")
        else:
            ok = False
            print(f"  {k:9s} FAILED (exit {p.returncode})\n" + "\n".join(out.splitlines()[-8:]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
