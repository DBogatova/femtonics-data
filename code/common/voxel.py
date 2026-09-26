"""One source of truth for voxel size (Z, Y, X) in micrometres.

Every script that needs physical distances should call `resolve_voxel(stack_path,
cli_value)`. Resolution order:

  1. explicit --voxel on the command line (always wins, printed as such)
  2. the run's own acquisition metadata: behavior_imaging_master.csv row whose
     run folder matches the stack (columns pixel_x_um / voxel_z_um, parsed from
     the .mesc MultiROI protocol)
  3. a *.summary.csv next to the raw .mesc for that session
  4. FAIL loudly. There is deliberately no hard-coded default: the previous
     per-script defaults ranged from 0.8 to 3.9 um in Z, and a wrong value
     silently changes every anatomical distance and skeleton split.

`--voxel auto` / omitting --voxel means "use metadata".
"""
from __future__ import annotations
import csv, re, sys
from pathlib import Path

PROJECT_MARKER = "behavior_imaging_master.csv"


def _project_root(p: Path) -> Path | None:
    for d in [p] + list(p.parents):
        if (d / PROJECT_MARKER).exists():
            return d
    return None


def _run_ident(stack_path: Path):
    """(mouse, MM-DD-YYYY, run_number) from <mouse>/<date>/.../run<NN>/file."""
    parts = stack_path.resolve().parts
    run_dir = next((q for q in reversed(parts) if re.fullmatch(r"run\d+", q)), None)
    date = next((q for q in parts if re.fullmatch(r"\d{2}-\d{2}-\d{4}", q)), None)
    if run_dir is None or date is None:
        return None
    mouse = parts[parts.index(date) - 1]
    return mouse, date, int(run_dir[3:])


def _from_master(stack_path: Path):
    root = _project_root(stack_path.resolve().parent)
    ident = _run_ident(stack_path)
    if root is None or ident is None:
        return None
    mouse, date, run = ident
    mm, dd, yyyy = date.split("-")
    date_short = f"{yyyy[2:]}-{mm}-{dd}"                 # filenames use yy-mm-dd
    with open(root / PROJECT_MARKER, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("mouse", "") != mouse:
                continue
            if date_short not in (row.get("date", ""), row.get("behavior_base", "")) \
                    and date not in row.get("date", ""):
                continue
            m = re.search(r"(\d+)", str(row.get("behavior_run_number") or row.get("behavior_run") or ""))
            if not m or int(m.group(1)) != run:
                continue
            try:
                xy = float(row["pixel_x_um"]); z = float(row["voxel_z_um"])
            except (KeyError, ValueError):
                return None
            if xy > 0 and z > 0:
                return (z, xy, xy), f"{PROJECT_MARKER} row {row.get('behavior_base', mouse)}"
    return None


def _from_summary(stack_path: Path):
    ident = _run_ident(stack_path)
    if ident is None:
        return None
    parts = stack_path.resolve().parts
    date_idx = parts.index(ident[1])
    session = Path(*parts[: date_idx + 1])
    for csvp in sorted((session / "raw").glob("*.summary.csv")) if (session / "raw").exists() else []:
        with open(csvp, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("snake_n_slices", "").strip()]
        vals = {(float(r["voxel_z_um"]), float(r["pixel_x_um"])) for r in rows
                if r.get("voxel_z_um") and r.get("pixel_x_um")}
        if len(vals) == 1:                               # unambiguous for the session
            z, xy = vals.pop()
            return (z, xy, xy), csvp.name
    return None


def resolve_voxel(stack_path, cli_value=None, *, quiet=False):
    """Return (z, y, x) in um. Raises SystemExit if nothing authoritative is found."""
    stack_path = Path(stack_path)
    if cli_value is not None and not (isinstance(cli_value, (list, tuple)) and
                                      len(cli_value) == 1 and str(cli_value[0]).lower() == "auto"):
        vox = tuple(float(v) for v in cli_value)
        src = "command line"
    else:
        hit = _from_master(stack_path) or _from_summary(stack_path)
        if hit is None:
            sys.exit(f"[voxel] cannot determine voxel size for {stack_path.name}: no matching row in "
                     f"{PROJECT_MARKER} and no unambiguous *.summary.csv in the session's raw/. "
                     f"Pass --voxel Z Y X explicitly (from summarize_mesc.py output).")
        vox, src = hit
    if not quiet:
        print(f"[voxel] Z Y X = {vox[0]:.3f} {vox[1]:.3f} {vox[2]:.3f} um  (source: {src})")
    return vox


def add_voxel_arg(parser):
    """Standard --voxel argument: omit or 'auto' for metadata lookup."""
    parser.add_argument("--voxel", nargs="+", default=None, metavar="Z_Y_X",
                        help="voxel size in um as Z Y X; omit (or 'auto') to read it from the "
                             "run's acquisition metadata. No hard-coded default.")
    return parser


if __name__ == "__main__":                              # quick manual check
    for p in sys.argv[1:]:
        resolve_voxel(p)
