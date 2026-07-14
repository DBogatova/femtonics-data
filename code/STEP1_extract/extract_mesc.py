#!/usr/bin/env python
"""
Extract 4D time-series from Femtonics .mesc files.

# Activate the venv
  source .venv311/bin/activate

Usage:
    python extract_mesc.py path/to/file.mesc
    python extract_mesc.py path/to/file.mesc --only-good
    python extract_mesc.py path/to/file.mesc --nz 14
    python extract_mesc.py path/to/file.mesc --out /path/to/output/

Options:
    --only-good   Only extract MUnits with "good" in their Comment attribute
    --nz N        Number of Z-planes (default: 14)
    --out DIR     Output directory (default: same folder as .mesc file)
    --info        Just print file structure, don't extract
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile


def scan_mesc(filepath, n_z=14):
    """Scan .mesc file and return info about all time-series MUnits."""
    units = []
    with h5py.File(filepath, "r") as f:
        for sess_key in sorted(f.keys()):
            session = f[sess_key]
            if not isinstance(session, h5py.Group):
                continue
            for unit_key in sorted(session.keys(), key=lambda x: int(x.split("_")[1])):
                unit = session[unit_key]
                if "Channel_0" not in unit:
                    continue
                ds = unit["Channel_0"]
                if len(ds.shape) != 3 or ds.shape[1] > 50 or ds.shape[0] < 1000:
                    continue
                if ds.shape[0] % n_z != 0:
                    continue
                comment = unit.attrs.get("Comment", "")
                if isinstance(comment, bytes):
                    comment = comment.decode("utf-8", errors="replace")
                T_raw, Y, X = ds.shape
                n_t = T_raw // n_z
                units.append({
                    "session": sess_key,
                    "unit": unit_key,
                    "path": f"{sess_key}/{unit_key}/Channel_0",
                    "raw_shape": ds.shape,
                    "shape_4d": (n_t, n_z, Y, X),
                    "comment": comment.strip(),
                })
    return units


def extract_unit(filepath, unit_info, out_dir, n_z=14):
    """Extract a single MUnit to a 4D TIFF."""
    with h5py.File(filepath, "r") as f:
        data = f[unit_info["path"]][:]

    n_t, nz, Y, X = unit_info["shape_4d"]
    stack_4d = data.reshape(n_t, nz, Y, X)

    stem = Path(filepath).stem
    unit_name = unit_info["unit"]
    comment_tag = f"_{unit_info['comment'].replace(' ', '_')}" if unit_info["comment"] else ""
    out_path = Path(out_dir) / f"{stem}_{unit_name}{comment_tag}_4D.tif"

    tifffile.imwrite(str(out_path), stack_4d, photometric="minisblack",
                     metadata={"axes": "TZYX"})
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract 4D time-series from Femtonics .mesc files")
    parser.add_argument("mesc_file", help="Path to .mesc file")
    parser.add_argument("--only-good", action="store_true", help="Only extract units marked 'good'")
    parser.add_argument("--nz", type=int, default=14, help="Number of Z-planes (default: 14)")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--info", action="store_true", help="Just print structure, don't extract")
    args = parser.parse_args()

    filepath = Path(args.mesc_file)
    if not filepath.exists():
        sys.exit(f"File not found: {filepath}")

    out_dir = Path(args.out) if args.out else filepath.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {filepath.name}")
    units = scan_mesc(filepath, n_z=args.nz)

    if not units:
        sys.exit("No time-series MUnits found.")

    # Display summary
    print(f"\nFound {len(units)} time-series recordings (Z={args.nz}):\n")
    print(f"  {'Unit':<12} {'Shape (T,Z,Y,X)':<25} {'Comment'}")
    print(f"  {'----':<12} {'---------------':<25} {'-------'}")
    for u in units:
        marker = " ★" if "good" in u["comment"].lower() else ""
        print(f"  {u['unit']:<12} {str(u['shape_4d']):<25} {u['comment']}{marker}")

    if args.info:
        return

    # Filter
    to_extract = units
    if args.only_good:
        to_extract = [u for u in units if "good" in u["comment"].lower()]
        if not to_extract:
            sys.exit("\nNo units marked 'good'. Run without --only-good to extract all.")
        print(f"\nExtracting {len(to_extract)} 'good' units...")
    else:
        print(f"\nExtracting all {len(to_extract)} units...")

    # Extract
    for u in to_extract:
        out_path = extract_unit(filepath, u, out_dir, n_z=args.nz)
        print(f"  ✓ {u['unit']} -> {out_path.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
