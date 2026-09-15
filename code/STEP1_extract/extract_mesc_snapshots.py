#!/usr/bin/env python
"""
Extract the *single-frame* units (snapshots: camera overviews, 2p reference
images) and the *zStack* anatomy volumes of a Femtonics .mesc/.hdf file as plain
TIFFs (2D for pictures, 3D ZYX for z-stacks).

Time-series units (multi-frame recordings: snake / ribbon / repeated frames) are
skipped -- use extract_mesc.py for those.

Usage:
    python extract_mesc_snapshots.py path/to/file.mesc
    python extract_mesc_snapshots.py path/to/file.mesc --info
    python extract_mesc_snapshots.py path/to/file.mesc --out some/dir
    python extract_mesc_snapshots.py path/to/file.mesc --max-frames 1

Options:
    --info         list what would be extracted, write nothing
    --out DIR      output directory (default: <date folder>/snapshots, i.e. next
                   to raw/ if the file lives in a raw/ folder)
    --max-frames N treat units with <= N frames as pictures (default 1)

Each TIFF is written with the pixel size in its ImageJ metadata (micron), taken
from ReferenceViewportJSON width/height divided by XDim/YDim (z-stacks also get
the z spacing from MinZ/MaxZ), so Fiji shows a correct scale bar. A
`*_snapshots_index.csv` summarises what was written.
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import tifffile


def decode(val):
    if isinstance(val, bytes):
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return val.decode(enc)
            except Exception:
                pass
        return repr(val)
    if isinstance(val, np.generic):
        return val.item()
    return val


def attr(unit, key, default=""):
    return decode(unit.attrs[key]) if key in unit.attrs else default


def parse_json_attr(unit, key):
    raw = attr(unit, key)
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def viewport(unit):
    """(width_um, height_um, z_um) of the imaged field, or (None, None, None)."""
    vps = parse_json_attr(unit, "ReferenceViewportJSON").get("viewports") or []
    if not vps:
        return None, None, None
    v = vps[0]
    transl = v.get("geomTransTransl")
    z = float(transl[2]) if isinstance(transl, list) and len(transl) == 3 else None
    w = float(v["width"]) if "width" in v else None
    h = float(v["height"]) if "height" in v else None
    return w, h, z


def slug(text, maxlen=40):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")[:maxlen]


def unit_order(key):
    m = re.search(r"(\d+)$", key)
    return int(m.group(1)) if m else 0


def posix_to_str(ts):
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ""


def scan_snapshots(filepath, max_frames=1):
    """List single-frame (picture) units, one entry per channel dataset."""
    out = []
    with h5py.File(filepath, "r") as f:
        for sess_key in sorted(f.keys()):
            session = f[sess_key]
            if not isinstance(session, h5py.Group):
                continue
            for unit_key in sorted(session.keys(), key=unit_order):
                unit = session[unit_key]
                if not isinstance(unit, h5py.Group):
                    continue
                chans = [k for k in sorted(unit.keys())
                         if k.startswith("Channel") and hasattr(unit[k], "shape")
                         and len(unit[k].shape) == 3]
                if not chans:
                    continue
                method = str(attr(unit, "MethodTypeDebugString", ""))
                is_zstack = "zstack" in method.lower()
                # a picture = a zStack (anatomy volume) or a unit whose channels hold
                # at most max_frames frames
                if not is_zstack and any(unit[k].shape[0] > max_frames for k in chans):
                    continue
                w_um, h_um, z_um = viewport(unit)
                x_dim = attr(unit, "XDim", None)
                y_dim = attr(unit, "YDim", None)
                # depth relative to the labeling origin (brain surface reference)
                origin = attr(unit, "LabelingOriginTransl", None)
                z_rel = None
                if z_um is not None and origin is not None and len(origin) == 3:
                    z_rel = float(z_um) - float(origin[2])
                min_z, max_z = attr(unit, "MinZ", None), attr(unit, "MaxZ", None)
                for cname in chans:
                    n_f, ny, nx = unit[cname].shape
                    z_step = None
                    if is_zstack and n_f > 1 and min_z is not None and max_z is not None:
                        z_step = (float(max_z) - float(min_z)) / (n_f - 1)
                    out.append({
                        "session": sess_key,
                        "unit": unit_key,
                        "channel": cname,
                        "path": f"{sess_key}/{unit_key}/{cname}",
                        "kind": "zstack" if is_zstack else "picture",
                        "channel_name": str(attr(unit, f"{cname}_Name", cname)),
                        "technology": str(attr(unit, "TechnologyTypeDebugString", "")),
                        "method": method,
                        "comment": str(attr(unit, "Comment", "")).strip(),
                        "n_frames": int(n_f),
                        "n_z": int(n_f) if is_zstack else 1,
                        "n_y": int(ny),
                        "n_x": int(nx),
                        "dtype": str(unit[cname].dtype),
                        "pixel_x_um": round(w_um / float(x_dim), 5) if w_um and x_dim else None,
                        "pixel_y_um": round(h_um / float(y_dim), 5) if h_um and y_dim else None,
                        "z_step_um": round(z_step, 4) if z_step else None,
                        "fov_width_um": round(w_um, 2) if w_um else None,
                        "fov_height_um": round(h_um, 2) if h_um else None,
                        "depth_z_um": round(z_um, 2) if z_um is not None else None,
                        "depth_rel_um": round(z_rel, 2) if z_rel is not None else None,
                        "z_range_um": (f"{min_z} to {max_z}"
                                       if is_zstack and min_z is not None else None),
                        "objective": objective_of(unit),
                        "frame_loop": attr(unit, "FrameLoop", None),
                        "timestamp": posix_to_str(attr(unit, "MeasurementDatePosix", "")),
                        "exposure_ms": attr(unit, "MeasurementLengthInMs", ""),
                    })
    return out


def objective_of(unit):
    mod = parse_json_attr(unit, "ModalityJSON")
    for m in mod.get("modalities", []) if isinstance(mod, dict) else []:
        if m.get("name") == "Objective":
            return m.get("currentState", "")
    return ""


def write_snapshot(filepath, info, out_dir):
    with h5py.File(filepath, "r") as f:
        img = f[info["path"]][:]
    if img.shape[0] == 1:
        img = img[0]  # 2D picture

    parts = [Path(filepath).stem,
             info["session"].replace("MSession_", "S"),
             info["unit"],
             slug(info["channel_name"] or info["channel"])]
    if info["kind"] == "zstack":
        parts.append("zstack")
    if info["comment"]:
        parts.append(slug(info["comment"]))
    out_path = Path(out_dir) / ("_".join(parts) + ".tif")

    px_x, px_y = info["pixel_x_um"], info["pixel_y_um"]
    kwargs = {}
    if px_x and px_y:
        kwargs["resolution"] = (1.0 / px_x, 1.0 / px_y)
    meta = {"axes": "YX" if img.ndim == 2 else "ZYX", "unit": "micron"}
    if info.get("z_step_um"):
        meta["spacing"] = info["z_step_um"]
    tifffile.imwrite(
        str(out_path), img, photometric="minisblack", imagej=True,
        metadata=meta, **kwargs,
    )
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Extract single-frame (picture) units of a .mesc/.hdf as TIFFs")
    ap.add_argument("mesc_file")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--max-frames", type=int, default=1,
                    help="units with <= N frames count as pictures (default 1)")
    ap.add_argument("--info", action="store_true", help="list only, write nothing")
    args = ap.parse_args()

    filepath = Path(args.mesc_file)
    if not filepath.exists():
        sys.exit(f"File not found: {filepath}")

    if args.out:
        out_dir = Path(args.out)
    elif filepath.parent.name.lower() == "raw":
        out_dir = filepath.parent.parent / "snapshots"
    else:
        out_dir = filepath.parent / "snapshots"

    print(f"Scanning: {filepath.name}")
    snaps = scan_snapshots(filepath, max_frames=args.max_frames)
    if not snaps:
        sys.exit("No single-frame units found.")

    print(f"\n{len(snaps)} picture(s) (<= {args.max_frames} frame, plus any zStack):\n")
    header = (f"  {'session':<12}{'unit':<10}{'chan':<9}{'kind':<9}"
              f"{'Z x Y x X':<18}{'px_um':<9}{'z_step':<8}{'depth_um':<11}comment")
    print(header)
    for s in snaps:
        px = f"{s['pixel_x_um']:.4f}" if s["pixel_x_um"] else "?"
        dz = f"{s['depth_rel_um']:.1f}" if s["depth_rel_um"] is not None else "?"
        zs = f"{s['z_step_um']:.2f}" if s["z_step_um"] else "-"
        dims = (f"{s['n_z']}x{s['n_y']}x{s['n_x']}" if s["n_z"] > 1
                else f"{s['n_y']}x{s['n_x']}")
        print(f"  {s['session']:<12}{s['unit']:<10}{s['channel_name'][:8]:<9}"
              f"{s['kind']:<9}{dims:<18}{px:<9}{zs:<8}{dz:<11}{s['comment']}")

    if args.info:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in snaps:
        p = write_snapshot(filepath, s, out_dir)
        row = {k: v for k, v in s.items() if k != "path"}
        row["tif"] = p.name
        rows.append(row)
        print(f"  ok {s['session']}/{s['unit']}/{s['channel']} -> {p.name}")

    index = out_dir / f"{filepath.stem}_snapshots_index.csv"
    with open(index, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} TIFF(s) to {out_dir}")
    print(f"Index: {index}")


if __name__ == "__main__":
    main()
