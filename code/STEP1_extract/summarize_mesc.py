"""
Produce a per-run summary of a Femtonics .mesc file.

Writes <file>.summary.csv next to the .mesc (works for .hdf-named files too) and
prints a recap of the real recordings (single-frame snapshots excluded).

Key columns:
    scan_type        snake / ribbon_transverse / timeseries / snapshot / camera_*
    frame_rate_hz    stored timepoints per second (1000 / TStepInMs)
    volume_rate_hz   same value, filled in only for snake (Z-stack) runs
    slice_rate_hz    frame_rate_hz x slices = individual planes per second (snake)
    snake_n_slices   number of Z-planes per volume, only for snake runs (--nz for
                     extract_mesc.py)
    pixel_average    samples averaged per pixel at acquisition (affects the data)
    display_average  MESc viewer running average only (does NOT affect the data)
    pixel_x/y_um, voxel_z_um   voxel size of the reconstructed volume (--voxel)
"""

import h5py
import numpy as np
import json
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


def decode(val):
    if isinstance(val, bytes):
        for enc in ['utf-8', 'utf-16', 'latin-1']:
            try:
                return val.decode(enc)
            except Exception:
                pass
        return repr(val)
    if isinstance(val, np.ndarray):
        if val.dtype.kind in ('S', 'U', 'O'):
            items = [decode(x) if isinstance(x, bytes) else str(x) for x in val.flat]
            return items[0] if len(items) == 1 else items
        return val.tolist()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def attr(unit, key, default=None):
    if key in unit.attrs:
        return decode(unit.attrs[key])
    return default


def parse_json_attr(unit, key):
    raw = attr(unit, key)
    if not raw or not isinstance(raw, str):
        return raw if isinstance(raw, dict) else {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def device_map(unit):
    """Return {device_name: position} from DeviceJSON."""
    d = parse_json_attr(unit, 'DeviceJSON')
    devices = d.get('devices', []) if isinstance(d, dict) else []
    return {dev['name']: dev['position'] for dev in devices if 'name' in dev}


def posix_to_str(ts):
    if ts is None:
        return 'unknown'
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    except Exception:
        return str(ts)


def main_pattern(unit):
    """The scan pattern actually used for this measurement (mainPatternIndex)."""
    mroi = parse_json_attr(unit, 'MultiROIProtocolJSON')
    if not isinstance(mroi, dict):
        return {}
    pats = (mroi.get('scanPatterns') or {}).get('patterns') or []
    idx = (mroi.get('protocol') or {}).get('mainPatternIndex')
    if isinstance(idx, int) and 0 <= idx < len(pats):
        return pats[idx]
    # fall back: first pattern that carries reconstructed-volume pixel sizes
    for p in pats:
        if 'pixelSizeL' in p:
            return p
    return pats[0] if pats else {}


def scan_type_of(method, tech, n_t):
    """Short, unambiguous label for what kind of acquisition this unit is."""
    m = (method or '').lower()
    if 'snake' in m:
        return 'snake'
    # check the orientation-qualified names before the bare 'ribbon' test,
    # otherwise 'multiROILongitudinalRibbonScan' would fall through to 'ribbon'
    if 'transverseribbon' in m or ('transverse' in m and 'ribbon' in m):
        return 'ribbon_transverse'
    if 'longitudinal' in m:
        return 'ribbon_longitudinal'
    if 'ribbon' in m:
        return 'ribbon'
    if 'zstack' in m:
        return 'zstack'
    if (tech or '').lower() == 'camera':
        return 'camera_snapshot' if n_t == 1 else 'camera_series'
    if n_t == 1:
        return 'snapshot'
    return method or 'timeseries'


def summarize_mesc(mesc_path, out_path=None):
    mesc_path = Path(mesc_path)
    if out_path is None:
        csv_path = mesc_path.with_suffix('.summary.csv')
    else:
        out_path = Path(out_path)
        # a path ending in .csv is the file itself, anything else is a directory
        if out_path.suffix.lower() == '.csv':
            csv_path = out_path
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            csv_path = out_path / (mesc_path.stem + '.summary.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    with h5py.File(mesc_path, 'r') as f:
        for sname in sorted(f.keys()):
            session = f[sname]
            for uname in sorted(session.keys()):
                unit = session[uname]

                channels = [k for k in unit.keys() if 'Channel' in k and hasattr(unit[k], 'shape')]
                if not channels:
                    continue

                method      = attr(unit, 'MethodTypeDebugString', '')
                section     = attr(unit, 'SectionTypeDebugString', '')
                tech        = attr(unit, 'TechnologyTypeDebugString', '')
                ts          = attr(unit, 'MeasurementDatePosix')
                duration    = attr(unit, 'MeasurementLengthInMs')
                comment     = attr(unit, 'Comment', '').strip()
                t_step      = attr(unit, 'TStepInMs')
                slices      = attr(unit, 'Slices')
                min_z       = attr(unit, 'MinZ')
                max_z       = attr(unit, 'MaxZ')
                dwell_ms    = attr(unit, 'DwellTime')
                pixelclock  = attr(unit, 'Pixelclock')
                disp_avg    = attr(unit, 'DisplayAverage')
                frame_loop  = attr(unit, 'FrameLoop')
                z_mode      = attr(unit, 'ZModeDebugString')
                is_zstack   = 'zstack' in str(method).lower()
                version     = parse_json_attr(unit, 'VersionInfoJSON')
                devs        = device_map(unit)
                ao_settings = parse_json_attr(unit, 'AOSettingsJSON')
                modality    = parse_json_attr(unit, 'ModalityJSON')
                pattern     = main_pattern(unit)
                # reconstructed-volume voxel: L=longitudinal(->X), T=transverse(->Y), Z=depth
                px_x        = pattern.get('pixelSizeL', pattern.get('pixelSizeX'))
                px_y        = pattern.get('pixelSizeT', pattern.get('pixelSizeY'))
                px_z        = pattern.get('voxelSizeZ')
                z_extent    = pattern.get('zSize')
                trans_size  = pattern.get('transverseSize')
                mes_ver     = version.get('mesVersion', '') if isinstance(version, dict) else ''

                # averaging: pixelAverage = samples averaged per pixel on acquisition
                # (affects the stored data); displayAverage = running average in the
                # MESc viewer only. Verified on these files: TStepInMs equals
                # (pixels per stored timepoint) / Pixelclock, i.e. every scanned line is
                # stored -- displayAverage does not reduce the data.
                px_avg = ao_settings.get('pixelAverage', '') if isinstance(ao_settings, dict) else ''

                objective = ''
                if isinstance(modality, dict):
                    for m in modality.get('modalities', []):
                        if m.get('name') == 'Objective':
                            objective = m.get('currentState', '')

                for cname in sorted(channels):
                    shape = unit[cname].shape
                    if is_zstack:
                        # anatomy stack: the frame axis is Z, not time
                        n_z, n_t = shape[0], 1
                    elif slices is not None and shape[0] > 1:
                        n_z = int(slices)
                        n_t = shape[0] // n_z if shape[0] % n_z == 0 else shape[0]
                    elif shape[0] == 1:
                        n_z, n_t = 1, 1
                    else:
                        n_z, n_t = 1, shape[0]

                    z_step = None
                    if min_z is not None and max_z is not None and n_z > 1:
                        z_step = (float(max_z) - float(min_z)) / (n_z - 1)

                    scan_type = scan_type_of(method, tech, n_t)
                    is_snake = scan_type == 'snake'

                    # TStepInMs is the period of one stored timepoint: one 2D frame for
                    # ribbon/timeSeries, one full Z-volume for snake.
                    rate_hz = 1000.0 / float(t_step) if t_step else None
                    slice_rate = rate_hz * n_z if (rate_hz and is_snake and n_z > 1) else None
                    # MeasurementLengthInMs is per frame on some aborted/free-running
                    # raster series, so also give the length implied by the data
                    dur_calc = n_t * float(t_step) / 1000 if (t_step and not is_zstack) else None

                    rows.append({
                        'file':          mesc_path.name,
                        'session':       sname,
                        'unit':          uname,
                        'channel':       cname,
                        'scan_type':     scan_type,
                        'type':          method or section or tech,
                        'comment':       comment,
                        'timestamp':     posix_to_str(ts),
                        'duration_s':    round(float(duration) / 1000, 2) if duration is not None else '',
                        'duration_calc_s': round(dur_calc, 2) if dur_calc else '',
                        # --- rate / averaging / slices -----------------------------
                        'frame_rate_hz': round(rate_hz, 3) if rate_hz else '',
                        'volume_rate_hz': round(rate_hz, 3) if (rate_hz and is_snake) else '',
                        'slice_rate_hz': round(slice_rate, 2) if slice_rate else '',
                        't_step_ms':     round(float(t_step), 4) if t_step is not None else '',
                        'snake_n_slices': n_z if is_snake else '',
                        'zstack_n_planes': n_z if is_zstack else '',
                        'pixel_average': px_avg,
                        'display_average': int(float(disp_avg)) if disp_avg not in (None, '') else '',
                        'frame_loop':    int(frame_loop) if frame_loop not in (None, '') else '',
                        'z_mode':        z_mode or '',
                        'pixel_dwell_us': round(float(dwell_ms) * 1000, 4) if dwell_ms else '',
                        'pixel_clock_hz': round(float(pixelclock), 1) if pixelclock else '',
                        # -----------------------------------------------------------
                        'n_t':           n_t,
                        'n_z':           n_z,
                        'n_y':           shape[1],
                        'n_x':           shape[2],
                        'pixel_x_um':    round(float(px_x), 4) if px_x is not None else '',
                        'pixel_y_um':    round(float(px_y), 4) if px_y is not None else '',
                        'voxel_z_um':    round(float(px_z), 4) if px_z is not None else '',
                        'z_extent_um':   round(float(z_extent), 2) if z_extent is not None else '',
                        'transverse_um': round(float(trans_size), 2) if trans_size is not None else '',
                        'min_z_um':      min_z if min_z is not None else '',
                        'max_z_um':      max_z if max_z is not None else '',
                        'z_step_um':     round(z_step, 4) if z_step is not None else '',
                        'stage_x_um':    round(devs['FemtoStageX'], 1) if 'FemtoStageX' in devs else '',
                        'stage_y_um':    round(devs['FemtoStageY'], 1) if 'FemtoStageY' in devs else '',
                        'stage_z_um':    round(devs['ObjectiveArm'], 1) if 'ObjectiveArm' in devs else '',
                        'wavelength_LP1': devs.get('wavelength_LP1', ''),
                        'power_LP1':     devs.get('power_LP1', ''),
                        'wavelength_LP2': devs.get('wavelength_LP2', ''),
                        'power_LP2':     devs.get('power_LP2', ''),
                        'UG':            devs.get('UG', ''),
                        'UR':            devs.get('UR', ''),
                        'AO':            devs.get('AO', ''),
                        'GDD2':          devs.get('GDD2', ''),
                        'objective':     objective,
                        'software':      mes_ver,
                    })

    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Written: {csv_path}")

        # short console recap of the recordings + anatomy z-stacks
        # (single-frame snapshots are left out; use extract_mesc_snapshots.py for those)
        recs = [r for r in rows if r['n_t'] > 1 or r['scan_type'] == 'zstack']
        if recs:
            print(f"\n{len(recs)} recording(s):\n")
            print(f"  {'unit':<10}{'chan':<6}{'scan_type':<19}{'rate':>9}  {'slices':>6}  "
                  f"{'px_avg':>6}  {'disp_av':>7}  {'T':>6}  {'dur_s':>6}  comment")
            for r in recs:
                rate = f"{r['frame_rate_hz']} Hz" if r['frame_rate_hz'] != '' else '-'
                unit_lbl = f"{r['session'].replace('MSession_', 'S')}/{r['unit'].replace('MUnit_', 'U')}"
                slices = r['snake_n_slices'] or r['zstack_n_planes']
                dur = r['duration_calc_s'] or r['duration_s']
                print(f"  {unit_lbl:<10}{r['channel'].replace('Channel_', 'c'):<6}"
                      f"{r['scan_type']:<19}{rate:>9}  "
                      f"{str(slices):>6}  {str(r['pixel_average']):>6}  "
                      f"{str(r['display_average']):>7}  "
                      f"{r['n_t']:>6}  {dur:>6}  {r['comment']}")
            print("\n  rate = stored timepoints/s (volumes/s for snake; "
                  "x slices = slice_rate_hz)")
            print("  slices = snake_n_slices, or zstack_n_planes for a zStack "
                  "(T=1: it is anatomy, not time)")
            print("  px_avg = samples averaged per pixel at acquisition; "
                  "disp_av = MESc viewer running average only (data not averaged)")
            print("  dur_s = n_t x t_step_ms where available (MeasurementLengthInMs is "
                  "per frame on some aborted raster series)")


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    out = None
    if '--out' in args:
        i = args.index('--out')
        out = args[i + 1]
        del args[i:i + 2]

    if args:
        mesc_file = Path(args[0])
    else:
        mesc_file = Path(__file__).parent / "rbp4_132_2026-05-20.mesc"

    if not mesc_file.exists():
        print(f"File not found: {mesc_file}")
        sys.exit(1)

    # --out DIR (or FILE) writes the CSV somewhere else, e.g. when the folder
    # holding the .mesc is read-only (someone else's project space).
    summarize_mesc(mesc_file, out)
