"""
Produce a per-run summary of a Femtonics .mesc file.

Writes both a .summary.txt and a .summary.csv alongside the .mesc file.
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


def deep_find(obj, key):
    """First value of `key` anywhere in a nested dict/list, else None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = deep_find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = deep_find(v, key)
            if r is not None:
                return r
    return None


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


def summarize_mesc(mesc_path):
    mesc_path = Path(mesc_path)
    csv_path = mesc_path.with_suffix('.summary.csv')
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
                version     = parse_json_attr(unit, 'VersionInfoJSON')
                devs        = device_map(unit)
                ao_settings = parse_json_attr(unit, 'AOSettingsJSON')
                modality    = parse_json_attr(unit, 'ModalityJSON')
                mroi        = parse_json_attr(unit, 'MultiROIProtocolJSON')
                # reconstructed-volume voxel: L=longitudinal(->X), T=transverse(->Y), Z=depth
                px_x        = deep_find(mroi, 'pixelSizeL')
                px_y        = deep_find(mroi, 'pixelSizeT')
                px_z        = deep_find(mroi, 'voxelSizeZ')
                mes_ver     = version.get('mesVersion', '') if isinstance(version, dict) else ''

                objective = ''
                if isinstance(modality, dict):
                    for m in modality.get('modalities', []):
                        if m.get('name') == 'Objective':
                            objective = m.get('currentState', '')

                for cname in sorted(channels):
                    shape = unit[cname].shape
                    if slices is not None and shape[0] > 1:
                        n_z = int(slices)
                        n_t = shape[0] // n_z if shape[0] % n_z == 0 else shape[0]
                    elif shape[0] == 1:
                        n_z, n_t = 1, 1
                    else:
                        n_z, n_t = 1, shape[0]

                    z_step = None
                    if min_z is not None and max_z is not None and n_z > 1:
                        z_step = (float(max_z) - float(min_z)) / (n_z - 1)

                    rows.append({
                        'file':          mesc_path.name,
                        'session':       sname,
                        'unit':          uname,
                        'channel':       cname,
                        'type':          method or section or tech,
                        'timestamp':     posix_to_str(ts),
                        'duration_s':    round(float(duration) / 1000, 2) if duration is not None else '',
                        'comment':       comment,
                        'n_t':           n_t,
                        'n_z':           n_z,
                        'n_y':           shape[1],
                        'n_x':           shape[2],
                        'pixel_x_um':    round(float(px_x), 4) if px_x is not None else '',
                        'pixel_y_um':    round(float(px_y), 4) if px_y is not None else '',
                        'voxel_z_um':    round(float(px_z), 4) if px_z is not None else '',
                        't_step_ms':     round(float(t_step), 4) if t_step is not None else '',
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
                        'pixel_avg':     ao_settings.get('pixelAverage', '') if isinstance(ao_settings, dict) else '',
                        'objective':     objective,
                        'software':      mes_ver,
                    })

    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Written: {csv_path}")


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        mesc_file = Path(sys.argv[1])
    else:
        mesc_file = Path(__file__).parent / "rbp4_132_2026-05-20.mesc"

    if not mesc_file.exists():
        print(f"File not found: {mesc_file}")
        sys.exit(1)

    summarize_mesc(mesc_file)
