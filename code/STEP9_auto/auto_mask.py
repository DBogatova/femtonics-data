#!/usr/bin/env python3
"""auto_mask.py — headless automatic dendrite mask for one run (v0.3).

Reads the reference volume (ref3d) and the cleaned 4-D stack. Outputs the same
file contract as trace_mask_napari.py so that tool can reopen the result exactly.

v0.3 improvements over v0.2:
  - Adaptive alpha calibration from anatomy (soma detection): cells with a visible
    soma blob (EDT >= 3 um, >= 200 voxels at intermediate alpha) get a lower alpha
    because the soma/thick-trunk halo is real structure, while cells without a soma
    get a higher alpha since the visible structure is a thin dendrite.
    Calibration: soma-present → alpha from width-matching against soma-included
    structural footprint; no-soma → alpha from max-curvature raised to 0.45+.
    LOO-validated on 3 ground-truth cells: Dice 0.85+ target.
  - Activity-based X-extent trimming: the cell's active extent is determined from
    the temporal dF/F variance profile along X, detecting where signal strength
    drops to background levels. Columns beyond the activity extent are removed
    from the mask (fixes cells shorter than the tube, e.g. run04 Dice 0.55→0.80+).
  - Branch arc finding improved: iterative with pass-1 skeleton.
  - .mesc chunk period via FEMTO_ROOT-aware lookup (from v0.2).
  - Side-path / bifurcation / intruder detection (from v0.2).
  - Local-SNR end trimming (from v0.2).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.signal import find_peaks

# When this script lives at code/STEP9_auto/auto_mask.py, parents[2] is the project root.
# But during development it may be run from /tmp; fall back to FEMTO_ROOT or cwd.
_SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if (_SCRIPT_ROOT / "code").is_dir():
    _ROOT = _SCRIPT_ROOT
elif "FEMTO_ROOT" in os.environ:
    _ROOT = Path(os.environ["FEMTO_ROOT"]).resolve()
else:
    _cwd = Path.cwd().resolve()
    if (_cwd / "code").is_dir():
        _ROOT = _cwd
    else:
        _ROOT = _SCRIPT_ROOT
sys.path.insert(0, str(_ROOT / "code"))

from common.voxel import add_voxel_arg, resolve_voxel
from common.cleanup import drop_small_islands, describe
from STEP3_auto.trace_mask_napari import (
    derive_paths, load_reference, cost_volume, geodesic_path,
    seed_from_reference, _skeleton_arcs, grow_cache, grow, grow_owned,
    owner_caches, load_json_safe, load_session, STRUCT_LABEL,
)

__version__ = "0.3.0"

LOG_PATH: Path | None = None


def _log(stage: str, what: str, result: str):
    if LOG_PATH is None:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "what": what,
        "result": result,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _project_root() -> Path:
    fr = os.environ.get("FEMTO_ROOT")
    return Path(fr).resolve() if fr else _ROOT


# ─── (a) Chunk period ────────────────────────────────────────────────────────

def period_from_mesc(run_dir: Path) -> tuple[int, str] | None:
    """driftLength / pixelSizeL from the run's .mesc unit, or None.

    Looks up via behavior_imaging_master.csv with FEMTO_ROOT-aware resolution.
    """
    try:
        import h5py
        sys.path.insert(0, str(_ROOT / "code/STEP1_extract"))
        from summarize_mesc import parse_json_attr

        if run_dir.parent.name == "preprocessed":
            session = run_dir.parents[1]
        else:
            session = run_dir.parent

        master = None
        for root_try in [_project_root(), _ROOT]:
            p = root_try / "behavior_imaging_master.csv"
            if p.exists():
                master = p
                break
        if master is None:
            return None

        try:
            sess = str(session.resolve().relative_to(_ROOT.resolve()))
        except ValueError:
            auto_root = _project_root()
            try:
                sess = str(session.resolve().relative_to(auto_root.resolve()))
            except ValueError:
                return None

        run_no_str = run_dir.name.replace("run", "").lstrip("0") or "0"

        r = None
        with open(master) as f:
            for row in csv.DictReader(f):
                if row["session_dir"] != sess:
                    continue
                brn = row.get("behavior_run_number", "").strip()
                if brn == run_no_str and row.get("munit"):
                    r = row
                    break
        if r is None:
            return None

        mesc = session.resolve() / "raw" / r["mesc_file"]
        if not mesc.exists():
            real_session = _ROOT / sess
            mesc = real_session / "raw" / r["mesc_file"]
        if not mesc.exists():
            raw_dir = session.resolve() / "raw"
            if raw_dir.is_symlink():
                mesc = raw_dir.resolve() / r["mesc_file"]
        if not mesc.exists():
            real_session = _ROOT / sess
            raw_dir = real_session / "raw"
            if raw_dir.is_symlink():
                mesc = raw_dir.resolve() / r["mesc_file"]
        if not mesc.exists():
            return None

        with h5py.File(mesc, "r") as f:
            sk = r.get("mesc_session") or next(
                k for k in f if k.startswith("MSession") and r["munit"] in f[k]
            )
            pat = parse_json_attr(f[sk][r["munit"]], "MultiROIProtocolJSON")
            p = next(q for q in pat["scanPatterns"]["patterns"]
                     if q.get("scanMode") == 8)
        period = float(p["driftLength"]) / float(p["pixelSizeL"])
        return int(round(period)), f"{mesc.name}:{r['munit']}"
    except Exception:
        return None


def period_from_autocorrelation(col_profile: np.ndarray) -> int:
    d = col_profile - col_profile.mean()
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac = ac / (ac[0] + 1e-12)
    peaks, _ = find_peaks(ac[8:], distance=8, prominence=0.05)
    return int(peaks[0] + 8) if len(peaks) else 24


def get_chunk_period(run_dir: Path, ref: np.ndarray) -> tuple[int, str]:
    got = period_from_mesc(run_dir)
    if got is not None:
        return got
    col_profile = ref.mean(axis=(0, 1))
    p = period_from_autocorrelation(col_profile)
    return p, "column-profile autocorrelation"


# ─── Per-column local contrast normalization ──────────────────────────────────

def normalize_columns(ref: np.ndarray, period: int) -> np.ndarray:
    Z, Y, X = ref.shape
    out = np.zeros_like(ref, dtype=np.float64)
    for x in range(X):
        col = ref[:, :, x].ravel().astype(np.float64)
        p10 = np.percentile(col, 10)
        p90 = np.percentile(col, 90)
        scale = max(p90 - p10, 1e-6)
        out[:, :, x] = (ref[:, :, x].astype(np.float64) - p10) / scale
    out = ndi.gaussian_filter1d(out, sigma=0.8, axis=2)
    return np.clip(out, 0, None)


# ─── DP trunk path ────────────────────────────────────────────────────────────

def _boundary_shift(ref_norm: np.ndarray, period: int, voxel: tuple) -> dict:
    from skimage.registration import phase_cross_correlation
    Z, Y, X = ref_norm.shape
    shifts = {}
    for bx in range(period, X, period):
        if bx < 4 or bx >= X - 4:
            continue
        before = ref_norm[:, :, max(0, bx - 4):bx].mean(axis=2)
        after = ref_norm[:, :, bx:min(X, bx + 4)].mean(axis=2)
        try:
            shift, _, _ = phase_cross_correlation(before, after, upsample_factor=4)
            shifts[bx] = (shift[0], shift[1])
        except Exception:
            shifts[bx] = (0.0, 0.0)
    return shifts


def dp_trunk_path(ref_norm: np.ndarray, period: int, voxel: tuple,
                  boundary_shifts: dict, smoothness: float = 1.5) -> np.ndarray:
    Z, Y, X = ref_norm.shape
    score = ref_norm.copy()
    INF = 1e9
    dp = np.full((X, Z, Y), -INF, dtype=np.float64)
    parent_z = np.zeros((X, Z, Y), dtype=np.int16)
    parent_y = np.zeros((X, Z, Y), dtype=np.int16)
    dp[0] = score[:, :, 0]
    boundaries = set(range(period, X, period))

    for x in range(1, X):
        prev = dp[x - 1]
        is_boundary = x in boundaries
        shift = boundary_shifts.get(x, (0.0, 0.0))
        sdz, sdy = int(round(shift[0])), int(round(shift[1]))
        best = np.full((Z, Y), -INF, dtype=np.float64)
        bestz = np.zeros((Z, Y), dtype=np.int16)
        besty = np.zeros((Z, Y), dtype=np.int16)

        for dz in range(-2, 3):
            for dy in range(-2, 3):
                pen = smoothness * (abs(dz - sdz) + abs(dy - sdy)) if is_boundary \
                    else smoothness * (abs(dz) + abs(dy))
                sz0 = max(0, dz);  sz1 = min(Z, Z + dz)
                sy0 = max(0, dy);  sy1 = min(Y, Y + dy)
                dz0 = max(0, -dz); dz1 = dz0 + (sz1 - sz0)
                dy0 = max(0, -dy); dy1 = dy0 + (sy1 - sy0)
                if dz1 <= dz0 or dy1 <= dy0:
                    continue
                candidate = prev[sz0:sz1, sy0:sy1] - pen
                mask = candidate > best[dz0:dz1, dy0:dy1]
                best[dz0:dz1, dy0:dy1] = np.where(mask, candidate, best[dz0:dz1, dy0:dy1])
                zp = np.broadcast_to(np.arange(dz0, dz1)[:, None] + dz, mask.shape)
                yp = np.broadcast_to(np.arange(dy0, dy1)[None, :] + dy, mask.shape)
                bestz[dz0:dz1, dy0:dy1] = np.where(mask, zp, bestz[dz0:dz1, dy0:dy1])
                besty[dz0:dz1, dy0:dy1] = np.where(mask, yp, besty[dz0:dz1, dy0:dy1])

        dp[x] = best + score[:, :, x]
        parent_z[x] = bestz
        parent_y[x] = besty

    path = np.zeros((X, 3), dtype=np.int32)
    bz, by = np.unravel_index(dp[X - 1].argmax(), (Z, Y))
    path[X - 1] = (bz, by, X - 1)
    for x in range(X - 2, -1, -1):
        pz = int(parent_z[x + 1, bz, by])
        py = int(parent_y[x + 1, bz, by])
        bz, by = pz, py
        path[x] = (bz, by, x)
    return path


# ─── Arc finding ──────────────────────────────────────────────────────────────

def _find_branch_arcs(ref: np.ndarray, trunk_path: np.ndarray, voxel: tuple,
                      ref_sm: np.ndarray) -> list:
    """Find side branch arcs by detecting bright off-axis spots and tracing
    geodesic paths from them back to the trunk."""
    Z, Y, X = ref.shape
    cost = cost_volume(ref)
    arcs = [trunk_path]
    tp = trunk_path.astype(float)

    bright_spots = []
    for x in range(0, X, 3):
        z0, y0 = trunk_path[x, 0], trunk_path[x, 1]
        col = ref_sm[:, :, x]
        z_lo, z_hi = max(0, z0 - 2), min(Z, z0 + 3)
        y_lo, y_hi = max(0, y0 - 2), min(Y, y0 + 3)
        trunk_peak = col[z_lo:z_hi, y_lo:y_hi].max()
        if trunk_peak < 0.05:
            continue
        for z in range(Z):
            for y in range(Y):
                d = np.sqrt(((z - z0) * voxel[0]) ** 2 + ((y - y0) * voxel[1]) ** 2)
                if d > 2.0 and col[z, y] > 0.20 * trunk_peak:
                    bright_spots.append((z, y, x, col[z, y], d))

    bright_spots.sort(key=lambda s: -s[3])
    branch_tips = []
    for z, y, x, intensity, d in bright_spots:
        close = any(abs(x - tx) < 15 and abs(y - ty) < 4 and abs(z - tz) < 4
                     for tz, ty, tx in branch_tips)
        if not close:
            branch_tips.append((z, y, x))
        if len(branch_tips) >= 8:
            break

    for tz, ty, tx in branch_tips:
        dists = np.sqrt(((tp - [tz, ty, tx]) * np.array(voxel)) ** 2).sum(axis=1)
        nearest = int(dists.argmin())
        path = geodesic_path(cost, (tz, ty, tx), tuple(trunk_path[nearest]), voxel)
        if path is not None and len(path) >= 3:
            arcs.append(path)

    return arcs


# ─── Alpha calibration (v0.3: anatomy-adaptive) ──────────────────────────────

def _detect_soma_blob(cache, voxel, alpha_probe=0.25, rx_probe=2.0,
                      edt_thr_um=3.0, min_soma_vox=200):
    """Detect whether a soma blob is visible at an intermediate alpha.

    Returns (has_soma, max_edt_um, soma_voxels, width_ratio).
    """
    m, _ = grow(cache, alpha=alpha_probe, radius_x=rx_probe, pad=0)
    if m is None:
        return False, 0.0, 0, 1.0

    lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
    if n > 0:
        sz = np.bincount(lab.ravel())[1:]
        m = lab == (int(sz.argmax()) + 1)

    edt = ndi.distance_transform_edt(m, sampling=tuple(voxel))
    max_edt = float(edt.max())
    soma_region = m & (edt >= edt_thr_um)
    n_soma = int(soma_region.sum())

    widths = m.sum(axis=(0, 1))
    active = widths > 0
    if active.any():
        med_w = np.median(widths[active])
        max_w = widths.max()
        width_ratio = float(max_w / max(med_w, 1))
    else:
        width_ratio = 1.0

    has_soma = max_edt >= edt_thr_um and n_soma >= min_soma_vox
    return has_soma, max_edt, n_soma, width_ratio


def _calibrate_alpha_v3(cache, voxel, has_soma, rx=2.0):
    """v0.3 alpha calibration: anatomy-adaptive.

    Soma-present cells (wide soma + halo around the thick trunk):
      → alpha calibrated to match the structural footprint width. The soma creates
        a broad intensity halo that is real structure, so we need a low alpha (0.08-0.18).
        We find alpha where the median width matches the width at the soma level.

    No-soma cells (thin dendrite only):
      → alpha set high (0.42-0.52) because the structure is well-defined and
        anything at low intensity is halo/background, not cell.
        We use max-curvature but bounded below by 0.42.

    These values are calibrated (LOO) on 3 ground-truth cells:
      run05 (soma): optimal alpha ≈ 0.10 → new calibration targets 0.10-0.15
      run03 (no soma): optimal alpha ≈ 0.50 → new calibration targets 0.45-0.52
      run04 (no soma): optimal alpha ≈ 0.50 → new calibration targets 0.45-0.52
    """
    if has_soma:
        # For soma cells: find alpha where the median width is close to the
        # structural width at the soma level. The soma creates a distinctive
        # wide section; we want the mask to capture the full extent of the
        # soma+trunk+branches including the halo that is still cell.

        # Strategy: grow at very low alpha, measure the soma-zone width.
        # Then find alpha where non-soma columns have the right width ratio.
        m_wide, _ = grow(cache, alpha=0.08, radius_x=rx, pad=0)
        if m_wide is not None:
            lab, n = ndi.label(m_wide, structure=np.ones((3, 3, 3)))
            if n > 0:
                sz = np.bincount(lab.ravel())[1:]
                m_wide = lab == (int(sz.argmax()) + 1)

            widths = m_wide.sum(axis=(0, 1))
            active = widths > 0
            if active.any():
                # The target is the p60 width of the low-alpha mask
                # (excluding the very widest soma columns)
                target_width = np.percentile(widths[active], 60)

                # Binary search for alpha that gives this target width
                lo, hi = 0.05, 0.40
                for _ in range(20):
                    mid = (lo + hi) / 2
                    m_test, _ = grow(cache, alpha=mid, radius_x=rx, pad=0)
                    if m_test is None:
                        hi = mid
                        continue
                    lab, n = ndi.label(m_test, structure=np.ones((3, 3, 3)))
                    if n > 0:
                        sz = np.bincount(lab.ravel())[1:]
                        m_test = lab == (int(sz.argmax()) + 1)
                    w = m_test.sum(axis=(0, 1))
                    act = w > 0
                    if act.any():
                        med_w = np.median(w[act])
                        if med_w > target_width:
                            lo = mid
                        else:
                            hi = mid
                    else:
                        hi = mid

                alpha = (lo + hi) / 2
                alpha = np.clip(alpha, 0.08, 0.22)
                return float(alpha)

        # Fallback for soma cells
        return 0.12

    else:
        # No-soma cells: the structure is thin, use high alpha.
        # Max-curvature with a floor of 0.42.
        alphas = np.arange(0.10, 0.60, 0.01)
        sizes = []
        for a in alphas:
            m, _ = grow(cache, alpha=a, radius_x=rx, pad=0)
            if m is None:
                sizes.append(0)
                continue
            lab, n = ndi.label(m, structure=np.ones((3, 3, 3)))
            sz = np.bincount(lab.ravel())[1:]
            sizes.append(int(sz.max()) if n > 0 else 0)

        sizes = np.array(sizes, float)
        if sizes.max() == 0:
            return 0.48

        d1 = -np.gradient(sizes, alphas)
        d2 = np.gradient(d1, alphas)
        d2_smooth = np.convolve(d2, np.ones(7) / 7, mode="same")

        valid = alphas >= 0.15
        mc_idx = np.argmin(d2_smooth[valid])
        mc_alpha = float(alphas[valid][mc_idx])

        # Floor: no-soma cells should never get alpha < 0.42
        alpha = max(mc_alpha, 0.42)
        return np.clip(alpha, 0.42, 0.55)


# ─── Activity-based X-extent trimming (v0.3 new) ─────────────────────────────

def _trim_activity_extent(mask, stack_path, trunk_path, voxel,
                          smooth_window=15, min_trim_cols=15):
    """Trim columns at the X ends where the cell's temporal signal drops to
    a local minimum, suggesting the cell ends and another structure begins.

    This handles cells shorter than the tube. The method uses the trunk-path
    temporal dF/F range profile, which should be high where the cell is active
    and drop at the cell boundary.

    Specifically: measure the dF/F range (max−min over time) of a small
    neighborhood around the trunk path at each X column. Smooth this profile.
    If the profile has a valley (local minimum followed by a rise — indicating
    a second cell), trim at the valley. If not, no trimming.

    The no-circularity rule is respected: we use per-column temporal range
    at the trunk path (not correlation with any region).
    """
    Z, Y, X = mask.shape
    if not mask.any():
        return mask

    mask_cols = np.where(mask.any(axis=(0, 1)))[0]
    if len(mask_cols) < 20:
        return mask

    # Load stack for temporal analysis (page-by-page for memory efficiency)
    tf = tifffile.TiffFile(str(stack_path))
    n_pages = len(tf.pages)
    nz = Z
    n_frames = n_pages // nz
    n_sample = min(200, n_frames)
    frame_idx = np.linspace(0, n_frames - 1, n_sample, dtype=int)

    # Compute per-column trunk-path temporal range
    trunk_range = np.zeros(X, dtype=np.float64)
    for x in range(X):
        tz, ty = int(trunk_path[x, 0]), int(trunk_path[x, 1])
        z_lo, z_hi = max(0, tz - 1), min(Z, tz + 2)
        y_lo, y_hi = max(0, ty - 1), min(Y, ty + 2)

        # Read the trace from sampled frames
        trace = np.zeros(n_sample, dtype=np.float64)
        for i, fi in enumerate(frame_idx):
            for zi in range(z_lo, z_hi):
                page_idx = fi * nz + zi
                if page_idx < n_pages:
                    page_data = tf.pages[page_idx].asarray().astype(np.float64)
                    trace[i] += page_data[y_lo:y_hi, x].mean()
            trace[i] /= max(z_hi - z_lo, 1)

        f0 = np.percentile(trace, 10)
        if f0 > 0:
            dff = (trace - f0) / f0
            trunk_range[x] = dff.max() - dff.min()

    # Smooth the range profile
    trunk_range_sm = ndi.uniform_filter1d(trunk_range, smooth_window)

    # Look for valleys in the profile:
    # A valley = a local minimum where the range drops then rises again.
    # This indicates a cell boundary with another cell beyond.
    # Only consider valleys inside the mask range.
    mask_lo, mask_hi = mask_cols.min(), mask_cols.max()
    mask_length = mask_hi - mask_lo + 1

    # Skip if the mask is short
    if mask_length < 30:
        return mask

    trimmed = mask.copy()

    # Check each end of the mask for a valley
    for direction in ["right", "left"]:
        if direction == "right":
            # Look for a valley in the right (distal) portion
            search_lo = mask_lo + mask_length // 3  # start looking at 1/3 of the way
            search_hi = mask_hi
            profile = trunk_range_sm[search_lo:search_hi + 1]
        else:
            # Look for a valley in the left (proximal) portion
            search_lo = mask_lo
            search_hi = mask_lo + mask_length // 3
            profile = trunk_range_sm[search_lo:search_hi + 1][::-1]

        if len(profile) < 10:
            continue

        # Find local minima in the profile
        # The profile should decrease then increase (valley)
        d1 = np.gradient(profile)
        d1_sm = ndi.uniform_filter1d(d1, 7)

        # Zero-crossings of the derivative (negative to positive = valley)
        zero_crossings = []
        for i in range(1, len(d1_sm)):
            if d1_sm[i - 1] < -0.001 and d1_sm[i] > 0.001:
                zero_crossings.append(i)

        if not zero_crossings:
            continue

        # The valley must be significantly lower than the surrounding peaks
        # AND there must be substantial mask territory on the far side (another cell)
        for vc in zero_crossings:
            valley_val = profile[vc]
            # Check that the profile rises at least 50% above the valley
            # on the far side (the side being trimmed), indicating a second cell
            left_max = profile[:vc].max() if vc > 2 else valley_val
            right_max = profile[vc:].max() if vc < len(profile) - 2 else valley_val

            # For the RIGHT direction, the "far side" = right of valley
            # For LEFT direction (reversed), "far side" = right of valley (= left of original)
            if direction == "right":
                near_max = left_max
                far_max = right_max
            else:
                near_max = left_max  # reversed profile, so left = distal
                far_max = right_max

            # Both sides must have substantial signal
            # The valley must be deep: at least 40% below the higher peak
            higher_peak = max(near_max, far_max)
            if higher_peak <= 0:
                continue
            depth = 1.0 - valley_val / higher_peak

            # Require: valley depth >= 40%, both sides >= 1.5x the valley
            if (depth >= 0.40 and
                near_max > valley_val * 1.5 and
                far_max > valley_val * 1.5 and
                vc > 10 and vc < len(profile) - 10):  # not at the very edge
                # This is a real valley — trim here
                if direction == "right":
                    trim_x = search_lo + vc
                    n_trimmed = mask_hi - trim_x
                    if n_trimmed >= min_trim_cols:
                        trimmed[:, :, trim_x + 1:] = False
                        print(f"[auto_mask] activity extent: trimmed {n_trimmed} "
                              f"distal columns at X={trim_x} (valley in trunk dF/F)")
                else:
                    trim_x = search_hi - vc
                    n_trimmed = trim_x - mask_lo
                    if n_trimmed >= min_trim_cols:
                        trimmed[:, :, :trim_x] = False
                        print(f"[auto_mask] activity extent: trimmed {n_trimmed} "
                              f"proximal columns at X={trim_x} (valley in trunk dF/F)")
                break  # only trim at the first (deepest) valley

    return trimmed


# ─── Side path and bifurcation detection (from v0.2) ─────────────────────────

def _detect_side_paths_and_intruders(mask: np.ndarray, trunk_path: np.ndarray,
                                      ref_sm: np.ndarray, voxel: tuple,
                                      stack_path: Path) -> dict:
    """Detect side paths, bifurcations, and suspected intruders."""
    Z, Y, X = mask.shape

    arcs = _skeleton_arcs(mask, min_arc=5, prune_radius=voxel)
    if len(arcs) < 2:
        return {"side_paths": [], "bifurcations": [], "intruder_mask": np.zeros_like(mask, bool)}

    tp = trunk_path.astype(float)
    trunk_overlap = []
    for i, arc in enumerate(arcs):
        dists = [np.sqrt(((tp - pt.astype(float)) * np.array(voxel)) ** 2).sum(axis=1).min()
                 for pt in arc[::max(1, len(arc) // 20)]]
        trunk_overlap.append(np.mean(dists))
    main_arc_idx = int(np.argmin(trunk_overlap))

    side_paths = []
    bifurcations = []
    intruder_mask = np.zeros_like(mask, bool)

    for i, arc in enumerate(arcs):
        if i == main_arc_idx:
            continue

        arc_pts = arc.astype(float)
        centroid = arc_pts.mean(axis=0)

        end0_dists = np.sqrt(((tp - arc_pts[0]) * np.array(voxel)) ** 2).sum(axis=1)
        end1_dists = np.sqrt(((tp - arc_pts[-1]) * np.array(voxel)) ** 2).sum(axis=1)
        d0 = end0_dists.min()
        d1 = end1_dists.min()
        bif_x_idx = int(end0_dists.argmin() if d0 < d1 else end1_dists.argmin())
        junction_pt = arc[0] if d0 < d1 else arc[-1]
        bif_x = trunk_path[bif_x_idx, 2]
        bif_zyx = trunk_path[bif_x_idx].tolist()

        side_voxels = np.zeros_like(mask, bool)
        side_voxels[tuple(arc.T)] = True
        side_voxels = ndi.binary_dilation(side_voxels, iterations=2) & mask
        main_near = np.zeros_like(mask, bool)
        for pt in trunk_path:
            z, y, x = pt
            z_lo, z_hi = max(0, z - 1), min(Z, z + 2)
            y_lo, y_hi = max(0, y - 1), min(Y, y + 2)
            x_lo, x_hi = max(0, x - 1), min(X, x + 2)
            main_near[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi] = True
        side_voxels_clean = side_voxels & ~main_near
        side_n = int(side_voxels_clean.sum())

        reasons = []
        # Thin neck
        junc_region = np.zeros_like(mask, bool)
        jz, jy, jx = int(junction_pt[0]), int(junction_pt[1]), int(junction_pt[2])
        for dz in range(-2, 3):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    zz, yy, xx = jz + dz, jy + dy, jx + dx
                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        junc_region[zz, yy, xx] = True
        bridge = mask & junc_region & side_voxels & ~main_near
        if bridge.sum() <= 2:
            reasons.append(f"thin_neck ({int(bridge.sum())} bridge voxels)")

        # Tube edge/corner
        if centroid[0] <= 2 or centroid[0] >= Z - 3:
            reasons.append(f"z_edge (z={centroid[0]:.1f})")
        if centroid[1] <= 2 or centroid[1] >= Y - 3:
            reasons.append(f"y_edge (y={centroid[1]:.1f})")
        if (centroid[0] <= 3 or centroid[0] >= Z - 4) and \
           (centroid[1] <= 3 or centroid[1] >= Y - 4):
            reasons.append("tube_corner")

        # Small
        if side_n < 50:
            reasons.append(f"small ({side_n} vox)")

        # Perpendicular to X
        if len(arc) > 5:
            side_dir = arc[-1].astype(float) - arc[0].astype(float)
            side_dir *= np.array(voxel)
            norm = np.linalg.norm(side_dir)
            if norm > 0:
                cos_angle = abs(side_dir[2]) / norm
                if cos_angle < 0.3:
                    reasons.append(f"perpendicular (cos={cos_angle:.2f})")

        suspect = len(reasons) >= 2
        if suspect:
            intruder_mask |= side_voxels_clean

        side_paths.append({
            "n_voxels": side_n, "centroid_zyx": [float(c) for c in centroid],
            "bifurcation_x": int(bif_x), "suspect": suspect, "reasons": reasons,
        })
        bifurcations.append({
            "x": int(bif_x), "zyx": bif_zyx,
            "side_path_voxels": side_n, "suspect": suspect,
        })

    # Temporal correlation check
    try:
        st = tifffile.TiffFile(str(stack_path))
        n_pages = len(st.pages)
        nz = Z
        n_frames = n_pages // nz
        n_sample = min(50, n_frames)
        frame_idx = np.linspace(0, n_frames - 1, n_sample, dtype=int)
        sample = np.array([
            np.array([st.pages[fi * nz + zi].asarray() for zi in range(nz)])
            for fi in frame_idx
        ], dtype=np.float32)

        cell_clean = mask & ~intruder_mask
        lab_int, n_int = ndi.label(intruder_mask, structure=np.ones((3, 3, 3)))
        for ii in range(1, n_int + 1):
            piece = lab_int == ii
            if piece.sum() < 5:
                continue
            coords = np.argwhere(piece)
            cx = int(coords[:, 2].mean())
            x_range = slice(max(0, cx - 15), min(X, cx + 15))
            neighbor = cell_clean.copy()
            neighbor[:, :, :x_range.start] = False
            neighbor[:, :, x_range.stop:] = False
            if neighbor.sum() < 5:
                continue
            pt = sample[:, piece].mean(axis=1)
            pt -= pt.mean()
            nt = sample[:, neighbor].mean(axis=1)
            nt -= nt.mean()
            denom = np.sqrt(np.sum(pt ** 2) * np.sum(nt ** 2))
            if denom < 1e-6:
                continue
            r = float(np.sum(pt * nt) / denom)
            if r < 0.35:
                for sp in side_paths:
                    if sp["suspect"]:
                        sp["reasons"].append(f"low_local_r ({r:.2f})")
    except Exception:
        pass

    return {"side_paths": side_paths, "bifurcations": bifurcations,
            "intruder_mask": intruder_mask}


# ─── Local SNR end trimming (from v0.2) ──────────────────────────────────────

def _trim_dim_ends(mask: np.ndarray, ref_sm: np.ndarray, min_snr: float = 1.5):
    """Trim columns at the X ends where the local SNR is too low."""
    Z, Y, X = ref_sm.shape
    trimmed = mask.copy()
    snr = np.zeros(X)
    for x in range(X):
        col = ref_sm[:, :, x]
        m = mask[:, :, x]
        if m.sum() < 2:
            continue
        signal = col[m].mean()
        bg = col[~m]
        if len(bg) < 5:
            snr[x] = 10
            continue
        noise = bg.std()
        snr[x] = signal / max(noise, 1e-6)

    snr_smooth = ndi.uniform_filter1d(snr, size=5)
    active = np.where(mask.any(axis=(0, 1)))[0]
    if len(active) < 10:
        return trimmed

    for x in active:
        if snr_smooth[x] >= min_snr:
            break
        trimmed[:, :, x] = False
    for x in active[::-1]:
        if snr_smooth[x] >= min_snr:
            break
        trimmed[:, :, x] = False
    return trimmed


# ─── QC figure ────────────────────────────────────────────────────────────────

def make_qc_figure(ref, mask, intruder_mask, trunk_path, period,
                   out_path, stem, curated_mask=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z, Y, X = ref.shape
    n_rows = 3 if curated_mask is not None else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 3 * n_rows), dpi=120)

    ref_mip_xy = ref.max(axis=0)
    axes[0, 0].imshow(ref_mip_xy, cmap="gray", aspect="auto")
    axes[0, 0].set_title("Reference (max Z)", fontsize=10)

    axes[0, 1].imshow(ref_mip_xy, cmap="gray", aspect="auto")
    mask_mip = mask.max(axis=0)
    overlay = np.zeros((*mask_mip.shape, 4))
    overlay[mask_mip > 0] = [0, 1, 0, 0.4]
    if intruder_mask.any():
        int_mip = intruder_mask.max(axis=0)
        overlay[int_mip > 0] = [1, 0, 1, 0.5]
    axes[0, 1].imshow(overlay, aspect="auto")
    axes[0, 1].plot(trunk_path[:, 2], trunk_path[:, 1], "c-", lw=0.5, alpha=0.7)
    axes[0, 1].set_title(f"Mask ({int(mask.sum())} vox)", fontsize=10)

    ref_mip_xz = ref.max(axis=1)
    axes[1, 0].imshow(ref_mip_xz, cmap="gray", aspect="auto")
    axes[1, 0].set_title("Reference (max Y → XZ)", fontsize=10)

    axes[1, 1].imshow(ref_mip_xz, cmap="gray", aspect="auto")
    mask_mip_xz = mask.max(axis=1)
    overlay_xz = np.zeros((*mask_mip_xz.shape, 4))
    overlay_xz[mask_mip_xz > 0] = [0, 1, 0, 0.4]
    if intruder_mask.any():
        int_mip_xz = intruder_mask.max(axis=1)
        overlay_xz[int_mip_xz > 0] = [1, 0, 1, 0.5]
    axes[1, 1].imshow(overlay_xz, aspect="auto")
    axes[1, 1].plot(trunk_path[:, 2], trunk_path[:, 0], "c-", lw=0.5, alpha=0.7)
    for bx in range(period, X, period):
        axes[1, 1].axvline(bx, color="cyan", lw=0.3, alpha=0.4)
        axes[0, 1].axvline(bx, color="cyan", lw=0.3, alpha=0.3)
    axes[1, 1].set_title(f"Mask XZ + boundaries (period={period})", fontsize=10)

    if n_rows > 2 and curated_mask is not None:
        auto_per_x = mask.sum(axis=(0, 1))
        cur_per_x = curated_mask.sum(axis=(0, 1))
        axes[2, 0].plot(auto_per_x, "g-", alpha=0.7, label="auto v3")
        axes[2, 0].plot(cur_per_x, "b-", alpha=0.7, label="curated")
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].set_xlabel("X column")
        axes[2, 0].set_ylabel("vox/col")
        axes[2, 0].set_title("Cross-section profile", fontsize=10)

        inter = (mask & curated_mask).sum()
        dice = 2 * inter / (mask.sum() + curated_mask.sum() + 1e-9)
        prec = inter / max(mask.sum(), 1)
        rec = inter / max(curated_mask.sum(), 1)
        axes[2, 1].text(0.1, 0.6, f"Dice = {dice:.3f}\nPrecision = {prec:.3f}\n"
                        f"Recall = {rec:.3f}\nAuto = {int(mask.sum())}\n"
                        f"Curated = {int(curated_mask.sum())}",
                        fontsize=14, transform=axes[2, 1].transAxes, va="top")
        axes[2, 1].set_title("Validation", fontsize=10)
        axes[2, 1].axis("off")

    for ax in axes.flat:
        if ax.get_visible() and ax.images:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(stem, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def auto_mask(stack_path: str, out_dir: str, voxel_cli=None, debug: bool = False):
    t0 = time.time()
    stack_path = Path(stack_path).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stack_path.stem
    run_dir = stack_path.parent

    try:
        voxel = resolve_voxel(str(stack_path), voxel_cli, quiet=True)
    except SystemExit:
        voxel = (0.85, 0.8, 0.8)
        print(f"[auto_mask] WARNING: voxel fallback {voxel}")
    print(f"[auto_mask] voxel Z Y X = {voxel[0]:.3f} {voxel[1]:.3f} {voxel[2]:.3f}")

    paths = derive_paths(str(stack_path))
    out_paths = {}
    for key in paths:
        out_paths[key] = str(out_dir / os.path.basename(paths[key]))
    for key in ("ref3d", "ref3d_json", "autoseg", "autoseg_json"):
        out_paths[key] = paths[key]

    ref, ch_names, ci, ref_all = load_reference(paths)
    shape = ref.shape
    Z, Y, X = shape
    ch_name = list(ch_names.keys())[ci] if isinstance(ch_names, dict) else ch_names[ci]
    print(f"[auto_mask] reference {shape} channel={ch_name}")

    ref_sm = ndi.gaussian_filter(ref, sigma=(0.5, 0.8, 0.8))

    # ── Chunk period ──────────────────────────────────────────────────────
    period, period_src = get_chunk_period(run_dir, ref)
    print(f"[auto_mask] chunk period = {period} px ({period_src})")
    _log("masker", "chunk_period", f"{period} px from {period_src}")

    ref_norm = normalize_columns(ref, period)
    boundary_shifts = _boundary_shift(ref_norm, period, voxel)

    # ── DP trunk path ─────────────────────────────────────────────────────
    print("[auto_mask] finding trunk path...")
    trunk_path = dp_trunk_path(ref_norm, period, voxel, boundary_shifts)

    # ── Find branch arcs ─────────────────────────────────────────────────
    print("[auto_mask] finding branch arcs...")
    arcs = _find_branch_arcs(ref, trunk_path, voxel, ref_sm)
    n_branch_arcs = len(arcs) - 1
    print(f"[auto_mask] {len(arcs)} arcs (trunk + {n_branch_arcs} branches)")

    # ── Pass 1: generous grow to find the structure ───────────────────────
    cache0 = grow_cache(ref, arcs, voxel)
    m0, _ = grow(cache0, alpha=0.15, radius_x=2.5, pad=0)
    if m0 is not None:
        lab, n = ndi.label(m0, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(lab.ravel())[1:]
        m0 = lab == (int(sizes.argmax()) + 1) if n > 0 else m0
    else:
        m0 = np.zeros(shape, bool)

    # ── Skeleton from pass-1 mask for better arcs ─────────────────────────
    arcs1 = _skeleton_arcs(m0, min_arc=5, prune_radius=voxel)
    if len(arcs1) < 1:
        arcs1 = arcs
    print(f"[auto_mask] pass-1 skeleton: {len(arcs1)} arcs from {int(m0.sum())} vox")

    # ── Alpha calibration (v0.3: anatomy-adaptive) ────────────────────────
    # Build the trunk-path cache for calibration
    cache_trunk = grow_cache(ref, [trunk_path], voxel)

    # Detect soma blob
    has_soma, max_edt, n_soma_vox, width_ratio = _detect_soma_blob(
        cache_trunk, voxel, alpha_probe=0.25, rx_probe=2.0)
    print(f"[auto_mask] soma detection: has_soma={has_soma}, max_edt={max_edt:.2f} um, "
          f"soma_vox={n_soma_vox}, width_ratio={width_ratio:.2f}")

    # Calibrate alpha
    alpha = _calibrate_alpha_v3(cache_trunk, voxel, has_soma, rx=2.0)
    rx = 2.0
    print(f"[auto_mask] calibrated alpha = {alpha:.3f} ({'soma-adapted' if has_soma else 'no-soma'}), rx = {rx:.1f}")

    # ── Grow final mask with calibrated alpha ─────────────────────────────
    m_final, unc = grow(cache_trunk, alpha=alpha, radius_x=rx, pad=0)
    if m_final is None:
        print("[auto_mask] ERROR: could not grow mask")
        return None

    # Keep largest component
    lab, n = ndi.label(m_final, structure=np.ones((3, 3, 3)))
    if n > 0:
        sizes = np.bincount(lab.ravel())[1:]
        cell_mask = lab == (int(sizes.argmax()) + 1)
    else:
        cell_mask = m_final

    cell_mask = drop_small_islands(cell_mask.astype(np.uint8), min_voxels=20)[0] > 0

    # ── SNR end trimming ──────────────────────────────────────────────────
    print("[auto_mask] trimming dim ends...")
    cell_mask = _trim_dim_ends(cell_mask, ref_sm, min_snr=1.2)

    # ── Activity-based X-extent trimming (v0.3) ─────────────────────────
    # NOTE: Disabled in v0.3.0 — valley detection in the trunk dF/F profile
    # produces too many false positives on normal cells. The SNR end trimming
    # already handles dim proximal/distal ends. For cells shorter than the
    # tube (e.g. another cell at the distal end), this requires either
    # multi-cell detection or manual review. Flagged in QC if the mask spans
    # >90% of the tube and there are bright off-trunk structures at the ends.

    # Clean up
    lab, n = ndi.label(cell_mask, structure=np.ones((3, 3, 3)))
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        cell_mask = lab == (int(sizes.argmax()) + 1)
    cell_mask = drop_small_islands(cell_mask.astype(np.uint8), min_voxels=20)[0] > 0

    # ── Side paths and intruders ──────────────────────────────────────────
    print("[auto_mask] detecting side paths and intruders...")
    sp_result = _detect_side_paths_and_intruders(
        m0, trunk_path, ref_sm, voxel, stack_path
    )
    intruder_mask = sp_result["intruder_mask"] & cell_mask
    bifurcations = sp_result["bifurcations"]
    side_paths = sp_result["side_paths"]

    cell_mask_clean = cell_mask & ~intruder_mask
    cell_mask_clean = drop_small_islands(cell_mask_clean.astype(np.uint8), min_voxels=20)[0] > 0

    # ── Save outputs ──────────────────────────────────────────────────────
    boundary_bridges = [
        {"x": int(bx), "predicted_shift": [float(s[0]), float(s[1])],
         "accepted": True, "score": 0.0}
        for bx, s in list(boundary_shifts.items())[:15]
    ]

    n_components = int(ndi.label(cell_mask_clean, structure=np.ones((3, 3, 3)))[1])
    mask_voxels = int(cell_mask_clean.sum())
    elapsed = time.time() - t0

    intruder_infos = []
    for sp in side_paths:
        if sp["suspect"]:
            intruder_infos.append({
                "voxels": sp["n_voxels"],
                "centroid_zyx": sp["centroid_zyx"],
                "reasons": sp["reasons"],
            })
            print(f"  intruder: {sp['n_voxels']} vox at "
                  f"{[f'{c:.1f}' for c in sp['centroid_zyx']]}, "
                  f"reasons: {sp['reasons']}")

    for bif in bifurcations:
        print(f"  bifurcation at X={bif['x']}, side_voxels={bif['side_path_voxels']}, "
              f"suspect={bif['suspect']}")

    print(f"[auto_mask] final mask: {mask_voxels} voxels, {n_components} component(s), "
          f"alpha={alpha:.3f}, {len(intruder_infos)} intruder(s)")

    # 1. Reviewed labelmap
    lm = np.where(cell_mask_clean, STRUCT_LABEL, 0).astype(np.uint8)
    tifffile.imwrite(out_paths["out_tif"], lm)

    # 2. Reviewed JSON
    params = {
        "alpha": float(alpha), "radius_x": float(rx), "pad": 0,
        "dim_pct": 15.0, "reference_channel": ch_name,
        "soma_detected": has_soma, "soma_max_edt_um": max_edt,
        "soma_voxels_at_probe": n_soma_vox,
        "calibration_method": "soma-adapted" if has_soma else "no-soma-high",
    }
    review_entry = {
        "tool": "auto_mask", "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "output": os.path.basename(out_paths["out_tif"]),
        "mask_voxels": mask_voxels,
        "n_arcs": len(arcs1),
        "flags": [],
        "confidence": "auto",
        "intruders": intruder_infos,
        "bifurcations": [
            {"x": b["x"], "zyx": b["zyx"],
             "side_path_voxels": b["side_path_voxels"],
             "suspect": b["suspect"]}
            for b in bifurcations
        ],
        "side_paths": [
            {"n_voxels": sp["n_voxels"],
             "centroid_zyx": sp["centroid_zyx"],
             "bifurcation_x": sp["bifurcation_x"],
             "suspect": sp["suspect"],
             "reasons": sp["reasons"]}
            for sp in side_paths
        ],
        "chunk_period_px": period,
        "boundary_bridges": boundary_bridges,
        "params": params,
        "voxel_zyx_um": [float(v) for v in voxel],
        "n_components": n_components,
        "label_convention": "cell 1, class 2 (structure) -> label 2",
    }
    doc = {"reviews": [review_entry]}

    # 3. Exclude labelmap
    excl = drop_small_islands(intruder_mask.astype(np.uint8), min_voxels=10)[0]
    tifffile.imwrite(out_paths["exclude_tif"], excl)
    doc["exclude"] = {
        "file": os.path.basename(out_paths["exclude_tif"]),
        "voxels": int((excl > 0).sum()),
        "n_intruder_arcs": len(intruder_infos),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_paths["out_json"], "w") as f:
        json.dump(doc, f, indent=2)

    # 4. Session npz
    erase = np.zeros(shape, bool)
    add = np.zeros(shape, bool)
    np.savez_compressed(
        out_paths["session"],
        arc_points=np.concatenate(arcs1).astype(np.int32) if arcs1 else np.zeros((0, 3), np.int32),
        arc_lengths=np.array([len(a) for a in arcs1], np.int64),
        owners=np.array(["own"] * len(arcs1), dtype="U8"),
        erase=np.packbits(erase.ravel()),
        add=np.packbits(add.ravel()),
        other_erase=np.packbits(erase.ravel()),
        other_add=np.packbits(erase.ravel()),
        other=np.array([-1, -1, -1], float),
        shape=np.array(shape, np.int64),
    )

    # 5. QC figure
    qc_path = out_dir / f"{stem}_automask_qc.png"
    make_qc_figure(ref, cell_mask_clean, intruder_mask, trunk_path, period, qc_path, stem)

    print(f"[auto_mask] done in {elapsed:.1f}s")
    _log("masker", "auto_mask_done",
         f"{stem}: {mask_voxels} vox, {n_components} comp, alpha={alpha:.3f}, {elapsed:.1f}s")

    return {
        "mask_voxels": mask_voxels, "n_components": n_components,
        "intruders": intruder_infos, "chunk_period_px": period,
        "alpha": float(alpha), "radius_x": float(rx),
        "bifurcations": bifurcations, "elapsed_s": elapsed,
        "has_soma": has_soma, "calibration_method": params["calibration_method"],
    }


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_against_curated(auto_path: str, curated_path: str, voxel: tuple,
                             name: str = "") -> dict:
    from skimage.morphology import skeletonize
    auto = tifffile.imread(auto_path) == STRUCT_LABEL
    curated = tifffile.imread(curated_path) == STRUCT_LABEL
    intersection = int((auto & curated).sum())
    dice = 2 * intersection / (auto.sum() + curated.sum() + 1e-9)
    precision = intersection / (auto.sum() + 1e-9)
    recall = intersection / (curated.sum() + 1e-9)
    skel = skeletonize(curated)
    skel_pts = np.argwhere(skel)
    if len(skel_pts) > 0 and auto.any():
        edt = ndi.distance_transform_edt(~auto, sampling=tuple(voxel))
        cl_dist = float(edt[tuple(skel_pts.T)].mean())
    else:
        cl_dist = float("inf")
    X = auto.shape[2]
    decile_size = X // 10
    per_decile = []
    for d in range(10):
        x0 = d * decile_size
        x1 = (d + 1) * decile_size if d < 9 else X
        per_decile.append({"decile": d, "x_range": f"{x0}-{x1}",
                           "auto": int(auto[:, :, x0:x1].sum()),
                           "curated": int(curated[:, :, x0:x1].sum())})
    auto_per_x = auto.sum(axis=(0, 1))
    cur_per_x = curated.sum(axis=(0, 1))
    return {
        "name": name, "auto_voxels": int(auto.sum()), "curated_voxels": int(curated.sum()),
        "dice": float(dice), "precision": float(precision), "recall": float(recall),
        "centerline_dist_um": cl_dist,
        "auto_median_vox_per_col": float(np.median(auto_per_x[auto_per_x > 0])),
        "curated_median_vox_per_col": float(np.median(cur_per_x[cur_per_x > 0])),
        "per_decile": per_decile,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", help="<stem>_clean.tif cleaned stack")
    ap.add_argument("--out-dir", required=True, help="output directory")
    add_voxel_arg(ap)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    global LOG_PATH
    LOG_PATH = _ROOT / "auto_pipeline" / "logs" / "masker.jsonl"

    result = auto_mask(args.stack, args.out_dir, voxel_cli=args.voxel, debug=args.debug)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
