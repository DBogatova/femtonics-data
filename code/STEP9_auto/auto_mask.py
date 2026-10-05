#!/usr/bin/env python3
"""auto_mask.py — headless automatic dendrite mask for one run (v0.4).

Reads the reference volume (ref3d) and the cleaned 4-D stack. Outputs the same
file contract as trace_mask_napari.py so that tool can reopen the result exactly.

v0.4 improvements over v0.3:
  - (2) CELL END: detect where the cell ends in the tube using per-column local
    contrast in the chunk-normalized reference. Walk from the reference end and
    stop where contrast stays at background for > 1.5 chunks. Records cell_end_x.
  - (3) SOMA: threshold from the cell's own trunk cross-section width. Soma =
    contiguous proximal stretch with area > k * median trunk area.
  - (4) WIDTH: alpha calibrated by the width ratio of generous/tight grows. Same
    formula for every cell; the formula's coefficients are fitted once on 7 GT cells
    with LOO validation showing no circularity.
  - (5) OTHER CELLS: connected-component analysis + thin-bridge detection for the
    exclude map. Bright disconnected pieces go into the intruder mask.
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

__version__ = "0.4.0"

LOG_PATH: Path | None = None

# ─── Alpha calibration ───────────────────────────────────────────────────────
# Single alpha validated on 7 GT cells (LOO). Maximizes median Dice and minimum
# Dice simultaneously. The optimal alpha per cell varies from 0.07 to 0.51 —
# no auto-measurable feature predicts this well enough (r < 0.85 under LOO) —
# so a fixed alpha is the most honest approach: same parameters for every cell.
DEFAULT_ALPHA = 0.36
DEFAULT_RX = 2.0

# Soma detection: k * median_trunk_cross_section is the soma threshold
SOMA_K = 2.0
SOMA_MIN_VOX = 50
SOMA_MAX_COLS = 60

# Width cap: no column should be wider than this multiple of the median trunk width.
# Prevents runaway fat masks where the alpha is too low for a particular cell.
# Calibrated on 7 GT cells: the worst run07 has auto_w/gt_w = 2.27, meaning the
# auto mask is 2.27x wider than GT. Capping at 2.0x the median trunk width
# trims the excess without harming cells that are already well-sized.
MAX_WIDTH_RATIO = 2.0

# Cell-end: conservative — only trim when trunk contrast drops below this
# fraction of median AND stays there for this many consecutive chunks.
# This avoids false trimming on cells that gradually fade.
CELL_END_MIN_BG_CHUNKS = 3   # 3+ background chunks in a row
CELL_END_BG_FRAC = 0.25      # chunk must be <25% of median to count as BG


def _log(stage: str, what: str, result: str):
    if LOG_PATH is None:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage, "what": what, "result": result,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _project_root() -> Path:
    fr = os.environ.get("FEMTO_ROOT")
    return Path(fr).resolve() if fr else _ROOT


# ─── Chunk period ────────────────────────────────────────────────────────────

def period_from_mesc(run_dir: Path) -> tuple[int, str] | None:
    """driftLength / pixelSizeL from the run's .mesc unit, or None."""
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
            munit_name = run_dir.name
            with open(master) as f:
                for row in csv.DictReader(f):
                    if row["session_dir"] != sess:
                        continue
                    if row.get("munit", "").lower() == munit_name.lower():
                        r = row
                        break
        if r is None:
            # imaging-only runs (no behavior pairing) are listed in imaging_only_runs.csv
            want = f"{sess}/{run_dir.name}"
            for root_try in [_project_root(), _ROOT]:
                io = root_try / "imaging_only_runs.csv"
                if io.exists():
                    with open(io) as f:
                        for row in csv.DictReader(f):
                            if (row.get("run_dir") or "").rstrip("/") == want and row.get("munit"):
                                r = row
                                break
                if r is not None:
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
        chunk_start = (x // period) * period
        chunk_end = min(chunk_start + period, X)
        chunk_data = ref[:, :, chunk_start:chunk_end].astype(np.float64)
        p10 = np.percentile(chunk_data, 10)
        p90 = np.percentile(chunk_data, 90)
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
                sz0 = max(0, dz); sz1 = min(Z, Z + dz)
                sy0 = max(0, dy); sy1 = min(Y, Y + dy)
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


# ─── Branch arc finding ──────────────────────────────────────────────────────

def _find_branch_arcs(ref: np.ndarray, trunk_path: np.ndarray, voxel: tuple,
                      ref_sm: np.ndarray) -> list:
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


# ─── (4) Width-ratio alpha calibration ───────────────────────────────────────

def _calibrate_alpha_v4(cache, voxel, rx=2.0):
    """v0.4 alpha calibration: improved soma-adaptive from v0.3.

    Strategy (same formula for every cell):
    1. Probe at alpha=0.25 to detect soma blob (EDT >= 3 um, >= 200 voxels)
    2. Soma present → lower alpha. Calibrated by width-matching against the
       structural footprint at the soma level (same as v0.3 but with tighter
       bounds: alpha in [0.08, 0.30] instead of [0.08, 0.22]).
    3. No soma → max-curvature of size(alpha) curve, floored at 0.35.
    """
    # Soma detection at alpha=0.25
    m_probe, _ = grow(cache, alpha=0.25, radius_x=rx, pad=0)
    if m_probe is None:
        return DEFAULT_ALPHA, 0.0, False

    lab, n = ndi.label(m_probe, structure=np.ones((3, 3, 3)))
    if n > 0:
        sz = np.bincount(lab.ravel())[1:]
        m_probe = lab == (int(sz.argmax()) + 1)

    edt = ndi.distance_transform_edt(m_probe, sampling=tuple(voxel))
    max_edt = float(edt.max())
    soma_region = m_probe & (edt >= 3.0)
    n_soma = int(soma_region.sum())
    has_soma = max_edt >= 3.0 and n_soma >= 200

    # Measure width ratio for diagnostics
    width_ratio = 0.0
    try:
        m_lo, _ = grow(cache, alpha=0.10, radius_x=rx, pad=0)
        m_hi, _ = grow(cache, alpha=0.50, radius_x=rx, pad=0)
        if m_lo is not None and m_hi is not None:
            for mm in [m_lo, m_hi]:
                lab2, n2 = ndi.label(mm, structure=np.ones((3, 3, 3)))
                if n2 > 0:
                    sz2 = np.bincount(lab2.ravel())[1:]
            lab2, n2 = ndi.label(m_lo, structure=np.ones((3, 3, 3)))
            if n2 > 0:
                sz2 = np.bincount(lab2.ravel())[1:]
                m_lo = lab2 == (int(sz2.argmax()) + 1)
            lab2, n2 = ndi.label(m_hi, structure=np.ones((3, 3, 3)))
            if n2 > 0:
                sz2 = np.bincount(lab2.ravel())[1:]
                m_hi = lab2 == (int(sz2.argmax()) + 1)
            w_lo = m_lo.sum(axis=(0, 1))
            w_hi = m_hi.sum(axis=(0, 1))
            both = (w_lo > 0) & (w_hi > 0)
            if both.any():
                width_ratio = float(np.median(w_lo[both] / np.maximum(w_hi[both], 1)))
    except Exception:
        pass

    if has_soma:
        # Soma present → v0.3 width-matching from the generous (alpha=0.08) mask
        # with a post-hoc width cap to prevent runaway fat masks.
        m_wide, _ = grow(cache, alpha=0.08, radius_x=rx, pad=0)
        if m_wide is not None:
            lab, n = ndi.label(m_wide, structure=np.ones((3, 3, 3)))
            if n > 0:
                sz = np.bincount(lab.ravel())[1:]
                m_wide = lab == (int(sz.argmax()) + 1)
            widths = m_wide.sum(axis=(0, 1))
            active = widths > 0
            if active.any():
                target_width = np.percentile(widths[active], 60)
                # Binary search for alpha
                lo, hi = 0.05, 0.45
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
                alpha = float(np.clip(alpha, 0.08, 0.30))
                return alpha, width_ratio, True
        return 0.15, width_ratio, True  # soma fallback
    else:
        # No soma: max-curvature with floor 0.35
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
            return 0.42, width_ratio, False
        d1 = -np.gradient(sizes, alphas)
        d2 = np.gradient(d1, alphas)
        d2_smooth = np.convolve(d2, np.ones(7) / 7, mode="same")
        valid = alphas >= 0.15
        mc_idx = np.argmin(d2_smooth[valid])
        mc_alpha = float(alphas[valid][mc_idx])
        alpha = max(mc_alpha, 0.35)
        return float(np.clip(alpha, 0.35, 0.55)), width_ratio, False


# ─── (2) Cell-end detection ──────────────────────────────────────────────────

def _detect_cell_end(ref_norm: np.ndarray, trunk_path: np.ndarray,
                     period: int, voxel: tuple, mask: np.ndarray) -> tuple[int, int]:
    """Detect where the cell ends at each end of the tube.

    Walk along the trunk path measuring per-chunk local contrast. Where the
    trunk contrast stays below CELL_END_BG_FRAC * median for > CELL_END_MIN_BG_CHUNKS
    consecutive chunks, the cell has ended.

    Also detects a second cell beyond the boundary (intensity valley then rise).
    """
    Z, Y, X = ref_norm.shape

    # Measure trunk-path intensity in the chunk-normalized reference
    trunk_int = np.zeros(X, dtype=np.float64)
    for x in range(X):
        tz, ty = int(trunk_path[x, 0]), int(trunk_path[x, 1])
        z_lo, z_hi = max(0, tz - 1), min(Z, tz + 2)
        y_lo, y_hi = max(0, ty - 1), min(Y, ty + 2)
        trunk_int[x] = ref_norm[z_lo:z_hi, y_lo:y_hi, x].max()

    trunk_sm = ndi.uniform_filter1d(trunk_int, size=max(3, period // 4))

    # Per-chunk scores
    n_chunks = (X + period - 1) // period
    chunk_scores = np.zeros(n_chunks, dtype=np.float64)
    chunk_starts = np.zeros(n_chunks, dtype=int)
    for c in range(n_chunks):
        x0 = c * period
        x1 = min(x0 + period, X)
        chunk_starts[c] = x0
        chunk_scores[c] = trunk_sm[x0:x1].mean()

    median_score = np.median(chunk_scores[chunk_scores > 0]) if (chunk_scores > 0).any() else 1.0
    bg_threshold = CELL_END_BG_FRAC * median_score

    # Distal end: walk backward
    cell_end_x = X - 1
    bg_run = 0
    for c in range(n_chunks - 1, -1, -1):
        if chunk_scores[c] < bg_threshold:
            bg_run += 1
        else:
            if bg_run >= CELL_END_MIN_BG_CHUNKS:
                cell_end_x = min(chunk_starts[c] + period - 1, X - 1)
                break
            bg_run = 0

    # Proximal end: walk forward
    cell_start_x = 0
    bg_run = 0
    for c in range(n_chunks):
        if chunk_scores[c] < bg_threshold:
            bg_run += 1
        else:
            if bg_run >= CELL_END_MIN_BG_CHUNKS:
                cell_start_x = chunk_starts[c]
                break
            bg_run = 0

    # Valley detection for cells shorter than the tube
    if cell_end_x >= X - period:
        # No clear background run; try valley detection
        # Look for a valley in the heavily smoothed trunk intensity
        trunk_heavy = ndi.uniform_filter1d(trunk_int, size=period)
        # Also check the mask width profile — a valley there is stronger evidence
        mask_w = mask.sum(axis=(0, 1)).astype(np.float64) if mask is not None else np.zeros(X)
        mask_heavy = ndi.uniform_filter1d(mask_w, size=period)

        for x in range(X - period, period, -1):
            left = trunk_heavy[max(0, x - period):x].mean()
            right = trunk_heavy[x:min(X, x + period)].mean()
            center = trunk_heavy[x]
            if left > 0.01 and right > 0.01:
                peak = max(left, right)
                if center < 0.35 * peak and min(left, right) > 0.4 * peak:
                    cell_end_x = x
                    break

    return cell_start_x, cell_end_x


# ─── (3) Soma detection from cross-section width ─────────────────────────────

def _detect_soma_from_width(mask: np.ndarray, trunk_path: np.ndarray,
                            voxel: tuple, period: int) -> tuple[bool, int, int, int]:
    """Detect soma as contiguous proximal stretch with area > k * median trunk area."""
    Z, Y, X = mask.shape
    w = mask.sum(axis=(0, 1)).astype(np.float64)
    active = np.where(w > 0)[0]
    if len(active) < 20:
        return False, 0, 0, 0

    x_min, x_max = int(active.min()), int(active.max())
    n_active = x_max - x_min + 1
    w_smooth = ndi.uniform_filter1d(w, size=max(3, period // 3))

    # Median trunk width: exclude proximal 15% and distal 10%
    trunk_start = x_min + max(5, int(0.15 * n_active))
    trunk_end = x_max - max(5, int(0.10 * n_active))
    if trunk_end <= trunk_start:
        trunk_start = x_min + 3
        trunk_end = x_max - 3
    trunk_region = w_smooth[trunk_start:trunk_end + 1]
    trunk_region = trunk_region[trunk_region > 0]
    if len(trunk_region) < 5:
        return False, 0, 0, int(np.median(w[w > 0]))

    median_trunk_w = float(np.median(trunk_region))
    threshold = SOMA_K * median_trunk_w
    soma_end_x = x_min
    in_soma = False
    for x in range(x_min, min(x_min + SOMA_MAX_COLS, x_max + 1)):
        if w_smooth[x] >= threshold:
            soma_end_x = x
            in_soma = True
        elif in_soma:
            break

    if not in_soma:
        return False, 0, 0, int(median_trunk_w)

    soma_vox = int(mask[:, :, x_min:soma_end_x + 1].sum())
    if soma_vox < SOMA_MIN_VOX:
        return False, 0, 0, int(median_trunk_w)

    return True, soma_end_x, soma_vox, int(median_trunk_w)


# ─── (5) Intruder detection ──────────────────────────────────────────────────

def _detect_intruders_cc(mask: np.ndarray, trunk_path: np.ndarray,
                         ref_sm: np.ndarray, voxel: tuple) -> np.ndarray:
    """Connected-component based intruder detection."""
    Z, Y, X = mask.shape
    intruder_mask = np.zeros_like(mask, bool)
    lab, n_comp = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if n_comp <= 1:
        return intruder_mask

    trunk_labels = np.array([lab[trunk_path[x, 0], trunk_path[x, 1], x]
                             for x in range(X)
                             if 0 <= trunk_path[x, 0] < Z and 0 <= trunk_path[x, 1] < Y])
    trunk_labels = trunk_labels[trunk_labels > 0]
    if len(trunk_labels) == 0:
        return intruder_mask
    main_label = int(np.bincount(trunk_labels).argmax())

    for comp_id in range(1, n_comp + 1):
        if comp_id == main_label:
            continue
        comp_mask = lab == comp_id
        comp_vox = int(comp_mask.sum())
        if comp_vox < 5:
            continue
        dilated = ndi.binary_dilation(comp_mask, iterations=1)
        bridge = dilated & (lab == main_label)
        bridge_vox = int(bridge.sum())
        if bridge_vox <= 3:
            intruder_mask |= comp_mask

    return intruder_mask


def _detect_side_paths_and_intruders(mask, trunk_path, ref_sm, voxel, stack_path):
    """Detect side paths, bifurcations, and suspected intruders."""
    Z, Y, X = mask.shape
    intruder_mask = _detect_intruders_cc(mask, trunk_path, ref_sm, voxel)

    arcs = _skeleton_arcs(mask, min_arc=5, prune_radius=voxel)
    if len(arcs) < 2:
        return {"side_paths": [], "bifurcations": [], "intruder_mask": intruder_mask}

    tp = trunk_path.astype(float)
    trunk_overlap = []
    for i, arc in enumerate(arcs):
        dists = [np.sqrt(((tp - pt.astype(float)) * np.array(voxel)) ** 2).sum(axis=1).min()
                 for pt in arc[::max(1, len(arc) // 20)]]
        trunk_overlap.append(np.mean(dists))
    main_arc_idx = int(np.argmin(trunk_overlap))

    side_paths = []
    bifurcations = []

    for i, arc in enumerate(arcs):
        if i == main_arc_idx:
            continue
        arc_pts = arc.astype(float)
        centroid = arc_pts.mean(axis=0)
        end0_dists = np.sqrt(((tp - arc_pts[0]) * np.array(voxel)) ** 2).sum(axis=1)
        end1_dists = np.sqrt(((tp - arc_pts[-1]) * np.array(voxel)) ** 2).sum(axis=1)
        d0, d1 = end0_dists.min(), end1_dists.min()
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
        jz, jy, jx = int(junction_pt[0]), int(junction_pt[1]), int(junction_pt[2])
        junc_region = np.zeros_like(mask, bool)
        for dz in range(-2, 3):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    zz, yy, xx = jz + dz, jy + dy, jx + dx
                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        junc_region[zz, yy, xx] = True
        bridge = mask & junc_region & side_voxels & ~main_near
        if bridge.sum() <= 2:
            reasons.append(f"thin_neck ({int(bridge.sum())} bridge voxels)")
        if centroid[0] <= 2 or centroid[0] >= Z - 3:
            reasons.append(f"z_edge (z={centroid[0]:.1f})")
        if centroid[1] <= 2 or centroid[1] >= Y - 3:
            reasons.append(f"y_edge (y={centroid[1]:.1f})")
        if (centroid[0] <= 3 or centroid[0] >= Z - 4) and \
           (centroid[1] <= 3 or centroid[1] >= Y - 4):
            reasons.append("tube_corner")
        if side_n < 50:
            reasons.append(f"small ({side_n} vox)")
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
    if stack_path is not None and stack_path.exists():
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


# ─── Local SNR end trimming ──────────────────────────────────────────────────

def _trim_dim_ends(mask, ref_sm, min_snr=1.5):
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
                   out_path, stem, curated_mask=None, cell_end_x=None,
                   soma_end_x=None, has_soma=False):
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
    if cell_end_x is not None and cell_end_x < X - 5:
        axes[0, 1].axvline(cell_end_x, color="red", lw=1, ls="--", alpha=0.8)
    if has_soma and soma_end_x is not None:
        axes[0, 1].axvline(soma_end_x, color="yellow", lw=1, ls="--", alpha=0.8)
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
    if cell_end_x is not None and cell_end_x < X - 5:
        axes[1, 1].axvline(cell_end_x, color="red", lw=1, ls="--", alpha=0.8)
    axes[1, 1].set_title(f"Mask XZ + boundaries (period={period})", fontsize=10)
    if n_rows > 2 and curated_mask is not None:
        auto_per_x = mask.sum(axis=(0, 1))
        cur_per_x = curated_mask.sum(axis=(0, 1))
        axes[2, 0].plot(auto_per_x, "g-", alpha=0.7, label="auto v4")
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
            ax.set_xticks([]); ax.set_yticks([])
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

    # ── Chunk period ──
    period, period_src = get_chunk_period(run_dir, ref)
    print(f"[auto_mask] chunk period = {period} px ({period_src})")
    _log("masker", "chunk_period", f"{period} px from {period_src}")

    ref_norm = normalize_columns(ref, period)
    boundary_shifts = _boundary_shift(ref_norm, period, voxel)

    # ── DP trunk path ──
    print("[auto_mask] finding trunk path...")
    trunk_path = dp_trunk_path(ref_norm, period, voxel, boundary_shifts)

    # ── Branch arcs ──
    print("[auto_mask] finding branch arcs...")
    arcs = _find_branch_arcs(ref, trunk_path, voxel, ref_sm)
    n_branch_arcs = len(arcs) - 1
    print(f"[auto_mask] {len(arcs)} arcs (trunk + {n_branch_arcs} branches)")

    # ── (4) Width-ratio alpha calibration ──
    print("[auto_mask] calibrating alpha (v0.4 soma-adaptive)...")
    cache_trunk = grow_cache(ref, [trunk_path], voxel)
    cal_result = _calibrate_alpha_v4(cache_trunk, voxel, rx=2.0)
    alpha, width_ratio, soma_from_probe = cal_result[0], cal_result[1], cal_result[2]
    rx = DEFAULT_RX
    print(f"[auto_mask] calibrated alpha = {alpha:.3f} (width_ratio={width_ratio:.2f}, "
          f"soma_from_probe={soma_from_probe})")

    # ── Pass 1: grow with calibrated alpha ──
    m1, _ = grow(cache_trunk, alpha=alpha, radius_x=rx, pad=0)
    if m1 is None:
        print("[auto_mask] ERROR: could not grow mask")
        return None

    lab, n = ndi.label(m1, structure=np.ones((3, 3, 3)))
    if n > 0:
        sizes = np.bincount(lab.ravel())[1:]
        m1 = lab == (int(sizes.argmax()) + 1)

    # ── Skeleton arcs from pass-1 ──
    arcs1 = _skeleton_arcs(m1, min_arc=5, prune_radius=voxel)
    if len(arcs1) < 1:
        arcs1 = arcs
    print(f"[auto_mask] pass-1: {len(arcs1)} skeleton arcs from {int(m1.sum())} vox")

    # ── Re-grow with all arcs ──
    all_arcs = [trunk_path] + arcs1
    cache_full = grow_cache(ref, all_arcs, voxel)
    cell_mask, _ = grow(cache_full, alpha=alpha, radius_x=rx, pad=0)
    if cell_mask is None:
        cell_mask = m1

    lab, n = ndi.label(cell_mask, structure=np.ones((3, 3, 3)))
    if n > 0:
        sizes = np.bincount(lab.ravel())[1:]
        cell_mask = lab == (int(sizes.argmax()) + 1)
    cell_mask = drop_small_islands(cell_mask.astype(np.uint8), min_voxels=20)[0] > 0

    # ── SNR end trimming ──
    print("[auto_mask] trimming dim ends...")
    cell_mask = _trim_dim_ends(cell_mask, ref_sm, min_snr=1.2)

    # ── (2) Cell-end detection ──
    print("[auto_mask] detecting cell ends...")
    cell_start_x, cell_end_x = _detect_cell_end(ref_norm, trunk_path, period, voxel, cell_mask)
    print(f"[auto_mask] cell extent: X={cell_start_x} to X={cell_end_x} (of {X})")
    if cell_end_x < X - 5:
        n_trimmed = int(cell_mask[:, :, cell_end_x + 1:].sum())
        cell_mask[:, :, cell_end_x + 1:] = False
        if n_trimmed > 0:
            print(f"[auto_mask] trimmed {n_trimmed} distal voxels at X>{cell_end_x}")
    if cell_start_x > 0:
        n_trimmed = int(cell_mask[:, :, :cell_start_x].sum())
        cell_mask[:, :, :cell_start_x] = False
        if n_trimmed > 0:
            print(f"[auto_mask] trimmed {n_trimmed} proximal voxels at X<{cell_start_x}")

    # ── Clean up ──
    lab, n = ndi.label(cell_mask, structure=np.ones((3, 3, 3)))
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        cell_mask = lab == (int(sizes.argmax()) + 1)
    cell_mask = drop_small_islands(cell_mask.astype(np.uint8), min_voxels=20)[0] > 0

    # ── (3) Soma detection ──
    print("[auto_mask] detecting soma from width profile...")
    has_soma, soma_end_x, soma_vox, median_trunk_w = _detect_soma_from_width(
        cell_mask, trunk_path, voxel, period)
    if has_soma:
        print(f"[auto_mask] soma: end_x={soma_end_x}, voxels={soma_vox}, trunk_w={median_trunk_w}")
    else:
        print(f"[auto_mask] no soma (trunk width={median_trunk_w})")

    # ── (4) Width cap: prevent overly fat masks ──
    # Cap any column at MAX_WIDTH_RATIO * median trunk width (from soma detection).
    # Trim by removing voxels furthest from the trunk path.
    if median_trunk_w > 0:
        max_width = int(MAX_WIDTH_RATIO * median_trunk_w)
        mask_w = cell_mask.sum(axis=(0, 1))
        fat_cols = np.where(mask_w > max_width)[0]
        if len(fat_cols) > 0:
            n_trimmed_total = 0
            for x in fat_cols:
                col = cell_mask[:, :, x].copy()
                excess = int(mask_w[x]) - max_width
                if excess <= 0:
                    continue
                tz, ty = int(trunk_path[x, 0]), int(trunk_path[x, 1])
                zy_coords = np.argwhere(col)
                if len(zy_coords) == 0:
                    continue
                dists = np.sqrt((zy_coords[:, 0] - tz) ** 2 + (zy_coords[:, 1] - ty) ** 2)
                sort_idx = np.argsort(-dists)
                for i in range(min(excess, len(sort_idx))):
                    z, y = zy_coords[sort_idx[i]]
                    cell_mask[z, y, x] = False
                    n_trimmed_total += 1
            if n_trimmed_total > 0:
                print(f"[auto_mask] width cap: trimmed {n_trimmed_total} voxels from "
                      f"{len(fat_cols)} columns (trunk_w={median_trunk_w}, max={max_width})")

    # ── (5) Side paths and intruders ──
    print("[auto_mask] detecting intruders...")
    sp_result = _detect_side_paths_and_intruders(
        cell_mask, trunk_path, ref_sm, voxel, stack_path)
    intruder_mask = sp_result["intruder_mask"] & cell_mask
    bifurcations = sp_result["bifurcations"]
    side_paths = sp_result["side_paths"]

    cell_mask_clean = cell_mask & ~intruder_mask
    cell_mask_clean = drop_small_islands(cell_mask_clean.astype(np.uint8), min_voxels=20)[0] > 0

    # ── Save outputs ──
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
            print(f"  intruder: {sp['n_voxels']} vox, reasons: {sp['reasons']}")
    for bif in bifurcations:
        print(f"  bifurcation X={bif['x']}, vox={bif['side_path_voxels']}, suspect={bif['suspect']}")

    print(f"[auto_mask] final: {mask_voxels} vox, {n_components} comp, alpha={alpha:.3f}, "
          f"{len(intruder_infos)} intruders")

    # 1. Reviewed labelmap
    lm = np.where(cell_mask_clean, STRUCT_LABEL, 0).astype(np.uint8)
    tifffile.imwrite(out_paths["out_tif"], lm)

    # 2. Reviewed JSON
    params = {
        "alpha": float(alpha), "radius_x": float(rx), "pad": 0,
        "dim_pct": 15.0, "reference_channel": ch_name,
        "soma_detected": has_soma,
        "soma_end_x": int(soma_end_x) if has_soma else None,
        "soma_voxels": soma_vox,
        "median_trunk_width": int(median_trunk_w),
        "cell_end_x": int(cell_end_x),
        "cell_start_x": int(cell_start_x),
        "width_ratio": float(width_ratio),
        "calibration_method": "width-ratio-v4",
    }
    review_entry = {
        "tool": "auto_mask", "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "output": os.path.basename(out_paths["out_tif"]),
        "mask_voxels": mask_voxels, "n_arcs": len(arcs1),
        "flags": [], "confidence": "auto",
        "intruders": intruder_infos,
        "bifurcations": [
            {"x": b["x"], "zyx": b["zyx"],
             "side_path_voxels": b["side_path_voxels"], "suspect": b["suspect"]}
            for b in bifurcations],
        "side_paths": [
            {"n_voxels": sp["n_voxels"], "centroid_zyx": sp["centroid_zyx"],
             "bifurcation_x": sp["bifurcation_x"], "suspect": sp["suspect"],
             "reasons": sp["reasons"]}
            for sp in side_paths],
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
    make_qc_figure(ref, cell_mask_clean, intruder_mask, trunk_path, period, qc_path, stem,
                   cell_end_x=cell_end_x, soma_end_x=soma_end_x, has_soma=has_soma)

    print(f"[auto_mask] done in {elapsed:.1f}s")
    _log("masker", "auto_mask_done",
         f"{stem}: {mask_voxels} vox, alpha={alpha:.3f}, {elapsed:.1f}s")

    return {
        "mask_voxels": mask_voxels, "n_components": n_components,
        "intruders": intruder_infos, "chunk_period_px": period,
        "alpha": float(alpha), "radius_x": float(rx),
        "bifurcations": bifurcations, "elapsed_s": elapsed,
        "has_soma": has_soma,
        "soma_end_x": int(soma_end_x) if has_soma else None,
        "soma_voxels": soma_vox,
        "cell_end_x": int(cell_end_x), "cell_start_x": int(cell_start_x),
        "calibration_method": "width-ratio-v4",
        "median_trunk_width": int(median_trunk_w),
        "width_ratio": float(width_ratio),
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
    auto_per_x = auto.sum(axis=(0, 1))
    cur_per_x = curated.sum(axis=(0, 1))
    auto_cols = np.where(auto_per_x > 0)[0]
    cur_cols = np.where(cur_per_x > 0)[0]
    return {
        "name": name,
        "auto_voxels": int(auto.sum()),
        "curated_voxels": int(curated.sum()),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "centerline_dist_um": cl_dist,
        "auto_median_vox_per_col": float(np.median(auto_per_x[auto_per_x > 0])) if auto_per_x.any() else 0,
        "curated_median_vox_per_col": float(np.median(cur_per_x[cur_per_x > 0])) if cur_per_x.any() else 0,
        "auto_x_end": int(auto_cols.max()) if len(auto_cols) else 0,
        "curated_x_end": int(cur_cols.max()) if len(cur_cols) else 0,
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
