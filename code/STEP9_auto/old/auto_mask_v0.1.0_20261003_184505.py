#!/usr/bin/env python3
"""auto_mask.py — headless automatic dendrite mask for one run.

Reads the reference volume (ref3d) and the cleaned 4-D stack. Outputs the same
file contract as trace_mask_napari.py so that tool can reopen the result exactly.

Algorithm
  (a) Chunk period from the .mesc (fallback: column-profile autocorrelation);
      per-column local-contrast normalization of the reference.
  (b) Main trunk = minimum-cost DP path along X through the whole tube, with a
      within-chunk smoothness penalty and predicted boundary shifts.
  (c) Adaptive alpha: sweep grow() alpha from 0.05..0.60, find the max-curvature
      point of the size(alpha) curve (transition from halo removal to structure
      removal), use that as the threshold.
  (d) Thickness from grow()/grow_owned with the adaptive alpha and rx=2.0.
  (e) Intruder detection: thin neck, tube-edge location, temporal decorrelation
      with immediate neighbor.
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

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))

from common.voxel import add_voxel_arg, resolve_voxel
from common.cleanup import drop_small_islands, describe
from STEP3_auto.trace_mask_napari import (
    derive_paths, load_reference, cost_volume, geodesic_path,
    seed_from_reference, _skeleton_arcs, grow_cache, grow, grow_owned,
    owner_caches, load_json_safe, load_session, STRUCT_LABEL,
)

__version__ = "0.1.0"

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


# ─── (a) Chunk period ────────────────────────────────────────────────────────

def period_from_mesc(run_dir: Path) -> tuple[int, str] | None:
    """driftLength / pixelSizeL from the run's .mesc unit, or None."""
    try:
        import h5py
        sys.path.insert(0, str(_ROOT / "code/STEP1_extract"))
        from summarize_mesc import parse_json_attr

        session = run_dir.parents[1] if run_dir.parent.name == "preprocessed" else run_dir.parent
        master = None
        for root_try in [_ROOT, _ROOT / "auto_pipeline"]:
            p = root_try / "behavior_imaging_master.csv"
            if p.exists():
                master = p
                break
        if master is None:
            return None

        sess = str(session.resolve().relative_to(_ROOT.resolve()))
        run_no = int(run_dir.name.replace("run", ""))
        r = next(
            (r for r in csv.DictReader(open(master))
             if r["session_dir"] == sess
             and r.get("behavior_run_number", "").strip() == str(run_no)
             and r.get("munit")),
            None,
        )
        if r is None:
            return None
        mesc = session / "raw" / r["mesc_file"]
        if not mesc.exists():
            for raw in [session / "raw"]:
                if raw.is_symlink():
                    mesc = raw.resolve() / r["mesc_file"]
        if not mesc.exists():
            return None
        with h5py.File(mesc, "r") as f:
            sk = r.get("mesc_session") or next(
                k for k in f if k.startswith("MSession") and r["munit"] in f[k]
            )
            pat = parse_json_attr(f[sk][r["munit"]], "MultiROIProtocolJSON")
            p = next(q for q in pat["scanPatterns"]["patterns"] if q.get("scanMode") == 8)
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


# ─── (a) Per-column local contrast normalization ─────────────────────────────

def normalize_columns(ref: np.ndarray, period: int) -> np.ndarray:
    """Per-column local contrast normalization so dim chunk-ends score like
    bright chunk-starts."""
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


# ─── (b) Main trunk: DP path along X ─────────────────────────────────────────

def _boundary_shift(ref_norm: np.ndarray, period: int, voxel: tuple) -> dict:
    """Predict centroid shift at each chunk boundary via phase cross-correlation."""
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
    """Vectorized DP: optimal (z, y) path along X maximizing brightness with
    a within-chunk smoothness penalty and predicted shifts at boundaries."""
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


# ─── (c) Adaptive alpha ──────────────────────────────────────────────────────

def select_alpha(cache, voxel, rx: float = 2.0) -> float:
    """Find the max-curvature alpha: the transition point between removing
    halo (rapid shrinkage) and removing real structure (slow shrinkage)."""
    alphas = np.arange(0.05, 0.60, 0.01)
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
    d1 = -np.gradient(sizes, alphas)
    d2 = np.gradient(d1, alphas)
    d2_smooth = np.convolve(d2, np.ones(7) / 7, mode="same")

    valid = alphas >= 0.10
    mc_idx = np.argmin(d2_smooth[valid])
    mc_alpha = float(alphas[valid][mc_idx])
    return np.clip(mc_alpha, 0.10, 0.55)


# ─── (e) Intruder detection ──────────────────────────────────────────────────

def detect_intruders(mask: np.ndarray, arcs: list, ref: np.ndarray,
                     stack_path: Path, voxel: tuple) -> list:
    """Detect suspected other-cell pieces (need >= 2 reasons to flag).

    Criteria:
    1. Thin neck (bridge <= 3 voxels after erosion)
    2. Tube edge/corner location
    3. Small isolated piece
    4. Low temporal correlation with immediate neighbor (if 4D stack available)
    """
    Z, Y, X = mask.shape
    intruders = []

    # Check for weakly connected pieces by erosion
    eroded = ndi.binary_erosion(mask, iterations=1)
    lab_e, n_e = ndi.label(eroded, structure=np.ones((3, 3, 3)))
    if n_e <= 1:
        return intruders

    sizes_e = np.bincount(lab_e.ravel())[1:]
    main_lab = int(sizes_e.argmax()) + 1

    for i in range(1, n_e + 1):
        if i == main_lab:
            continue
        piece = lab_e == i
        if piece.sum() < 10:
            continue

        coords = np.argwhere(piece)
        centroid = coords.mean(axis=0)
        reasons = []

        # Thin neck: dilate piece back to find bridge
        dilated = ndi.binary_dilation(piece, iterations=1)
        bridge = dilated & mask & ~piece & ~(lab_e == main_lab)
        if bridge.sum() <= 3:
            reasons.append(f"thin_neck ({int(bridge.sum())} bridge voxels)")

        # Tube edge
        if centroid[0] <= 2 or centroid[0] >= Z - 3:
            reasons.append(f"z_edge (z={centroid[0]:.1f})")
        if centroid[1] <= 2 or centroid[1] >= Y - 3:
            reasons.append(f"y_edge (y={centroid[1]:.1f})")

        # Small size
        if piece.sum() < 80:
            reasons.append(f"small ({int(piece.sum())} vox)")

        if len(reasons) >= 2:
            intruders.append({
                "voxels": int(piece.sum()),
                "centroid_zyx": [float(c) for c in centroid],
                "reasons": reasons,
                "piece_mask": piece,
            })

    # Also check disconnected components in the uneroded mask
    lab_full, n_full = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if n_full > 1:
        sizes_full = np.bincount(lab_full.ravel())[1:]
        main_full = int(sizes_full.argmax()) + 1
        for i in range(1, n_full + 1):
            if i == main_full:
                continue
            piece = lab_full == i
            # Skip if already flagged by erosion check
            if any((piece & intr["piece_mask"]).any() for intr in intruders):
                continue
            coords = np.argwhere(piece)
            centroid = coords.mean(axis=0)
            reasons = [f"disconnected ({int(piece.sum())} vox)"]
            if centroid[0] <= 2 or centroid[0] >= Z - 3:
                reasons.append(f"z_edge (z={centroid[0]:.1f})")
            if centroid[1] <= 2 or centroid[1] >= Y - 3:
                reasons.append(f"y_edge (y={centroid[1]:.1f})")
            if piece.sum() < 100:
                reasons.append(f"small ({int(piece.sum())} vox)")
            if len(reasons) >= 2:
                intruders.append({
                    "voxels": int(piece.sum()),
                    "centroid_zyx": [float(c) for c in centroid],
                    "reasons": reasons,
                    "piece_mask": piece,
                })

    # Temporal correlation check: load a subsample of the 4D stack
    try:
        st = tifffile.TiffFile(str(stack_path))
        pages = st.pages
        n_pages = len(pages)
        nz = Z
        n_frames = n_pages // nz
        n_sample = min(50, n_frames)
        frame_idx = np.linspace(0, n_frames - 1, n_sample, dtype=int)
        sample = np.array([
            np.array([pages[fi * nz + zi].asarray() for zi in range(nz)])
            for fi in frame_idx
        ], dtype=np.float32)  # (T_sample, Z, Y, X)

        main_mask = mask.copy()
        for intr in intruders:
            main_mask &= ~intr["piece_mask"]

        for intr in intruders:
            piece = intr["piece_mask"]
            centroid = intr["centroid_zyx"]
            cx = int(centroid[2])
            x_range = slice(max(0, cx - 10), min(X, cx + 10))
            neighbor = np.zeros_like(main_mask)
            neighbor[:, :, x_range] = main_mask[:, :, x_range]
            if neighbor.sum() < 5 or piece.sum() < 5:
                continue
            pt = sample[:, piece].mean(axis=1)
            pt -= pt.mean()
            nt = sample[:, neighbor].mean(axis=1)
            nt -= nt.mean()
            denom = np.sqrt(np.sum(pt**2) * np.sum(nt**2))
            if denom < 1e-6:
                continue
            r = float(np.sum(pt * nt) / denom)
            if r < 0.35:
                intr["reasons"].append(f"low_local_r ({r:.2f})")
    except Exception:
        pass

    return intruders


# ─── QC figure ────────────────────────────────────────────────────────────────

def make_qc_figure(ref: np.ndarray, mask: np.ndarray, intruder_mask: np.ndarray,
                   trunk_path: np.ndarray, period: int, out_path: Path, stem: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z, Y, X = ref.shape
    fig, axes = plt.subplots(2, 2, figsize=(16, 6), dpi=120)

    ref_mip_xy = ref.max(axis=0)
    axes[0, 0].imshow(ref_mip_xy, cmap="gray", aspect="auto")
    axes[0, 0].set_title("Reference (max Z)", fontsize=10)

    # Mask overlay
    axes[0, 1].imshow(ref_mip_xy, cmap="gray", aspect="auto")
    mask_mip = mask.max(axis=0)
    overlay = np.zeros((*mask_mip.shape, 4))
    overlay[mask_mip > 0] = [0, 1, 0, 0.4]
    if intruder_mask.any():
        int_mip = intruder_mask.max(axis=0)
        overlay[int_mip > 0] = [1, 0, 1, 0.5]
    axes[0, 1].imshow(overlay, aspect="auto")
    # Trunk path
    axes[0, 1].plot(trunk_path[:, 2], trunk_path[:, 1], "c-", linewidth=0.5, alpha=0.7)
    axes[0, 1].set_title(f"Mask ({int(mask.sum())} vox) + trunk path", fontsize=10)

    ref_mip_xz = ref.max(axis=1)
    axes[1, 0].imshow(ref_mip_xz, cmap="gray", aspect="auto")
    axes[1, 0].set_title("Reference (max Y → XZ side)", fontsize=10)

    axes[1, 1].imshow(ref_mip_xz, cmap="gray", aspect="auto")
    mask_mip_xz = mask.max(axis=1)
    overlay_xz = np.zeros((*mask_mip_xz.shape, 4))
    overlay_xz[mask_mip_xz > 0] = [0, 1, 0, 0.4]
    if intruder_mask.any():
        int_mip_xz = intruder_mask.max(axis=1)
        overlay_xz[int_mip_xz > 0] = [1, 0, 1, 0.5]
    axes[1, 1].imshow(overlay_xz, aspect="auto")
    axes[1, 1].plot(trunk_path[:, 2], trunk_path[:, 0], "c-", linewidth=0.5, alpha=0.7)
    for bx in range(period, X, period):
        axes[1, 1].axvline(bx, color="cyan", linewidth=0.3, alpha=0.4)
        axes[0, 1].axvline(bx, color="cyan", linewidth=0.3, alpha=0.3)
    axes[1, 1].set_title(f"Mask XZ + boundaries (period={period})", fontsize=10)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(stem, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def auto_mask(stack_path: str, out_dir: str, voxel_cli=None, debug: bool = False):
    """Run the automatic mask pipeline on one run."""
    t0 = time.time()
    stack_path = Path(stack_path).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stack_path.stem
    run_dir = stack_path.parent

    # ── Voxel ─────────────────────────────────────────────────────────────
    try:
        voxel = resolve_voxel(str(stack_path), voxel_cli, quiet=True)
    except SystemExit:
        voxel = (0.85, 0.8, 0.8)
        print(f"[auto_mask] WARNING: voxel fallback {voxel}")
    print(f"[auto_mask] voxel Z Y X = {voxel[0]:.3f} {voxel[1]:.3f} {voxel[2]:.3f}")

    # ── Output paths ──────────────────────────────────────────────────────
    paths = derive_paths(str(stack_path))
    out_paths = {}
    for key in paths:
        out_paths[key] = str(out_dir / os.path.basename(paths[key]))
    for key in ("ref3d", "ref3d_json", "autoseg", "autoseg_json"):
        out_paths[key] = paths[key]

    # ── Load reference ────────────────────────────────────────────────────
    ref, ch_names, ci, ref_all = load_reference(paths)
    shape = ref.shape
    Z, Y, X = shape
    ch_name = list(ch_names.keys())[ci] if isinstance(ch_names, dict) else ch_names[ci]
    print(f"[auto_mask] reference {shape} channel={ch_name}")

    # ── (a) Chunk period + normalization ──────────────────────────────────
    period, period_src = get_chunk_period(run_dir, ref)
    print(f"[auto_mask] chunk period = {period} px ({period_src})")
    _log("masker", "chunk_period", f"{period} px from {period_src}")

    ref_norm = normalize_columns(ref, period)
    boundary_shifts = _boundary_shift(ref_norm, period, voxel)

    # ── (b) Main trunk path ───────────────────────────────────────────────
    print("[auto_mask] finding main trunk path (DP)...")
    trunk_path = dp_trunk_path(ref_norm, period, voxel, boundary_shifts, smoothness=1.5)
    _log("masker", "trunk_path", f"{len(trunk_path)} pts, X 0-{X-1}")

    # ── (c) Adaptive alpha ────────────────────────────────────────────────
    cache = grow_cache(ref, [trunk_path], voxel)
    rx = 2.0
    alpha = select_alpha(cache, voxel, rx=rx)
    print(f"[auto_mask] adaptive alpha = {alpha:.2f} (max-curvature), rx = {rx:.1f}")

    # ── (d) Grow mask ─────────────────────────────────────────────────────
    mask_raw, unc = grow(cache, alpha=alpha, radius_x=rx, pad=0)
    if mask_raw is None:
        print("[auto_mask] ERROR: could not grow mask")
        return None

    # Keep only the largest connected component
    lab, n = ndi.label(mask_raw, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    cell_mask = lab == (int(sizes.argmax()) + 1) if n > 0 else mask_raw

    # Clean up small islands
    cell_mask = drop_small_islands(cell_mask.astype(np.uint8), min_voxels=20)[0] > 0

    # ── (e) Intruder detection ────────────────────────────────────────────
    print("[auto_mask] detecting intruders...")
    arcs = [trunk_path]
    owners = ["own"]
    intruders = detect_intruders(cell_mask, arcs, ref, stack_path, voxel)

    intruder_mask = np.zeros(shape, bool)
    intruder_infos = []
    for intr in intruders:
        piece = intr["piece_mask"]
        intruder_mask |= piece
        intruder_infos.append({
            "voxels": intr["voxels"],
            "centroid_zyx": intr["centroid_zyx"],
            "reasons": intr["reasons"],
        })
        print(f"  intruder: {intr['voxels']} vox at "
              f"{[f'{c:.1f}' for c in intr['centroid_zyx']]}, "
              f"reasons: {intr['reasons']}")

    # Remove intruders from cell mask
    cell_mask = cell_mask & ~intruder_mask
    cell_mask = drop_small_islands(cell_mask.astype(np.uint8), min_voxels=20)[0] > 0

    # Update arc owners
    final_owners = []
    for arc in arcs:
        in_intruder = intruder_mask[tuple(arc.T)].mean()
        final_owners.append("intruder" if in_intruder > 0.5 else "own")
    owners = final_owners

    # ── Boundary bridge info ──────────────────────────────────────────────
    boundary_bridges = [
        {"x": int(bx), "predicted_shift": [float(s[0]), float(s[1])],
         "accepted": True, "score": 0.0}
        for bx, s in list(boundary_shifts.items())[:15]
    ]

    n_components = int(ndi.label(cell_mask, structure=np.ones((3, 3, 3)))[1])
    mask_voxels = int(cell_mask.sum())
    elapsed = time.time() - t0

    print(f"[auto_mask] final mask: {mask_voxels} voxels, {n_components} component(s), "
          f"alpha={alpha:.2f}, rx={rx:.1f}, "
          f"{len(intruder_infos)} intruder(s)")

    # ── Save outputs ──────────────────────────────────────────────────────
    # 1. Reviewed labelmap (uint8, label 2 = cell)
    lm = np.where(cell_mask, STRUCT_LABEL, 0).astype(np.uint8)
    tifffile.imwrite(out_paths["out_tif"], lm)

    # 2. Reviewed JSON
    params = {
        "alpha": float(alpha), "radius_x": float(rx), "pad": 0,
        "dim_pct": 15.0, "reference_channel": ch_name,
    }
    review_entry = {
        "tool": "auto_mask", "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "output": os.path.basename(out_paths["out_tif"]),
        "mask_voxels": mask_voxels,
        "n_arcs": len(arcs),
        "flags": [],
        "confidence": "auto",
        "intruders": intruder_infos,
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
        "n_intruder_arcs": int(sum(o == "intruder" for o in owners)),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_paths["out_json"], "w") as f:
        json.dump(doc, f, indent=2)

    # 4. Session npz (trace_mask_napari reopening)
    erase = np.zeros(shape, bool)
    add = np.zeros(shape, bool)
    np.savez_compressed(
        out_paths["session"],
        arc_points=np.concatenate(arcs).astype(np.int32) if arcs else np.zeros((0, 3), np.int32),
        arc_lengths=np.array([len(a) for a in arcs], np.int64),
        owners=np.array(owners, dtype="U8"),
        erase=np.packbits(erase.ravel()),
        add=np.packbits(add.ravel()),
        other_erase=np.packbits(erase.ravel()),
        other_add=np.packbits(erase.ravel()),
        other=np.array([-1, -1, -1], float),
        shape=np.array(shape, np.int64),
    )

    # 5. QC figure
    qc_path = out_dir / f"{stem}_automask_qc.png"
    make_qc_figure(ref, cell_mask, intruder_mask, trunk_path, period, qc_path, stem)

    print(f"[auto_mask] done in {elapsed:.1f}s")
    _log("masker", "auto_mask_done",
         f"{stem}: {mask_voxels} vox, {n_components} comp, alpha={alpha:.2f}, {elapsed:.1f}s")

    return {
        "mask_voxels": mask_voxels, "n_components": n_components,
        "intruders": intruder_infos, "chunk_period_px": period,
        "alpha": float(alpha), "radius_x": float(rx), "elapsed_s": elapsed,
    }


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_against_curated(auto_path: str, curated_path: str, voxel: tuple,
                             name: str = "") -> dict:
    """Dice, precision, recall, centerline distance, components."""
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

    return {
        "name": name,
        "auto_voxels": int(auto.sum()),
        "curated_voxels": int(curated.sum()),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "centerline_dist_um": cl_dist,
        "auto_components": int(ndi.label(auto, structure=np.ones((3, 3, 3)))[1]),
        "curated_components": int(ndi.label(curated, structure=np.ones((3, 3, 3)))[1]),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
