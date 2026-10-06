#!/usr/bin/env python3
"""Common amplitude scale for trace figures, so figures of different cells compare.

The coherence figure (code/extra/segment_event_coherence.py) stacks one dF/F trace per
region. Its historical look ('auto') spaces the traces by the run's own largest range, so
the vertical scale differs from run to run and amplitudes cannot be compared by eye
(the 0.5 dF/F scale bar is the only anchor). Two alternatives use ONE value chosen for the
whole curated cohort and stored in stats/plot_scale.json:

  common   traces in dF/F, lane spacing = dff_per_lane, drawn at inch_per_lane inches:
           the same dF/F per inch on every figure, no per-trace rescaling, fixed scale bar.
  zscore   each trace divided by its own noise SD (robust_noise below), lane spacing =
           z_per_lane: shows signal-to-noise, not amplitude.

stats/plot_scale.json is written by code/STEP8_stats/normalized_plots.py --write-scale.
Keep robust_noise identical to code/STEP8_stats/run_metrics.robust_noise (the noise used
for soma_snr), so 'SD' means the same thing in figures and statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

MODES = ("auto", "common", "zscore")
_CODE_ROOT = Path(__file__).resolve().parents[2]


def default_scale_file(root: Path | None = None) -> Path:
    return Path(root or _CODE_ROOT) / "stats" / "plot_scale.json"


def robust_noise(t) -> float:
    """Noise SD of a trace: 1.4826 x MAD of the high-pass residual (trace minus its 15-frame
    running mean). Slow calcium transients are removed by the high-pass, so this measures
    the frame-to-frame (shot) noise, not the activity. Same formula as run_metrics.py."""
    t = np.asarray(t, np.float64)
    hp = t - ndi.uniform_filter1d(t, 15)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-9)


def load_scale(path: Path | str | None = None) -> dict | None:
    p = Path(path) if path else default_scale_file()
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    need = ("dff_per_lane", "z_per_lane", "inch_per_lane", "dff_scale_bar", "z_scale_bar")
    return d if all(k in d for k in need) else None


def nice_ceil(x: float) -> float:
    """Round up to 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^k (readable lane values)."""
    if not np.isfinite(x) or x <= 0:
        return float("nan")
    k = np.floor(np.log10(x)); m = x / 10 ** k
    for s in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m <= s + 1e-9:
            return float(round(s * 10 ** k, 6))
    return float(10 ** (k + 1))


def nice_bar(lane: float) -> float:
    """A round scale-bar length of about a third of a lane."""
    target = lane / 3.0
    k = np.floor(np.log10(target)); m = target / 10 ** k
    s = 1 if m < 1.5 else 2 if m < 3.5 else 5 if m < 7.5 else 10
    return float(round(s * 10 ** k, 6))
