"""Progress reporting for long steps. Scripts call progress(done, total, label); the control
panel turns the lines into a progress bar and hides them from its log. On a terminal they
print as a compact one-line-per-update status. Zero cost when total is small."""
from __future__ import annotations
import sys, time

_last = {"t": 0.0}


def progress(done: int, total: int, label: str = "", every_s: float = 0.25, force: bool = False):
    """Print '##PROGRESS done/total label'. Throttled to one line per every_s seconds
    (the first and the last update always print)."""
    now = time.time()
    if not force and done not in (0, total) and now - _last["t"] < every_s:
        return
    _last["t"] = now
    print(f"##PROGRESS {int(done)}/{int(total)} {label}", flush=True)
