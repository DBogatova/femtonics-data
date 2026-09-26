#!/usr/bin/env python
"""
review_autoseg_napari.py - REVIEW step for the automatic 3D neuron segmentation.

The automatic stage (auto_segment.py, built in parallel) proposes a labelmap and a
sidecar JSON. Hand-tracing a run used to take ~1 hour; this tool exists so the human
step is ~2 minutes: open, glance, fix, save. You never draw from scratch here - you
accept what is right and correct the few things that are wrong.

Input contract (fixed - conform to THIS, not to the auto stage's code)
----------------------------------------------------------------------
For a stack <dir>/<stem>.tif there exist:
  <dir>/<stem>_autoseg_labelmap.tif  uint8 (Z,Y,X); 0=bg; per-cell offset 10
        (cell1 -> labels 1-9, cell2 -> 11-19, cell3 -> 21-29, ...); within a cell
        1=soma, 2=trunk, 3+=branches.  So for a label L: cell index = L//10 (0-based),
        anatomy class = L%10.
  <dir>/<stem>_autoseg.json          {"cells": {name: {"labels":[...], ...}}, "params":..., "stats":...}
Stacks are 4D (T,Z,Y,X) uint16 thin slabs (e.g. 21x19x336), voxel ~0.8 um near-isotropic.

What it does on save (Ctrl+S) - NON-DESTRUCTIVE by construction
---------------------------------------------------------------
  * writes  <stem>_autoseg_labelmap_reviewed.tif  (uint8, same convention, normalized so
    each cell's labels are contiguous and cells are renumbered from 1 with offset 10)
  * appends a review record into  <stem>_autoseg_reviewed.json  (a COPY of the original
    sidecar that accumulates one record per save).
  * It NEVER writes the original <stem>_autoseg_labelmap.tif / <stem>_autoseg.json, and it
    refuses to write any path that looks like a hand-made labelmap
    (*_guided_labelmap.tif, *_segments_labelmap.tif, bare *_labelmap.tif).

Keys (mirror the STEP3/STEP4 napari tools so it feels familiar)
---------------------------------------------------------------
  a            : cycle projection axis  XY -> XZ -> ZY  (in 3D, snap the camera)
  j / k / l    : jump to XY / XZ / ZY directly
  d            : toggle 2D projection painting <-> rotatable 3D volume edit
  f            : MERGE two cells - press over cell A (picks it), then over cell B (merges B into A)
  x            : SPLIT - 1st press arms a scratch brush; paint the region to peel off, then
                 press x again to reassign that region to a brand-new cell id
  c            : relabel the segment under the cursor - cycle anatomy class soma -> trunk -> branch
  h            : hide/show the cell under the cursor (per-cell visibility; also via the dock checkboxes)
  u            : undo the last structural edit (merge / split / relabel / committed paint)
  Ctrl+S       : save reviewed labelmap + append review record to the reviewed JSON

Painting uses napari's own Labels controls
-------------------------------------------
  P / E        : paint / erase        -/= : prev / next label id     M : new label
  [ / ]        : smaller / larger brush        Ctrl+Z : undo a single paint/erase stroke
  In 2D projection, paint reassigns / erases the labelmap's own footprint (extruded along the
  view axis, hidden cells protected); to ADD genuinely new voxels, switch to 3D (d).

Usage
-----
  PY=/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/.venv311/bin/python
  $PY code/STEP3_auto/review_autoseg_napari.py <dir>/<stem>.tif --voxel 0.8 0.8 0.8
  $PY code/STEP3_auto/review_autoseg_napari.py --check [<dir>/<stem>.tif]   # headless self-test

--check runs headless (napari is imported lazily and is NOT required): it verifies the input
contract, exercises the save-path logic into a temp dir, confirms the originals are untouched,
prints PASS/FAIL and exits nonzero on FAIL. With no path it builds its own tiny fixture.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import tifffile
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from common.voxel import add_voxel_arg, resolve_voxel

# ------------------------------------------------------------------ contract constants
CELL_OFFSET = 10          # labels of cell i (0-based) occupy 10*i+1 .. 10*i+9
SCRATCH = 250             # transient paint id used by SPLIT (class 0 -> never a real cell label)
HANDMADE_HINTS = ("_guided_labelmap", "_segments_labelmap", "_guideline_")


def cell_of(label):
    return int(label) // CELL_OFFSET


def class_of(label):
    return int(label) % CELL_OFFSET


def label_of(cell_idx, cls):
    return CELL_OFFSET * int(cell_idx) + int(cls)


def class_name(cls):
    return {1: "soma", 2: "trunk"}.get(cls, "branch%d" % (cls - 2))


def derive_paths(stack_path):
    """From <dir>/<stem>.tif return (stem_path, autoseg_labelmap, autoseg_json)."""
    stem = stack_path.rsplit(".", 1)[0]
    return stem, stem + "_autoseg_labelmap.tif", stem + "_autoseg.json"


def reviewed_paths(stack_path, out_dir=None):
    stem, _, _ = derive_paths(stack_path)
    tif = stem + "_autoseg_labelmap_reviewed.tif"
    js = stem + "_autoseg_reviewed.json"
    if out_dir is not None:
        tif = os.path.join(out_dir, os.path.basename(tif))
        js = os.path.join(out_dir, os.path.basename(js))
    return tif, js


def is_protected(path, original_labelmap=None, original_json=None):
    """Return a reason string if `path` must NOT be written to (an original autoseg output
    or anything that looks like a hand-made labelmap), else None."""
    ap = os.path.abspath(path)
    for orig in (original_labelmap, original_json):
        if orig and ap == os.path.abspath(orig):
            return "is an original autoseg output (%s)" % os.path.basename(orig)
    base = os.path.basename(path).lower()
    for hint in HANDMADE_HINTS:
        if hint in base:
            return "looks like a hand-made labelmap (%s)" % os.path.basename(path)
    # bare *_labelmap.tif that is not one of OUR reviewed/autoseg outputs
    if base.endswith("_labelmap.tif") and "_autoseg" not in base:
        return "looks like a hand-made labelmap (%s)" % os.path.basename(path)
    return None


# ------------------------------------------------------------------ label bookkeeping
def _labels_from_value(val):
    """Pull label ids from a per-cell 'labels' value that may be a flat list
    [1,2,3] or a dict {"soma":1,"trunk":2,"branches":[...]}."""
    ids = set()
    if isinstance(val, dict):
        for key in ("soma", "trunk"):
            if isinstance(val.get(key), (int, np.integer)):
                ids.add(int(val[key]))
        for b in (val.get("branches") or []):
            if isinstance(b, (int, np.integer)):
                ids.add(int(b))
        if "labels" in val:
            ids |= _labels_from_value(val["labels"])
    elif isinstance(val, (list, tuple)):
        for x in val:
            if isinstance(x, (int, np.integer)):
                ids.add(int(x))
    return ids


def collect_json_labels(cells):
    """Robustly pull label ids out of a json 'cells' block, accepting either the
    dict-form {name: {"labels": [...]}} / {name: [...]} or the list-form
    [{"id":1, "labels": {"soma":1,"trunk":2,"branches":[...]}}, ...]."""
    out = set()
    if isinstance(cells, dict):
        entries = cells.values()
    elif isinstance(cells, (list, tuple)):
        entries = cells
    else:
        return out
    for v in entries:
        if isinstance(v, dict):
            out |= _labels_from_value(v.get("labels", v))
        else:
            out |= _labels_from_value(v)
    return out


def normalize_labelmap(seg):
    """Return a labelmap where each cell's labels are contiguous (soma=1, trunk=2 if present,
    branches renumbered from 3) and cells are renumbered from 0 with offset 10. Leftover
    scratch / class-0 voxels are dropped. Preserves the contract exactly."""
    seg = np.asarray(seg).astype(np.int64)
    out = np.zeros_like(seg)
    present = [int(v) for v in np.unique(seg) if v > 0 and (v % CELL_OFFSET) != 0]
    cells = sorted({cell_of(v) for v in present})
    for new_i, ci in enumerate(cells):
        classes = sorted({class_of(v) for v in present if cell_of(v) == ci})
        cmap = {}
        next_branch = 3
        for c in classes:
            if c == 1:
                cmap[c] = 1
            elif c == 2:
                cmap[c] = 2
            else:
                cmap[c] = next_branch
                next_branch += 1
        for c in classes:
            m = (seg == label_of(ci, c))
            out[m] = label_of(new_i, cmap[c])
    return out


def cell_stats(seg):
    """Per-cell summary keyed by 1-based cell number: labels, per-label voxel counts,
    soma/trunk/branch label ids, total voxels."""
    seg = np.asarray(seg)
    labels, counts = np.unique(seg[seg > 0], return_counts=True)
    stats = {}
    for lid, cnt in zip(labels.tolist(), counts.tolist()):
        if lid % CELL_OFFSET == 0:
            continue
        ci = cell_of(lid)
        c = class_of(lid)
        d = stats.setdefault(ci, {"cell": ci + 1, "labels": [], "voxels": {},
                                  "soma": None, "trunk": None, "branches": [], "total": 0})
        d["labels"].append(int(lid))
        d["voxels"][str(int(lid))] = int(cnt)
        d["total"] += int(cnt)
        if c == 1:
            d["soma"] = int(lid)
        elif c == 2:
            d["trunk"] = int(lid)
        else:
            d["branches"].append(int(lid))
    return {str(d["cell"]): d for d in stats.values()}


def build_review_record(seg, source_labelmap, reviewed_labelmap, actions):
    cells = cell_stats(seg)
    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "tool": "review_autoseg_napari.py",
        "source_labelmap": os.path.basename(source_labelmap) if source_labelmap else None,
        "reviewed_labelmap": os.path.basename(reviewed_labelmap),
        "n_cells": len(cells),
        "actions": dict(actions or {}),
        "cells": cells,
    }


# ------------------------------------------------------------------ save (pure, testable)
def save_review(seg, stack_path, out_dir=None, actions=None):
    """Write the reviewed labelmap + append a record into the reviewed JSON. Returns
    (tif_path, json_path). Raises if it would clobber a protected file or overflow uint8."""
    _, orig_lm, orig_js = derive_paths(stack_path)
    tif_path, json_path = reviewed_paths(stack_path, out_dir=out_dir)

    reason = is_protected(tif_path, orig_lm, orig_js)
    if reason:
        raise RuntimeError("refusing to write labelmap: target %s" % reason)
    reason = is_protected(json_path, orig_lm, orig_js)
    if reason:
        raise RuntimeError("refusing to write json: target %s" % reason)

    norm = normalize_labelmap(seg)
    if int(norm.max()) > 255:
        raise RuntimeError("normalized labelmap exceeds uint8 (max=%d); too many cells/classes"
                           % int(norm.max()))
    tifffile.imwrite(tif_path, norm.astype(np.uint8))

    # accumulate into a COPY of the original sidecar; never touch the original
    if os.path.exists(json_path):
        with open(json_path) as fh:
            doc = json.load(fh)
    elif os.path.exists(orig_js):
        with open(orig_js) as fh:
            doc = json.load(fh)
    else:
        doc = {}
    if not isinstance(doc, dict):
        doc = {"original_json_was_not_an_object": True}
    doc.setdefault("reviews", []).append(
        build_review_record(norm, orig_lm, tif_path, actions))
    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return tif_path, json_path


# ------------------------------------------------------------------ contract verification
def verify_contract(stack_path, lm_path, js_path):
    """Return (ok, lines). Checks the fixed input contract without needing napari."""
    lines = []
    state = {"ok": True}

    def add(cond, msg):
        state["ok"] = state["ok"] and bool(cond)
        lines.append(("PASS" if cond else "FAIL") + ": " + msg)
        return bool(cond)

    def warn(msg):
        lines.append("WARN: " + msg)

    have_lm = add(os.path.exists(lm_path), "labelmap exists: %s" % os.path.basename(lm_path))
    have_js = add(os.path.exists(js_path), "json exists: %s" % os.path.basename(js_path))
    add(os.path.exists(stack_path), "stack exists: %s" % os.path.basename(stack_path))
    if not (have_lm and have_js):
        return state["ok"], lines

    lm = tifffile.imread(lm_path)
    add(lm.dtype == np.uint8, "labelmap dtype uint8 (got %s)" % lm.dtype)
    add(lm.ndim == 3, "labelmap is 3D (Z,Y,X) (got ndim %d)" % lm.ndim)

    if os.path.exists(stack_path):
        st = tifffile.imread(stack_path)
        add(st.ndim in (3, 4), "stack ndim in (3,4) (got %d)" % st.ndim)
        spatial = tuple(st.shape[-3:]) if st.ndim == 4 else tuple(st.shape)
        if lm.ndim == 3:
            add(spatial == tuple(lm.shape),
                "stack spatial dims %s == labelmap %s" % (spatial, tuple(lm.shape)))
        if st.dtype != np.uint16:
            warn("stack dtype %s (contract says uint16)" % st.dtype)

    labels = [int(v) for v in np.unique(lm) if v > 0]
    add(all((l % CELL_OFFSET) != 0 for l in labels),
        "every label has anatomy class 1-9 (no multiple of %d)" % CELL_OFFSET)
    add(all(1 <= (l % CELL_OFFSET) <= 9 for l in labels),
        "every label class within 1..9")

    try:
        with open(js_path) as fh:
            doc = json.load(fh)
        json_ok = add(isinstance(doc, dict), "json top-level is an object")
    except Exception as exc:  # noqa: BLE001
        add(False, "json parses (%s)" % exc)
        json_ok = False
    if json_ok:
        add("cells" in doc, "json has a 'cells' block")
        jlabels = collect_json_labels(doc.get("cells", {}))
        add(set(jlabels) == set(labels),
            "json cell labels match labelmap labels (json=%s map=%s)"
            % (sorted(jlabels), sorted(labels)))
        if "params" not in doc:
            warn("json has no 'params' block")
        if "stats" not in doc:
            warn("json has no 'stats' block")
    return state["ok"], lines


# ------------------------------------------------------------------ synthetic fixture
def make_fixture(dirpath):
    """Write a tiny contract-conformant (stack, labelmap, json) trio; return the stack path."""
    T, Z, Y, X = 5, 6, 8, 10
    rng = np.random.default_rng(0)
    st = rng.integers(0, 500, size=(T, Z, Y, X)).astype(np.uint16)
    lm = np.zeros((Z, Y, X), np.uint8)
    # cell 1 (labels 1,2,3): soma / trunk / branch1
    lm[1, 1, 1] = 1; lm[1, 1, 2] = 1
    lm[1, 2, 2] = 2; lm[1, 2, 3] = 2
    lm[1, 3, 3] = 3; lm[1, 3, 4] = 3
    # cell 2 (labels 11,12): soma / trunk
    lm[3, 5, 6] = 11; lm[3, 5, 7] = 11
    lm[3, 6, 7] = 12
    stem = os.path.join(dirpath, "fix")
    stack_path = stem + ".tif"
    tifffile.imwrite(stack_path, st)
    tifffile.imwrite(stem + "_autoseg_labelmap.tif", lm)
    doc = {
        "cells": {
            "1": {"labels": [1, 2, 3], "soma": 1, "trunk": 2, "branches": [3]},
            "2": {"labels": [11, 12], "soma": 11, "trunk": 12, "branches": []},
        },
        "params": {"thr": 0.8, "method": "actstat_mask+NMF+skeleton"},
        "stats": {"n_cells": 2},
    }
    with open(stem + "_autoseg.json", "w") as fh:
        json.dump(doc, fh, indent=2)
    return stack_path


# ------------------------------------------------------------------ headless self-test
def run_check(stack_path):
    made_tmp = None
    if stack_path is None:
        made_tmp = tempfile.mkdtemp(prefix="reviewcheck_")
        stack_path = make_fixture(made_tmp)
        print("[--check] no path given; built synthetic fixture at %s" % stack_path)
    else:
        print("[--check] running against %s" % stack_path)

    all_lines = []
    state = {"ok": True}

    def add(cond, msg):
        state["ok"] = state["ok"] and bool(cond)
        all_lines.append(("PASS" if cond else "FAIL") + ": " + msg)
        return bool(cond)

    _, lm_path, js_path = derive_paths(stack_path)

    # 1) input contract
    ok, clines = verify_contract(stack_path, lm_path, js_path)
    all_lines.extend(clines)
    state["ok"] = state["ok"] and ok

    if os.path.exists(lm_path) and os.path.exists(js_path):
        with open(lm_path, "rb") as fh:
            orig_lm_bytes = fh.read()
        with open(js_path, "rb") as fh:
            orig_js_bytes = fh.read()

        # 2) save-path logic into a temp dir
        out_dir = tempfile.mkdtemp(prefix="reviewout_")
        try:
            seg = tifffile.imread(lm_path).astype(np.uint16)
            actions = {"merges": 1, "splits": 0, "relabels": 2, "paint_strokes": 3}
            tif_path, json_out = save_review(seg, stack_path, out_dir=out_dir, actions=actions)
            add(os.path.exists(tif_path), "reviewed labelmap written")
            rr = tifffile.imread(tif_path)
            add(rr.dtype == np.uint8, "reviewed labelmap dtype uint8 (got %s)" % rr.dtype)
            add(rr.shape == seg.shape,
                "reviewed labelmap shape preserved %s" % (tuple(rr.shape),))
            add(set(int(v) for v in np.unique(rr) if v > 0) ==
                set(int(v) for v in np.unique(normalize_labelmap(seg)) if v > 0),
                "reviewed labels == normalized input labels")
            with open(json_out) as fh:
                doc = json.load(fh)
            add(isinstance(doc.get("reviews"), list) and len(doc["reviews"]) >= 1,
                "reviewed json has a 'reviews' list with >=1 record")
            add("cells" in doc, "reviewed json preserves the original 'cells' block")
            add(doc.get("params", {}).get("thr", None) is not None or "params" not in doc,
                "reviewed json preserves original 'params'")
            rec = doc["reviews"][-1]
            add(all(k in rec for k in ("timestamp", "tool", "reviewed_labelmap", "cells")),
                "review record has the expected keys")
            add(rec["actions"] == actions, "review record stored the action counts")

            # second save must ACCUMULATE, not overwrite
            save_review(seg, stack_path, out_dir=out_dir, actions={"merges": 0})
            with open(json_out) as fh:
                doc2 = json.load(fh)
            add(len(doc2["reviews"]) == 2, "second save appends a second review record")

            # 3) protection logic
            add(is_protected(lm_path, lm_path, js_path) is not None,
                "original autoseg labelmap is refused as a target")
            add(is_protected(js_path, lm_path, js_path) is not None,
                "original autoseg json is refused as a target")
            add(is_protected("/tmp/run7_clean_guided_labelmap.tif") is not None,
                "hand-made guided labelmap is refused")
            add(is_protected("/tmp/run7_clean_segments_labelmap.tif") is not None,
                "hand-made segments labelmap is refused")
            add(is_protected("/tmp/run7_labelmap.tif") is not None,
                "bare hand-made *_labelmap.tif is refused")
            add(is_protected(tif_path, lm_path, js_path) is None,
                "the reviewed output path itself is allowed")
        except Exception as exc:  # noqa: BLE001
            add(False, "save-path exercise raised: %r" % exc)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

        # 4) originals must be byte-for-byte untouched
        with open(lm_path, "rb") as fh:
            add(fh.read() == orig_lm_bytes, "original labelmap bytes unchanged after save")
        with open(js_path, "rb") as fh:
            add(fh.read() == orig_js_bytes, "original json bytes unchanged after save")

    if made_tmp:
        shutil.rmtree(made_tmp, ignore_errors=True)

    print("\n".join(all_lines))
    verdict = "PASS" if state["ok"] else "FAIL"
    print("\n==== --check %s ====" % verdict)
    return 0 if state["ok"] else 1


# ------------------------------------------------------------------ napari GUI
def run_gui(args):
    import napari  # lazy: napari must NOT be required for --check

    stack_path = args.stack
    stem, lm_path, js_path = derive_paths(stack_path)
    for p in (stack_path, lm_path, js_path):
        if not os.path.exists(p):
            print("missing required file: %s" % p, file=sys.stderr)
            return 2

    ok, clines = verify_contract(stack_path, lm_path, js_path)
    print("\n".join(clines))
    if not ok:
        print("contract check FAILED - fix the autoseg outputs before reviewing.",
              file=sys.stderr)
        return 1

    vox = list(resolve_voxel(args.stack, args.voxel))
    stack = tifffile.imread(stack_path)
    if stack.ndim == 4:
        vmax = stack.max(0).astype(np.float32)          # temporal-MIP anatomy
    else:
        vmax = stack.astype(np.float32)
    seg0 = tifffile.imread(lm_path).astype(np.uint16)
    Z, Y, X = seg0.shape

    S = {"seg3d": seg0, "a": {"z": 0, "y": 1, "x": 2}[args.axis], "mode": "2d",
         "hidden": set(), "known_cells": None, "base2d": None, "undo": [],
         "merge_pick": None, "cell_boxes": {}, "L3": {},
         "actions": {"merges": 0, "splits": 0, "relabels": 0, "paint_strokes": 0}}

    AXNAME = {0: "XY (top)", 1: "XZ (side)", 2: "ZY (end-on)"}

    def scale_for(a):
        return tuple(v for i, v in enumerate(vox) if i != a)

    def hidden_zeroed():
        m = S["seg3d"].copy()
        if S["hidden"]:
            m[np.isin(m // CELL_OFFSET, list(S["hidden"]))] = 0
        return m

    def labels2d(a):
        return hidden_zeroed().max(a).astype(np.uint16)

    def clim(a2d):
        lo, hi = np.percentile(a2d, (2, 99.7))
        return float(lo), float(hi)

    def snapshot():
        S["undo"].append(S["seg3d"].copy())
        if len(S["undo"]) > 25:
            S["undo"].pop(0)

    # ----- viewer + layers
    v = napari.Viewer()
    a0 = S["a"]
    anat = v.add_image(vmax.max(a0), name="anatomy (MIP)", colormap="gray",
                       contrast_limits=clim(vmax.max(a0)), scale=scale_for(a0))
    seg_layer = v.add_labels(labels2d(a0), name="autoseg labels",
                             opacity=0.6, scale=scale_for(a0))
    S["base2d"] = np.asarray(seg_layer.data).copy()
    seg_layer.mode = "paint"
    seg_layer.selected_label = 1
    v.scale_bar.visible = True
    v.scale_bar.unit = "um"

    # ----- dockable status + per-cell visibility panel (Qt, GUI-only)
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel, QCheckBox,
                                QGroupBox, QScrollArea)
    from qtpy.QtCore import Qt

    # The panel must never steal space from the canvas: without a width cap an
    # unwrapped status line (per-cell voxel breakdowns get long) forces the
    # dock as wide as the text and leaves no room to draw. Cap the width, wrap
    # the text, and scroll anything that does not fit.
    inner = QWidget()
    pl = QVBoxLayout(inner)
    status_lbl = QLabel("")
    status_lbl.setTextFormat(Qt.PlainText)
    status_lbl.setWordWrap(True)
    pl.addWidget(status_lbl)
    cell_box = QGroupBox("cells (uncheck to hide)")
    cell_layout = QVBoxLayout(cell_box)
    pl.addWidget(cell_box)
    pl.addStretch(1)

    panel = QScrollArea()
    panel.setWidget(inner)
    panel.setWidgetResizable(True)
    panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    panel.setMinimumWidth(180)
    panel.setMaximumWidth(280)
    v.window.add_dock_widget(panel, name="review", area="right")

    def toggle_cell(ci, checked):
        if checked:
            S["hidden"].discard(ci)
        else:
            S["hidden"].add(ci)
        refresh(rebuild=False)

    def rebuild_panel(cells):
        while cell_layout.count():
            item = cell_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        S["cell_boxes"].clear()
        for ci in cells:
            cb = QCheckBox("cell %d" % (ci + 1))
            cb.setChecked(ci not in S["hidden"])
            cb.toggled.connect(lambda checked, c=ci: toggle_cell(c, checked))
            cell_layout.addWidget(cb)
            S["cell_boxes"][ci] = cb

    def update_counts():
        stats = cell_stats(S["seg3d"])
        cells = sorted(int(k) for k in stats)
        title_bits = ["review %s" % os.path.basename(stem)]
        title_bits.append("cells=%d" % len(cells))
        lines = ["mode: %s   proj: %s" % ("3D edit" if S["mode"] == "3d" else "2D",
                                          AXNAME[S["a"]])]
        for k in cells:
            d = stats[str(k)]
            title_bits.append("c%d:%d" % (k, d["total"]))
            parts = []
            for lid in sorted(d["voxels"], key=int):
                parts.append("%s=%d" % (class_name(class_of(int(lid))), d["voxels"][lid]))
            hidden_tag = "  [hidden]" if (k - 1) in S["hidden"] else ""
            lines.append("cell %d (%d vox)%s: %s" % (k, d["total"], hidden_tag, ", ".join(parts)))
        if S["merge_pick"] is not None:
            lines.append("merge: cell %d picked - press f over the cell to merge into it"
                         % (S["merge_pick"] + 1))
        status_lbl.setText("\n".join(lines))
        v.title = "  |  ".join(title_bits)

    def refresh(rebuild=True):
        if S["mode"] == "2d":
            d = labels2d(S["a"])
            seg_layer.data = d
            S["base2d"] = d.copy()
        elif "seg" in S["L3"]:
            S["L3"]["seg"].data = hidden_zeroed()
        cells = sorted({cell_of(v_) for v_ in np.unique(S["seg3d"]) if v_ > 0})
        if rebuild and cells != S["known_cells"]:
            rebuild_panel(cells)
            S["known_cells"] = cells
        else:
            for ci, cb in S["cell_boxes"].items():
                want = ci not in S["hidden"]
                if cb.isChecked() != want:
                    cb.blockSignals(True)
                    cb.setChecked(want)
                    cb.blockSignals(False)
        update_counts()

    # ----- 2D projection <-> 3D volume editing
    def commit2d():
        a = S["a"]
        cur = np.asarray(seg_layer.data).astype(np.uint16)
        base = S["base2d"]
        if base is None or base.shape != cur.shape:
            base = np.zeros_like(cur)
        changed = cur != base
        if changed.any():
            occ = S["seg3d"] > 0
            if S["hidden"]:
                occ &= ~np.isin(S["seg3d"] // CELL_OFFSET, list(S["hidden"]))
            chg3 = np.broadcast_to(np.expand_dims(changed, a), S["seg3d"].shape)
            new3 = np.broadcast_to(np.expand_dims(cur, a), S["seg3d"].shape)
            setm = chg3 & occ & (new3 > 0)
            S["seg3d"][setm] = new3[setm]
            clr = chg3 & occ & (new3 == 0)
            S["seg3d"][clr] = 0
        S["base2d"] = cur.copy()

    def read_from_3d():
        new3 = np.asarray(S["L3"]["seg"].data).astype(np.uint16)
        if S["hidden"]:
            hid = np.isin(S["seg3d"] // CELL_OFFSET, list(S["hidden"]))
            S["seg3d"] = np.where(hid, S["seg3d"], new3)
        else:
            S["seg3d"] = new3

    def sync():
        if S["mode"] == "2d":
            commit2d()
        elif "seg" in S["L3"]:
            read_from_3d()

    def paint_callback(layer, event):
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged and layer.mode in ("paint", "erase"):
            S["actions"]["paint_strokes"] += 1
            sync()
            refresh()

    seg_layer.mouse_drag_callbacks.append(paint_callback)

    def to_3d():
        if S["mode"] == "3d":
            return
        commit2d()
        anat.visible = False
        seg_layer.visible = False
        clim3 = (float(np.percentile(vmax, 2)), float(np.percentile(vmax, 99.7)))
        S["L3"]["anat"] = v.add_image(vmax, name="anatomy 3D", colormap="gray",
                                      scale=tuple(vox), contrast_limits=clim3,
                                      rendering="attenuated_mip")
        lyr = v.add_labels(hidden_zeroed(), name="autoseg labels 3D",
                           scale=tuple(vox), opacity=0.6)
        lyr.n_edit_dimensions = 3
        lyr.brush_size = 3
        lyr.selected_label = 1
        lyr.mode = "paint"
        lyr.mouse_drag_callbacks.append(paint_callback)
        S["L3"]["seg"] = lyr
        S["mode"] = "3d"
        v.dims.ndisplay = 3
        v.layers.selection.active = lyr
        print("3D edit: rotatable volume. PAINT (P) / ERASE (E), -/= pick label, [ / ] brush, "
              "SPACE+drag rotate. Structural keys still work; d returns to 2D.", flush=True)
        update_counts()

    def to_2d():
        if S["mode"] != "3d":
            return
        read_from_3d()
        for ly in list(S["L3"].values()):
            v.layers.remove(ly)
        S["L3"].clear()
        S["mode"] = "2d"
        v.dims.ndisplay = 2
        anat.visible = True
        seg_layer.visible = True
        set_axis(S["a"])
        v.layers.selection.active = seg_layer
        print("2D projection painting mode.", flush=True)

    def set_axis(a):
        if S["mode"] == "2d":
            commit2d()
        S["a"] = a % 3
        a = S["a"]
        if S["mode"] == "3d":
            snap_camera(a)
        else:
            anat.data = vmax.max(a)
            anat.scale = scale_for(a)
            anat.contrast_limits = clim(vmax.max(a))
            d = labels2d(a)
            seg_layer.data = d
            seg_layer.scale = scale_for(a)
            S["base2d"] = d.copy()
            v.reset_view()
        update_counts()
        print("projection: %s" % AXNAME[a], flush=True)

    CANON = {0: ((1, 0, 0), (0, 1, 0)),
             1: ((0, 1, 0), (-1, 0, 0)),
             2: ((0, 0, 1), (-1, 0, 0))}

    def snap_camera(a):
        v.dims.ndisplay = 3
        vd, up = CANON[a]
        try:
            v.camera.set_view_direction(view_direction=vd, up_direction=up)
        except Exception:  # noqa: BLE001
            v.reset_view()

    # ----- picking helpers
    def label_at_cursor():
        layer = S["L3"]["seg"] if S["mode"] == "3d" else seg_layer
        try:
            val = layer.get_value(
                v.cursor.position, world=True,
                view_direction=(v.camera.view_direction if v.dims.ndisplay == 3 else None),
                dims_displayed=list(v.dims.displayed))
        except Exception:  # noqa: BLE001
            try:
                val = layer.get_value(v.cursor.position, world=True)
            except Exception:  # noqa: BLE001
                val = None
        if val is None:
            return 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    def cell_at_cursor():
        lid = label_at_cursor()
        return cell_of(lid) if lid > 0 else None

    # ----- structural operations
    def do_merge(_vw):
        sync()
        ci = cell_at_cursor()
        if ci is None:
            print("merge: hover over a cell first", flush=True)
            return
        if S["merge_pick"] is None:
            S["merge_pick"] = ci
            print("merge: picked cell %d - hover the 2nd cell and press f" % (ci + 1), flush=True)
            update_counts()
            return
        a_cell, b_cell = S["merge_pick"], ci
        S["merge_pick"] = None
        if a_cell == b_cell:
            print("merge: same cell, cancelled", flush=True)
            update_counts()
            return
        snapshot()
        bmask = (S["seg3d"] // CELL_OFFSET == b_cell) & (S["seg3d"] > 0)
        S["seg3d"][bmask] = S["seg3d"][bmask] - CELL_OFFSET * b_cell + CELL_OFFSET * a_cell
        S["actions"]["merges"] += 1
        print("merged cell %d into cell %d" % (b_cell + 1, a_cell + 1), flush=True)
        refresh()

    def do_split(_vw):
        sync()
        if (S["seg3d"] == SCRATCH).any():
            snapshot()
            cells = [cell_of(v_) for v_ in np.unique(S["seg3d"]) if v_ > 0 and v_ != SCRATCH]
            new_i = (max(cells) + 1) if cells else 0
            S["seg3d"][S["seg3d"] == SCRATCH] = label_of(new_i, 1)  # new cell soma
            S["actions"]["splits"] += 1
            print("split: painted region -> new cell %d (relabel its parts with c)"
                  % (new_i + 1), flush=True)
            refresh()
        else:
            layer = S["L3"]["seg"] if S["mode"] == "3d" else seg_layer
            layer.selected_label = SCRATCH
            layer.mode = "paint"
            v.layers.selection.active = layer
            print("split armed: paint the region to peel off (brush is on a scratch id), "
                  "then press x again to make it a new cell.", flush=True)

    def do_relabel(_vw):
        sync()
        lid = label_at_cursor()
        if lid <= 0 or (lid % CELL_OFFSET) == 0:
            print("relabel: hover over a labeled segment first", flush=True)
            return
        ci, c = cell_of(lid), class_of(lid)
        if c == 1:
            nc = 2
        elif c == 2:
            branches = [class_of(v_) for v_ in np.unique(S["seg3d"])
                        if v_ > 0 and cell_of(v_) == ci and class_of(v_) >= 3]
            nc = (max(branches) + 1) if branches else 3
        else:
            nc = 1
        snapshot()
        S["seg3d"][S["seg3d"] == lid] = label_of(ci, nc)
        S["actions"]["relabels"] += 1
        print("cell %d: %s -> %s" % (ci + 1, class_name(c), class_name(nc)), flush=True)
        refresh()

    def do_hide(_vw):
        ci = cell_at_cursor()
        if ci is None:
            print("hide: hover over a cell first", flush=True)
            return
        if ci in S["hidden"]:
            S["hidden"].discard(ci)
            print("cell %d shown" % (ci + 1), flush=True)
        else:
            S["hidden"].add(ci)
            print("cell %d hidden" % (ci + 1), flush=True)
        refresh(rebuild=False)

    def do_undo(_vw):
        if not S["undo"]:
            print("nothing to undo (structural). Ctrl+Z undoes a paint stroke.", flush=True)
            return
        S["seg3d"] = S["undo"].pop()
        print("undid last structural edit", flush=True)
        refresh()

    def do_save(_vw):
        sync()
        try:
            tif_path, json_path = save_review(S["seg3d"], stack_path, actions=S["actions"])
        except Exception as exc:  # noqa: BLE001
            print("SAVE FAILED: %s" % exc, flush=True)
            return
        print("saved -> %s\n         %s (review record appended)"
              % (tif_path, json_path), flush=True)
        update_counts()

    # ----- key bindings (mirror STEP3/STEP4 conventions)
    v.bind_key("a", lambda vw: set_axis(S["a"] + 1), overwrite=True)
    v.bind_key("j", lambda vw: set_axis(0), overwrite=True)
    v.bind_key("k", lambda vw: set_axis(1), overwrite=True)
    v.bind_key("l", lambda vw: set_axis(2), overwrite=True)
    v.bind_key("d", lambda vw: (to_2d() if S["mode"] == "3d" else to_3d()), overwrite=True)
    v.bind_key("f", do_merge, overwrite=True)
    v.bind_key("x", do_split, overwrite=True)
    v.bind_key("c", do_relabel, overwrite=True)
    v.bind_key("h", do_hide, overwrite=True)
    v.bind_key("u", do_undo, overwrite=True)
    v.bind_key("Control-s", do_save, overwrite=True)

    refresh()
    print("REVIEW ready. KEYS: a=cycle proj, j/k/l=XY/XZ/ZY, d=2D<->3D, f=merge, x=split, "
          "c=relabel class, h=hide cell, u=undo, Ctrl+S=save. Paint/erase use napari (P/E, "
          "-/=, M, [ / ]).", flush=True)
    napari.run()
    return 0


# ------------------------------------------------------------------ entrypoint
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", nargs="?", default=None,
                    help="4D (T,Z,Y,X) stack <dir>/<stem>.tif; the autoseg outputs are "
                         "derived as <stem>_autoseg_labelmap.tif / <stem>_autoseg.json")
    add_voxel_arg(ap)
    ap.add_argument("--axis", choices=["z", "y", "x"], default="z",
                    help="starting projection axis (z=XY, y=XZ, x=ZY)")
    ap.add_argument("--check", action="store_true",
                    help="headless contract + save-path self-test (no napari GUI); "
                         "exits nonzero on FAIL")
    args = ap.parse_args(argv)

    if args.check:
        return run_check(args.stack)
    if args.stack is None:
        ap.error("a stack path is required unless --check is used")
    return run_gui(args)


if __name__ == "__main__":
    sys.exit(main())
