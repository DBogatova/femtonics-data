#!/usr/bin/env python3
"""femto_gui.py - button-driven control panel for the dendrite pipeline.

One window: the ranked run table (live stage detection, same logic as
femto_status.py - imported, not duplicated), and buttons that run the next
step for the selected run:

  [Run next automatic step]  registration / reference / autoseg / composite,
                             executed as a subprocess with live log output;
                             chains until the run needs a human or is complete
                             (checkbox controls chaining).
  [Open review GUI]          launches trace_mask_napari.py detached (path-guided mask)
  [Open wrap GUI]            launches wrap_segments_napari.py detached
  [Build figure + movies]    coherence + behavior composite, then the ticked movies
  [Build movies only]        just the ticked 3D movies (dual / dynamic / structural)
  [Statistics (all cells)]   per-run metrics + cohort table / summary / figures in stats/
  [Refresh]                  re-scan the disk, update stages

GUI steps are launched as separate processes so napari's own event loop never
fights this panel's. The log pane shows every command verbatim, so anything the
panel does can be reproduced in a terminal.

Run:  $PY code/STEP7_workflow/femto_gui.py
      --selftest   headless check: builds the table model + commands without
                   showing a window, prints PASS/FAIL.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]   # real project root (for finding scripts; always from __file__)
ROOT = Path(os.environ["FEMTO_ROOT"]).resolve() if os.environ.get("FEMTO_ROOT") else CODE_ROOT
sys.path.insert(0, str(HERE))

import femto_status as fs  # single source of truth for stages/commands

PYEXE = sys.executable

# stage -> (label, script, is_gui) ; script args are built per run
AUTO_STAGES = {"stack", "reference", "auto_segmented_pending"}  # informational


def build_runs():
    """femto_status.build_status is the single source of truth."""
    return fs.build_status(ROOT)


def next_command(run: dict, auto: bool = False) -> tuple[str, list[str], bool]:
    """(description, argv, needs_gui) for this run's next step.
    Mirrors femto_status's next_action strings; commands identical.
    auto=True: the mask and the regions are made by the program (STEP9_auto) instead of
    opening the napari tools. It only ever runs where no mask / no regions exist yet, so
    your curated files are never touched; edit the automatic ones with Edit mask/regions."""
    stage = run.get("stage", "?")
    if run.get("mark"):                                   # your decision in run_marks.csv
        return (run["next"]["label"], [], False)
    d = run.get("run_dir")
    stem = run.get("stem") or ""
    stack = run.get("stack")
    base = run.get("behavior_base", "")
    if stage == "not_local":
        return ("fetch 4D stack from cluster (extract on SCC: extract_top7.qsub)", [], False)
    if stage == "stack":
        return ("build reference volume",
                [PYEXE, str(CODE_ROOT / "code/STEP3_auto/make_reference_volume.py"),
                 str(stack), "--register-blocks"], False)
    rd = (ROOT / d) if d else None
    if auto and stage in ("reference", "auto_segmented") and stack:
        return ("automatic mask (program; edit later with 'Edit mask')",
                [PYEXE, str(CODE_ROOT / "code/STEP9_auto/auto_mask.py"), str(stack), "--out-dir", str(rd)], False)
    if auto and stage == "mask_reviewed" and stack:
        m = rd / f"{stem}_autoseg_labelmap_reviewed.tif"; ex = rd / f"{stem}_exclude_labelmap.tif"
        return ("automatic regions (program; edit later with 'Edit regions')",
                [PYEXE, str(CODE_ROOT / "code/STEP9_auto/auto_regions.py"), str(stack), "--mask", str(m),
                 "--out-dir", str(rd)] + (["--exclude", str(ex)] if ex.exists() else []), False)
    if stage == "reference":
        return ("auto-segment cells",
                [PYEXE, str(CODE_ROOT / "code/STEP3_auto/auto_segment.py"), str(stack)], False)
    if stage == "auto_segmented":
        return ("trace + grow mask (napari)",
                [PYEXE, str(CODE_ROOT / "code/STEP3_auto/trace_mask_napari.py"), str(stack)], True)
    if stage == "mask_reviewed":
        return ("one-click wrap soma/trunk/branches (napari)",
                [PYEXE, str(CODE_ROOT / "code/STEP7_workflow/wrap_segments_napari.py"), str(stack)], True)
    if stage in ("segments_located", "coherence_built", "behavior_added"):
        return ("build coherence + behavior composite",
                [PYEXE, str(CODE_ROOT / "code/STEP7_workflow/coherence_with_behavior.py"),
                 "--run", base], False)
    return ("complete - use 'Edit mask' / 'Edit regions' to revise it, then 'Build figure + movies'", [], False)


def provenance(run: dict) -> str:
    """'' (nothing yet), 'auto', 'yours' or 'mixed' for the mask + regions of a run."""
    d, stem = run.get("run_dir"), run.get("stem")
    if not d or not stem:
        return ""
    rd = ROOT / d
    def who(p, key):
        if not p.exists():
            return None
        try:
            j = json.loads(p.read_text())
        except Exception:
            return "yours"
        if key == "mask":
            revs = j.get("reviews") or [{}]
            return "auto" if revs[-1].get("tool") == "auto_mask" else "yours"
        return "auto" if "auto_regions" in j else "yours"
    m = who(rd / f"{stem}_autoseg_reviewed.json", "mask")
    g = who(rd / f"{stem}_segments_final.json", "regions")
    vals = {v for v in (m, g) if v}
    return "" if not vals else (vals.pop() if len(vals) == 1 else "mixed")


# ---------------------------------------------------------------------------
def selftest() -> int:
    runs = build_runs()
    assert len(runs) >= 44, f"expected at least 44 runs, got {len(runs)}"
    stages = {}
    n_cmd = 0
    for r in runs:
        desc, argv, gui = next_command(r)
        next_command(r, auto=True)
        stages[r.get("stage")] = stages.get(r.get("stage"), 0) + 1
        if argv:
            assert Path(argv[1]).exists(), f"missing script: {argv[1]}"
            n_cmd += 1
    print("stage tally:", stages)
    print(f"runnable commands built: {n_cmd}")
    # Count imaging-only runs if present
    n_io = sum(1 for r in runs if r.get("_imaging_only"))
    if n_io:
        print(f"imaging-only runs: {n_io}")
    print("SELFTEST PASS")
    return 0


# ---------------------------------------------------------------------------
def run_gui() -> int:
    from qtpy import QtWidgets, QtCore, QtGui

    class Panel(QtWidgets.QMainWindow):
        log_signal = QtCore.Signal(str)
        progress_signal = QtCore.Signal(int, int, str)        # done, total, label (current step)
        chain_signal = QtCore.Signal(int, int, str)           # step k, n steps, step name
        refresh_signal = QtCore.Signal()

        def __init__(self):
            super().__init__()
            self.setWindowTitle("Femtonics dendrite pipeline")
            self.resize(1120, 640)
            w = QtWidgets.QWidget(); self.setCentralWidget(w)
            lay = QtWidgets.QVBoxLayout(w)

            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(
                ["rank", "run", "quality", "stage", "mask/regions by", "next step"])
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            lay.addWidget(self.table, stretch=3)

            btns = QtWidgets.QHBoxLayout()
            self.b_auto = QtWidgets.QPushButton("Run next automatic step")
            self.b_gui = QtWidgets.QPushButton("Open GUI step")
            self.b_emask = QtWidgets.QPushButton("Edit mask")
            self.b_emask.setToolTip("Open the mask tool on this run, whatever its stage (resumes your saved session)")
            self.b_ereg = QtWidgets.QPushButton("Edit regions")
            self.b_ereg.setToolTip("Open the region tool on this run, whatever its stage")
            self.b_mark = QtWidgets.QPushButton("Mark run…")
            self.b_mark.setToolTip("Exclude this run from everything, or set it aside to re-analyze later. "
                                   "Files are kept; clear the mark to bring it back.")
            self.b_ign = QtWidgets.QPushButton("Ignore regions…")
            self.b_ign.setToolTip("Leave chosen regions (e.g. branch2) out of the figure and all statistics, "
                                  "without changing the mask or regions. Untick to bring them back.")
            self.b_fig = QtWidgets.QPushButton("Build figure + movies")
            self.b_mov = QtWidgets.QPushButton("Build movies only")
            self.b_stats = QtWidgets.QPushButton("Statistics (all cells)")
            self.b_stats.setToolTip("Recompute per-run metrics for every cell with regions and rebuild "
                                    "stats/cohort_* (table, summary, figures)")
            self.b_ref = QtWidgets.QPushButton("Refresh")
            self.chain = QtWidgets.QCheckBox("chain automatic steps")
            self.chain.setChecked(True)
            self.full_auto = QtWidgets.QCheckBox("fully automatic (program makes mask + regions)")
            self.full_auto.setChecked(True)
            self.full_auto.setToolTip("The program draws the mask and picks the regions instead of opening the tools. "
                                      "It never replaces a mask or regions that already exist. Correct anything with "
                                      "Edit mask / Edit regions afterwards.")
            self.b_all = QtWidgets.QPushButton("Automate all runs")
            self.b_all.setToolTip("Every local, unmarked run that is not complete: all automatic steps up to the "
                                  "figure, then the statistics once. Existing masks/regions are kept.")
            for b in (self.b_auto, self.b_all, self.b_gui, self.b_emask, self.b_ereg, self.b_ign, self.b_mark, self.b_fig, self.b_mov, self.b_stats, self.b_ref):
                btns.addWidget(b)
            btns.addWidget(self.chain)
            btns.addWidget(self.full_auto)
            btns.addStretch()
            lay.addLayout(btns)

            movs = QtWidgets.QHBoxLayout()
            movs.addWidget(QtWidgets.QLabel("movies:"))
            self.mv = {}
            for key, label in (("dual", "dual (structure + activity)"),
                               ("time", "dynamic (activity in 3D)"),
                               ("structure", "structural rotation")):
                cb = QtWidgets.QCheckBox(label); cb.setChecked(True)
                self.mv[key] = cb; movs.addWidget(cb)
            self.mv_force = QtWidgets.QCheckBox("rebuild figure + movies even if up to date")
            movs.addWidget(self.mv_force)
            self.hide_other = QtWidgets.QCheckBox("hide other cells (fill with background)")
            self.hide_other.setChecked(True)
            self.hide_other.setToolTip("Cells you marked 'other cell' in the mask tool are replaced by nearby "
                                       "background flicker in movies and the figure picture. Traces never change.")
            movs.addWidget(self.hide_other)
            self.bg_black = QtWidgets.QCheckBox("black background")
            self.bg_black.setChecked(False)          # default: the original, unmasked look
            self.bg_black.setToolTip("Movies and the figure's cell picture show only your cell on black. "
                                     "Traces are never affected.")
            movs.addWidget(self.bg_black)
            movs.addWidget(QtWidgets.QLabel("edge (um):"))
            self.edge = QtWidgets.QDoubleSpinBox(); self.edge.setRange(0.0, 10.0); self.edge.setSingleStep(0.5)
            self.edge.setValue(2.0); self.edge.setToolTip("soft falloff outside the cell; 0 = hard cut")
            movs.addWidget(self.edge); movs.addStretch()
            lay.addLayout(movs)

            prog = QtWidgets.QGridLayout()
            self.chain_label = QtWidgets.QLabel("idle"); self.chain_bar = QtWidgets.QProgressBar()
            self.chain_bar.setRange(0, 1); self.chain_bar.setValue(0); self.chain_bar.setTextVisible(True); self.chain_bar.setFormat("")
            self.step_label = QtWidgets.QLabel(""); self.step_bar = QtWidgets.QProgressBar()
            self.step_bar.setRange(0, 1); self.step_bar.setValue(0); self.step_bar.setFormat("")
            for b in (self.chain_bar, self.step_bar):
                b.setFixedHeight(16)
            prog.addWidget(QtWidgets.QLabel("steps:"), 0, 0); prog.addWidget(self.chain_bar, 0, 1); prog.addWidget(self.chain_label, 0, 2)
            prog.addWidget(QtWidgets.QLabel("current:"), 1, 0); prog.addWidget(self.step_bar, 1, 1); prog.addWidget(self.step_label, 1, 2)
            prog.setColumnStretch(1, 3); prog.setColumnStretch(2, 2)
            lay.addLayout(prog)

            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(5000)
            f = QtGui.QFont("Menlo"); f.setPointSize(11)
            self.log.setFont(f)
            lay.addWidget(self.log, stretch=2)

            self.b_ref.clicked.connect(self.refresh)
            self.b_all.clicked.connect(self.automate_all)
            self.full_auto.toggled.connect(lambda _=None: self.refresh())
            self.b_auto.clicked.connect(lambda: self.dispatch(gui_ok=False))
            self.b_gui.clicked.connect(lambda: self.dispatch(gui_ok=True))
            self.b_stats.clicked.connect(self.build_stats)
            self.b_emask.clicked.connect(lambda: self.reopen("mask"))
            self.b_ereg.clicked.connect(lambda: self.reopen("regions"))
            self.b_ign.clicked.connect(self.edit_ignore)
            self.b_mark.clicked.connect(self.edit_mark)
            self.b_fig.clicked.connect(lambda: self.build_figure(with_movies=True))
            self.b_mov.clicked.connect(lambda: self.build_figure(with_movies=True, figure=False))
            self.log_signal.connect(self.log.appendPlainText)
            self.progress_signal.connect(self._on_progress)
            self.chain_signal.connect(self._on_chain)
            self.refresh_signal.connect(self.refresh)
            self.busy = False
            self.runs = []
            self.refresh()

        # ---- table ------------------------------------------------------
        def refresh(self):
            self.runs = build_runs()
            self.table.setRowCount(len(self.runs))
            for i, r in enumerate(self.runs):
                desc, argv, gui = next_command(r, auto=self.full_auto.isChecked())
                cells = [str(r.get("rank", "")), r.get("behavior_base", ""),
                         str(r.get("quality", r.get("priority", ""))),
                         r.get("stage", "?"), {"auto": "program", "yours": "you", "mixed": "both"}.get(provenance(r), ""),
                         ("[GUI] " if gui else "") + desc]
                for j, txt in enumerate(cells):
                    it = QtWidgets.QTableWidgetItem(txt)
                    if r.get("mark") == "excluded":
                        it.setForeground(QtGui.QColor("#c62828"))
                    elif r.get("mark") == "revisit":
                        it.setForeground(QtGui.QColor("#ef6c00"))
                    elif r.get("stage") == "complete":
                        it.setForeground(QtGui.QColor("#2e7d32"))
                    elif r.get("stage") == "not_local":
                        it.setForeground(QtGui.QColor("#9e9e9e"))
                    self.table.setItem(i, j, it)
            self.table.resizeColumnsToContents()
            self.logline("table refreshed from disk")

        def selected(self):
            i = self.table.currentRow()
            if i < 0 or i >= len(self.runs):
                self.logline("!! select a run first")
                return None
            return self.runs[i]

        def _on_progress(self, done, total, label):
            if total <= 0:
                self.step_bar.setRange(0, 0); self.step_bar.setFormat("")          # busy (unknown length)
            else:
                self.step_bar.setRange(0, total); self.step_bar.setValue(min(done, total))
                self.step_bar.setFormat(f"{100 * min(done, total) // total}%")
            self.step_label.setText(label)

        def _on_chain(self, k, n, name):
            if n <= 0:
                self.chain_bar.setRange(0, 1); self.chain_bar.setValue(0); self.chain_bar.setFormat(""); self.chain_label.setText("idle")
                self.step_bar.setRange(0, 1); self.step_bar.setValue(0); self.step_bar.setFormat(""); self.step_label.setText("")
            else:
                self.chain_bar.setRange(0, n); self.chain_bar.setValue(k); self.chain_bar.setFormat(f"{k}/{n}")
                self.chain_label.setText(name)

        def logline(self, s):
            self.log_signal.emit(s)

        # ---- actions ----------------------------------------------------
        def dispatch(self, gui_ok: bool):
            r = self.selected()
            if r is None or self.busy:
                return
            desc, argv, gui = next_command(r, auto=self.full_auto.isChecked())
            if not argv:
                self.logline(f"[{r.get('behavior_base')}] {desc}")
                return
            if gui and not gui_ok:
                self.logline(f"[{r.get('behavior_base')}] next step is a GUI step "
                             f"({desc}) -> use 'Open GUI step'")
                return
            if gui:
                self.launch_gui(argv, desc)
                return
            threading.Thread(target=self._run_auto, args=(r,), daemon=True).start()

        def launch_gui(self, argv, desc):
            if True:
                self.logline("launch: " + " ".join(argv))
                # start_new_session detaches the child into its own process
                # group: closing napari can never take this panel down, and
                # closing the panel leaves napari alive. A watcher thread
                # reports the child's exit code so crashes are visible.
                child = subprocess.Popen(argv, cwd=str(ROOT),
                                         start_new_session=True,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                def _watch(proc=child, name=desc):
                    rc = proc.wait()
                    self.log_signal.emit(f"[napari closed: {name} exit {rc}] "
                                         "- press Refresh to update stages")
                threading.Thread(target=_watch, daemon=True).start()
                self.logline("napari launched in its own window; press Refresh "
                             "after you Ctrl+S there.")

        def reopen(self, which):
            """Open the mask or region tool on ANY local run, whatever its stage (e.g. to
            revise a completed cell). The tools resume the saved mask/session."""
            r = self.selected()
            if r is None:
                return
            stack = self.stack_path(r)
            if stack is None:
                self.logline(f"[{r.get('behavior_base')}] no local stack - nothing to open"); return
            if which == "mask":
                if not Path(str(stack).replace(".tif", "_ref3d.tif")).exists():
                    self.logline(f"[{r.get('behavior_base')}] no reference volume yet - run the automatic steps first"); return
                self.launch_gui([PYEXE, str(CODE_ROOT / "code/STEP3_auto/trace_mask_napari.py"), str(stack)], "mask tool")
            else:
                if not Path(str(stack).replace(".tif", "_autoseg_labelmap_reviewed.tif")).exists():
                    self.logline(f"[{r.get('behavior_base')}] no saved mask yet - use Edit mask first"); return
                self.launch_gui([PYEXE, str(CODE_ROOT / "code/STEP7_workflow/wrap_segments_napari.py"), str(stack)], "region tool")

        def edit_mark(self):
            """Exclude / revisit later / analyze normally -> run_marks.csv."""
            r = self.selected()
            if r is None:
                return
            sys.path.insert(0, str(CODE_ROOT / "code"))
            from common import run_marks as rm
            base = r.get("behavior_base", "")
            dlg = QtWidgets.QDialog(self); dlg.setWindowTitle(f"Mark run - {base}")
            v = QtWidgets.QVBoxLayout(dlg)
            opts = [(None, "Analyze normally"), ("revisit", "Set aside - re-analyze later"),
                    ("excluded", "Exclude - not analyzable")]
            radios = []
            for key, label in opts:
                rb = QtWidgets.QRadioButton(label); rb.setChecked(r.get("mark") == key); v.addWidget(rb); radios.append((key, rb))
            v.addWidget(QtWidgets.QLabel("Reason:"))
            reason = QtWidgets.QLineEdit(r.get("mark_reason", ""))
            reason.setPlaceholderText("e.g. multiple cells, part of the cell out of frame")
            v.addWidget(reason)
            v.addWidget(QtWidgets.QLabel("Files are kept. A marked run is greyed out here, its automatic steps\n"
                                         "are not run, and it is left out of all statistics."))
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); v.addWidget(bb)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            key = next(k for k, rb in radios if rb.isChecked())
            rm.set_mark(base, key, reason.text().strip())
            self.logline(f"[{base}] " + ({None: "analyzed normally", "revisit": "set aside to re-analyze later",
                                          "excluded": "excluded"}[key]) + (f" - {reason.text().strip()}" if reason.text().strip() else "")
                         + " - press 'Statistics' to update the cohort")
            self.refresh()

        def edit_ignore(self):
            """Tick-box dialog over the run's region names -> <stem>_ignore.json."""
            r = self.selected()
            if r is None:
                return
            stack = self.stack_path(r)
            seg = Path(str(stack).replace(".tif", "_segments_final.tif")) if stack else None
            if seg is None or not seg.exists():
                self.logline(f"[{r.get('behavior_base')}] no regions yet - use Edit regions first"); return
            sys.path.insert(0, str(CODE_ROOT / "code"))
            from common import regions as rg
            names = [n for _, n in sorted(rg._names_from_json(seg).items())]
            cur = {n.lower() for n in rg.ignored_names(seg)}
            dlg = QtWidgets.QDialog(self); dlg.setWindowTitle(f"Ignore regions - {r.get('behavior_base')}")
            v = QtWidgets.QVBoxLayout(dlg)
            v.addWidget(QtWidgets.QLabel("Ticked regions are left out of the figure and all statistics.\n"
                                         "The mask and regions are not changed."))
            boxes = []
            for n in names:
                cb = QtWidgets.QCheckBox(n); cb.setChecked(n.lower() in cur); v.addWidget(cb); boxes.append(cb)
            reason = QtWidgets.QLineEdit(); reason.setPlaceholderText("reason (optional), e.g. suspected other cell")
            v.addWidget(reason)
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); v.addWidget(bb)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            chosen = [cb.text() for cb in boxes if cb.isChecked()]
            rg.main([str(seg), "--clear"])
            if chosen:
                rg.main([str(seg), *chosen] + (["--reason", reason.text().strip()] if reason.text().strip() else []))
            self.logline(f"[{r.get('behavior_base')}] ignored in figure + statistics: {', '.join(chosen) or '(none)'}"
                         " - press 'Build figure + movies' and 'Statistics' to update")

        def stack_path(self, r):
            rd, st = r.get("run_dir"), r.get("stem")
            if not rd or not st:
                return None
            pth = ROOT / rd / f"{st}.tif"
            return pth if pth.exists() else None

        def stats_argv(self):
            return [[PYEXE, str(CODE_ROOT / "code/STEP8_stats/run_metrics.py"), "--all"],
                    [PYEXE, str(CODE_ROOT / "code/STEP8_stats/coupling_phenotype.py"), "--all"],
                    [PYEXE, str(CODE_ROOT / "code/STEP8_stats/behavior_coupling.py"), "--all"],
                    [PYEXE, str(CODE_ROOT / "code/STEP8_stats/cohort_stats.py")]]

        def build_stats(self):
            if self.busy:
                return
            threading.Thread(target=self._run_argv_seq, args=(self.stats_argv(),), daemon=True).start()

        def display_args(self):
            return ((["--mask"] if self.bg_black.isChecked() else ["--no-mask"]) + ["--edge-um", f"{self.edge.value():g}"]
                    + (["--hide-other"] if self.hide_other.isChecked() else ["--show-other"]))

        def build_figure(self, with_movies=True, figure=True):
            r = self.selected()
            if r is None or self.busy:
                return
            base = r.get("behavior_base", "")
            seq = []
            disp = self.display_args()
            if figure:
                seq.append([PYEXE, str(CODE_ROOT / "code/STEP7_workflow/coherence_with_behavior.py"),
                            "--run", base, *disp] + (["--force"] if self.mv_force.isChecked() else []))
            kinds = [k for k, cb in self.mv.items() if cb.isChecked()]
            if with_movies and kinds:
                mv = [PYEXE, str(CODE_ROOT / "code/STEP7_workflow/make_movies.py"), "--run", base,
                      "--kinds", *kinds, *disp]
                if self.mv_force.isChecked():
                    mv.append("--force")
                seq.append(mv)
            if not seq:
                self.logline("nothing selected to build"); return
            if figure:
                seq += self.stats_argv()                 # cohort statistics follow every new figure
            threading.Thread(target=self._run_argv_seq, args=(seq,), daemon=True).start()

        def automate_all(self):
            if self.busy:
                return
            todo = [r for r in self.runs if r.get("stack") and not r.get("mark") and r.get("stage") != "complete"]
            if not todo:
                self.logline("nothing to automate: every local, unmarked run is complete"); return
            self.logline(f"automating {len(todo)} run(s); existing masks/regions are kept")
            threading.Thread(target=self._run_all, args=(todo,), daemon=True).start()

        def _run_all(self, todo):
            self.busy = True
            try:
                for i, r in enumerate(todo, 1):
                    self.logline(f"=== [{i}/{len(todo)}] {r.get('behavior_base')}")
                    for _ in range(8):
                        fresh = [x for x in build_runs() if x.get("behavior_base") == r.get("behavior_base")]
                        if not fresh:
                            break
                        desc, argv, gui = next_command(fresh[0], auto=self.full_auto.isChecked())
                        self.chain_signal.emit(i, len(todo), f"{r.get('behavior_base')}: {desc}")
                        if not argv or gui:
                            break
                        if argv[1].endswith("coherence_with_behavior.py"):
                            argv = argv + self.display_args()
                        if not self._exec(argv, desc):
                            break
                self.logline("=== statistics (all cells)")
                for a in self.stats_argv():
                    self._exec(a)
            finally:
                self.busy = False
                self.refresh_signal.emit()
                QtCore.QTimer.singleShot(4000, lambda: self.chain_signal.emit(0, 0, ""))

        def _run_auto(self, r):
            """Run automatic steps, optionally chaining until GUI/complete."""
            self.busy = True
            try:
                for step_i in range(8):
                    fresh = [x for x in build_runs()
                             if x.get("behavior_base") == r.get("behavior_base")]
                    if not fresh:
                        break
                    desc, argv, gui = next_command(fresh[0], auto=self.full_auto.isChecked())
                    # chain bar: stages left until the GUI step / completion
                    stages = (["stack", "reference", "mask_reviewed", "segments_located"] if self.full_auto.isChecked()
                              else ["stack", "reference", "auto_segmented"])
                    st = fresh[0].get("stage", ""); k = stages.index(st) if st in stages else 0
                    self.chain_signal.emit(k, len(stages), desc)
                    if argv and argv[1].endswith("coherence_with_behavior.py"):
                        argv = argv + self.display_args()          # same options as the buttons
                    if not argv or gui:
                        self.logline(f"[{r.get('behavior_base')}] stopping: {desc}")
                        break
                    if not self._exec(argv, desc):
                        break
                    if not self.chain.isChecked():
                        break
                # the chain reached the end (figure built): make the ticked movies too.
                # make_movies skips anything already up to date, so this is cheap to repeat.
                fresh = [x for x in build_runs() if x.get("behavior_base") == r.get("behavior_base")]
                kinds = [k for k, cb in self.mv.items() if cb.isChecked()]
                if fresh and fresh[0].get("stage") == "complete" and kinds and self.chain.isChecked():
                    mv = [PYEXE, str(CODE_ROOT / "code/STEP7_workflow/make_movies.py"),
                          "--run", r.get("behavior_base"), "--kinds", *kinds,
                          *self.display_args()]
                    if self.mv_force.isChecked():
                        mv.append("--force")
                    self._exec(mv)
                    for a in self.stats_argv():
                        self._exec(a)
            finally:
                self.busy = False
                self.refresh_signal.emit()
                QtCore.QTimer.singleShot(4000, lambda: self.chain_signal.emit(0, 0, ""))

        def _run_argv_seq(self, seq):
            self.busy = True
            try:
                for i, argv in enumerate(seq):
                    self.chain_signal.emit(i, len(seq), Path(argv[1]).stem.replace("_", " "))
                    if not self._exec(argv):
                        break
                else:
                    self.chain_signal.emit(len(seq), len(seq), "all done")
            finally:
                self.busy = False
                self.refresh_signal.emit()
                QtCore.QTimer.singleShot(4000, lambda: self.chain_signal.emit(0, 0, ""))

        def _exec(self, argv, step=None) -> bool:
            name = step or Path(argv[1]).stem.replace("_", " ")
            self.logline("$ " + " ".join(argv))
            self.progress_signal.emit(0, 0, name)                       # busy until the step reports
            p = subprocess.Popen(argv, cwd=str(ROOT), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                line = line.rstrip()
                if line.startswith("##PROGRESS "):
                    try:
                        frac, _, label = line[11:].partition(" ")
                        done, total = (int(x) for x in frac.split("/"))
                        self.progress_signal.emit(done, total, label or name)
                    except ValueError:
                        pass
                    continue                                            # keep the log readable
                self.logline(line)
            p.wait()
            self.progress_signal.emit(1, 1, f"{name}: done" if p.returncode == 0 else f"{name}: FAILED")
            self.logline(f"[exit {p.returncode}]")
            return p.returncode == 0

    app = QtWidgets.QApplication(sys.argv)
    panel = Panel()
    panel.show()
    return app.exec_()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run_gui())
