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
  [Build figure + movies]    coherence + behaviour composite, then the ticked movies
  [Build movies only]        just the ticked 3D movies (dual / dynamic / structural)
  [Refresh]                  re-scan the disk, update stages

GUI steps are launched as separate processes so napari's own event loop never
fights this panel's. The log pane shows every command verbatim, so anything the
panel does can be reproduced in a terminal.

Run:  $PY code/STEP7_workflow/femto_gui.py
      --selftest   headless check: builds the table model + commands without
                   showing a window, prints PASS/FAIL.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import femto_status as fs  # single source of truth for stages/commands

PYEXE = sys.executable

# stage -> (label, script, is_gui) ; script args are built per run
AUTO_STAGES = {"stack", "reference", "auto_segmented_pending"}  # informational


def build_runs():
    """femto_status.build_status is the single source of truth."""
    return fs.build_status(ROOT)


def next_command(run: dict) -> tuple[str, list[str], bool]:
    """(description, argv, needs_gui) for this run's next step.
    Mirrors femto_status's next_action strings; commands identical."""
    stage = run.get("stage", "?")
    d = run.get("run_dir")
    stem = run.get("stem") or ""
    stack = run.get("stack")
    base = run.get("behavior_base", "")
    if stage == "not_local":
        return ("fetch 4D stack from cluster (ask Kiro / see extract_top7.qsub)", [], False)
    if stage == "stack":
        return ("build reference volume",
                [PYEXE, str(ROOT / "code/STEP3_auto/make_reference_volume.py"),
                 str(stack), "--register-blocks"], False)
    if stage == "reference":
        return ("auto-segment cells",
                [PYEXE, str(ROOT / "code/STEP3_auto/auto_segment.py"), str(stack)], False)
    if stage == "auto_segmented":
        return ("trace + grow mask (napari)",
                [PYEXE, str(ROOT / "code/STEP3_auto/trace_mask_napari.py"), str(stack)], True)
    if stage == "mask_reviewed":
        return ("one-click wrap soma/trunk/branches (napari)",
                [PYEXE, str(ROOT / "code/STEP7_workflow/wrap_segments_napari.py"), str(stack)], True)
    if stage in ("segments_located", "coherence_built", "behavior_added"):
        return ("build coherence + behaviour composite",
                [PYEXE, str(ROOT / "code/STEP7_workflow/coherence_with_behavior.py"),
                 "--run", base], False)
    return ("complete - nothing to do", [], False)


# ---------------------------------------------------------------------------
def selftest() -> int:
    runs = build_runs()
    assert len(runs) == 44, f"expected 44 runs, got {len(runs)}"
    stages = {}
    n_cmd = 0
    for r in runs:
        desc, argv, gui = next_command(r)
        stages[r.get("stage")] = stages.get(r.get("stage"), 0) + 1
        if argv:
            assert Path(argv[1]).exists(), f"missing script: {argv[1]}"
            n_cmd += 1
    print("stage tally:", stages)
    print(f"runnable commands built: {n_cmd}")
    print("SELFTEST PASS")
    return 0


# ---------------------------------------------------------------------------
def run_gui() -> int:
    from qtpy import QtWidgets, QtCore, QtGui

    class Panel(QtWidgets.QMainWindow):
        log_signal = QtCore.Signal(str)
        refresh_signal = QtCore.Signal()

        def __init__(self):
            super().__init__()
            self.setWindowTitle("Femtonics dendrite pipeline")
            self.resize(1120, 640)
            w = QtWidgets.QWidget(); self.setCentralWidget(w)
            lay = QtWidgets.QVBoxLayout(w)

            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(
                ["rank", "run", "quality", "stage", "next step"])
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            lay.addWidget(self.table, stretch=3)

            btns = QtWidgets.QHBoxLayout()
            self.b_auto = QtWidgets.QPushButton("Run next automatic step")
            self.b_gui = QtWidgets.QPushButton("Open GUI step")
            self.b_fig = QtWidgets.QPushButton("Build figure + movies")
            self.b_mov = QtWidgets.QPushButton("Build movies only")
            self.b_ref = QtWidgets.QPushButton("Refresh")
            self.chain = QtWidgets.QCheckBox("chain automatic steps")
            self.chain.setChecked(True)
            for b in (self.b_auto, self.b_gui, self.b_fig, self.b_mov, self.b_ref):
                btns.addWidget(b)
            btns.addWidget(self.chain)
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
            self.mv_force = QtWidgets.QCheckBox("rebuild even if up to date")
            movs.addWidget(self.mv_force); movs.addStretch()
            lay.addLayout(movs)

            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(5000)
            f = QtGui.QFont("Menlo"); f.setPointSize(11)
            self.log.setFont(f)
            lay.addWidget(self.log, stretch=2)

            self.b_ref.clicked.connect(self.refresh)
            self.b_auto.clicked.connect(lambda: self.dispatch(gui_ok=False))
            self.b_gui.clicked.connect(lambda: self.dispatch(gui_ok=True))
            self.b_fig.clicked.connect(lambda: self.build_figure(with_movies=True))
            self.b_mov.clicked.connect(lambda: self.build_figure(with_movies=True, figure=False))
            self.log_signal.connect(self.log.appendPlainText)
            self.refresh_signal.connect(self.refresh)
            self.busy = False
            self.runs = []
            self.refresh()

        # ---- table ------------------------------------------------------
        def refresh(self):
            self.runs = build_runs()
            self.table.setRowCount(len(self.runs))
            for i, r in enumerate(self.runs):
                desc, argv, gui = next_command(r)
                cells = [str(r.get("rank", "")), r.get("behavior_base", ""),
                         str(r.get("quality", r.get("priority", ""))),
                         r.get("stage", "?"), ("[GUI] " if gui else "") + desc]
                for j, txt in enumerate(cells):
                    it = QtWidgets.QTableWidgetItem(txt)
                    if r.get("stage") == "complete":
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

        def logline(self, s):
            self.log_signal.emit(s)

        # ---- actions ----------------------------------------------------
        def dispatch(self, gui_ok: bool):
            r = self.selected()
            if r is None or self.busy:
                return
            desc, argv, gui = next_command(r)
            if not argv:
                self.logline(f"[{r.get('behavior_base')}] {desc}")
                return
            if gui and not gui_ok:
                self.logline(f"[{r.get('behavior_base')}] next step is a GUI step "
                             f"({desc}) -> use 'Open GUI step'")
                return
            if gui:
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
                return
            threading.Thread(target=self._run_auto, args=(r,), daemon=True).start()

        def build_figure(self, with_movies=True, figure=True):
            r = self.selected()
            if r is None or self.busy:
                return
            base = r.get("behavior_base", "")
            seq = []
            if figure:
                seq.append([PYEXE, str(ROOT / "code/STEP7_workflow/coherence_with_behavior.py"),
                            "--run", base])
            kinds = [k for k, cb in self.mv.items() if cb.isChecked()]
            if with_movies and kinds:
                mv = [PYEXE, str(ROOT / "code/STEP7_workflow/make_movies.py"), "--run", base,
                      "--kinds", *kinds]
                if self.mv_force.isChecked():
                    mv.append("--force")
                seq.append(mv)
            if not seq:
                self.logline("nothing selected to build"); return
            threading.Thread(target=self._run_argv_seq, args=(seq,), daemon=True).start()

        def _run_auto(self, r):
            """Run automatic steps, optionally chaining until GUI/complete."""
            self.busy = True
            try:
                for _ in range(6):
                    fresh = [x for x in build_runs()
                             if x.get("behavior_base") == r.get("behavior_base")]
                    if not fresh:
                        break
                    desc, argv, gui = next_command(fresh[0])
                    if not argv or gui:
                        self.logline(f"[{r.get('behavior_base')}] stopping: {desc}")
                        break
                    if not self._exec(argv):
                        break
                    if not self.chain.isChecked():
                        break
            finally:
                self.busy = False
                self.refresh_signal.emit()

        def _run_argv_seq(self, seq):
            self.busy = True
            try:
                for argv in seq:
                    if not self._exec(argv):
                        break
            finally:
                self.busy = False
                self.refresh_signal.emit()

        def _exec(self, argv) -> bool:
            self.logline("$ " + " ".join(argv))
            p = subprocess.Popen(argv, cwd=str(ROOT), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                self.logline(line.rstrip())
            p.wait()
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
