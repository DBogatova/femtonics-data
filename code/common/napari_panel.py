"""Shared napari side panel: every action as a button with its key shown, plus a
'next step' hint and a live status area. Used by trace_mask_napari.py and
wrap_segments_napari.py so the two GUIs look and behave the same.

    panel = ActionPanel(viewer, title="wrap regions")
    panel.hint("Press W, then click a piece of dendrite")
    panel.section("Select")
    panel.button("Wrap piece under click", key="w", cb=toggle_wrap, toggle=True)
    panel.button("Undo last", key="u", cb=undo)
    panel.status("3 regions: soma, trunk, branch1")

Buttons call the SAME functions as the key bindings, so nothing can drift apart.
Toggle buttons show their on/off state. The panel is width-capped and scrollable so
it can never crush the canvas (the failure mode we hit with the first review panel).
"""
from __future__ import annotations


class ActionPanel:
    def __init__(self, viewer, title="actions", min_width=220, max_width=320):
        from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame)
        from qtpy.QtCore import Qt
        self._Qt = Qt
        self.viewer = viewer
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumWidth(min_width); self.scroll.setMaximumWidth(max_width)
        self.w = QWidget(); self.lay = QVBoxLayout(self.w); self.lay.setSpacing(4)
        self.scroll.setWidget(self.w)

        self._hint = QLabel(""); self._hint.setWordWrap(True)
        self._hint.setStyleSheet("font-weight: bold; color: #ffd166; padding: 4px;")
        self.lay.addWidget(self._hint)

        self._status = QLabel(""); self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #cfd8dc; padding: 2px 4px;")
        self.lay.addWidget(self._status)
        line = QFrame(); line.setFrameShape(QFrame.HLine); self.lay.addWidget(line)

        self._toggles = {}
        self._stretch_added = False
        viewer.window.add_dock_widget(self.scroll, name=title, area="right")

    # ---- text areas
    def hint(self, text):
        self._hint.setText(text)

    def status(self, text):
        self._status.setText(text)

    # ---- structure
    def section(self, title):
        from qtpy.QtWidgets import QLabel
        lab = QLabel(title.upper()); lab.setStyleSheet("color: #90a4ae; font-size: 10px; margin-top: 6px;")
        self.lay.addWidget(lab)

    def note(self, text):
        from qtpy.QtWidgets import QLabel
        lab = QLabel(text); lab.setWordWrap(True); lab.setStyleSheet("color: #b0bec5; font-size: 11px;")
        self.lay.addWidget(lab)

    def button(self, label, key=None, cb=None, toggle=False, tooltip=None):
        """Add a button. `cb` is called with no arguments (wrap key-bound callbacks with
        a lambda). For toggles, call panel.set_toggle(key, state) to reflect state."""
        from qtpy.QtWidgets import QPushButton
        text = f"{label}   [{key}]" if key else label
        b = QPushButton(text)
        b.setStyleSheet("text-align: left; padding: 4px 6px;")
        b.setFocusPolicy(self._Qt.NoFocus)        # never steal keyboard/scroll focus from the canvas
        if tooltip:
            b.setToolTip(tooltip)
        if cb is not None:
            def _run(*_, cb=cb):
                cb()
                self.refocus_canvas()
            b.clicked.connect(_run)
        if toggle:
            b.setCheckable(True)
            self._toggles[key or label] = (b, label)
        self.lay.addWidget(b)
        return b

    def slider(self, label, lo, hi, value, scale, cb, fmt="{:.2f}"):
        """Labelled horizontal slider; cb(float_value) on change. Returns (slider, label)."""
        from qtpy.QtWidgets import QLabel, QSlider
        lab = QLabel(f"{label}: {fmt.format(value)}"); self.lay.addWidget(lab)
        s = QSlider(self._Qt.Horizontal); s.setRange(int(lo), int(hi)); s.setValue(int(round(value * scale)))
        s.setFocusPolicy(self._Qt.ClickFocus)     # focus only while dragging; wheel over canvas still zooms
        s.sliderReleased.connect(self.refocus_canvas)
        def on(v_i):
            val = v_i / scale; lab.setText(f"{label}: {fmt.format(val)}"); cb(val)
        s.valueChanged.connect(on); self.lay.addWidget(s)
        return s, lab

    def set_toggle(self, key, state: bool):
        if key in self._toggles:
            b, label = self._toggles[key]
            b.blockSignals(True); b.setChecked(bool(state)); b.blockSignals(False)
            b.setText(f"{'● ' if state else '○ '}{label}   [{key}]")

    def refocus_canvas(self):
        """Give keyboard + wheel focus back to the napari canvas (so scroll = zoom, keys work)."""
        try:
            self.viewer.window._qt_viewer.canvas.native.setFocus()
        except Exception:
            try:
                self.viewer.window._qt_window.activateWindow()
            except Exception:
                pass

    def finish(self):
        if not self._stretch_added:
            self.lay.addStretch(1); self._stretch_added = True
