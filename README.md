# Femtonics 3D dendrite pipeline

Tools for turning Femtonics 2p (multi-ROI snake / piezo) `.mesc` recordings into a
reconstructed **single-dendrite 3D mask**, hand-defined **segments**, per-segment
**ΔF/F traces**, and **overlay movies** — built to study how different parts of one
dendrite behave (e.g. branches that fire independently of soma/trunk).

Everything runs from the **project root** with the project virtualenv:

```bash
source .venv/bin/activate          # Python 3.11 + numpy/scipy/scikit-image/tifffile/napari/imageio
# then:  python code/STEPn_.../script.py ...
```

If the venv is missing: `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`

---

## Data layout

Raw and processed data live outside `code/`, per mouse / date / run, e.g.:

```
rbp4_141_phpeb/06-25-2026/
  raw/            rbp4_141_2026_06_25.mesc          # raw acquisition
  preprocessed/runN/  runN3d.tif                    # extracted 4D volume (T,Z,Y,X)
                      *.csv                          # ImageJ ROI traces (optional)
```

**Voxel size matters** and differs per recording. Get it from the metadata (STEP 1b)
and pass it as `--voxel Z Y X` to every step below (e.g. `--voxel 0.8 0.9 0.9`).

---

## Pipeline

### STEP 1 — Extract from `.mesc`  (`code/STEP1_extract/`)

**`extract_mesc.py`** — reshape a recording's raw frames into a 4D `(T,Z,Y,X)` TIFF.
```bash
python code/STEP1_extract/extract_mesc.py path/to/file.mesc --nz 19 --out out_dir/
```
- `--nz N` number of Z-planes per volume (must match acquisition; check STEP 1b).
- `--info` just prints the unit list without extracting.

**`summarize_mesc.py`** — write a metadata CSV (`*.summary.csv`) next to the `.mesc`,
including recording dims, comments, laser, and **pixel size** (`pixel_x_um`,
`pixel_y_um`, `voxel_z_um`, parsed from `MultiROIProtocolJSON`).
```bash
python code/STEP1_extract/summarize_mesc.py path/to/file.mesc
```
Use this to find which MUnit = which run and its voxel size.

---

### STEP 2 — Clean scan lines  (`code/STEP2_clean/`)

**`clean_register_3d.py`** — the snake scan leaves periodic dark columns. This detects
that grid and inpaints it in every frame. Motion correction is available but tends to
warp thin slabs, so we normally skip it (`--no-register`).
```bash
python code/STEP2_clean/clean_register_3d.py runN3d.tif --no-register --out runN/runN_clean.tif
```
- `--no-register` clean only (recommended). Drop it to also 3D-motion-correct
  (`--two-pass`, `--ref-frame N`, `--ref-tif anat.tif`, `--compare-refs` available).

Output: `*_clean.tif` (used by all later steps).

---

### STEP 3 — Build & refine the mask  (`code/STEP3_mask/`)

**`guideline_mask_napari.py`** — *select the dendrite.* Trace the dendrite path on the
MIP; the mask = the bright structure within a thin corridor around your line (captures
exactly what you draw, incl. faint branches; stays on-structure).
```bash
python code/STEP3_mask/guideline_mask_napari.py runN_clean.tif --voxel 0.8 0.9 0.9
```
napari: select **guideline** layer → path tool → click along the dendrite (multiple
paths for branches). `b` = build (footprint shows on the MIP), `Ctrl+S` = save
→ `*_guided_labelmap.tif`.
- `--radius` corridor half-width (thinner/thicker), `--thr-pct` structure sensitivity
  (lower = more), `--enhance` cell (vesselness) boost on the background.

**`refine_mask_napari.py`** — *fix the mask.* Paint to add / erase to remove,
editing 2D slices in any orientation and flipping to 3D to check.
```bash
python code/STEP3_mask/refine_mask_napari.py runN_clean.tif runN_guided_labelmap.tif --voxel 0.8 0.9 0.9 --ndisplay 2
```
Select the mask layer → paintbrush/eraser → `Ctrl+S` saves `*_edited.tif`. Keys:
**`a`** rotate the slice view 90° around X (XY top ↔ XZ side), **`j`** XY, **`k`** XZ,
**`d`** toggle 2D↔3D (drag to rotate). `--frame N` shows one timepoint; `--movie`
overlays on the 4D series. **`--rot-x DEG`** (e.g. 45) rotates the whole volume around
X so you can see/edit a branch hidden in XY/XZ — additions are rotated back and unioned
onto the mask on save (add-only in that mode).

---

### STEP 4 — Pick segments  (`code/STEP4_segments/`)

**`segment_mask_napari.py`** — paint 3–5 segments on the mask footprint; they extrude
through the projection axis onto the 3D mask and accumulate across views.
```bash
python code/STEP4_segments/segment_mask_napari.py runN_clean.tif runN_edited_labelmap.tif --voxel 0.8 0.9 0.9
```
napari: on the **segments** layer, set the label number (1,2,3,4…) and paint each
segment over the footprint. Keys: **`a`** cycle projection (XY→XZ→ZY), **`j`/`k`/`l`**
jump to XY/XZ/ZY, **`s`** save → `*_segments_labelmap.tif`. Paint each part on the view
where it doesn't overlap others; only mask voxels get labeled.

---

### STEP 5 — Per-segment traces  (`code/STEP5_traces/`)

**`extract_segment_traces.py`** — per-segment ΔF/F over time + event timestamps, plus a
figure (segments marked on the cell MIP + stacked traces with a ΔF/F scale bar).
```bash
python code/STEP5_traces/extract_segment_traces.py runN_clean.tif runN_segments_labelmap.tif \
    --proj-axis y --mask runN_edited_labelmap.tif --voxel 0.8 0.9 0.9
```
- Uses your painted labels as segments (or `--force-split --n-segments N` for an auto
  PCA split of a single-label mask).
- `--proj-axis {z,y,x}` view for the segment map (z=XY, y=XZ, x=ZY).
- `--mask` full mask to draw semi-transparently under the segments.
- `--f0-pct` F0 baseline percentile (default 10), `--prom-frac` event sensitivity.

Outputs (next to the stack): per-segment `*_segNN.csv` (Slice,Mean=ΔF/F), a combined
`*_segment_traces.csv`, event timestamps `*_segment_events.csv`, the segment labelmap,
and `*_segment_traces.png`.

---

### STEP 6 — Movie  (`code/STEP6_movie/`)

**`segment_overlay_movie.py`** — MP4 of the projected activity with the segments
overlaid, so you watch each segment light up over time.
```bash
python code/STEP6_movie/segment_overlay_movie.py runN_clean.tif runN_segments_labelmap.tif \
    --proj-axis y --fps 20 --scale 4 --out runN_segments_overlay.mp4
```
- `--proj-axis` projection, `--fps`, `--scale` upscale, `--alpha` overlay opacity.

**`segment_3d_movie.py`** — 3D movie of the dendrite rotating around the X axis with
the segments overlaid semi-transparently. Three modes:
```bash
# static turntable (rotate the structure)
python code/STEP6_movie/segment_3d_movie.py runN_clean.tif runN_segments_labelmap.tif --fps 8

# activity over time in 3D (tilted view)
python code/STEP6_movie/segment_3d_movie.py runN_clean.tif runN_segments_labelmap.tif --time --angle 45 --fps 8

# stacked: structure on top, live activity on bottom, spinning in sync
python code/STEP6_movie/segment_3d_movie.py runN_clean.tif runN_segments_labelmap.tif --dual --rotations 6 --fps 8
```
- `--rotations N` turns across the movie (spin speed), `--fps`, `--angle` tilt,
  `--frames` (turntable resolution), `--time-subsample` to shorten, `--alpha`, `--scale`.

---

## `code/extra/`

Kept but not part of the main pipeline:
- `plot_traces_tool.py` — stacked/overlay plots of ImageJ ROI `Slice,Mean` CSVs
  (anatomical ordering, role colors, `--dff`, Arial, PDF). `plot_run12.py`,
  `plot_traces*.py` are older/one-off versions.
- `extract_dendrite_3d.py` — automatic mask extractor (high-pass + Sato vesselness +
  seed/candidate hysteresis, co-fluctuation, brightest-frame, backbone). Superseded for
  routine use by the guideline workflow, but useful for batch/auto masks.
- `connectivity_napari.py` — 3D connected-component viewer.
- `compartment_coupling.py` — soma/trunk/branch correlation & event-coincidence analysis.
- `make_rotation_movie.py` — movie from pre-rendered rotation-view 4D stacks.

---

## Typical run, end to end

```bash
source .venv/bin/activate
python code/STEP1_extract/summarize_mesc.py rbp4_141_phpeb/06-25-2026/raw/rbp4_141_2026_06_25.mesc
python code/STEP2_clean/clean_register_3d.py rbp4_141_phpeb/06-25-2026/preprocessed/run7/run73d.tif --no-register --out .../run7/run7_clean.tif
python code/STEP3_mask/guideline_mask_napari.py .../run7/run7_clean.tif --voxel 0.8 0.9 0.9        # b, Ctrl+S
python code/STEP3_mask/refine_mask_napari.py   .../run7/run7_clean.tif .../run7_guided_labelmap.tif --voxel 0.8 0.9 0.9 --ndisplay 2   # Ctrl+S
python code/STEP4_segments/segment_mask_napari.py .../run7/run7_clean.tif .../run7_guided_labelmap_edited.tif --voxel 0.8 0.9 0.9      # paint, s
python code/STEP5_traces/extract_segment_traces.py .../run7/run7_clean.tif .../run7_guided_segments_labelmap.tif --proj-axis y --voxel 0.8 0.9 0.9
python code/STEP6_movie/segment_overlay_movie.py   .../run7/run7_clean.tif .../run7_guided_segments_labelmap.tif --proj-axis y
```
