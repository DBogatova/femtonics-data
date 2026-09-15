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
# one specific unit, with that unit's own slice count and an explicit name:
python code/STEP1_extract/extract_mesc.py file.mesc --unit MUnit_7 --nz 19 \
       --out .../preprocessed/run5 --out-name run53d
```
- `--nz N` number of Z-planes per volume (must match acquisition; check STEP 1b).
- `--info` just prints the unit list without extracting.
- `--unit MUnit_N` extract a single unit, `--out-name STEM` name the output. Needed
  when one `.mesc` mixes slice counts (a single `--nz` for the whole file is then
  wrong); it aborts if the raw frame count is not divisible by `--nz`.

**Getting `--nz` wrong is silent.** The frame count still divides, so you get a
plausible-looking stack with the planes interleaved incorrectly. Always take `--nz`
from `snake_n_slices` in the summary CSV. `match_behavior_imaging.py` cross-checks
every extracted TIFF against the metadata shape and reports offenders in
`extracted_tif_suspect_nz`.

**`summarize_mesc.py`** — write a metadata CSV (`*.summary.csv`) next to the `.mesc`,
including recording dims, comments, laser, and **pixel size** (`pixel_x_um`,
`pixel_y_um`, `voxel_z_um`, parsed from `MultiROIProtocolJSON`).
```bash
python code/STEP1_extract/summarize_mesc.py path/to/file.mesc
```
Use this to find which MUnit = which run and its voxel size. It also prints a recap of the
real recordings (snapshots excluded) and marks explicitly:
- `scan_type` — `snake` / `ribbon_transverse` / `zstack` / `timeseries` / `snapshot` / `camera_*`.
- `frame_rate_hz` — stored timepoints per second (`1000 / t_step_ms`); for a snake this is
  the **volume** rate, also repeated in `volume_rate_hz`, and `slice_rate_hz`
  (= rate × slices) is the individual-plane rate.
- `snake_n_slices` — Z-planes per volume, filled in only for snake runs; this is the
  `--nz` to pass to `extract_mesc.py` (`n_z`/`z_extent_um`/`z_step_um` give the depth span).
- `pixel_average` — samples averaged per pixel at acquisition (affects the stored data)
  vs `display_average` — the MESc viewer's running average, which does **not**.
  (Checked: `t_step_ms` = pixels-per-timepoint / `pixel_clock_hz` exactly, i.e. every
  scanned line is stored.)
- `duration_calc_s` — length implied by the data (`n_t × t_step_ms`). Prefer it over
  `duration_s` (= `MeasurementLengthInMs`), which is only per frame on some aborted or
  free-running raster series.
- `zstack_n_planes` / `z_step_um` / `frame_loop` / `z_mode` — for `zStack` anatomy units,
  whose frame axis is Z, not time (so they show `n_t = 1`).
- one row **per channel**, so dual-detector units (UG + UR) appear twice.

**`match_behavior_imaging.py`** — pair the behavior recordings to the imaging runs and
write one master table (`behavior_imaging_master.csv` at the project root).
```bash
python code/STEP1_extract/match_behavior_imaging.py .            # scans every mouse/date
```
The pairing key is a hard count, not a guess: the behavior `*_info.txt` reports
**`AndorXylaTrigger` rising edges**, and that is the same pulse train as
**`n_t × snake_n_slices`** in the `.mesc` (one pulse per scanned plane). Equal counts ⇒
same recording. Cross-checked against the behavior `imaging window` length vs
`duration_calc_s` (agrees to <15 ms on every pair).

When several runs in one session share the same count (identical settings repeated),
the tie is broken by (1) a run number written in the `.mesc` comment (`"good run7"`),
else (2) acquisition order — those rows carry `match_confidence =
fingerprint+comment_run` / `fingerprint+order` and an explicit `flags` entry, plus
`order_based_alternative` showing what pure ordering would have said.

Each row also gets `imaging_quality` (parsed from your comment), `rank` (1 = best
usable run), `priority` **P1**→**P4** and `priority_reason`, so sorting by `rank` gives
the usable runs first and pushes bad / behavior-less / ambiguous ones to the bottom.
Rows with no partner are kept and flagged (`no_behavior` / `no_imaging`) rather than
dropped. `extracted_tif` is filled only when an existing TIFF both matches the unit's
`(T,Z,Y,X)` metadata shape *and* is identifiable by filename or run folder — folder
numbering alone is not trusted, since some sessions number `run<N>` by MUnit and
others by behavior run.

**One `.mesc` holding two mice** (e.g. `rbp4_140_141_2026_06_23.mesc`): keep the real
file in one mouse's `raw/` and symlink it (plus its `.summary.csv`) into the other's.
The script detects the shared file and assigns each unit to whichever mouse's behavior
its trigger count matches, so neither session is polluted by the other's runs.
Add folder-name aliases (behavior and `.mesc` filed under different mouse names) to
`FOLDER_ALIASES` at the top of the script.


**`extract_mesc_snapshots.py`** — extract the **single-frame units** (snapshots: camera
overview, 2p reference images taken while hunting for a cell) as 2D TIFFs and the
**`zStack` anatomy volumes** as 3D TIFFs; multi-frame time series are skipped.
```bash
python code/STEP1_extract/extract_mesc_snapshots.py path/to/file.mesc          # writes ../snapshots/
python code/STEP1_extract/extract_mesc_snapshots.py path/to/file.mesc --info   # list only
```
Output: `<date>/snapshots/<stem>_S<session>_MUnit_N_<channel>[_zstack][_comment].tif` with
the pixel size (and z spacing for stacks) stored in ImageJ metadata (µm, so Fiji scale bars
are right), plus `*_snapshots_index.csv` (dims, px size, z step, depth below the labeling
origin, objective, time). Dual-detector units give one TIFF per channel (UG and UR).
`--out DIR` changes the destination, `--max-frames N` also treats short N-frame units as pictures.

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
napari: select the **trace** layer → path tool → click along the dendrite (multiple
paths for branches). `b` = build (footprint shows on the MIP), `Ctrl+S` = save
→ `*_guided_labelmap.tif`.
- `--radius` corridor half-width (thinner/thicker), `--thr-pct` structure sensitivity
  (lower = more), `--enhance` cell (vesselness) boost on the background.
- **Trace from multiple views.** `--proj-axis {z,y,x}` sets the starting projection
  (z=XY top, y=XZ side, x=ZY end-on); switch live with **`a`** (cycle), **`j`/`k`/`l`**
  (XY/XZ/ZY). Each plane has its own **trace** layer, and `b` unions the corridors from
  every projection you drew on (each extruded along its own axis) — so you can pick up a
  branch that's hidden behind others in the top view.
- **Catch briefly-firing branches.** `--activity-pct N` (e.g. `90`) also pulls in corridor
  voxels whose *transient* activity (per-voxel `max − mean` over time) exceeds that
  percentile, so episodically-active branches aren't lost (default `0` = structure only).
  The magenta **transient branches (max−mean)** layer highlights where those are, to guide
  your tracing.
- **Structure guide layer.** A green **structure guide** layer (the thresholding score,
  projected) is shown so you can trace right along the detected structure.
- **Structure reference (experimental).** By default the structure is detected from the
  **temporal max** (each branch caught at its own brightest moment). `--struct-agg cofire`
  instead averages the `--cofire-n` brightest *co-firing* frames (the moments the whole
  dendrite lights up), `--struct-frame N` uses one specific frame you picked (e.g. the
  single clearest frame in the movie), and `--vesselness W` (e.g. `0.5`) adds a Sato
  tubular term to the score. Note: on our test recordings temporal max gave the cleanest
  structure/background separation, and vesselness tends to suppress the soma/thick trunk
  (which belong in the mask) — so these are opt-in knobs to try per-recording, not defaults.
- **Sharpen the guide.** If the structure guide looks blurred, lower `--hp-small` (the
  high-pass low-pass sigma, default `0.5 1 1`) to `0 0.6 0.6` (or `0 0.4 0.4`) for a
  thinner, crisper backbone; raise it to smooth. On our noisier recordings
  `--struct-agg cofire --cofire-n 5 --hp-small 0 0.6 0.6` gave the cleanest guide.

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
- **See what the mask misses.** For 4D input, two reference layers are added from the
  stack: **activity (max)** and the magenta **transient branches** (`max − mean` high-pass,
  on by default). Toggle them with the eye icon and paint the mask to include branches
  that only light up briefly. `--no-refs` skips them (they are temporal-max based and can
  look noisy); combine with **`--frame N`** to show one clean frame (e.g. the same frame
  you traced in the guideline step) as the structure instead of the temporal max.
- **Paint segments here too.** `--segments` turns this into a segment painter: it shows
  the given mask as faint context and lets you paint segment numbers (1,2,3…) on a fresh
  layer with the same slice/3D controls, saving `*_segments_labelmap.tif` clamped to the
  mask (an alternative to STEP 4).

---

### STEP 4 — Pick segments  (`code/STEP4_segments/`)

**`segment_skeleton_napari.py`** — *recommended.* Segment by **clicking, not drawing.**
The mask is skeletonized and auto-broken into many pieces (one per inter-bifurcation
branch); each piece is inflated back to fill the mask by geodesic (through-the-mask,
anisotropic) nearest-seed, so branches that pass close in space but are far along the
dendrite never bleed together. Every piece starts as its own part — a valid
segmentation already — and you just click pieces to group them.
```bash
python code/STEP4_segments/segment_skeleton_napari.py runN_clean.tif runN_edited_labelmap.tif \
    --voxel 0.8 0.9 0.9 --min-branch 4
```
napari (3D): pick a segment number (**`1`..`9`**, **`m`** = new segment), then **click**
a branch to add it to that segment; drag = rotate. Keys: **`u`** undo click, **`r`** reset,
**`[`/`]`** coarser/finer decomposition (fewer/more pieces), **`n`** preview final ids,
**`s`** save → `*_segments_labelmap.tif`. Pieces you never click stay as distinct parts;
grouped pieces share a label; labels are renumbered 1..N and clamped to the mask.
- `--min-branch` drops skeleton spurs shorter than N voxels (absorbed into a neighbour);
  raise it for fewer/larger pieces (or use `[`/`]` live). `--agg`, `--ndisplay {3,2}`.

**`segment_mask_napari.py`** — *paint-based alternative.* Paint 3–5 segments on the mask
footprint; they extrude through the projection axis onto the 3D mask and accumulate across
views.
```bash
python code/STEP4_segments/segment_mask_napari.py runN_clean.tif runN_edited_labelmap.tif --voxel 0.8 0.9 0.9
```
napari: on the **segments** layer, set the label number (1,2,3,4…) and paint each
segment over the footprint. Keys: **`a`** cycle projection (XY→XZ→ZY), **`j`/`k`/`l`**
jump to XY/XZ/ZY, **`d`** toggle 2D projection painting ↔ rotatable **3D edit** (the mask
is pre-filled so the brush always has a surface), **`g`** grow segments from your seeds
through the mask (geodesic nearest-seed: dab one stroke per branch, then `g` fills each
connected branch and splits touching branches at their junction), **`u`** undo the last
grow, **`s`** save → `*_segments_labelmap.tif`. Label #, brush size and paint/erase use
napari's own controls (`P`/`E`, `-`/`=`, `M`, `[`/`]`). Only mask voxels get labeled.
- Paint each part on the view where it doesn't overlap others; for many thin overlapping
  branches, prefer seed + **`g`** grow (which stays precise) over 2D extrusion (which
  labels every mask voxel through the depth).
- `--agg {max,mean}` temporal projection for the anatomy background (default `max`,
  matches refine), `--axis {z,y,x}` starting projection.

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
- `--exclude LABEL [LABEL ...]` drop one or more segment labels (e.g. `--exclude 3`);
  the remaining segments keep their original numbers and colors.
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
- `segment_event_coherence.py` — how coherent are events across the painted segments
  (STEP 4)? Orders segments soma→branch by mean X and reports pairwise ΔF/F correlation,
  event co-participation (global vs isolated "network events"), per-event leader, and the
  soma-end↔branch-end lead/lag (onset-delta + cross-correlation). Figure + `*_network_events.csv`.
  `--window` groups cross-segment events into one network event; `--order`/`--names` label
  roles; `--frame-ms` reports lags in ms.
- `segment_branch_propagation.py` — for each branch-initiated event, how far toward the
  soma does it propagate? Classifies each event as branch-local / reached-trunk /
  reached-soma using a per-segment empirical response threshold (null distribution of
  sliding random windows at false-positive rate `--p`). Figure (propagation profile +
  per-event reach raster) + `*_events.csv`.
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
