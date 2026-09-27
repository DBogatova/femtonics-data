# Dendrite pipeline — from 4D stack to coherence+behavior figure

One command drives everything:

```bash
cd /Users/daria/Desktop/femtonics-data
PY=/Users/daria/Desktop/Boston_University/Devor_Lab/apical-dendrites-2025/.venv311/bin/python
$PY code/STEP7_workflow/femto_status.py --next
```

It prints which run to work on and the exact command for its next stage.
`--next --run-it` executes non-GUI stages directly. Or use the button panel:

```bash
$PY code/STEP7_workflow/femto_gui.py
```

## The stages

| # | stage | command | manual? |
|---|-------|---------|---------|
| 0 | get stack local | extract MUnit from .mesc on SCC (`Femtonics/deepcad/extract_top7.qsub` is the template), rsync to `<mouse>/<MM-DD-YYYY>/preprocessed/runN/` | needs live `ssh scc` |
| 1 | register | `code/STEP2_clean/clean_register_3d.py <4D.tif> --out <dir>/runN_clean.tif` | no |
| 2 | reference volume | `code/STEP3_auto/make_reference_volume.py <dir>/runN_clean.tif` (add `--register-blocks` if it refuses on drift) | no |
| 3 | cell proposal | `code/STEP3_auto/auto_segment.py <dir>/runN_clean.tif` | no |
| 4 | trace + grow mask | `code/STEP3_auto/trace_mask_napari.py <dir>/runN_clean.tif` — centreline seeded from reference ridges; `t`+click two points traces a geodesic path, `x`+click deletes an arc; **alpha slider** = relative threshold (keep voxels ≥ alpha × local centreline intensity: removes halo around trunk and faint branch alike), radius-x and pad sliders; orange = dim centreline (uncertain, never auto-bridged); Ctrl+S. (`review_autoseg_napari.py` remains as the cell-toggle alternative.) | ~2 min GUI |
| 5 | pick regions | `code/STEP7_workflow/wrap_segments_napari.py <dir>/runN_clean.tif` — `w` then click: whole junction-to-junction piece becomes a region (soma/trunk/branch auto-guessed; `s`/`t`/`b` override). `i` = **interval mode**: click two points and the stretch between them along the skeleton becomes a region (e.g. trunk 50–80 µm from soma, proximal 20 µm of a branch). `m` merges the last two regions, `n` names the last one (names go to the JSON and figures), `u` undo, Ctrl+S | ~1–2 min GUI |
| 6 | figure | `code/STEP7_workflow/coherence_with_behavior.py --run <behavior_base>` | no |

Stage detection is by disk presence (which files exist), so
`processing_status.csv` cannot disagree with reality.

## Naming conventions (load-bearing)

- cleaned stack: `runN_clean.tif` — everything downstream finds files by this stem
- outputs: `runN_clean_ref3d.tif`, `runN_clean_autoseg_labelmap.tif` (+`_reviewed`),
  `runN_clean_segments_final.tif`, `runN_clean_coherence_full.{png,pdf}`
- hand-made labelmaps and figures are never overwritten by any tool

## Prerequisites per run

1. behavior published under `<mouse>/<MM-DD-YYYY>/{behavior,trigger}`
   (SCC behavior pipeline; done for all 44 current runs)
2. run matched in `behavior_imaging_master.csv`
   (`code/STEP1_extract/match_behavior_imaging.py`; matches by trigger-pulse
   fingerprint — rerun it when adding new sessions)

## Which image feeds what (explicit, recorded)

- **Traces and events**: always `runN_clean.tif` (registered raw). Never denoised.
- **Mask geometry** (`auto_segment.py`): controlled by `--mask-source`
  - `auto` (default): `runN_clean_denoised.tif` if it exists, else raw
  - `raw` / `denoised`: force one (denoised errors out if the file is missing)
  The choice actually used is written to `runN_clean_autoseg.json` under
  `params.mask_source` / `params.computed_from`, so every mask records its input.
- **Denoising** (`code/STEP2b_denoise/`, DeepCAD-RT on the SCC GPU): processes each
  Z-plane as an independent 2D movie (x, y, t); it does not use structure across
  adjacent planes. Effect on masks measured on 10 runs: masks 5–30% larger than from
  raw (more faint branch picked up). Whether that is signal or halo is decided in review.

## Voxel size: one source of truth

Every script takes `--voxel Z Y X` in um, but **omit it**: `code/common/voxel.py`
reads the run's own acquisition metadata (`behavior_imaging_master.csv`
`pixel_x_um` / `voxel_z_um`, parsed from the `.mesc` protocol) and prints what it
used. There is no hard-coded default any more; if the run cannot be found the
script stops and tells you. The mice differ (0.70–0.90 um lateral, 0.80–1.00 um Z),
and the old per-script defaults ranged from 0.8 to 3.9 um in Z, which silently
changed anatomical distances and skeleton splits.
