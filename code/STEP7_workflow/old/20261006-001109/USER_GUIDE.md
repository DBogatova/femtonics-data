# User guide: from a recording to statistics

The pipeline takes a Femtonics snake recording of one L5 pyramidal cell and produces a
mask, regions, a figure with behavior, 3D movies, per-cell metrics and cohort statistics.
Everything is automatic except two short steps you do by eye: checking the mask and
picking regions.

## Starting

Double-click **Femto Panel** on the Desktop, or type `femto` in a terminal.

## Automatic mask, regions by hand (default)

Two checkboxes control what the program does automatically:

- **automatic mask** (default ON) — the program draws the mask. When the run reaches the
  mask step, `auto_mask.py` runs instead of opening the mask napari tool. It never
  replaces a mask that already exists; your curated cells stay as they are.
- **automatic regions** (default OFF) — the program picks the regions. Because region
  placement is subjective, the default is OFF: after the mask is made, the **region tool**
  opens for you to pick regions by hand. Tick this checkbox only if you want the program
  to pick regions too (e.g. for a first pass over many runs).

With the defaults (**automatic mask** ON, **automatic regions** OFF, **chain steps** ON):

1. Click **Run next step** — the program builds the reference volume and the mask
   automatically, then opens the **region tool** for you.
2. Pick regions in the region tool (see below), **Ctrl+S**, close napari.
3. Click **Refresh**, then **Run next step** — the figure, movies and statistics are built.

**Automate all runs** does this for every local, unmarked run: reference + automatic mask,
stopping at the region step. Runs that already have regions continue to the figure. Then
the statistics are rebuilt once.

Other features:

- The **mask/regions by** column shows **program**, **you** or **both**.
- To correct the program's mask, use **Edit mask** (it opens the automatic result with your
  saved session), then **Build figure**.
- **Mark / rate…** sets a whole run aside (excluded / revisit later); **Ignore regions…** leaves
  single regions out of the figure and statistics.
- The same dialog rates a processed run **very good / good / questionable** (or none), shown
  in the **rating** column (`femto rate RUN very_good|good|questionable|clear --reason "..."`;
  stored in `run_quality.csv`). A rating never removes a run: the statistics keep every rated
  run and add a sensitivity table next to the main tests, (a) without 'questionable' runs and
  (b) 'very good' only (`stats/cohort_summary.txt`, `stats/cohort_tests_sensitivity.csv`, and
  the automatic report). The rating also appears in the cell atlas and overview headers.
  (The **priority** column, P1–P4, is the old ranking from your imaging comments.)
- Runs without behavior (imaging only, e.g. the September sessions) show "needs behavior";
  they get the imaging figure and are left out of behavior statistics.
- Removed 4D stacks: `femto restore PATH|all`.
- Every cell on one page with a 3D view: `python code/STEP8_stats/cell_atlas.py` (writes `stats/cell_atlas/index.html`).
- **common amplitude scale** (Options, default ON): **Build figure** draws traces at one cohort-wide dF/F per inch (`stats/plot_scale.json`), so amplitudes compare between cells. Untick for the old per-run scaling. All curated runs on one page: `stats/normalized/sheet_common.pdf` and `sheet_zscore.pdf` (`python code/STEP8_stats/normalized_plots.py`).

## Processing one cell fully by hand (untick "automatic mask")

1. In the panel, select the highlighted run and click **Open tool**. Each time a step finishes, click **Refresh**, then **Open tool** again.
2. **Mask tool** opens. Get the mask right (see below), press **Ctrl+S**, then close napari.
3. **Region tool** opens. Pick regions (see below), press **Ctrl+S**, then close napari.
4. Click **Run next step**. The figure, movies and statistics are built for you. The two progress bars show what is running.

To redo a cell later: **Edit mask** or **Edit regions**, save, then **Build figure** (or **Build movies**).
The mask tool reopens exactly where you left it. Previous files are moved to `old/`, never
deleted.

## Mask tool

| Section | What it does |
|---|---|
| Mask Thickness | **Tightness**: higher = thinner (removes glow). **Max width**: limits how far the mask spreads. **Extra margin**: dilates the mask. |
| Change Cell Mask | **Add a missing branch** (`t`, two clicks: start then end). **Remove a branch** (`x`, click it). **Erase / Add voxels** brushes (`e` / `a`). **Undo** (`u`). **Start over from automatic** (`r`). |
| Mark Other Cells | **Trace another cell** (`i`, two clicks). **Switch a branch: mine / other** (`f`). Their own tightness / width sliders and brushes (Shift-E / Shift-A). |
| View | **Image** (`c`): co-firing moments / average / peak activity, display only. **2D / 3D** (`d`). |

Rules worth knowing:

- **Your cell always wins.** Other cells can never take a voxel your mask claims, so marking or shaping them never changes your cell.
- **Every edit is permanent.** That includes erasing with napari's own brush directly on the "Your cell" or "Other cells" layer. It survives slider changes and reopening.
- In the 2D view, a click finds the brightest point through all planes, so a connection is drawn once rather than slice by slice.
- **Orange** marks spots where the image is dim. The tool never bridges a gap there for you: decide by eye.

## Region tool

- **Whole piece** (`w`): click to take a skeleton piece. Clicking anywhere on the soma takes the whole soma.
- **Stretch between 2 clicks** (`i`): selects the dendrite between two points.
- **Cut a region here** (`k`): splits a region at the clicked cross-section.
- **Name**: `s` soma, `t` trunk, `b` branch, `n` a free name. Names that start with soma / trunk decide the compartment; anything else counts as a branch.
- **Merge** (`m`), **Erase** (`e`), **Undo** (`u`).

**Recipe for comparable cells.** Use the same compartments in every cell: soma (when it is in the scan), a proximal trunk stretch, a distal trunk stretch, and each clear branch. Sampling the clearest stretch of each compartment is fine. Keep regions roughly similar in size across cells, at least about 50 voxels.

## Outputs per run (in the run folder)

| File | Content |
|---|---|
| `runNN_clean_coherence_full.png/.pdf` | figure: correlation matrix, cell picture, traces, behavior. The PDF is vector. |
| `runNN_final_3d_{dual,time,structure}.mp4` | 3D movies |
| `runNN_clean_metrics.json` | per-cell metrics (below) |
| `runNN_clean_coupling.png/.json` | co-firing groups among your regions |
| `runNN_clean_autoseg_labelmap_reviewed.tif` | your mask; `_exclude_labelmap.tif` other cells; `_segments_final.tif` regions |

## Cohort statistics (`stats/`, rebuilt by **Statistics (all cells)** or `femto stats`)

- `cohort_metrics.csv`: one row per cell.
- `cohort_summary.txt`: the tests, written in words.
- `fig_independence`, `fig_coupling_vs_distance`, `fig_within_mouse`, `fig_expression`.
- `phenotype/cell_profiles.csv` and `fig_cell_phenotypes`. Cell types are assigned automatically once there are 6 or more cells.

## What the metrics mean

| Metric | Definition |
|---|---|
| reference | the soma region; if the soma is below the scanned tube, the trunk region closest to the soma (`proximal_trunk`, shown hollow in the figures and reported separately) |
| r_soma_branch | correlation of dF/F, reference vs each branch (mean over branches) |
| r_soma_branch_corr | the same, corrected for each region's noise (split-half reliability), so different ROI sizes compare fairly |
| r_soma_branch_core | the same, using only the inner core of each region. If the coupling were driven by scattered light (halo), this would be much lower. |
| frac_branch_independent | branch events with no reference event within ±2 frames |
| branch_first_frac | among paired events, the fraction where the branch peaked first |
| distance_um | distance from the reference, measured along the dendrite through the mask |
| n_groups | co-firing groups: average-linkage clustering of the correlation, cut at r = 0.75. `r_cut_sweep` shows how the count changes between 0.60 and 0.90. Treat the count as a relative measure between cells, not an absolute one. |
| r_soma_branch_quiet / _active | coupling in low vs high arousal frames (pupil or whisking above their median) |

## Caveats to keep in mind

- **Shared signals.** The whole scan tube glows in step with the cell. The noise correction does not remove shared signal, and neither does masking.
- **Region sampling.** With sampled (not tiled) regions, the results describe the compartments you chose. They do not show *where* along the dendrite coupling breaks down.
- **Other cells.** Small, poorly correlated regions at the tube edge may belong to another cell. Check them in 3D before relying on them; run03's `branch2` is an example to check.
- **Nucleus state** is recorded per cell from a dedicated soma snapshot (see `code/STEP8_stats/NUCLEUS_FILLING.md`). It cannot be recovered retrospectively for the rbp4_132/139/140/141 mice.
