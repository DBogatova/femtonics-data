# STEP 9 — Automatic pipeline infrastructure

Build and maintain a mirror tree (`auto_pipeline/`) that lets the full
figure/stats toolchain run on automatic results without touching Daria's curated
data.

## Layout

```
auto_pipeline/                          # "AUTO_ROOT"
  behavior_imaging_master.csv           # COPY of root CSV  (never hard-linked)
  ranked_runs.csv                       # COPY
  mice.csv                              # COPY
  run_identity.csv                      # COPY
  rbp4_<mouse>/<date>/.../runNN/
    runNN_clean.tif                     # HARD LINK (same inode, zero extra space)
    runNN_4d.tif                        # HARD LINK (if present)
    runNN_clean_ref3d.tif/.json         # HARD LINK
    runNN_clean_autoseg_labelmap.tif    # HARD LINK
    runNN_clean_autoseg.json            # HARD LINK
    runNN_clean_metrics.json            # generated here (auto results)
    runNN_clean_coherence_full.png      # generated here
    ...
  <mouse>/<date>/behavior/              # DIRECTORY SYMLINK -> real session
  <mouse>/<date>/trigger/               # DIRECTORY SYMLINK -> real session
  <mouse>/<date>/raw/                   # DIRECTORY SYMLINK -> real session
  stats/                                # cohort outputs (auto)
  logs/                                 # JSON-lines logs per stage
```

Hard links share the inode with the original file; `Path.resolve()` stays inside
the mirror, so no script accidentally writes back into the real tree.  Directory
symlinks point to the real session sub-folders for read-only access to behavior
data and trigger files.

Root CSVs are copies (not hard links) because some tools write to them; a
hard-linked CSV would modify the original.

## How to run

```bash
source .venv/bin/activate

# Build / refresh the mirror for all 20 local runs
python code/STEP9_auto/make_auto_tree.py

# Build for one run only
python code/STEP9_auto/make_auto_tree.py --run <run_dir>

# Custom destination (e.g. for testing)
python code/STEP9_auto/make_auto_tree.py --dest /tmp/test_mirror

# Include curated files (testing only: _reviewed, _segments_final, etc.)
python code/STEP9_auto/make_auto_tree.py --include-curated
```

Idempotent: re-running skips files whose hard links already share the inode.

## Running tools on the mirror

Set `FEMTO_ROOT` to point all tools at the mirror instead of the real tree:

```bash
# Direct env var
FEMTO_ROOT=auto_pipeline python code/STEP8_stats/run_metrics.py --all

# Or use the femto CLI shortcut
femto auto status       # run table on the mirror
femto auto stats        # per-run metrics + cohort on the mirror
femto auto panel        # control panel on the mirror
```

`femto auto <cmd>` sets `FEMTO_ROOT=auto_pipeline/` and re-dispatches.

## Overriding automatic results

Open the mirror run in the normal tools — they work unchanged:

```bash
femto auto panel        # -> select a run -> "Edit mask" / "Edit regions"
```

Edits save into the mirror's run directory, not the real tree.

## What is hard-linked vs copied

| Item | Method | Why |
|------|--------|-----|
| `*_clean.tif`, `*_4d.tif` | hard link | large, read-only, zero extra space |
| `*_ref3d.tif/.json` | hard link | read-only reference volume |
| `*_autoseg_labelmap.tif/.json` | hard link | read-only autoseg proposal |
| Root CSVs | copy | tools may write to them |
| `behavior/`, `trigger/`, `raw/`, `snapshots/` | dir symlink | read-only session data |
| `*_reviewed`, `*_segments_final`, metrics, figures | **not mirrored** | auto pipeline generates its own |

## Environment variable: FEMTO_ROOT

When set (absolute path), honored by:
- `femto_status.py` (`project_root()`)
- `femto_gui.py` (ROOT for data, CODE_ROOT for scripts)
- `run_metrics.py`, `cohort_stats.py`, `coupling_phenotype.py`, `behavior_coupling.py` (ROOT)
- `coherence_with_behavior.py` (via `--root` or `fs.project_root()`)
- `voxel.py` (`_project_root()`)

Without `FEMTO_ROOT`, all scripts behave exactly as before (derive the root from
`__file__`).
