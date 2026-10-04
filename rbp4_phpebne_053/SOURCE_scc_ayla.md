# Source of this session's data

Pulled from **Ayla's SCC tree** (read-only; nothing was written there):

    scc1.bu.edu:/projectnb/devorlab/aosgood/femtonics/

Note the path in the original request, `.../femtonics/download`, does not exist —
the `.mesc` files and Ayla's per-unit TIFF exports live in `.../femtonics/data/`.

Mouse `rbp4_phpebNe_53` on SCC is the same animal as `rbp4_phpebne_053` here
(we already have its 08-11-2026 session).

## What was copied

| local | from (SCC) |
|---|---|
| `raw/MSession_0_MUnit_*.tiff` | `femtonics/data/<mesc stem>/MSession_0_MUnit_*.tiff` (Ayla's export) |
| `behavior_raw/trigger/RunNNN_t1.mat` | `femtonics/behavior/26-MM-DD/rbp4_phpebne_53/trigger/` |
| `behavior_raw/camera/<run>/*.tiff` | `femtonics/behavior/2026-MM-DD/rbp4_phpebne_53/camera/<run>/` |

The `.mesc` itself was **left on SCC** (4.5 GB + 1.6 GB). Only the requested
units were pulled. To get more units later, copy from the same `data/<stem>/`
folder — Ayla has already exported every unit to its own TIFF.

`raw/*.summary.csv` was **regenerated with our own** `summarize_mesc.py`
(run on SCC against the read-only `.mesc`, output written to `~/femtonics_pull`
on the cluster, then downloaded). Ayla's own `.summary.csv` is an older column
format that has **no pixel-size fields**, which STEP 3-5 need for `--voxel`.

## Run pairing

`behavior_imaging_match.csv` in this folder pairs every imaging unit to its
behavior run. The key is the same exact fingerprint used elsewhere in this
project: **`AndorXylaTrigger` rising edges == `n_t * slices`** (one pulse per
scanned plane; `slices == 1` for a ribbon scan, so it is just `n_t`).

Edge counts were read straight out of the MATLAB v7.3 trigger files with
`code/STEP1_extract/trigger_counts_from_mat.py`, then matched by
`code/STEP1_extract/match_raw_trigger_to_mesc.py`.

The behavior here is **raw** — trigger `.mat` plus camera frame folders. There
are no `*_behavior.csv` / `*_info.txt` yet; those come from running the behavior
pipeline. Until then these runs sit at P4 in `behavior_imaging_master.csv` with
`match_confidence = raw_behavior_unprocessed`, and the real pairing lives in
`behavior_imaging_match.csv`.

## Note on ribbon scans

`ribbon_longitudinal` units have **no Z axis**: the stack is `(T, Y, X)`, a strip
along the dendrite over time. The STEP 2-6 pipeline in this repo expects 4D
`(T, Z, Y, X)` volumes, so it does not apply to these unchanged.

## Copied overnight 2026-10-04

Source: `scc1.bu.edu:/projectnb/devorlab/aosgood/femtonics/` (READ ONLY)

| local | SCC source | size |
|---|---|---|
| `09-14-2026/raw/rbp4_phpebNe_53_20260914_145023.mesc` | `data/rbp4_phpebNe_53_20260914_145023.mesc` | 1.54 GB |
| `09-18-2026/raw/rbp4_phpebNe_53_20260918_112243.mesc` | `data/rbp4_phpebNe_53_20260918_112243.mesc` | 1.34 GB |
| `09-10-2026/raw/rbp4_phpebNe_53_20260910_142830.mesc` | `data/rbp4_phpebNe_53_20260910_142830.mesc` | 6.18 GB |

