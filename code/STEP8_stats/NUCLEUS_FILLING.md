# Nuclear filling (GCaMP7s) - what the existing data can say

**Marker.** Healthy expression excludes GCaMP from the nucleus: a soma seen from above
is a bright ring with a dark center. Once the sensor enters the nucleus the soma is a
flat bright disc. Binary, visible by eye, independent of laser power.

**What exists.** Of 108 two-photon snapshots across all sessions, 81 are tilted side
views (apical trunks seen as vertical bars - useless for this), 27 are top-down, and
only these are in the L5 soma layer (>= 450 um):

| session | top-down L5 frames | usable? |
|---|---|---|
| rbp4_phpebne_053 08-11 | 13 (0.11-0.32 um/px) | yes - rings AND filled discs both visible (`sheets/`) |
| rbp4_phpebach 06-26 | 1 (0.75 um/px, single noisy frame) | marginal |
| rbp4_132 / 139 / 140 / 141 (all dates) | 0 | **no soma-layer top-down frame exists** |
| rbp4_ai162_055 | 0 at L5 (3 shallow) | no |

So for the four Rbp4 GCaMP7s mice whose dendrites were analyzed, nuclear state cannot be
determined retrospectively. The one high-resolution session (phpebne_053) shows that
both states coexist in the same animal at the same depth, which is itself informative:
"filled vs ring" varies cell by cell, so it must be recorded per recorded cell, not per
mouse or per session.

**Automatic scoring was tried and discarded** (`old_automatic/`): a blob detector on
single noisy frames mistook dendrite cross-sections for somata and mis-sized real ones;
its 98-100 % "filled" numbers were artifacts. Scoring by eye on the contact sheets is
the honest method until a dedicated image exists.

**Protocol going forward (per recorded cell).** Before the snake scan, one top-down frame
of the soma at <= 0.3 um/px, averaged over 10-30 frames (not a single frame), centered
on the cell about to be recorded. Record in the run's notes: ring / filled / unsure.
This becomes the `nucleus_state` covariate in `stats/cohort_metrics.csv`, and the
statistics compare soma-dendrite coupling between ring and filled cells directly.
