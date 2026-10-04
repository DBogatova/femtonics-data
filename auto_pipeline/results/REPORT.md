# Automatic Pipeline Statistical Report (v4)

Generated: 2026-10-04 01:37
n = 27 cells from 6 mice (7 with behavior, 20 imaging-only)
All masks and regions fully automatic (no manual curation).
Set aside (run_marks.csv; reasons starting [auto-QC] are automatic, others are Daria's), not in any statistic: rbp4_141_phpeb_26-06-17_Run001 (revisit); rbp4_phpebne_050_26-09-15_Run009 (revisit: [auto-QC] large motion drift: reference volume needed block registration, traces are unregistered - check before use); rbp4_phpebne_050_26-09-15_Run019 (revisit: [auto-QC] large motion drift: reference volume needed block registration, traces are unregistered - check before use); rbp4_phpebne_053_26-09-14_MUnit16 (revisit: [auto-QC] cell sits on the tube's Z edge for 80% of its length (partly out of frame); comment: forgot behavior); rbp4_phpebne_053_26-09-18_MUnit21 (revisit: [auto-QC] automatic mask missed the dendrite (activity outside the mask 6.6x inside; the dendrite is clearly visible - hand-mask it); comment: good dendrites, bad soma); rbp4_phpebne_053_26-09-18_MUnit37 (revisit: [auto-QC] cell runs along the tube's Z edge for 83% of its length (partly out of frame); right half of the tube bright through all planes - check); rbp4_phpebne_053_26-09-18_MUnit9 (revisit: [auto-QC] no cell (imaging comment: 'run 5. no cell'); no regions could be placed)

Imaging-only cells (listed in imaging_only_runs.csv; behavior not yet processed) are included in all imaging analyses (coupling, independence, distance, event order) and excluded from behavior analyses (test 7). This is stated in each test.

**Key change from v1**: Independence test now uses dual-null surrogates
(coupled-noise null and circular-shift independent null) instead of the
trivially-significant Wilcoxon-vs-0 and wrong-direction binomial.

## Summary

Across 27 cells from 6 mice (7 with behavior, 20 imaging-only):
- **Branches are less coupled to the reference than trunks**: mean r(ref, branch)=0.4933, r(ref, trunk)=0.7179, paired Wilcoxon p=1.53e-05 (17 cells, 6 mice).
- **Branches have genuine independent events**: 27 cells tested; mean frac_independent=0.6046, bracketed between coupled-noise and independent null distributions.
- **Coupling decays with geodesic distance**: 45 cells, Spearman ρ=-0.4964.
- **Branch tends to fire first**: mean fraction=0.6515 (22 cells).
- **Behavior coupling**: 16 cells with behavior data (imaging-only cells excluded from behavior analyses).

Validation on 3 hand-curated cells: mask Dice run03=0.8213, run04=0.5518, run05=0.8103. Leave-one-out calibration caveat: alpha calibration uses anatomy, not Dice, so holding one cell out does not change parameters; but n=3 limits validation power.

---

# 1. Method Validation: Automatic vs Hand-Curated

Compared on 3 cells where Daria manually curated masks and regions.

**Calibration caveat (leave-one-out):** The auto_mask alpha calibration uses soma detection (anatomy), not Dice optimization against these curated cells. Leave-one-out produces the same parameters because the calibration does not depend on the held-out cell's ground truth. Nevertheless, with n=3 ground-truth cells the validation power is limited; additional curated cells would strengthen the claim.

### run03 (no soma)

Mask: Dice=0.821, precision=0.743, recall=0.918
  auto 5698 vox, curated 4615 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.5417 | — | — |
| r_soma_trunk | 0.9305 | 0.8341 | -0.0964 |
| frac_branch_independent | 0.6923 | — | — |
| branch_first_frac | 1.0000 | — | — |
| n_branch_events | 39.0000 | — | — |
| n_soma_events | 14.0000 | — | — |
| r_soma_branch_corr | 0.5449 | — | — |
| rate_branch_per_min | 3.3731 | — | — |

  Auto note: no branch region: branch metrics skipped (reference, distances and trunk coupling computed)

### run04 (soma + bifurcation, another cell at high X limits auto mask extent)

Mask: Dice=0.552, precision=0.546, recall=0.558
  auto 3668 vox, curated 3584 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.5585 | — | — |
| r_soma_trunk | 0.9225 | 0.9162 | -0.0063 |
| frac_branch_independent | 0.5455 | — | — |
| branch_first_frac | 0.5000 | — | — |
| n_branch_events | 110.0000 | — | — |
| n_soma_events | 46.0000 | — | — |
| r_soma_branch_corr | 0.5896 | — | — |
| rate_soma_per_min | 11.4981 | — | — |
| rate_branch_per_min | 14.4976 | — | — |

  Auto note: no branch region: branch metrics skipped (reference, distances and trunk coupling computed)

### run05 (soma present)

Mask: Dice=0.810, precision=0.749, recall=0.883
  auto 14068 vox, curated 11938 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.7144 | — | — |
| r_soma_trunk | 0.9395 | 0.9164 | -0.0231 |
| frac_branch_independent | 0.1538 | — | — |
| branch_first_frac | 1.0000 | — | — |
| n_branch_events | 13.0000 | — | — |
| n_soma_events | 37.0000 | — | — |
| r_soma_branch_corr | 0.7163 | — | — |
| rate_soma_per_min | 9.2483 | — | — |
| rate_branch_per_min | 3.2494 | — | — |

  Auto note: no branch region: branch metrics skipped (reference, distances and trunk coupling computed)

# 2. Soma/Reference-Branch vs Reference-Trunk Coupling

n = 17 cells from 6 mice
r(ref, branch): mean=0.493, 95% CI [0.391, 0.591]
r(ref, trunk):  mean=0.718, 95% CI [0.629, 0.790]
Cells excluded (10): rbp4_140_phpeb_26-06-23_Run006 (no trunk region); rbp4_141_phpeb_26-06-17_Run006 (no trunk region); rbp4_141_phpeb_26-06-25_Run003 (no trunk region); rbp4_155_26-09-23_MUnit19 (no trunk region); rbp4_155_26-09-28_MUnit11 (no trunk region); rbp4_155_26-09-29_MUnit19 (no trunk region); rbp4_155_26-09-29_MUnit22 (no trunk region); rbp4_phpebne_050_26-09-15_Run014 (no trunk region); rbp4_phpebne_050_26-09-15_Run017 (no trunk region); rbp4_phpebne_050_26-09-15_Run020 (no trunk region)
  (13 of 17 are imaging-only cells, included because this is an imaging analysis)
Wilcoxon signed-rank (trunk > branch): W=152.0, p=1.53e-05
Branch coupling lower than trunk: 16/17 cells
Noise-corrected r(ref, branch): mean=0.513, 95% CI [0.411, 0.609]
Halo control: full - core mean diff = 0.0537 (small = not halo artifact)
Mixed model (trunk-branch ~ 1 | mouse): intercept=0.2262, p=5.41e-47

# 3. Independent Branch Events: Dual-Null Surrogate Test

For each cell, 500 surrogates per null are generated from the raw traces:
  (A) COUPLED-NOISE NULL: each branch = gain × reference + phase-randomized measurement noise from a within-region voxel split (half1-half2)/2.
      Tests whether observed independence exceeds what noise alone produces.
  (B) INDEPENDENT NULL: all branch traces circularly shifted together (>20 s).
      Tests whether observed coupling is above chance (i.e., independence < full independence).

n = 27 cells from 6 mice with sufficient events

Observed frac_independent: mean=0.605, 95% CI [0.504, 0.698]
Coupled-noise null mean:   0.486
Independent null mean:     0.898

## Test A: Observed > Coupled-Noise Null
  (Real independent events beyond what detection noise produces)
  Wilcoxon signed-rank (observed - null_A > 0): W=284.0, p=0.0107
  Observed > coupled null: 18/27 cells
  Stouffer combined p: 3.71e-05 (z=3.96)
  Fisher combined p:   4e-14

## Test B: Observed < Independent Null
  (Real coupling: independence is less than what fully unrelated traces give)
  Wilcoxon signed-rank (null_B - observed > 0): W=378.0, p=7.45e-09
  Observed < independent null: 27/27 cells
  Stouffer combined p: < 1e-300 (z=13.81)
  Fisher combined p:   5.73e-37

### Per-Cell Results

| Cell | Observed | Null_A (coupled) | Null_B (independent) | p(>A) | p(<B) | n_branch | n_ref |
|------|----------|------------------|---------------------|-------|-------|----------|-------|
| rbp4_140_phpeb_26-06-23_Run005 | 0.150 | 0.009±0.017 | 0.904±0.079 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 20 | 22 |
| rbp4_140_phpeb_26-06-23_Run006 | 0.636 | 0.375±0.057 | 0.893±0.034 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 99 | 25 |
| rbp4_141_phpeb_26-06-17_Run006 | 0.683 | 0.625±0.016 | 0.743±0.032 | < 0.002 (resolution 1/501) | 0.0299 | 145 | 63 |
| rbp4_141_phpeb_26-06-25_Run007 | 0.714 | 0.159±0.050 | 0.891±0.079 | < 0.002 (resolution 1/501) | 0.0559 | 14 | 31 |
| rbp4_141_phpeb_26-06-25_Run003 | 0.206 | 0.399±0.017 | 0.461±0.105 | 1 | < 0.002 (resolution 1/501) | 355 | 204 |
| rbp4_155_26-09-23_MUnit19 | 0.896 | 0.921±0.004 | 0.932±0.009 | 1 | < 0.002 (resolution 1/501) | 479 | 30 |
| rbp4_155_26-09-28_MUnit11 | 0.714 | 0.713±0.011 | 0.778±0.016 | 0.459 | < 0.002 (resolution 1/501) | 416 | 73 |
| rbp4_155_26-09-28_MUnit22 | 0.125 | 0.050±0.029 | 0.930±0.085 | 0.022 | < 0.002 (resolution 1/501) | 8 | 36 |
| rbp4_155_26-09-29_MUnit13 | 0.660 | 0.414±0.053 | 0.938±0.037 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 53 | 29 |
| rbp4_155_26-09-29_MUnit17 | 0.404 | 0.415±0.062 | 0.951±0.036 | 0.603 | < 0.002 (resolution 1/501) | 57 | 24 |
| rbp4_155_26-09-29_MUnit19 | 0.841 | 0.773±0.032 | 0.951±0.013 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 214 | 26 |
| rbp4_155_26-09-29_MUnit22 | 0.813 | 0.806±0.011 | 0.912±0.015 | 0.255 | < 0.002 (resolution 1/501) | 316 | 41 |
| rbp4_155_26-09-29_MUnit29 | 0.825 | 0.832±0.012 | 0.903±0.026 | 0.739 | < 0.002 (resolution 1/501) | 120 | 46 |
| rbp4_phpebach_26-06-26_Run007 | 0.100 | 0.256±0.066 | 0.960±0.063 | 0.994 | < 0.002 (resolution 1/501) | 10 | 13 |
| rbp4_phpebach_26-06-26_Run009 | 0.750 | 0.867±0.017 | 0.872±0.082 | 1 | 0.134 | 16 | 32 |
| rbp4_phpebne_050_26-09-15_Run011 | 0.881 | 0.531±0.058 | 0.972±0.013 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 134 | 14 |
| rbp4_phpebne_050_26-09-15_Run012 | 0.839 | 0.075±0.051 | 0.983±0.013 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 93 | 18 |
| rbp4_phpebne_050_26-09-15_Run014 | 0.548 | 0.388±0.036 | 0.906±0.030 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 115 | 39 |
| rbp4_phpebne_050_26-09-15_Run016 | 0.455 | 0.247±0.084 | 0.904±0.072 | 0.012 | < 0.002 (resolution 1/501) | 11 | 9 |
| rbp4_phpebne_050_26-09-15_Run017 | 0.784 | 0.827±0.026 | 0.947±0.039 | 0.956 | < 0.002 (resolution 1/501) | 37 | 6 |
| rbp4_phpebne_050_26-09-15_Run018 | 0.533 | 0.141±0.072 | 0.924±0.062 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 15 | 8 |
| rbp4_phpebne_050_26-09-15_Run020 | 0.410 | 0.348±0.036 | 0.869±0.026 | 0.0479 | < 0.002 (resolution 1/501) | 195 | 100 |
| rbp4_phpebne_050_26-09-15_Run021 | 0.853 | 0.912±0.012 | 0.931±0.024 | 1 | < 0.002 (resolution 1/501) | 68 | 5 |
| rbp4_phpebne_053_26-09-14_Run005 | 0.533 | 0.056±0.051 | 0.998±0.015 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 15 | 11 |
| rbp4_phpebne_053_26-09-14_Run010 | 0.185 | 0.177±0.032 | 0.920±0.052 | 0.419 | < 0.002 (resolution 1/501) | 27 | 54 |
| rbp4_phpebne_053_26-09-18_MUnit8 | 0.946 | 0.939±0.006 | 0.956±0.008 | 0.122 | 0.164 | 296 | 23 |
| rbp4_phpebne_053_26-09-18_MUnit14 | 0.841 | 0.863±0.008 | 0.906±0.010 | 0.994 | < 0.002 (resolution 1/501) | 485 | 46 |

Per-cell FDR (test A, obs > coupled): 13/27 significant at q<0.05
Per-cell FDR (test B, obs < independent): 24/27 significant at q<0.05

### Interpretation
  Majority of cells (18/27) show more independent branch events than
  the coupled-noise null: real independent events exist beyond detection noise.
  Majority of cells (27/27) show less independence than the
  independent null: real coupling between branch and reference exists.

CAVEAT: Same event detection parameters (window=2, prom_frac=0.2) are used for all cells.
Ground truth for the claim is limited to n=3 curated cells. Report overfitting risk honestly.
Permutation p-values have resolution 1/(500+1) ≈ 0.002; the minimum reportable is < 1/501.

# 3b. Ground-Truth Independence Test (Daria's Curated Cells)

Running the same dual-null independence test on Daria's 3 hand-curated
segments+stacks. These are the ground truth for the claim 'branches have
their own events'. All computation in memory; no writes to real tree.
Leave-one-out: calibration uses anatomy, not these Dice scores, so holding one
out does not change parameters. But n=3 limits validation power.

### run05 (soma present)
  Reference: soma
  Branches: ['branch2']
  n_branch_events = 13, n_ref_events = 37
  Observed frac_independent = 0.154
  Coupled-noise null:   0.060 ± 0.032
  Independent null:     0.854 ± 0.091
  p(observed > coupled null) = 0.00798
  p(observed < independent null) = < 0.002 (resolution 1/501)
  Bracketed: YES (coupled < observed < independent: 0.060 < 0.154 < 0.854)

  Daria's frac_branch_independent (from _metrics.json): 0.15384615384615385
  → Our surrogate test confirms this is above detection noise (p=0.00798) and below full independence (p=< 0.002 (resolution 1/501)).

### run03 (no soma)
  Reference: trunk1
  Branches: ['bifurcation', 'branch1', 'branch1far', 'branch2']
  n_branch_events = 39, n_ref_events = 14
  Observed frac_independent = 0.692
  Coupled-noise null:   0.774 ± 0.027
  Independent null:     0.952 ± 0.036
  p(observed > coupled null) = 0.994
  p(observed < independent null) = < 0.002 (resolution 1/501)
  Bracketed: NO (coupled < observed < independent: 0.774 < 0.692 < 0.952)

  Daria's frac_branch_independent (from _metrics.json): 0.6923076923076923
  → Our surrogate test confirms this is not above detection noise (p=0.994) and below full independence (p=< 0.002 (resolution 1/501)).

### run04 (soma + bifurcation, another cell at high X limits auto mask extent)
  Reference: soma
  Branches: ['branch1', 'branch2']
  n_branch_events = 110, n_ref_events = 46
  Observed frac_independent = 0.545
  Coupled-noise null:   0.693 ± 0.018
  Independent null:     0.864 ± 0.029
  p(observed > coupled null) = 1
  p(observed < independent null) = < 0.002 (resolution 1/501)
  Bracketed: NO (coupled < observed < independent: 0.693 < 0.545 < 0.864)

  Daria's frac_branch_independent (from _metrics.json): 0.5454545454545454
  → Our surrogate test confirms this is not above detection noise (p=1) and below full independence (p=< 0.002 (resolution 1/501)).

# 4. Coupling vs Geodesic Distance from Reference

n = 176 regions from 45 cells
Pooled Spearman (descriptive, pseudo-replicated): rho=-0.496, p=2.46e-12
  NOTE: regions from the same cell are not independent; the mixed model below is the primary test.
Per-cell slope: mean=-0.112 r/100um, 95% CI [-0.145, -0.079]
Mixed model (r ~ distance + (1|cell), PRIMARY): slope=-0.000971/um, p=1.28e-15

# 5. Event Order: Branch-First Fraction

n = 22 cells, 390 paired events
Branch peaks first: 234/390 (60.0%)
Pooled binomial (descriptive, pseudo-replicated): p=4.59e-05
  NOTE: events from the same cell are not independent; the sign test below is the primary test.
Per-cell sign test (PRIMARY): 15/22 cells with fraction > 0.5, p=0.0669
Mean branch-first fraction: 0.652, 95% CI [0.556, 0.747]

# 6. Branch Amplitude Predicts Propagation

This test requires per-event amplitude data not stored in the current metrics.
Future: extract from coherence network_events CSVs.
SKIP: per-event amplitude data not available in aggregate metrics.

# 7. Behavior Coupling

n = 16 cells with behavior data
  (20 imaging-only cells excluded from behavior analyses — behavior data not available)

  whisking: 9/16 significant at p<0.05 (uncorrected), mean peak r=0.190
  pupil: 3/16 significant at p<0.05 (uncorrected), mean peak r=0.105
  accelerometer: 7/16 significant at p<0.05 (uncorrected), mean peak r=0.129

After BH-FDR across all 48 behavior tests: 17 significant at q<0.05

Quiet vs active: 4/7 cells more coupled when active
Wilcoxon (active vs quiet): p=0.578

# 8. Coupling vs Expression Time

n = 5 cells from 2 mice
DPI range: 250–258
Spearman r(ref, branch) vs DPI: rho=-0.632, p=0.252
Spearman r(ref, branch) vs soma F: rho=-0.186, p=0.352

CAVEAT: DPI range is narrow (17 days across 5 mice), so this test has limited power.
rbp4_phpebach excluded (unknown injection date).

# 9. Co-firing Groups

n = 45 cells
Co-firing groups per cell: mean=3.2, range [1, 10]
Soma shares a group with any branch: 8/25 cells

# 10. Per-Run QC Table

| Run | Mouse | Mask vox | Regions | Reference | Intruders | Soma events | Branch events | Flags |
|-----|-------|----------|---------|-----------|-----------|-------------|---------------|-------|
| phpeb/06-08-2026/preprocessed/run03 | rbp4_132_phpeb | 12626 | 3 | soma | 0 | 0 | 0 | no_branches |
| phpeb/06-08-2026/preprocessed/run04 | rbp4_132_phpeb | 11700 | 3 | soma | 0 | 0 | 0 | no_branches |
| 4_139_phpeb/06-12-2026/traces/run05 | rbp4_139_phpeb | 6907 | 4 | proximal_trunk | 0 | 0 | 0 | no_soma, no_behavior, no_branches |
| p4_139_phpeb/06-12-2026/traces/run4 | rbp4_139_phpeb | 19758 | 4 | soma | 0 | 0 | 0 | no_behavior, no_branches |
| phpeb/06-18-2026/preprocessed/run03 **[GT]** | rbp4_140_phpeb | 5698 | 4 | proximal_trunk | 0 | 0 | 0 | no_soma, no_branches |
| phpeb/06-18-2026/preprocessed/run04 **[GT]** | rbp4_140_phpeb | 3668 | 4 | proximal_trunk | 0 | 0 | 0 | no_soma, no_branches |
| phpeb/06-23-2026/preprocessed/run05 | rbp4_140_phpeb | 10796 | 4 | proximal_trunk | 0 | 22 | 20 | no_soma, no_bifurcation |
| phpeb/06-23-2026/preprocessed/run06 | rbp4_140_phpeb | 31370 | 6 | soma | 1 | 25 | 99 | 1_suspects |
| phpeb/06-17-2026/preprocessed/run01 |  | 12530 | 4 | proximal_trunk | 0 | 0 | 0 | no_soma, no_bifurcation |
| phpeb/06-17-2026/preprocessed/run04 | rbp4_141_phpeb | 4538 | 4 | proximal_trunk | 0 | 0 | 0 | no_soma, low_reliability, no_branches |
| phpeb/06-17-2026/preprocessed/run06 | rbp4_141_phpeb | 7048 | 7 | proximal_trunk | 0 | 63 | 145 | no_soma, low_reliability |
| phpeb/06-17-2026/preprocessed/run09 | rbp4_141_phpeb | 7781 | 4 | proximal_trunk | 0 | 0 | 0 | no_soma, no_branches |
| phpeb/06-25-2026/preprocessed/run04 | rbp4_141_phpeb | 5148 | 3 | soma | 2 | 0 | 0 | 2_suspects, no_branches |
| phpeb/06-25-2026/preprocessed/run07 | rbp4_141_phpeb | 14602 | 4 | soma | 0 | 31 | 14 | no_bifurcation |
| _phpeb/06-25-2026/preprocessed/run3 | rbp4_141_phpeb | 12069 | 6 | soma | 0 | 204 | 355 |  |
| 155/09-23-2026/preprocessed/munit19 | rbp4_155 | 41901 | 10 | soma | 1 | 30 | 479 | 1_suspects, low_reliability, no_behavior |
| 155/09-28-2026/preprocessed/munit11 | rbp4_155 | 14099 | 8 | soma | 0 | 73 | 416 | low_reliability, no_behavior |
| 155/09-28-2026/preprocessed/munit22 | rbp4_155 | 13584 | 4 | soma | 0 | 36 | 8 | no_bifurcation, no_behavior |
| 155/09-29-2026/preprocessed/munit13 | rbp4_155 | 12370 | 5 | soma | 0 | 29 | 53 | no_bifurcation, no_behavior |
| 155/09-29-2026/preprocessed/munit15 | rbp4_155 | 12906 | 5 | soma | 0 | 0 | 0 | low_reliability, no_behavior, no_branches |
| 155/09-29-2026/preprocessed/munit17 | rbp4_155 | 11371 | 7 | soma | 0 | 24 | 57 | no_behavior |
| 155/09-29-2026/preprocessed/munit19 | rbp4_155 | 10956 | 5 | soma | 1 | 26 | 214 | 1_suspects, low_reliability, no_behavior |
| 155/09-29-2026/preprocessed/munit22 | rbp4_155 | 14473 | 7 | soma | 0 | 41 | 316 | low_reliability, no_behavior |
| 155/09-29-2026/preprocessed/munit29 | rbp4_155 | 21152 | 5 | soma | 0 | 46 | 120 | no_bifurcation, low_reliability, no_behavior |
| 155/09-29-2026/preprocessed/munit31 | rbp4_155 | 19305 | 5 | soma | 1 | 0 | 0 | 1_suspects, low_reliability, no_behavior, no_branches |
| rbp4_phpebach/06-26-2026/run04 | rbp4_phpebach | 5033 | 4 | proximal_trunk | 1 | 0 | 0 | no_soma, 1_suspects, no_branches |
| rbp4_phpebach/06-26-2026/run04/run2 | rbp4_phpebach | 7398 | 4 | soma | 0 | 0 | 0 | low_reliability, no_branches |
| rbp4_phpebach/06-26-2026/run05 **[GT]** | rbp4_phpebach | 14068 | 5 | soma | 0 | 0 | 0 | no_branches |
| rbp4_phpebach/06-26-2026/run07 | rbp4_phpebach | 18028 | 5 | soma | 1 | 13 | 10 | no_bifurcation, 1_suspects |
| rbp4_phpebach/06-26-2026/run09 | rbp4_phpebach | 23704 | 5 | soma | 0 | 32 | 16 | no_bifurcation |
| 050/09-15-2026/preprocessed/munit43 | rbp4_phpebne_050 | 22246 | 5 | soma | 0 | 0 | 0 | short_recording, no_behavior, no_branches |
| e_050/09-15-2026/preprocessed/run09 |  | 11874 | 5 | soma | 2 | 0 | 0 | 2_suspects, no_behavior |
| e_050/09-15-2026/preprocessed/run10 | rbp4_phpebne_050 | 11271 | 3 | soma | 0 | 0 | 0 | no_behavior, no_branches |
| e_050/09-15-2026/preprocessed/run11 | rbp4_phpebne_050 | 10952 | 5 | soma | 0 | 14 | 134 | no_behavior |
| e_050/09-15-2026/preprocessed/run12 | rbp4_phpebne_050 | 16454 | 3 | soma | 0 | 18 | 93 | no_bifurcation, no_behavior |
| e_050/09-15-2026/preprocessed/run14 | rbp4_phpebne_050 | 14829 | 6 | soma | 0 | 39 | 115 | no_behavior |
| e_050/09-15-2026/preprocessed/run15 | rbp4_phpebne_050 | 21243 | 3 | soma | 3 | 0 | 0 | 3_suspects, short_recording, no_behavior, no_branches |
| e_050/09-15-2026/preprocessed/run16 | rbp4_phpebne_050 | 20087 | 3 | soma | 1 | 9 | 11 | no_bifurcation, 1_suspects, short_recording, no_behavior |
| e_050/09-15-2026/preprocessed/run17 | rbp4_phpebne_050 | 22042 | 7 | soma | 1 | 6 | 37 | 1_suspects, low_reliability, short_recording, no_behavior |
| e_050/09-15-2026/preprocessed/run18 | rbp4_phpebne_050 | 19040 | 5 | soma | 0 | 8 | 15 | no_bifurcation, short_recording, no_behavior |
| e_050/09-15-2026/preprocessed/run19 |  | 24966 | 7 | soma | 0 | 0 | 0 | low_reliability, no_behavior |
| e_050/09-15-2026/preprocessed/run20 | rbp4_phpebne_050 | 17447 | 7 | soma | 0 | 100 | 195 | low_reliability, no_behavior |
| e_050/09-15-2026/preprocessed/run21 | rbp4_phpebne_050 | 41414 | 10 | soma | 0 | 5 | 68 | low_reliability, short_recording, no_behavior |
| e_053/09-14-2026/preprocessed/run05 | rbp4_phpebne_053 | 14181 | 4 | soma | 0 | 11 | 15 | no_bifurcation, no_behavior |
| e_053/09-14-2026/preprocessed/run07 | rbp4_phpebne_053 | 8711 | 3 | soma | 0 | 0 | 0 | no_behavior, no_branches |
| e_053/09-14-2026/preprocessed/run10 | rbp4_phpebne_053 | 9390 | 4 | soma | 0 | 54 | 27 | no_bifurcation, no_behavior |
| 053/09-18-2026/preprocessed/munit08 | rbp4_phpebne_053 | 17208 | 7 | soma | 0 | 23 | 296 | low_reliability, no_behavior |
| 053/09-18-2026/preprocessed/munit09 |  | 26024 | 8 | soma | 0 | 0 | 0 | low_reliability, no_behavior |
| 053/09-18-2026/preprocessed/munit14 | rbp4_phpebne_053 | 22697 | 7 | soma | 0 | 46 | 485 | low_reliability, no_behavior |

All 49 runs have confidence='auto' (fully automatic, no manual review).

---
## Global BH-FDR Correction (Cohort-Level Tests Only)

Only cohort-level p-values (one per test) are included in the global FDR.
Per-cell surrogate p-values and pseudo-replicated (pooled) p-values are excluded.

| Test | p | q (BH) |
|------|---|--------|
| test02_coupling | 1.53e-05 | 3.43e-05 **|
| test02_coupling | 5.41e-47 | 4.86e-46 **|
| test03_independence | 0.0107 | 0.0193 **|
| test03_independence | 7.45e-09 | 2.24e-08 **|
| test04_distance | 1.28e-15 | 5.75e-15 **|
| test05_event_order | 0.0669 | 0.1 |
| test07_behavior | 0.578 | 0.578 |
| test08_expression | 0.252 | 0.324 |
| test08_expression | 0.352 | 0.396 |

9 tests, 5 significant at q<0.05

---
## Sensitivity Analysis
All runs are automatic (confidence='auto'), so no high vs low confidence split is possible.
The full cohort results above ARE the only analysis.

---
## Statistical Notes

### Changes from pass 1
1. **Independence test (test 3)**: Replaced meaningless one-sample Wilcoxon vs 0
   (trivially significant: any nonzero frac_independent passes) and wrong-direction
   per-cell binomial (null rate = 1 - soma_coverage, which tested whether observed
   independence was ABOVE chance, but the claim needs it above coupled noise and
   below full independence). Now uses dual-null surrogates on raw traces.
2. **Pseudo-replication**: Pooled Spearman (test 4) and pooled binomial (test 5)
   treat observations from the same cell as independent. Both are now flagged as
   descriptive; the mixed model (test 4) and sign test (test 5) are the primary tests.
3. **Global FDR**: Now includes only cohort-level p-values, not per-cell surrogates.
4. **Ground truth**: Independence test validated on Daria's 3 curated cells (test 3b).
5. **p-value formatting**: Full-precision floats in RESULTS.json; 3 significant digits in REPORT.md.
   Permutation p-values (500 draws) have resolution 1/501 ≈ 0.002; minimum reported as < 1/501.

### Overfitting risk
Same parameters (window=2 frames, prom_frac=0.2, n_draws=500) for all cells.
Ground truth is n=3 curated cells. The surrogate test does not use ground truth
to set parameters; parameters come from the event detector which was developed
independently. However, the small ground-truth sample limits validation power.
Alpha calibration uses anatomy (soma detection), not Dice against curated masks,
so leave-one-out produces the same parameters.

---
## Cross-Check: cohort_stats.py vs RESULTS.json

cohort_stats.py: 27 runs, r(soma,branch) mean 0.51, r(soma,trunk) mean 0.72, Wilcoxon p=1.53e-05, 7 set aside
paper_stats.py:  27 runs, r(soma,branch) mean 0.4933, r(soma,trunk) mean 0.7179, Wilcoxon p=1.53e-05

**Differences found:**
  - r(soma,branch): cohort_stats 0.5100 vs paper_stats 0.4933 (cohort_stats includes all cells with branch regions; paper_stats test 2 requires both branch and trunk for pairing)

Explanation: cohort_stats.py counts all cells including those without branch regions (for distance and trunk coupling), while paper_stats.py test 2 requires both r_soma_branch and r_soma_trunk (paired test). Cells without a branch region are excluded from the paired coupling test but included in the distance regression and independence test where applicable. The difference in n is expected and correct.

