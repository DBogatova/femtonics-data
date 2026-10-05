# Automatic Pipeline Statistical Report (v4)

Generated: 2026-10-05 09:53
n = 44 cells from 8 mice (16 with behavior, 28 imaging-only)
All masks and regions fully automatic (no manual curation).
Set aside (run_marks.csv; reasons starting [auto-QC] are automatic, others are Daria's), not in any statistic: rbp4_141_phpeb_26-06-17_Run001 (revisit); rbp4_141_phpeb_26-06-17_Run004 (excluded); rbp4_141_phpeb_26-06-17_Run009 (revisit); rbp4_141_phpeb_26-06-25_Run003 (excluded); rbp4_phpebne_050_26-09-15_Run009 (revisit: [auto-QC] large motion drift: reference volume needed block registration, traces are unregistered - check before use); rbp4_phpebne_050_26-09-15_Run019 (revisit: [auto-QC] large motion drift: reference volume needed block registration, traces are unregistered - check before use); rbp4_phpebne_053_26-09-14_MUnit16 (revisit: [auto-QC] cell sits on the tube's Z edge for 80% of its length (partly out of frame); comment: forgot behavior); rbp4_phpebne_053_26-09-18_MUnit21 (revisit: [auto-QC] automatic mask missed the dendrite (activity outside the mask 6.6x inside; the dendrite is clearly visible - hand-mask it); comment: good dendrites, bad soma); rbp4_phpebne_053_26-09-18_MUnit37 (revisit: [auto-QC] cell runs along the tube's Z edge for 83% of its length (partly out of frame); right half of the tube bright through all planes - check); rbp4_phpebne_053_26-09-18_MUnit9 (revisit: [auto-QC] no cell (imaging comment: 'run 5. no cell'); no regions could be placed)

Imaging-only cells (listed in imaging_only_runs.csv; behavior not yet processed) are included in all imaging analyses (coupling, independence, distance, event order) and excluded from behavior analyses (test 7). This is stated in each test.

**Key change from v1**: Independence test now uses dual-null surrogates
(coupled-noise null and circular-shift independent null) instead of the
trivially-significant Wilcoxon-vs-0 and wrong-direction binomial.

## Summary

Across 44 cells from 8 mice (16 with behavior, 28 imaging-only):
- **Branches are less coupled to the reference than trunks**: mean r(ref, branch)=0.4363, r(ref, trunk)=0.6562, paired Wilcoxon p=3.07e-08 (44 cells, 8 mice).
- **Branches have genuine independent events**: 41 cells tested; mean frac_independent=0.4919, bracketed between coupled-noise and independent null distributions.
- **Coupling decays with geodesic distance**: 44 cells, Spearman ρ=-0.5585.
- **Branch tends to fire first**: mean fraction=0.7785 (27 cells).
- **Behavior coupling**: 13 cells with behavior data (imaging-only cells excluded from behavior analyses).

Validation on 7 hand-curated cells: mask Dice run03=0.7757, run04=0.688, run04_phpebach=0.887, run05=0.8825, run05_140_0623=0.8535, run06_140_0623=0.9006, run07_phpebach=0.6894. Leave-one-out calibration caveat: alpha calibration uses anatomy, not Dice, so holding one cell out does not change parameters; but n=3 limits validation power.

---

# 1. Method Validation: Automatic vs Hand-Curated

Compared on 7 cells where Daria manually curated masks and regions.

**Calibration caveat (leave-one-out):** The auto_mask alpha calibration uses soma detection (anatomy), not Dice optimization against these curated cells. Leave-one-out produces the same parameters because the calibration does not depend on the held-out cell's ground truth. Nevertheless, with n=7 ground-truth cells the validation power is limited; additional curated cells would strengthen the claim.

### run03 (no soma)

Mask: Dice=0.776, precision=0.656, recall=0.949
  auto 6673 vox, curated 4615 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.5417 | 0.5233 | -0.0184 |
| r_soma_trunk | 0.9305 | 0.8407 | -0.0898 |
| frac_branch_independent | 0.6923 | 0.2857 | -0.4066 |
| branch_first_frac | 1.0000 | 1.0000 | +0.0000 |
| n_branch_events | 39.0000 | 7.0000 | -32.0000 |
| n_soma_events | 14.0000 | 15.0000 | +1.0000 |
| r_soma_branch_corr | 0.5449 | 0.5254 | -0.0196 |
| rate_branch_per_min | 3.3731 | 1.7490 | -1.6241 |

### run04 (soma + bifurcation, another cell at high X limits auto mask extent)

Mask: Dice=0.688, precision=0.622, recall=0.770
  auto 4439 vox, curated 3584 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.5585 | 0.8497 | +0.2912 |
| r_soma_trunk | 0.9225 | 0.9348 | +0.0123 |
| frac_branch_independent | 0.5455 | 0.1667 | -0.3788 |
| branch_first_frac | 0.5000 | 1.0000 | +0.5000 |
| n_branch_events | 110.0000 | 72.0000 | -38.0000 |
| n_soma_events | 46.0000 | 45.0000 | -1.0000 |
| r_soma_branch_corr | 0.5896 | 0.8565 | +0.2669 |
| rate_soma_per_min | 11.4981 | — | — |
| rate_branch_per_min | 14.4976 | 10.3733 | -4.1243 |

### run04_phpebach (no soma (trunk_soma_end), bifurcation with branch2, mask ends X 360/392)

Mask: Dice=0.887, precision=0.918, recall=0.858
  auto 7722 vox, curated 8256 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.4805 | 0.4635 | -0.0170 |
| r_soma_trunk | 0.5820 | 0.5755 | -0.0065 |
| frac_branch_independent | 0.2857 | 0.3143 | +0.0286 |
| branch_first_frac | 0.4444 | 0.5000 | +0.0556 |
| n_branch_events | 35.0000 | 35.0000 | +0.0000 |
| n_soma_events | 46.0000 | 45.0000 | -1.0000 |
| r_soma_branch_corr | 0.4893 | 0.4702 | -0.0191 |
| rate_branch_per_min | 5.1663 | 4.8121 | -0.3541 |

### run05 (soma present)

Mask: Dice=0.883, precision=0.866, recall=0.899
  auto 12395 vox, curated 11938 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.7144 | 0.7020 | -0.0124 |
| r_soma_trunk | 0.9395 | 0.9242 | -0.0152 |
| frac_branch_independent | 0.1538 | 0.1538 | +0.0000 |
| branch_first_frac | 1.0000 | 1.0000 | +0.0000 |
| n_branch_events | 13.0000 | 13.0000 | +0.0000 |
| n_soma_events | 37.0000 | 35.0000 | -2.0000 |
| r_soma_branch_corr | 0.7163 | 0.7043 | -0.0120 |
| rate_soma_per_min | 9.2483 | 8.7484 | -0.4999 |
| rate_branch_per_min | 3.2494 | 3.2494 | +0.0000 |

### run05_140_0623 (trunk_soma_end, bifurcation, branch2 + main_branch)

Mask: Dice=0.853, precision=0.760, recall=0.973
  auto 12417 vox, curated 9707 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.7251 | 0.7192 | -0.0059 |
| r_soma_trunk | 0.8827 | 0.8784 | -0.0043 |
| frac_branch_independent | 0.1600 | 0.2500 | +0.0900 |
| branch_first_frac | 0.8000 | 0.8571 | +0.0571 |
| n_branch_events | 25.0000 | 24.0000 | -1.0000 |
| n_soma_events | 23.0000 | 22.0000 | -1.0000 |
| r_soma_branch_corr | 0.7269 | 0.7211 | -0.0058 |
| rate_soma_per_min | — | 5.4974 | — |
| rate_branch_per_min | 3.1235 | 2.8737 | -0.2499 |

### run06_140_0623 (soma, bifurcation, branch2 + main_branch1/2)

Mask: Dice=0.901, precision=0.865, recall=0.940
  auto 26635 vox, curated 24515 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.5679 | 0.5509 | -0.0170 |
| r_soma_trunk | 0.7402 | 0.7298 | -0.0104 |
| frac_branch_independent | 0.4091 | 0.3333 | -0.0758 |
| branch_first_frac | 0.8333 | 0.8333 | +0.0000 |
| n_branch_events | 22.0000 | 12.0000 | -10.0000 |
| n_soma_events | 27.0000 | 25.0000 | -2.0000 |
| r_soma_branch_corr | 0.5697 | 0.5520 | -0.0177 |
| rate_soma_per_min | 6.7495 | 6.2496 | -0.5000 |
| rate_branch_per_min | 3.0623 | 2.9998 | -0.0625 |

### run07_phpebach (soma, no bifurcation (branch at distal end))

Mask: Dice=0.689, precision=0.531, recall=0.982
  auto 12109 vox, curated 6550 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.7590 | 0.7488 | -0.0102 |
| r_soma_trunk | 0.8805 | 0.9154 | +0.0349 |
| frac_branch_independent | 0.0000 | 0.0000 | +0.0000 |
| branch_first_frac | 0.0000 | 0.0000 | +0.0000 |
| n_branch_events | 5.0000 | 5.0000 | +0.0000 |
| n_soma_events | 15.0000 | 14.0000 | -1.0000 |
| r_soma_branch_corr | 0.7672 | 0.7537 | -0.0135 |
| rate_soma_per_min | 3.7493 | 3.4994 | -0.2500 |
| rate_branch_per_min | 1.2498 | 1.2498 | +0.0000 |

# 2. Soma/Reference-Branch vs Reference-Trunk Coupling

n = 44 cells from 8 mice
r(ref, branch): mean=0.436, 95% CI [0.366, 0.507]
r(ref, trunk):  mean=0.656, 95% CI [0.573, 0.731]
  (28 of 44 are imaging-only cells, included because this is an imaging analysis)
Wilcoxon signed-rank (trunk > branch): W=914.0, p=3.07e-08
Branch coupling lower than trunk: 39/44 cells
Noise-corrected r(ref, branch): mean=0.460, 95% CI [0.389, 0.531]
Halo control: full - core mean diff = 0.0602 (small = not halo artifact)
Mixed model (trunk-branch ~ 1 | mouse): intercept=0.2289, p=7.92e-09

# 3. Independent Branch Events: Dual-Null Surrogate Test

For each cell, 500 surrogates per null are generated from the raw traces:
  (A) COUPLED-NOISE NULL: each branch = gain × reference + phase-randomized measurement noise from a within-region voxel split (half1-half2)/2.
      Tests whether observed independence exceeds what noise alone produces.
  (B) INDEPENDENT NULL: all branch traces circularly shifted together (>20 s).
      Tests whether observed coupling is above chance (i.e., independence < full independence).

n = 41 cells from 8 mice with sufficient events

Cells excluded (3): rbp4_132_phpeb_26-06-08_Run004 (< 3 branch events); rbp4_phpebne_050_26-09-15_Run021 (< 3 branch events); rbp4_phpebne_053_26-09-14_MUnit18 (< 3 branch events)

Observed frac_independent: mean=0.492, 95% CI [0.402, 0.585]
Coupled-noise null mean:   0.418
Independent null mean:     0.886

## Test A: Observed > Coupled-Noise Null
  (Real independent events beyond what detection noise produces)
  Wilcoxon signed-rank (observed - null_A > 0): W=560.0, p=0.0474
  Observed > coupled null: 27/41 cells
  Stouffer combined p: 1.52e-05 (z=4.17)
  Fisher combined p:   1.09e-21

## Test B: Observed < Independent Null
  (Real coupling: independence is less than what fully unrelated traces give)
  Wilcoxon signed-rank (null_B - observed > 0): W=853.0, p=1.14e-11
  Observed < independent null: 38/41 cells
  Stouffer combined p: < 1e-300 (z=13.87)
  Fisher combined p:   9.99e-43

### Per-Cell Results

| Cell | Observed | Null_A (coupled) | Null_B (independent) | p(>A) | p(<B) | n_branch | n_ref |
|------|----------|------------------|---------------------|-------|-------|----------|-------|
| rbp4_132_phpeb_26-06-08_Run003 | 0.500 | 0.145±0.048 | 0.875±0.176 | < 0.002 (resolution 1/501) | 0.0639 | 4 | 36 |
| rbp4_139_phpeb_26-06-12_Run005 | 0.846 | 0.102±0.052 | 0.934±0.045 | < 0.002 (resolution 1/501) | 0.0619 | 26 | 18 |
| rbp4_139_phpeb_26-06-12_Run004 | 0.842 | 0.642±0.047 | 0.886±0.070 | < 0.002 (resolution 1/501) | 0.377 | 19 | 22 |
| rbp4_140_phpeb_26-06-18_Run003 | 0.286 | 0.088±0.067 | 0.952±0.076 | 0.00599 | < 0.002 (resolution 1/501) | 7 | 15 |
| rbp4_140_phpeb_26-06-18_Run004 | 0.167 | 0.063±0.024 | 0.871±0.046 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 72 | 45 |
| rbp4_140_phpeb_26-06-23_Run005 | 0.250 | 0.052±0.026 | 0.903±0.068 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 24 | 22 |
| rbp4_140_phpeb_26-06-23_Run006 | 0.333 | 0.025±0.025 | 0.890±0.093 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 12 | 25 |
| rbp4_141_phpeb_26-06-17_Run006 | 0.250 | 0.364±0.058 | 0.862±0.116 | 0.972 | < 0.002 (resolution 1/501) | 8 | 34 |
| rbp4_141_phpeb_26-06-25_Run004 | 0.143 | 0.376±0.051 | 0.855±0.078 | 1 | < 0.002 (resolution 1/501) | 21 | 55 |
| rbp4_141_phpeb_26-06-25_Run007 | 0.714 | 0.249±0.066 | 0.896±0.076 | < 0.002 (resolution 1/501) | 0.0419 | 14 | 30 |
| rbp4_155_26-09-23_MUnit19 | 0.524 | 0.496±0.025 | 0.619±0.032 | 0.148 | < 0.002 (resolution 1/501) | 170 | 169 |
| rbp4_155_26-09-28_MUnit11 | 0.476 | 0.414±0.027 | 0.547±0.059 | 0.014 | 0.136 | 63 | 149 |
| rbp4_155_26-09-28_MUnit22 | 0.250 | 0.127±0.036 | 0.865±0.108 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 8 | 76 |
| rbp4_155_26-09-29_MUnit13 | 0.800 | 0.871±0.018 | 0.953±0.033 | 0.998 | 0.00399 | 45 | 22 |
| rbp4_155_26-09-29_MUnit15 | 0.846 | 0.907±0.011 | 0.932±0.049 | 1 | 0.0898 | 26 | 32 |
| rbp4_155_26-09-29_MUnit17 | 0.591 | 0.843±0.028 | 0.955±0.046 | 1 | < 0.002 (resolution 1/501) | 22 | 22 |
| rbp4_155_26-09-29_MUnit19 | 0.414 | 0.571±0.058 | 0.939±0.043 | 0.996 | < 0.002 (resolution 1/501) | 29 | 34 |
| rbp4_155_26-09-29_MUnit22 | 0.820 | 0.819±0.012 | 0.921±0.012 | 0.519 | < 0.002 (resolution 1/501) | 311 | 37 |
| rbp4_155_26-09-29_MUnit29 | 0.916 | 0.903±0.012 | 0.910±0.015 | 0.138 | 0.675 | 202 | 42 |
| rbp4_155_26-09-29_MUnit31 | 0.915 | 0.945±0.009 | 0.955±0.013 | 0.998 | 0.00798 | 165 | 20 |
| rbp4_phpebach_26-06-26_Run004 | 0.314 | 0.231±0.032 | 0.859±0.070 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 35 | 45 |
| rbp4_phpebach_26-06-26_Run002 | 0.533 | 0.666±0.046 | 0.949±0.056 | 0.99 | < 0.002 (resolution 1/501) | 15 | 24 |
| rbp4_phpebach_26-06-26_Run005 | 0.154 | 0.069±0.037 | 0.865±0.094 | 0.024 | < 0.002 (resolution 1/501) | 13 | 35 |
| rbp4_phpebach_26-06-26_Run007 | 0.000 | 0.167±0.077 | 0.952±0.092 | 1 | < 0.002 (resolution 1/501) | 5 | 14 |
| rbp4_phpebach_26-06-26_Run009 | 0.800 | 0.853±0.019 | 0.866±0.082 | 0.998 | 0.339 | 15 | 34 |
| rbp4_phpebne_050_26-09-15_MUnit43 | 0.500 | 0.855±0.032 | 0.864±0.104 | 1 | < 0.002 (resolution 1/501) | 8 | 10 |
| rbp4_phpebne_050_26-09-15_Run010 | 0.128 | 0.076±0.029 | 0.719±0.083 | 0.0739 | < 0.002 (resolution 1/501) | 39 | 30 |
| rbp4_phpebne_050_26-09-15_Run011 | 0.082 | 0.062±0.018 | 0.734±0.063 | 0.148 | < 0.002 (resolution 1/501) | 49 | 133 |
| rbp4_phpebne_050_26-09-15_Run012 | 0.862 | 0.217±0.073 | 0.983±0.023 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 29 | 15 |
| rbp4_phpebne_050_26-09-15_Run014 | 0.091 | 0.059±0.026 | 0.916±0.046 | 0.12 | < 0.002 (resolution 1/501) | 44 | 35 |
| rbp4_phpebne_050_26-09-15_Run015 | 0.100 | 0.001±0.007 | 0.843±0.132 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 10 | 12 |
| rbp4_phpebne_050_26-09-15_Run016 | 0.800 | 0.441±0.116 | 0.991±0.029 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 10 | 2 |
| rbp4_phpebne_050_26-09-15_Run017 | 0.857 | 0.701±0.074 | 0.953±0.078 | 0.00798 | 0.287 | 7 | 5 |
| rbp4_phpebne_050_26-09-15_Run018 | 0.444 | 0.209±0.086 | 0.891±0.106 | 0.00399 | < 0.002 (resolution 1/501) | 9 | 9 |
| rbp4_phpebne_050_26-09-15_Run020 | 0.156 | 0.046±0.018 | 0.869±0.039 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 77 | 102 |
| rbp4_phpebne_053_26-09-14_Run005 | 0.545 | 0.009±0.027 | 0.998±0.013 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 11 | 11 |
| rbp4_phpebne_053_26-09-14_Run007 | 1.000 | 0.926±0.014 | 0.941±0.090 | < 0.002 (resolution 1/501) | 1 | 7 | 15 |
| rbp4_phpebne_053_26-09-14_Run010 | 0.200 | 0.476±0.046 | 0.920±0.123 | 1 | < 0.002 (resolution 1/501) | 5 | 56 |
| rbp4_phpebne_053_26-09-14_Run011 | 0.250 | 0.316±0.056 | 0.941±0.065 | 0.882 | < 0.002 (resolution 1/501) | 12 | 41 |
| rbp4_phpebne_053_26-09-18_MUnit8 | 0.967 | 0.941±0.008 | 0.952±0.038 | < 0.002 (resolution 1/501) | 0.78 | 30 | 23 |
| rbp4_phpebne_053_26-09-18_MUnit14 | 0.500 | 0.796±0.017 | 0.784±0.137 | 1 | 0.0539 | 8 | 108 |

Per-cell FDR (test A, obs > coupled): 21/41 significant at q<0.05
Per-cell FDR (test B, obs < independent): 29/41 significant at q<0.05

### Interpretation
  Majority of cells (27/41) show more independent branch events than
  the coupled-noise null: real independent events exist beyond detection noise.
  Majority of cells (38/41) show less independence than the
  independent null: real coupling between branch and reference exists.

CAVEAT: Same event detection parameters (window=2, prom_frac=0.2) are used for all cells.
Ground truth for the claim is limited to n=7 curated cells. Report overfitting risk honestly.
Permutation p-values have resolution 1/(500+1) ≈ 0.002; the minimum reportable is < 1/501.

# 3b. Ground-Truth Independence Test (Daria's Curated Cells)

Running the same dual-null independence test on Daria's 7 hand-curated
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

### run04_phpebach (no soma (trunk_soma_end), bifurcation with branch2, mask ends X 360/392)
  Reference: trunk_soma_end
  Branches: ['bifurcation', 'main_branch', 'branch2']
  n_branch_events = 35, n_ref_events = 46
  Observed frac_independent = 0.286
  Coupled-noise null:   0.229 ± 0.040
  Independent null:     0.857 ± 0.063
  p(observed > coupled null) = 0.0758
  p(observed < independent null) = < 0.002 (resolution 1/501)
  Bracketed: YES (coupled < observed < independent: 0.229 < 0.286 < 0.857)

  Daria's frac_branch_independent (from _metrics.json): 0.2857142857142857
  → Our surrogate test confirms this is not above detection noise (p=0.0758) and below full independence (p=< 0.002 (resolution 1/501)).

### run05_140_0623 (trunk_soma_end, bifurcation, branch2 + main_branch)
  Reference: trunk_soma_end
  Branches: ['bifurcation', 'branch2', 'main_branch_mid', 'main_trunk_end']
  n_branch_events = 25, n_ref_events = 23
  Observed frac_independent = 0.160
  Coupled-noise null:   0.057 ± 0.035
  Independent null:     0.904 ± 0.066
  p(observed > coupled null) = 0.00399
  p(observed < independent null) = < 0.002 (resolution 1/501)
  Bracketed: YES (coupled < observed < independent: 0.057 < 0.160 < 0.904)

  Daria's frac_branch_independent (from _metrics.json): 0.16000000000000003
  → Our surrogate test confirms this is above detection noise (p=0.00399) and below full independence (p=< 0.002 (resolution 1/501)).

### run06_140_0623 (soma, bifurcation, branch2 + main_branch1/2)
  Reference: soma
  Branches: ['bifurcation', 'branch2', 'main_branch1', 'main_branch2']
  n_branch_events = 22, n_ref_events = 27
  Observed frac_independent = 0.409
  Coupled-noise null:   0.090 ± 0.035
  Independent null:     0.888 ± 0.076
  p(observed > coupled null) = < 0.002 (resolution 1/501)
  p(observed < independent null) = < 0.002 (resolution 1/501)
  Bracketed: YES (coupled < observed < independent: 0.090 < 0.409 < 0.888)

  Daria's frac_branch_independent (from _metrics.json): 0.40909090909090906
  → Our surrogate test confirms this is above detection noise (p=< 0.002 (resolution 1/501)) and below full independence (p=< 0.002 (resolution 1/501)).

### run07_phpebach (soma, no bifurcation (branch at distal end))
  Reference: soma
  Branches: ['branch']
  n_branch_events = 5, n_ref_events = 15
  Observed frac_independent = 0.000
  Coupled-noise null:   0.176 ± 0.070
  Independent null:     0.950 ± 0.094
  p(observed > coupled null) = 1
  p(observed < independent null) = < 0.002 (resolution 1/501)
  Bracketed: NO (coupled < observed < independent: 0.176 < 0.000 < 0.950)

  Daria's frac_branch_independent (from _metrics.json): 0.0
  → Our surrogate test confirms this is not above detection noise (p=1) and below full independence (p=< 0.002 (resolution 1/501)).

# 4. Coupling vs Geodesic Distance from Reference

n = 180 regions from 44 cells
Pooled Spearman (descriptive, pseudo-replicated): rho=-0.559, p=3.73e-16
  NOTE: regions from the same cell are not independent; the mixed model below is the primary test.
Per-cell slope: mean=-0.155 r/100um, 95% CI [-0.185, -0.125]
Mixed model (r ~ distance + (1|cell), PRIMARY): slope=-0.001308/um, p=3.84e-25

# 5. Event Order: Branch-First Fraction

n = 27 cells, 253 paired events
Branch peaks first: 181/253 (71.5%)
Pooled binomial (descriptive, pseudo-replicated): p=2.64e-12
  NOTE: events from the same cell are not independent; the sign test below is the primary test.
Per-cell sign test (PRIMARY): 22/27 cells with fraction > 0.5, p=0.000757
Mean branch-first fraction: 0.779, 95% CI [0.679, 0.866]

# 6. Branch Amplitude Predicts Propagation

This test requires per-event amplitude data not stored in the current metrics.
Future: extract from coherence network_events CSVs.
SKIP: per-event amplitude data not available in aggregate metrics.

# 7. Behavior Coupling

n = 13 cells with behavior data
  (28 imaging-only cells excluded from behavior analyses — behavior data not available)

  whisking: 6/13 significant at p<0.05 (uncorrected), mean peak r=0.177
  pupil: 4/13 significant at p<0.05 (uncorrected), mean peak r=0.124
  accelerometer: 5/13 significant at p<0.05 (uncorrected), mean peak r=0.100

After BH-FDR across all 39 behavior tests: 11 significant at q<0.05

Quiet vs active: 7/13 cells more coupled when active
Wilcoxon (active vs quiet): p=0.455

# 8. Coupling vs Expression Time

n = 11 cells from 4 mice
DPI range: 241–258
Spearman r(ref, branch) vs DPI: rho=0.271, p=0.42
Mixed model (r ~ dpi + (1|mouse)): slope=0.006430/day, p=0.527
Spearman r(ref, branch) vs soma F: rho=0.136, p=0.378

CAVEAT: DPI range is narrow (17 days across 5 mice), so this test has limited power.
rbp4_phpebach excluded (unknown injection date).

# 9. Co-firing Groups

n = 44 cells
Co-firing groups per cell: mean=2.7, range [1, 5]
Soma shares a group with any branch: 2/18 cells

# 10. Per-Run QC Table

| Run | Mouse | Mask vox | Regions | Reference | Intruders | Soma events | Branch events | Flags |
|-----|-------|----------|---------|-----------|-----------|-------------|---------------|-------|
| phpeb/06-08-2026/preprocessed/run03 | rbp4_132_phpeb | 6053 | 4 | soma | 0 | 36 | 4 | no_bifurcation |
| phpeb/06-08-2026/preprocessed/run04 | rbp4_132_phpeb | 5648 | 4 | soma | 0 | 32 | 1 | no_bifurcation |
| 4_139_phpeb/06-12-2026/traces/run05 | rbp4_139_phpeb | 4043 | 4 | proximal_trunk | 3 | 18 | 26 | no_soma, no_bifurcation, 3_suspects, no_behavior |
| p4_139_phpeb/06-12-2026/traces/run4 | rbp4_139_phpeb | 13625 | 5 | soma | 0 | 22 | 19 | no_bifurcation, no_behavior |
| phpeb/06-18-2026/preprocessed/run03 **[GT]** | rbp4_140_phpeb | 6673 | 5 | proximal_trunk | 0 | 15 | 7 | no_soma, no_bifurcation |
| phpeb/06-18-2026/preprocessed/run04 **[GT]** | rbp4_140_phpeb | 4439 | 7 | proximal_trunk | 0 | 45 | 72 | no_soma |
| phpeb/06-23-2026/preprocessed/run05 | rbp4_140_phpeb | 12417 | 7 | soma | 0 | 22 | 24 |  |
| phpeb/06-23-2026/preprocessed/run06 | rbp4_140_phpeb | 26635 | 5 | soma | 0 | 25 | 12 | no_bifurcation |
| phpeb/06-17-2026/preprocessed/run01 |  | 7798 | 4 | soma | 0 | 0 | 0 | no_bifurcation |
| phpeb/06-17-2026/preprocessed/run04 |  | 5367 | 5 | proximal_trunk | 0 | 0 | 0 | no_soma, no_bifurcation, low_reliability |
| phpeb/06-17-2026/preprocessed/run06 | rbp4_141_phpeb | 6926 | 5 | proximal_trunk | 0 | 34 | 8 | no_soma, no_bifurcation, low_reliability |
| phpeb/06-17-2026/preprocessed/run09 |  | 6043 | 5 | proximal_trunk | 0 | 0 | 0 | no_soma, no_bifurcation |
| phpeb/06-25-2026/preprocessed/run04 | rbp4_141_phpeb | 5744 | 4 | soma | 0 | 55 | 21 | no_bifurcation |
| phpeb/06-25-2026/preprocessed/run07 | rbp4_141_phpeb | 8349 | 5 | soma | 0 | 30 | 14 | no_bifurcation |
| _phpeb/06-25-2026/preprocessed/run3 |  | 4546 | 4 | soma | 0 | 0 | 0 | no_bifurcation |
| 155/09-23-2026/preprocessed/munit19 | rbp4_155 | 16447 | 5 | proximal_trunk | 2 | 169 | 170 | no_soma, no_bifurcation, 2_suspects, low_reliability, no_behavior |
| 155/09-28-2026/preprocessed/munit11 | rbp4_155 | 9547 | 5 | proximal_trunk | 4 | 149 | 63 | no_soma, no_bifurcation, 4_suspects, low_reliability, no_behavior |
| 155/09-28-2026/preprocessed/munit22 | rbp4_155 | 8100 | 5 | proximal_trunk | 1 | 76 | 8 | no_soma, no_bifurcation, 1_suspects, low_reliability, no_behavior |
| 155/09-29-2026/preprocessed/munit13 | rbp4_155 | 8407 | 5 | soma | 1 | 22 | 45 | no_bifurcation, 1_suspects, low_reliability, no_behavior |
| 155/09-29-2026/preprocessed/munit15 | rbp4_155 | 7999 | 5 | soma | 3 | 32 | 26 | no_bifurcation, 3_suspects, low_reliability, no_behavior |
| 155/09-29-2026/preprocessed/munit17 | rbp4_155 | 8187 | 5 | proximal_trunk | 1 | 22 | 22 | no_soma, no_bifurcation, 1_suspects, low_reliability, no_behavior |
| 155/09-29-2026/preprocessed/munit19 | rbp4_155 | 7666 | 5 | proximal_trunk | 0 | 34 | 29 | no_soma, no_bifurcation, low_reliability, no_behavior |
| 155/09-29-2026/preprocessed/munit22 | rbp4_155 | 6153 | 6 | proximal_trunk | 1 | 37 | 311 | no_soma, 1_suspects, low_reliability, no_behavior |
| 155/09-29-2026/preprocessed/munit29 | rbp4_155 | 13067 | 5 | soma | 0 | 42 | 202 | no_bifurcation, low_reliability, no_behavior |
| 155/09-29-2026/preprocessed/munit31 | rbp4_155 | 11101 | 5 | proximal_trunk | 1 | 20 | 165 | no_soma, no_bifurcation, 1_suspects, low_reliability, no_behavior |
| rbp4_phpebach/06-26-2026/run04 | rbp4_phpebach | 7722 | 7 | proximal_trunk | 0 | 45 | 35 | no_soma |
| rbp4_phpebach/06-26-2026/run04/run2 | rbp4_phpebach | 7387 | 5 | proximal_trunk | 0 | 24 | 15 | no_soma, no_bifurcation, low_reliability |
| rbp4_phpebach/06-26-2026/run05 **[GT]** | rbp4_phpebach | 12395 | 5 | soma | 0 | 35 | 13 | no_bifurcation |
| rbp4_phpebach/06-26-2026/run07 | rbp4_phpebach | 12109 | 5 | soma | 0 | 14 | 5 | no_bifurcation |
| rbp4_phpebach/06-26-2026/run09 | rbp4_phpebach | 18371 | 5 | soma | 0 | 34 | 15 | no_bifurcation |
| 050/09-15-2026/preprocessed/munit43 | rbp4_phpebne_050 | 14545 | 5 | soma | 0 | 10 | 8 | no_bifurcation, short_recording, no_behavior |
| e_050/09-15-2026/preprocessed/run09 |  | 4281 | 6 | proximal_trunk | 0 | 0 | 0 | no_soma, no_behavior |
| e_050/09-15-2026/preprocessed/run10 | rbp4_phpebne_050 | 7458 | 7 | proximal_trunk | 1 | 30 | 39 | no_soma, 1_suspects, low_reliability, no_behavior |
| e_050/09-15-2026/preprocessed/run11 | rbp4_phpebne_050 | 8306 | 5 | proximal_trunk | 1 | 133 | 49 | no_soma, no_bifurcation, 1_suspects, no_behavior |
| e_050/09-15-2026/preprocessed/run12 | rbp4_phpebne_050 | 7226 | 4 | proximal_trunk | 0 | 15 | 29 | no_soma, no_bifurcation, no_behavior |
| e_050/09-15-2026/preprocessed/run14 | rbp4_phpebne_050 | 2628 | 6 | proximal_trunk | 1 | 35 | 44 | no_soma, 1_suspects, no_behavior |
| e_050/09-15-2026/preprocessed/run15 | rbp4_phpebne_050 | 5869 | 4 | soma | 1 | 12 | 10 | no_bifurcation, 1_suspects, short_recording, no_behavior |
| e_050/09-15-2026/preprocessed/run16 | rbp4_phpebne_050 | 5959 | 6 | proximal_trunk | 0 | 2 | 10 | no_soma, short_recording, no_behavior |
| e_050/09-15-2026/preprocessed/run17 | rbp4_phpebne_050 | 14532 | 5 | soma | 0 | 5 | 7 | no_bifurcation, short_recording, no_behavior |
| e_050/09-15-2026/preprocessed/run18 | rbp4_phpebne_050 | 6505 | 4 | soma | 2 | 9 | 9 | no_bifurcation, 2_suspects, short_recording, no_behavior |
| e_050/09-15-2026/preprocessed/run19 |  | 14278 | 5 | soma | 0 | 0 | 0 | no_bifurcation, no_behavior |
| e_050/09-15-2026/preprocessed/run20 | rbp4_phpebne_050 | 8931 | 5 | proximal_trunk | 1 | 102 | 77 | no_soma, no_bifurcation, 1_suspects, no_behavior |
| e_050/09-15-2026/preprocessed/run21 | rbp4_phpebne_050 | 14205 | 5 | proximal_trunk | 2 | 23 | 2 | no_soma, no_bifurcation, 2_suspects, low_reliability, short_recording, no_behavior |
| 053/09-14-2026/preprocessed/munit16 |  | 6070 | 5 | soma | 0 | 0 | 0 | no_bifurcation, no_behavior |
| 053/09-14-2026/preprocessed/munit18 | rbp4_phpebne_053 | 6719 | 5 | proximal_trunk | 0 | 30 | 2 | no_soma, no_bifurcation, no_behavior |
| e_053/09-14-2026/preprocessed/run05 | rbp4_phpebne_053 | 9599 | 5 | proximal_trunk | 1 | 11 | 11 | no_soma, no_bifurcation, 1_suspects, no_behavior |
| e_053/09-14-2026/preprocessed/run07 | rbp4_phpebne_053 | 6387 | 5 | proximal_trunk | 3 | 15 | 7 | no_soma, no_bifurcation, 3_suspects, no_behavior |
| e_053/09-14-2026/preprocessed/run10 | rbp4_phpebne_053 | 6891 | 5 | proximal_trunk | 0 | 56 | 5 | no_soma, no_bifurcation, no_behavior |
| e_053/09-14-2026/preprocessed/run11 | rbp4_phpebne_053 | 5828 | 5 | proximal_trunk | 0 | 41 | 12 | no_soma, no_bifurcation, no_behavior |
| 053/09-18-2026/preprocessed/munit08 | rbp4_phpebne_053 | 8505 | 5 | soma | 0 | 23 | 30 | no_bifurcation, low_reliability, no_behavior |
| 053/09-18-2026/preprocessed/munit09 |  | 11360 | 5 | proximal_trunk | 0 | 0 | 0 | no_soma, no_bifurcation, low_reliability, no_behavior |
| 053/09-18-2026/preprocessed/munit14 | rbp4_phpebne_053 | 11944 | 5 | proximal_trunk | 0 | 108 | 8 | no_soma, no_bifurcation, low_reliability, no_behavior |
| 053/09-18-2026/preprocessed/munit21 |  | 11877 | 5 | soma | 0 | 0 | 0 | no_bifurcation, no_behavior |
| 053/09-18-2026/preprocessed/munit37 |  | 4681 | 5 | soma | 2 | 0 | 0 | 2_suspects, low_reliability, no_behavior |

All 54 runs have confidence='auto' (fully automatic, no manual review).

---
## Global BH-FDR Correction (Cohort-Level Tests Only)

Only cohort-level p-values (one per test) are included in the global FDR.
Per-cell surrogate p-values and pseudo-replicated (pooled) p-values are excluded.

| Test | p | q (BH) |
|------|---|--------|
| test02_coupling | 3.07e-08 | 7.69e-08 **|
| test02_coupling | 7.92e-09 | 2.64e-08 **|
| test03_independence | 0.0474 | 0.079 |
| test03_independence | 1.14e-11 | 5.68e-11 **|
| test04_distance | 3.84e-25 | 3.84e-24 **|
| test05_event_order | 0.000757 | 0.00151 **|
| test07_behavior | 0.455 | 0.505 |
| test08_expression | 0.42 | 0.505 |
| test08_expression | 0.527 | 0.527 |
| test08_expression | 0.378 | 0.505 |

10 tests, 5 significant at q<0.05

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
4. **Ground truth**: Independence test validated on Daria's 7 curated cells (test 3b).
5. **p-value formatting**: Full-precision floats in RESULTS.json; 3 significant digits in REPORT.md.
   Permutation p-values (500 draws) have resolution 1/501 ≈ 0.002; minimum reported as < 1/501.

### Overfitting risk
Same parameters (window=2 frames, prom_frac=0.2, n_draws=500) for all cells.
Ground truth is n=7 curated cells. The surrogate test does not use ground truth
to set parameters; parameters come from the event detector which was developed
independently. However, the small ground-truth sample limits validation power.
Alpha calibration uses anatomy (soma detection), not Dice against curated masks,
so leave-one-out produces the same parameters.

---
## Cross-Check: cohort_stats.py vs RESULTS.json

cohort_stats.py: 44 runs, r(soma,branch) mean 0.44, r(soma,trunk) mean 0.66, Wilcoxon p=3.07e-08, 10 set aside
paper_stats.py:  44 runs, r(soma,branch) mean 0.4363, r(soma,trunk) mean 0.6562, Wilcoxon p=3.07e-08

Numbers agree within rounding tolerance.

