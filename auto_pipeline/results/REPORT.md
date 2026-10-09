# Automatic Pipeline Statistical Report (v4)

Generated: 2026-10-09 13:28
n = 17 cells from 7 mice (14 with behavior, 3 imaging-only)
All masks and regions fully automatic (no manual curation).
Set aside (run_marks.csv; reasons starting [auto-QC] are automatic, others are Daria's), not in any statistic: rbp4_139_phpeb_26-06-12_Run002 (revisit); rbp4_139_phpeb_26-06-12_Run004 (excluded); rbp4_140_phpeb_26-06-18_Run002 (excluded: very noisy (Daria, 2026-10-05)); rbp4_140_phpeb_26-06-23_Run002 (excluded: very noisy (Daria, 2026-10-05)); rbp4_141_phpeb_26-06-17_Run001 (revisit); rbp4_141_phpeb_26-06-17_Run004 (excluded); rbp4_141_phpeb_26-06-17_Run009 (revisit); rbp4_141_phpeb_26-06-23_Run001 (excluded: pretty noisy (Daria, 2026-10-05)); rbp4_141_phpeb_26-06-25_Run003 (excluded); rbp4_141_phpeb_26-06-25_Run004 (excluded); rbp4_155_26-09-23_MUnit19 (excluded); rbp4_155_26-09-28_MUnit11 (excluded); rbp4_155_26-09-28_MUnit22 (excluded); rbp4_155_26-09-29_MUnit15 (excluded); rbp4_155_26-09-29_MUnit17 (excluded); rbp4_155_26-09-29_MUnit19 (excluded); rbp4_155_26-09-29_MUnit22 (excluded); rbp4_155_26-09-29_MUnit29 (excluded); rbp4_155_26-09-29_MUnit31 (excluded); rbp4_phpebne_050_26-09-15_MUnit43 (excluded); rbp4_phpebne_050_26-09-15_Run009 (revisit: [auto-QC] large motion drift: reference volume needed block registration, traces are unregistered - check before use); rbp4_phpebne_050_26-09-15_Run010 (excluded: [z-QC] thin slab: pc1 90.6%, 2 effective planes of 10; Z not independently resolved); rbp4_phpebne_050_26-09-15_Run011 (excluded: [z-QC] thin slab: pc1 93.1%, 2 effective planes of 10; worst of the set); rbp4_phpebne_050_26-09-15_Run012 (excluded: [z-QC] thin slab: pc1 88.9%, 2 effective planes of 10); rbp4_phpebne_050_26-09-15_Run014 (excluded: [z-QC] thin slab: pc1 84.2%, 2 effective planes of 17); rbp4_phpebne_050_26-09-15_Run015 (excluded); rbp4_phpebne_050_26-09-15_Run016 (excluded); rbp4_phpebne_050_26-09-15_Run017 (excluded); rbp4_phpebne_050_26-09-15_Run018 (revisit: [z-QC] borderline: pc1 84.5% (above the 77% June max) but 5 effective planes of 17 - 3D geometry weak, in-plane analysis fine); rbp4_phpebne_050_26-09-15_Run019 (revisit: [z-QC] borderline: pc1 83.1% but 9 effective planes of 17 - the best-resolved of the 09-15 set after Run011/MUnit_22); rbp4_phpebne_050_26-09-15_Run020 (excluded); rbp4_phpebne_050_26-09-15_Run021 (excluded); rbp4_phpebne_053_26-09-14_MUnit16 (revisit: [auto-QC] cell sits on the tube's Z edge for 80% of its length (partly out of frame); comment: forgot behavior); rbp4_phpebne_053_26-09-14_Run005 (excluded); rbp4_phpebne_053_26-09-14_Run007 (excluded: [z-QC] thin slab: 1 shared image explains 90.4% of across-plane variance, 2 components for 95% (June P1 range 33-77%, 4-14 comps); Z is not independently resolved); rbp4_phpebne_053_26-09-14_Run010 (excluded); rbp4_phpebne_053_26-09-18_MUnit14 (excluded); rbp4_phpebne_053_26-09-18_MUnit21 (revisit: [auto-QC] automatic mask missed the dendrite (activity outside the mask 6.6x inside; the dendrite is clearly visible - hand-mask it); comment: good dendrites, bad soma); rbp4_phpebne_053_26-09-18_MUnit37 (revisit: [auto-QC] cell runs along the tube's Z edge for 83% of its length (partly out of frame); right half of the tube bright through all planes - check); rbp4_phpebne_053_26-09-18_MUnit8 (excluded); rbp4_phpebne_053_26-09-18_MUnit9 (revisit: [auto-QC] no cell (imaging comment: 'run 5. no cell'); no regions could be placed)

Imaging-only cells (listed in imaging_only_runs.csv; behavior not yet processed) are included in all imaging analyses (coupling, independence, distance, event order) and excluded from behavior analyses (test 7). This is stated in each test.

**Key change from v1**: Independence test now uses dual-null surrogates
(coupled-noise null and circular-shift independent null) instead of the
trivially-significant Wilcoxon-vs-0 and wrong-direction binomial.

## Summary

Across 17 cells from 7 mice (14 with behavior, 3 imaging-only):
- **Branches are less coupled to the reference than trunks**: mean r(ref, branch)=0.5345, r(ref, trunk)=0.7574, paired Wilcoxon p=7.63e-06 (17 cells, 7 mice).
- **Branches have genuine independent events**: 15 cells tested; mean frac_independent=0.4019, bracketed between coupled-noise and independent null distributions.
- **Coupling decays with geodesic distance**: 17 cells, Spearman ρ=-0.5121.
- **Branch tends to fire first**: mean fraction=0.8522 (11 cells).
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
| r_soma_branch | 0.6563 | 0.5233 | -0.1330 |
| r_soma_trunk | 0.9305 | 0.8407 | -0.0898 |
| frac_branch_independent | 0.5769 | 0.2857 | -0.2912 |
| branch_first_frac | 1.0000 | 1.0000 | +0.0000 |
| n_branch_events | 26.0000 | 7.0000 | -19.0000 |
| n_soma_events | 14.0000 | 15.0000 | +1.0000 |
| r_soma_branch_corr | 0.6597 | 0.5254 | -0.1343 |
| rate_branch_per_min | 3.3315 | 1.7490 | -1.5825 |

### run04 (soma + bifurcation, another cell at high X limits auto mask extent)

Mask: Dice=0.688, precision=0.622, recall=0.770
  auto 4439 vox, curated 3584 vox

| Metric | Manual | Auto | Δ |
|--------|--------|------|---|
| r_soma_branch | 0.5585 | 0.8497 | +0.2912 |
| r_soma_trunk | 0.9225 | 0.9348 | +0.0123 |
| frac_branch_independent | 0.4818 | 0.1389 | -0.3429 |
| branch_first_frac | 0.5789 | 1.0000 | +0.4211 |
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
| frac_branch_independent | 0.2000 | 0.2000 | +0.0000 |
| branch_first_frac | 0.5455 | 0.6000 | +0.0545 |
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

n = 17 cells from 7 mice
r(ref, branch): mean=0.535, 95% CI [0.425, 0.642]
r(ref, trunk):  mean=0.757, 95% CI [0.657, 0.846]
  (3 of 17 are imaging-only cells, included because this is an imaging analysis)
Wilcoxon signed-rank (trunk > branch): W=153.0, p=7.63e-06
Branch coupling lower than trunk: 17/17 cells
Noise-corrected r(ref, branch): mean=0.545, 95% CI [0.432, 0.655]
Halo control: full - core mean diff = 0.0487 (small = not halo artifact)
Mixed model (trunk-branch ~ 1 | mouse): intercept=0.2228, p=1.41e-05

# 3. Independent Branch Events: Dual-Null Surrogate Test

For each cell, 500 surrogates per null are generated from the raw traces:
  (A) COUPLED-NOISE NULL: each branch = gain × reference + phase-randomized measurement noise from a within-region voxel split (half1-half2)/2.
      Tests whether observed independence exceeds what noise alone produces.
  (B) INDEPENDENT NULL: all branch traces circularly shifted together (>20 s).
      Tests whether observed coupling is above chance (i.e., independence < full independence).

n = 15 cells from 7 mice with sufficient events

Cells excluded (2): rbp4_132_phpeb_26-06-08_Run004 (< 3 branch events); rbp4_phpebne_053_26-09-14_MUnit18 (< 3 branch events)

Observed frac_independent: mean=0.402, 95% CI [0.278, 0.535]
Coupled-noise null mean:   0.245
Independent null mean:     0.906

## Test A: Observed > Coupled-Noise Null
  (Real independent events beyond what detection noise produces)
  Wilcoxon signed-rank (observed - null_A > 0): W=99.0, p=0.0128
  Observed > coupled null: 10/15 cells
  Stouffer combined p: 2.29e-05 (z=4.08)
  Fisher combined p:   2.69e-12

## Test B: Observed < Independent Null
  (Real coupling: independence is less than what fully unrelated traces give)
  Wilcoxon signed-rank (null_B - observed > 0): W=120.0, p=3.05e-05
  Observed < independent null: 15/15 cells
  Stouffer combined p: < 1e-300 (z=9.46)
  Fisher combined p:   8.82e-19

### Per-Cell Results

| Cell | Observed | Null_A (coupled) | Null_B (independent) | p(>A) | p(<B) | n_branch | n_ref |
|------|----------|------------------|---------------------|-------|-------|----------|-------|
| rbp4_132_phpeb_26-06-08_Run003 | 0.500 | 0.145±0.048 | 0.875±0.176 | < 0.002 (resolution 1/501) | 0.0639 | 4 | 36 |
| rbp4_139_phpeb_26-06-12_Run005 | 0.846 | 0.102±0.052 | 0.934±0.045 | < 0.002 (resolution 1/501) | 0.0619 | 26 | 18 |
| rbp4_140_phpeb_26-06-18_Run003 | 0.286 | 0.088±0.067 | 0.952±0.076 | 0.00599 | < 0.002 (resolution 1/501) | 7 | 15 |
| rbp4_140_phpeb_26-06-18_Run004 | 0.167 | 0.063±0.024 | 0.871±0.046 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 72 | 45 |
| rbp4_140_phpeb_26-06-23_Run005 | 0.250 | 0.052±0.026 | 0.903±0.068 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 24 | 22 |
| rbp4_140_phpeb_26-06-23_Run006 | 0.333 | 0.025±0.025 | 0.890±0.093 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 12 | 25 |
| rbp4_141_phpeb_26-06-17_Run006 | 0.250 | 0.364±0.058 | 0.862±0.116 | 0.972 | < 0.002 (resolution 1/501) | 8 | 34 |
| rbp4_141_phpeb_26-06-25_Run007 | 0.714 | 0.249±0.066 | 0.896±0.076 | < 0.002 (resolution 1/501) | 0.0419 | 14 | 30 |
| rbp4_155_26-09-29_MUnit13 | 0.800 | 0.871±0.018 | 0.953±0.033 | 0.998 | 0.00399 | 45 | 22 |
| rbp4_phpebach_26-06-26_Run002 | 0.364 | 0.081±0.064 | 0.967±0.053 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 11 | 13 |
| rbp4_phpebach_26-06-26_Run004 | 0.314 | 0.231±0.032 | 0.859±0.070 | < 0.002 (resolution 1/501) | < 0.002 (resolution 1/501) | 35 | 45 |
| rbp4_phpebach_26-06-26_Run005 | 0.154 | 0.069±0.037 | 0.865±0.094 | 0.024 | < 0.002 (resolution 1/501) | 13 | 35 |
| rbp4_phpebach_26-06-26_Run007 | 0.000 | 0.167±0.077 | 0.952±0.092 | 1 | < 0.002 (resolution 1/501) | 5 | 14 |
| rbp4_phpebach_26-06-26_Run009 | 0.800 | 0.853±0.019 | 0.866±0.082 | 0.998 | 0.339 | 15 | 34 |
| rbp4_phpebne_053_26-09-14_Run011 | 0.250 | 0.316±0.056 | 0.941±0.065 | 0.882 | < 0.002 (resolution 1/501) | 12 | 41 |

Per-cell FDR (test A, obs > coupled): 10/15 significant at q<0.05
Per-cell FDR (test B, obs < independent): 11/15 significant at q<0.05

### Interpretation
  Majority of cells (10/15) show more independent branch events than
  the coupled-noise null: real independent events exist beyond detection noise.
  Majority of cells (15/15) show less independence than the
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
  Branches: ['bifurcation', 'branch1', 'branch1far']
  n_branch_events = 26, n_ref_events = 14
  Observed frac_independent = 0.577
  Coupled-noise null:   0.196 ± 0.052
  Independent null:     0.955 ± 0.047
  p(observed > coupled null) = < 0.002 (resolution 1/501)
  p(observed < independent null) = < 0.002 (resolution 1/501)
  Bracketed: YES (coupled < observed < independent: 0.196 < 0.577 < 0.955)

  Daria's frac_branch_independent (from _metrics.json): 0.5769230769230769
  → Our surrogate test confirms this is above detection noise (p=< 0.002 (resolution 1/501)) and below full independence (p=< 0.002 (resolution 1/501)).

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

  Daria's frac_branch_independent (from _metrics.json): 0.4818181818181818
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

  Daria's frac_branch_independent (from _metrics.json): 0.19999999999999996
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

n = 71 regions from 17 cells
Pooled Spearman (descriptive, pseudo-replicated): rho=-0.512, p=4.99e-06
  NOTE: regions from the same cell are not independent; the mixed model below is the primary test.
Per-cell slope: mean=-0.149 r/100um, 95% CI [-0.196, -0.108]
Mixed model (r ~ distance + (1|cell), PRIMARY): slope=-0.001309/um, p=2.08e-18

# 5. Event Order: Branch-First Fraction

n = 11 cells, 58 paired events
Branch peaks first: 49/58 (84.5%)
Pooled binomial (descriptive, pseudo-replicated): p=4.48e-08
  NOTE: events from the same cell are not independent; the sign test below is the primary test.
Per-cell sign test (PRIMARY): 11/11 cells with fraction > 0.5, p=0.000488
Mean branch-first fraction: 0.852, 95% CI [0.761, 0.941]

# 6. Branch Amplitude Predicts Propagation

This test requires per-event amplitude data not stored in the current metrics.
Future: extract from coherence network_events CSVs.
SKIP: per-event amplitude data not available in aggregate metrics.

# 7. Behavior Coupling

n = 13 cells with behavior data
  (3 imaging-only cells excluded from behavior analyses — behavior data not available)

  whisking: 6/13 significant at p<0.05 (uncorrected), mean peak r=0.186
  pupil: 4/13 significant at p<0.05 (uncorrected), mean peak r=0.138
  accelerometer: 5/13 significant at p<0.05 (uncorrected), mean peak r=0.091

After BH-FDR across all 39 behavior tests: 11 significant at q<0.05

Quiet vs active: 7/13 cells more coupled when active
Wilcoxon (active vs quiet): p=0.455

# 8. Coupling vs Expression Time

n = 9 cells from 4 mice
DPI range: 241–258
Spearman r(ref, branch) vs DPI: rho=0.093, p=0.812
Mixed model (r ~ dpi + (1|mouse)): slope=-0.002402/day, p=0.881
Spearman r(ref, branch) vs soma F: rho=-0.471, p=0.0566

CAVEAT: DPI range is narrow (17 days across 5 mice), so this test has limited power.
rbp4_phpebach excluded (unknown injection date).

# 9. Co-firing Groups

n = 18 cells
Co-firing groups per cell: mean=1.9, range [1, 3]
Soma shares a group with any branch: 2/9 cells

# 10. Per-Run QC Table

| Run | Mouse | Rating | Mask vox | Regions | Reference | Intruders | Soma events | Branch events | Flags |
|-----|-------|--------|----------|---------|-----------|-----------|-------------|---------------|-------|
| phpeb/06-08-2026/preprocessed/run03 | rbp4_132_phpeb | - | 6053 | 4 | soma | 0 | 36 | 4 | no_bifurcation |
| phpeb/06-08-2026/preprocessed/run04 | rbp4_132_phpeb | - | 5648 | 4 | soma | 0 | 32 | 1 | no_bifurcation |
| 4_139_phpeb/06-12-2026/traces/run04 |  | - | 9582 | 5 | proximal_trunk | 0 | 0 | 0 | no_soma, no_bifurcation, no_behavior |
| 4_139_phpeb/06-12-2026/traces/run05 | rbp4_139_phpeb | - | 4043 | 4 | proximal_trunk | 3 | 18 | 26 | no_soma, no_bifurcation, 3_suspects, no_behavior |
| phpeb/06-18-2026/preprocessed/run03 **[GT]** | rbp4_140_phpeb | - | 6673 | 5 | proximal_trunk | 0 | 15 | 7 | no_soma, no_bifurcation |
| phpeb/06-18-2026/preprocessed/run04 **[GT]** | rbp4_140_phpeb | - | 4439 | 7 | proximal_trunk | 0 | 45 | 72 | no_soma |
| phpeb/06-23-2026/preprocessed/run05 | rbp4_140_phpeb | - | 12417 | 7 | soma | 0 | 22 | 24 |  |
| phpeb/06-23-2026/preprocessed/run06 | rbp4_140_phpeb | - | 26635 | 5 | soma | 0 | 25 | 12 | no_bifurcation |
| phpeb/06-17-2026/preprocessed/run01 |  | - | 7798 | 4 | soma | 0 | 0 | 0 | no_bifurcation |
| phpeb/06-17-2026/preprocessed/run04 |  | - | 5367 | 5 | proximal_trunk | 0 | 0 | 0 | no_soma, no_bifurcation, low_reliability |
| phpeb/06-17-2026/preprocessed/run06 | rbp4_141_phpeb | - | 6926 | 5 | proximal_trunk | 0 | 34 | 8 | no_soma, no_bifurcation, low_reliability |
| phpeb/06-17-2026/preprocessed/run09 |  | - | 6043 | 5 | proximal_trunk | 0 | 0 | 0 | no_soma, no_bifurcation |
| _WRONG_NZ15_was_MUnit4_misextracted |  | - | 4546 | 4 | soma | 0 | 0 | 0 | no_bifurcation, no_behavior |
| phpeb/06-25-2026/preprocessed/run03 |  | - | 8133 | 4 | soma | 0 | 0 | 0 | no_bifurcation |
| phpeb/06-25-2026/preprocessed/run07 | rbp4_141_phpeb | - | 8349 | 5 | soma | 0 | 30 | 14 | no_bifurcation |
| 155/09-29-2026/preprocessed/munit13 | rbp4_155 | - | 8407 | 5 | soma | 1 | 22 | 45 | no_bifurcation, 1_suspects, low_reliability, no_behavior |
| _WRONG_NZ19_was_MUnit3_misextracted | rbp4_phpebach | - | 7387 | 5 | proximal_trunk | 0 | 24 | 15 | no_soma, no_bifurcation, low_reliability, no_behavior |
| rbp4_phpebach/06-26-2026/run02 | rbp4_phpebach | - | 6040 | 5 | proximal_trunk | 0 | 13 | 11 | no_soma, no_bifurcation |
| rbp4_phpebach/06-26-2026/run04 | rbp4_phpebach | - | 7722 | 7 | proximal_trunk | 0 | 45 | 35 | no_soma |
| rbp4_phpebach/06-26-2026/run05 **[GT]** | rbp4_phpebach | - | 12395 | 5 | soma | 0 | 35 | 13 | no_bifurcation |
| rbp4_phpebach/06-26-2026/run07 | rbp4_phpebach | - | 12109 | 5 | soma | 0 | 14 | 5 | no_bifurcation |
| rbp4_phpebach/06-26-2026/run09 | rbp4_phpebach | - | 18371 | 5 | soma | 0 | 34 | 15 | no_bifurcation |
| e_050/09-15-2026/preprocessed/run09 |  | - | 4281 | 6 | proximal_trunk | 0 | 0 | 0 | no_soma, no_behavior |
| e_050/09-15-2026/preprocessed/run19 |  | - | 14278 | 5 | soma | 0 | 0 | 0 | no_bifurcation, no_behavior |
| 053/09-14-2026/preprocessed/munit16 |  | - | 6070 | 5 | soma | 0 | 0 | 0 | no_bifurcation, no_behavior |
| 053/09-14-2026/preprocessed/munit18 | rbp4_phpebne_053 | - | 6719 | 5 | proximal_trunk | 0 | 30 | 2 | no_soma, no_bifurcation, no_behavior |
| e_053/09-14-2026/preprocessed/run11 | rbp4_phpebne_053 | - | 5828 | 5 | proximal_trunk | 0 | 41 | 12 | no_soma, no_bifurcation, no_behavior |
| 053/09-18-2026/preprocessed/munit09 |  | - | 11360 | 5 | proximal_trunk | 0 | 0 | 0 | no_soma, no_bifurcation, low_reliability, no_behavior |
| 053/09-18-2026/preprocessed/munit21 |  | - | 11877 | 5 | soma | 0 | 0 | 0 | no_bifurcation, no_behavior |
| 053/09-18-2026/preprocessed/munit37 |  | - | 4681 | 5 | soma | 2 | 0 | 0 | 2_suspects, low_reliability, no_behavior |

All 30 runs have confidence='auto' (fully automatic, no manual review).
Rating = Daria's quality rating (run_quality.csv: very good / good / questionable; '-' = unrated). A rating never removes a run; see the Sensitivity Analysis section.

---
## Global BH-FDR Correction (Cohort-Level Tests Only)

Only cohort-level p-values (one per test) are included in the global FDR.
Per-cell surrogate p-values and pseudo-replicated (pooled) p-values are excluded.

| Test | p | q (BH) |
|------|---|--------|
| test02_coupling | 7.63e-06 | 3.81e-05 **|
| test02_coupling | 1.41e-05 | 4.71e-05 **|
| test03_independence | 0.0128 | 0.0213 **|
| test03_independence | 3.05e-05 | 7.63e-05 **|
| test04_distance | 2.08e-18 | 2.08e-17 **|
| test05_event_order | 0.000488 | 0.000977 **|
| test07_behavior | 0.455 | 0.569 |
| test08_expression | 0.812 | 0.881 |
| test08_expression | 0.881 | 0.881 |
| test08_expression | 0.0566 | 0.0808 |

10 tests, 6 significant at q<0.05

---
## Sensitivity Analysis
All runs are automatic (confidence='auto'), so no high vs low confidence split is possible.
The full cohort results above are the main analysis; the quality-rating checks below sit next to it.

## Sensitivity to Daria's quality ratings (run_quality.csv)

Ratings among the 17 cells: very good 0, good 0, questionable 0, unrated 17. Ratings are human judgments made after seeing the data, so these are robustness checks next to the main result, not new tests. Each column is BH-FDR corrected on its own; test 3 reuses the per-cell surrogate results (no new surrogates).

- **main**: 17 cells, 7 mice
- **(a) without 'questionable' (unrated kept)**: 17 cells, 7 mice - identical to main (no cell removed)
- **(b) 'very good' only**: 0 cells, 0 mice - not computable (needs >= 2 mice)

| Test | main p (q) | (a) no questionable p (q) | (b) very good only p (q) |
|------|-----------|---------------------------|--------------------------|
| test02_coupling | 7.63e-06 (3.81e-05) | 7.63e-06 (3.81e-05) | n/a |
| test02_coupling #2 | 1.41e-05 (4.71e-05) | 1.41e-05 (4.71e-05) | n/a |
| test03_independence | 0.0128 (0.0213) | 0.0128 (0.0213) | n/a |
| test03_independence #2 | 3.05e-05 (7.63e-05) | 3.05e-05 (7.63e-05) | n/a |
| test04_distance | 2.08e-18 (2.08e-17) | 2.08e-18 (2.08e-17) | n/a |
| test05_event_order | 0.000488 (0.000977) | 0.000488 (0.000977) | n/a |
| test07_behavior | 0.455 (0.569) | 0.455 (0.569) | n/a |
| test08_expression | 0.812 (0.881) | 0.812 (0.881) | n/a |
| test08_expression #2 | 0.881 (0.881) | 0.881 (0.881) | n/a |
| test08_expression #3 | 0.0566 (0.0808) | 0.0566 (0.0808) | n/a |

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

cohort_stats.py: 17 runs, r(soma,branch) mean 0.53, r(soma,trunk) mean 0.76, Wilcoxon p=7.63e-06, 41 set aside
paper_stats.py:  17 runs, r(soma,branch) mean 0.5345, r(soma,trunk) mean 0.7574, Wilcoxon p=7.63e-06

Numbers agree within rounding tolerance.

