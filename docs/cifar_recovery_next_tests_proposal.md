# Proposed next CIFAR recovery tests

Status: proposal, 20 September 2026. This is a plan for new studies, not a
change to an existing experiment contract or a completed scientific review.

The central question is whether adapting the deployed physical weights adds
useful accuracy and teacher fidelity after a strong offline source and an
adequate digital correction. Keep CIFAR-10 and CIFAR-100, each fault type,
and PCM and OM separate in every analysis.

## Why these tests

All numbers below are test accuracy (%) / teacher KL in nats at T=1,
three-array means. They use stronger-noise HWA, eight analog convolutions
plus classifier, 5% stuck-high faults and the historical five-epoch budget.

| Dataset / device | Calibration only | Weights only | Joint weights and calibration |
|---|---:|---:|---:|
| CIFAR-100 / PCM | 48.16 / 1.3738 | 57.98 / 0.9685 | 59.65 / 0.8775 |
| CIFAR-100 / OM | 42.18 / 1.5755 | 15.04 / 2.9469 | 46.76 / 1.4063 |
| CIFAR-10 / OM | 86.80 / 0.3643 | 69.29 / 1.1083 | 86.52 / 0.3485 |

The weight-only arms started from the deployed HWA model without a preceding
calibration stage. These results therefore do not answer whether weight-only
updates work well *after* the digital parameters have been calibrated.
Calibration currently fits BN affine parameters, output gains and classifier
bias; it keeps BN running means and variances fixed.

The completed OM learning-rate study tuned digital and normal-HWA sources,
not stronger-noise HWA or CDT. In the offline fits, 31 of 40 selected HWA
sources retain a late-convergence flag. These are separate reasons to avoid
treating the current best pipelines as fully optimized.

## Initial case panel

Use stronger-noise HWA first, eight analog convolutions plus classifier.

- CIFAR-100: PCM and OM, nominal plus 5% stuck-low, random-stuck and
  stuck-high. This gives eight dataset/device/fault conditions.
- CIFAR-10: PCM 5% stuck-low and OM 5% stuck-high, with a nominal reference
  for each device. These give a small-gain reference and a difficult case
  where joint recovery currently has an accuracy/KL trade-off.
- Use two development arrays for this exploratory screen: 12 conditions,
  24 paired deployments per arm. Source checkpoints and exact deployment
  states are shared across methods within each condition.
- Add 3% and four-convolution deployments for follow-up comparisons that
  remain informative on development data. Confirm the selected comparisons
  on fresh arrays after freezing the settings; the old three final arrays
  are historical evidence, not a new independent confirmation set.

## Priority 1: identify the minimum useful digital calibration

The hypothesis is that correcting activation statistics or a small subset
of digital parameters can remove much of the deployment loss without
physical-weight adaptation. Test:

| Arm | Trainable or recalculated quantities | Physical weights |
|---|---|---|
| BN-statistics refresh | Re-estimated suffix BN running means and variances; no gradient training | Fixed |
| BN affine only | Suffix BN scale and offset | Fixed |
| Gains and classifier bias | Analog-matrix output gains and classifier bias | Fixed |
| Existing full calibration | BN affine, output gains and classifier bias | Fixed |
| BN-statistics refresh + full calibration | Refreshed statistics followed by the existing trainable digital parameters | Fixed |

All gradient-based digital arms use the same five-epoch image budget,
teacher-KL objective, data order and dataset-specific calibration LR.
The statistics-only arm receives a separately recorded training-data pass;
do not describe its cost as equal to five gradient epochs. Collect
deployment activation means/variances and logit scales before and after.
The digital prefix remains fixed. The BN-statistics intervention is an
explicit new diagnostic; ordinary historical arms retain frozen statistics.

Readout: does a simpler digital correction approach the existing joint
recovery endpoint? Does the apparent need for physical updates survive the
stronger digital-only control? Report accuracy/KL separately when they
disagree. A one-pass statistics correction is a proposed experiment, not an
assumed explanation of the present results.

## Priority 2: calibrate first, then isolate the effect of weight updates

Compare the following schedules using ten total full-data passes each.
Save exact physical states, digital parameters, optimizer states and RNGs
at the branch point. Arms A, B, C and E share the same five-epoch calibration
prefix, including the same deployed array.

| Arm | Passes 1–5 | Passes 6–10 |
|---|---|---|
| A | Digital calibration | Continue digital calibration |
| B | Digital calibration | Freeze calibration parameters; update physical weights only |
| C | Digital calibration | Joint weights and calibration |
| D | Joint weights and calibration | Continue joint weights and calibration |
| E | Digital calibration | Frozen-target rewrite + digital calibration |

The primary contrasts are B versus A, C versus A, and C versus E. They test
the contribution of physical-weight adaptation after an identical digital
calibration prefix. B versus C tests whether calibration must keep adapting
as weights change. C versus D compares schedules at equal total image
exposure; it does not give both schedules the same number of weight updates.
Compare with the saved epoch-five endpoints as learning curves, not as
equal-budget ten-pass controls.

Start with the historical weight LR, fixed calibration LRs and unchanged
device laws. Keep OM tolerance, one-pulse-per-update rule and cumulative
pulse cap identical across these schedule arms; record any cap saturation.
PCM retains the complete Gaussian endpoint reprogramming abstraction.
The failed devices remain permanently fixed. If the selected digital
calibration includes refreshed BN statistics from Priority 1, include a
matched schedule block with that same initialization for every arm.

## Priority 3: tune recovery on the stronger sources

Apply a separate development-only weight-LR grid of 1e-4, 3e-4 and 1e-3 to
stronger-noise HWA and the corresponding fault-specific CDT sources. Select
separately for PCM and OM; the earlier OM optimum on normal HWA is not an
established optimum for these sources or for PCM.

First compare the LRs at a common five-epoch budget and common calibration
settings. Then extend the selected candidates with checkpoints at 5, 10 and
20 full epochs, alongside calibration-only and frozen-target rewrite
controls that receive the same total data exposure. Freeze the source,
schedule, LR, epoch-selection rule and evaluation cases before fresh-array
confirmation. Nominal deployments remain in the development panel to expose
regressions.

Hold OM tolerance and the pulse-cap policy fixed during the primary LR
comparison. Any tolerance or cap study is a separate matched intervention.
Report the pulse/verify-read expenditure and saturation, rather than
attributing improvements from a simultaneous policy change to LR alone.
Do not infer incremental PCM pulse behavior from the endpoint model.

Readout: does recovery after stronger-noise HWA or CDT saturate at five
epochs, and what additional accuracy/KL is obtained per data pass and write?

## Priority 4: strengthen the HWA baseline and confirm the conclusion

Before making a general necessity claim, compare stronger-noise HWA and CDT
under a common offline budget and a common deployment-oriented development
criterion. Their historical selection criteria differ, so the current
cross-source comparison does not isolate fault injection during training.

- Use the same teacher, architecture, optimizer search budget, noise
  strength, nominal/fault development cases and calibration budget for each
  comparison; explicit stuck-fault injection is the training difference
  between stronger-noise HWA and its matched CDT comparator.
- For the primary deployment-oriented selection, measure development KL
  after an identical digital-calibration budget. Retain uncalibrated scores
  and the historical source choices as references. Source selection must
  not use confirmation arrays or test labels.
- Evaluate the selected recipes at 60 and 120 offline epochs with a
  predeclared development plateau check. If the improvement persists at the
  cap, keep the convergence limitation visible rather than declaring it
  solved by a larger epoch count.
- Repeat the retained HWA recipes with three independent training seeds.
  Confirm the selected calibration/recovery comparisons on five fresh
  arrays per seed; report seed and array variation and paired gains against
  both calibration-only and rewrite controls.

The question is whether the recovery advantage survives a stronger,
more reproducible HWA baseline, not whether any one weak HWA model can be
improved. Final reports should distinguish: target met without recovery;
target met only with a tested recovery method; and target unmet by all
tested methods. A proposed practical accuracy target is within one point of
the clean teacher, with a two-point sensitivity analysis and teacher KL
reported independently. These practical bands are proposed reporting
criteria, not evidence of universal physical necessity.

## Follow-ups after the main comparison

The [partial fine-tuning exploration](cifar_partial_finetuning_recovery.md)
adds classifier-only recovery, selected-layer/weight updates and separate
digital low-rank corrections to Priority 2's calibrated branch point. It
includes checked parameter budgets for both datasets and both deployment
depths, and distinguishes fewer trainable weights from fewer physical writes.
Start with a small matched method screen before expanding ranks or masks.

Two useful deployment tests become meaningful once the source and recovery
recipe are fixed:

1. Repeat calibration/recovery with 500, 5,000 and 45,000 distinct training
   images, reporting both distinct images and total presentations. This
   measures the data requirement of the gain.
2. Transfer the learned digital target from array A to independent arrays,
   reprogramming through each target array's device model and allowing the
   same local calibration budget. Compare with the unrecovered HWA source
   and local recovery on each target array. This tests how array-specific
   the learned correction is. It does not by itself establish that the
   optimizer must physically reside on the chip.

## Common reporting and execution requirements

Every result reports accuracy and teacher KL together, paired array
differences, nominal performance, image presentations, digital-calibration
updates, and physical-write costs. OM reports pulse counts, pulse coverage
and verify reads; PCM reports full-array reprogram counts. These quantities
do not establish a hardware energy measurement.

New runs require a new native EBL study contract and exact source/input
identities before execution, with saved-state replay and the existing
monitoring/coverage workflow. This proposal is not a launcher or an approved
modification of historical studies.

Evidence: [fault sweep](cifar_fault_rate_sweep.md),
[OM LR protocol](cifar_om_closed_loop_lr.md),
[OM LR results](cifar_om_closed_loop_lr_results.md),
[CIFAR-10 analysis](../artifacts/cifar_fault_regime_analysis_20260920/cifar10_analysis.html),
[CIFAR-100 analysis](../artifacts/cifar_fault_regime_analysis_20260920/cifar100_analysis.html),
and `results/cifar-sweep-analysis/summary.csv`.
