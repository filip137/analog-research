# OM closed-loop learning-rate sweep: execution complete

The loulou-only campaign has passed four native smoke studies and their 96
exact physical-state replays. Full development for both datasets is collected
and artifact-verified: 192 five-epoch trajectories on development arrays
211001 and 211002. The development-only selection receipt is frozen.
Confirmation for both datasets is collected and verified. Both compare
uncapped LR 1e-4 with selected LR 1e-3 on three exact historical final-array
deployments. All 24 native runs completed with no failed attempts or active
runs: 336 full-cohort trajectories plus 96 smoke trajectories, and 432 exact
epoch-five physical-state replays. All 12 prepared study coverage/artifact
audits are ready for review. Scientific review and manifest finalization are
pending the user's interpretation and next-test decisions.

The following **CIFAR-10 test results** are means across three arrays, using
standard HWA and joint recovery. Each entry is accuracy (%) / teacher KL.

| Analog convolutions | Faults | New closed-loop 1e-4 | Selected closed-loop 1e-3 | Historical open-loop 1e-4 |
|---:|---|---:|---:|---:|
| 4 | No injected faults | 93.35 / 0.0040 | 93.39 / 0.0097 | 92.57 / 0.0591 |
| 4 | 3% stuck high | 92.16 / 0.1006 | 92.95 / 0.0544 | 92.09 / 0.0864 |
| 4 | 5% stuck high | 89.54 / 0.3593 | 91.71 / 0.1102 | 90.21 / 0.2623 |
| 8 | No injected faults | 93.15 / 0.0232 | 93.16 / 0.0431 | 89.47 / 0.2128 |
| 8 | 3% stuck high | 75.78 / 0.6696 | 84.03 / 0.3685 | 81.25 / 0.4690 |
| 8 | 5% stuck high | 39.21 / 44.4673 | 40.03 / 1.5730 | 42.74 / 25.7589 |

The corresponding **CIFAR-100 test means** are:

| Analog convolutions | Faults | New closed-loop 1e-4 | Selected closed-loop 1e-3 | Historical open-loop 1e-4 |
|---:|---|---:|---:|---:|
| 4 | No injected faults | 69.39 / 0.0742 | 69.78 / 0.0463 | 60.65 / 0.7004 |
| 4 | 3% stuck high | 60.08 / 0.7146 | 66.99 / 0.3533 | 55.18 / 0.9630 |
| 4 | 5% stuck high | 37.39 / 1.8542 | 50.84 / 1.1682 | 40.92 / 1.6607 |
| 8 | No injected faults | 69.39 / 0.1529 | 69.32 / 0.1371 | 52.45 / 1.1709 |
| 8 | 3% stuck high | 30.60 / 2.1895 | 41.52 / 1.6741 | 31.66 / 2.1647 |
| 8 | 5% stuck high | 2.33 / 6.0211 | 4.72 / 3.8201 | 2.14 / 9.5659 |

At eight convolutions and 3% stuck high, higher closed-loop LR improves both
test metrics on every array in both datasets versus both the fresh 1e-4
baseline and historical open-loop. Array standard deviations for selected
accuracy are 1.31 points on CIFAR-10 and 9.52 points on CIFAR-100, so physical
array variation remains substantial. At 5%, KL improves substantially but
eight-convolution test accuracy remains poor; CIFAR-10 mean accuracy is below
open-loop and CIFAR-100 remains only 4.72%. On nominal CIFAR-10 arrays, higher
LR worsens teacher KL despite nearly unchanged top-1 accuracy.

The digital-source arms are also retained in the full CSV tables. For eight
convolutions and 3% stuck high, their selected endpoints are 85.17% / 0.3329
on CIFAR-10 and 34.29% / 1.9731 on CIFAR-100. Every fault-case mean improves
in both accuracy and KL versus the corresponding new 1e-4 baseline, across
both digital and standard-HWA sources. This focused sweep did not tune the
corruption-aware HWA sources or test other fault kinds.

The larger LR also uses more writes. In the eight-convolution, 3% stuck-high
standard-HWA case, mean recovery pulses per cell rise from 0.204 to 2.956
(about 14.5 times), and the fraction of cells receiving a pulse rises from
2.66% to 29.18%. Epoch and image budgets are matched; physical pulse expenditure
is a measured outcome and differs between methods. This does not establish
an energy advantage. The corresponding CIFAR-100 case rises from 0.619 to
7.189 pulses per cell, with pulsed-cell coverage increasing 6.60% to 37.87%.

Development selection scores were:

| Dataset | Analog convolutions | Mean development KL, 1e-4 | 3e-4 | 1e-3 (selected) |
|---|---:|---:|---:|---:|
| CIFAR-10 | 4 | 0.038285 | 0.015224 | 0.015109 |
| CIFAR-10 | 8 | 19.441313 | 73.834440 | 0.654483 |
| CIFAR-100 | 4 | 1.044809 | 0.776622 | 0.598977 |
| CIFAR-100 | 8 | 3.054210 | 2.890776 | 2.113260 |

These means include both sources, all three fault cases and both development
arrays. CIFAR-10/four-convolution candidates 3e-4 and 1e-3 are nearly tied;
the fixed numerical minimum rule selected 1e-3 without a post-hoc margin.

The next table contains **development results, not test accuracy**. Selection uses the
existing 1000-image development cohort and two development arrays. The
following CIFAR-10 means use the standard-HWA source, eight analog
convolutions plus classifier, and joint recovery after five full epochs:

| Faults | Weight LR | Development accuracy (%) | Teacher KL | Cells receiving a pulse (%) |
|---|---:|---:|---:|---:|
| No injected faults | 1e-4 | 100.00 | 0.00075 | 0.00 |
| No injected faults | 3e-4 | 100.00 | 0.00070 | 1.00 |
| No injected faults | 1e-3 | 100.00 | 0.00201 | 28.83 |
| 3% stuck high | 1e-4 | 80.75 | 0.55536 | 2.92 |
| 3% stuck high | 3e-4 | 86.20 | 0.38622 | 10.59 |
| 3% stuck high | 1e-3 | 87.05 | 0.36609 | 35.63 |
| 5% stuck high | 1e-4 | 39.90 | 67.33409 | 1.28 |
| 5% stuck high | 3e-4 | 38.70 | 421.48028 | 7.48 |
| 5% stuck high | 1e-3 | 40.60 | 1.65711 | 83.46 |

Higher LR increases pulse coverage and improves the 3% fault case, supporting
the hypothesis that the old LR is small relative to the unchanged 0.04745
verify tolerance. The severe 5% case is nonmonotonic and varies across arrays;
the 3e-4 candidate includes an accuracy collapse, retained as scientific evidence.
The confirmed test results above must be distinguished from this selection
evidence. No inherent controller
superiority follows from these particular settings.

The selection rule remains the predeclared mean raw epoch-five development
KL across both sources, three cases and two arrays, separately by dataset
and analog depth. Severe cases can dominate that objective. All candidate
curves and clean-array regressions are retained. Calibration LR, tolerance,
one-pulse-per-update constraint and five-epoch budget are fixed; cumulative
recovery pulse caps are absent. HWA fits are reused without refitting.

The primary LR contrast uses two new uncapped runs on the same loulou
software/backend. Historical V2 confirmation arrays also used nom-cool-1
and Akib with ATen and older PyTorch builds. Although exact physical P0 and
initial test metrics are verified, small calibration endpoint differences
occur even with zero recovery pulses. Historical comparisons therefore
include this numerical-execution caveat; their differences cannot all be
attributed to removing the old pulse cap.

Protocol: [cifar_om_closed_loop_lr.md](cifar_om_closed_loop_lr.md).
Live state: `artifacts/cifar_om_closed_loop_lr_sources/full-progress.json` and
`execution-v1/watchdog-latest.json` in that directory. Collected raw runs:
`results/cifar-om-closedloop-lr-collected/v1/results/`. Every candidate/epoch,
array means and figures: `results/cifar-om-closedloop-lr-analysis/`.
The completed numerical campaign is ready for scientific review. Per
`/home/filip/.codex/skills/ebl-study-closeout/SKILL.md`, the user's scientific
interpretation and next tests are required before creating review records
or finalizing managed manifest entries. No such interpretation is inferred
from run completion.
