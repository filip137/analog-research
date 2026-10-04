# OM open-loop recovery: execution complete

Completion update, 2026-09-20 (Europe/Paris).

Both datasets are complete: each has 12/12 full native array runs, 2,592 controls, 1,944 exact endpoint replays, and four artifact-verified study receipts. The final pipeline finished on 2026-09-19 at 23:39 CEST. The current watchdog reports zero active campaign runs; the collected studies have no failed or invalid runs. Scientific review and manifest finalization remain pending.

## Completed CIFAR-100 results

The following are three-array test means after five full recovery epochs,
using the standard-HWA source and the exact saved V2 deployment. Entries
are accuracy (%) / KL(original digital teacher || student), nats at T=1.
Joint open-loop recovery includes the same digital calibration parameters.

| Analog convolutions + classifier | Faults | Calibration | Open-loop + calibration |
|---:|---|---:|---:|
| 4 | No injected faults | 69.27 / 0.0877 | 60.65 / 0.7004 |
| 4 | 3% stuck high | 53.50 / 1.0010 | 55.18 / 0.9630 |
| 4 | 5% stuck high | 33.31 / 2.1034 | 40.92 / 1.6607 |
| 8 | No injected faults | 69.09 / 0.1690 | 52.45 / 1.1709 |
| 8 | 3% stuck high | 22.40 / 2.6423 | 31.66 / 2.1647 |
| 8 | 5% stuck high | 2.67 / 6.0819 | 2.14 / 9.5659 |

With eight convolutions, the 3% stuck-high mean improves over calibration,
but the 5% case remains near chance and worsens in both metrics. Nominal
arrays regress at both depths under this inherited open-loop learning rate.
Fresh RESET open-loop programming remains a separate initialization arm:
even with no injected faults, joint recovery reaches only 7.02% / 3.6727
with four convolutions and 0.99% / 4.6413 with eight. These are measured
outcomes, not operational failures.

Complete CIFAR-100 tables are in
[the analysis directory](../results/cifar-om-openloop-cifar100-analysis).
The completed [closed-loop LR sweep](cifar_om_closed_loop_lr_results.md)
selects its learning rate on development arrays and confirms it on saved
final deployments. Its primary contrast uses a fresh matched-backend 1e-4
baseline; historical comparisons retain the documented execution caveat.

## CIFAR-10 results

All numbers below are means across three physical-array identities, evaluated on 10,000 test images after five full 45,000-image recovery epochs. Entries are accuracy (%) / KL(original digital teacher || student), nats at T=1. Every open-loop update retains at most one pulse per cell, with no cumulative cap. The largest observed recovery count in the completed CIFAR-10 sweep was 10 pulses per cell; all recovery verification-read counts were zero.

## Recovery from the same saved deployment

The following comparisons reuse the exact V2 deployment state and standard HWA weights. Both pulse-training columns include the same digital calibration parameters. The historical closed-loop reference retains its 640-pulse cumulative cap and one-pulse update limit.

| Analog convolutions + classifier | Fault | Calibration | Historical closed-loop + calibration | Open-loop + calibration |
|---|---|---|---|---|
| 4 | No injected fault | 93.34 / 0.004 | 93.35 / 0.004 | 92.57 / 0.059 |
| 4 | 2% stuck high | 92.69 / 0.067 | 92.76 / 0.062 | 92.47 / 0.073 |
| 4 | 3% stuck high | 91.86 / 0.109 | 92.16 / 0.101 | 92.09 / 0.086 |
| 4 | 5% stuck high | 88.23 / 0.406 | 89.52 / 0.359 | 90.21 / 0.262 |
| 8 | No injected fault | 93.14 / 0.023 | 93.14 / 0.023 | 89.47 / 0.213 |
| 8 | 2% stuck high | 89.26 / 0.216 | 90.18 / 0.194 | 88.72 / 0.227 |
| 8 | 3% stuck high | 70.48 / 0.832 | 75.69 / 0.672 | 81.25 / 0.469 |
| 8 | 5% stuck high | 24.85 / 50.006 | 38.09 / 47.498 | 42.74 / 25.759 |

At eight analog convolutions and 3% stuck-high faults, open-loop recovery improves both accuracy and KL relative to calibration and historical closed-loop recovery on all three arrays. At 5%, average improvement is larger relative to calibration, but absolute performance remains poor and array variation is substantial. With no injected faults, open-loop recovery worsens accuracy and KL on all three arrays at both depths. These are results for the inherited learning rate, not an optimized open-loop recipe.

## Stronger HWA controls

| Analog convolutions + classifier | Source / 5% fault | Calibration | Historical closed-loop + calibration | Open-loop + calibration |
|---|---|---|---|---|
| 4 | Noise HWA / stuck high | 92.45 / 0.089 | 92.45 / 0.089 | 92.59 / 0.077 |
| 4 | Corruption-aware HWA / stuck high | 92.64 / 0.076 | 92.66 / 0.075 | 92.52 / 0.077 |
| 4 | Corruption-aware HWA / stuck low | 92.71 / 0.069 | 92.73 / 0.068 | 92.34 / 0.078 |
| 4 | Corruption-aware HWA / stuck random | 92.92 / 0.051 | 92.90 / 0.050 | 92.85 / 0.054 |
| 8 | Noise HWA / stuck high | 86.80 / 0.364 | 86.52 / 0.349 | 87.95 / 0.276 |
| 8 | Corruption-aware HWA / stuck high | 84.19 / 0.368 | 87.76 / 0.257 | 85.98 / 0.316 |
| 8 | Corruption-aware HWA / stuck low | 89.60 / 0.204 | 89.97 / 0.195 | 87.83 / 0.266 |
| 8 | Corruption-aware HWA / stuck random | 91.03 / 0.155 | 91.18 / 0.150 | 89.83 / 0.208 |

The larger gains against standard HWA do not establish a general advantage against corruption-aware HWA. For example, four-convolution stuck-high CDT already reaches 92.64% with calibration; open-loop recovery reaches 92.52%. Source convergence flags are retained in the raw report.

## Fresh RESET open-loop programming

These runs program the same pretrained/HWA targets using the public nominal soft-bounds inverse, then adapt on that array. HWA was inherited from the earlier programming model and was not refitted for this programmer. This changes initialization as well as using open-loop recovery; it is not a matched-P0 recovery-only comparison.

| Analog convolutions + classifier | Standard HWA, no injected faults | Accuracy / KL |
|---|---|---|
| 4 | Immediately after open-loop programming | 13.04 / 322.077 |
| 4 | After calibration | 66.30 / 0.972 |
| 4 | After open-loop recovery + calibration | 85.08 / 0.364 |
| 8 | Immediately after open-loop programming | 9.90 / 119511.399 |
| 8 | After calibration | 26.03 / 54.994 |
| 8 | After open-loop recovery + calibration | 42.44 / 23.654 |

The nominal RESET programmer creates a much worse deployment. Five recovery epochs help substantially, but do not restore the accuracy of the saved-deployment arm, especially with eight analog convolutions. This result is specific to this nominal programmer and reused HWA weights.

Complete CIFAR-10 tables, including every 1/2/3/5% fault case, source, epoch, array spread and pulse cost: [analysis directory](../results/cifar-om-openloop-cifar10-analysis). Native handles and current status: [campaign records](../artifacts/cifar_om_open_loop_sources/README.md).

## Why the two recovery rules differ

Read-only endpoint diagnostics were performed on three arrays for four
selected CIFAR-10 conditions. Both methods receive digital Adam updates from
teacher KL. Closed-loop accumulates those updates into a digital target and
issues a pulse only when the apparent weight differs from that target by more
than 0.04745 (half the nominal OM step of 0.0949). Open-loop instead draws a
pulse with probability `min(abs(Adam update)/0.0949, 1)`, with no target-error
tolerance. These are different update controllers, not simply the same pulse
schedule with feedback enabled or disabled.

For eight analog convolutions, standard HWA, and 3% stuck-high faults:

| Five-epoch checkpoint diagnostic, three-array mean | Closed-loop | Open-loop |
|---|---:|---:|
| Cells receiving at least one recovery pulse | 2.66% | 16.49% |
| Recovery pulses per cell | 0.201 | 0.196 |
| Mean absolute change in apparent normalized weight | 0.00156 | 0.03242 |

The closed-loop digital target moved by only 0.01285 on average, below its
0.04745 tolerance. The recorded pulses are concentrated on fewer cells; the
open-loop rule distributes a similar total pulse count across more cells and
changes the realized weights more. This supports tolerance-suppressed
adaptation as an explanation of the severe-fault result, but is not a causal
ablation. Almost no closed-loop cells reach the 640-pulse cap in this case.

On nominal eight-convolution arrays, closed-loop leaves essentially every
cell unwritten, whereas open-loop writes 53.89% of cells at least once. That
is consistent with preserving a good deployment under closed-loop while
open-loop perturbs it and lowers accuracy. Apparent write noise is redrawn
after a pulse, not on each repeated verify read; it adds uncertainty to each
physical update. Its absolute preset scale is 0.13393 in normalized weight
coordinates.

These observations do not establish an inherent advantage of open-loop
training. A fair controller comparison would also examine the closed-loop
tolerance and select each method's learning rate on development arrays,
while retaining matched initial states and the one-pulse update limit.
The checkpoint paths, hashes, and per-array distributions are saved in
[writer_diagnostics.json](../results/cifar-om-openloop-cifar10-analysis/writer_diagnostics.json).
