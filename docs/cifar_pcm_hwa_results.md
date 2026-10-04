# CIFAR: HWA/CDT followed by fixed-array recovery

The subsequent [one-to-twenty-pass recovery comparison](cifar_pcm_recovery_epochs_results.md)
is complete for both datasets, including matched calibration/rewrite controls.
The measurements below remain the original one-pass study on its original arrays.

Numerical comparison completed and collected on 2026-09-17. Both native
studies passed full artifact-hash verification and report ready_for_review.
Scientific review and managed-manifest finalization await the user's
interpretation and next-test decisions.

The clearest additional recovery benefit is **CIFAR-100 at 1% Gmax failures,
using weight updates together with calibration**. It improves both teacher KL
and accuracy on all three arrays beyond every tested HWA/CDT control without
weight updates. Most of the total gain comes from stronger-noise HWA and
calibration. Weight-only recovery does not beat calibration on average in
this condition. This is a fixed-budget, model-based result; several selected
HWA/CDT fits still require convergence review.

## What ran

The [frozen protocol](cifar_pcm_hwa_comparison.md) compares standard HWA,
noise-strength-selected HWA and corrupt-device training (CDT), inspired by
[Li et al. (2023)](https://doi.org/10.1063/5.0131797). CDT resamples physical-bank
faults each minibatch. Final deployment and recovery keep the fault map fixed.
The [method audit](cifar_pcm_paper_comparison.md) explains differences from the
paper's logical training corruption, full-network mapping and retention model.

- Full pretrained CIFAR ResNet-32, with its final four convolutions and
  classifier mapped to nine logical tiles of capacity 512 by 512. The prefix
  and BN running statistics stay frozen. The two networks have 148,096 and
  153,856 mapped weights, each represented by two physical PCM devices.
- Gaussian programming endpoints at 25 microSiemens; open, random and Gmax
  failures at 100, 1,000 and 10,000 ppm per physical device, plus nominal
  conditions. One percent per device gives approximately 1.99% of logical
  differential pairs at least one failed device. No additional pulse/P&V loop, read noise, drift or
  peripheral quantization is included.
- HWA uses all 45,000 adaptation-training images with fresh crops/flips;
  source selection uses 1,000 held-out adaptation images and independent
  development arrays. Each dataset attempts 20 candidates: 19 complete and
  SGD at 0.5 is rejected as nonfinite. Completed fits total 750 training epochs
  for CIFAR-10 and 870 for CIFAR-100, across the candidates.
- Recovery makes one pass over 5,000 images, using teacher KL. Calibration and
  weight learning rates are selected on separate development arrays. Every
  writing control makes 79 complete Gaussian endpoint reprogramming requests.
  Gradients and Adam are digital; this is an abstraction of array-specific
  recovery, not a measured incremental-pulse or fully analog backward pass.
- Three fresh arrays per condition and all 10,000 test images. Each dataset
  has 630 measurements: 42 source/case combinations, five controls and three
  arrays. The combined comparison has 1,260 measurements, not 1,260 arrays.
- Primary metric: mean KL(p_teacher || p_student), in nats at temperature 1,
  against the original digital teacher. Accuracy is reported alongside it.

## The decisive condition: CIFAR-100, 1% Gmax failures

Mean plus/minus sample SD over three matched fresh arrays:

| Source and control | Teacher KL | Accuracy (%) |
|---|---:|---:|
| Original digital weights, deployed | 3.2103 ± 0.4851 | 33.05 ± 4.10 |
| Standard HWA, deployed | 2.9052 ± 0.4177 | 36.08 ± 3.65 |
| Noise-selected HWA (5x), deployed | 0.9292 ± 0.0190 | 56.94 ± 0.55 |
| HWA + CDT, deployed | 0.8761 ± 0.0278 | 57.60 ± 0.71 |
| CDT + calibration | 0.5431 ± 0.0512 | 63.11 ± 0.78 |
| CDT + unchanged-target rewriting and calibration | 0.5616 ± 0.0412 | 62.74 ± 0.47 |
| CDT + weight recovery | 0.5443 ± 0.0274 | 62.90 ± 0.40 |
| CDT + weight recovery and calibration | **0.5025 ± 0.0312** | **64.02 ± 0.46** |

Adding weight updates to CDT plus calibration reduces KL by 0.04067 nats
(7.49%) and adds 0.91 percentage points of accuracy. The paired sample SDs
are 0.02288 nats and 0.35 percentage points. Both metrics improve on all three
arrays, including comparisons with calibrated noise-HWA and matched
unchanged-target rewriting. Weight-only updates provide no mean advantage over
calibration. Recovery remains partial: 64.02% is below the original digital
teacher's 70.16% accuracy.

## Where the added benefit is weak or absent

For CIFAR-10, calibration is the strong control across this screen. At 1%
Gmax, noise-HWA plus weight-only recovery has a very small mean advantage over
noise-HWA plus calibration: KL 0.03756 to 0.03724 (0.84%). The paired reduction
is 0.000314 ± 0.003482 nats, with KL improving on two arrays and accuracy on
only one. Mean accuracy changes by just +0.01 percentage points. No other
CIFAR-10 condition has a recovery arm with lower mean KL than the best tested
HWA/CDT control without weight updates. This does not establish a consistent
recovery advantage on CIFAR-10.

For CIFAR-100 at 1% random faults, CDT plus joint recovery has a small,
inconsistent advantage over CDT plus calibration: KL 0.28028 to 0.27418 and
accuracy 66.87% to 67.01%. Each metric improves on two of three arrays. Open
faults and the 100/1,000 ppm conditions have no recovery arm with lower mean
KL than the best tested HWA/CDT non-weight control. The full contrast CSV
retains all source/control comparisons, including losses.

Examples of nominal conditions also favor calibration. With standard HWA,
CIFAR-10 KL is 0.003700 after deployment, 0.001863 after calibration and
0.005893 after joint recovery. CIFAR-100 values are 0.087956, 0.038711 and
0.086941. In this endpoint model, rewriting resamples healthy-device write
errors; calibration adapts to the existing programmed state. Matched rewrite
controls expose this effect rather than attributing every change to learning.

CDT can also sacrifice fidelity on healthy hardware. Its Gmax source has clean
KL 0.03085 and 93.38% accuracy on CIFAR-10, versus the original teacher's
93.53%; on CIFAR-100 it has clean KL 0.16956 and 69.39% accuracy, versus 70.16%.
Thus accuracy alone can conceal substantial changes in the teacher's output
distribution. Apparent recovery of an over-regularized CDT source at low fault
rates should be compared with calibrated standard HWA as well.

## Selection and limits

All selected fits use Adam with teacher KL and learning rate 1e-4.

| Dataset | Source | Training noise | CDT device probability | Selected epoch / run | Late flag |
|---|---|---:|---:|---:|---|
| CIFAR-10 | Standard HWA | 1x | 0 | 15 / 30 | No |
| CIFAR-10 | Noise HWA | 2x | 0 | 50 / 60 | Yes |
| CIFAR-10 | CDT open | 1x | 0.001 | 15 / 30 | No |
| CIFAR-10 | CDT random | 2x | 0.003 | 45 / 60 | No |
| CIFAR-10 | CDT Gmax | 5x | 0.003 | 50 / 60 | Yes |
| CIFAR-100 | Standard HWA | 1x | 0 | 25 / 60 | No |
| CIFAR-100 | Noise HWA | 5x | 0 | 60 / 60 | Yes |
| CIFAR-100 | CDT open | 1x | 0.001 | 45 / 60 | No |
| CIFAR-100 | CDT random | 2x | 0.003 | 60 / 60 | Yes |
| CIFAR-100 | CDT Gmax | 5x | 0.001 | 60 / 60 | Yes |

Recovery rates are 1e-4 for weights on both datasets; calibration uses 3e-4
on CIFAR-10 and 1e-3 on CIFAR-100. Rates are global per dataset, selected before
final deployment. Noise-HWA is selected globally across the declared fault
cases; CDT is selected separately for each fault type. These are comparisons
with the tested development-selected sources, not with every possible
per-condition HWA configuration.

The CIFAR-100 noise/Gmax/random fits selected their final epoch and continued
to improve late. Consequently, the data establish an incremental benefit over
these fixed-budget baselines, not a necessity beyond fully converged HWA/CDT.
There is one HWA/data-order seed per dataset; the three-array spread does not
measure training-seed uncertainty or provide a significance test. Development
images were held out from our adaptation, but were part of the public model's
original pretraining data. The conclusions apply to the converted suffix and
the stated immediate-endpoint failure model.

The most useful unresolved checks are longer or independently tuned HWA/CDT
for severe Gmax faults, additional training seeds and arrays, and an explicit
incremental device-update model if the result is to support physical on-chip
training claims. These are proposed follow-ups, not completed evidence or a
substitute for the user's scientific review.

## Artifacts and verification

- [Full numerical report](../results/cifar-hwa-cdt-analysis/report.md),
  [summary CSV](../results/cifar-hwa-cdt-analysis/summary.csv),
  [per-array data](../results/cifar-hwa-cdt-analysis/per_array.csv) and
  [cross-source paired contrasts](../results/cifar-hwa-cdt-analysis/cross_source_comparisons.csv).
- [CIFAR-10 KL curves](../results/cifar-hwa-cdt-analysis/cifar10_hwa_cdt_kl.png)
  and [CIFAR-100 KL curves](../results/cifar-hwa-cdt-analysis/cifar100_hwa_cdt_kl.png);
  corresponding PDF files are in the same directory.
- [CIFAR-10 development histories](../results/cifar-hwa-cdt-analysis/cifar10_hwa_development.png)
  and [CIFAR-100 development histories](../results/cifar-hwa-cdt-analysis/cifar100_hwa_development.png).
- Verified native roots are under results/cifar-hwa-collected/v3/results,
  named cifar10-pcm-hwa-cdt-comparison-v1 and cifar100-pcm-hwa-cdt-comparison-v1.
  Each has 6/6 complete stages, exact paired P0 checks, fixed fault identities,
  full test/recovery budgets and ready_for_review=true with full artifact hashes.
- [Sixteen saved endpoints were replayed](../results/cifar-hwa-collected/v3/results/endpoint-replay.json)
  on all 10,000 test images. Prediction
  hashes and accuracies match exactly; the largest KL difference is below
  4e-8 nats. The replay covers nominal standard HWA and 1% Gmax CDT, with no
  recovery, calibration, weight-only and joint recovery, on both datasets.
- Regression gate: 102 passed, 1 skipped. Source archives and operational
  history are retained in artifacts/cifar_pcm_hwa_sources. V2 failed attempts
  are preserved; [all 120 earlier validation points reproduced exactly in V3](../results/cifar-hwa-collected/v3/results/restart-parity.json).
  Final numerical source identity:
  dc9997efffcf151f9cfc1b8fc56afc3b144270a343b23c845a8640c63137392d.

Managed manifest finalization is pending the user's interpretation and next
tests, as required by the EBL study-closeout workflow. No scientific review
has been invented or finalized from run completion alone.
