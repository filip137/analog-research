# CIFAR recovery: one to twenty passes

Numerical work completed on 2026-09-17 at 22:22 UTC (September 18 in
Europe/Paris). Both datasets completed their four native stages and passed
full artifact-hash checks remotely and after local collection. Scientific
review and managed-manifest finalization await the user's interpretation.

## What was extended

The [declared protocol](cifar_pcm_recovery_epochs.md) extends recovery on the
same 5000-image cohort to 1, 2, 5, 10 and 20 passes. Twenty passes means 100000
image presentations and 1580 optimizer steps, not twenty full CIFAR epochs.
Every fixed milestone uses all 10000 official test images. The pretrained
digital ResNet-32 and frozen prefix are unchanged; the last four convolutions
and classifier remain on the simulated Gaussian PCM arrays.

Starting points are the original digital weights, standard HWA, noise-selected
HWA, and the three fault-aware CDT sources from the previous comparison.
The first three cover nominal arrays plus 1% open/Gmax/random failures per
physical device. Each CDT covers nominal arrays and its own 1% failure type.
Three new paired arrays per condition are independent of development arrays
and the previous final arrays. Compare one and twenty passes within this
experiment; the earlier study used different final array draws.

Recovery remains digital Adam/backprop with straight-through gradients and
complete Gaussian endpoint reprogramming after each minibatch. Permanent
failures stay fixed; healthy programming noise is redrawn. This is an
array-specific endpoint-model experiment, not incremental pulse training.
The teacher is always the original digital network, and the objective is
mean KL(p_teacher || p_student), in nats at temperature 1.

## More passes help, particularly on CIFAR-100 with Gmax failures

Means across three matched arrays at **1% Gmax failures**:

| Dataset | Starting point / recovery | KL, pass 1 | KL, pass 20 | Accuracy, pass 1 | Accuracy, pass 20 |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | Digital / joint | 0.04613 | 0.03193 | 92.64% | 92.97% |
| CIFAR-100 | Digital / joint | 0.95188 | 0.57219 | 54.73% | 62.32% |
| CIFAR-10 | Standard HWA / joint | 0.04596 | 0.03191 | 92.58% | 92.98% |
| CIFAR-100 | Standard HWA / joint | 0.87945 | 0.54859 | 56.20% | 62.84% |
| CIFAR-10 | Noise HWA / joint | 0.04227 | 0.02995 | 92.76% | 93.03% |
| CIFAR-100 | Noise HWA / joint | 0.48715 | 0.41152 | 64.09% | 66.00% |
| CIFAR-10 | CDT Gmax / joint | 0.04502 | 0.03044 | 92.98% | 93.10% |
| CIFAR-100 | CDT Gmax / joint | 0.48236 | 0.41224 | 64.26% | 66.04% |
| CIFAR-100 | CDT Gmax / weights only | 0.52921 | 0.38372 | 63.14% | 65.99% |

Joint recovery updates both weight targets and digital calibration parameters.
Weight-only recovery keeps calibration at the source values. On CIFAR-100,
the additional passes change the comparison: weight-only recovery now has
lower KL than joint recovery in this severe Gmax condition, while their
accuracies are close.

Much of the gain occurs by pass five. For CIFAR-100 CDT Gmax weight-only
recovery, KL at passes 1/5/10/20 is 0.52921/0.40411/0.39231/0.38372, with
accuracy 63.14/65.70/65.94/65.99%. These fixed-budget curves do not establish
full convergence.

## Equal-budget controls remain essential

All rows below start at the same CDT Gmax source and receive 20 passes.
Mean plus/minus sample SD over three arrays:

| Dataset | Control | Teacher KL | Accuracy (%) |
|---|---|---:|---:|
| CIFAR-10 | Calibration | 0.03680 ± 0.00220 | 93.090 ± 0.140 |
| CIFAR-10 | Rewrite + calibration | 0.03713 ± 0.00227 | 93.043 ± 0.190 |
| CIFAR-10 | Weight recovery | 0.03354 ± 0.00253 | 93.067 ± 0.185 |
| CIFAR-10 | Joint recovery | 0.03044 ± 0.00244 | 93.103 ± 0.182 |
| CIFAR-100 | Calibration | 0.44359 ± 0.02879 | 65.040 ± 0.420 |
| CIFAR-100 | Rewrite + calibration | 0.46769 ± 0.03513 | 64.793 ± 0.754 |
| CIFAR-100 | Weight recovery | 0.38372 ± 0.02045 | 65.987 ± 0.599 |
| CIFAR-100 | Joint recovery | 0.41224 ± 0.02550 | 66.040 ± 0.593 |

For CIFAR-100, weight-only recovery reduces KL **13.50%** relative to equally
trained CDT calibration and adds **0.947 percentage points**. The paired
differences are 0.05987 ± 0.00838 nats and 0.947 ± 0.179 points. Joint recovery
reduces KL 7.07% and adds 1.000 ± 0.185 points. Both methods improve both
metrics on all three arrays against every applicable tested HWA/CDT
non-weight control in this condition. Noise-HWA recovery also improves both
metrics against those controls on all three arrays. These are descriptive
paired results, not significance tests.

For CIFAR-10, CDT joint recovery reduces KL **17.28%** relative to equally
trained CDT calibration, improving KL on all three arrays. Accuracy changes
only +0.013 points, with paired SD 0.060 points and improvement on two arrays.
The additional recovery is visible in distribution fidelity, without a
consistent classification-accuracy advantage. Original digital accuracies
remain 93.53% and 70.16%; recovery is partial.

## Other cases and losses

At 1% random faults, some recovery arms have lower mean KL than every tested
HWA/CDT non-weight control, but accuracy does not consistently exceed all of
those alternatives. For example, CIFAR-100 CDT random calibration gives
KL 0.21668 / accuracy 67.68%; weight recovery gives 0.20515 / 67.96%; joint
recovery gives 0.20653 / 68.05%. Other HWA calibration sources retain higher
accuracy, despite their higher KL. All cross-source contrasts are retained.

For nominal arrays and open faults, no weight-recovery arm has lower mean KL
than the strongest tested HWA/CDT non-weight control at pass 20. On nominal
standard HWA, calibration versus joint recovery gives:

- CIFAR-10: KL 0.001418 versus 0.003255; accuracy 93.43% versus 93.37%.
- CIFAR-100: KL 0.029134 versus 0.078513; accuracy 69.76% versus 69.10%.

More passes therefore do not remove the cost of repeatedly redrawing
programming errors. Calibration adapts to the existing programmed state;
writing controls keep introducing a fresh healthy-device realization.

## Selection and limitations

Base rates remain the previously development-selected 1e-4 for weights,
3e-4 for CIFAR-10 calibration, and 1e-3 for CIFAR-100 calibration. Each dataset
compared constant rates with factors 0.3 and 0.09 after passes 5 and 10 using
48 development trajectories on separate arrays and held-out adaptation images.

CIFAR-10 selects step decay for all methods; development-selected reporting
passes are 20 for calibration/rewrite, 5 for weight-only and 10 for joint
recovery. CIFAR-100 selects constant calibration/rewrite and step-decayed
weight/joint recovery, with pass 20 selected for each. All final curves still
run through 20 and report every fixed milestone. The report also exports the
development-selected endpoints; no final test chooses settings or stopping.

HWA itself was not extended. The reused sources retain their one-training-seed
and late-improvement limitations from the [earlier results](cifar_pcm_hwa_results.md).
No superiority to fully converged HWA, full-network analog deployment, or a
physical incremental training algorithm is established. The data exclude
read noise, drift and peripheral quantization. Three-array SD does not quantify
HWA training-seed uncertainty. Development images were excluded from this
adaptation, but belonged to the public teacher's original pretraining data.

## Artifacts and verification

- [Full report and curves](../results/cifar-epochs-analysis/report.md).
- [CIFAR-10 Gmax curves](../results/cifar-epochs-analysis/cifar10_gmax_10000ppm.png)
  and [CIFAR-100 Gmax curves](../results/cifar-epochs-analysis/cifar100_gmax_10000ppm.png);
  PDFs and all nominal/open/random figures are alongside them.
- [Per-array values](../results/cifar-epochs-analysis/per_array.csv),
  [mean/SD summary](../results/cifar-epochs-analysis/summary.csv),
  [paired comparisons](../results/cifar-epochs-analysis/paired_comparisons.csv),
  [cross-source comparisons](../results/cifar-epochs-analysis/cross_source_comparisons.csv),
  and [development-selected endpoints](../results/cifar-epochs-analysis/development_selected_endpoints.csv).
- Eight completed native stages; 96 development trajectories, 540 final
  control records, 432 final learning trajectories, and 2160 post-adaptation
  full-test endpoints. Every writing trajectory makes 1580 additional array
  programming requests; calibration makes zero. No pulse energy is inferred.
- Both native studies report ready_for_review=true with full artifact hashes,
  remotely and after local collection. No failed or missing runs; launchers
  exit zero and the GPU is released. Raw roots are
  results/cifar-epochs-collected/v1/results/cifar10-pcm-recovery-epochs-20-v1
  and the corresponding CIFAR-100 directory.
- [Twenty-four saved endpoints replay exactly](../results/cifar-epochs-collected/v1/results/endpoint-replay.json),
  including predictions, accuracy and KL (maximum KL difference zero).
  Seventy-eight relevant checks passed. The Gmax figures were visually inspected.
- Numerical source identity:
  109493e9ed9f61babdb37c9e631f4fa0a5c51ecfa159a2376da57dc7c99c0bbd.
  The source archive and audit helpers are under artifacts/cifar_pcm_epoch_sources.
  Post-run analysis code has its own digest in results.json.

Formal scientific interpretation, next-test decisions and managed-manifest
finalization remain pending the user review required by EBL study closeout.
