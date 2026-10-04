# CIFAR PCM permanent-failure results, 2026-09-17

Numerical collection is complete and artifact-verified. Scientific review and
formal study finalization await the user’s interpretation and next-test decisions.

## Tested scope

The question was whether one pass of array-specific weight adaptation lowers
teacher KL beyond digital calibration and equally frequent frozen-target rewrites.
The [frozen protocol](cifar_pcm_fault_recovery.md) specifies the complete experiment.

- Full pretrained CIFAR-10/100 ResNet-32; final four convolutions and classifier
  mapped to nine logical tiles, with the digital prefix frozen.
- Three arrays per dataset; nominal plus open, stuck-high and random permanent
  physical-device failures at 100, 1000 and 10000 ppm (0.01%, 0.1%, 1%).
- Open devices stay at 0 µS, stuck-high at 25 µS, random at a fixed draw in
  [0,25] µS. Either or both devices of a differential pair can fail.
- Gaussian programming-endpoint noise, no explicit pulse/P&V, drift or read noise.
- Digital gradients and Adam, learning rates 1e-3, one pass over 5000 balanced
  recovery images (79 minibatches), not a pass over the entire 50000-image set.
- All 10000 official test images; mean KL(p_teacher || p_student), nats at T=1.
- Every control starts from the identical saved physical state and RNG. Failed
  devices remain stuck; the optimizer does not receive the fault map.

The source is the pretrained digital network, without a new HWA stage. These
results do not answer whether recovery outperforms a converged HWA baseline.
Recovery reprograms complete target endpoints after each minibatch; it is not
a measured incremental PCM pulse-update algorithm or fully analog backpropagation.

## Measured teacher KL

Values are mean ± sample SD across three arrays. Lower is better. The table
includes nominal and 1% failures; all tested rates and five controls are in the
[complete report](../results/cifar-pcm-fault-analysis/report.md).

| Dataset | Failure | Rate | Deployed | Calibration | Weight recovery |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | none | 0% | 0.0052404 ± 0.0041 | 0.003638 ± 0.0026 | 0.041085 ± 0.0086 |
| CIFAR-10 | open | 1% | 0.0069336 ± 0.0041 | 0.0040381 ± 0.001 | 0.042507 ± 0.014 |
| CIFAR-10 | random | 1% | 0.045566 ± 0.018 | 0.017416 ± 0.0031 | 0.045108 ± 0.016 |
| CIFAR-10 | gmax | 1% | 0.14461 ± 0.053 | 0.04651 ± 0.014 | 0.056901 ± 0.0026 |
| CIFAR-100 | none | 0% | 0.080868 ± 0.011 | 0.038842 ± 0.0024 | 0.16045 ± 0.026 |
| CIFAR-100 | open | 1% | 0.14541 ± 0.028 | 0.074086 ± 0.018 | 0.16246 ± 0.018 |
| CIFAR-100 | random | 1% | 0.84089 ± 0.15 | 0.34626 ± 0.043 | 0.29092 ± 0.0097 |
| CIFAR-100 | gmax | 1% | 3.1662 ± 0.29 | 1.239 ± 0.036 | 0.56761 ± 0.017 |

For CIFAR-100 at 1% stuck-high faults, weight recovery lowers mean KL from
3.16625 to 0.56761, an 82.1% reduction. Calibration reaches 1.23896 and matched
rewriting reaches 1.29554. The paired improvement over calibration is
0.67134 ± 0.04410 nats; all three arrays improve. Recovery plus calibration
reaches 0.56315 ± 0.01599, close to weight-only recovery.

For CIFAR-100 at 1% random faults, weight recovery lowers mean KL from
0.84089 to 0.29092, compared with 0.34626 for calibration and 0.37945 for
rewriting. The paired improvement over calibration is 0.05534 ± 0.04352 nats;
all three arrays improve. Hybrid recovery beats calibration on two of three.

No other tested case has a positive three-array mean KL advantage over
calibration for either recovery method. In particular, the nominal recovery
setting worsens KL on both datasets. This is evidence about the fixed one-pass
1e-3 protocol, not proof that another recovery algorithm or learning rate cannot
help. These three-array comparisons are exploratory, without significance claims.

## Measured accuracy

Full digital test accuracies were reproduced as 93.53% (CIFAR-10) and 70.16%
(CIFAR-100). Gaussian programming without permanent faults gives 93.42% and
68.71% respectively; calibration gives 93.41% and 69.55%.

| Dataset, 1% failures | Deployed | Calibration | Weight recovery |
|---|---:|---:|---:|
| CIFAR-10, open | 93.37% ± 0.04 pp | 93.40% ± 0.11 pp | 92.97% ± 0.33 pp |
| CIFAR-10, random | 92.68% ± 0.45 pp | 93.21% ± 0.05 pp | 92.97% ± 0.26 pp |
| CIFAR-10, gmax | 90.48% ± 1.32 pp | 92.71% ± 0.31 pp | 92.70% ± 0.17 pp |
| CIFAR-100, open | 67.79% ± 0.46 pp | 68.95% ± 0.37 pp | 67.83% ± 0.43 pp |
| CIFAR-100, random | 58.65% ± 1.90 pp | 64.95% ± 0.74 pp | 65.93% ± 0.20 pp |
| CIFAR-100, gmax | 34.43% ± 0.73 pp | 50.01% ± 0.61 pp | 61.84% ± 0.57 pp |

Even the strongest recovery remains below the clean digital model. For
CIFAR-100 stuck-high faults, accuracy improves by 27.41 percentage points,
but remains 8.32 points below 70.16%.

## Artifacts and verification

- [CIFAR-10 KL curves](../results/cifar-pcm-fault-analysis/cifar10_fault_kl.png)
  and [PDF](../results/cifar-pcm-fault-analysis/cifar10_fault_kl.pdf).
- [CIFAR-100 KL curves](../results/cifar-pcm-fault-analysis/cifar100_fault_kl.png)
  and [PDF](../results/cifar-pcm-fault-analysis/cifar100_fault_kl.pdf).
- [Per-array metrics](../results/cifar-pcm-fault-analysis/per_array.csv),
  [aggregate metrics](../results/cifar-pcm-fault-analysis/summary.csv), and
  [paired comparisons](../results/cifar-pcm-fault-analysis/paired_comparisons.csv).
- Native studies: `cifar10-pcm-faults-one-epoch-v1` and
  `cifar100-pcm-faults-one-epoch-v1`, each with 4/4 stages and 150 measurements.
  Both report `ready_for_review=true` with `full_artifact_hashes` validation.
- All 60 physical cases have five matched controls, all 300 records use
  10000 test images; each adapting control sees 5000 recovery images.
- Each rewrite/recovery arm requests 79 complete array reprogramming operations.
  These are endpoint requests, not physical pulse or energy estimates.
- CPU workers completed all six screens without failed attempts. Both campaigns
  exited zero. No study run remains active.
- Regression checks: 93 passed, 1 skipped. Two representative recovered
  checkpoints exactly reproduce full-test metrics and prediction hashes when
  reloaded: [replay receipt](../results/cifar-pcm-fault-analysis/checkpoint_replay.json).
- Numerical source identity:
  `25781b02c833dfec4710e081e12e292a17af39fa9f824d93510f4e4ee8145034`.
  Archive and receipt: `artifacts/cifar_pcm_faults_source_v1/`.

Failure definitions follow Li et al., [Impact of analog memory device failure
on in-memory computing inference accuracy](https://doi.org/10.1063/5.0131797).
The paper’s HWA benchmark and failure-tolerance thresholds are not reproduced
by this suffix-only, digital-source recovery screen.
