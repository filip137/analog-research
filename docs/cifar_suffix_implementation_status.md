# CIFAR suffix implementation validation

Branch: `codex/cifar-resnet-suffix-recovery`, based on MNIST recovery commit
`27ad7c4d5f0bd8c4b0e75d8cf9f138f4b4de2626`.

The user subsequently clarified the PCM benchmark: use **Gaussian programming
endpoint noise**, with no additional pulse/P&V simulation. The `pcm` columns
in the historical pilot below use a different, explicit pulse/refresh model.
They are retained as legacy diagnostics and do not characterize the intended
Gaussian PCM benchmark. The separate `pcm_inference` run near the end of this
note uses the intended programming abstraction. New default campaigns no
longer select the PCM pulse model implicitly.

A subsequent Gaussian PCM permanent-failure screen completed on both CIFAR
datasets, with three arrays per dataset and full official test evaluation.
See [the failure-screen results](cifar_pcm_fault_results.md) and
[its protocol](cifar_pcm_fault_recovery.md). That screen adds one-pass recovery
through Gaussian endpoint reprogramming and does not use the legacy pulse
model or a newly trained HWA source. The historical pilot statements below
describe the earlier evidence only.

The subsequent matched HWA/CDT comparison is also numerically complete on
both datasets: full 45,000-image HWA training, development-only source and
recovery selection, three fresh final arrays and 1,260 paired measurements.
See [the HWA/CDT results](cifar_pcm_hwa_results.md) for accuracy, teacher KL,
convergence limitations and verification. Scientific review remains pending.

## Declared pilot

The exploratory implementation pilot covers 38 native stages: full digital
test-set audits for CIFAR-10/100, and main-suffix characterization, one-epoch
500-image HWA, deployment and all five paired controls for OM and PCM. It is
defined in `examples/cifar_crossbar/pilot_nom_cool_1/campaign.json` before launch.
These truncated adaptation runs validate execution and estimate runtime; they
cannot establish the relative scientific performance of HWA and on-chip
training. No full scientific screening or confirmation is complete yet.

Both 5090s were occupied by unrelated jobs when checked. The initial pilot uses
the idle RTX 3090 on nom-cool-1 with cuDNN disabled. An isolated CPU environment
at `/home/filip/server_code/venvs/cifar-aihwkit-1.1.0` supplies AIHWKit 1.1.0 and
Torch 2.11 CPU, leaving the host's GPU Torch 2.5.1 installation untouched.

Initial staged source: `/home/filip/server_code/cifar_suffix_snapshots/pilot-v1`.
Its `source-receipt.json` records each source digest and the base commit.
Remote results live below the snapshot's `results/cifar-suffix-pilot-v1`;
`campaign-execution.json` records the active PID, exact command and log, plus
every completed native result and dependency artifact digest. The launcher
log and exit-code receipt are recorded beside it. Expect an artifact at each
stage and epoch, with a 30-minute watchdog interval and a 45-minute stall limit.
Safe retries retain the same frozen settings and failed bundles. Numerical
failures, bad checkpoint accuracy or inadequate kernels are not silently
ignored. Source corrections require a new frozen snapshot.

## Checks completed before launch

- Initial regression run: 71 passed, one skipped across CIFAR and relevant
  inherited CLI, crossbar and study-workflow tests.
- Digital mapping agrees with ordinary PyTorch logits and gradients for all
  three suffix sizes; prefix parameters and BN buffers remain frozen.
- Both devices and both pulse writers replay exact state and RNG after resume;
  zero learning rate performs no writes.
- Native AIHWKit PCM SET: 0.5880295634 versus equation 0.5880295849. Native
  RESET: 0.0099999998. Inference drift/noise parity at all four retention times
  had maximum absolute difference zero. Global compensation matches the native
  identity-probe implementation.

## Completed validation, 2026-09-17

The final `pilot-v4` snapshot completed all **38/38** native stages with exit
code zero. Full artifact verification passed both on nom-cool-1 and after local
collection. Its frozen source identity is
`0c6f05de5cfb0962997288ac3212cbfd8aecaa8dc36e859220a661bc6e19a890`.
Final targeted regression suite: **82 passed, one skipped**. The final source
also advertises CIFAR artifact inputs in the public CLI, records selection
receipt hashes, rejects cross-device HWA inputs, and disables TF32 in both
matmul and cuDNN; these guards are covered by the regression checks. They do
not change the completed cuDNN-disabled pilot's numerical settings.
Collected source, checkpoints, costs and logs are under
`results/cifar-pilot-collected/pilot-v4/`. The native study lives at
`results/cifar-pilot-collected/pilot-v4/results/cifar-suffix-pilot-v1/`.
Original manifests retain their remote paths and hashes; they were not rewritten.

| Full digital audit | Test images | Digital accuracy | Tiled suffix accuracy | Prediction agreement |
|---|---:|---:|---:|---:|
| CIFAR-10 | 10,000 | 93.53% | 93.53% | 100% |
| CIFAR-100 | 10,000 | 70.16% | 70.16% | 100% |

All runs retained the frozen prefix hash. Every five-way recovery comparison
restored exactly the same P0 and initial apparent-state hash. Reproducibility
checks also cover repeated native population sampling and an interrupted HWA
run, including optimizer/RNG restoration and export of the inherited best model.

## What actually ran in the pilot

The 38 stages comprise two full digital audits, eight characterization stages
(HWA kernel and deployment-population preparation for each dataset/device),
four HWA runs, four deployments, and twenty paired control evaluations.
They are not 38 independent array trials. Each dataset/device combination has
one HWA seed, one target-array assignment and one programming realization.

Every analog run uses the full pretrained ResNet-32 with the last two residual
blocks (four convolutions) and classifier converted. This is 148,096 logical
weights for CIFAR-10 or 153,856 for CIFAR-100, distributed over nine logical
tiles of capacity 512 x 512. The earlier network, residual additions, ReLU,
pooling and normalization remain digital. HWA and recovery each use one epoch
on 500 balanced training images, with batch size 64 and teacher KL at
temperature one. Recovery uses the same 500 training-image identities as HWA,
without HWA's crop/flip augmentation. Learning and calibration rates are 1e-3.

Only nominal noise and variation were run. OM uses the **repaired** population
policy: native corrupt cells are replaced by independently sampled functional
donors. There are no extra stuck-cell faults or MVM output read noise. The
native-corruption, noise-severity, variation and fault sweeps have not run.
The pilot endpoint kernels use 41 target bins and 64 samples per bin, reduced
from the planned scientific characterization budget of 512 samples per bin.

### HWA checkpoint selection correction

All four HWA runs selected **epoch zero**. One epoch was executed, but its
mean development accuracy over three held-out endpoint draws was lower:

| Dataset/device | Before HWA updates | After one HWA epoch | Selected epoch |
|---|---:|---:|---:|
| CIFAR-10 OM | 99.87% | 99.80% | 0 |
| CIFAR-10 PCM | 97.33% | 96.73% | 0 |
| CIFAR-100 OM | 93.13% | 90.80% | 0 |
| CIFAR-100 PCM | 30.60% | 29.87% | 0 |

Consequently, deployment used the original digital model's targets exported
by the HWA selection stage. This exercises HWA training and selection, but
does **not** provide a successfully adapted or converged HWA baseline. Calling
these results a tuned HWA-versus-recovery comparison would be misleading.
A read-only checkpoint audit confirmed selected epoch zero, identical saved
HWA/deployment target tensors, and identical P0/apparent-state hashes across
all five controls for each of the four comparisons.

KL provides a different view of this one-epoch result. Mean
KL(p_teacher || p_student), in nats at temperature 1, changed as follows:

| Dataset/device | Before HWA updates | After one HWA epoch |
|---|---:|---:|
| CIFAR-10 OM | 0.00467610 | 0.00456019 |
| CIFAR-10 PCM pulse/refresh | 0.12303619 | 0.09426759 |
| CIFAR-100 OM | 0.17964137 | 0.21413993 |
| CIFAR-100 PCM pulse/refresh | 3.76393075 | 3.27062040 |

Thus three of four HWA runs improved KL despite lower accuracy. The original
selection ranked accuracy first and KL second, so it selected epoch zero in
all four. The historical selections have not been changed retrospectively;
the one-epoch runs are not evidence that HWA cannot improve teacher fidelity.

### Paired development accuracies

All values in the following table use a separate balanced **500-image
development subset**, not the official test split. These images were excluded
from our HWA/recovery updates but belonged to the public checkpoints' original
digital pretraining set. This explains why their clean accuracy is much higher
than the 93.53%/70.16% official test accuracy. Each table column is one array,
not a multi-array mean.

| State/control | CIFAR-10 OM | CIFAR-10 PCM | CIFAR-100 OM | CIFAR-100 PCM |
|---|---:|---:|---:|---:|
| Clean selected weights, before programming | 100.0% | 100.0% | 98.2% | 98.2% |
| Programmed array, no recovery | 100.0% | 97.6% | 91.8% | 29.4% |
| Digital calibration only | 100.0% | 99.0% | 92.8% | 35.8% |
| Rewrite unchanged targets, then calibration | 100.0% | 99.2% | 94.6% | 32.4% |
| Open-loop pulse Adam plus calibration | 99.6% | 95.6% | 90.0% | 28.0% |
| Closed-loop pulse Adam/P&V plus calibration | 100.0% | 47.4% | 92.8% | 4.4% |

Calibration changes suffix BN affine parameters, converted-layer output gains
and classifier bias; it does not change crossbar states. Rewrite retries the
unchanged selected targets. The pulse arms use digitally computed gradients
and Adam updates to change the same simulated devices. There is no fully
analog backward pass in this implementation.

Teacher KL was measured throughout and used as the training objective. The
initial summary omitted it. Lower KL is better:

| State/control | CIFAR-10 OM | CIFAR-10 PCM pulse/refresh | CIFAR-100 OM | CIFAR-100 PCM pulse/refresh |
|---|---:|---:|---:|---:|
| Programmed array, no recovery | 0.00096609 | 0.12468005 | 0.19850684 | 3.30317029 |
| Digital calibration only | 0.00089536 | 0.06459032 | 0.14471026 | 2.68646143 |
| Rewrite unchanged targets, then calibration | 0.00048282 | 0.12353556 | 0.13027901 | 2.94971130 |
| Open-loop pulse Adam plus calibration | 0.01210301 | 0.14819513 | 0.23586938 | 2.96769006 |
| Closed-loop pulse Adam/P&V plus calibration | 0.00089536 | 1.29733856 | 0.14471026 | 5.33905322 |

For example, CIFAR-100 PCM open-loop decreases KL from 3.30317 to 2.96769
even while accuracy decreases from 29.4% to 28.0%. It still has worse KL than
calibration alone. CIFAR-10 OM rewrite halves KL despite unchanged 100%
accuracy. Accuracy alone conceals these changes.

Read-only reanalysis exports all existing HWA/deployment/recovery KL values,
absolute control metrics and paired KL reductions under
`results/cifar-pilot-collected/analysis-kl/`. KL reduction is control minus
recovery, with endpoint means computed within each array first. It is a
descriptive secondary outcome; original accuracy inference is unchanged.

The clean-to-deployment losses are respectively 0.0, 2.4, 6.4 and 68.8
percentage points. Neither pulse method beats both calibration and rewrite
in any of these four pilot comparisons. OM closed-loop performs **zero physical
writes** in both datasets and exactly matches calibration; it does not yet
exercise an active closed-loop recovery regime.

PCM has material programming/controller diagnostics. Initially 36,028 of
148,096 CIFAR-10 targets (24.33%) and 40,071 of 153,856 CIFAR-100 targets
(26.04%) exhaust the 128-pulse programming budget without meeting the verify
tolerance. Closed-loop recovery then uses 44,941,602 and 45,958,001 additional
physical writes, respectively, including refresh, while accuracy collapses.
The current custom PCM refresh policy runs after pulse-training updates;
initial programming and the rewrite control do not invoke that refresh hook.
The pilot does not isolate gradient updates, step-size choice, verification
and refresh as causes. These outcomes require controller/programming diagnosis
as well as tuning; they do not show that PCM recovery is intrinsically harmful.

Raw evidence is retained under
`results/cifar-pilot-collected/pilot-v4/results/cifar-suffix-pilot-v1/runs/`,
with per-run `result.json`, resolved configs and checkpoints. No new numerical
experiments were run for this scope audit.

## Runtime and outstanding scientific comparison

The 2x pilot extrapolation for 5,000 images and 30 epochs was 1.02–1.30 minutes
for OM and 15.52–15.73 minutes for PCM, below the two-GPU-hour sizing gate.
Peak allocated training tensor memory was 0.125–0.140 GB. These are projections,
not measured full-budget runtimes: the one-epoch OM closed-loop pilot issued
zero pulses, so its later write activity can increase runtime. Retain the main
four-convolution suffix and check throughput again during development tuning.

The pilot **does not show a recovery advantage**. At the untuned default 1e-3
rate, PCM closed-loop recovery degraded strongly on the 500-image development
subset: 47.4% versus 99.0% calibration-only for CIFAR-10, and 4.4% versus 35.8%
for CIFAR-100. OM closed-loop matched calibration while making no writes during
this short run. These are implementation diagnostics after only one HWA epoch,
not test-set comparisons or tuned scientific results. They motivate the declared
HWA and recovery learning-rate selection stages. Failed/negative outcomes have
been retained in the paired CSV and plots; no on-chip-necessity claim is made.

`analysis/paired/` contains the remote paired report and plots. A read-only
analysis of the collected mirror is in `results/cifar-pilot-collected/analysis-local/`.
`analysis/runtime-selection.json` contains the sizing receipt. The complete
nominal tuning, severity screening, confirmation and separate inference-reference
graphs have been materialized and checked for valid dependencies. One
dataset/device confirmation graph has 1,054 stages, including exactly 900
recovery arms (3 cases x 3 HWA seeds x 5 arrays x 2 endpoints x 2 budgets x 5 controls).
Full scientific tuning/screening/confirmation has **not** been run or finalized.

## Retained operational corrections

- `pilot-v1`: a string/Path mismatch in the source-receipt hash guard stopped
  execution before numerical work; the guard and dependency hashing were fixed.
- `pilot-v2`: the full CIFAR-10 audit passed; OM sampling then found an incorrect
  published-preset lookup key. Native sampling now uses the existing preset
  identifier and excludes constructor-random weight buffers from identity hashes.
- `pilot-v3`: all numerical stages completed, but the study verifier rejected
  evaluation-only arms because they lacked `metrics.jsonl`. Every stage now
  emits terminal metrics, and a regression test exercises full study verification.
- `pilot-v4`: complete and artifact-verified. Prior snapshots and their logs
  remain on nom-cool-1. No pilot process remains active.

Implementation validation is finished. Scientific closeout is intentionally
pending the full declared experiment coverage and interpretation; this pilot is
not a finalized study demonstrating a need for on-chip training.

A separate 500-image CIFAR-10 smoke also completed the one-day PCM inference,
global drift compensation and suffix-BN recalibration path from digital weights.
Its exact config, native result and log are collected under
`results/cifar-reference-collected/`. This tests the independent reference path;
it is not a tuned reproduction of the IBM accuracy benchmark.

In that **Gaussian PCM endpoint** run, mean teacher KL was 0.00051182 just
after programming, 0.00094272 after one day with global drift compensation,
and 0.00047482 after suffix BN recalibration. All three had 100% accuracy on
the same 500-image CIFAR-10 development subset. The pulse pilot's initial
PCM KL of 0.12468005 is about 244 times the Gaussian endpoint run's 0.00051182,
but this is a single-realization comparison between different models, not a
device-performance estimate. At that point no equivalent CIFAR-100 Gaussian
endpoint run had been completed; the later failure screen linked above now
includes both datasets. The reference run had no HWA adaptation or pulse recovery.

The Gaussian programming distribution summarizes residual error after
write/read/verify; it is not noise to add to every individual programming
pulse. See the [IBM model documentation](https://aihwkit.readthedocs.io/en/latest/pcm_inference.html#programming-noise).
The old pulse pilot did not combine both noise models; it selected a different
programming model. Using its behavior as the classical PCM baseline was the
model-selection/reporting error.
