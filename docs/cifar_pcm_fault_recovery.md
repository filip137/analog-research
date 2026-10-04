# PCM device failures and one-epoch recovery

This exploratory study implements the user's requested failure screen inspired
by Li, Tsai, Narayanan and Rasch, *Impact of analog memory device failure on
in-memory computing inference accuracy*, APL Machine Learning 1, 016104 (2023),
[doi:10.1063/5.0131797](https://doi.org/10.1063/5.0131797).
The Methods/Inference section assigns failures to either or both conductances
of a differential pair. We follow that physical-device definition, not a
replacement of every failed logical weight by a fixed signed value.

## Frozen experiment

- Full pretrained CIFAR-10 and CIFAR-100 ResNet-32, last four convolutions and
  classifier mapped to the existing nine logical crossbar tiles. Earlier
  layers, residual arithmetic and activations remain digital.
- Start from the original digital checkpoints. This screen has **no HWA
  adaptation** and cannot establish superiority over a converged HWA model.
- Three independently sampled array identities per dataset. Each covers
  nominal plus open, Gmax and random failures at 100, 1000 and 10000 ppm.
  Rates mean probability per physical device, not per logical weight.
- Open devices are fixed at 0 uS; Gmax at 25 uS; random values are sampled
  uniformly from [0, 25] uS once. Masks, failed banks and random values are
  permanent. Identical uniform draws give nested masks across rates and
  matched locations across failure types. Either bank or both may fail.
- Healthy devices use the AIHWKit 1.1.0 Gaussian programming-endpoint law.
  The endpoint is sampled once per programming operation and held during
  inference. No pulse/P&V controller is invoked. This screen measures
  immediate post-programming behavior; no drift or extra read noise is added.
- Exactly one epoch on the fixed balanced 5000-image recovery cohort, batch
  size 64 (79 minibatches), Adam weight-target and calibration rates 1e-3.
  This is one pass over the recovery cohort, not all 50000 CIFAR training images.
  Teacher KL at T=1 is the only optimization objective. No test labels or
  test checkpoint selection enter adaptation.
- Evaluate all 10000 official test images before and after recovery. Primary
  metric: mean KL(p_teacher || p_student), natural logs, nats per image.
  Report accuracy and programming-request counts as secondary measures.

## Recovery abstraction and controls

On-chip recovery here means digital gradients evaluated through the deployed
array, followed by complete Gaussian endpoint reprogramming of the requested
weight targets after each minibatch. This was stated to the user before
execution. It is not a simulation of incremental PCM pulses or fully analog
backpropagation. Fault maps remain internal to the array; optimizer updates
use observed effective weights and do not use a fault mask or hidden bounds.
The straight-through gradient is an explicit approximation.

Every control restores the exact saved P0, programming RNG and fault identity:

1. No recovery.
2. Digital calibration only: suffix BN affine terms, layer gains and head bias.
3. Frozen-target rewrite plus calibration: reprogram the original target after
   **every minibatch**, matching the recovery rewrite schedule.
4. Weight-only on-chip recovery: only crossbar targets change; digital
   calibration parameters remain fixed.
5. On-chip recovery plus the same digital calibration as controls 2 and 3.

Programming the same array cannot repair its permanently stuck devices.
Reprogramming uses new independent programming errors on healthy devices.
All methods share minibatch order; all rewrite methods start with the same
programming RNG. A zero weight learning rate performs no recovery rewrites.
Count complete endpoint-programming requests rather than inventing pulse,
verification, latency or energy costs.

An explicitly labelled `ideal_update_upper_bound` option exists for future
controls. It sets recovery programming noise to zero and is **not selected**
by the Gaussian experiment configs.

## Execution and analysis

Native experiment: `cifar_pcm_fault_recovery.v1`, through `python -m ebl train`.
The cache stage reproduces digital test accuracy and records frozen-prefix
features and teacher outputs. Three screen stages each execute the ten
physical cases and five controls. There are two four-stage dataset studies,
60 physical deployment cases and 300 control measurements in total.

Configs and study plans live in
`examples/cifar_crossbar/pcm_faults_cifar10/` and
`examples/cifar_crossbar/pcm_faults_cifar100/`. The existing campaign executor
handles native subprocesses, explicit cache inputs, logs and result receipts.
Run each dataset campaign once; never run two writers on one execution receipt.

`python -m experiments.cifar_crossbar.fault_campaign report` reads those
receipts, verifies exact coverage and pairing, and produces per-array CSV,
aggregate CSV/JSON, Markdown and Matplotlib PNG/PDF curves. Show mean and
sample standard deviation over three arrays and retain all negative outcomes.
No confirmatory statistical significance or paper benchmark reproduction is
claimed from this exploratory screen.

All authorized GPUs were occupied when the study was prepared. The local
24-thread CPU was mostly idle; two dataset workers, four Torch threads each,
are the initial execution target. A small native canary must complete before
the full recovery screen. Keep log/artifact progress visible; inspect at least
every 30 minutes and diagnose any 45-minute stall. Preserve failed attempts.

## Reproducing the analysis

The numerical source archive and source receipt are retained in
`artifacts/cifar_pcm_faults_source_v1/`. Execution receipts refer to its
original extracted location, `/tmp/cifar-pcm-faults-source-v1`. Restore the
archive there, or use the existing `EBL_CIFAR_MIRROR` path mapping if needed;
do not rewrite the original execution receipts. Reporting improvements made
after launch do not change the frozen numerical implementation.

After both campaigns complete, generate paired KL tables, accuracy tables,
per-array CSV and standalone PNG/PDF curves with:

```bash
python -m experiments.cifar_crossbar.fault_campaign report \
  --execution results/cifar10-pcm-faults-one-epoch-v1/campaign-execution.json \
              results/cifar100-pcm-faults-one-epoch-v1/campaign-execution.json \
  --output results/cifar-pcm-fault-analysis
```

Final checkpoints save the actual conductances and programming RNG under
`array`, the controller target under `master_target`, and digital parameters
under `model`. For replay, load `model`, restore `array`, and pass `array.read()`
as the forward weight argument. The shadow `model.q` is the original digital
target and must not be mistaken for the deployed physical weights.
