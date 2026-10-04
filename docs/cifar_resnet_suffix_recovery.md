# CIFAR ResNet suffix recovery protocol

This branch extends the MNIST work at `27ad7c4d` into full CIFAR-10 and
CIFAR-100 classification. The question is whether adaptation on a particular
deployed array improves on population hardware-aware training (HWA), digital
calibration, and rewriting the unchanged HWA targets. All device evidence is
simulation using AIHWKit 1.1.0 presets. This is same-task recovery; labels and
class counts do not change between digital pretraining and deployment.

## PCM model clarification, 2026-09-17

The user clarified that the intended classical PCM benchmark uses the
**Gaussian programming-endpoint model** (`pcm_inference`). Its conductance-
dependent programming distribution represents residual error after iterative
write/read/verify, as described in the
[IBM statistical-model documentation](https://aihwkit.readthedocs.io/en/latest/pcm_inference.html#programming-noise)
and [AIHWKit 1.1.0 implementation](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/pcm.py).
Sample that endpoint once for deployment and hold its programming realization
fixed during evaluation; do not add another pulse/P&V controller. Drift and
read-noise processes remain separately defined. The existing inference path
already follows this separation.

The completed `pcm` pilot instead uses an explicit SET/RESET pulse model and
custom refresh. It remains a legacy alternate-model experiment; its failure
does not establish poor performance of the Gaussian endpoint benchmark.
Descriptions of PCM pulses below apply only to that explicit alternate model.
New campaign defaults use OM for pulse phases, OM plus `pcm_inference` for HWA,
and `pcm_inference` for reference. Explicit `--backends pcm` is still available
for reproducing the old model. At the time of that clarification, Gaussian
PCM recovery had not been implemented: the endpoint distribution alone does
not define incremental write behavior. The subsequent
[permanent-failure screen](cifar_pcm_fault_recovery.md) and
[matched HWA/CDT comparison](cifar_pcm_hwa_comparison.md) explicitly define
complete Gaussian endpoint reprogramming for recovery.
The [completed matched results](cifar_pcm_hwa_results.md) report accuracy,
teacher KL and convergence limitations. This is an explicitly declared
statistical recovery abstraction; incremental pulse dynamics require separate
device characterization.

Report **mean KL(p_teacher || p_student)** in nats, at temperature 1, alongside
accuracy at every measured stage. KL reduction is initial/control KL minus
recovered KL, so positive is better. Raw runs already store this metric; the
analysis now exports HWA histories, deployment and recovery values in
`teacher_metrics.csv`, plus absolute and paired KL in the report and plots.
This reporting correction does not retrospectively change accuracy-based
checkpoint selection or the predeclared accuracy inference family.

## Architecture and inputs

Use the public pretrained ReLU ResNet-32 checkpoints from
[chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models).
The architecture has 16/32/64 channels, five blocks in each stage, projection
shortcuts, and the original 10- or 100-class head. `prepare.py` downloads only
state dictionaries, validates full SHA-256 digests, and records source URLs;
it does not execute downloaded model code. The author's reported test
accuracies are 93.53% and 70.16%. A full 10,000-image digital audit must reproduce
the relevant number within 0.25 percentage points before using a checkpoint.
Normalization follows the author's training logs, including CIFAR-100 mean
`[0.507, 0.4865, 0.4409]`, std `[0.2673, 0.2564, 0.2761]`.

The primary analog suffix is `layer3.3.conv1`, `layer3.3.conv2`,
`layer3.4.conv1`, `layer3.4.conv2`, and `fc`: four convolutions plus the
classifier. Alternatives are `last_block` and `head`. Everything before the
cut is frozen, including BN statistics. The primary CIFAR-10 suffix has
148,096 logical weights; CIFAR-100 has 153,856. Each uses nine logical tiles
of capacity 512 x 512. A 64 x 576 convolution matrix occupies two row tiles;
partial outputs are summed digitally. Residual additions, ReLU, pooling, BN,
and output scaling remain digital. OM uses the existing apparent effective
state `q = active - reference`; PCM uses two SET/RESET conductances per logical
weight. Logical tiles are not a fabricated physical device count for OM.

Digital convolution and explicit tiled MVM logits/gradients must agree before
any non-idealities. Each layer has a fixed original absolute-maximum scale.
Weights are never rescaled using a target array's hidden bounds. Calibration
can train only suffix BN affine terms, one output gain per converted layer,
and classifier bias. Ordinary HWA/recovery keeps all BN running buffers fixed.
The separate PCM inference reference can explicitly recalibrate suffix BN
running statistics; it still cannot change the prefix.

## Data and selection

All tasks use all ten or one hundred classes. A fixed seed, 20260917, reserves
a class-balanced 5,000-image development set from the original 50,000 training
images. HWA uses the other 45,000 with crop/flip augmentation. Public digital
pretraining used the original training split; the development split is held
out from our adaptation, not from that earlier pretraining. The official
10,000-image test split is used for the checkpoint audit and frozen final
confirmation, never HWA/recovery selection.

Recovery uses nested balanced cohorts of 500 and 5,000 training images and
digital-teacher KL at temperature 1. Labels are only used in offline accuracy
evaluation; on-chip updates use teacher targets. Cached prefix features retain
the full CIFAR-sized network's function. They do not substitute a small MNIST
model or train the prefix. The teacher can produce targets off chip before
deployment; this cost must not be misreported as free on-device computation.

Development HWA compares CE/KL and learning rates 1e-4, 3e-4, 1e-3, initially
30 epochs with a 10-epoch noise ramp. Selection averages three independent
held-out population draws on the development images. A winner within the final
20% of the epoch budget blocks automatic selection and requires a newly
declared longer HWA study. Epoch zero is a valid candidate. A short pilot is
never accepted as a scientific HWA baseline.

Recovery tuning uses two arrays at nominal settings and two with 2x write
noise, both cohort budgets, pulse Adam rates 1e-5/1e-4/1e-3, and calibration
Adam rates 1e-4/1e-3 for calibration/rewrite controls. Pulse arms use calibration
rate 1e-3. Epoch 5/15/30 is chosen from pooled development results and then
fixed for all confirmation arrays. No per-target best-checkpoint selection is
allowed. All recovery arms use batches of 64 and at most 30 epochs.

## Device and deployment contracts

`native.py` samples real AIHWKit 1.1.0 hidden parameters using an isolated CPU
interpreter. It never supplies those hidden parameters to the controller.
The OM plant reuses the existing validated pulse engine and explicit counter
RNG. Its `noise_scale` multiplies **post-write apparent noise**; its native
cycle-to-cycle step noise remains unchanged. PCM's multiplier changes
**cycle-to-cycle SET update noise**. These are different physical interventions
and must be labelled separately in plots. Variation multipliers apply to native
bound, step-size, and asymmetry device-to-device spreads.

PCM uses the native ExpStep SET equation and native RESET distribution with a
differential pair. The refresh thresholds are 0.75/0.25 of nominal maximum
conductance. Our explicitly declared refresh policy resets both banks and
reprograms the observable pre-refresh difference with at most 128 pulses.
This is a custom observable P&V refresh policy, **not AIHWKit native SGD refresh
or TTv2**. SET, RESET, refresh, and verify costs are all recorded. The PCM
640-write cap applies separately to each physical bank and includes refresh
SETs and RESETs. The controller also has a conservative 640-command logical
cap. OM's 640-write cap counts both pulse directions per active cell.

Population endpoint characterization programs independent calibration and
held-out populations with the same one-pulse P&V routine used at deployment.
Its 41 target bins and 512 endpoints/bin include failed programming terminals
and stuck cells. It never fits only successful endpoints. A held-out mean
adequacy check requires each bin discrepancy to be below max(0.02, four pooled
standard errors). This is a screening gate, not proof of full distributional
fidelity. HWA samples this population kernel with a straight-through gradient;
it never observes a confirmation array's identity, bounds or endpoints.

Screening uses **severity-matched population HWA**: each physical case gets
its own generic population kernel and fresh HWA fit, using the development
selected HWA settings. The target array is independently sampled afterward.
This distinguishes inability to recover target-specific deployment error from
merely failing to expose HWA to a known error distribution. A late-converging
case must be resolved before drawing a need-for-recovery conclusion.

OM starts at the sampled lower bound, an assumed conditioned reset state whose
preparation cost is not measured. PCM starts with an explicitly counted noisy
RESET of both banks. Native tile constructors' random initial weight buffers
are excluded from the physical-parameter population and its fingerprint.
Programming targets the HWA
master, and stops at half a nominal update step or 128 pulses per cell. The
resulting P0 saves persistent/apparent state, identities, RNG counters, targets,
calibration and source hashes. Every paired arm restores that exact P0:

1. Raw HWA deployment, no changes.
2. Digital calibration only.
3. Rewrite the frozen HWA targets with P&V, then digital calibration.
4. Stochastic open-loop pulse Adam plus digital calibration.
5. Accumulated-target closed-loop pulse Adam/P&V plus digital calibration.

Gradients and Adam state are computed digitally. The simulated on-chip action
is updating the same array's persistent weights through physical pulse laws.
This does not model a fully analog backward pass, ADC/DAC quantization, or
optimizer energy. Negative results are conditional on these two controllers;
they cannot prove that all possible on-chip algorithms would fail.

The zero-learning-rate test performs no writes or refresh. Both writer types
have exact state/RNG continuation tests. A controller sees only apparent
weights, its target/history and a nominal population step. Verification counts
every cell read, including unsuccessful passes; no energy or latency claim is
inferred from these logical counters.

## Matrix, compute gate and confirmation

For each dataset and pulse device, screen nominal, OM native corruption versus
independent donor repair, 0.5x/2x write noise, 2x device variation, low/middle/high
stuck cells at 0.1/1/5%, and fresh iid MVM output noise of 0.01/0.05/0.1 in
normalized pre-layer-scale tile-output units. Stuck states are fixed before
programming. Read noise is sampled anew for each tile MVM and is not persistent
weight drift. Screening uses one HWA seed and two independent development
arrays. Keep negative cases and programming failures.

A primary-suffix pilot measures epoch time and peak memory. Conservatively
extrapolate the largest 5,000-image, 30-epoch recovery arm (double the measured
per-batch pilot cost). If it exceeds two GPU-hours or fails memory, choose
`last_block`, then `head` if needed, consistently for that device/dataset
comparison and repeat development selection. A suffix reduction may not be
chosen because its accuracy looks favorable.

Confirm nominal, the strongest non-read-noise development case at 5,000 images,
and the fixed 0.05 read-noise control. Even if all gains are negative, retain the
best nonnominal case and report the negative screening outcome. Use HWA seeds
41/43/47, five new arrays per seed and two programming endpoint replicas per
array. Development and confirmation use disjoint assignment/endpoint seed
namespaces. Both recovery budgets and all five controls remain included.

Primary effects are paired accuracy gains over calibrated HWA and over the
no-learning rewrite. Average endpoints within each array; bootstrap arrays
within fixed HWA seed strata for 95% intervals. Use one-sided paired array
sign-flip tests with Holm adjustment across the complete set of confirmation
comparisons. A supported gain must have positive interval and adjusted p<0.05
against **both** controls. This is conditional on the three trained HWA seeds;
two endpoints from one array are not two independent arrays. Any reliable gain
is relevant; complete restoration of digital accuracy is not required.

Report digital/HWA/deployed/recovered accuracy, CE/KL/agreement, deployment loss,
raw and control-adjusted recovery gain, image presentations, SET/RESET/refresh
counts and verification reads. The protocol makes no hardware energy claim.

## Separate IBM PCM inference reference

`pcm_reference.py` transcribes AIHWKit's measured PCM inference laws at 25 uS,
t0=20 s and read time=250 ns. It evaluates 1 s, 1 h, 1 day and 1 year, with and
without per-tile global drift compensation and with explicit suffix BN
recalibration. Fixed original layer scales are retained. `native_check.py`
compares conductance noise/drift and identity-probe compensation directly to
AIHWKit; it also checks native PCM SET and RESET. Inference noise does not
provide a physical update model, so this reference has no pulse recovery arm.
The device-focused implementation omits the complete ADC/DAC, IR-drop and
peripheral configurations of classical IBM system studies.

The two transfer-learning papers motivate suffix replacement and realistic
device sweeps, rather than furnishing a directly reproduced benchmark:
[Assessing the Performance of Analog Training for Transfer Learning](https://arxiv.org/abs/2505.11067)
and [Transfer Learning on Edge Using 14nm CMOS-compatible ReRAM Array and Analog In-memory Training Algorithm](https://doi.org/10.1109/IEDM50572.2025.11353900).
Classical IBM comparisons should retain each publication's architecture and
noise assumptions; public ResNet-32 accuracy is not interchangeable with a
modified ResNet-16/28/56 benchmark.

## Running the implementation

Inputs are CLI artifacts; scientific configs cannot select paths. All numeric
runs enter `python -m ebl train` with experiment
`cifar_resnet_suffix_recovery.v1`. The campaign module only generates visible
configs/study plans and invokes that CLI. It records dependency digests and
does not search for a newest checkpoint.

```bash
python -m experiments.cifar_crossbar.prepare --output artifacts/cifar_inputs --download-data
python -m experiments.cifar_crossbar.campaign prepare \
  --phase pilot --study-id cifar-suffix-pilot-v1 --device cuda:0 \
  --output examples/cifar_crossbar/my-pilot
python -m experiments.cifar_crossbar.campaign run \
  --campaign examples/cifar_crossbar/my-pilot/campaign.json \
  --inputs artifacts/cifar_inputs --cifar-root artifacts/cifar_inputs/data \
  --native-python /path/to/aihwkit-1.1.0/bin/python
python -m experiments.cifar_crossbar.analysis report \
  --execution results/cifar-suffix-pilot-v1/campaign-execution.json \
  --output results/cifar-suffix-pilot-v1/analysis/paired
```

Run `analysis runtime-gate` on the completed pilot and pass its receipt through
`--runtime-selection` to every scientific pulse campaign. Failed gates require
a new pilot with a smaller suffix. Generate `--phase hwa`, run it, then
`analysis select-hwa`. Pass its immutable
receipt through `--hwa-selection` to `--phase tune`. Use
`analysis select-recovery` and `--recovery-selection` for `--phase screen`.
Finally use `analysis select-screen` and `--screen-selection` for `--phase
confirm`. The generator rejects missing selections. For the separate inference
track generate HWA with `--backends pcm_inference`, select it, then generate
`--phase reference --backends pcm_inference`. Use new study IDs/directories for
new scientific budgets. Generated files must be reviewed/frozen before formal
execution. `--only` selects named nodes and their explicit prerequisites for
operational checks; it does not mark missing declared coverage complete.

Set `EBL_CIFAR_ROOT` and `EBL_AIHWKIT_PYTHON` for individual native commands.
Use `--disable-cudnn` in generated configs on nom-cool-1: its installed cuDNN
fails initialization, while the native CUDA backend works. Source snapshots
can supply `EBL_SOURCE_RECEIPT`; every listed source file is checked before a
run. Keep Git discovery from accidentally using a parent checkout when staging.
Use `python -m experiments.cifar_crossbar.snapshot --output /tmp/source.tgz`
to package source independently of datasets and outputs. For a collected mirror,
analysis accepts `EBL_CIFAR_MIRROR` as a JSON mapping of remote source roots to
local roots; this changes read locations, never the original run manifests.
Check campaign receipt, process, GPU and actual artifact progress after launch
and every 30 minutes. Preserve failed attempts. Full studies use native
artifact-verified summary and human scientific closeout; implementation pilots
are never finalized as a conclusion that on-chip recovery is needed.
