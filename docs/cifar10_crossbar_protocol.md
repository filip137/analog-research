# CIFAR-10 crossbar extension, 17 September 2026

This exploratory extension addresses the requested HWA-versus-recovery
comparison. It uses the AIHWKit 1.1.0 IBM OM fitted preset, not measured trace
replay or fabricated hardware. The CIFAR-10 comparator is a bias-free dense
3072–256–10 network with digital ReLU. This scope supersedes the older MNIST
dimensions in the inherited worktree notes; no DRN comparison is included.

## Frozen digital source

The source was pretrained locally from seed 42 for 100 epochs with AdamW,
batch 256, initial LR 0.001, weight decay 0.0001 and cosine decay to 1% of the
initial LR. It uses no augmentation. A fixed permutation with seed 42 splits
the official 50,000 training images into 45,000 training and 5,000 validation
images. Per-channel normalization is fitted on those 45,000 images only; CHW
inputs are flattened. Minimum validation CE, then higher accuracy, then
earlier epoch selects the digital checkpoint. The minimum validation accuracy
gate was 45%. The official test set is evaluated after selection.

Exact source:
`results/cifar10-digital-relu-pretrain-20260917-v1/runs/digital/20260917T105249.415682Z-5fd72081-a5f3b148/checkpoints/teacher.pt`

SHA-256: `ee173c2accad7ae9490613b0a629e7cebf7beb9900f2bfd44cd199ac7c7ddaa4`.
Selected epoch 6, validation accuracy 52.24%, test accuracy 50.97%.

## Mapping and hardware

Each layer uses one absolute-maximum scale (omega 1), frozen from the digital
source for every arm. Six 512×256 input tiles and one 256×10 output tile give
788,992 active programmable states. Fixed fitted references are retained;
effective state is q=a−r. This is an ideal standard crossbar MVM with the
declared device nonidealities; line resistance, ADC/DAC quantization and
physical energy are not modeled.

Population assignment 42011 is used only for HWA training. Assignment 43011
is the development array. Held-out arrays are 44011 and 45011. Their respective
programming seeds are 53011, 54011 and 55011. Healthy arrays use the
counterfactual-repaired native sampled population. The published companion
population of each assignment supplies its post-deployment stuck-cell mask
and collapsed persistent states (corruption probability 0.1348, range 0.01).
Native OM defaults have corruption disabled; this is an explicitly enabled
stress condition. Healthy and faulted arms share the healthy programming
realization before the fault transition.

Programming starts at the sampled lower persistent bound. One-pulse P&V uses
held apparent verify values, tolerance 0.04745 in q, maximum 128 pulses per
cell, and no target clipping. Direct and HWA deployments use the same array
identities and initial per-cell random streams, with different frozen targets.
Apparent q is the primary forward state; persistent q is diagnostic only.

## HWA selection

Four arms begin at the exact digital source with noise multipliers
0, 0.25, 0.5 and 1. A frozen repaired training population clips the forward
state to each cell's native support. The forward then adds Gaussian apparent
noise of sigma = nominal_dw_min × write_noise_std × multiplier. An identity
STE trains FP32 master q bounded to [-1,1] with Adam, LR 0.0001 in q, batch
128 and ten epochs. The objective is KL(teacher || student), temperature 1.
The data and standard-noise random sequences match across multiplier arms.

At each epoch the master is programmed on the healthy development array using
the same endpoint seed. Minimum development validation KL, higher validation
accuracy, earlier epoch selects within each arm (epochs 1–10). The same rank,
then smaller multiplier as a final tie break, selects across arms. No HWA test
accuracy enters either choice. The selected clean master is independently
programmed on every target array; no realized source endpoint is transferred.
The zero-noise arm still adapts to sampled support and is not the direct
digital control. Direct deployment is reported separately.

## Recovery

Each of three selected-HWA deployments supplies two saved P0 states: healthy
and post-deployment faulted. Each state is the identical starting point for
open-loop pulse Adam and closed-loop apparent-feedback Adam/P&V. Recovery
restores all persistent, apparent, pulse and random-stream state; no remapping
or resampling occurs. Faulted restores require the published source population
and exact healthy pre-fault state.

Both writers use LR 0.00001 in q, batch 64, constant schedule, 30 epochs,
KL(teacher || student), beta=(0.9,0.999), epsilon=1e-8, and a 640-pulse per-cell
recovery cap. Open loop emits at most one stochastic pulse per cell per
minibatch. Closed loop accumulates Adam commands into a target and verifies
held apparent state with tolerance 0.04745 and at most 128 pulses per update.
Neither writer receives hidden bounds or a fault mask. Digital gradients and
Adam memories assist the physical updates; this is not autonomous analog
backpropagation. Verify counts represent cell-value observations, not time or
energy.

Recovery selection uses minimum validation KL, higher accuracy, then earlier
epoch; P0 (epoch zero) is eligible. Test metrics are evaluated after selection.
The final epoch is also reported as a diagnostic. All arms have the same
training split and per-epoch sample permutation. The batch size is larger than
the earlier MNIST recovery runs, so pulse exposure is not matched between
datasets. Rates are transferred exploratory choices, not a CIFAR-10 optimum.

Selected endpoints are retrospective saved simulation states. Returning a
physical array to an earlier selected endpoint would require a separate
reprogramming operation. Report both pulses through the selected epoch and
the full training budget; do not treat retrospective selection as cost-free
physical rollback. HWA and recovery also have different training budgets and
information: recovery observes its particular deployed array. The comparison
estimates its incremental benefit, not superiority over every possible
off-chip method with a target-array fault map.

## Analysis and execution

The plan declares 22 hardware runs: four HWA arms, six deployments, and twelve
recoveries. Report each development result separately, followed by the mean
and range across the two held-out arrays. Primary effects are paired recovery
minus HWA-only accuracy (percentage points) and relative teacher-KL reduction.
Also report direct-to-HWA effects, teacher agreement, prediction flips,
selected epochs, pulse costs and persistent diagnostics. Do not use the test
set to choose a writer or revise settings. A negative or zero selected effect
is valid evidence.

Execution uses the native `python -m ebl train` CLI under the prepared study.
The campaign script only orders subprocesses and pins their input artifacts.
Operational smoke configs use two minibatches and 64 evaluation images and
are excluded from scientific evidence. Monitor launch, semantic progress,
logs, metrics, and GPU/process state, then every 30 minutes until all terminal
artifacts are collected. Preserve failed attempts; retry only operational
defects without changing scientific settings. Final scientific review remains
subject to the repository study closeout workflow.
