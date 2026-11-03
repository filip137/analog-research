---
id: "exp-006"
title: "PCM/CMO repeated noisy recovery"
status: "imported"
hypotheses: []
---

# exp-006 — PCM/CMO repeated noisy recovery

## Question and provenance

PCM/CMO repeated noisy recovery. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-006-pcm-cmo-noisy.md).

Historical document locations (now references/forwarding pages): [mnist_ibm_pcm_cmo_noisy_recovery.md](../../../../../docs/mnist_ibm_pcm_cmo_noisy_recovery.md).

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-mnist-ibm-pcmcmo-noisy-recovery-study"></a>

## Historical source: MNIST IBM PCM/CMO noisy-recovery study

Last updated: 2026-07-30

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-question"></a>

## Question

Starting from the existing FP32 MNIST DRN, compare two measured IBM endpoint
device models:

1. IBM Analog Foundation Models PCM;
2. IBM CMO/HfOx ReRAM.

For each device, measure FP32 deployment, HWA deployment, full BPTT recovery,
and rank-4 physical LoRA recovery. Unlike the earlier Wan study, every
trainable device array is reprogrammed noisily after every recovery update.

The original nine runs documented below used `normalized_offset` for CMO,
which subtracts its `9 µS` floor after applying device noise. Two
floor-retaining mappings are now separate: `literal_conductance` directly
scales and hard-clips sub-floor targets, while `affine_floor` spans the full
physical window without merging them. The follow-ups are reported separately
so the original result remains reproducible.

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-controlled-sequence"></a>

## Controlled sequence

| Stage | Starting point | Trainable arrays | Device write |
| --- | --- | --- | --- |
| FP32 reference | existing selected FP32 checkpoint | none | none |
| FP32 deployment | FP32 reference | none | one endpoint write |
| HWA | FP32 reference | W1, W2, bias | temporary 3% channel-wise noise during BPTT; clean digital update |
| HWA deployment | selected HWA checkpoint | none | one endpoint write |
| Full recovery | selected HWA checkpoint | W1 and W2; bias frozen | endpoint write after every BPTT update |
| LoRA recovery | selected HWA checkpoint | A1, B1, A2, B2; base and bias frozen | base written once; every factor rewritten after every BPTT update |

The HWA checkpoint is shared between the two devices. Device seed `17` and
the same per-array RNG order are used for matched full and LoRA deployments.
This is one device realization, not a replicated estimate.

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-hwa-procedure"></a>

## HWA procedure

HWA starts from the selected 96.15% FP32 checkpoint and runs for two epochs:

- W1/W2 learning rates: `0.008`, `0.005`;
- hidden-bias learning rate: `0.015`;
- Gaussian modifier standard deviation: `0.03`;
- scale: independent absolute maximum of every output channel;
- modifier seed: `101`;
- noisy validation: disabled.

For each minibatch, the modifier samples a temporary noisy W1/W2 realization,
runs the forward equilibrium and BPTT gradient computation, restores the clean
master tensors, and applies the optimizer update. This matches the
`ADD_NORMAL_PER_CHANNEL` and `std_dev=0.03` choices in IBM's released Analog
Foundation Models configuration. The DRN study does not copy that LLM's
Gaussian clipping, input/output quantization, optimizer, distillation, or
architecture. It also applies the temporary perturbation to nonnegative
conductances and clamps it to the DRN bounds; IBM applies the modifier to
signed neural-network weights.

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-device-models"></a>

## Device models

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-ibm-analog-foundation-models-pcm"></a>

### IBM Analog Foundation Models PCM

The implementation is an equation-level transcription of IBM's released
`add_PCM_noise` helper:

- nominal mapped maximum: `180` in the helper's unitless fit coordinate;
- exact-zero threshold: `0.6` in the same coordinate;
- low/high polynomial switch: `52.5` in the same coordinate;
- maximum input tile: 512;
- each output channel/tile has its own absolute-maximum normalization;
- polynomial standard deviations retain the helper's FP16 conversion.

This is an endpoint programming-error model, not a pulse model. It has no
retention law or separate read-noise term. Because every channel/tile is
renormalized, it does not define an absolute DRN-to-conductance calibration.
The released signed-weight helper can also produce Gaussian tails below zero or
above 180 in fit units; this positive-conductance DRN clamps the mapped-back
tensor to its configured logical bounds.

The provenance matters: Figures 7a and 9 in the AFM paper show the PCM
signal-to-noise/programming-error relationship on normalized axes. They do
not provide a physical min/max-conductance figure. IBM's released helper names
the constants `GMAX=180` and `OD_TD_DIVERGENCE=52.5`, but assigns no physical
unit to them. The underlying Hermes-chip paper reports a nominal per-device
resistance range of `100 kΩ–10 MΩ`, equivalent to `0.1–10 µS`, while its
weight mapping uses unit-cell maxima of 80 or 160 **ADC counts**. Those
physical/device and ADC-count ranges cannot be identified one-to-one with the
AFM helper's 180 coordinate.

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-cmohfox-reram"></a>

### CMO/HfOx ReRAM

The implementation follows AIHWKit 1.1's `ReRamCMONoiseModel` coefficients:

- measured window: `9` to `88.199997 µS`;
- program-and-verify acceptance fit: `0.2%`;
- programming-noise scale: `1.0`;
- inference age: 1 second;
- relaxation and read-noise scales: `1.0`;
- archived logical calibration: `[0, 1]` maps to
  `[9, 88.199997] µS`, followed by normalized-offset conversion.

Unlike the AFM PCM case, Falcone et al.'s Figure 3 and Supplementary
Figure S5 directly show 35 programmed levels over approximately
`10–90 µS`; the AIHWKit fit uses the slightly more specific
`9–88.199997 µS` endpoints.
Falcone et al. report about 89 set/reset pulses on average to converge at the
selected `0.2%` acceptance range. This experiment samples the fitted residual
endpoint error; it does not replay those 89 pulses or their trajectory.

The last mapping is explicit and nontrivial. It allows a logical zero and a
zero-initialized LoRA output factor only by subtracting the physical `9 µS`
floor on conversion back to the DRN. A literal passive circuit would require
a reference/differential cancellation path; without one, the CMO floor would
be an actual nonzero connection.

This arm uses all three stochastic terms exposed by AIHWKit's CMO class.
Falcone et al.'s own reported MVM simulation omitted read noise while adding
6-bit/8-bit input/output quantization and wire IR drop. This DRN experiment
does the converse for those terms: it enables the class's read-noise law but
does not model converter quantization or IR drop.

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-noisy-recovery-update"></a>

## Noisy recovery update

BPTT always computes its gradient at the current realized device tensors. The
optimizer then:

1. restores a clean digital target tensor;
2. applies the gradient update to that target;
3. clamps the target to its configured logical conductance bounds;
4. samples a fresh program-and-verify endpoint realization;
5. installs that realization for the next minibatch.

Full recovery uses 20 epochs, W1/W2 learning rates `0.008/0.005`, and a frozen
hidden bias. Rank-4 LoRA uses 20 epochs and learning rate `0.01` for each of
A1, B1, A2, and B2. The LoRA base W1/W2 realization is frozen.

This serial closed-loop interpretation deliberately satisfies the requirement
that every post-HWA BPTT update include write noise. It does not claim
constant-time in-situ training: it rewrites a complete trainable array after
every minibatch and omits pulse count, latency, energy, and endurance.
With 60,000 examples, batch size 16, and 20 epochs, each recovery arm performs
75,000 optimizer/write steps. That corresponds to 11.91 billion individual
W1/W2 endpoint programs for full recovery (`158,800` values per step), versus
536.4 million factor endpoint programs for rank-4 LoRA (`7,152` values per
step), before counting verification pulses. These are modeled write events,
not a hardware time or energy estimate.

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-reproducibility"></a>

## Reproducibility

Study configs are under
[`examples/small_drn/mnist_ibm_devices`](../../../../../examples/small_drn/mnist_ibm_devices).
Focused model and resume tests are in
[`tests/test_device_noise_backends.py`](../../../../../tests/test_device_noise_backends.py).

The source bundle staged for the GPU run has SHA-256
`2aedd9fde0cb7799678e57255557d69d56123ba1b8d215e46c0f17579eef36c4`.
The FP32 input checkpoint has SHA-256
`ee7dd6c9936e051e54c38e5167133f72b64624946183b86e349b3e878ebbc27e`.
The archived run configuration calls the PCM fit fields `g_max_us` and
`zero_threshold_us`, reflecting an initial unit interpretation. The current
strict schema corrects those names to `fit_max` and `zero_threshold`; the
numerical values and equations are unchanged. For the same reason, archived
PCM programming-report keys beginning `physical_` and ending `_us` contain
fit-coordinate values, not measured microSiemens. Current runs emit
`fit_target_*` and `fit_realized_*` instead.
The complete raw study, corrected summary, and exact source bundle are local
ignored artifacts under `results/mnist-ibm-pcm-cmo-noisy-recovery/`. An
original remote copy is retained outside the repository; its
machine-specific location is intentionally omitted.

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-primary-sources"></a>

## Primary sources

- [Analog Foundation Models](https://arxiv.org/abs/2505.09663)
- [IBM Analog Foundation Models code, commit 4f4c505](https://github.com/IBM/analog-foundation-models/tree/4f4c505b9d9af43af289788b3a29eeecaeb4e1ab)
- [IBM Hermes 64-core PCM chip](https://arxiv.org/abs/2212.02872)
- [Falcone et al., CMO/HfOx ReRAM](https://arxiv.org/abs/2502.04524)
- [AIHWKit 1.1 ReRAM model source](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/reram.py)

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
