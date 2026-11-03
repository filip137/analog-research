---
id: "exp-004"
title: "MNIST HWA and physical LoRA rank study"
status: "imported"
hypotheses: []
---

# exp-004 — MNIST HWA and physical LoRA rank study

## Question and provenance

MNIST HWA and physical LoRA rank study. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-004-rank-study.md).

Historical document locations (now references/forwarding pages): [mnist_hwa_lora_experiment_setup.md](../../../../../docs/mnist_hwa_lora_experiment_setup.md).

<a id="source-mnist_hwa_lora_experiment_setup-mnist-hwa-and-lora-experiment-setup"></a>

## Historical source: MNIST HWA and LoRA Experiment Setup

Last updated: 2026-07-29

<a id="source-mnist_hwa_lora_experiment_setup-purpose"></a>

## Purpose

This experiment asks two separate questions:

1. Does hardware-aware training (HWA) make a DRN more robust when its trained
   conductances are programmed into noisy ReRAM devices?
2. After deployment, can physical low-rank conductance branches recover some
   of the remaining accuracy without changing the deployed base
   conductances?

The baseline depends on which question is being answered:

- **HWA baseline:** an ordinarily trained FP32 DRN programmed into ReRAM.
- **LoRA baseline:** the HWA-trained DRN after it has been programmed into
  ReRAM.
- **Clean FP32 accuracy:** a reference for the accuracy available before
  device programming. It is not the direct LoRA baseline.

This distinction is important because LoRA never starts from the clean FP32
network. It starts from the already degraded HWA-to-ReRAM checkpoint.

<a id="source-mnist_hwa_lora_experiment_setup-four-arm-comparison"></a>

## Four-arm comparison

The controlled comparison has four arms:

| Arm | Training and deployment path | Role |
| --- | --- | --- |
| a | Ordinary FP32 BPTT, evaluated before device programming | Clean reference |
| b | Ordinary FP32 BPTT, then programmed into ReRAM | HWA baseline |
| c | HWA BPTT, then programmed into the same ReRAM realizations | LoRA baseline |
| d | HWA BPTT, programmed into ReRAM, then adapted with physical LoRA | Recovered model |

Arms b and c use the same device model, conductance calibration, retention
time, and device seed pairs. The ordinary and HWA base trainings share their
initialization, data order, optimizer, solver, and epoch budget. Their
training-time hardware perturbation is the intended difference.

<a id="source-mnist_hwa_lora_experiment_setup-dataset-and-drn"></a>

## Dataset and DRN

- Dataset: full torchvision MNIST
- Training examples: 60,000
- Test examples: 10,000
- Input normalization: mean `0.1307`, standard deviation `0.3`
- Batch size: 16
- Logical network dimensions: `[784, 100, 20]`
- Physical DRN dimensions: `[1568, 100, 20]`
- Nonlinearity: perfect diode
- Input gain: 50
- Numerical precision: FP32

Each of the 784 flattened input pixels is represented by positive and
negative input nodes, giving 1,568 physical input nodes. The 20 output nodes
form differential pairs for the 10 MNIST classes.

The DRN equilibrium is solved with:

- asynchronous coordinate updates;
- 4 iterations in training and inference;
- overrelaxation factor 1.1;
- BPTT through the unrolled DRN iterations.

In the strict experiment schema, BPTT is selected with:

```json
"algorithm": "backprop"
```

<a id="source-mnist_hwa_lora_experiment_setup-stage-1-ordinary-fp32-training"></a>

## Stage 1: ordinary FP32 training

The ordinary base DRN is trained for 20 epochs with:

- W1 learning rate: `0.08`
- W2 learning rate: `0.05`
- hidden-bias learning rate: `0.15`
- no training-time weight modifier

Its selected clean accuracy is:

```text
Ordinary FP32 DRN: 96.150%
```

This is arm a and serves as the clean reference.

<a id="source-mnist_hwa_lora_experiment_setup-stage-2-hardware-aware-training"></a>

## Stage 2: hardware-aware training

The HWA base uses the same model initialization, data seed, batches, solver,
learning rates, and 20-epoch budget as the ordinary base. During training,
an additive normal weight modifier is applied:

```text
standard deviation = 0.0025
modifier seed       = 101
```

The selected HWA checkpoint reaches 96.240% before ReRAM programming. The
small clean difference from ordinary training is not the main HWA result.
The meaningful HWA comparison is made after both checkpoints are programmed
into matched ReRAM realizations.

<a id="source-mnist_hwa_lora_experiment_setup-stage-3-reram-deployment"></a>

## Stage 3: ReRAM deployment

The selected ordinary and HWA checkpoints are programmed after training
using the AIHWKit `ReRamWan2022NoiseModel`.

- Device model: Wan-2022 ReRAM
- AIHWKit version: 1.1.0
- Retention time: 86,400 seconds, or one day
- Device maximum conductance: `40 uS`
- DRN conductance represented by `G_max`: `1.0` for W1 and W2
- Noise scale: `1.0`
- Original W1/W2 seed pairs:
  `17/29`, `31/37`, `41/43`, `47/53`, and `59/61`

The conductance reference is above the largest conductance in both selected
base checkpoints, so no target base conductance saturates at the ReRAM high
bound.

Device programming creates a new named-weights checkpoint containing the
realized ReRAM base conductances. Training does not resume during this
stage. Each programmed checkpoint is evaluated directly on the DRN.

<a id="source-mnist_hwa_lora_experiment_setup-stage-4-physical-lora-recovery"></a>

## Stage 4: physical LoRA recovery

LoRA is added to both dense edges of the already programmed HWA DRN:

```text
input  -> A1 -> rank nodes -> B1 -> hidden
hidden -> A2 -> rank nodes -> B2 -> output
```

The effective network therefore contains three distinct conductance paths:

```text
W1_device in parallel with A1/B1
W2_device in parallel with A2/B2
```

`W1_device`, `W2_device`, `A1`, `B1`, `A2`, and `B2` are separate physical
conductance arrays. They are not combined and clipped as one effective
matrix.

For the primary rank-4 experiment:

- rank: 4 on both dense edges
- factor gain: `0.01`
- A1/A2 bounds: `[1e-7, 1.0]`
- B1/B2 bounds: `[0.0, 1.0]`
- A1/B1/A2/B2 learning rates: `0.01`
- recovery algorithm: BPTT
- recovery epochs: 20

B1 and B2 are initialized to zero. Consequently, inserting the LoRA
branches initially makes no contribution and exactly preserves the
corresponding deployed accuracy.

During recovery:

- the programmed W1 and W2 conductances remain frozen;
- the hidden bias remains frozen;
- only A1, B1, A2, and B2 train;
- each base or factor array is subject to its own conductance bounds.

The LoRA arrays are modeled as physical conductances in the DRN equations,
but their own programming noise, read noise, and retention drift are kept
ideal in this experiment. The Wan-2022 device effects apply to the deployed
base arrays.

<a id="source-mnist_hwa_lora_experiment_setup-artifact-flow"></a>

## Artifact flow

The experiment artifacts are under:

```text
labs/cases/mnist_perfect_diode_hwa_lora_comparison/
```

The main flow is:

```text
fp32_base_run/.../checkpoints/weights.pt
    -> deployed/fp32/seeds_*/weights.pt
    -> deployed_validation/fp32/seeds_*/*

hwa_base_run/.../checkpoints/weights.pt
    -> deployed/hwa/seeds_*/weights.pt
    -> deployed_validation/hwa/seeds_*/*
    -> lora_initial/seeds_*/weights.pt
    -> lora_initial_validation/seeds_*/*
    -> lora_run/seeds_*/*/checkpoints/weights.pt
```

Expansion artifacts are grouped under:

```text
expansion/deployed/
expansion/deployed_validation/
expansion/rank1/
expansion/rank2/
expansion/rank4_new/
expansion/rank8/
```

The full-model control follows the new raw-result convention:

```text
results/mnist-hwa-reram-full-finetune/
  runs/baseline-validation/
  runs/full-finetune/
  analysis/summary.json
```

The selected named-weights artifact for each completed training run is:

```text
checkpoints/weights.pt
```

<a id="source-mnist_hwa_lora_experiment_setup-configuration-files"></a>

## Configuration files

- Ordinary clean base:
  `examples/small_drn/mnist_perfect_diode.json`
- HWA base:
  `labs/cases/mnist_perfect_diode_hwa_lora_comparison/hwa_base.json`
- Physical LoRA recovery:
  `labs/cases/mnist_perfect_diode_hwa_lora_comparison/lora_hwa_deployed.json`
- Full-model post-deployment fine-tuning:
  `examples/small_drn/mnist_hwa_reram_full_finetune.json`
- Device-programming records:
  `labs/cases/mnist_perfect_diode_hwa_lora_comparison/deployed/deployment_manifest.json`

Training and validation use the stable public CLI:

```bash
python -m ebl train \
  --config <config.json> \
  --output-dir <new-run-directory>

python -m ebl validate \
  --config <config.json> \
  --output-dir <new-validation-directory> \
  --weights <explicit-checkpoints/weights.pt>
```

Each command owns a new run directory. Checkpoints are always supplied by an
explicit path; the workflow does not scan for the newest model.

<a id="source-mnist_hwa_lora_experiment_setup-related-records"></a>

## Related records

- [Current research state](../../../../../docs/current_state.md)
- [Experimental manifest](../../../../../docs/experimental_manifest.md)
- [Physical layerwise low-rank recovery](../../../../../docs/passive_layerwise_low_rank_recovery.md)

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
