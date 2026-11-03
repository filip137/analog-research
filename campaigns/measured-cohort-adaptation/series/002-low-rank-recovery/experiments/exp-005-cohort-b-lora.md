---
id: "exp-005"
title: "Corrected RESET passive LoRA recovery"
status: "imported"
hypotheses: []
---

# exp-005 — Corrected RESET passive LoRA recovery

## Question and provenance

Corrected RESET passive LoRA recovery. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-005-cohort-b-lora.md).

Dependencies and related controls: [Cohort-B transfer and learning-rate controls](../../001-transfer-and-write-policies/results/exp-002-cohort-b-transfer.md), [Fixed-rate programming deadbands](../../001-transfer-and-write-policies/results/exp-003-deadband.md).

Historical document locations (now references/forwarding pages): [measured_cohort_b_lora_recovery.md](../../../../../docs/measured_cohort_b_lora_recovery.md).

<a id="source-measured_cohort_b_lora_recovery-measured-cohort-b-memristor-lora-recovery"></a>

## Historical source: Measured cohort-B memristor LoRA recovery

`measured_cohort_b_lora` is the physical low-rank accuracy-recovery protocol
for a one-hidden-layer MNIST DRN. It combines the
`passive_layerwise_low_rank` model adapter with backpropagation and measured
cohort-B projection.

The loaded cohort-A checkpoint supplies only the two dense base matrices and
hidden bias. On deployment, each base conductance is independently assigned a
virtual cohort-B device curve and written once to the globally nearest
measured conductance. Those base tensors then remain frozen.

Four trainable memristor arrays form two explicit rank-node branches:

```text
input  -- A1 -- rank1 -- B1 -- hidden
hidden -- A2 -- rank2 -- B2 -- output
```

Every element of `A1`, `B1`, `A2`, and `B2` starts at the final measured
RESET-sweep point (pulse index 4999 in the example).
There is no random or ideal floating-point LoRA initialization. The optimizer
keeps a digital target shadow for these four arrays only and projects every
write onto that element's assigned measured curve. Checkpoints retain the
realized frozen base, factor targets, pulse indices, assignments, and optimizer
state.

The base edges retain `voltage_amp=4` and `current_amp=0.25`. The internal
rank branches are passive unity-gain connections. Amplification depth is
derived from the logical DRN topology, so constructing an LR-selection model
before the production model cannot alter the circuit equations.

Run the nested example with an explicit cohort-A base checkpoint and device
file:

```bash
python -m ebl train \
  --config examples/small_drn/measured_cohort_b_lora_mnist.json \
  --base-weights /path/to/cohort-a/checkpoints/weights.pt \
  --device-data data/march_slope_x3_5k.hdf5 \
  --output-dir results/measured-cohort-b-lora
```

The initialization artifact separately records the frozen-base deployment and
proves a last-pulse fraction of one for every LoRA factor. The example uses
per-factor learning rates because the measured reset-state gradient RMS differs
substantially between the four physical arrays.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
