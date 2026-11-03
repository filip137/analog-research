---
id: "exp-006"
title: "Ten-epoch teacher-initialized differential control"
status: "imported"
hypotheses: []
---

# exp-006 — Ten-epoch teacher-initialized differential control

## Question and provenance

Ten-epoch teacher-initialized differential control. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-006-differential-ten-epoch.md).

Dependencies and related controls: [Teacher-initialized measured DRN distillation](../results/exp-005-teacher-distillation.md), [Ten-epoch teacher-initialized single-conductance control](../results/exp-007-single-ten-epoch.md).

Historical document locations (now references/forwarding pages): [mnist_relu_drn_initialized_differential_10ep.md](../../../../../docs/mnist_relu_drn_initialized_differential_10ep.md).

<a id="source-mnist_relu_drn_initialized_differential_10ep-teacher-initialized-differential-reram-mnist-fine-tuning-10-epochs"></a>

## Historical source: Teacher-initialized differential ReRAM MNIST fine-tuning (10 epochs)

This exploratory follow-up measures the implemented differential `G+ - G-`
student after teacher mapping and exactly ten epochs of measured-trace KL
fine-tuning. It uses the registered `mnist_relu_drn_kd.v1` experiment.

<a id="source-mnist_relu_drn_initialized_differential_10ep-controlled-setup"></a>

## Controlled setup

- Teacher: the frozen bias-free `784 -> 50 -> 10` ReLU checkpoint used by the
  RESET studies. Its tensors are exactly equal to the checkpoint used by the
  earlier 20-epoch initialized-differential study.
- Student: bias-free dual-rail `[1568, 100, 20]` DRN with one independently
  assigned measured cohort-A `G+`/`G-` pair per physical edge.
- Mapping: search the fixed non-clipping layer-scale grid, map the teacher into
  each pair's shared affine reachable window, and fit one positive readout gain
  on the 1,024-example calibration subset.
- Programming: begin each device at raw pulse-zero RESET, perform exactly one
  global-nearest teacher-target write, then project fine-tuning targets onto
  the assigned raw trace.
- Fine-tuning: pure `KL(teacher || student)`, seed 42, batch size 16, calibrated
  differential learning rates `[4.2e-10, 1.14e-12]`, and exactly 10 epochs.
- Circuit scaling: model-local logical indexing, `voltage_amp = 4`, and
  `current_amp = 0.25`.

The campaign invokes training and then a separate held-out test validation of
the selected named-weights checkpoint. Checkpoints record the resolved
model-local differential stages and measured-device assignments, and
validation fails closed if the numerical topology or teacher provenance does
not match.

<a id="source-mnist_relu_drn_initialized_differential_10ep-entry-points"></a>

## Entry points

- Config:
  `examples/mnist_relu_drn/measured_raw_differential_finetune_10ep.json`
- Campaign:
  `campaigns/manifests/mnist_relu_drn_initialized_differential_10ep.json`
- Raw output:
  `results/mnist-differential-reram-initialized-finetune-10ep-20260817-v1/`

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
