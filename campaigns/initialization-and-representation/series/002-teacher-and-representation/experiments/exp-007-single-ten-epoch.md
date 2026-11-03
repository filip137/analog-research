---
id: "exp-007"
title: "Ten-epoch teacher-initialized single-conductance control"
status: "imported"
hypotheses: []
---

# exp-007 — Ten-epoch teacher-initialized single-conductance control

## Question and provenance

Ten-epoch teacher-initialized single-conductance control. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-007-single-ten-epoch.md).

Dependencies and related controls: [Teacher-initialized measured DRN distillation](../results/exp-005-teacher-distillation.md), [Ten-epoch teacher-initialized differential control](../results/exp-006-differential-ten-epoch.md).

Historical document locations (now references/forwarding pages): [mnist_relu_drn_initialized_single_10ep.md](../../../../../docs/mnist_relu_drn_initialized_single_10ep.md).

<a id="source-mnist_relu_drn_initialized_single_10ep-teacher-initialized-single-conductance-reram-mnist-fine-tuning-10-epochs"></a>

## Historical source: Teacher-initialized single-conductance ReRAM MNIST fine-tuning (10 epochs)

This exploratory control measures the original one-conductance-per-physical-
edge student after ReLU-teacher programming and exactly ten epochs of
measured-trace KL fine-tuning. It uses the registered
`mnist_relu_drn_kd.v1` experiment.

<a id="source-mnist_relu_drn_initialized_single_10ep-controlled-setup"></a>

## Controlled setup

- Teacher: the same frozen bias-free `784 -> 50 -> 10` ReLU checkpoint used by
  the initialized differential run.
- Student: bias-free dual-rail `[1568, 100, 20]` DRN with one independently
  assigned measured cohort-A conductance per physical edge.
- Mapping: lift each signed teacher matrix into non-negative dual rails, search
  the same non-clipping layer-scale grid, map every target to the assigned
  device's affine trace range, and fit one positive readout gain on the same
  1,024-example calibration subset.
- Programming: begin each device at raw pulse-zero RESET, perform exactly one
  global-nearest teacher-target write, then project fine-tuning targets onto
  that assigned raw trace.
- Fine-tuning: pure `KL(teacher || student)`, seed 42, batch size 16,
  calibrated one-conductance learning rates `[2.1e-10, 5.7e-13]`, and exactly
  10 epochs.
- Circuit scaling: model-local logical indexing, `voltage_amp = 4`, and
  `current_amp = 0.25`.

The differential arm uses twice these learning rates because its effective
weight update is the difference of two branch updates. Thus the declared rates
are gradient-equivalent in the ideal-device limit.

<a id="source-mnist_relu_drn_initialized_single_10ep-entry-points"></a>

## Entry points

- Config: `examples/mnist_relu_drn/measured_raw_single_finetune_10ep.json`
- Campaign: `campaigns/manifests/mnist_relu_drn_initialized_single_10ep.json`
- Raw output:
  `results/mnist-single-reram-initialized-finetune-10ep-20260817-v1/`

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
