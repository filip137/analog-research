---
id: "exp-004"
title: "Ten-epoch differential RESET"
status: "imported"
hypotheses: []
---

# exp-004 — Ten-epoch differential RESET

## Question and provenance

Ten-epoch differential RESET. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-004-differential-reset.md).

Dependencies and related controls: [RESET loss × bias × indexing factorial](../results/exp-003-reset-factorial.md).

Historical document locations (now references/forwarding pages): [mnist_relu_drn_reset_differential_10ep.md](../../../../../docs/mnist_relu_drn_reset_differential_10ep.md).

<a id="source-mnist_relu_drn_reset_differential_10ep-differential-reset-trained-mnist-drn-10-epochs"></a>

## Historical source: Differential RESET-trained MNIST DRN (10 epochs)

This exploratory experiment applies the implemented weighted differential-pair
equilibrium to the strongest corrected arm of the recent single-device RESET
factorial. It is registered as `mnist_relu_drn_reset_differential.v1`.

<a id="source-mnist_relu_drn_reset_differential_10ep-controlled-setup"></a>

## Controlled setup

- Student: bias-free dual-rail `[1568, 100, 20]` DRN.
- Synapse: one independently assigned measured `G+`/`G-` pair for every
  physical dual-rail edge.
- Amplification: model-local logical indices with `voltage_amp = 4` and
  `current_amp = 0.25`. The resolved differential stages are `(0, 1)` and
  `(1, 2)`, with forward gains `1` and `4` and post-layer energy metrics `1`
  and `0.0625`.
- Initialization: every physical branch starts at its assigned cohort-A raw
  trace's pulse-zero RESET state. There is no teacher-weight mapping or
  common-window pre-programming.
- Supervision: paired-output squared error against MNIST labels. The frozen
  ReLU teacher is used for agreement and KL diagnostics only.
- Production budget: 10 epochs after bounded learning-rate selection.

The two branches of each logical edge share one selected learning rate. The
relative-update probe computes an element-weighted RMS across both branch
gradients and normalizes it by the combined physical conductance RMS. The same
bounded center reduction, safety canaries, and 3-by-3 candidate grid used by
the RESET factorial then select one rate for each of the two logical layers.

Learning-rate selection and production are separate constructions. Production
is exactly reseeded and rebuilt, and the run fails closed unless device
assignments and the model-local differential stage scales match.

<a id="source-mnist_relu_drn_reset_differential_10ep-scope"></a>

## Scope

This run tests literal dual RESET initialization. It does not guarantee
`G+ = G-` at initialization because independently assigned physical devices
can have different pulse-zero conductances. It therefore differs from the
teacher-initialized differential experiment, which first writes both devices
into a shared reachable window. The common conductance still cancels from
signed transfer only to the extent that the two realized branch baselines
match; it remains present in loading and power in every case.

The result is exploratory, single-seed numerical evidence rather than a
fabricated-circuit claim or a replicate-based uncertainty estimate.

<a id="source-mnist_relu_drn_reset_differential_10ep-versioned-entry-points"></a>

## Versioned entry points

- Config:
  `examples/mnist_relu_drn_reset_differential/cohort_a_paired_squared_error_10ep.json`
- Campaign:
  `campaigns/manifests/mnist_relu_drn_reset_differential_10ep.json`
- Raw output:
  `results/mnist-differential-reram-reset-10ep-20260817-v1/`

The campaign runs training followed by explicit held-out test validation using
the selected named-weights checkpoint.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
