---
id: "exp-002"
title: "Cohort-B transfer and learning-rate controls"
status: "imported"
hypotheses: []
---

# exp-002 — Cohort-B transfer and learning-rate controls

## Question and provenance

Cohort-B transfer and learning-rate controls. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-002-cohort-b-transfer.md).

Dependencies and related controls: [Raw and isotonic cohort-A feasibility](../results/exp-001-cohort-a.md), [RESET loss × bias × indexing factorial](../../../../initialization-and-representation/series/001-reset-and-objective/results/exp-003-reset-factorial.md), [Corrected RESET passive LoRA recovery](../../002-low-rank-recovery/results/exp-005-cohort-b-lora.md).

Historical document locations (now references/forwarding pages): [measured_cohort_b_finetuning.md](../../../../../docs/measured_cohort_b_finetuning.md).

<a id="source-measured_cohort_b_finetuning-measured-cohort-b-deployment-and-fine-tuning"></a>

## Historical source: Measured cohort-B deployment and fine-tuning

This exploratory phase starts from the explicitly selected named-weights
checkpoint produced by raw cohort-A training. It tests transfer to the other
half of the physical devices without allowing cohort-A traces into the
cohort-B numerical path.

<a id="source-measured_cohort_b_finetuning-deployment-semantics"></a>

## Deployment semantics

- The HDF5 screening, physical-cell split, seeds, and conductance conversion
  are identical to cohort A. Cohort B contains 632 formed traces from 317
  physical cells for the current dataset.
- Each dense weight receives a deterministic convex interpolation of two
  distinct cohort-B traces. The cohort-specific assignment seed namespace
  makes this mapping independent of cohort A.
- The loaded cohort-A conductance is the desired deployment target. It is
  clamped to the model bounds and globally projected to the nearest point on
  the assigned raw cohort-B curve; ties select the lower pulse index.
- The desired loaded value remains the initial digital shadow while the live
  model receives the projected cohort-B conductance. This matches the
  accumulated desired-update semantics used during cohort-A training.
- Hidden biases remain digital and are loaded unchanged.
- Validation is measured once before remapping and once immediately after
  remapping. Both use only the held-out 5,000-example validation split.

This remains a global-nearest study, not a local or directional sequential
pulse model. The deployment report therefore records target-to-realized
conductance error and pulse-index statistics so that optimistic long jumps are
visible rather than hidden.

<a id="source-measured_cohort_b_finetuning-fine-tuning-protocol"></a>

## Fine-tuning protocol

Learning rates are selected again after the cohort-B remap. The full
bounded-relative-update probe, 640-minibatch canaries, 3x3 restarted
three-epoch candidate grid, safety gates, and fresh 20-epoch production run
are the same as for cohort A. Every restart reloads the same named cohort-A
checkpoint and repeats the deterministic cohort-B deployment before any
gradient update.

<a id="source-measured_cohort_b_finetuning-command"></a>

## Command

```bash
python -m ebl train \
  --config examples/small_drn/measured_cohort_b_raw_mnist.json \
  --output-dir results/measured-cohort-b-raw-mnist-exploratory-20260814-v1 \
  --weights results/measured-cohort-a-raw-mnist-exploratory-20260814-v1/<run-id>/checkpoints/weights.pt \
  --device-data data/march_slope_x3_5k.hdf5
```

New cohort-B runs require `--weights`; exact continuation requires
`--resume`. The two initialization sources remain mutually exclusive.

<a id="source-measured_cohort_b_finetuning-exploratory-feasibility-gate"></a>

## Exploratory feasibility gate

Treat transfer as viable when the run completes, post-deployment validation
is at least 85%, selected fine-tuned validation is at least 90%, held-out test
accuracy is within five percentage points of raw cohort A, and the late
fine-tuning curve drops by less than five points from its peak. Projection
diagnostics still require physical interpretation even when these accuracy
gates pass.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
