---
id: "exp-014"
title: "SGD and Adam within heterogeneous OM cell bounds"
status: "imported"
hypotheses: ["H-014"]
---

# exp-014 — SGD and Adam within heterogeneous OM cell bounds

## Question and provenance

SGD and Adam within heterogeneous OM cell bounds. For a bias-free four-cell perfect-diode DRN initialized independently and uniformly inside each seed-84001 IBM OM cell's effective controller interval, array-specific bounds-only FP32 Adam will reach at least 97% selected validation accuracy within ten epochs and outperform the validation-selected bounds-only FP32 SGD schedule.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-014-cell-bounds-optimizers.md).

Dependencies and related controls: [Full-range uniform bounded initialization](../results/exp-012-uniform-initialization.md).

## Frozen plans

- [Ten-epoch FP32 SGD and Adam references inside heterogeneous IBM OM cell bounds](../../../../../studies/mnist-om-range-uniform-fp32-optimizer-reference-20260826-v1.json) — `mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.


## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
