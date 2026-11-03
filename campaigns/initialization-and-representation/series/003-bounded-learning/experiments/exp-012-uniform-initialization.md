---
id: "exp-012"
title: "Full-range uniform bounded initialization"
status: "imported"
hypotheses: ["H-012"]
---

# exp-012 — Full-range uniform bounded initialization

## Question and provenance

Full-range uniform bounded initialization. Changing only the bounded DRN initialization to independent uniform draws over the complete permitted conductance interval [0.1020408197973068, 1.0] will still produce at least 90% selected validation accuracy after ten epochs of otherwise unchanged clean FP32 BPTT training.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-012-uniform-initialization.md).

## Frozen plans

- [Bounded DRN training from a full-range uniform initialization](../../../../../studies/mnist-bounded-drn-uniform-initialization-20260826-v1.json) — `mnist-bounded-drn-uniform-initialization-20260826-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.


## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
