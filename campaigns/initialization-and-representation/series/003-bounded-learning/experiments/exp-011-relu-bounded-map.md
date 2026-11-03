---
id: "exp-011"
title: "Direct ReLU-to-bounded FP32 mapping"
status: "imported"
hypotheses: ["H-011"]
---

# exp-011 — Direct ReLU-to-bounded FP32 mapping

## Question and provenance

Direct ReLU-to-bounded FP32 mapping. A signed 784-50-10 ReLU checkpoint can be compressed directly into the continuous bounded [0.1020408197973068, 1.0] four-device dual-rail DRN with no more than one percentage point of clean test-accuracy loss before any device programming.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-011-relu-bounded-map.md).

Dependencies and related controls: [Independently trained bounded DRN reference](../results/exp-010-bounded-teacher.md).

## Frozen plans

- [Direct ReLU compression into the bounded FP32 DRN](../../../../../studies/mnist-relu-to-bounded-fp32-map-20260824-v1.json) — `mnist-relu-to-bounded-fp32-map-20260824-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.


## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
