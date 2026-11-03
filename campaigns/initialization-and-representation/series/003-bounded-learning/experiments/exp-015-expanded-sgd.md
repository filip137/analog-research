---
id: "exp-015"
title: "Expanded SGD learning-rate control"
status: "imported"
hypotheses: ["H-015"]
---

# exp-015 — Expanded SGD learning-rate control

## Question and provenance

Expanded SGD learning-rate control. The original SGD search stopped at a monotonically improving boundary. Extending zero-momentum layerwise SGD geometrically to W1 in {0.512,1.024,2.048,4.096} and W2 in {0.02,0.04,0.08} will produce at least 96.0% selected validation accuracy within ten epochs from the fixed per-cell OM-bounds uniform initialization, reducing the prior 1.94-percentage-point Adam-versus-SGD validation gap by at least 1.18 percentage points.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-015-expanded-sgd.md).

Dependencies and related controls: [SGD and Adam within heterogeneous OM cell bounds](../results/exp-014-cell-bounds-optimizers.md).

## Frozen plans

- [Expanded ten-epoch SGD learning-rate search inside heterogeneous IBM OM cell bounds](../../../../../studies/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1.json) — `mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.


## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
