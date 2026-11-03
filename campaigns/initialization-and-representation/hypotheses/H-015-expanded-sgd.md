---
id: "H-015"
title: "Expanded SGD learning-rate control"
---

# H-015 — Expanded SGD learning-rate control

## Provenance and claim

Imported verbatim from the first linked historical JSON plan. This record was reconstructed on 2026-10-02; it is not a newly preregistered hypothesis. The original plan owns its timing, arms and decision thresholds.

The original SGD search stopped at a monotonically improving boundary. Extending zero-momentum layerwise SGD geometrically to W1 in {0.512,1.024,2.048,4.096} and W2 in {0.02,0.04,0.08} will produce at least 96.0% selected validation accuracy within ten epochs from the fixed per-cell OM-bounds uniform initialization, reducing the prior 1.94-percentage-point Adam-versus-SGD validation gap by at least 1.18 percentage points.

## Decision rule and scope

[Original plan](../../../studies/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1.json) contains the completion criteria and analysis plan. Preserve its original numerical gates and evidence regime. [Experiment](../series/003-bounded-learning/experiments/exp-015-expanded-sgd.md) and [result and interpretation](../series/003-bounded-learning/results/exp-015-expanded-sgd.md) record coverage and the scoped outcome.
