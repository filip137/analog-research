---
id: "H-012"
title: "Full-range uniform bounded initialization"
---

# H-012 — Full-range uniform bounded initialization

## Provenance and claim

Imported verbatim from the first linked historical JSON plan. This record was reconstructed on 2026-10-02; it is not a newly preregistered hypothesis. The original plan owns its timing, arms and decision thresholds.

Changing only the bounded DRN initialization to independent uniform draws over the complete permitted conductance interval [0.1020408197973068, 1.0] will still produce at least 90% selected validation accuracy after ten epochs of otherwise unchanged clean FP32 BPTT training.

## Decision rule and scope

[Original plan](../../../studies/mnist-bounded-drn-uniform-initialization-20260826-v1.json) contains the completion criteria and analysis plan. Preserve its original numerical gates and evidence regime. [Experiment](../series/003-bounded-learning/experiments/exp-012-uniform-initialization.md) and [result and interpretation](../series/003-bounded-learning/results/exp-012-uniform-initialization.md) record coverage and the scoped outcome.
