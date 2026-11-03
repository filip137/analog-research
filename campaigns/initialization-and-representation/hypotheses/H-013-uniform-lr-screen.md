---
id: "H-013"
title: "Uniform-initialization learning-rate screen"
---

# H-013 — Uniform-initialization learning-rate screen

## Provenance and claim

Imported verbatim from the first linked historical JSON plan. This record was reconstructed on 2026-10-02; it is not a newly preregistered hypothesis. The original plan owns its timing, arms and decision thresholds.

At least one predeclared layerwise schedule with W1 learning rate in {0.128,0.256,0.512} and W2 learning rate in {0.01,0.02} will achieve at least 90% validation accuracy after ten epochs from the fixed full-range uniform initialization, improving by at least five percentage points over the 83.74% validation result obtained with inherited learning rates [0.008,0.005].

## Decision rule and scope

[Original plan](../../../studies/mnist-bounded-drn-uniform-learning-rate-screen-20260826-v1.json) contains the completion criteria and analysis plan. Preserve its original numerical gates and evidence regime. [Experiment](../series/003-bounded-learning/experiments/exp-013-uniform-lr-screen.md) and [result and interpretation](../series/003-bounded-learning/results/exp-013-uniform-lr-screen.md) record coverage and the scoped outcome.
