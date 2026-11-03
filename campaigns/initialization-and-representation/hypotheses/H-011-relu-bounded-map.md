---
id: "H-011"
title: "Direct ReLU-to-bounded FP32 mapping"
---

# H-011 — Direct ReLU-to-bounded FP32 mapping

## Provenance and claim

Imported verbatim from the first linked historical JSON plan. This record was reconstructed on 2026-10-02; it is not a newly preregistered hypothesis. The original plan owns its timing, arms and decision thresholds.

A signed 784-50-10 ReLU checkpoint can be compressed directly into the continuous bounded [0.1020408197973068, 1.0] four-device dual-rail DRN with no more than one percentage point of clean test-accuracy loss before any device programming.

## Decision rule and scope

[Original plan](../../../studies/mnist-relu-to-bounded-fp32-map-20260824-v1.json) contains the completion criteria and analysis plan. Preserve its original numerical gates and evidence regime. [Experiment](../series/003-bounded-learning/experiments/exp-011-relu-bounded-map.md) and [result and interpretation](../series/003-bounded-learning/results/exp-011-relu-bounded-map.md) record coverage and the scoped outcome.
