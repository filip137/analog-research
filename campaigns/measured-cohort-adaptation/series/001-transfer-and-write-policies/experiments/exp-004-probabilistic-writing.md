---
id: "exp-004"
title: "Matched probabilistic write policies"
status: "imported"
hypotheses: []
---

# exp-004 — Matched probabilistic write policies

## Question and provenance

Matched probabilistic write policies. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-004-probabilistic-writing.md).

Dependencies and related controls: [Fixed-rate programming deadbands](../results/exp-003-deadband.md), [Corrected RESET passive LoRA recovery](../../002-low-rank-recovery/results/exp-005-cohort-b-lora.md).

Historical document locations (now references/forwarding pages): [measured_cohort_b_finetuning.md](../../../../../docs/measured_cohort_b_finetuning.md).

<a id="source-measured_cohort_b_finetuning-probabilistic-writing"></a>

## Probabilistic writing

`measured_cohort_b` also supports two stochastic programming policies. Both
continue to apply every gradient to the bounded digital shadow and only gate
the subsequent physical programming request:

- `uniform_bernoulli` writes each element independently with a fixed
  `probabilistic_write_probability`.
- `displacement_proportional` writes with probability
  `min(1, displacement / scale)`, where displacement is from the last
  programming target and `scale` is `probabilistic_write_scale_relative`
  times the layer's initial shadow RMS.

Suppressed displacement therefore accumulates rather than being discarded.
Each parameter owns a deterministically seeded random generator, and exact
checkpoints persist its state together with the last programmed shadow.
Probabilistic writing is mutually exclusive with the deterministic deadband.
Reports record configured and realized write probability, programming-request
fraction, pending displacement, and actual pulse-index changes. The nested
example `examples/small_drn/measured_cohort_b_probabilistic_mnist.json`
defaults to the selected uniform policy at p=0.85.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
