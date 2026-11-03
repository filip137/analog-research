---
id: "exp-003"
title: "Fixed-rate programming deadbands"
status: "imported"
hypotheses: []
---

# exp-003 — Fixed-rate programming deadbands

## Question and provenance

Fixed-rate programming deadbands. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-003-deadband.md).

Dependencies and related controls: [Cohort-B transfer and learning-rate controls](../results/exp-002-cohort-b-transfer.md).

Historical document locations (now references/forwarding pages): [measured_cohort_b_finetuning.md](../../../../../docs/measured_cohort_b_finetuning.md).

<a id="source-measured_cohort_b_finetuning-accumulated-programming-deadband"></a>

## Accumulated programming deadband

`measured_cohort_b` optionally separates digital gradient accumulation from
device programming. With
`programming_deadband_mode: accumulated_shadow_relative_rms`, every gradient
still updates the bounded digital shadow, but an individual device is
reprogrammed only after the absolute shadow displacement from its last
programming target reaches its layer threshold. The threshold is fixed at
initialization as `programming_deadband_relative` times that layer's initial
shadow RMS. After a programming request, the current shadow becomes the new
accumulation reference even when global-nearest projection selects the same
pulse index.

This is an accumulated deadband rather than gradient clipping: sub-threshold
updates are retained and can trigger a later write. Checkpoints persist both
the shadow at the last programming request and the absolute per-layer
threshold, so pending updates survive exact resume. Reports distinguish the
fraction of element-steps that request programming from the fraction that
actually change the selected pulse index.

The example
`examples/small_drn/measured_cohort_b_threshold_mnist.json` fixes the learning
rates selected by the 2026-08-14 cohort-B search and uses a relative deadband
of 0.003. This isolates deadband behavior from learning-rate selection; sweep
variants should change only `programming_deadband_relative` and write to a
new result root.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
