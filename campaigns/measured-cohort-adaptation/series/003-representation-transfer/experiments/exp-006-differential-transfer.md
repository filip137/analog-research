---
id: "exp-006"
title: "Differential A-to-B adaptation"
status: "imported"
hypotheses: []
---

# exp-006 — Differential A-to-B adaptation

## Question and provenance

Differential A-to-B adaptation. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-006-differential-transfer.md).

Dependencies and related controls: [Ten-epoch teacher-initialized differential control](../../../../initialization-and-representation/series/002-teacher-and-representation/results/exp-006-differential-ten-epoch.md).

Historical document locations (now references/forwarding pages): [mnist_differential_cohort_ab_adaptation.md](../../../../../docs/mnist_differential_cohort_ab_adaptation.md).

<a id="source-mnist_differential_cohort_ab_adaptation-differential-reram-cohort-a-to-b-adaptation"></a>

## Historical source: Differential ReRAM cohort A-to-B adaptation

Status: complete (2026-08-17)

<a id="source-mnist_differential_cohort_ab_adaptation-question"></a>

## Question

Does adapting differential ReRAM conductances on a held-out hardware cohort
recover performance that is lost when a cohort-A model is deployed onto
cohort-B devices?

The A/B cohorts here are **measured device-trace cohorts**, not disjoint MNIST
example cohorts. The deterministic split contains 317 physical cells / 633
traces in A and 317 physical cells / 632 traces in B. The active trace sets are
disjoint.

<a id="source-mnist_differential_cohort_ab_adaptation-frozen-protocol"></a>

## Frozen protocol

- Differential `G+ / G-` encoding, no biases.
- Model-local amplifier semantics: `voltage_amp=4.0`, `current_amp=0.25`.
- Cohort A: teacher mapping into paired common device windows, followed by 10
  epochs of measured-curve training.
- Cohort B receives the selected cohort-A checkpoint through two transfer
  rules using identical B virtual-device assignments:
  - `literal`: project each loaded A branch conductance independently onto its
    B curve;
  - `paired_affine_common_window`: map both branches into their shared
    per-pair B window before projection.
- A positive scalar readout gain is fitted on the 1,024-example calibration
  set after B deployment and then frozen. The source gain and recalibrated
  deployment metrics are both recorded.
- Each transfer rule has a matched zero-learning-rate deployment control and a
  10-epoch B adaptation run. Control and adaptation initial validation metrics
  and conductance tensors matched exactly.
- Both B adaptation arms reuse the cohort-A learning rates
  `(4.2e-10, 1.14e-12)`, seed, data order, solver, and epoch budget; there is no
  B-specific learning-rate search.
- Selection uses validation `KL(teacher || student)`. Every reported test
  endpoint is evaluated by a separate `ebl validate` process on 10,000 MNIST
  test examples.

<a id="source-mnist_differential_cohort_ab_adaptation-reproducibility"></a>

## Reproducibility

- Branch: `codex/mnist-differential-cohort-ab-adaptation`
- Implementation commit: `4ef5cb8b`
- Campaign manifest:
  `campaigns/manifests/mnist_relu_drn_differential_cohort_ab_10ep.json`
- Raw ignored artifacts:
  `results/mnist-differential-reram-cohort-ab-adaptation-10ep-20260817-v1/`
- Terminal state: 10/10 stages complete; a resume/reuse pass revalidated all
  recorded result and artifact hashes.
- Verification: 492 repository tests passed (2 skipped), 42 legacy
  `labs/tests` passed (1 skipped), public describe checks passed, and GPU smoke
  plus exact-resume checks completed.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
