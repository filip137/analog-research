---
id: "exp-008"
title: "Pairwise common window versus bounded FP32"
status: "imported"
hypotheses: []
---

# exp-008 — Pairwise common window versus bounded FP32

## Question and provenance

Pairwise common window versus bounded FP32. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-008-common-window-fp32.md).

Historical document locations (now references/forwarding pages): [mnist_single_rail_common_window_vs_fp32.md](../../../../../docs/mnist_single_rail_common_window_vs_fp32.md).

<a id="source-mnist_single_rail_common_window_vs_fp32-single-rail-common-window-initialization-versus-bounded-fp32"></a>

## Historical source: Single-rail common-window initialization versus bounded FP32

<a id="source-mnist_single_rail_common_window_vs_fp32-question"></a>

## Question

Can the existing one-conductance-per-dual-rail-edge MNIST DRN retain a ReLU
teacher initialization when complementary rail conductances are programmed
through a shared reachable window, and how does it compare with a uniform
bounded FP32 network without device-to-device variability?

This is an exploratory, single-seed screen rather than paper-facing evidence.

<a id="source-mnist_single_rail_common_window_vs_fp32-frozen-protocol"></a>

## Frozen protocol

- Teacher, MNIST split, minibatch order, solver, model-local amplification,
  objective, and 10-epoch budget match the earlier initialized single-device
  study.
- The measured arm uses the same raw cohort-A device split and virtual-device
  assignment (`split_seed=42`, `assignment_seed=42`). It still uses exactly
  one conductance per physical dual-rail edge.
- Complementary output-rail conductances share a pairwise reachable window.
  The nominal encoding is centered inside that window so compressed mappings
  retain symmetric programming margin.
- Candidate layer-scale pairs are `(1, 0.125)`, `(1, 0.25)`, `(1, 0.5)`,
  `(1, 0.75)`, and `(1, 1)`. The measured arm selects on the 1,024-example
  calibration subset after global-nearest measured projection. Test data are
  not used for mapping, gain, epoch, or checkpoint selection.
- The FP32 control has the same single-device topology and centered encoding,
  exact float32 updates, uniform bounds of 70--90 uS, and no device assignment
  or measured-state projection. This range approximates cohort A's 70.7 uS
  pairwise baseline and 18.4 uS mean overlap span.

<a id="source-mnist_single_rail_common_window_vs_fp32-monitoring-contract"></a>

## Monitoring contract

- Campaign ID:
  `mnist-single-rail-common-window-vs-fp32-10ep-20260818-v1`
- Manifest:
  `campaigns/manifests/mnist_relu_drn_single_pairwise_common_window_vs_fp32_10ep.json`
- Expected stages: measured train/test and bounded-FP32 train/test, four total.
- Target and launcher: local subprocess campaign on the available CUDA GPU.
- Operational output root:
  `/home/filip/server_code/.codex/worktrees/tiki-taka-lora-integration/results/`
- Progress evidence: initialization metric, one metric record per completed
  epoch, selected named weights, epoch-boundary resume state, fresh-process
  test result, and campaign aggregate.
- Expected cadence: initialization within a few minutes and epoch metrics at
  sub-minute cadence; total runtime approximately ten minutes.
- Safe recovery: rerun the identical campaign with `--resume --fail-fast`.
  Completed stages may be reused only when campaign fingerprints and artifact
  hashes validate.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
