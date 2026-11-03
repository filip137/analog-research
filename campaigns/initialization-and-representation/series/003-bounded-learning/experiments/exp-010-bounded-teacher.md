---
id: "exp-010"
title: "Independently trained bounded DRN reference"
status: "imported"
hypotheses: []
---

# exp-010 — Independently trained bounded DRN reference

## Question and provenance

Independently trained bounded DRN reference. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-010-bounded-teacher.md).

Historical document locations (now references/forwarding pages): [mnist_bounded_drn_teacher.md](../../../../../docs/mnist_bounded_drn_teacher.md).

<a id="source-mnist_bounded_drn_teacher-mnist-bounded-drn-teacher-run"></a>

## Historical source: MNIST bounded-DRN teacher run

<a id="source-mnist_bounded_drn_teacher-monitoring-contract"></a>

## Monitoring contract

- Scientific tier: formal, intended as the frozen teacher/source checkpoint
  for the CMO versus Wan-2022 deployment study.
- Coverage: one bias-free `[1568, 100, 20]` perfect-diode DRN, seed 17,
  trained from scratch for exactly 10 epochs on the 55,000-example MNIST
  training split; clean model selection uses the fixed 5,000-example
  validation split. The 10,000-example MNIST test split is untouched during
  training and will be evaluated after collection.
- Conductance constraint: every weight remains in
  `[9 / 88.199997, 1] = [0.1020408197973068, 1]` from initialization onward.
  Deployment is multiplicative only: CMO uses `88.199997 uS * w`; Wan-2022
  uses `40 uS * w`.
- Frozen source commit: `6aa32237ea5894939af9a9a271b3081fcb983fc9`.
- Frozen source tree: `6e70efc440b4db953d7559c8344d21018842b6f9`.
- Source bundle SHA-256:
  `2af61ccb8ac35ce73ad563a82d42c6e172509e0d7b229eb96b7c22436b449501`.
- Config SHA-256:
  `29fc5094f997fa0670e67c1635f5accc21dde749a940463c955fa4888cdd6040`.
- Akib source:
  `/home/filiposana/staged/mnist_bounded_drn_teacher_6aa32237_v2`.
- Akib result root:
  `/home/filiposana/server_code/results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237`.
- Akib launcher state:
  `/home/filiposana/server_code/results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237.launcher`.
- Combined log: `<launcher state>/job.log`; lifecycle evidence is
  `<launcher state>/exit_code`, `pid`, `started_at`, and `finished_at`.
- Required terminal artifacts: one successful run directory containing
  `result.json`, `metrics.jsonl`, `checkpoints/weights.pt`, and
  `checkpoints/resume.pt`; then a separate clean test-validation run and a
  local copy with matching hashes.
- Progress evidence: launcher/PID and GPU occupancy plus a new epoch metric
  and updated resume checkpoint. Check immediately twice, then every 30
  minutes; diagnose after 45 minutes without semantic artifact growth.
- Expected runtime: approximately 30--90 minutes on the Akib RTX 3080. A
  retry is safe only into a new result and launcher path using the same commit
  and config; do not alter the learning rates, seed, bounds, or epoch budget.

<a id="source-mnist_bounded_drn_teacher-pre-launch-evidence"></a>

## Pre-launch evidence

- Local clean-worktree gates: 117 focused tests passed; legacy `labs/tests`
  passed 42 tests with 1 skipped.
- Local numerical canary: one epoch horizon, two training minibatches and one
  validation minibatch completed; named weights and exact-resume checkpoints
  were written with all conductances inside the configured interval.
- Akib environment: passwordless SSH succeeded; PyTorch 2.5.1 reported CUDA
  available on an NVIDIA GeForce RTX 3080; no compute process occupied the
  GPU at preflight.
- Staging recovery: an initial shallow-bundle clone failed before launch due
  to an omitted parent object. The complete bundle was checked at the `v2`
  path above and preserved the intended commit and tree exactly.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
