---
id: "exp-003"
title: "RESET loss × bias × indexing factorial"
status: "imported"
hypotheses: []
---

# exp-003 — RESET loss × bias × indexing factorial

## Question and provenance

RESET loss × bias × indexing factorial. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-003-reset-factorial.md).

Dependencies and related controls: [Bias, loss and historical amplifier indexing](../results/exp-002-bias-and-indexing.md).

Historical document locations (now references/forwarding pages): [mnist_relu_drn_reset_factorial.md](../../../../../docs/mnist_relu_drn_reset_factorial.md).

<a id="source-mnist_relu_drn_reset_factorial-controlled-reset-factorial"></a>

## Historical source: Controlled RESET Factorial

<a id="source-mnist_relu_drn_reset_factorial-question"></a>

## Question

The historical 95.97% test result and the newer 85.52% teacher-KL result do
not isolate a single cause. They jointly changed the hidden digital bias,
supervision loss, amplifier indexing, and minibatch input-reset behavior.
Consequently, neither result can by itself establish whether the historical
number is optimistic or the newer number is pessimistic.

<a id="source-mnist_relu_drn_reset_factorial-design"></a>

## Design

`mnist_relu_drn_reset_factorial.v1` is a closed 2 x 2 x 2 experiment:

- hidden digital bias: disabled or enabled;
- supervision: historical paired-output squared error or teacher KL;
- amplifier indexing: model-local logical indices or the archived
  process-global behavior.

All eight arms use the same frozen teacher checkpoint (SHA-256
`42b0526c4a433057b0ad0f09a7b36ede64afd7d47a3a8985724cac0e46358d54`),
the same cohort-A device source (SHA-256
`207b143fc1a63710c944db74bc573f16b616ce50e22c6605c12068bf6a3f230d`),
seed 42, `voltage_amp = 4`, and `current_amp = 0.25`. Each arm explicitly
resets input state between minibatches and performs the same reseed/rebuild
between learning-rate selection and production training. Thus input-state
carryover and construction lifecycle are controls rather than hidden proxies
for the bias factor.

The legacy-index arms intentionally train the rebuilt production model with
resolved edges 3->4 and 4->5. Logical arms remain 0->1 and 1->2 across both
constructions. Fresh-process validation resolves 0->1 and 1->2 in either
case, matching how the archived indexing defect affected training but not a
fresh deployment process.

Execution is parallelized over the local RTX 3090, Trex RTX 5090, and Akib
RTX 3080. Each host receives complementary factorial cells, so both levels of
bias, loss, and indexing occur equally often within that host. This blocks a
host-wide accuracy offset out of every main-effect contrast. With one run per
cell, host-by-interaction effects remain inseparable from factorial
interactions; the cross-host reduced smoke is a compatibility gate rather than
an uncertainty estimate.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
