---
id: "exp-001"
title: "Learning from measured RESET"
status: "imported"
hypotheses: []
---

# exp-001 — Learning from measured RESET

## Question and provenance

Learning from measured RESET. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-001-reset.md).

Historical document locations (now references/forwarding pages): [mnist_relu_drn_reset.md](../../../../../docs/mnist_relu_drn_reset.md).

<a id="source-mnist_relu_drn_reset-reset-trained-cohort-a-mnist-drn"></a>

## Historical source: RESET-trained cohort-A MNIST DRN

This exploratory experiment tests whether the original dual-rail DRN can learn
from raw measured ReRAM trajectories without first mapping a digital teacher
into conductances. It is registered as `mnist_relu_drn_reset.v1`.

The frozen teacher is the bias-free `784 -> 50 -> 10` ReLU network. The student
is the bias-free `1568 -> 100 -> 20` DRN with one non-negative memristor on each
physical edge. Its ten logits are the differences between paired output rails.
The fixed readout gain is exactly one.

Every assigned cohort-A virtual device starts at measured pulse index zero. No
teacher weight, affine conductance mapping, initial target write, deadband, or
probabilistic write is used. Training retains the established per-device
global-nearest projection of each digital-shadow update onto its raw measured
pulse-conductance graph. The circuit uses `voltage_amp = 4` and
`current_amp = 0.25`.

Two arms use the same MNIST split, device split, virtual-device assignments,
RESET state, and training budget:

- `teacher_kl` minimizes `KL(softmax(teacher) || softmax(DRN))`; labels are used
  only for metrics.
- `cross_entropy` is the matched hard-label control.

Each arm independently selects its two conductance learning rates from RESET
using the bounded relative-update probe, safety canaries, and 3-by-3 validation
grid. Every cell is restored to the exact pre-probe state before it runs, and
the production training starts from that same state.

The canonical configs are
`examples/mnist_relu_drn_reset/cohort_a_kl.json` and
`examples/mnist_relu_drn_reset/cohort_a_cross_entropy.json`. The campaign is
`campaigns/manifests/mnist_relu_drn_reset.json`. Raw outputs belong below
`results/mnist-relu-drn-reset-exploratory-20260816/`.

The hypothesis is supported when the KL arm reaches at least 94% held-out test
accuracy and is no more than one percentage point below the matched
cross-entropy arm. Test data are read only after validation-based learning-rate
and checkpoint selection. A post-hoc positive logit scale is fitted on the
validation split only to report calibrated test KL; it never changes training,
checkpoint selection, or accuracy.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
