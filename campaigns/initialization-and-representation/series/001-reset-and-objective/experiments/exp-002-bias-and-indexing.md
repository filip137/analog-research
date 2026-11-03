---
id: "exp-002"
title: "Bias, loss and historical amplifier indexing"
status: "imported"
hypotheses: []
---

# exp-002 — Bias, loss and historical amplifier indexing

## Question and provenance

Bias, loss and historical amplifier indexing. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-002-bias-and-indexing.md).

Dependencies and related controls: [Raw and isotonic cohort-A feasibility](../../../../measured-cohort-adaptation/series/001-transfer-and-write-policies/results/exp-001-cohort-a.md).

Historical document locations (now references/forwarding pages): [mnist_relu_drn_reset_bias.md](../../../../../docs/mnist_relu_drn_reset_bias.md).

<a id="source-mnist_relu_drn_reset_bias-bias-inclusive-reset-loss-control"></a>

## Historical source: Bias-inclusive RESET loss control

This exploratory study answers whether the archived approximately 95% cohort-A
MNIST result was reproduced simply by replacing its paired squared-error loss
with KL divergence. It was not. During the comparison, the archived result was
also traced to a process-global amplifier-index artifact, so it is retained as
an explicit provenance control rather than evidence for the intended circuit.

<a id="source-mnist_relu_drn_reset_bias-two-distinct-amplifier-semantics"></a>

## Two distinct amplifier semantics

The corrected experiment is `mnist_relu_drn_reset_bias.v1`. It assigns the
three DRN layers the logical indices 0, 1, and 2 every time the model is built.
This is the intended interpretation of `voltage_amp = 4` and
`current_amp = 0.25`.

The historical code inferred amplifier exponents from process-global names
such as `Layer_3`. Its LR selector built one stack (`Layer_0..2`), destroyed
it, and rebuilt the production stack (`Layer_3..5`). Consequently:

- LR selection used dense edges 0→1 and 1→2;
- production training used dense edges 3→4 and 4→5;
- held-out validation in a fresh process again used 0→1 and 1→2.

The archived training and evaluation circuits therefore differed even though
the JSON still said `voltage_amp = 4` and `current_amp = 0.25`. The explicit
historical replay is registered separately as
`mnist_relu_drn_reset_bias_legacy.v1`, with
`amplification_indexing = legacy_process_global`. Checkpoint metadata and
result artifacts record the resolved indices so this behavior cannot be used
silently.

<a id="source-mnist_relu_drn_reset_bias-controlled-setup"></a>

## Controlled setup

All three historical-control arms use the same single-device dual-rail
`[1568, 100, 20]` DRN:

- one non-negative measured ReRAM conductance per physical edge;
- one trainable digital bias on the 100-node hidden layer and no output bias;
- paired class score `s_k = y_(2k) - y_(2k+1)`;
- raw measured cohort-A traces with deterministic virtual-device assignment;
- global nearest-state projection after every minibatch;
- every conductance initialized at pulse index zero and the bias initialized at
  exact zero;
- `input_gain = 100`, four asynchronous iterations, and overrelaxation `1.1`;
- 20 production epochs with a validation-selected checkpoint.

Each objective independently runs the same bounded relative-update LR probe,
safety canaries, and 3×3 validation grid from an exactly restored RESET state.
The teacher, measured-device assignments, RESET fingerprint, and initial
metrics are identical across arms. Test data are used only after selection.

The objectives are:

- paired squared error:
  `0.5 * sum_k((s_k - one_hot(label)_k)^2)`, exactly the archived cost;
- cross-entropy on the hard label and paired scores;
- `KL(softmax(teacher) || softmax(s))` from the frozen bias-free
  `784 -> 50 -> 10` ReLU teacher.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
