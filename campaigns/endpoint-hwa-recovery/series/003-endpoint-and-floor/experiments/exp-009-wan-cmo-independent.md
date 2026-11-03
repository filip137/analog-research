---
id: "exp-009"
title: "Independently initialized CMO/Wan comparison"
status: "imported"
hypotheses: []
---

# exp-009 — Independently initialized CMO/Wan comparison

## Question and provenance

Independently initialized CMO/Wan comparison. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-009-wan-cmo-independent.md).

Dependencies and related controls: [PCM/CMO repeated noisy recovery](../results/exp-006-pcm-cmo-noisy.md).

Historical document locations (now references/forwarding pages): [mnist_wan_cmo_head_to_head.md](../../../../../docs/mnist_wan_cmo_head_to_head.md).

<a id="source-mnist_wan_cmo_head_to_head-mnist-cmo-versus-wan-2022-head-to-head"></a>

## Historical source: MNIST CMO versus Wan-2022 head-to-head

<a id="source-mnist_wan_cmo_head_to_head-matched-protocol"></a>

## Matched protocol

- DRN: `[1568, 100, 20]`, perfect diode, four asynchronous coordinate
  iterations, input gain 50, voltage amplification 4, and current
  amplification 0.25.
- Ordinary FP32 source: 96.15% test accuracy.
- HWA: two BPTT epochs from the FP32 source with additive Gaussian parameter
  noise, output-channel absolute-maximum scaling, `std_dev=0.03`, and seed
  101. The clean HWA checkpoint is 95.67%.
- Device seed: 17; device age: 86,400 seconds.
- Full BPTT: ten epochs, W1/W2 learning rates `0.008/0.005`, frozen hidden
  bias, and device seed 17. Gradients are evaluated at the current realized
  state; the digital shadow target is updated and both dense arrays are then
  freshly reprogrammed after every minibatch.
- Checkpoint selection: minimum validation mean cost. Under this MNIST
  configuration, `validation` resolves to the held-out test split, so the
  reported endpoint is test-selected and should not be treated as an unbiased
  generalization estimate.

The physical affine maps retain each finite floor in the DRN equations:

```text
CMO:      G_target = 9 + 79.199997 w  uS;  effective DRN G = G_realized / 88.199997
Wan-2022: G_target = 1 + 39 w         uS;  effective DRN G = G_realized / 40
```

This is one device per nonnegative DRN edge. It is not NeuRRAM's signed
two-device differential encoding. It is also an endpoint program-and-verify
experiment, not a Tiki-Taka pulse-update experiment.

<a id="source-mnist_wan_cmo_head_to_head-provenance-and-exclusions"></a>

## Provenance and exclusions

- Ordinary FP32 weights SHA-256:
  `ee7dd6c9936e051e54c38e5167133f72b64624946183b86e349b3e878ebbc27e`.
- Shared HWA weights SHA-256:
  `0055b685b1ea573dabfde127ec9a8be772bf6ce18525ab1e5db6b0d7d037afed`.
- Exact ten-epoch source commit:
  `a7bfac494b4834efc625e25ebc44f787772422cd` (clean worktree).
- Selected CMO weights SHA-256:
  `f24e0faedd342140e491aebdbec4b1330fb1112a13371cac6afdc309eb3e4ccd`.
- Selected Wan weights SHA-256:
  `bb03f4ed6dedd82b9cd48820b8fc2fcbfdcbc4b0dc2e233ebc37a03a60f37356`.
- Local raw-artifact copy: `results/mnist_wan_cmo_head_to_head_seed17_10ep/`
  (ignored by Git).

The earlier sequential 20-epoch campaign was intentionally terminated after
the user reduced the budget. Its interrupted CMO checkpoint had already
crossed into an eleventh completed epoch and is excluded from every headline
number above.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
