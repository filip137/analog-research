---
id: "exp-010"
title: "Teacher-initialized CMO/Wan comparison"
status: "imported"
hypotheses: []
---

# exp-010 — Teacher-initialized CMO/Wan comparison

## Question and provenance

Teacher-initialized CMO/Wan comparison. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-010-wan-cmo-teacher.md).

Dependencies and related controls: [Independently initialized CMO/Wan comparison](../results/exp-009-wan-cmo-independent.md), [Teacher-initialized measured DRN distillation](../../../../initialization-and-representation/series/002-teacher-and-representation/results/exp-005-teacher-distillation.md).

Historical document locations (now references/forwarding pages): [mnist_wan_cmo_teacher_initialized.md](../../../../../docs/mnist_wan_cmo_teacher_initialized.md).

<a id="source-mnist_wan_cmo_teacher_initialized-teacher-initialized-cmo-versus-wan-2022-endpoint-comparison"></a>

## Historical source: Teacher-initialized CMO versus Wan-2022 endpoint comparison

Last updated: 2026-08-20

<a id="source-mnist_wan_cmo_teacher_initialized-exact-initialization"></a>

## Exact initialization

The source checkpoint is a frozen FP32 ReLU teacher with test accuracy
`97.36%`. Its signed matrices are lifted directly into a bias-free,
single-conductance dual-rail DRN with dimensions `[1568, 100, 20]`. Positive
and negative teacher coefficients are placed on their corresponding rails;
there is no DRN optimization before the ideal-map row.

The mapping search selected layer scale fractions `(1.0, 0.125)` in the
shared logical DRN window `0..110 uS` and a positive output gain of
`0.630957`. The immediate mapped checkpoint has SHA-256
`25cdfcbc8fe6dc67e66ab8b847f5ddfe25bd691f0b5be052cec91c0983a45899`.
This mapped checkpoint, rather than the older independently trained
`96.15%` DRN, is the source of both direct device-deployment arms.

The DRN uses `input_gain=100`, `voltage_amp=4`, `current_amp=0.25`, four
asynchronous coordinate iterations, perfect-diode nonlinearities, no learned
bias, and model-local amplification indices. Mapping-scale and output-gain
selection use only the validation data.

<a id="source-mnist_wan_cmo_teacher_initialized-hwa-stage"></a>

## HWA stage

HWA continues from the teacher-mapped checkpoint for two epochs using pure
teacher KL. During each minibatch, temporary independent Gaussian noise is
added to every dense output channel with

```text
sigma_channel = 0.03 * max(abs(channel)).
```

Gradients are computed at that noisy realization; the clean master is then
restored and updated. Validation is noise-free. Epoch 0 was selected by
validation KL (`0.002550`, `97.18%` validation accuracy); epoch 1 was slightly
worse. The selected HWA checkpoint has SHA-256
`e69c10cf560f9082c148ec030c06b5c4141aa1856977a377387caf4d00ab2f18`.
This modifier is a generic IBM-style HWA approximation, not a fit to either
endpoint model.

<a id="source-mnist_wan_cmo_teacher_initialized-physical-endpoint-mappings-and-noise"></a>

## Physical endpoint mappings and noise

The finite conductance offsets are retained in the DRN equations. They are
not subtracted on conversion back from the device:

```text
CMO target [uS] = 9 + (88.199997 - 9) * G_DRN / 110 uS
Wan target [uS] = 1 + (40 - 1) * G_DRN / 110 uS

G_effective = G_realized / G_max * 110 uS
```

Consequently, a logical zero becomes an effective `11.22449 uS` CMO edge or
`2.75 uS` Wan edge. Both models use one-day age and device seed 17.
This one-day CMO realization is not interchangeable with earlier one-second
CMO deployment rows.

- CMO samples the AIHWKit CMO/HfOx program-error fit at `0.2%` acceptance,
  followed by the fitted relaxation and read-error terms at one day. The
  result is clamped to `9..88.199997 uS`.
- Wan samples additive Gaussian endpoint error whose conductance-dependent
  standard deviation is the fourth-order one-day Wan-2022 fit, then clamps
  the result to `1..40 uS`. It has no separate read-noise term.

Each endpoint write produces a stored conductance tensor. Noise is not
resampled on every inference. During post-deployment BPTT, however, a fresh
endpoint realization is sampled after every minibatch update.

<a id="source-mnist_wan_cmo_teacher_initialized-post-deployment-bptt"></a>

## Post-deployment BPTT

Both BPTT arms start from the exact selected HWA checkpoint. For every
minibatch, gradients are evaluated at the currently realized conductances, a
clean digital shadow target is updated, and both dense arrays are programmed
again through the corresponding endpoint model. This is closed-loop
program-and-verify modeling, not a Tiki-Taka pulse update.

Ten epochs produced `34,380` rewrite steps. CMO selected epoch 9, reaching
`96.72%` validation accuracy and `0.031577` validation KL. Wan selected epoch
5, reaching `91.36%` and `0.206599`; later Wan epochs had worse validation KL.
Fresh-process test evaluation used the selected checkpoints.

Relative to the HWA first write, noisy BPTT added `+4.28` accuracy points on
CMO and `+11.53` points on Wan. This recovered `82.9%` and `67.7%` of the
respective clean-HWA accuracy gaps. Teacher KL fell by `79.3%` on CMO and
`63.4%` on Wan. HWA alone added only `+1.03` points to the CMO first write and
`+1.69` points to the Wan first write.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
