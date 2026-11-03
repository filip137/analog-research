---
id: "exp-004"
title: "Shared RESET-relative nine-level QAT"
status: "imported"
hypotheses: ["H-004"]
---

# exp-004 — Shared RESET-relative nine-level QAT

## Question and provenance

Shared RESET-relative nine-level QAT. On an untouched repaired IBM OM array, nine-level quantization-aware HWA using one global RESET-relative contrast codebook will achieve higher apparent-forward test accuracy than a matched continuous shared-target HWA control because the minimum nonzero logical contrast is larger than the four-cell program-and-verify acceptance-error envelope.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-004-shared-quantization.md).

Dependencies and related controls: [Corrected fixed-array common-window HWA](../../001-common-window/results/exp-002-fixed-array.md).

## Frozen plans

- [MNIST four-device IBM OM shared RESET-relative quantized HWA](../../../../../studies/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1.json) — `mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.

Historical document locations (now references/forwarding pages): [ibm_om_reset_relative_quantization.md](../../../../../docs/ibm_om_reset_relative_quantization.md).

<a id="source-ibm_om_reset_relative_quantization-ibm-om-shared-reset-relative-quantization"></a>

## Historical source: IBM OM shared RESET-relative quantization

<a id="source-ibm_om_reset_relative_quantization-question"></a>

## Question

The earlier four-device common-window experiment used the sampled minimum and
maximum of every assigned cell to construct a separate affine target window
for each logical synapse. That is useful for an array-specific upper bound,
but it assumes information that a newly deployed array does not provide
without characterizing every cell's complete range.

This protocol instead asks whether a small global codebook, positioned by a
RESET/read commissioning measurement that is available on the new array, can
preserve enough differential signal for HWA training and program-and-verify
deployment. It deliberately does not answer whether a subsequent on-chip
update closes any remaining gap.

<a id="source-ibm_om_reset_relative_quantization-logical-state-and-four-cell-convention"></a>

## Logical state and four-cell convention

One signed logical value is represented by the four dual-rail cells
`(++,+-,-+,--)`. From a clean normalized four-cell tensor, reconstruct

```text
u = 0.5 * (q++ - q+- - q-+ + q--),  u in [-1, 1].
```

The frozen clean shadow uses the full nominal fractions `(1,1)`, so the
available logical range is not silently inherited from the previous
`(0.25,0.125)` initialization. The derivation is linear, performs zero
optimizer updates, retains the exact signed logical matrices, and selects the
already computed full-span output gain from the source mapping artifact.

<a id="source-ibm_om_reset_relative_quantization-commissioning-without-per-cell-range-characterization"></a>

## Commissioning without per-cell range characterization

On each fixed physical assignment:

1. Apply one RESET pulse to every cell and read its apparent state.
2. Repeat this RESET/read observation `M=8` times.
3. For cell `k`, compute its observed mean `r_k`, standard error `s_k`, and
   guarded estimate `r_k + 3*s_k`.
4. For each logical four-cell group, set
   `B = max_k(r_k + 3*s_k)` and use that same `B` for all four cells.
5. If `B` is below the public characterized coordinate zero, raise the shared
   baseline to zero. Do not clip it against any cell-specific bound.

The durable commissioning bundle contains the RESET observations, standard
errors, expanded quad baselines, layout, physical-assignment fingerprint,
and pulse cost. It contains no per-cell minimum or maximum. Recommissioning a
new array is allowed and necessary; copying the development array's baseline
is not.

The simulator retains hidden per-cell device parameters because it still
needs them to execute physical pulses. They are used after mapping to report
whether a requested target is physically supported and to run the exact
controller for a target/class combination absent from the compact endpoint
model. They never feed back into target construction.

<a id="source-ibm_om_reset_relative_quantization-shared-codebook-and-quantizer"></a>

## Shared codebook and quantizer

The program-and-verify tolerance is
`tau = 0.5 * 0.04745 = 0.023725` in normalized OM coordinates. Four
independent cell errors can change the signed four-cell contrast by at most
`4*tau`. The frozen adjacent contrast spacing is therefore

```text
delta = 1.01 * 4 * tau = 0.095849.
```

For the nine-level path, quantize before endpoint sampling or programming:

```text
n = clamp(round_half_away_from_zero(4*u), -4, 4)
C = n * delta.
```

Positive and negative targets are

```text
C >= 0: (B + C/2, B,       B,       B + C/2)
C <  0: (B,       B - C/2, B - C/2, B      ).
```

Thus all devices receive one of the five non-negative global offsets
`{0, delta/2, delta, 3*delta/2, 2*delta}` relative to their quad's observed
RESET baseline. There are nine signed logical contrasts: zero and four
magnitudes per sign. Absolute conductances need not be identical across the
array because the measured RESET floors differ; the information-bearing
offsets are identical.

The continuous control replaces integer `n` with `4*u`. It has the same
maximum contrast, commissioning, physical assignment, controller, and pulse
budget, so it isolates quantization rather than available range.

<a id="source-ibm_om_reset_relative_quantization-qat-semantics"></a>

## QAT semantics

The optimizer owns clean continuous FP32 master tensors. For each minibatch:

1. derive `u` from the clean master;
2. quantize it and construct shared RESET-relative targets;
3. sample compact programmed apparent endpoints, with exact pulse fallback
   only for audited non-corrupt out-of-support targets;
4. use those apparent endpoints in the DRN forward and backward pass;
5. restore the FP32 master and apply the optimizer update.

This is a straight-through hardware-aware update: the discontinuous mapper is
outside autograd, while the gradient computed at the mapped physical forward
state is applied to the clean master. It is not on-chip training and it does
not accumulate persistent conductance across minibatches.

Physical checkpoint selection and deployment use the exact cap-128 adaptive
program-and-verify controller. Forward inference uses the saved apparent
endpoint. The corresponding persistent endpoint, accepted/failure masks,
pulse costs, population, commissioning bundle, and RNG state remain together
for a possible later on-chip recovery study.

<a id="source-ibm_om_reset_relative_quantization-matched-controls-and-held-out-deployment"></a>

## Matched controls and held-out deployment

The predeclared study contains four development pipelines:

| Pipeline | Minibatch forward | Physical selection/deployment |
| --- | --- | --- |
| Zero-update quantized | None | Nine-level RESET-relative |
| Clean BPTT | Clean FP32 | Nine-level RESET-relative |
| Continuous HWA | Continuous RESET-relative compact endpoint | Continuous RESET-relative pulse-resolved |
| Quantized QAT | Nine-level RESET-relative compact endpoint | Nine-level RESET-relative pulse-resolved |

All development roles use repaired assignment seed `84001`. Training endpoint
seed `84002` and selection endpoint seed `84003` are intentionally distinct.
After all selected checkpoints are frozen, each is deployed to the untouched
repaired assignment seed `85001` with five predeclared endpoint seeds
`85101` through `85105`. The held-out array is a generalization test, not a
second opportunity to choose the codebook, read count, guard, gain, learning
rate, or checkpoint.

The primary comparison is held-out quantized-QAT accuracy against the matched
continuous-HWA accuracy. Zero-update and clean-BPTT deployments separate the
initial quantization loss from the benefit of ordinary retraining and
hardware exposure. Report ideal mapped targets separately from programmed
apparent endpoints so codebook loss and programming loss are not conflated.

<a id="source-ibm_om_reset_relative_quantization-scope-and-failure-conditions"></a>

## Scope and failure conditions

The mapper fails closed when targets leave the public `[0,1]` coordinate or
when fewer than 95% of quads are fully supported in the hidden post-mapping
audit. A formal run also fails its programming gate below 99% accepted cells,
above 1% budget exhaustion, on a non-finite endpoint, or if training and
selection population fingerprints differ.

These checks do not make the initialization device-independent in the sense
of using no array observations. They make it deployable with RESET/read
commissioning instead of a full per-cell SET/RESET range sweep. If the
held-out array fails the predeclared support gate, that is a result: the
global codebook plus RESET observation did not transfer conservatively enough.

No on-chip recovery is part of this protocol. A Tiki-Taka, LoRA, or soft
SET/RESET arm may be added only in a later study that starts from one frozen
two-state deployed bundle and includes a matched no-update control.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
