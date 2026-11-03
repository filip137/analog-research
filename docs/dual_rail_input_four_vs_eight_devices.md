# Clamped dual-rail input: four versus eight devices

## Campaign records

The study narratives and current interpretations now live in the following campaigns. Original dates/status text in the retained preamble is historical; the sections below remain shared technical reference.

- [Four versus eight measured devices](../campaigns/initialization-and-representation/series/002-teacher-and-representation/experiments/exp-009-four-versus-eight.md) — [results and interpretation](../campaigns/initialization-and-representation/series/002-teacher-and-representation/results/exp-009-four-versus-eight.md)

## Question

Does the double-differential input interface introduce independent nodes, or
does it reduce to the original dual-rail interface because the input is
clamped to `[x, -x]`? If it reduces, can a four-device logical synapse retain
the measured-device benefit previously obtained with eight devices?

The first part has an exact circuit answer. The second part depends on device
range and mismatch and is therefore covered by a frozen MNIST experiment.

## Exact input-boundary identity

For one logical input,

```text
x+ = x                         x- = -x
x++ = x+                      x+- = -x+ = x-
x-+ = x-                      x-- = -x- = x+
```

Thus `x+-` is exactly `x-`; it is not another independently evolving node.
Likewise, `x--` is `x+`. The apparent four input ports contain only the two
original rail voltages.

Let `G+` and `G-` be the two non-negative branch matrices of the signed
interaction. Split their source rows into the `x+` and `x-` halves. On the
clamped subspace, the eight physical devices of each logical input/output
rail block can be regrouped into the four post-facing conductances

```text
C[x+, :] = G+[x+, :] + G-[x-, :]
C[x-, :] = G-[x+, :] + G+[x-, :].
```

For any hidden voltage, this regrouping preserves both the edge energy and
the current entering every hidden node. It also preserves the derivative in
the permitted logical-input direction, where a change in `x+` is accompanied
by the opposite change in `x-`.

This is deliberately a boundary result. The individual source-port currents
are not preserved, and the identity fails if the two source rails are free
dynamic variables. Therefore the same reduction cannot generally be applied
between hidden layers, even when the desired equilibrium is approximately
antipodal.

For the ideal teacher map, with common branch baseline `b`, scale `s`, and
signed differential lift `D`,

```text
G+ = b + s max(D, 0)
G- = b + s max(-D, 0)
C  = 2 b + s dual_rail_lift(W).
```

So, at the clamped input, the eight-device implementation is exactly a
four-conductance dual-rail block whose conductances are sums of device pairs.

## Why this is not merely four unchanged devices

Replacing each sum by one existing-range memristor is exact only if that
memristor can realize the summed conductance, or if all relevant circuit terms
are rescaled consistently. Two devices in parallel can provide twice one
device's reachable conductance and change mismatch statistics. Consequently,
the node identity removes redundant *ports*, but it does not automatically
remove the range and averaging supplied by the second device.

This distinction is what the experiment tests:

The canonical construction and programming rules for both schemes are in
[`initialization_protocols.md`](initialization_protocols.md#four-device-and-eight-device-dual-rail-initialization).

- The numerical equivalence tests use arbitrary `G+` and `G-` and require
  float64 agreement of input-edge energy, hidden-node quadratic and linear
  coefficients, hidden-node KCL, and the logical input derivative.
- A negative control releases the two input rails and requires the proposed
  collapse to cease being equivalent.
- The teacher-map test requires the collapsed differential pair to equal
  `2 b + s dual_rail_lift(W)` for both repository rail layouts.
- The measured four-device arm assigns all four cells of one logical
  synapse a single reachable baseline and span. This
  `dual_rail_quad_common_window` mapping is stricter than the earlier mapping,
  which shared a window only between each output-rail pair.
- The measured eight-device control retains the existing `G+`/`G-` common-
  window implementation.

## Moved study sections

<a id="frozen-end-to-end-comparison"></a>

[frozen end to end comparison](../campaigns/initialization-and-representation/series/002-teacher-and-representation/experiments/exp-009-four-versus-eight.md#source-dual_rail_input_four_vs_eight_devices-frozen-end-to-end-comparison)

<a id="result"></a>

[result](../campaigns/initialization-and-representation/series/002-teacher-and-representation/results/exp-009-four-versus-eight.md#source-dual_rail_input_four_vs_eight_devices-result)

<a id="held-out-cohort-b-follow-up"></a>

[held out cohort b follow up](../campaigns/initialization-and-representation/series/002-teacher-and-representation/experiments/exp-009-four-versus-eight.md#source-dual_rail_input_four_vs_eight_devices-held-out-cohort-b-follow-up)
