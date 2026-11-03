---
id: "exp-009"
title: "Held-out raw-active seven-level QAT"
status: "imported"
hypotheses: ["H-009"]
---

# exp-009 — Held-out raw-active seven-level QAT

## Question and provenance

Held-out raw-active seven-level QAT. Seven-level quantization-aware mapped HWA using one global D90 differential codebook and one variable within-quad baseline will achieve at least two percentage points higher mean held-out apparent programmed test accuracy than the matched continuous mapped-HWA control.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-009-seven-level-qat.md).

Dependencies and related controls: [Raw-active canary and learning-rate screen](../results/exp-008-readiness-and-rates.md), [Array-wide nine-level physical-cell programming](../../../../ibm-om-programming/series/002-common-cell-levels/results/exp-005-nine-cell-levels.md).

## Frozen plans

- [MNIST raw-active IBM OM p90 common-differential seven-level HWA](../../../../../studies/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1.json) — `mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.

Historical document locations (now references/forwarding pages): [ibm_om_raw_active_state_program_verify.md](../../../../../docs/ibm_om_raw_active_state_program_verify.md).

<a id="source-ibm_om_raw_active_state_program_verify-ibm-om-raw-active-p90-deployment-and-qat-protocol"></a>

## Historical source: IBM OM raw-active p90 deployment and QAT protocol

> This historical protocol uses a quad-local baseline `B_Q` that varies
> between logical weights.  It is not the array-wide common-baseline
> nine-cell-state protocol in
> [`ibm_om_raw_active_common_cell_9_program_verify.md`](../../../../ibm-om-programming/series/002-common-cell-levels/experiments/exp-005-nine-cell-levels.md#source-ibm_om_raw_active_common_cell_9_program_verify-ibm-om-raw-active-p90-common-cell-nine-level-pv).

- Status: implemented protocol for the `raw_active_p90_quad` target mapper and
  pulse-resolved deployment path
- Device model: AIHWKit 1.1.0 `ReRamArrayOMPresetDevice`
- Evidence class: normalized hardware-derived fitted model, not an absolute
  conductance calibration in siemens

<a id="source-ibm_om_raw_active_state_program_verify-one-cell-is-one-drn-conductance"></a>

## One cell is one DRN conductance

For physical cell `i`, the programmable active state is `a_i`. Its own
sampled parameters are

```text
a_min,i, a_max,i, d_up,i, d_down,i.
```

The AIHWKit tile-facing state is `w_i = a_i - r_i`, but the sampled reference
`r_i` is not a second programmable device. This protocol excludes `r_i` from
target construction, bounds, pulse dynamics, verify observations, acceptance,
and inference. It is retained only as identity provenance. Tests require a
change to `r_i` alone to leave mapping and programming bit-identical.

<a id="source-ibm_om_raw_active_state_program_verify-frozen-array-wide-coordinate"></a>

## Frozen array-wide coordinate

The repaired development assignment at seed 84001 freezes one common affine
coordinate for every cell and every later array:

```text
A_min = -3.455834150314331
A_max =  2.5384607315063477
S     =  5.994294881820679
g     = (a - A_min) / S.
```

The calibration population contains 158,800 cells in binding order
`base.dense_weight.0`, `base.dense_weight.1`, with shapes `[1568,100]` and
`[100,20]`. Its population fingerprint is
`4887ad89abdd16193448c54a0cbe97be915cb4b9bc04cf1151c5e2a96ee5a3ac`.

The same transform is reused on later arrays. Cell-wise normalization

```text
(a_i - a_min,i) / (a_max,i - a_min,i)
```

is prohibited because it would give each cell a different unit and erase
span and pulse-response variation. The DRN model therefore uses conductance
bounds `[0,1]` directly for this protocol; the runtime rejects a second affine
mapping through the historical `[0.1020408,1]` bounds.

All sampled RESET bounds are below raw `a=0`, and all sampled SET bounds are
above it. Consequently their displayed `g` distributions meet at

```text
g(a=0) = 0.576520544692435.
```

That boundary is not plot clipping.

<a id="source-ibm_om_raw_active_state_program_verify-four-cell-logical-coordinate"></a>

## Four-cell logical coordinate

The canonical cell order is `(++,+-,-+,--)`. Define

```text
D_Q = (g_++ - g_+- - g_-+ + g_--) / 2
B_Q = (g_++ + g_+- + g_-+ + g_--) / 4.
```

For requested signed differential `D*_Q`, construct

```text
g*_++ = g*_-- = B_Q + D*_Q/2
g*_+- = g*_-+ = B_Q - D*_Q/2.
```

Thus all four cells in a quad have exactly the same baseline, while the
information-bearing differential has one common array-wide unit. `B_Q` may
vary between quads because it cancels from `D_Q`, but it remains in the
physical network and therefore changes loading and power.

<a id="source-ibm_om_raw_active_state_program_verify-frozen-lower-tail-p90-budget"></a>

## Frozen lower-tail p90 budget

For each complete quad, transform its four exact raw supports and compute

```text
C_Q = min_i(g_max,i) - max_i(g_min,i).
```

For `N=39700` candidate quads, the largest threshold supported by at least
`ceil(0.9*N)=35730` quads is the lower-tail order statistic

```text
D_90 = 0.10354409442884083.
```

The one-cell full-scale displacement is `D_90/2 = 0.051772047214420414`.
The frozen development yields are:

| Binding | Eligible quads | Total quads | Fraction |
| --- | ---: | ---: | ---: |
| `base.dense_weight.0` | 35,270 | 39,200 | 0.8997449 |
| `base.dense_weight.1` | 460 | 500 | 0.92 |
| Total | 35,730 | 39,700 | 0.90 |

For an eligible quad,

```text
B_low,Q  = max_i(g_min,i) + D_90/2
B_high,Q = min_i(g_max,i) - D_90/2
B_Q      = (B_low,Q + B_high,Q)/2.
```

`B_Q` is fixed across every sign and magnitude requested from that quad. The
runtime never shrinks `D_90`, rescales by `C_Q`, or clips an unsupported
target. The first implemented stress path declares unsupported quads as
structural target-assignment failures with zero target-programming pulses.
A future functional deployment must version a donor/reassignment stream
before replacing those quads; it may not select replacements from accuracy.

<a id="source-ibm_om_raw_active_state_program_verify-continuous-and-seven-level-training-coordinates"></a>

## Continuous and seven-level training coordinates

The clean FP32 master reconstructs `u in [-1,1]` from its four rails. The
continuous mapper uses

```text
D* = clamp(u,-1,1) D_90.
```

The program-and-verify tolerance in the shared coordinate is

```text
tau_g = 0.00791586015294392.
```

Four independent cell acceptance errors produce a worst-case separation
requirement of `4*tau_g = 0.03166344061177568` between adjacent requested
logical differentials. Seven symmetric levels fit while preserving that
spacing because `D_90/3 = 0.03451469814294694 > 4*tau_g`:

```text
n  = clamp(round_half_away_from_zero(3u), -3, 3)
D* = n D_90/3.
```

This is the QAT codebook. The optimizer owns the continuous `[0,1]` FP32
master. During a QAT forward, eligible quads are temporarily replaced by the
seven-level array-aware targets and restored before the ideal off-chip update.
Unsupported quads use their transformed RESET bound in this deterministic
mapping-only surrogate.

The historical compact endpoint density was fitted in the tile-facing `q`
coordinate for an adaptive controller and is not reused. Until an adequate
raw-coordinate one-pulse compact model is fitted, minibatch training is
explicitly mapping-only; checkpoint selection and deployment remain exact
pulse-resolved one-pulse operations.

<a id="source-ibm_om_raw_active_state_program_verify-raw-pulse-plant-and-standard-program-and-verify"></a>

## Raw pulse plant and standard program-and-verify

The physical pulse equation is evaluated in `a`. SET and RESET use each
cell's own sampled response, bounds, and cycle draw. The global affine map is
only the controller/reporting coordinate.

Each new trajectory starts at raw `a=0`. RESET conditioning continues until
four consecutive persistent raw changes are below `2e-6`, or until 4096
pulses. Conditioning pulses have their own seed and are excluded from the
128-pulse target budget.

The canonical controller sees only apparent `g`, its continuous scalar
target, and its own history:

```text
if abs(g_apparent - g_target) <= tau_g: accept
elif g_apparent < g_target - tau_g:     apply one SET pulse
else:                                  apply one RESET pulse
```

The four cells of a quad run this controller independently. A RESET after an
overshoot does not couple their states or change `B_Q`.

Every deployment keeps exact target support, structural p90 eligibility,
conditioning success, apparent acceptance, persistent-within-tolerance after
acceptance, budget exhaustion, SET/RESET counts, verifies, reversals, raw and
transformed endpoints, population parameters, and both RNG continuation
states as separate facts. Apparent acceptance is never relabelled as exact
persistent reachability.

<a id="source-ibm_om_raw_active_state_program_verify-evaluation-contract"></a>

## Evaluation contract

The matched study follows the RESET-QAT decomposition surface:

1. zero-update seven-level deployment;
2. clean FP32 BPTT followed by seven-level deployment;
3. continuous array-aware mapped-target training and deployment;
4. seven-level QAT and deployment.

Report clean, ideal mapped, apparent programmed, and persistent diagnostic
accuracy; teacher agreement and KL; layerwise `D` signal and endpoint error;
sign flips; prediction flips and margins; p90 eligibility; conditioning cost;
SET, RESET, verify, reversal, acceptance, persistent-tolerance, saturation,
and exhaustion statistics. Held-out assignments reuse the frozen coordinate,
`D_90`, codebook, learning rates, and gains without adaptation.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
