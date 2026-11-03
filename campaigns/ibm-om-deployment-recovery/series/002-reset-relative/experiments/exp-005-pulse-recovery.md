---
id: "exp-005"
title: "Matched refresh, direct-pulse and Tiki-Taka recovery"
status: "imported"
hypotheses: ["H-005"]
---

# exp-005 — Matched refresh, direct-pulse and Tiki-Taka recovery

## Question and provenance

Matched refresh, direct-pulse and Tiki-Taka recovery. Teacher-KL updates expressed through physical single-pulse writes can recover validation accuracy from a frozen nine-level RESET-relative QAT deployment, and a Tiki-Taka accumulator can use the same slow-array pulse budget more effectively than direct stochastic writes.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-005-pulse-recovery.md).

Dependencies and related controls: [Shared RESET-relative nine-level QAT](../results/exp-004-shared-quantization.md).

## Frozen plans

- [One-seed RESET-relative deployed-array pulse recovery pilot](../../../../../studies/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1.json) — `mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.

Historical document locations (now references/forwarding pages): [ibm_om_reset_relative_on_chip_recovery.md](../../../../../docs/ibm_om_reset_relative_on_chip_recovery.md).

<a id="source-ibm_om_reset_relative_on_chip_recovery-reset-relative-deployed-array-pulse-recovery"></a>

## Historical source: RESET-relative deployed-array pulse recovery

This protocol tests whether training the already programmed IBM OM array can
recover error that off-chip RESET-relative QAT did not anticipate. It starts
from one immutable pulse-resolved deployment, not from the clean checkpoint
or from newly reprogrammed weights.

<a id="source-ibm_om_reset_relative_on_chip_recovery-source-state"></a>

## Source state

The source is the selected nine-level RESET-relative QAT checkpoint (selection
epoch 4, SHA-256
`537681598b887d4d63c6429c7f7bee3a7082041215c09d6cb3c25947452f311a`).
It is freshly deployed once on repaired assignment seed `84501` with endpoint
seed `84601`. The sidecar contains both states needed for continuation:

- `apparent_endpoint`: clipped conductances used by the network forward pass;
- `persistent_endpoint`: the physical simulator state from which the next
  SET or RESET pulse evolves; and
- `generator_state_after_programming`: the exact continuation of pulse and
  write-noise randomness.

Persistent accuracy is a hidden-state diagnostic. It is not a later read,
retention measurement, or noisy reread. Apparent-forward accuracy remains the
deployment result.

The runner validates the checkpoint, teacher, device model, assignment,
endpoint seed, target mapper, RESET commissioning, rail layouts, codebook,
population fingerprint, and full deployment hash before applying a pulse.
An arm cannot remap targets, inspect a hidden bound to choose an update, or
replace an unsupported cell.

<a id="source-ibm_om_reset_relative_on_chip_recovery-physical-scale"></a>

## Physical scale

The nine-level codebook has logical contrast spacing `delta = 0.095849`.
Because one signed contrast is split across two same-sign cells, adjacent
per-cell offsets are `delta / 2 = 0.0479245` in the public normalized
conductance coordinate. The OM preset's nominal minimum one-pulse change in
that coordinate is `0.04745`. Thus one nominal pulse is approximately one
adjacent cell-offset step, but an actual pulse is nonlinear, stochastic, and
state dependent; it is not a guaranteed transition between QAT codes.

Recovery deliberately allows conductances to leave the original five cell
offsets. It never rounds a programmed endpoint back to a code or projects the
persistent state to the public range.

<a id="source-ibm_om_reset_relative_on_chip_recovery-matched-update-paths"></a>

## Matched update paths

All gradient-directed paths use teacher-KL BPTT on the host. Only the physical
weight mutation differs.

<a id="source-ibm_om_reset_relative_on_chip_recovery-rail-refresh"></a>

### Rail refresh

This no-gradient control spreads a fixed number of single pulses across the
slow array. A cell originally above its quad RESET baseline receives SET; a
baseline cell receives RESET. It asks whether merely refreshing the intended
rail orientation helps, independently of learning.

<a id="source-ibm_om_reset_relative_on_chip_recovery-direct-pulses"></a>

### Direct pulses

For each physical conductance `G_i`, host BPTT computes its own gradient. The
desired update is `-learning_rate_i * gradient_i`. Its magnitude, divided by
`conductance_span * 0.04745`, becomes an unscaled one-pulse probability. One
positive scale is fitted on 64 replayed source-state minibatches so the
expected full-run requests equal the declared cap. A sampled positive update
issues one SET; a sampled negative update issues one RESET.

The calibration restores the training dataloader and global RNG states, so it
does not consume the first epoch's minibatch order.

<a id="source-ibm_om_reset_relative_on_chip_recovery-ideal-fast-tiki-taka"></a>

### Ideal-fast Tiki-Taka

Every minibatch adds the normalized continuous desired update to a zeroed
host-side fast matrix. For each slow layer, a cyclic column schedule makes
`budget * number_of_columns` transfer visits over the full ten epochs. At a
visit, the accumulated row signal is stochastically converted to at most one
slow SET or RESET per cell and that fast column is cleared.

This separates the value of accumulation and sparse transfer from physical
fast-array imperfections.

<a id="source-ibm_om_reset_relative_on_chip_recovery-physical-fast-tiki-taka"></a>

### Physical-fast Tiki-Taka

Each slow conductance receives a repaired-OM fast differential pair. The fast
assignment is `85501` and endpoint seed is `85601`. Commissioning applies
eight RESET/read samples, takes the guarded maximum of each pair, and
equalizes both rails to that observed baseline by adaptive verify. Gradient
sign selects the fast plus or minus rail for a stochastic SET pulse. A cyclic
column transfer reads the apparent pair contrast, requests slow pulses, and
adaptively RESETs the transferred pair slice.

Fast SET/RESET pulses, commissioning, equalization, verify reads, reversals,
failures, and unique cells are reported separately from the matched slow
budget.

<a id="source-ibm_om_reset_relative_on_chip_recovery-budgets-and-selection"></a>

## Budgets and selection

The slow array has `158,800` conductances. The three whole-run caps are:

| Budget per slow cell | Maximum requested slow pulses |
| ---: | ---: |
| 0.1 | 15,880 |
| 1 | 158,800 |
| 4 | 635,200 |

Every arm runs exactly ten epochs (`34,380` minibatches) and retains epoch 10.
There is no validation-based checkpoint selection. All twelve arms use one
seed initially; the pilot can check mechanism and implementation, but cannot
rank methods statistically.

<a id="source-ibm_om_reset_relative_on_chip_recovery-gain-and-evaluation"></a>

## Gain and evaluation

The original deployed gain is `562.341325190349`. Before training and after
each epoch, the runner also fits one positive gain on the fixed 1,024-example
training calibration cohort. It evaluates apparent and persistent states at
both gains on all 5,000 validation examples and proves that positive gain
does not change argmax accuracy. Gain can improve KL and gradient scale; it
cannot itself explain an accuracy change.

Each epoch checkpoint retains every slow and fast state, RNG stream, direct
probability calibration, transfer position, pulse ledger, logit gain,
dataloader generator, and metric history. Resume therefore continues the
same physical trajectory rather than starting a statistically similar one.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
