---
id: "exp-003"
title: "Sensitivity to mean and spread time exponents"
status: "planned"
hypotheses: ["H-001"]
---
# exp-003 — Sensitivity to mean and spread time exponents

## Question and dependencies

Does the recovery conclusion from [exp-002](exp-002-joint-digital-recovery.md)
survive changes in the time evolution of mean shift and spread?
Use the same scalar m and s at t_ref=1 h. Alpha and beta are independent
configurable parameters, with nominal values 0.23 and 0.11.

## Cases and controls

First vary alpha at fixed beta; then vary beta at fixed alpha. Use development
conditions near the observed benefit boundary and non-reference ages within
the supplied 25-hour measurement horizon. At one hour the uncorrected model
is independent of either exponent, so retain that point only as an invariance
check. The user's early-window beta estimate near 0.15 is a candidate sensitivity
case; the numerical grid is still to be frozen.

Reuse programmed state and standardized device draws across exponent settings,
with the synthetic temporal-coupling assumption explicit. Hold m, s, mapping,
teacher/cache, data, optimizer and budgets fixed. Compare calibration, refresh,
global_joint and digital_only under the same rules as exp-002. Select sparse
combinations only after the one-factor comparisons; do not start a full grid.

## Execution contract

The [power-law model](../../../../../../on-chip-learning/campaigns/drift/power-law-model.md)
defines positive exponents, the t=0 baseline and optional delay correction.
Use the initial uncorrected first-read convention. Any future delay/read-noise
or heavy-tail refinement is a separately named intervention.

Check power-law factors at non-reference times and exponent invariance at t_ref.
Record realized mean/spread and clipping because changes in beta can increase
bound occupancy. Preserve the exact same physical/update and calibration policies.

## Metrics and decision

Report paired quality/cost and recovery gaps for each (alpha, beta, age, sigma)
condition separately. Robustness means the previously declared useful recovery
effect persists across the tested parameter range. A loss of benefit narrows
the supported regime; it is not an operational failure to retry away.

Do not interpret the supplied ±0.05 exponent uncertainty as a validated
probability distribution. Do not label extrapolated ages as measured device
retention. Choose confirmation cases before accessing held-out chips.

## Budget, stop and expected outputs

The exponent/age grid, chip count, resource caps and target remain open.
Inherit the paired recovery data/step allowance from exp-002; any increase is
an explicit design change. Preserve invalid solves and exhausted budgets.

Expected raw root: `results/opt-drift-recovery/exp-003/`.
Expected review note: `series/001-dispersion/results/exp-003.md`.
Produce separate quality-versus-age and recovery-versus-exponent tables plus
the frozen condition list for independent confirmation.

## Execution and monitoring handoff

Not launched. Dependencies are the resolved exp-002 comparison and an explicit
bounded exponent/age grid. Add native configs, source/input hashes, commands,
target, progress paths and owner before numerical execution.
