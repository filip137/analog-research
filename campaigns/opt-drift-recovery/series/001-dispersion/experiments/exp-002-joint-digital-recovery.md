---
id: "exp-002"
title: "Joint learning versus digital-only recovery and refresh"
status: "planned"
hypotheses: ["H-001"]
---
# exp-002 — Joint learning versus digital-only recovery and refresh

## Question and dependencies

At low/high dispersion selected by [exp-001](exp-001-dispersion-calibration.md),
do analog updates add useful recovery beyond digital-only training and simpler
interventions? This is the direct development test of
[H-001](../../../hypotheses/H-001-dispersion-recovery.md).

Use the same scalar power law, age, window/mapping, frozen std3 source and chip
identities as the selected development comparison. Both learning arms minimize
end-to-end KL to cached logits of this architecture's clean post-HWA model.

## Cases and controls

Carry deployed reference and both calibration alternatives from compatible
paired bundles. Add bounded P&V refresh toward the frozen HWA targets, followed
by recalibration. The two active learning arms start from clones of the same
validation-selected calibrated snapshot:

| Arm | Analog conductances | Digital parameters |
| --- | --- | --- |
| global_joint | All programmable devices in mapped MLPs | Attention projections, LayerNorm affine parameters and digital biases in decoder blocks 7–12 |
| digital_only | Frozen aged state | Exactly the same digital mask |

Freeze blocks 1–6, embeddings, final normalization outside the selected blocks,
and the language-model head. Serialize masks and trainable counts. Calibration
parameters have the same separately declared policy in both arms.

Use the fixed 1,024-sequence, length-256 corpus subset, effective batch 32,
and 128 updates/four passes per learning arm (1,048,576 input-token
presentations). Match order, digital optimizer protocol and validation allowance.
Keep the 256-sequence validation cohort and held-out cohorts separate.
Prepare teacher targets before drawing deployment chips; do not rerun a teacher
during recovery. Include no-impairment learning controls and a small matched
joint-arm control with noise-free writes, as specified by the campaign defaults.

## Execution contract

The [native handoff](../../../lifecycles/README.md) identifies required
extensions. Verify exact joint/digital masks, cached-target use, held-state
forwards, zero physical writes in digital-only recovery, and noisy physical
updates in the joint arm. Freeze drift time; never compound aging per update.

Declare analog/digital optimizers and learning rates, physical write law/noise,
bounds, per-device attempt cap and checkpoint/stopping selection before running.
Use development tuning or a justified prior recipe; do not inherit unrelated
fault-recovery settings. Bounded P&V requires actual tolerance, attempts and
verification accounting. An endpoint-noise draw cannot supply its pulse cost.
A missing refresh comparison limits any necessity claim.

## Metrics and decision

Report KL/perplexity, remaining clean-to-calibrated gap closed when positive,
and paired joint-minus-digital and joint-minus-no-learning improvements.
Keep negative improvements and failed cases. Track attempted/realized writes,
per-device distributions, verification/calibration reads, data/cache traffic,
compute and peak memory, with shared deployment costs separately identified.

Freeze L_target, delta_min and cost limits before final comparison. Joint
learning is useful if its paired improvement meets the declared margin.
It is needed within the tested method/budget set only if it reaches the target
and eligible alternatives do not. Refresh or digital-only success contradicts
necessity in that regime. If none succeeds, recovery remains unresolved.

## Budget, stop and expected outputs

The 128-update allowance is the agreed pilot starting point for each learning
arm; production duration is fixed after development. Sigma cases, learning-rate
selection allowance, physical write/P&V caps, compute/storage caps and execution
target remain open. Failed/stuck write attempts consume their declared cap.
Stop on exhausted budget or invalid state/solve, preserving evidence.

Expected raw root: `results/opt-drift-recovery/exp-002/`.
Expected review note: `series/001-dispersion/results/exp-002.md`.
Retain states/checkpoints, masks, cache manifests, counters and paired
quality-versus-cost curves. Native result bundles own actual evidence.

## Execution and monitoring handoff

Not launched. After exp-001 resolves conditions and the implementation is
validated, record exact configurations/input hashes, target, commands, cap,
progress paths and monitoring owner here. Scientific changes are recorded
before new runs; failed attempts are retained and are not replicates.
