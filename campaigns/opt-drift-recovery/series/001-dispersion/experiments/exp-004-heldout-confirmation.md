---
id: "exp-004"
title: "Independent confirmation of the recovery boundary"
status: "planned"
hypotheses: ["H-001"]
---
# exp-004 — Independent confirmation of the recovery boundary

## Question and dependencies

Does the selected recovery comparison generalize to independent deployment
chips at low dispersion and near/above the proposed benefit boundary?
Depend on [exp-002](exp-002-joint-digital-recovery.md) and
[exp-003](exp-003-exponent-sensitivity.md), or explicitly record which exponent
robustness conditions remain outside the confirmed scope.

## Cases and controls

Freeze checkpoint/cache identities, sigma/age/exponent conditions, m and window,
calibration selection, update law, learning rates, masks, target quality,
minimum useful effect, data/write/compute budgets and analysis rule before
drawing final chips.

Proposed minimum: three independent held-out chips per selected condition,
with additional sampling if the development uncertainty requires it. Freeze
the final count and sampling/stopping policy before observing their results.
Include representative low sigma and no-impairment controls; do not select only
favorable high-sigma conditions. Preserve all eligible no-learning controls,
refresh and the same global_joint/digital_only comparison.

Within a chip, pair interventions on the same aged state. Independent
programming/noise draws on one assignment are not automatically independent
chips. Training/cache seed repetitions are a separate axis and are selected
prospectively for representative cases.

## Execution contract

Reuse the validated native path and exact frozen configs from the prior
experiments. Preserve shared-specification and code snapshots, state/cache
hashes, training masks, counters and native bundles. Do not retune on test
text, test chips, failed conditions or recovery endpoints.

Each failed/partial case remains visible. A replacement is recorded with its
reason and original evidence, not counted as an extra successful replicate.
Operational completion, numerical validity, coverage and scientific verdict
remain separate.

## Metrics and decision

Apply the frozen H-001 rule to paired chip effects and target attainment.
Report individual chips and paired uncertainty at each condition, with
training/cache-seed variation separate. Do not pool architectures, sigma levels,
age/exponent settings or fault regimes.

Support the necessity claim only within the tested alternatives and budgets
where joint recovery meets the target and the alternatives do not. If simpler
methods meet it, report their sufficiency. If uncertainty or missing controls
prevents attribution, report an inconclusive scope. BPTT remains a simulation
gradient reference, not a hardware-cost validation.

## Budget, stop and expected outputs

Final chip count, selected conditions, resource/storage caps, deadline and target
are pending the development evidence. Freeze them before evaluation, including
failure handling and checkpoint retention. Stop per the predefined rule rather
than extending until a favorable result appears.

Expected raw root: `results/opt-drift-recovery/exp-004/`.
Expected review note: `series/001-dispersion/results/exp-004.md`.
Deliver the paired confirmation table, uncertainty plots, complete failure/
coverage accounting and a scoped verdict linked back to the common campaign.

## Execution and monitoring handoff

Not launched. Before running, record the frozen scientific specification,
exact commands, identities, target, bounded coverage, monitoring owner and
collection/review commands. No pilot output counts as held-out confirmation.
