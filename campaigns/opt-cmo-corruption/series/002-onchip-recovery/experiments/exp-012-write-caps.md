---
id: "exp-012"
title: "Recovery retained under hard per-device write-attempt caps"
status: "running"
hypotheses: []
---
# exp-012: maximum writes per device

## Question and authorization

On October 12 the user assigned the next question: "see how much can we reduce
the maximum number of writes per device". This follows the positive
[thresholded-write result](../results/exp-011.md): its 2.1M-token screen reached
development KL 0.068020 with maximum 15 successful writes per active cell;
50M tokens gave KL 0.068731 with maximum 152. The shorter result therefore offers
a useful first reference for reducing the maximum further.

The disclosed default is a short cap sweep, with an optional user preference
requested between short screens and full-epoch comparisons. In the absence of
steering, use the short screens. Prior authorization covers the idle 5090 hosts,
implementation, native execution and persistent monitoring. This experiment
introduces no new HWA fit, corpus, analog subset, defect regime or digital adapter.

## Frozen comparison

Reuse exp-011's exact P0, teacher, standard 3% Gaussian whole-model HWA source,
temporary training fault rate zero, permanent 3% random faults and assignment
271001. The final six OPT-125M MLPs contain 28,311,552 logical weights. Digital
attention, embeddings, base parameters and calibration remain frozen. This is
a one-array exploratory simulation, not a viability or fabricated-hardware claim.

Five independent fits start from P0 with fresh optimizer state. Caps are
**1, 2, 4, 8 and 16 issued write attempts per active device**. All use the same
8,192 training sequences / 2,097,152 input tokens / 256 optimizer updates,
batch size 32, order seed 53, teacher-KL objective, constant LR **3e-3**, signed
accumulation and nominal **2-sigma** threshold (0.0282790843 q units), selected
in exp-011. All upstream lifecycle sections and separated development/generated/
WikiText cohorts are unchanged. The existing uncapped 2-sigma short result is
the explicit reference; each new native run also contains the held-P0 control.

Count issued commands using digital controller-owned per-cell counters,
including attempts on stuck cells. Exclude initial deployment programming;
reference devices receive no adaptation writes. Once an active cell reaches
its cap, freeze it and stop accumulating further commands for it; other cells
continue. Cap enforcement never consults hidden faults, realized write noise or
successful-write counts. Issue the complete pending command at threshold crossing,
clear it after an issued attempt regardless of success, and never flush endpoint
residuals. The original unclamped conductance-dependent noisy-write law remains
unchanged. All counts describe idealized write events, not measured pulses.

## Measurements and decision

Retain fixed final checkpoints, with development evaluation every 64 updates.
Report final development KL and the **fraction of the uncapped short run's gain
retained**, `(P0_KL - capped_KL) / (P0_KL - uncapped_short_KL)`, without clipping.
Report every cap, including detrimental results; no acceptance threshold or
automatic winner/full-pass gate is introduced. Generated and WikiText endpoints
are reporting cohorts and do not select caps or checkpoints.

Also report successful writes, issued attempts, stuck attempts, maximum successful
writes and attempts per cell, distinct touched cells, cells at their cap, verify
reads and out-of-range devices. Validate that maximum attempts never exceed the
declared cap, per-cell successful writes never exceed attempts, and total attempts
equal successful writes plus stuck attempts. A failed validation invalidates that
case and prevents interpreting its apparent gain.

This measures the cap/performance tradeoff at one fixed LR, threshold and short
duration; it does not establish an optimized result for each cap. Infer no full-
epoch result or independent-array robustness. Review the complete curve to decide
whether a promising cap merits a separate longer or LR-tuned comparison.

## Budget and execution

Targets: idle RTX5090s on **loulou and trex**, one worker per GPU, preserving
other users' work. Budget **0.5 GPU-hours**, **20 GB additional storage**, cutoff
**17:30 Paris on October 12**. Prior screens took 102–112 seconds each including
validation; allow 180 seconds expected and 600 seconds maximum per case. Five
screens should need about 10–20 minutes after preparation, including queue cadence
and collection. The new int32 count adds approximately 113 MB per saved learning
trajectory and on GPU. Retain all native final states and failed evidence.

Use the public lifecycle CLI and the existing single run-watch service, tmux
bridge, native validators and compact collector. No interactive coordinator is
needed. No automatic scientific retries, LR changes or additional runs. Source,
input, native-plan and counter-validity checks must pass before dispatch.

Lifecycle IDs: `opt125m-cmo-l6-exp6-writecap-cap{1,2,4,8,16}-dev`.
Preparation helper: `campaigns/opt-cmo-corruption/launch/exp6_writecap.py`.
Evidence root: `results/lifecycles/opt125m-cmo-l6-exp6-writecap/`.

## Current execution and monitoring handoff

The controller-owned attempt cap is implemented in the shared lifecycle parser
and `ThresholdedWriteAdam`, with physical accounting gates in training and native
collection. Focused tests pass (20), including stuck-attempt charging, exhausted
cell freezing, cap-zero no-write behavior, exact nonbinding-cap trajectories,
strict configuration rejection and a tiny native lifecycle with exact replay.
The workflow and campaign checks pass **334 tests with one existing optional
AIHWKit skip**. `workflow describe` gives a 12.17 GB
conservative bound for the generic five-assignment/ten-trajectory lifecycle.
Each recovery-only manifest selects only assignment 271001 and two trajectories,
or 2.43 GB; all five cases together retain the same 12.17 GB conservative bound
before source/collection overhead. All five lifecycle files preserve upstream
provenance and differ only in their declared attempt cap and question identity.
Existing watchdog, restart/idempotence and bounded read-only incident-agent
handoff checks are recorded in exp-011.
The shared service remains the sole dispatcher and monitoring owner, polling
semantic progress, budget and storage every five minutes and preserving incidents.

All five cases are registered and released under campaign
`opt-cmo-exp6-writecap-20261012`, immutable manifest digest
`312269b7a4fcc7d92df05380fd1246768ba025add64ea624c74436a97fc14129`.
Worker and completion-controller source is
`bd8cad10c212638b345ea732dc59ce06cb4a8de4`. All ten host/case native plans and
dry runs passed; `readiness/preflight.json` records the passed launch gate.
The prior uncapped reference is frozen by SHA-256
`7f0ff1dceb4a5c8cf6ca4f25e3ec786bd918d4a929b3b512bdcbbecb4ec99771`.

At the first live check, cap 1 on loulou (attempt
`fdcc7c9162ae4dd1b8945714ec570db8`, worker PID 1229384) and cap 2 on trex
(attempt `8b84ccf27b304551addb472e3f8d554f`, PID 2278768) both reached all 256
updates with finite metrics and maximum issued counts of exactly 1 and 2.
Peak reserved memory was 19.99 GB, within the 26,000 MiB reservation.
The other three cases remain with the dispatcher and need no interactive launch.

Caps 1 and 2 subsequently completed all validation and collection gates, charging
214.84 GPU-seconds together. Their [partial result](../results/exp-012.md) is
reviewed. The dispatcher then launched cap 4 on loulou (attempt
`6c04a369eaec4227b82082abe5b4fbb4`, PID 1231147) and cap 8 on trex
(`776934e837be46e8834ac5a2925c86dd`, PID 2281315). Both have real training
metrics; `readiness/second-wave-live.json` records their identities and progress.
Cap 16 remains queued while those workers hold their reservations. Its current
budget-wait reason reflects reserved allowances, not an exhausted campaign;
the prior measured durations leave ample allowance when the workers finish.
The service has no incident for this campaign and retains a fresh heartbeat.

Worker roots on each host:
`/home/filip/analog-research/opt-cmo-exp1/hwa-exp6-writecap-20261012/`.
Native run roots are `results/lifecycles/<job-id>/runs/<job-id>` beneath that
checkout. Logs and tmux receipts are under
`/home/filip/analog-research/opt-cmo-exp1/worker-receipts/bd8cad10c212638b345ea732dc59ce06cb4a8de4/<job-id>/<attempt>/`.
Read-only attachment is `ssh -t <host> tmux attach -r -t <job-id>-<attempt>`
while the session exists. The durable queue owns attempt identity and cleanup.

Completion hooks run native inspection, exact-replay/lineage and cap-accounting
validation, then collect into `collected/<job-id>/<attempt>/` and
`reviews/cap<N>-dev.json` under the evidence root. The last successful hook writes
`reviews/summary.json`; it launches nothing further. Prepared commands, inputs,
paths and hashes are in `prepared/{loulou,trex}.json` and
`prepared/queue/jobs/`. The unchanged service is independently supervised by
systemd; budgets and history remain in the shared durable campaign ledger.
