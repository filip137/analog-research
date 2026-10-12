---
id: "exp-010"
title: "Development-triggered LR follow-up for dense recovery and LoRA"
status: "blocked"
hypotheses: []
---
# exp-010: bounded LR follow-up

## Authorization and question

On October 12 the user instructed: **"if the loss worsens tune the lr"**,
retaining the 09:00 Paris delivery deadline. This authorizes a bounded scientific
follow-up to [exp-008](exp-008-overnight-recovery.md) and
[exp-009](exp-009-digital-lora.md). Original configurations, LR-selection receipts,
endpoints and negative results remain intact. Additional fits are identified
separately and their tuning exposure and compute are reported.

Does a different LR improve the dense on-chip recipe that worsened development
KL, and can a lower LR address any development-loss worsening during the full
LoRA fits? This is exploratory follow-up after inspecting the original results,
not a new confirmatory comparison or independent-array replicate.

## Fixed science and decision rules

Retain the exact standard 3% Gaussian HWA source, held deployment P0, permanent
3% random faults, assignment 271001, final six OPT-125M MLPs (layers 6–11,
28,311,552 logical weights), teacher, objective, tokenization, batch size 32,
on-chip/adapter seed 53, data seed 20261008 and its existing epoch shuffle,
optimizer, update law and all evaluation cohorts.
Use the explicit portable P0 already validated in exp-009; its original parent
SHA-256 is `f80a9447469697cf9e4ff657cb17545d2824e60ae4f12d4694f01f3e4ebaf873`.
Every fit restarts from P0 with fresh optimizer state. No mid-run LR changes,
extra epochs, new scheduler or test-based selection.

**LoRA:** keep the current full passes running. At native validated completion,
compare final development teacher KL against the lower of the unchanged
deployment's development KL and the original selected LR's short-screen
development KL. A relative worsening greater than 2% releases exactly one
additional full pass at the best previously validated lower LR, selected by
short-screen development KL (exact ties: lower LR). Otherwise no extra run.
This means LR 1e-3 for ranks 4/16 if triggered, and 3e-4 for rank 64 if triggered
by the current evidence. Reuse the existing prepared held jobs, native gates
and remaining 7.9 GPU-hour/38 GB LoRA campaign allowance. The original selection
receipt is never rewritten. The new full pass cannot trigger recursive tuning.
All adapters remain digital on the same faulty MLPs; physical writes stay zero.

**Dense:** the trigger already holds: development KL rose from 0.136692 to
0.402117 at LR 5e-3. Screen constant normalized-state Adam LRs **3e-4, 1e-3,
3e-3, 1e-2 and 3e-2**, each for one pass over the same 8,192-sequence recovery
subset (2,097,152 input tokens; 256 updates). The grid spans both sides of the
original LR because repeated noisy writes can make LR effects nonmonotonic.
The physical law, dense write mask, fixed digital model and uncapped writes are
unchanged. Select minimum final development teacher KL, exact ties lower LR.
Only if the best screen improves its unchanged-deployment development KL by
at least 2%, release one fresh full 49,977,856-token pass (6,101 updates).
Otherwise stop with the screen's negative result. Any failed mandatory native
validation gate blocks full dispatch. A worse full endpoint is retained; this
bounded follow-up does not recursively expand the grid.

Train on adaptation tokens only. Use the 256-sequence development cohort for
all LR decisions. Generated and WikiText test metrics remain reporting outputs;
they never enter triggers, LR selection or checkpoint selection. Retain fixed
final endpoints. Report dense device writes, stuck attempts, touched devices,
out-of-range states and verify reads as in exp-008. The short screens are extra
tuning exposure, separate from each matched full pass.

## Execution and budget

Use the existing single persistent run-watch dispatcher and durable ledger.
LoRA completion callbacks apply the declared rule automatically; active workers
and scientific training code are unchanged. Dense screens use the idle loulou
RTX5090 in sequential detached tmux jobs; all full candidates are prepared but
only the selected one can pass its gate. The five screens plus at most one full
fit have a **2 GPU-hour cap**, within the unused original 4 GPU-hour dense
allocation, and **20 GB additional storage**. They reserve 26,000 MiB plus
2,700 MiB headroom, using the measured exp-008 peak and throughput. Expected
elapsed time is 20–60 minutes including a conditional full fit and collection.
The compute cutoff remains **07:30 Paris October 12**, before delivery at 09:00.
No operational retry grants a fresh budget or changes the scientific recipe.

Strict dense lifecycle files are
`campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp4-dense-lr-{LR}-{dev,50m}.json`.
The public preparation/collection helper is
[`exp4_dense_lr.py`](../../../launch/exp4_dense_lr.py). Native plans, validated
compact bundles, selection receipts and per-case reports live under
`results/lifecycles/opt125m-cmo-l6-exp4-dense-lr/`.
LoRA authorization and per-rank decisions live in exp-009's existing
`readiness/lr-followup-policy.json` and `readiness/lr-followup-r{rank}.json`.

## Handoff

Dense queue campaign: `opt-cmo-exp4-dense-lr-20261012`. Job IDs are
`opt125m-cmo-l6-exp4-dense-lr-{3em4,1em3,3em3,1em2,3em2}-{dev,50m}`.
No independent monitoring daemon or live interactive coordinator is required.
The shared service polls semantic progress every five minutes, invokes bounded
read-only investigators for anomalies, validates native artifacts and collects
reports. Preparation and first live observation must be recorded before calling
the handoff armed.

All ten dense native plans and dry-runs passed on loulou at source
`4e2ddc74011676556ba315c8790b723e91f1e90d`. The local dense collection controller
uses that same source. Root and deployed LoRA decision/replay checks passed
(11 tests); its completion controller is `93f8bdf`. The policy was exercised on
the completed rank-4 endpoint and correctly requested no follow-up.

The dense manifest SHA-256 is
`da3876f7a6af01f9bbaa70acc82f8ed47ac2a0cc6430fe01ff150b690e9b7c6b`.
Five development jobs were released; all five full candidates remain held.
Native validation and full-coverage development selection gate their release.
The prepared source and receipt trees plus local collection root are included in
storage accounting (0.433 GB before execution). A controller-storage amendment
initially rejected the local collector of this remote-only campaign; the tested
client fix at `8fa415c` admits only the prepared controller directory or its
immediate evidence parent, without adding any GPU placement. Dispatcher and
decision regression checks pass 145 tests. Submission and storage receipts are
in `readiness/submission.json` and `readiness/storage-preflight.json`.

The first dense screen, LR 1e-2, started on loulou in attempt
`31cec0bd192641c8b9617fa50c33f413`. The immediate live probe confirmed both tmux
processes and finite training loss at update 200/256; receipt:
`readiness/first-live-probe.json`. The other four development jobs wait for that
same GPU. The existing `run-watch-queue.service` remains their sole dispatcher
and monitor; the local completion controller collects each validated endpoint
and evaluates the full-fit selection gate only after all five screens finish.

That first screen subsequently completed, including native validation and
automatic compact collection, charging 96.42 GPU-seconds. The original
`readiness/unattended-handoff.json` reported a passing handoff from worker and
heartbeat checks. The morning review found that this missed an existing storage
observation incident; it was not sufficient evidence of healthy dispatch.

Morning outcome: all three LoRA fits completed without triggering retuning.
Only dense LRs 1e-3 and 1e-2 completed; the remaining three development jobs
expired at their deadline without launching. No full follow-up was selected.
The local storage-probe bug, repair and partial scientific evidence are recorded
in the [result note](../results/exp-010.md). Execution history, charged budgets,
original failed handoff evidence and both LR-selection policies are preserved.
Further GPU execution requires an extended window beyond the user's 09:00
deadline; the operational correction alone does not authorize that extension.
