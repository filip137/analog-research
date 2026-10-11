---
id: "exp-006"
title: "Corruption-aware HWA with smaller digital learning rates"
status: "complete"
hypotheses: []
---
# exp-006: corruption-aware HWA with split learning rates

## Question and authorization

On 2026-10-11 the user approved the proposed next comparison and requested prompt
execution while the two RTX 5090s remain free until the following morning.
Does corruption-aware HWA benefit from the smaller digital learning rates that
improved standard HWA, with analog training and the selection rule held fixed?
Run two full fits on fifi/loulou, then deploy only the development-selected
checkpoint. The two-additional-training-seed confirmation is a later phase with
its own cases and budget; this phase does not dispatch it or on-chip learning.

Evidence: [exp-005](../results/exp-005.md#completed-frozen-checkpoint-transfer-comparison-2026-10-11).
Standard HWA's digital peak 1e-5 gave modest held-out gains under corruption,
below the 10% material-effect threshold, with a small healthy WikiText cost.
Existing corruption-aware analog HWA is a stronger faulty-array control but has
a larger healthy-performance penalty. Its selected mean development KL is
0.07576591790306803 at update 25,088. These references are one training seed,
not new independent replicates.

## Frozen comparison

| Setting | Value |
|---|---|
| New training cases | Corruption-aware HWA, model scope, digital peaks 1e-5 and 1e-4 |
| Reused zero endpoint | Existing `cdt_random_3pct` analog-only source from exp-005 |
| Analog/calibration LR | 5e-3 peak, cosine decay to 1e-3 |
| Digital schedule | Same factor: finals 2e-6 and 2e-5 |
| Training budget | 205 warm-up updates, 25,600 updates, batch 32, 4.196 corpus passes |
| Initialization | Original pinned FP32 OPT-125M teacher, not a prior trained source |
| Network | Last six analog MLPs, 28,311,552 logical analog weights; other components digital |
| Data | Pinned synthetic 50M-token corpus, 195,226 sequences of length 256; original seed/order and HWA seed 41 |
| HWA noise | Per-output-channel abs-max Gaussian 3%, plus temporary i.i.d. 3% random stuck active/reference masks |
| Selection data | Original 256 development sequences and arrays 211001/211002, healthy and 3% permanent random faults |
| Selection rule | Minimum mean healthy/3%-fault KL; compare peaks 0, 1e-5, 1e-4, within 1% of the minimum prefer smaller LR |
| Unchanged optimizer | Adam defaults; no weight decay/gradient clipping; q clamp [-1,1]; original calibration group |
| Selected-source deployment | Five assignments 271001–271005; endpoint offset 10000; permanent random faults 0–5% separately |
| Test domains | Original pinned WikiText 1,024 sequences and generated 384 sequences, length 256 |

The zero endpoint is the existing analog implementation, supported by the prior
zero-digital equivalence checks. It is not silently relabelled as a new model-scope
fit. Source and comparison receipts are pinned in
`campaigns/opt-cmo-corruption/launch/exp1f_references.json`. The existing standard
grid is reused; each method chooses its own digital rate from the same three
values. Standard 9% analog remains an additional robustness reference.

This retains one joint healthy/faulty checkpoint per method. No separate healthy
or 3%-specialist checkpoint rule is introduced. The original best-development
selection and late-convergence flag remain. Test metrics never affect selection.
If zero wins, reuse its existing complete test deployments; do not duplicate them.
Otherwise run exactly one new 30-row deployment job, with both domains per row.

Both HWA reference papers were consulted. Per-channel Gaussian noise and synthetic
distillation follow the Büchel-inspired choices; backward propagation through
noisy weights follows the Rasch-style choice. Corruption masks, the split LR grid,
cosine floor, Adam settings, token budget and best-development checkpoint remain
campaign-specific choices, not a paper-matched recipe. No clipping, noise ramp,
quantization, objective or optimizer changes are bundled into this comparison.

## Decision and limits

Report both development components and clean KL for every candidate. On test
arrays, report KL and perplexity and paired differences/wins separately for each
fault rate and domain, against corruption-aware analog, selected standard 3% and
standard 9% analog. Retain the descriptive material-effect rule: at least 4/5
array wins and at least 10% mean paired relative KL reduction. Healthy costs
remain explicit. A smaller or negative gain is valid evidence, not a retry cause.

This is exploratory, model-based and adaptive, with one training seed and reused
test cohorts/arrays. The result does not establish optimal rates, independent
training reproducibility or on-chip recovery. Freeze candidate recipes for later
seed confirmation only after this comparison is reviewed.

## Execution, budget and ownership

Strict lifecycle files: `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1f-cdt-*.json`.
Preparation/completion: `campaigns/opt-cmo-corruption/launch/exp1f_cdt_lr.py`.
Evidence root: `results/lifecycles/opt125m-cmo-l6-exp1f-cdt-lr/`.
Remote source/results: `/home/filip/analog-research/opt-cmo-exp1/hwa-exp1f-cdt-lr-20261011/`.
Execution phase: `opt-cmo-exp1f-cdt-lr-20261011`.

Incremental cap: **6 GPU-hours, 16 GB retained storage, six hours from first
admission**, and no worker beyond **2026-10-12 08:00 Paris**. Prior whole-model
corruption-aware training took 6,113 s; expect about **2–2.5 h on the two 5090s**
including readiness, deployment and collection, around 4 GPU-hours. Each full
worker has a 9,000 s hard allowance; the sole 512-update canary has 900 s; each
conditional deployment has 1,200 s. Training reserves 21,500 MiB plus 2,200 MiB
headroom; deployment reserves 17,000 MiB plus 2,200 MiB. One worker per GPU.

One short fifi canary combines the temporary corruption masks with digital peak
1e-5. It uses 512 updates and a separately declared 512-update cosine horizon,
retaining 205 warm-up updates. Its scores are readiness evidence only. Both
full fits remain held until native validation/collection and the immutable
handoff gate pass. Fifi then trains 1e-5, loulou 1e-4 concurrently. Both fits
must validate before the frozen development rule can release one deployment.
Nonselected deployment alternatives are cancelled before any attempt, with their
history preserved. No automatic compute retry or scientific configuration change.

The existing `run-watch-queue.service` is the sole dispatcher/monitor, with its
durable SQLite ledger, locks, idempotent submissions, bounded read-only incident
investigator and self-heartbeat supervision. It polls campaign semantics every
five minutes and distinguishes queued work from active workers. Expected native
stage durations: prepare/devices 120 s, HWA 6,600 s (canary 300 s), deployment
180 s per array, validation and collection 600 s. Startup 600 s and stale 900 s.
Reuse prior verified dispatcher restart, investigator and duplicate-prevention
receipts for the unchanged service; test the new conditional completion logic.
Any mandatory failed gate or insufficient remaining budget stops downstream work.

Native execution uses `workflow plan` and `ebl campaign run` in detached tmux.
Training collection explicitly requires source-only coverage; selected deployment
requires deployment-only coverage, full original-source artifact validation and
exact source lineage. Completion hooks run independently of this interactive
session, collect compact native evidence and produce development/transfer reports.
Large artifacts and failed/partial evidence remain on their execution hosts.
Exact prepared specs, hashes, attempts and first semantic observations are
recorded in the handoff below.

## Completed execution and monitoring handoff

Frozen worker/controller revision: `176bce6bd226918858c701f84bf394178e6011d3`.
Campaign manifest SHA-256:
`abd7482f5b0c3b4cad0e1d7bcfed085435581912443431f4bb223e109f00431b`.
Both hosts passed pinned teacher/corpus checks and native planning/dry runs.
Each full lifecycle describes prepare, devices, HWA and five deployments;
execution slices run only three stages for training and seven for selected-source
deployment, with zero on-chip trajectories. The source-dependent sections of the
reused analog control match the new candidates outside the declared scope/LR
intervention. All five strict lifecycle files passed `workflow check`.

145 focused checks passed, covering conditional selection/replay, invalid or
partial coverage, native collection and the existing durable execution service.
Repeat preparation/registration produced the same manifest hash and no duplicate
jobs or attempts. The unchanged dispatcher's prior live restart, investigator
invocation and ownership receipts were hash-verified. Service PID 342022 is active,
restarts on failure, has an 11-minute systemd watchdog and user lingering enabled;
four of six rolling-day investigator invocations remain available. Initial
incremental storage is 0.9835 GB. Preflight evidence is in `readiness/preflight.json`.

The fifi canary completed in attempt
`a1777f0165764adabcf16ea4e681113e`, native worker PID 1120506. Its tmux session was
`opt125m-cmo-l6-exp1f-cdt-lr1em5-smoke-a1777f0165764adabcf16ea4e681113e`.
The service observed its live worker and GPU reservation. Direct native
inspection confirmed prepare/devices completion and finite HWA progress; all
512 updates subsequently passed native validation and collection. HWA took
132.493 s with 19.711 GB peak allocated memory; total charged worker time was
145.893 s (0.040526 GPU-hours). Final rates were 1e-3 and 2e-6. All 26 compact
file hashes were rechecked. The first observation is
`readiness/canary-first-observation.json`; the passed immutable gate is
`readiness/handoff.json`.

The canary completion hook passed the handoff gate and released both full runs
without an interactive release. Full-fit IDs are
`opt125m-cmo-l6-exp1f-cdt-lr1em5` (fifi) and
`opt125m-cmo-l6-exp1f-cdt-lr1em4` (loulou). Conditional evaluation IDs insert
`-transfer` before `-lr`; only the development-selected alternative may launch.
Exact placements, native manifests and worker specs are in
`prepared/{fifi,loulou}.json`; all five queue specs and the idempotent registration
receipt are under `prepared/queue/`. Completion hooks use the clean local
`controller/` checkout. Native tmux names are `<job-id>-<attempt>`.

Native run roots are specified exactly by each placement's
`campaign_result_root`; retained logs are under
`/home/filip/analog-research/opt-cmo-exp1/worker-receipts/176bce6bd226918858c701f84bf394178e6011d3/<job-id>/<attempt>/run.log`.
Collection writes `collected/source/`, the immutable `selection.json`, then
`collected/transfer/` and `collected/transfer-review.json`. If zero wins, the last
report explicitly identifies reused test evidence. No live coordinator is needed
for these frozen transitions. Leave the shared service running after this phase.

Both full fits started around **15:14:25 Paris on 2026-10-11**. Their first
observations confirmed live workers, reservations and finite progress at update
206/25,600 with the correct group rates. Historical observations are
`readiness/{fifi,loulou}-first-full-observation.json` and
`readiness/full-handoff.json`.

At the **17:47 Paris completion check**, the canary, both 25,600-update fits and
the selected-source deployment were complete, validated and collected. The
service selected digital peak **1e-5** from development data alone and evaluated
its frozen checkpoint on all five arrays and six fault rates. The unselected
1e-4 deployment was cancelled before any attempt, as required by the frozen
winner-only rule. This is complete declared coverage, not a missing test run.

| Host | Completed work | Attempt | Charged GPU-hours |
|---|---|---|---:|
| fifi | 512-update canary | `a1777f0165764adabcf16ea4e681113e` | 0.040526 |
| fifi | Digital peak 1e-5 fit | `a6e0289cbcbf439eab24303b19bc1ed9` | 1.730585 |
| loulou | Digital peak 1e-4 fit | `f5cd534869a54627841159f92f1f12e3` | 1.701213 |
| fifi | Selected 1e-5 deployment | `c51a3dc1611c4b35b6187d937e8ba603` | 0.091384 |

All four executed attempts have execution and validation exit codes zero. All
140 compact-file hashes across their four collection receipts were rechecked;
the development selection was reproduced, and transfer source lineage and
matched physical identities were verified. No incident, duplicate launch or
compute retry occurred. Native deployment `attempt-002` follows the preparatory
dry run; it is not a second GPU attempt after a failed execution.

Final charge is **3.563708 of 6 GPU-hours**, leaving **2.436292 GPU-hours**
unspent, with no active reservations. Retained storage is **7.242 of 16 GB**.
`readiness/latest-observation.json` records the terminal job states and preserved
budgets. The shared watchdog remains active with a fresh heartbeat and no host
probe errors. Both 5090s have no compute processes or reservations at this check;
their experiment tmux workers have exited normally. Logs and native artifacts
remain at the paths above.

The [completed result and review](../results/exp-006.md) records the development
choice, all held-out comparisons, healthy-performance costs and material-effect
verdict. Small digital updates help corruption-aware HWA, but the improvement
over its analog-only counterpart is below the declared material threshold at
3% faults. Additional training seeds remain a later phase; nothing further was
dispatched. The automatic transfer report retains its collection-time
`awaiting_scientific_review` marker; the linked result note owns the completed
scientific review.

## User decision after review, 2026-10-11

The user confirmed: “ok I'd freeze this lr”, then asked to recall the scientific
questions and design the larger training comparison. Freeze the HWA analog and
existing calibration peak/final rates at **5e-3 / 1e-3**, and the additional
digital peak/final rates at **1e-5 / 2e-6**, with the current warm-up/cosine form.
These are HWA rates, not rates for future on-chip adaptation. The larger run's
proposed cases, replication and budget are in
[exp-007](exp-007-corruption-rate-sweep.md); the completed exp-006 configurations and
source identities are unchanged.
