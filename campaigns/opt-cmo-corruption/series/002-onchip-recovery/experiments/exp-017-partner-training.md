---
id: "exp-017"
title: "On-chip learning through the verified partner after fault-aware rewriting"
status: "complete"
hypotheses: []
---
# exp-017: train through the verified partner

The user assigned implementation and execution on October 14, 2026, selecting
**std3 only** and the unchanged training recipe. This is a new comparison after
[exp-016](../results/exp-016.md): that experiment's learning arm updated only
original active devices, including when a healthy reference had absorbed a
stuck active. No hyperparameter search or extension to other sources is assigned.

## Frozen comparison and decision

- OPT-125M's last six MLPs, layers 6–11: 28,311,552 logical weights and
  56,623,104 devices. Same pinned teacher and frozen exp-007 std3 source.
- Exact exp-013 held P0s, assignments 271002–271004; select `nominal` and
  `random_30000ppm` without changing upstream defect declarations/provenance.
- Four arms: original hold; verify pair rewrite then hold; the same rewrite
  then original-active training; the same rewrite then verified-partner training.
  Three native jobs, six starting conditions, 24 endpoints, 12 learning fits.
- Rewrites match exp-016 exactly: 5-sigma detection, seed 241016, one apparent
  per-device scan, fitted 2% CMO endpoint programming. Targets and subtraction
  polarity stay unchanged. No calibration or adapter.
- New routing uses that original scan only: unflagged active -> active;
  flagged active/unflagged reference -> reference; both flagged -> blocked.
  The map stays fixed. Reference writes use the opposite delta-q sign. Hidden
  faults remain analysis-only; missed faults consume attempts and never heal.
- Same exp-013/016 recipe: thresholded Adam, LR .003, accumulated nominal
  2-sigma threshold, physical write noise, four issued attempts per selected
  physical device after rewriting, 256 updates, batch 32, one epoch over 8,192
  sequences / 2,097,152 input tokens. Training seed 53; data seed 20261008.
- Same separate development/generated-test/WikiText-test cohorts: 256/384/1,024
  sequences at length 256. All evaluation uses held apparent state. Final
  checkpoint fixed before running; development every 64 steps is diagnostic.
- Primary teacher KL; report perplexity. Separately for each case/domain/control,
  verified-partner learning improves consistently only if final KL is lower on
  all three assignments, harms if higher on all three, otherwise mixed.
  Compare with rewrite hold and original-active learning. No significance claim
  or general physical-viability claim follows from a sign test.

Expected possibility: opening the stuck-active weights to reference updates
improves the remaining error. Alternative: write noise or the unchanged recipe
dominates, and reference updates add cost without a held-out gain. Either outcome
is retained; scientific quality never triggers a retry.

## Implementation and validation

The optional arm field `write_roles: "verified_partner"` extends the native
thresholded CMO writer. The controller retains an int8 routing map from the
rewrite, consuming 28,311,552 bytes. No additional verify reads are introduced.
Counters distinguish active/reference learning writes, issued/stuck attempts
and touched devices; rewrite programs/reads remain separate. Map, moments,
pending commands, counters and RNG states are serialized. Legacy configurations
and active-only trajectories preserve their behavior; final-state replay is
required, while interrupted-stage resume remains unsupported.

CPU tests cover reference polarity, fixed partners, all detection combinations,
missed faults, blocked pairs, cap accounting, legacy equivalence and native
tiny-OPT execution/replay. Run the workflow suite and campaign-specific checks.
The first real array is the canary: its original hold, rewrite and original-active
endpoints must reproduce exp-016 state hashes and KL within 1e-6 relative before
the other two jobs are released. Every learning arm starts from the identical
rewrite-state hash, with matched rewrite receipts and input hashes.

## Budget and execution

Caps: **2 GPU-hours, 80 GB of new declared artifacts, 24 hours**, including
readiness and understood operational retries. Reuse previously staged P0s and
datasets without copying or modifying them. Preserve failed/partial evidence.
The unchanged network/batch has prior measurements: exp-016 jobs used about
310 seconds and 17.25 GB peak allocated GPU memory. Two learning arms replace
the former single learning arm; reserve 650 seconds and 15 GB per job, with
21,500 MiB peak plus 2,048 MiB headroom. Canary measurement checks these estimates.

Idle fifi/loulou 5090s are the targets. The shared queue is the sole monitoring
owner, with native validation and controller collection. Use explicit safe
windows within 09:00–17:20 and/or 22:00–07:20 Paris because the generic queue's
daytime policy is narrower than the repository's. Worker timeout is 1,800 seconds;
per-job total is 2,100 seconds and the aggregate two-hour cap remains controlling.
No shared policy, unrelated job, or GPU setting is changed.

Evidence: `results/lifecycles/opt125m-cmo-l6-exp17-partner-training/`.
Lifecycle: `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp17-partner-training.json`.
The launch helper plans one native on-chip stage per array using public CLIs.
Collection revalidates native exports and prior controls, releases jobs through
the canary gate, and writes complete coverage and per-array decisions after all
three collect. The result note will own interpretation.

## Execution handoff

CPU validation passed: 477 workflow tests and one optional skip, including
12 new partner-training checks; two campaign coverage/control tests also passed.
The strict lifecycle digest is `4305f2580a47c94e88ed88de3df9c0cc5ca0fb40dfb9d90dc83da5d38525507f`.
The full reusable lifecycle retains five upstream assignments for provenance
(40 endpoints, 48.7 GB estimated); the execution manifest selects only the
three approved assignments (24 endpoints, about 29.2 GB of estimated state).
Exact source revision, placements, manifest, deadline and monitoring observations
will be recorded before release. Model-based assumptions remain programmable per-weight
references, synthetic permanent faults, fitted endpoint programming, no read
noise or drift. Programming/write events are not measured pulse counts.

### Registered execution, October 14

- Worker and collection-controller source: `d12d779f04522b41a2b13efaeebb5abd1bb5b746`; shared-worktree implementation commit: `2a65d1c`. The isolated source excludes concurrent retention changes.
- Workers on fifi/loulou: `/home/filip/analog-research/opt-cmo-exp1/hwa-exp17-partner-training-20261014`. Both hosts passed all three native dry runs and full frozen-input validation. Inputs reuse the exp-016 staging; no new input copy was made.
- Campaign `opt-cmo-exp17-partner-training-20261014`, manifest SHA-256 `7fbe1a9c70d399716b3e8b3be12556f5b5b94d62f6354295599031cb8c181a10`. Three exact job specs and placements are under `prepared/queue/`; preflight passed. Only array 271002 is initially released; 271003/271004 require its successful collection gate.
- Deadline: `2026-10-15T11:27:20.145986+02:00`. Explicit safe admission windows are pinned in `readiness/execution-window.json`; all charges remain inside 2 GPU-hours / 80 GB.
- Shared `run-watch-queue.service`, PID 1384148 at registration, owns admission, five-minute semantic checks, native validation and controller collection. The local pinned controller is `<evidence>/controller/`; automatic final review writes `reviews/summary.json` and `reviews/summary.md` after all three jobs collect. There is no duplicate watcher.
- Job IDs: `opt125m-cmo-l6-exp17-partner-training-std3-{271002,271003,271004}`. `reviews/progress.json` tracks collected endpoints. `prepared/{fifi,loulou}.json` names worker receipts, native roots, input hashes and exact commands.
- No automatic compute retry is enabled. Operational failures require diagnosis within the remaining allowance; collection is separately retryable without rerunning successful compute. Numerical results never trigger retuning or replacement.

- Initial handoff: array 271002 started on fifi, attempt `c414c1fadabd4c2889b98ab3f2997f17`; see the incident below for its terminal state. Arrays 271003/271004 remain held behind the canary gate. Evidence: `readiness/handoff-observation.json`.

### Canary driver failure and unchanged restart, October 14

The first canary failed at 11:33 Paris with native return code -11. The kernel
records PID 1950362 faulting inside `libcuda.so.610.57.04`. Three nominal control
endpoints were written; the new partner arm reached step 256 but did not finish
endpoint evaluation or export. These are partial artifacts, not a completed
canary or a scientific result. The queue correctly withheld the two later arrays.

The failed worker, supervisor and GPU reservation are confirmed terminal. Its
174.318738 seconds remain charged. A guarded, full-size GPU accounting diagnostic
passed in another 1.277132 seconds; this extra diagnostic time also counts against
the original two-hour allowance. Passing the diagnostic does not establish that
the driver fault is fixed.

Pinned evidence is `readiness/canary-driver-failure.json`; the public
`campaigns.execution_service recover-driver-crash` command accepted
`readiness/canary-driver-recovery.json` (receipt SHA-256
`a031026c23bd8105d1052231b68483a65a6eadce28b3aff27c2b60e5e1aa0f3d`).
The one authorized restart uses original inputs and the unchanged frozen source,
excluding fifi for this canary and retaining its already prepared loulou placement.
No scientific setting, deadline or resource allowance changed. Failed fifi
artifacts remain intact. The shared queue remains the only admission and monitoring
owner; the canary's successful collection is still required before release of
arrays 271003/271004.

At 12:01 Paris the restarted canary is running on loulou, attempt
`4f793a3a903f45579ce24cd554448162`, native PID 1942639. Fresh metrics contain
the nominal original-active arm's first training batch after its hold/rewrite
controls; NVIDIA reports 17,912 MiB allocated to this process. The shared queue
has acknowledged the live worker, with arrays 271003/271004 still held. This is
progress evidence, not canary completion. The saved observation is
`readiness/recovery-handoff-observation.json`.

### Two arrays collected; remaining driver retry, October 14

Arrays 271002 and 271004 completed on loulou and passed native collection,
exact replay, physical accounting and exp-016 control reproduction: 16 of the
24 endpoints are validated. The canary gate released both later arrays as
specified. Preliminary measurements belong to the [result note](../results/exp-017.md).

Array 271003's initial attempt `88c9f13657ad496cbde812595d825b3f` failed on
fifi at 12:13 Paris, again with return code -11 and a kernel fault inside
`libcuda.so.610.57.04` (PID 1962365). Its worker and reservation are terminal;
141.263882 seconds remain charged. The public driver-recovery command accepted
`readiness/271003-driver-recovery.json`, backed by
`readiness/271003-driver-failure.json`; receipt SHA-256 is
`f640e10d5372b6f824f24f3eb2c8e3722275848e8d0db0a64dd5f094ae8475e7`.
The same array is queued for its one unchanged restart on prepared loulou;
fifi is excluded from that job. Completed arrays are not rerun. Before this
restart, total charged execution plus the earlier GPU diagnostic is 1,105.251
seconds (0.307 GPU-hours) of the original two-hour allowance. The deadline,
inputs, code, recipe and final three-array decision rule remain unchanged.

### Complete and reviewed, October 14

Array 271003 completed on loulou in retry attempt
`aae28af34dda4d45b9969d0ad4388b1f`. All three jobs are collected and validated:
six starting conditions, 24 endpoints, 12 learning trajectories. The frozen
controller's `exp17_partner_training.py review --evidence <evidence-root>` passed
at the final review, rechecking collection hashes, native exports, prior controls,
exact replay, costs and all eight predeclared comparisons. The
[result note](../results/exp-017.md) records the full three-array conclusion;
it supersedes the earlier partial summary.

Total queue charge is 1,502.481636 seconds including both failed attempts;
with the 1.277132-second diagnostic this is **0.418 GPU-hours** of two.
Declared storage roots occupy **28.25 GB** of 80 GB at the final observation.
No experiment jobs remain active or queued. The unchanged recipe did not improve
WikiText teacher KL after rewriting on any of the three faulty arrays; no tuning
or further compute was launched in response to that result.
