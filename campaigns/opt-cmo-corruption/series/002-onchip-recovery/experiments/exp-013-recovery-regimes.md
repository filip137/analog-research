---
id: "exp-013"
title: "Recovery across three HWA recipes, six fault rates and three assignments"
status: "complete"
hypotheses: []
---
# exp-013: recovery across HWA and corruption regimes

## Question and authorization

On October 12 the user approved extending recovery to standard 3% Gaussian HWA,
3%-fault-aware HWA and standard 9% Gaussian HWA, actual faults 0–5%, caps 2 and 4
plus uncapped recovery, a frozen recipe and three assignments. The user then
instructed implementation of the complete plan on one free 5090 and two 3090s.
This asks how much additional performance recovery supplies after HWA and how
much of that improvement survives very few writes per device. It does not claim
optimized recovery or superiority over rewrite, calibration or digital LoRA.

Prior evidence: [exp-007](../../001-hwa-robustness/results/exp-007.md) compares
matched HWA sources; [exp-011](../results/exp-011.md) selects accumulation and
shows little full-pass benefit; [exp-012](../results/exp-012.md) establishes the
single-array write-cap curve. Source training remains frozen; no new HWA choices
or departures from its recorded paper-informed protocol are introduced here.

## Frozen comparison

- Sources `std3`, `cdt3`, `std9`: exact selected exp-007 whole-model sources.
  Six analog OPT-125M MLPs, 28,311,552 logical weights; other components digital.
- Actual fault rates 0%, 1%, 2%, 3%, 4%, 5%, permanent random stuck devices,
  same exact-count homogeneous placement and CMO device/programming model.
- Assignments **271002, 271003, 271004**, chosen before new recovery results.
  The earlier development assignment 271001 is excluded from this comparison.
- Each condition starts from its own named held P0. Three independent recovery
  trajectories restart there: 2, 4 or unlimited issued attempts per active cell.
  Every native run also contains the unchanged-deployment arm. Reference cells,
  digital weights and calibration are held; faults cannot be repaired by writes.
- `cmo_thresholded_write_adam`, constant LR 0.003, signed accumulation, nominal
  2-sigma threshold, no verify/retry/clamp and no residual flush. Stuck attempts
  consume caps; initial programming is excluded. These are idealized write events,
  not measured hardware pulses.
- 8,192 sequences of 256 tokens: 2,097,152 tokens, batch 32, 256 updates, one pass.
  Preserve data seed 20261008 and on-chip seed 53. Fixed final checkpoint; evaluate
  development every 64 updates without selecting intermediate states.
- Existing disjoint cohorts: 256 development sequences, 384 generated-test and
  1,024 WikiText-test sequences, length 256. Preserve pinned corpora and teacher.

Coverage is **54 starting conditions and 162 recovery endpoints**. Nine lifecycle
variants (source × cap) retain original upstream provenance, including the original
five-assignment device catalog; native execution slices select only the declared
three assignments. Twenty-seven jobs each execute all six fault rates with paired
held/learned trajectories. Duplicate held controls are never extra replicates.

## Measurements and decision

Report absolute teacher KL, perplexity and paired change from the same P0 by
source, rate, assignment and domain. Compare final performance against the best
HWA-only endpoint among the three frozen sources on that same assignment/rate.
Retained recovery is `(initial KL - capped KL)/(initial KL - uncapped KL)`;
report null when uncapped improvement is nonpositive, and never clip negative or
above-one values. Such scientific outcomes are valid, not operational failures.
Show all three assignments, mean, range and paired signs, with no pooling of rates
or domains and no test-selected source, cap, LR or checkpoint.

Report maximum/total attempts, successful writes, touched cells, stuck attempts,
verify reads and out-of-range states. Completion requires all 162 validated
endpoints and review. No automatic follow-up, full-pass extension or retuning.
One HWA training seed and reused test cohorts retain exploratory, model-based
limits; the three chips have not been used to tune this recovery recipe.

## Implementation, budget and execution

Helper: `campaigns/opt-cmo-corruption/launch/exp7_recovery_regimes.py`.
The adjacent `exp7_recovery_sources.json` pins nine original deployment bundles,
their input/source hashes and all 54 held apparent states. Materialization on the
5090 preserves each state without new writes before transport to 3090 workers.
Frozen original bundles are never edited. New lifecycle IDs begin
`opt125m-cmo-l6-exp7-recovery-regimes-`; all execution uses the public native CLI.

Targets: **loulou RTX5090, nom-cool-1 RTX3090, local RTX3090**, one worker each.
The local six-job allowance is standard HWA/271002 and 9%-Gaussian/271003, all
three caps. Project at least 40 GB free local storage after their completion.
Budget **12 GPU-hours**, **600 GB additional storage**, **24 hours from first
admission**, including the preparation job, initial runs, validation and retries.
The same durable ledger accounts for two execution phases: up to 0.5 GPU-hours
for held-state materialization, then 11.5 for recovery. Main preflight requires
successful preparation; the final job deadline retains the first phase's 24-hour
cutoff, so phase registration cannot reset the total allowance or time window.
Previous short runs used about 106 seconds each on a 5090 and 19.99 GB reserved
CUDA memory; the first jobs calibrate the new six-rate stages and 3090 placement.
Preliminary elapsed estimate is 3–5 hours after preparation, not a guaranteed ETA.

First jobs are cap 4 on assignment 271002: std9 on loulou, cdt3 on nom-cool-1,
std3 locally. These are full, predeclared scientific jobs, counted if valid.
The remaining 24 jobs require all three initial validations and the handoff gate.
No loss-improvement threshold gates dispatch. A failure of physical/artifact
validation, exhausted budget or elapsed window blocks downstream admission.

One existing persistent run-watch service owns preparation, execution and
collection, with five-minute monitoring, durable locks and idempotent submissions.
Workers run in tmux; queue ownership survives the interactive session. Read-only
bounded incident investigation preserves exact manifests, logs and remaining budget.
Only classified operational recovery within the original allowance is permitted.
Never alter scientific settings or steal another workload's GPU.

Evidence root: `results/lifecycles/opt125m-cmo-l6-exp7-recovery-regimes/`.
The frozen GPU worker checkout on each host is
`/home/filip/analog-research/opt-cmo-exp1/hwa-exp7-recovery-20261012`.
Preparation checks input binding, native plans and dry runs; execution checks
finite metrics, exact held-state replay, fault persistence, full coverage and cap
accounting. Final plots show recovery versus corruption and writes by source.

## Current execution handoff

**Reviewed October 13: complete.** All 27 recovery jobs and 162 endpoints passed
native validation and local collection; the final worker finished October 12 at
22:22 Europe/Paris and collection completed by 22:24. The [final result](../results/exp-013.md)
records the scoped scientific review, all-rate comparisons, write costs and
limitations. All 297 compact artifact hashes passed the final audit. Execution
used 7.616 GPU-hours including preparation; the main monitor observes 353.7 GB
against the overall 600 GB allowance. There were no execution retries or
excluded endpoints. All exp-013 GPU reservations have ended. No further jobs or
scientific choices are dispatched by this comparison.

The preparation storage monitor raised incident 17 after its job completed:
its 120 GB sublimit covered the whole loulou checkout, including later recovery
outputs. The result reconciles this overlapping scope and the read-only diagnosis;
the incident and original manifest remain preserved. Native results are valid.
The service's generic `awaiting_scientific_review` label denotes terminal jobs;
this note and the result record the completed scientific review.

The following snapshots are retained as execution history.

**October 12, 20:24 Europe/Paris:** eight jobs / 48 endpoints have completed
collection and validation; three jobs are running and 16 queued. All initial
gates passed automatically. The [partial result](../results/exp-013.md) records
the first matched comparisons. Native elapsed times are about 9.4 minutes on
the 5090 and 23–25 minutes on the 3090s; allow another 2–3 hours including queue
turnover and collection. This remains an estimate.

All remaining inputs have reached nom-cool-1. Fourteen still-unstarted jobs
received validated additional placements through amendment
`942471495d9283ee852e4ebac22d6fce33e4f539765dc167b9029d1998a79edd`;
their identities, scientific inputs and original budgets are unchanged. The CPU
controller advanced to `7bf27fd2579e019930425c23644f2e5c12b6cbfc` for a report-only
fix: uncapped native bundles record successful and stuck writes separately, but
omit the cap controller's attempted-write counters. Total attempts are their
sum; the unrecorded per-cell attempt maximum remains null, and the recorded
maximum successful writes is retained. Eleven focused tests and a full synthetic
figure render passed. GPU worker revision remains `9adbd20` and all running
workers survived the brief controller maintenance. Evidence is in
`readiness/reporting-and-placement-update.json`.

The initial launch handoff follows as execution history.

**October 12, 19:12 Europe/Paris:** all 27 jobs are registered; the three initial
cap-4 jobs are running, and the other 24 remain held behind their validation gate.
The existing service has observed all three live workers and native semantic
progress, with finite metrics and no artifact errors. This is execution progress,
not completed scientific coverage.

| Host | Initial job suffix | Attempt |
| --- | --- | --- |
| loulou RTX5090 | `std9-271002-cap4` | `ce0abf35cc9a48b59c5454818d33e860` |
| nom-cool-1 RTX3090 | `cdt3-271002-cap4` | `af6a73cfb7074082a87e60fb7585311f` |
| local RTX3090 | `std3-271002-cap4` | `0d3cd4a8fa4d4d778c19137966c4659c` |

Each job ID is `opt125m-cmo-l6-exp7-recovery-regimes-` plus its suffix. Its tmux
session adds `-` and the attempt token. On the corresponding host, for example:
`tmux attach -r -t opt125m-cmo-l6-exp7-recovery-regimes-std3-271002-cap4`.
Exact handles and log paths are in `readiness/{local,nom-cool-1,loulou}-live.json`
under the evidence root. The prefix is unambiguous for these first attempts.

GPU workers are frozen at `9adbd20b7cf16962b8a48a658a909b97150679c4`; the separate
CPU dispatcher/collector is at `20acc18cefeace618ea8099abb82d8ad72e4cbc2`, in
`<evidence-root>/controller`. The later controller change fixes incident-owner
CLI arguments, admits only validated placements and calibrates native on-chip
progress from the three initial jobs. It changes no worker science or inputs.
The main manifest is pinned by
`ece11e5b6ed9269d0ed50ebfde7e8fe5689d8ed0d3dc9b1f2d4980082d83bdfd`.

Nine original deployments were materialized and validated with no new writes;
preparation consumed 83.82 GPU-seconds. Native planning, dry runs and hash binding
passed on all admitted placements; 187 targeted tests passed. Observed GPU usage
was 20,369 MiB locally (including display usage), 19,393 MiB on nom-cool-1 and
19,620 MiB on loulou; all are within their memory allowance.

The nom-cool-1 Ethernet link is 100 Mb/s. To avoid delaying all workers for its
full input transfer, launch used its first source/assignment group, then added
the other two cdt3 assignments through an explicit prepared-placement amendment.
Validated placement counts are loulou 25, local 6, nom-cool-1 9. Every scientific
job already has a prepared placement. Further copies do not authorize additional
submissions or change job identities, budgets or source selection.

Handoff checks are recorded in `readiness/supervision.json` and
`readiness/active-worker-restart.json`. The read-only investigator's test write
was rejected. The first bounded repair-owner invocation exposed mutually
exclusive CLI flags; that failed attempt is preserved. After correction, the
second test produced a `no_action` report and a clean terminal receipt. Dispatcher
restart preserved ledger state, incident history and budgets; duplicate prepare
and registration were idempotent. A second restart with three active jobs
preserved all worker/supervisor identities and attempt tokens. Systemd monitors
the service heartbeat and restarts it on failure; user lingering is enabled.
The single service, not an interactive agent, owns subsequent polling and dispatch.

The deadline remains **October 13, 18:46:44 Europe/Paris**, measured from initial
preparation admission. Limits remain 12 GPU-hours total and 600 GB. Successful
native validation and collection of all three initial jobs automatically opens
the next gate. The final collector requires all 162 endpoints and produces
`reviews/summary.json`, assignment-level CSV, grouped descriptive statistics and
PNG/PDF recovery-versus-corruption and recovery-versus-writes figures. Scientific
interpretation remains pending completed evidence.
