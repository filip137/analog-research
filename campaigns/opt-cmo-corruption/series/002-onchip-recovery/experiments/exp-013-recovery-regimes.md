---
id: "exp-013"
title: "Recovery across three HWA recipes, six fault rates and three assignments"
status: "ready"
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
The frozen worker checkout on each host is
`/home/filip/analog-research/opt-cmo-exp1/hwa-exp7-recovery-20261012`.
Preparation checks input binding, native plans and dry runs; execution checks
finite metrics, exact held-state replay, fault persistence, full coverage and cap
accounting. Final plots show recovery versus corruption and writes by source.

## Current execution handoff

Implementation and frozen input staging are in progress. No new recovery job has
launched. Source revision, service incident/restart checks, readiness receipts,
timing measurements and active handles will be recorded here before handoff.
