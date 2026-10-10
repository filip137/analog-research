---
id: "exp-005"
title: "Experiment 1e: develop whole-model HWA, then compare matched scopes"
status: "running"
hypotheses: []
---
# exp-005: whole-transformer HWA

## Question and current assignment

First determine how standard 3% whole-transformer HWA responds to learning rate
and decay on synthetic text. Review this focused evidence before committing to
all remaining recipes and analog-only controls. This sequencing supersedes the
unexecuted fixed-1e-4 plan and the older digital-only LR search.

The user authorized implementation, launch, persistent monitoring and tmux access.
On 2026-10-10 the user lifted the weekend two-GPU admission cap and authorized a
higher compute allowance; the bounded campaign allowance is now 16 GPU-hours.
The six-cell grid below is the recommended default offered to the user, using the
previously discussed 1e-5/1e-4/1e-3 range. Later scientific changes require review.
Both readiness smokes passed. The calibrated handoff gate is open and all six full scientific candidates are released to the persistent queue.

## Frozen focused comparison

| Field | Setting |
|---|---|
| Network | FP32 OPT-125M; last six MLPs analog, 28,311,552 logical weights; attention and all other components digital |
| Trainable scope | Whole transformer, including analog weights, calibration, embeddings, attention, other MLPs and tied head |
| Data | Pinned `opt125m-synthetic-50m`, 195,226 training sequences of 256 tokens; 256 generated validation sequences |
| Budget per candidate | 12,800 updates, batch 32, 104,857,600 token exposures, about 2.098 corpus passes |
| Intervals | 50 intervals of 8,192 sequences; these are **not** corpus epochs |
| Recipe | Standard HWA, per-output-channel abs-max Gaussian noise 3%; no temporary corruption masks |
| LR candidates | 1e-5, 1e-4, 1e-3, applied equally to every trainable parameter group |
| Warm-up | 205 updates (ceil of 1.6% of total), identical for both schedule families |
| Linear schedule | Linear decay from peak at update 205 to zero at update 12,800 |
| Step schedule | Peak through update 10,240, then 0.1 × peak for the remaining updates |
| Held fixed | Adam defaults, no weight decay or gradient clipping; current q bounds [-1,1]; teacher-KL objective; HWA seed 41 and data seed 20261008 |
| Selection | Minimum mean development KL across healthy and 3% random-stuck cases, using selection arrays 211001 and 211002; evaluate at initialization and every 512 updates |

Training has exactly the same batches and noise stream across the six cells.
The rate and decay family are the only intended interventions. Schedule factors
are evaluated before each one-based optimizer update: warm-up is s/W; linear
decay is (T-s)/(T-W); step decay changes only for s > 10,240.

The fit never scores either final test cohort. Development lifecycles use
`data.evaluation: development` and omit `transfer_tests`. Later final evaluation
adds WikiText and generated test cohorts as named transfer tests without changing
the HWA source identity. No fingerprint is weakened to enable reuse.

## Decision and scientific limits

All six candidates are attempted at equal budgets; do not eliminate a candidate
because its early curve is worse. Operationally incomplete attempts do not count
as completed comparisons. A diagnosed nonfinite fit is an unstable candidate,
is retained as negative evidence, and is never automatically retried at a new LR.
Independent candidates may continue while its downstream source use stays blocked.

For stable completed candidates, rank selected development KL. Within 1% relative
of the lowest score, prefer the lower peak LR, then linear decay at equal LR.
Report initial, best and final development scores, healthy and 3%-fault components,
selected update, clean development KL, training loss, duration and memory.
Plot development KL and actual LR against optimizer updates. Flag an edge winner,
a best checkpoint at initialization, late improvement, or inconsistent schedule
ordering; do not silently extend the grid. One training seed gives exploratory
hyperparameter evidence, not an estimate of training variance.

The summary recommends a recipe but writes `awaiting_scientific_review`; it cannot
release the broader campaign. Review the curves and cost before freezing the common
recipe. Neither final test domain participates in this decision.

## Broader comparison, held until development review

After the recipe is frozen, reuse the selected standard-3% whole-model source
verbatim and train the other five sources: standard 9% and corruption-aware 3%
whole-model HWA, and all three analog-only counterparts. Both scopes use the same
frozen LR, schedule, data and update budget. Historical analog-only sources are
contextual evidence and are not matched controls for the new optimization recipe.

Together with pretrained, evaluate seven sources on assignments 271001–271005 and
random stuck-device fractions 0–5%. CMO 2% P&V endpoints and held apparent-state
semantics remain unchanged; no drift or read noise. This is 210 deployment rows.
Final test columns are `deployed_transfer_wikitext_teacher_kl` (1,024 sequences)
and `deployed_transfer_generated_teacher_kl` (384 sequences). The plain deployed KL
column describes development and must not be mistaken for WikiText.

A wins over B only with lower KL on at least four of five matched assignments and
mean per-assignment relative reduction at least 10%, separately by fault fraction
and domain. Report clean-source versus healthy-deployment versus corrupt-deployment
contrasts and absolute scores on both domains; a specialization ratio alone cannot
establish an absolute loss out of domain. Reproduce the unchanged pretrained
baseline; do not demand that newly trained analog controls reproduce old checkpoints.

## Reference choices and deviations

Consult [the reference papers](../../../../../hwa_papers/README.md). The 1.6% warm-up
ratio follows Büchel et al. 2025 Appendix I; our explicit linear and late-step
comparison is a declared schedule experiment. The paper describes polynomial
scheduling with AdamW; this campaign retains Adam without weight decay or gradient
clipping. Whole-model training and synthetic distillation follow its motivation,
while OPT-125M, our smaller batch/token budget, analog MLP subset, checkpoint
selection and CMO device regime differ.

Per-channel noise follows the Analog Foundation Models convention. Gradients pass
through noisy weights as in the existing implementation, and q is clamped to
[-1,1] after updates. Adaptive alpha-times-standard-deviation clipping, input/output
quantization and learned input ranges are not part of this experiment. Generated
training text uses temperature 0.9 and top-k 50; record this precisely rather than
claiming exact reproduction of either paper's complete data-generation procedure.

## Inputs, lifecycles and outputs

Teacher: `facebook/opt-125m`, revision
`27dcfa74d334bc871f3234de431e71c6eeba5dd6`, weights SHA-256
`2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9`.

Training corpus SHA-256:
`ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984`.
Future generated-test corpus `wikitext2-train-synthtest` SHA-256:
`8e73b6b43aafb92b8b52e11cdd060d573331fae410350cf2427f3eeba640e224`.

The strict lifecycles are generated by
[exp1e_development.py](../../../launch/exp1e_development.py):
`opt125m-cmo-l6-exp1e-dev-{linear,step}-lr1em{5,4,3}.json`.
Each has one `standard_hwa_model` arm. Separate labelled 512-update smokes at
1e-4 use nine warm-up updates and scaled decay endpoints. Old unlaunched
`exp1e-lrdev` and `exp1e-smoke` files remain historical preparation, not active jobs.

Use `python -m workflow plan`, then a dependency-preserving stage slice containing
only prepare, devices and HWA, executed by `python -m ebl campaign run`.
Do not launch deployment stages during the focused comparison. The full lifecycle describes prepare=1, devices=1, HWA=1 and deploy=5 (30 case rows), with 28,311,552 analog weights; the execution slice contains only the first three stages and creates no deployed trajectories.

Remote base: `~/analog-research/opt-cmo-exp1/`; stage a clean, pinned source checkout.
Native outputs: `results/lifecycles/<lifecycle-id>/runs/` under that checkout.
Local compact evidence and plots: `results/lifecycles/opt125m-cmo-l6-exp1e-development/`.
Final interpretation belongs to [results/exp-005.md](../results/exp-005.md).
Copy back small results/configs/metrics/inspection receipts, not large raw bundles.

## Budget and admission

16 GPU-hours total, 30 GB of incremental artifacts/staged copies, and a 12-hour
execution window beginning with the first GPU smoke. All failed attempts, readiness
runs and retries consume this same budget. Full candidate allowance is initially
90 minutes, subject to the measured readiness gate; do not shorten scientific
training or enlarge limits silently to fit a resource estimate.

All four RTX 5090 hosts (fifi, loulou, riri, trex) are eligible when idle, with the
user's weekend fleet-cap override. Prefer the already compatible PyTorch 2.11
hosts, one worker per GPU. Do not disturb or share with existing jobs. The user
also explicitly authorized nom-cool-1 and the local nom-cool-2 RTX 3090s on
2026-10-10. The local full-batch smoke fits, but full 3090 admission is held by
the cross-hardware randomness failure described below. Jean Zay is held for the later
deployment phase and requires its own validated adapter and production canary.

Measure full-batch peak memory, throughput, and actual storage during the 512-update
smokes before admitting six full trainings. Prefer 1e-4 linear, 1e-4 step, 1e-5
linear, 1e-5 step, 1e-3 linear, 1e-3 step as queue priority; all run in full.

## Persistent monitoring and tmux handoff

One existing `run-watch-queue.service` on nom-cool-2 owns campaign dispatch and
monitoring through a repository-owned reusable extension. Its existing SQLite ledger
records jobs, attempts, manifests, dependencies, leases, incidents and budgets.
Register this focused phase only; broader jobs are not submitted before review.

Poll every 300 seconds. Observe actual workers, nested native-stage progress,
finite metrics, artifacts and storage, separate queued/running states, and integrate
GPU allocation time. Calibrate phase-specific progress thresholds from smoke
measurements; a heartbeat alone is not semantic progress.

One visible tmux session per attempt carries live logs and remains available after
completion. Record its session name plus actual worker PID/start/boot identity,
output root, log path and terminal receipt. Tmux existence does not imply success.
Budget and lease enforcement must reach the actual worker descendants. Remote
lease expiry uses the remote monotonic clock, avoiding known host wall-clock offsets.

Systemd restarts a failed controller and supervises an 11-minute watchdog. Workers
have renewable 15-minute supervision leases. Only bounded read-only Codex incident
investigators are permitted for this campaign (five minutes, one at a time, two
calls per incident, six per rolling day). The service captures their exact evidence,
diagnosis and recommended intervention. Investigators cannot launch, edit configs,
change LR/selection, mutate state or use network/submission authority. The read-only sandbox does not hide all readable host files; investigators receive only the bounded incident snapshot.

One automatic operational replay per stage is allowed only after confirmed terminal
ownership: preemption/node failure, transport failure before execution, failed
collection alone, or supervision-loss termination. Nonfinite science, OOM,
provenance failures and unknown failures require diagnosis, not a changed recipe.
There is no intermediate training resume; a training retry starts from the declared
initial state and preserves the failed attempt.

Before expensive focused runs, require: observer coverage of all readiness jobs;
real read-only Codex test-incident invocation/report; restart and lost-ack recovery
without duplicate launches; preserved budgets/incidents; tamper/invalid-metric and
failed-gate checks; tmux detach/terminal receipts; watchdog and lease expiry tests.
The future Slurm phase stays disabled until its adapter and live canary pass.

### Current execution and monitoring handoff (2026-10-10)

Owner: persistent `run-watch-queue.service` on nom-cool-2, with user lingering
and `WatchdogSec=660`. It owns every case; interactive agents are implementation
helpers, not execution owners. Training source is commit `237de7d49b05fd485a817f73d8b46d3cc30e4592`;
controller-only reservation repair is `bdbee05`.

Durable queue campaign: `opt-cmo-exp1e-development-20261010`.
Manifest SHA-256: `e8c2b404b808786a8daad7edf474a74568eb76014161334ea2a930764b8619b7`.
The eight job IDs equal their lifecycle IDs. Native plans, slices and input/config
hashes are in the four host receipts under the local
`results/lifecycles/opt125m-cmo-l6-exp1e-development/hosts/` directory.
The exact job specifications and campaign contract are under its `queue/` directory.

Remote checkout on each host:
`/home/filip/analog-research/opt-cmo-exp1/hwa-exp1e-development-20261010a`.
Worker receipts/logs are under
`/home/filip/analog-research/opt-cmo-exp1/worker-receipts/237de7d49b05fd485a817f73d8b46d3cc30e4592/<job-id>/<attempt-token>/`.
Tmux sessions are named `<job-id>-<attempt-token>` and retain the terminal pane.
Use `ssh -F ~/.ssh/config -t HOST tmux attach -t SESSION` to attach.

Current scientific execution: `opt125m-cmo-l6-exp1e-dev-linear-lr1em4` is running
on loulou, attempt `01eede827d1a4cd7af528580992c7bc6`; the other five candidates
are queued for idle prepared RTX 5090s. Exact tmux session:
`opt125m-cmo-l6-exp1e-dev-linear-lr1em4-01eede827d1a4cd7af528580992c7bc6`.
The full-trial handoff gate is recorded in `readiness/handoff-gate.json`.
At 09:28:44 UTC the service observed the live worker (PID 527684, recorded
boot/start identity), finite loss 0.08011 at update 206, and completed warm-up.
The next regular campaign check is due around 09:33:44 UTC. The complete live
observation, ownership and remaining-budget receipt is `readiness/live-handoff.json`.
At release, smokes consumed 282.57 GPU-seconds (0.0785 GPU-hours), leaving
15.9215 GPU-hours including the first candidate's 1.5-hour reservation.

The initial smokes both executed on loulou after fifi became occupied by another
user. Step attempt: `3568591cb3c943b1b8633530c3efd11c`; linear attempt:
`d5fbd9d96016443e8781cc39b45b4e5c`. The linear placement amendment was recorded
before any attempt and preserved its exact configuration and budget.

## Additional RTX 3090 readiness and scientific hold

Both added GPUs are healthy outside the interactive sandbox. Stage the same
training source and pinned inputs in the dedicated checkout above, with an
isolated runtime at `~/analog-research/opt-cmo-exp1/runtime-exp1e-cu128`.
It matches the 5090 runtime: PyTorch 2.11.0+cu128, Transformers 5.8.0,
NumPy 2.4.4 and tokenizers 0.22.2. Shared existing environments remain intact.
Runtime copies are included in the original 30 GB storage allowance.

Controller revision `3890610` adds local collection and append-only operational
amendments for readiness cases, storage roots and target progress calibration.
The original manifest hash, handoff gate, 16 GPU-hour cap, start time and deadline
are unchanged. Old and new execution records remain in the same locked ledger.
Amendment receipts and diagnostics are under `readiness/3090/` in the local
evidence root. The local readiness job is
`opt125m-cmo-l6-exp1e-dev-linear-lr1em4-smoke-readiness-local`, attempt
`7c1fa9e007a443e0a03de990c5113996`. Its visible tmux session is
`opt125m-cmo-l6-exp1e-dev-linear-lr1em4-smoke-7c1fa9e007a443e0a03de990c5113996`.

Nom-cool-1 uses the separate readiness queue ID
`opt125m-cmo-l6-exp1e-dev-linear-lr1em4-smoke-readiness-nom-cool-1`.
Its first attempt, `c106f2a82c4140c089ad4957a2297f7c`, failed at the first
`tmux` lookup before creating a training worker. Preserve that failed attempt.
An isolated tmux executable was staged and checked with a disposable non-GPU
session; its hashes and the exact required placement `PATH` are recorded in
`readiness/3090/nom-cool-1-tmux-fix.json`. Future nom-cool-1 placements must retain
that PATH. One classified pre-start operational retry uses the original
readiness allowance and campaign budget; it grants no full scientific admission.
The retry is running with attempt `262ff8d09a454354943964610f842919`, in tmux
session `opt125m-cmo-l6-exp1e-dev-linear-lr1em4-smoke-262ff8d09a454354943964610f842919`.
Use the isolated runtime's `bin/tmux` to attach on nom-cool-1. The fresh service
observation and restart-recovery receipt is `readiness/3090/live-handoff.json`.

The local smoke completed all 512 updates with finite metrics, but initial
development KL differs from loulou by about 1%. Paired exact-shape probes confirm
different hashes for the minibatch permutation, CMO programming endpoints and
HWA noise despite equal seeds and Torch/CUDA versions. Permanent CPU-generated
fault maps remain matched. The current CUDA sampling implementation therefore
does not satisfy the frozen matched-randomness contract across GPU models.

Do not add full 3090 placements to the current grid. Incident 2,
`cross_hardware_randomness_mismatch`, has a durable bounded read-only investigator
report. The existing 5090 trial continues and the five pending scientific cases
retain their 5090 placements. The user has been asked to choose between retaining
that grid and explicitly restarting a matched grid with versioned portable
randomness on all authorized GPUs. A portable profile must pass complete tensor
hash checks across targets and preserve legacy bundle replay; seeds or matching
prefixes alone are insufficient. No scientific configuration has been changed.
The immutable `rtx3090_matched_randomness` gate records `passed: false`; the
original 5090 handoff gate remains unchanged.

Measured 3090 throughput projects to roughly 2–2.5 hours for 12,800 updates.
If full 3090 training is later admitted, record a measured wall-time amendment;
the current 90-minute worker limit is insufficient. Readiness success is not
permission to change the update budget, batch size or checkpoint-selection rule.

The campaign window started at 09:19:34 UTC (11:19:34 Paris) and closes at
21:19:34 UTC (23:19:34 Paris). Full candidates have 90-minute allowances;
smoke measurements project about 50–60 minutes each. Preparation and CMO devices
complete in seconds; HWA metrics advance roughly every 100 updates, with selection
every 512. Training startup/stall thresholds are 600 seconds, with a 3,600-second expected
case duration. The handoff receipt freezes this measured calibration.

Readiness evidence and active-restart receipts are under the local `readiness/`
directory. Full-hash validation and compact collection are native queue phases;
failed collection retries do not retrain. Remaining budgets, leases, events and
incident reports persist in `~/.local/state/run-watch/queue/queue.sqlite3`.
`systemctl --user status run-watch-queue.service` verifies the controller;
`python -m campaigns.execution_service status` exposes durable observations.
The service heartbeat is `~/.local/state/run-watch/queue/service-status.json`.

The automatic collector writes `collected/index.json` and
`collected/review/development.{json,md}` plus PNG/PDF curves. Its report always
leaves the broader comparison at `awaiting_scientific_review`. No broader jobs
are registered. Slurm execution is disabled for this focused phase.
