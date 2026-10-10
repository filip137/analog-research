---
id: "exp-005"
title: "Experiment 1e: develop whole-model HWA, then compare matched scopes"
status: "planned"
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
Implementation and readiness checks are in progress; no scientific candidates have launched yet.

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
hosts, one worker per GPU. Do not disturb or share with existing jobs. RTX 3090
placement is not assumed to fit the whole-model run. Jean Zay is held for the later
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

Current state: scheduler/service/tmux implementation and readiness validation in
progress. No production job IDs or tmux handles have been issued yet. Replace this
paragraph with the verified live handoff before transferring execution ownership.
