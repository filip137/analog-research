---
id: "exp-005"
title: "Experiment 1e: matched HWA scopes with cosine decay"
status: "running"
hypotheses: []
---
# exp-005: whole-transformer HWA

## Question and current assignment

On 2026-10-11, after reviewing the scope gap, the user approved the focused test:
hold analog training fixed and vary only the additional digital LR toward zero.
The completed cosine comparison below is retained as its reference. Execute four
fresh standard-3% HWA sources on **fifi and loulou only**, with the shared persistent
dispatcher, one worker per RTX 5090. The existing analog and equal-rate controls
are references, not new replicates.

### Frozen digital-LR comparison

| Setting | Declared value |
|---|---|
| Additional digital peak LR | 0, 1e-5, 1e-4, 1e-3; final rates 0, 2e-6, 2e-5, 2e-4 |
| Analog/calibration LR | Fixed peak 5e-3, cosine decay to 1e-3 |
| Schedule and budget | 205 warm-up updates; 25,600 updates, batch 32, about 4.196 synthetic-corpus passes |
| Training scope | Model scope for all four, including the zero-digital-update endpoint |
| Teacher/network | Same pinned FP32 OPT-125M, last six analog MLPs, 28,311,552 logical analog weights |
| Data and noise | Same synthetic cohort, seeds and batch stream; standard per-output-channel 3% Gaussian HWA; no corruption masks |
| Selection | Unchanged mean healthy/3%-random-fault development KL on selection arrays 211001/211002, 256 development sequences, initialization plus every 512 updates |
| Held fixed | Adam defaults, no weight decay or gradient clipping, q clamp [-1,1], calibration group and CMO endpoint model |
| Evaluation boundary | Prepare, devices and HWA only; no deployment to test arrays and no held-out test corpus loaded |

Rank completed candidates by the existing development mean; within 1% prefer
the smaller digital peak LR, including zero. Report both components, clean KL,
selected update and late-convergence flag. Compare the zero case with the prior
analog-only mean 0.085279 and all candidates with the prior equal-rate mean
0.305032. A nonzero rate must improve on the zero endpoint beyond the tie range
to motivate retaining digital updates. No new healthy-performance threshold is
introduced: report the tradeoff explicitly. One training seed and an adaptively
chosen development grid remain exploratory; no automatic broader rollout follows.

The LR split tests a specific optimization hypothesis, not a claim that the
previous implementation is faulty. All runs start from the original teacher,
not the trained analog source. The zero case retains the whole-model forward,
backward and optimizer path, with a runtime assertion that its digital weights
stay unchanged. A tiny CPU equivalence test and two 512-update full-batch smokes
(analog and zero digital LR) check this boundary before expensive dispatch.
Cross-host exact GPU matching remains diagnostic under the user's earlier waiver;
finite execution, exact schedule coverage and zero digital-weight changes are
mandatory gates. Smokes retain 205 warm-up steps but have their own 512-update
cosine horizon; they do not select a scientific rate.

The two reference papers and their local PDFs were consulted again. Per-channel
noise and synthetic distillation retain the earlier Büchel-inspired choices;
backpropagation through noisy weights retains the Rasch-style choice. The LR
split, zero endpoint, cosine floor, Adam defaults, smaller token budget and best
development checkpoint remain campaign-specific deviations. Appendix I of Büchel
uses different AdamW settings and polynomial scheduling; neither paper establishes
the digital/analog rate ratio tested here. No paper-matched recipe is claimed.

Strict definitions: `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1e-digital-*.json`.
Preparation: `campaigns/opt-cmo-corruption/launch/exp1e_digital_lr.py`.
Local evidence: `results/lifecycles/opt125m-cmo-l6-exp1e-digital-lr/`.
Remote source/results: `/home/filip/analog-research/opt-cmo-exp1/hwa-exp1e-digital-lr-20261011/`.
New phase ID: `opt-cmo-exp1e-digital-lr-20261011`. Bound the incremental phase to
**10 GPU-hours, 16 GB storage and 8 hours from registration**, including readiness,
failed attempts and any authorized operational retry. Full-worker hard cap 3 h;
smoke cap 30 min. The measured whole-model fit takes about 6,057 s; four fits are
about 6.8 GPU-hours plus readiness/validation, or **3.5–4 h on the pair**.
The service polls campaign semantics every five minutes, distinguishes queued
from running, and applies phase thresholds: prepare/devices 120 s, HWA 6,500 s
(smoke 300 s), validation/collection 600 s; stale 900 s, startup 600 s.
Failed mandatory gates or insufficient budget stop downstream dispatch. No
automatic scientific changes or compute retries. The existing tested incident
investigator, restart-safe ownership, locking and durable ledger remain in use.

### Completed cosine comparison protocol

The focused learning-rate and duration experiments are complete. On 2026-10-10,
the user selected a pragmatic common recipe: “5e-3 with cos decay to 1e-3” for
“the whole experiment”, then explicitly requested “run it on two 5090s”. This
supersedes the pending broader-release decision below. Execute on **fifi and
loulou only**, one worker per RTX 5090, through the existing persistent service.

The new recipe uses 25,600 updates (100 intervals of 8,192 sequences, batch 32;
about 4.196 passes over the 195,226-sequence synthetic training cohort), retaining
205 warm-up updates. All trained parameter groups warm up to 0.005 and then use
`0.001 + 0.004 * (1 + cos(pi * (s - 205) / (25600 - 205))) / 2`.
The final optimizer update has LR 0.001. This smooth schedule and floor were
**user-selected, not evaluated or established optimal by the previous sweep**.
The floor is twice the 0.0005 tail of the completed 5e-3 step trial. Warm-up is
0.80% of this longer budget; the original 1.6% ratio described the short grid.

Train all six sources afresh from the pinned teacher: standard 3%, standard 9%,
and temporary random-corruption-aware 3%, each at analog and whole-model scope.
No old step/linear source is reused. Include the unchanged pretrained logical
source. Preserve the selection rule, data, seeds, Adam settings and physical
regime below. Score seven sources on five independent assignments and all six
fault fractions: **210 deployment rows**, with both held-out domains per row.
Transfer cohorts may be prepared with the source cache, but neither training nor
checkpoint selection reads them. Only deployment evaluates the test domains.

Strict main lifecycle: [opt125m-cmo-l6-exp1e.json](../../../lifecycles/opt125m-cmo-l6-exp1e.json).
Two labelled numerical canaries use 512 updates (205 warm-up), corruption-aware
3% HWA at each scope, one assignment and all six fault fractions, with the full
held-out evaluation cohorts. These canaries test the changed scheduler, source
validation, deployment, collection and memory/time estimates; their test metrics
must not be used to retune the scientific recipe. Main jobs wait for the frozen
handoff gate. Deployment itself requires a complete finite source fit and exact
update/schedule/selection coverage; a numerically invalid source never advances.

One queue job owns a native dependency-preserving source slice: prepare, devices,
HWA and its five deployment stages. Source artifacts are referenced by exact
native dependencies and hashes. Native resume reuses validated completed stages;
it does not infer training-state resume from a selected checkpoint. The reusable
collector inspects hashes, validates exact coverage and lineage, invokes native
`workflow collect`, and copies compact evidence. Failed collection does not rerun
successful GPU work. The service observes each actual stage with separate
training, deployment, validation and collection expectations; per-defect metrics
make deployment progress visible. No additional dispatcher or polling agent.

New phase ledger ID: `opt-cmo-exp1e-cosine-20261010`, in the existing shared SQLite
ledger. **18 incremental GPU-hours, 40 GB incremental storage, 12 hours from the
first canary**, including readiness, failed attempts and retries. The completed
focused phase retains its original records and 15.329 GPU-hours charged; the new
phase does not reset or erase them. Estimated work is 8.6 GPU-hours training plus
roughly 2 GPU-hours deployment, subject to the live canary, about 5–6 hours across
the pair. Worker hard caps are 3 h whole-model, 2.25 h analog, 45 min pretrained,
and 30 min per readiness job. Reserve full allowances before dispatch.

Both test corpora and the teacher/training corpus are hash-verified on both hosts.
Remote source and native outputs are under
`/home/filip/analog-research/opt-cmo-exp1/hwa-exp1e-cosine-20261010/`;
local compact evidence and queue receipts are under
`results/lifecycles/opt125m-cmo-l6-exp1e-cosine/`.
The former pretrained results were generated on V100. Cross-GPU exact stochastic
matching was explicitly waived by the user. Historical baseline differences are
reported as diagnostics, not an automatic cancellation gate; the freshly deployed
pretrained arm supplies the matched baseline for this two-5090 comparison.

Implementation: strict `cosine_warmup_decay` with required `min_factor: 0.2`;
existing optimizer code applies the factor to both analog/calibration and digital
parameter groups. Legacy lifecycle/source identities remain unchanged. The
scheduler tests include the actual final optimizer update and both parameter
groups. Validation passed: 276 workflow/collector tests (one skipped), 109 service
tests, 50 development/decision tests; the updated collector suite has 19 tests.
Before deployment, complete native dry-runs,
then the two full-batch canaries. Handoff receipts below will record actual launch
handles and measured expectations once those gates pass.

## Completed focused development (historical protocol)

First determine how standard 3% whole-transformer HWA responds to learning rate
and decay on synthetic text. Review this focused evidence before committing to
all remaining recipes and analog-only controls. This sequencing supersedes the
unexecuted fixed-1e-4 plan and the older digital-only LR search.

The user authorized implementation, launch, persistent monitoring and tmux access.
On 2026-10-10 the user lifted the weekend two-GPU admission cap and authorized a
higher compute allowance; the original bounded allowance was 16 GPU-hours.
The authorized duration comparison below raises the cumulative cap to 20 GPU-hours,
including all prior work and retries.
The initial six-cell grid used the previously discussed 1e-5/1e-4/1e-3 range.
After four completed candidates favored its upper LR boundary, the user explicitly
requested adding 5e-3 on RTX 5090s: "please schedule also 5e-3 on the 5090s -- so
that we dont finish on the upper bound of good performance". This authorizes two
additional candidates, one per existing schedule. The expansion is an adaptive
development decision made after partial results, not part of the initial grid.
After all eight completed, the user authorized a second extension to 1e-2 for
both schedules ("its just strange that lr has to be so high? but yes leta do
1e-2 as well"). The LR development grid therefore has ten candidates. The user
then accepted a separate two-case duration comparison at 1e-3 and 5e-3 after
reviewing the late improvements ("ok yes..."). Its frozen protocol is below;
unequal training budgets do not enter the ten-case LR ranking.
Both readiness smokes passed and the calibrated handoff gate remains open.

As of 2026-10-10 18:52 Paris, all ten short candidates and both duration trials
are complete, validated and collected. The focused execution is finished and
all campaign GPUs are released. The short-grid rule recommends linear 1e-2;
the separate four-pass comparison recommends step 5e-3. Longer training improves
both healthy and faulty performance, but both duration runs select their final
checkpoint. The result note contains the review. Experiment status is partial
because the broader scope/recipe comparison was then unsubmitted. The user
subsequently selected the cosine protocol above; the following focused rules
remain the historical record of how its evidence was obtained.

## Frozen focused comparison

| Field | Setting |
|---|---|
| Network | FP32 OPT-125M; last six MLPs analog, 28,311,552 logical weights; attention and all other components digital |
| Trainable scope | Whole transformer, including analog weights, calibration, embeddings, attention, other MLPs and tied head |
| Data | Pinned `opt125m-synthetic-50m`, 195,226 training sequences of 256 tokens; 256 generated validation sequences |
| Budget per candidate | 12,800 updates, batch 32, 104,857,600 token exposures, about 2.098 corpus passes |
| Intervals | 50 intervals of 8,192 sequences; these are **not** corpus epochs |
| Recipe | Standard HWA, per-output-channel abs-max Gaussian noise 3%; no temporary corruption masks |
| LR candidates | 1e-5, 1e-4, 1e-3, plus the user-authorized 5e-3 and 1e-2 extensions; applied equally to every trainable parameter group |
| Warm-up | 205 updates (ceil of 1.6% of total), identical for both schedule families |
| Linear schedule | Linear decay from peak at update 205 to zero at update 12,800 |
| Step schedule | Peak through update 10,240, then 0.1 × peak for the remaining updates |
| Held fixed | Adam defaults, no weight decay or gradient clipping; current q bounds [-1,1]; teacher-KL objective; HWA seed 41 and data seed 20261008 |
| Selection | Minimum mean development KL across healthy and 3% random-stuck cases, using selection arrays 211001 and 211002; evaluate at initialization and every 512 updates |

All cells retain the same corpus, cohorts, seeds and noise distributions. On
2026-10-10 the user explicitly waived exact cross-GPU batch/noise matching
("i dont care about the matching check") and authorized continuing this grid
on both GPU models. Realized batch orders and noise differ between 3090s and
5090s; record host/GPU with each result. No portable-RNG restart is requested.
The rate and decay family are the intended interventions. Schedule factors
are evaluated before each one-based optimizer update: warm-up is s/W; linear
decay is (T-s)/(T-W); step decay changes only for s > 10,240.

The fit never scores either final test cohort. Development lifecycles use
`data.evaluation: development` and omit `transfer_tests`. Later final evaluation
adds WikiText and generated test cohorts as named transfer tests without changing
the HWA source identity. No fingerprint is weakened to enable reuse.

## Decision and scientific limits

All ten candidates are attempted at equal update budgets; do not eliminate a candidate
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

The focused summary retains its original `awaiting_scientific_review` marker.
The subsequent user decision above authorizes the new cosine comparison; no
summary script makes this scientific decision. Neither final test domain
participates in recipe selection.

## Broader comparison and frozen decision rule

The user-selected cosine recipe requires six fresh sources: standard 3%, standard 9% and corruption-aware 3%
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
establish an absolute loss out of domain. Deploy the unchanged pretrained logical
source as the fresh matched baseline. Historical V100 reproduction is a diagnostic
under the explicit GPU-matching waiver; newly trained controls do not reproduce
old checkpoints. The `cosine-5090-v1` analysis contract records that distinction.

## Reference choices and deviations

The broader cosine-to-0.2-floor schedule is a declared user-selected deviation
from the papers and from both schedules tested during focused development.
Retaining 205 warm-up updates on the longer budget also changes the warm-up ratio.

Consult [the reference papers](../../../../../hwa_papers/README.md). The 1.6% warm-up
ratio follows Büchel et al. 2025 Appendix I; our explicit linear and late-step
comparison is a declared schedule experiment. The paper describes polynomial
scheduling with AdamW; this campaign retains Adam without weight decay or gradient
clipping. Whole-model training and synthetic distillation follow its motivation,
while OPT-125M, our smaller batch/token budget, analog MLP subset, checkpoint
selection and CMO device regime differ.

The LR scale also differs substantially. Appendix I of
[Büchel et al. v3](https://arxiv.org/html/2505.09663v3#A9) gives base rates
1e-6 (Phi-3) and 5e-7 (Llama), multiplied by 96 GPUs for the main experiments:
effective peaks 9.6e-5 and 4.8e-5. Our 5e-3 and 1e-2 tests are exploratory
extensions well above those rates, not paper-matched defaults. Its AdamW
settings also use beta2=0.98, epsilon=1e-6, weight decay 0.01 and gradient norm
clipping at 1.0; this campaign retains the previously declared Adam defaults.
The observed faulted/healthy tradeoff does not establish a general need for
larger learning rates.

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
`opt125m-cmo-l6-exp1e-dev-{linear,step}-lr1em{5,4,3}.json`, plus the authorized
`opt125m-cmo-l6-exp1e-dev-{linear,step}-lr5em3.json` and
`opt125m-cmo-l6-exp1e-dev-{linear,step}-lr1em2.json` extensions.
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

## Focused-development budget and admission (historical)

16 GPU-hours total, 34 GB of incremental artifacts/staged copies (30 GB initially,
32 GB after the 5e-3 extension, then 34 GB after the 1e-2 extension), and a 12-hour
execution window beginning with the first GPU smoke. All failed attempts, readiness
runs and retries consume this same budget. Full candidate allowance is initially
90 minutes, subject to the measured readiness gate; do not shorten scientific
training or enlarge limits silently to fit a resource estimate.

All four RTX 5090 hosts (fifi, loulou, riri, trex) are eligible when idle, with the
user's weekend fleet-cap override. Prefer the already compatible PyTorch 2.11
hosts, one worker per GPU. Do not disturb or share with existing jobs. The user
also explicitly authorized nom-cool-1 and the local nom-cool-2 RTX 3090s on
2026-10-10. Both full-batch smokes passed and full 3090 admission is enabled
under the user's explicit matching waiver below. Jean Zay is held for the later
deployment phase and requires its own validated adapter and production canary.

Measure full-batch peak memory, throughput, and actual storage during the 512-update
smokes before admitting six full trainings. Prefer 1e-4 linear, 1e-4 step, 1e-5
linear, 1e-5 step, 1e-3 linear, 1e-3 step as queue priority; all run in full.

## Persistent monitoring and tmux handoff

One existing `run-watch-queue.service` on nom-cool-2 owns campaign dispatch and
monitoring through a repository-owned reusable extension. Its existing SQLite ledger
records jobs, attempts, manifests, dependencies, leases, incidents and budgets.
The focused phase and the newly authorized cosine phase have separate frozen
contracts in this same ledger; the completed focused history remains intact.

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

## Additional RTX 3090 readiness and admission

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
The retry completed and passed collection with attempt `262ff8d09a454354943964610f842919`, in tmux
session `opt125m-cmo-l6-exp1e-dev-linear-lr1em4-smoke-262ff8d09a454354943964610f842919`.
Use the isolated runtime's `bin/tmux` to attach on nom-cool-1. The fresh service
observation and restart-recovery receipt is `readiness/3090/live-handoff.json`.

The local smoke completed all 512 updates with finite metrics, but initial
development KL differs from loulou by about 1%. Paired exact-shape probes confirm
different hashes for the minibatch permutation, CMO programming endpoints and
HWA noise despite equal seeds and Torch/CUDA versions. Permanent CPU-generated
fault maps remain matched. The current CUDA sampling implementation therefore
does not satisfy the frozen matched-randomness contract across GPU models.

The user explicitly instructed continuing despite this matching check. Both
3090s are now placements for all five previously unlaunched scientific cases;
the running 5090 case continues unchanged. Incident 2 and its failed diagnostic
gate remain preserved, with an append-only user waiver bound to that exact
gate receipt. Controller revision `4b2c66e` honors this waiver; all numerical,
artifact, budget and other validation requirements remain enforced. The previous
gate also blocked queued 5090 jobs because it was applied campaign-wide; the
waiver removes that unintended block. Do not restart or retune this grid to
obtain identical random streams. The authorization, old/new job specs, measured
runtime amendments and calibration are in
`readiness/3090/full-admission-user-waiver.json`.

Full trials admitted after the user waiver:

| Host | Candidate | Attempt |
|---|---|---|
| nom-cool-1, RTX 3090 | step, 1e-4 | `f7e698abff434db7b1f2f7834bf8d895` |
| local nom-cool-2, RTX 3090 | linear, 1e-5 | `080f40c75ed84fdd9869f3a7a5c20bc5` |
| fifi, RTX 5090 | step, 1e-5 | `f138205dded54b32afcf0d065dcd6df8` |

The original loulou linear-1e-4 attempt continues. Sessions are named
`<lifecycle-id>-<attempt>`; `tmux ls` lists them (on nom-cool-1 use
`~/analog-research/opt-cmo-exp1/runtime-exp1e-cu128/bin/tmux`). The remaining
two 1e-3 candidates stay queued under the same resource and budget accounting.
Fresh worker observations and preserved waiver after restart are recorded in
`readiness/3090/full-trials-live-handoff.json`.

Measured 3090 throughput projects to roughly 2–2.5 hours for 12,800 updates.
The new 3090 worker wrappers permit 10,800 seconds (three hours), with 9,000
seconds expected runtime. Pending job allowances are three hours within the
same 16 GPU-hour campaign cap. The measured memory reservation is 21,500 MiB
plus 1,024 MiB headroom. Existing 5090 worker wrappers retain their 90-minute
timeout. Update counts, batch size and checkpoint selection are unchanged.

The campaign window started at 09:19:34 UTC (11:19:34 Paris) and closes at
21:19:34 UTC (23:19:34 Paris). The 5090 smoke measurements project about
50–60 minutes per candidate; 3090 calibration is recorded above. Preparation and CMO devices
complete in seconds; HWA metrics advance roughly every 100 updates, with selection
every 512. Training startup/stall thresholds are 600 seconds, with a 3,600-second expected
case duration on 5090s. The handoff receipt freezes this measured calibration;
the additive target calibration uses 9,000 seconds for either 3090.

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

### Live handoff, 2026-10-10 13:06 Paris

Linear 1e-4 on loulou and step 1e-5 on fifi are complete, collected and
validated; partial scientific interpretation is in the result note. Both 3090
trials continue with finite metrics. The final two candidates were admitted:

| Host | Candidate | Attempt |
|---|---|---|
| fifi, RTX 5090 | linear, 1e-3 | `09a9d8e4acb14c9089e3baef3ecd1137` |
| loulou, RTX 5090 | step, 1e-3 | `1f71c30ed2ac45cca765a43b90df8e6f` |

Controller `782cd5c` supports audited storage estimate amendments. A completed
full case occupies 918,639,322 bytes, so the six training cases now reserve
1,000,000,000 bytes each instead of the preliminary 1,500,000,000. The original
manifest, 30 GB cap, GPU/time budgets, all scientific inputs and active attempts
are preserved. This removes an overly conservative admission block after the
two 3090 runtimes were staged. The measured evidence, old/new reservations and
preservation checks are in `readiness/full-storage-calibration.json`; 64 service
tests passed, including restart recovery and rejection of stale estimates.

At measured throughput, the new 5090 trials should finish near 14:00 Paris and
the 3090 trials near 14:25–14:40. These are estimates, subject to validation and
host performance. The persistent watchdog continues to own all four attempts.
After a controller restart, it observed all four live workers and supervisors,
finite training metrics, and unchanged attempt tokens. The durable receipt is
`readiness/full-storage-live-handoff.json`; no duplicate launches occurred.

### Current status, 2026-10-10 14:03 Paris

Both 1e-3 candidates completed native validation and collection; fifi and
loulou are now free of this campaign's workers. Four of six candidates are
complete. Only step 1e-4 on nom-cool-1 and linear 1e-5 locally remain active,
under their existing attempt tokens and tmux sessions. The 13:59 semantic
observation shows 9,573/12,800 and 10,341/12,800 updates respectively, with finite
metrics and live workers. The 14:02 watchdog heartbeat is fresh, without host
errors or new incidents. Measured throughput projects local completion near
14:25 and nom-cool-1 near 14:35–14:40 Paris, including a small validation margin.
No full candidates remain queued; the broader comparison remains unsubmitted.

### Authorized 5e-3 extension, 2026-10-10

Add `opt125m-cmo-l6-exp1e-dev-linear-lr5em3` and
`opt125m-cmo-l6-exp1e-dev-step-lr5em3` on fifi/loulou RTX 5090s. Peak LR is the
only training change relative to their 1e-3 schedule counterparts. The existing
12,800 updates, 205 warm-up updates, synthetic corpus/cohorts, standard 3% HWA,
whole-transformer scope, selection rules and held-out test domains are preserved.
The summary now requires eight completed candidates before recommending a recipe.
Keep 5e-3 as the new upper boundary and retain the edge-winner review rule.

Measured expected execution is 50–60 minutes per candidate, with a 5,400-second
hard allowance each and the same 900-second renewable tmux lease. Reserve
21,500 MiB plus 1,024 MiB headroom. The two jobs add roughly 1.7 GPU-hours and
1.84 GB of retained artifacts. The total allowance remains 16 GPU-hours and
the original 12-hour window; the expanded scientific scope raises the incremental
storage cap from 30 to 32 GB (projected final usage approximately 30.23 GB).
The original manifest and execution history are retained with an append-only
extension in the same campaign ledger; no fresh compute allowance is granted.

Worker code remains the clean frozen revision `237de7d49b05fd485a817f73d8b46d3cc30e4592`.
New pinned lifecycle inputs and public CLI plans are staged under each existing
worker checkout's ignored `results/lifecycles/<new-case-id>/` directory. This
avoids modifying the original tracked configs or running code. Host preparation,
dry-run, queue and extension receipts belong in
`results/lifecycles/opt125m-cmo-l6-exp1e-development/readiness/lr5em3/`.
Native results stay under each new case root; compact collection joins the existing
`collected/` index and eight-candidate review. The persistent watchdog is the
single dispatch and monitoring owner. No broader comparison is released.

The extension was admitted at 14:09 Paris under controller `402acee`, with
76 service tests and 71 experiment preparation/collection/review tests passing.
All four host/case planning dry-runs passed. The append-only extension digest is
`bfe35d99f678f1d72c55f0861879183eb9a77793a53f549203b56ac10042ea3f`.
Its receipt and unchanged prior-attempt/budget checks are in
`readiness/lr5em3/extension.json` and `admission.json`.

| Host | Candidate | Attempt |
|---|---|---|
| fifi, RTX 5090 | linear, 5e-3 | `08f61584d19c4b9783a7f01aefc18739` |
| loulou, RTX 5090 | step, 5e-3 | `4d4d220426ad4d49a6e917aad98de1ca` |

Tmux sessions use `<candidate-lifecycle-id>-<attempt>`. Both workers passed
numerical preparation and emitted their initial HWA selection. The watchdog
observed each worker and supervisor, finite metrics, and at least 101 updates;
subsequent direct reads showed further progress past warm-up. Restarting the
controller preserved the extension, 32 GB cap, original 16 GPU-hour allowance,
matching waiver and exact single attempt per added case. The receipt is
`readiness/lr5em3/live-handoff.json`. The expanded review reports four of eight
completed candidates and no recommendation. Expected completion for the new
5090 pair is 15:00–15:10 Paris, subject to measured throughput and validation.

### Terminal development handoff, 2026-10-10 15:37 Paris

All eight scientific cases have one completed, validated and collected attempt,
with 12,800 updates each. Last collection finished at 15:03 Paris. The local
3090 and nom-cool-1 cases completed collection at 14:25 and 14:37 respectively.
The two 5e-3 cases completed at 15:03 (fifi linear) and 15:02 (loulou step).
The native collection index and eight-case summary are complete; the selected
development winner is `opt125m-cmo-l6-exp1e-dev-step-lr5em3` with KL 0.4299463.
The upper-boundary and late-checkpoint review flags remain unresolved; this is
not an automatic authorization for higher LR, more updates or the broader grid.

Remaining compute allowance is 5.9731 GPU-hours, with zero active reservations;
charged time is 10.0269 GPU-hours including readiness and its failed pre-start
attempt. Observed incremental storage is 30.228 GB against 32 GB. All campaign
GPUs are released. The persistent shared watchdog remains healthy, with a fresh
15:36 heartbeat, no host errors and no new incidents. Preserve its other jobs.
The scientific next decision and limits are in the result note; no further
scientific cases are queued.

### Authorized 1e-2 extension, 2026-10-10

The user's explicit 1e-2 instruction extends both existing schedules to
`opt125m-cmo-l6-exp1e-dev-{linear,step}-lr1em2`, on fifi/loulou RTX 5090s. This
is a second adaptive boundary test after the complete eight-case comparison.
Peak LR changes to 0.01 in both Adam parameter groups; data, initialization,
3% HWA noise, whole-transformer scope, 12,800 updates, 205 warm-up updates,
schedule shape, selection cases and final-test holdout remain unchanged. The
review now requires ten candidates, with 1e-2 the upper LR boundary.

Both new cases retain the measured 50–60-minute expectation, 5,400-second hard
allowance, 900-second renewable tmux lease and 21,500 MiB plus 1,024 MiB memory
reservation. They add about 1.7 GPU-hours and 1.84 GB of artifacts. The existing
16 GPU-hour/12-hour allowances cover them, and the incremental storage cap rises
from 32 to 34 GB (projected final usage approximately 32.07 GB). The same ledger,
append-only scientific-extension mechanism and frozen worker revision `237de7d`
apply. Do not reset charged time, rewrite previous cases, or alter the matching
waiver. The broader comparison remains unsubmitted.

Host, queue, extension and monitoring receipts are under
`results/lifecycles/opt125m-cmo-l6-exp1e-development/readiness/lr1em2/`.
Native outputs use each new lifecycle's existing-host case directory; compact
results join the shared `collected/` index and ten-candidate review. Numerical
divergence is retained as an outcome; no automatic LR change is authorized.

Both candidates were admitted at 15:49 Paris under controller `c29330c` after
71 experiment tests and all four host/case dry-runs passed. Existing lifecycle
files remain byte-identical. Extension SHA-256:
`4f9b2abbeb15583f1193666fdd5180183e1e54e1dc2b43ce237644fb83d2c8e4`.
Preparation and admission evidence is in `readiness/lr1em2/{hosts,extension.json,admission.json}`.

| Host | Candidate | Attempt |
|---|---|---|
| fifi, RTX 5090 | linear, 1e-2 | `e96e67eee9504c558becd405bc1ad42a` |
| loulou, RTX 5090 | step, 1e-2 | `299d83d879ab458f84cacbbc2acd3cd7` |

Each tmux session is `<candidate-lifecycle-id>-<attempt>`. Both workers passed
numerical preparation and initial HWA selection, then reached update 206 with
finite training loss and the intended LR (linear 0.009999206, step 0.01).
The persistent watchdog observed live workers/supervisors and finite metrics.
After controller restart, both attempts remained unique, the two scientific
extensions and 34 GB cap persisted, and the original 16 GPU-hour allowance,
charged time, window and matching waiver remained intact. This handoff is in
`readiness/lr1em2/live-handoff.json`. The regenerated review has eight of ten
completed candidates and withholds a recommendation. Expected finish for both
new runs is 16:40–16:50 Paris, subject to throughput and validation.

### Authorized duration comparison, 2026-10-10

Question: after the current step-schedule runs, does more training at the same
post-decay rate improve development performance, and does it change the ordering
of peak LR 1e-3 and 5e-3? The user accepted this comparison following the
training-duration interpretation in the result note. This is adaptive development,
not an independent confirmation or a new source of training-seed replication.

Add exactly two cases:
`opt125m-cmo-l6-exp1e-dev-step-lr1em3-tail4pass` and
`opt125m-cmo-l6-exp1e-dev-step-lr5em3-tail4pass`. Both start from the same pinned
teacher and run 25,600 updates (100 intervals, 209,715,200 token exposures,
approximately 4.196 corpus passes). Preserve batch 32, sequences per interval
8,192, the 205-update warm-up, the peak through update 10,240, and the tenfold
drop beginning at 10,241. Thus every scheduled LR through update 12,800 is
unchanged, and the additional 12,800 updates use 1e-4 or 5e-4 respectively.
Do not stretch warm-up or move the decay point. This tests a longer low-LR tail;
it does not test stretching the whole schedule over four passes.

All other scientific settings in the frozen comparison remain fixed: teacher,
corpus/cohorts, whole-transformer scope, six analog MLPs, 3% standard HWA noise,
optimizer, seeds, development cases/arrays and 512-update selection cadence.
The selected source still minimizes mean healthy/faulted development KL, with
earliest ties. The native saved source lacks optimizer/RNG state, so these are
fresh complete runs with a matching declared prefix, not exact checkpoint resumes.
Retain the existing cross-GPU randomness waiver; prefix differences are reported
as diagnostics and never trigger an automatic scientific restart.

Report the duration comparison separately from the equal-budget ten-case grid.
For each long run, compare terminal development KL at update 25,600 with that
same trajectory's update 12,800, including healthy and faulty components.
Also report selected checkpoints, late-curve improvement and comparison of the
12,800-update midpoint with the prior short run. Best-score improvement alone
cannot demonstrate a duration benefit because longer runs offer more selection
opportunities. Between the two equal-budget long runs, retain the existing
within-1% preference for lower peak LR. No result automatically releases broader
recipes, extra epochs or final-test evaluation.

The duration choice is a declared deviation from Büchel et al. Appendix I's
two-epoch recipe. Its Appendix H motivates checking slower noisy-training
convergence, but neither its corpus size nor its optimizer is matched here.
Rasch et al. use task-dependent training durations (including 5–20 epochs for
BERT in Appendix A.2.3); that is context, not a prescribed OPT epoch count.
Both reference PDFs were consulted. The specific four-pass budget and fixed
decay point are justified by this campaign's late development improvements.

Dispatch only after both current 1e-2 cases finish execution, collection and
their declared validation. Use the existing fifi/loulou RTX 5090 placements,
one job per GPU, through the same persistent queue and tmux supervisors.
Measured 50–52-minute short trials project to 100–105 minutes per long trial;
allow 7,200 seconds expected and 10,800 seconds hard runtime per case, with a
900-second renewable supervision lease. Memory remains 21,500 MiB plus 1,024 MiB
headroom. Add approximately 3.4 GPU-hours and 1.84 GB retained evidence; expected
cumulative usage is about 15.2 GPU-hours and 33.9 GB. Raise the cumulative cap
to 20 GPU-hours and 36 GB through an append-only user-authorized receipt,
preserving all charged work and the original 23:19:34 Paris deadline.

Use the clean worker revision `237de7d` and existing native training implementation.
New host/case plans and pinned configs go under each worker's ignored
`results/lifecycles/<case-id>/`; preparation and queue receipts go under local
`results/lifecycles/opt125m-cmo-l6-exp1e-development/readiness/tail4pass/`.
Native outputs stay in each case root and compact exports join the existing
`collected/` index, with a separate duration section in its review. The original
manifest, earlier lifecycle files and completed artifacts remain immutable.

The two duration cases were admitted to the persistent queue at 16:09 Paris
under controller `c859f5b`. All four host/case planning dry runs passed, as did
81 experiment tests and 88 service tests. The original twelve grid/smoke
lifecycle files remain byte-identical. The append-only extension SHA-256 is
`429953c5adffd6d58bae8d650e28b362ef4a04654ea275401bc1df2f1484c9a9`.
Its receipt and admission audit are `readiness/tail4pass/extension.json` and
`admission.json` in the evidence root.

At the verified handoff, both new cases are **queued**, with no attempts or GPU
allocations. Each depends on both 1e-2 cases plus its completed short-run
reference; the dispatcher explicitly reports waiting for dependency validation.
The restarted watchdog observed both existing 1e-2 workers and supervisors at
update 4,965 with finite metrics, preserving their exact attempt tokens. Its
heartbeat was fresh, with no host errors or new incidents. The original
manifest, charges, deadline, matching waiver and incident history were retained;
reapplying the extension was idempotent. Effective limits are 20 GPU-hours,
36 GB and the unchanged 12-hour window. Already charged work is 10.0269 GPU-hours;
the active pair reserves 3 GPU-hours, leaving 6.9731 unreserved GPU-hours, enough
for both new three-hour hard allowances. Storage observed was 31.0634 GB.

The single monitoring owner remains enabled `run-watch-queue.service`, with
five-minute semantic checks, bounded read-only incident investigation and its
11-minute systemd watchdog. New cases use their own 7,200-second expected runtime,
not the earlier 3,600-second calibration. Queue and restart-preservation evidence
is in `readiness/tail4pass/live-handoff.json`. Once dependencies pass, tmux
sessions will be named `<duration-case-id>-<attempt>` on fifi/loulou; use
`ssh fifi tmux ls` or `ssh loulou tmux ls` to find them. Expected start remains
16:40–16:50 Paris, with completion roughly 18:25–18:40 if current throughput holds.

The compact review preserves the original ten-case LR comparison (currently
eight complete) and separately records zero of two duration cases in
`collected/review/duration/duration.{json,md}`. Separate duration plots will be
created when evidence arrives. Both sections retain the scientific-review gate;
no broader recipes or deployment rows have been released.

### Live duration handoff, 2026-10-10 18:08 Paris

Both 1e-2 predecessors completed validation and collection. The automatic
dispatcher then launched exactly one attempt for each duration case at 16:56
Paris:

| Host | Duration case | Attempt |
|---|---|---|
| fifi / RTX 5090 | step 1e-3, tail4pass | `c82f05e4104d4c528b41a120864fd4e7` |
| loulou / RTX 5090 | step 5e-3, tail4pass | `d99f78c4ba8c40e2ade57b2c160b6103` |

The watchdog has observed both workers and supervisors with valid metrics;
its heartbeat is fresh, with no host errors or new incidents. Direct artifact
reads showed 17,509 and 17,765 of 25,600 updates respectively. Both emitted
finite selection metrics through update 17,408. Measured tail throughput
projects training completion near 18:38–18:40, with validation and collection
expected by approximately 18:40–18:45 Paris. The later start reflects the
1e-2 step candidate's longer runtime, not a dispatch interruption.

Charged completed work is 11.9285 GPU-hours; the active pair has consumed about
2.3241 additional GPU-hours and retains its six-hour hard reservation. The
unreserved allowance is 2.0715 GPU-hours against the cumulative 20-hour cap.
Observed storage is 32.9006 GB against 36 GB. Existing tmux session naming,
five-minute supervision and the original deadline are unchanged. The result
note records the complete ten-case outcome and explicitly partial duration
observations; no broader jobs have been submitted.

### Terminal duration handoff, 2026-10-10 18:52 Paris

Both duration attempts completed all 25,600 updates and passed native validation
and automatic collection without retries. Loulou's step 5e-3 case used 6,066.08
charged seconds and fifi's step 1e-3 case used 6,174.45. Both GPU reservations
are released, and no campaign jobs remain queued. The twelve scientific cases
and four successful readiness cases are terminal; the historical pre-start
failure remains preserved separately.

The terminal check verified each duration case's 22 collected files against its
receipt, the exact source inspection bindings, one completed attempt per case,
and complete short-reference/duration review coverage. Evidence is in
`readiness/tail4pass/terminal-handoff.json`; final duration tables and figures
are under `collected/review/duration/`. The result note owns interpretation.

Total charged consumption is 15.3287 GPU-hours against 20, with 4.6713 remaining
and no active reservation. Observed storage is 33.9029 GB against 36 GB. The
persistent shared watchdog remains enabled and healthy with no host errors or
new campaign incidents; leave it running for other work. The original deadline
and matching waiver are preserved. Broader dispatch remains
`awaiting_scientific_review`; this terminal handoff does not authorize another
duration extension or an unequal-budget recipe comparison.


### Current cosine execution handoff, 2026-10-10 19:27 Paris

The new comparison is launched. Both 512-update readiness jobs completed native
execution, full artifact validation and compact collection. The immutable handoff
gate passed; all seven main jobs are released. At this observation two are running
and five are queued. Only **fifi and loulou** are eligible, one worker per RTX 5090.

Worker and isolated controller revision:
`b31202f8ad7a373717095c9dfd45303dfef68f0e`.
Main lifecycle digest:
`7daec35c3d992cc2ef028156e55673ab10de91e7441d34a13c11c416324dde88`.
Campaign ledger manifest digest:
`f6704ae69ce65a0a38cef5dd906b54d42670febefee9f21c1381d6828be134f9`.
Every frozen host receipt and job specification is under the local cosine evidence
root's `prepared/`; all 18 host-specific native dry-runs passed.

| Host | Current case | Worker PID | Tmux session |
|---|---|---:|---|
| loulou | `exp1e-cosine-cdt_random_3pct_model` | 630949 | `exp1e-cosine-cdt_random_3pct_model-ed6cb7620d514b338d9b73610c974092` |
| fifi | `exp1e-cosine-pretrained` | 908359 | `exp1e-cosine-pretrained-a9a549c4b40a423a8267e8b9a9ead7d8` |

Attach read-only with `ssh -t <host> 'tmux attach -r -t <session>'`; later jobs
get their own sessions, visible with `ssh <host> 'tmux ls'`. Panes are retained
after completion. Exact boot/start identities, native stage roots and live logs
are in `readiness/live-handoff.json`. Worker logs follow
`/home/filip/analog-research/opt-cmo-exp1/worker-receipts/<revision>/<case>/<attempt>/run.log`.

The whole-model worker has passed its initial selection and update 512; the
pretrained job is on its third assignment. Both advanced on two direct native
checks with finite metrics, and both outer worker observations are persisted by
the service. The next semantic campaign poll is due around **19:29 Paris**, then
at five-minute cadence. Queued jobs consume no GPU time and do not count as stalled.

The existing `run-watch-queue.service` is active with PID 211221, systemd watchdog
11 min, automatic restart, user lingering and renewable 15-minute worker leases.
Its working directory is the pinned local cosine evidence root's `controller/`.
The SQLite ledger remains `/home/filip/.local/state/run-watch/queue/queue.sqlite3`;
there is no second dispatcher or interactive monitoring owner. A live restart
retained both readiness worker identities and produced no duplicate attempts.
Injected incident 5 successfully invoked the bounded read-only investigator and
preserved its diagnosis, recommendation and immutable snapshot. Old jobs,
incidents and charges also survived the controller update.

Readiness charged **0.1012 GPU-hours**. The two active jobs reserve 3.75 GPU-hours,
leaving **14.1488 unreserved GPU-hours** within the 18-hour phase cap (actual active
usage accrues against those reservations). The most recent storage observation is
3.284 GB; finished phase storage is projected at about 21.5 GB against the 40 GB
cap. The phase window ends **2026-10-11 07:17:43 Paris**. Normal completion is
expected around **00:15–00:45 Paris**, roughly five hours after the 19:24 launch.

Measured readiness freezes prepare/devices expectations at 120 s and each
six-case deployment array at 121 s; start/stale thresholds are 600 s. HWA keeps
scope-specific expected duration 6,500 s whole-model / 4,500 s analog and a
900 s stale threshold. Validation and controller collection each allow 600 s.
The service raises an incident beyond 1.5 times a stage expectation. These
thresholds flag anomalies; they do not change LR, data, selection or training
length. Read-only investigation and explicitly classified operational recovery
remain bounded by the existing budgets. No scientific retuning is authorized.

Each source job validates its complete finite fit before native deployment.
On exit it undergoes remote full-artifact inspection, then CPU-only controller
collection. The reusable collector preserves compact control files and exact
artifact receipts; large native caches, sources and deployments stay on their
execution host. `collected/deployments.json` contains only the main comparison;
`collected/readiness-deployments.json` holds canaries. At all 210 main rows, the
completion hook applies the frozen paired rule and writes `collected/review.txt`.
There is no downstream on-chip dispatch in this experiment. Scientific
interpretation remains partial until all required coverage has been reviewed.

Verified handoff evidence: `readiness/handoff-tests.json`, `handoff-gate.json`,
`live-handoff.json` and `dispatch-handoff.json` under
`results/lifecycles/opt125m-cmo-l6-exp1e-cosine/`.


### Current cosine handoff, 2026-10-11 00:10 Paris

Six of seven main jobs are complete, validated and collected (five trained
sources plus pretrained), covering 180/210 deployment rows. Only
`exp1e-cosine-standard_hwa_9pct` is active on loulou, attempt
`db7bc812848943f8a40c163db783a7bb`; direct native observation reached update
24,777/25,600 with finite metrics and live worker/supervisor. Fifi has no remaining
campaign allocation. The active tmux session is
`exp1e-cosine-standard_hwa_9pct-db7bc812848943f8a40c163db783a7bb`.

The shared watchdog remains active, with a fresh heartbeat and no host errors.
No new operational incidents or retries occurred. Collection and the automatic
full-coverage comparison remain armed for the final source; expected finish is
around 00:20 Paris. The result note records partial paired findings without
changing the frozen protocol or dispatching any further experiment.

Charged GPU time is 8.0377 h, plus about 1.1324 h of active allocation
(9.1700 h estimated total against 18 h); observed storage is
20.746 GB against 40 GB. The original 07:17:43 Paris deadline
and all prior charges are preserved. Exact observations are in the cosine
evidence root’s `readiness/progress-20261011-0010*.json`.
