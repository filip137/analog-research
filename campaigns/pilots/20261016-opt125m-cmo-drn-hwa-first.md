# CMO-initialized OPT-125M DRNs: optimizer selection and five-pass HWA

This executes the final plan in thread `01a140d3-3043-7f63-b0f3-b7930a7c56f5`,
including the user's subsequent choice **T=4, K=4, A=5, B=0.1** and authorization
to implement, tune and run on trex, fifi, loulou and riri. The first study asks
whether the six initialized coupled DRN MLPs train successfully and whether
std3 HWA improves programmed performance over matched clean adaptation.
On 2026-10-17 the user assigned the complete implementation, HWA execution and
on-chip recovery, requested a monitoring agent, and authorized adjustments to
smokes/protocol where necessary for execution. Recovery is now an assigned phase
after the two matched HWA sources; this does not add the superseded crossbar,
std9 or cdt3 studies.

## Frozen comparison

OPT decoder layers 6–11 each use a perfect-diode 1536–6144–1536 DRN, replacing
one complete logical 768–3072–768 MLP. Attention, residuals, embeddings and the
tied head remain digital. All student parameters train. The pretrained OPT
teacher remains frozen. Four graph-free settling sweeps precede four BPTT
sweeps in training; evaluation uses four settling sweeps. There is no solver
depth search; residuals are reported at the fixed settings.

Initialize from the std3 CMO digital master with SHA-256
`86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028`.
Preserve its digital weights, channel scales, gains and biases. The separate
teacher SHA-256 is
`2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9`.
The pinned teacher revision is `27dcfa74d334bc871f3234de431e71c6eeba5dd6`.
Source device arithmetic comes from HWA commit `b0c1f06`.

Map each MLP into 9–88.199997 µS with the existing positive/negative rail lift,
using the full conductance span independently for both matrices (`full_span_quad.v1`),
input gain 100 and a positive least-squares readout gain. The user's 2026-10-17
instruction replaces the prior conductance-scale search. For each logical matrix,
normalize `w_hat = w / max(abs(W))`. The four targets are
`G++ = G-- = G_MIN + (G_MAX-G_MIN) max(w_hat, 0)` and
`G+- = G-+ = G_MIN + (G_MAX-G_MIN) max(-w_hat, 0)`, where the first sign is
the source rail and the second is the destination rail. They are four distinct
physical devices, initialized in equal pairs and independently trainable/noisy.
Use the first shuffled training sequence (256 tokens) to fit only the readout
gain; report mapping error and assembled development metrics. Mapped artifacts
record the initialization contract, preventing reuse of the prior scale-search mapping.
Stored conductance masters use `u=(G−MID)/HALF_RANGE`; forward conductances use
`g=G/G_MAX`, giving gradient factor `HALF_RANGE/G_MAX` automatically.

Training uses the pinned 195,226-sequence synthetic corpus, sequence length
256, effective batch 32, sequence microbatch 1. Five complete passes give
30,505 updates and 249,889,280 input-token presentations per final arm.
Hold one std3 device view throughout each effective batch and checkpoint
recomputation. Project negative surrogate conductances to zero with identity
straight-through gradients, counting projections. Every programmed forward
uses held apparent conductances; the CMO model has identical persistent and
apparent states. Reject invalid physical conductances without modifying them.

Select on development teacher KL after programming and verify partner rewrite,
nominal/random-3%, assignments 211001 and 211002. Keep the source's trained
digital parameters during selection. Report loading changes caused by rewrite.
No hidden fault information reaches the rewrite controller. Selection retains
the earlier checkpoint on ties; tuning uses final-budget scores.

## Tuning and execution budget

Three optimizer groups: conductance masters, 32,268 calibration parameters,
and remaining digital parameters. Recipe A is Adam `(0.9,0.999), eps=1e-8`,
no clipping. Recipe B is Adam `(0.9,0.98), eps=1e-6`, global norm clip 1.
Neither uses weight decay. Clip after gradient accumulation and coordinate
conversion.

Eight screens use conductance LR `{1e-4,3e-4,1e-3,3e-3}` × recipes A/B,
digital LR `1e-5`, calibration LR equal to conductance LR. Eight refinements
retain the best conductance rate per recipe and compare digital rates
`{3e-6,1e-5}` and calibration rate equal to conductance or digital rate.
Confirm one winner per recipe before choosing the final recipe. Within 1%
ties prefer smaller rates, then A. All candidates restart from the same mapped
state, with matched batches/noise and 32-update warm-up then constant LR.

The tuning allowance is 24 GPU-hours including readiness and failed attempts.
After measuring update, selection and checkpoint costs, choose the largest
complete tier projected to fit 19.2 GPU-hours: screen/refinement/confirmation
updates are 256/512/2048, 256/512/1024, or 256/256/512. Do not rank incomplete
coverage or launch long arms from an unconfirmed recipe. No improving valid
candidate means an inconclusive tuning outcome.

The clean and std3 final arms restart from the mapped state with fresh Adam,
245-update warm-up and cosine decay to 20% of peak. Select every 512 updates
and at epoch boundaries; retain selected and fixed-final weights. Evaluate
generated text and WikiText only after selection. Their token budgets are
separate from the tuning GPU-hour cap. Estimate runtime from readiness before
admission. Aggregate artifacts must remain within 100 GiB; retain tuning final
weights/metrics, and best/latest-resumable/final for long runs. Preserve failed
and partial evidence.

## Assigned physical recovery

For each selected clean/std3 source, deploy nominal and random-3% devices with
assignment 271002, independently of selection assignments. Save raw endpoints,
the canonical partner-rewritten apparent state, original verify flags, frozen
source targets, physical fault identities, write RNG and separate initial costs.
Each source/case has five controls: hold, calibration, physical learning,
learning followed by calibration, and uncapped frozen-target rewriting.
Learning-plus-calibration consumes the completed learning state, sharing exactly
its physical trajectory. Other controls restore the canonical deployment.

Physical learning uses the pinned b0c1f06 thresholded Adam law, LR .003,
betas (.9,.999), epsilon 1e-8, signed command accumulation, strict 2-sigma
nominal-midrange threshold and four attempted writes per physical device.
Failed/stuck attempts consume the cap. Initial programming/partner rewrite do
not consume that cap. All digital parameters stay fixed during physical learning;
every forward uses the current held apparent conductances. Physical writes are
unclamped; negative/nonfinite realized states fail explicitly. The HWA STE now
uses `held + (master-master.detach())` so the forward equals held values exactly.

Predeclare 2048 effective-batch-32 physical update opportunities per learning or
rewrite control (16,777,216 training-token presentations for learning).
Calibration fits the existing 32,268 parameters on the development cohort using
200 full-cohort Adam updates at LR 1e-4, selecting minimum development KL with
earlier ties, then freezes before held-out evaluation. No conductance writes
occur during calibration. The rewrite control targets every device on each
opportunity and is explicitly uncapped. Report costs per device and logical
weight and compare fixed-final physical-learning states. Measure GPU duration
before admitting these substantial runs; retain the aggregate 100 GiB limit.
Completed recovery retains its final state and removes redundant latest partial
checkpoints with an audit receipt. Partial runs retain full continuation state.

The public `ebl train` stages `deploy` and `recover` use the existing experiment
ID; recovery receives explicit `--weights` and `--device-state`, or `--resume`.
`ebl validate` with a recovery config restores the saved physical state, verifies
development replay and evaluates held-out corpora. Local tiny tests cover all
five controls and exact optimizer/device RNG continuation; full-size execution
remains to be measured.

## Current handoff

**2026-10-18: paused for the user's experiment review.** No study job is queued
or running at the latest read-only queue check. The unclipped noiseless .0003
trajectory was cancelled at update 646 for the user's Gmin correction; it did
not fail numerically. The .0001 case is unsubmitted and paused, and the older
unsubmitted recovery branches remain paused. The user subsequently requested
review before further execution. Gmin-floor code is present in the working
tree and its 16 recovery tests pass, but no clipped numerical snapshot or job
has been launched. Preserve the failed and cancelled evidence; do not advance
the controller or launch replacement cases during this review. The newest
interpretation is recorded in the review below. The following handoff history
preserves earlier execution decisions and is not current launch authority.

Implementation and CPU validation now include physical recovery; full-size local
mapping and HWA readiness passed. Subsequent CUDA initialization began failing
before numerical execution, including a direct driver check outside the sandbox.
On 2026-10-17 the user explicitly approved transferring this experiment's code,
configs and required checkpoints to `/home/filip/analog-research/opt-cmo-drn-first-v1/`
on loulou, trex, fifi and riri, and said these transfers should always be approved.
This is continuing authorization for the assigned experiment's staging and
checkpoint transfers; do not ask again for the same scope. Remote execution
resumes with source v9 and the prepared readiness job after digest verification.
Public runtime:
`python -m ebl train --config examples/opt_cmo_drn/prepare.json --teacher-weights
TEACHER --weights STD3 --output-dir OUTPUT`; HWA consumes the explicit mapped
artifact with `--weights`. Resume uses `--resume` and a full training checkpoint.
`python -m workflow plan` emits the native campaign manifest; numerical stages
remain public ebl subprocesses.

Outputs belong under `results/opt125m-cmo-drn-hwa-first-v1/`. Local numerical
tests use CPU. Full-size readiness uses one authorized 5090, bounded to 30
minutes initially, with four development sequences and two training updates.
Readiness counts against the tuning allowance and establishes operation only.
GPU inventory on 2026-10-16 found all four 5090s idle. Recheck at admission;
preserve unrelated workloads. The shared run-watch queue will own admitted
jobs. Monitoring agent `/root/run_monitor` verified the live shared controller
(PID 2529587), idle loulou/riri, and no existing study jobs on 2026-10-17.
The local RTX 3090 also has no compute workload/reservation and 22,777 MiB free.
The earlier authorized local 20-minute readiness envelope permits a 20 GiB peak
plus 2 GiB headroom. Prepare and two-update HWA each receive at most ten minutes,
using local exact teacher, std3 source and corpora. Admission rechecks resources.
The local source snapshot is `results/opt125m-cmo-drn-hwa-first-v1/source-readiness-v6/`.
Preparation v1 failed before numerical execution: the new configs used `experiment`
instead of the public dispatcher's `experiment_id`. The failed attempt and immutable
v6 snapshot remain preserved; it consumed 1.665 s. The schema/examples and a public
CLI regression were fixed. Snapshot v7 includes the correction and a local 20 GiB
CUDA allocator cap. Retry job `opt-cmo-drn-local-prepare-v2`, attempt
`c20d669da654479d8918c52a19b7af30`, completed with run/validator exit 0 in 12.704 s
(5.514 s native preparation). Its native bundle is
`results/opt125m-cmo-drn-hwa-first-v1/local-prepare-readiness-v2/run`.
Mapped weights SHA-256:
`5ab69be92c6eb17dff5c46d299066727cd8a96a7515dd49b9fd4c003fedf951a`.
On four development sequences (1020 scored tokens), clean KL was 0.159108 and
perplexity 11.4944; all denominators were positive and maximum projected voltage
defect was 1.19e-7. These are readiness measurements, not study coverage.

Two-update std3 HWA completed as `opt-cmo-drn-local-hwa-v1`, consuming that exact
mapped checkpoint and snapshot v7. Native output:
`results/opt125m-cmo-drn-hwa-first-v1/local-hwa-readiness-v1/run`.
The shared queue owns launch/monitor/validation; `/root/run_monitor` performs bounded
live checks, while root owns readiness incident fixes (no duplicate automatic incident
agent). Maximum job time is 600 s, with a 480 s internal guard and no automatic replay.
Inspect with `python ~/.codex/skills/run-watch/scripts/gpu_queue.py status opt-cmo-drn-local-hwa-v1`.
Attempt `8252c983b2a742b0ad395e6141c61a7b` passed native artifact validation with
both exit codes zero. Updates took 4.7907/4.7313 s; peak CUDA allocation was
5,419,230,208 bytes (5.05 GiB). All three parameter groups had finite nonzero
gradients; no surrogate cells were projected. Selection took 6.2503/5.9713 s
on the four-sequence readiness cohort. KL changed from 0.193944 to 0.160149,
chiefly through nominal-device improvement; random-3% KL was approximately
unchanged (0.22810 to 0.22777). This is execution evidence only. The final
weights digest is `4e0fb2f0aacdf7e19e656d90fe424a050c57bc8111c262bcfd25d7087c3`.
Combined preparation and HWA attempts consumed 66.520 s of the existing
24 GPU-hour tuning allowance and local 20-minute readiness envelope.

The attempted remaining local readiness uses snapshot v8 and the same 20 GiB allocator cap:
one HWA update with the actual 256-sequence selection cohort (300 s queue cap,
240 s inner guard), random-3% deployment of the explicit two-update HWA final
state (120 s queue cap), then 16 physical recovery updates from that saved
deployment (300 s queue cap, 240 s inner guard). Source, device assignment,
writer LR .003 and 2-sigma threshold are unchanged. Sixteen updates replace
the original two-update physical smoke because two commands cannot cross the
declared write threshold; the readiness check must exercise actual writes.
Checkpoint every eight updates and record write counts, timing and peak memory.
These three jobs reserve at most another 720 s, within the remaining 1133.480 s.
Expected durations are approximately 90/30/120 s; artifacts remain under the
existing 20 GiB local readiness cap (currently 5.49 GiB, projected final 13 GiB).
Each job uses public ebl execution and native artifact validation. Root owns
fixes; the shared queue owns processes; `/root/run_monitor` checks progress.

Selection job `opt-cmo-drn-local-selection-v1`, attempt
`a74688c73f134af7bd3d17e278bbc0a0`, failed at CUDA initialization in 1.865 s;
deployment job `opt-cmo-drn-local-deploy-v1`, attempt
`f8898c4cbb0649068ea676209d841de6`, failed at the same call in 2.657 s.
Both exited 1 before native run creation. Their receipts, commands and logs
are copied into their respective `local-*-readiness-v1/launcher-evidence/`
directories. No physical recovery job was submitted. The monitor confirmed
both processes exited. Total readiness charge is now 71.043 s; 1128.957 s
remain in the local allowance. `nvidia-smi` still sees the idle card, device
nodes exist and driver/runtime versions match, but direct `cuInit(0)` returns
error 3 outside the sandbox. Root cause is not established; no module reload,
GPU reset or unrelated workload change was attempted. Retry only after CUDA
initialization succeeds. There are no active numerical jobs for this study.
The next goal-turn check reproduced direct CUDA error 3 outside the sandbox.
A read-only syscall trace opened all three NVIDIA device nodes successfully,
with no device-access permission error; this does not identify the driver's
internal failure. Queue receipts still show both failed checks terminal.
Read-only remote probes confirmed working CUDA on loulou and riri (PyTorch
2.11.0+cu128, RTX 5090, 32,149 MiB free each). Their capacity must still be
rechecked at admission. These probes transferred no experiment files and do
not resolve the pending source-staging approval.

Latest source snapshot v9 contains 68 files, 745,753 bytes. It adds lossless
`bank.v2` checkpoint encoding: pack the three Boolean masks into bits and save
stuck conductances only at corrupt cells. Runtime tensors, fault identities,
held state, write RNG and optimizer state are unchanged; legacy dense `bank.v1`
inputs still load. Exact continuation tests pass for nominal and random devices.
Tensor payload estimates for all 18 tuning finals, two full HWA arms and all
four deployment/five-control groups total about 77.4 GiB; with the maximum
20 GiB local readiness retention this remains below 100 GiB. Check actual
checkpoint and transient-save usage before each substantial phase. No existing
evidence was pruned or rewritten. Compact measurements and estimates are in
`results/opt125m-cmo-drn-hwa-first-v1/readiness-summary.json`.

The current prepared remote job is
[`20261016-opt125m-cmo-drn-hwa-first.prepare-v2-job.json`](20261016-opt125m-cmo-drn-hwa-first.prepare-v2-job.json),
replacing the unsubmitted v1 proposal. It allows one placement on loulou, riri,
trex or fifi, reserves 20 GiB plus 2 GiB headroom, expects 30 s and caps the job
at 300 s. It uses v9 code, exact public commands, hash-pinned existing teacher
and source, native validation, and remote output
`/home/filip/analog-research/opt-cmo-drn-first-v1/results/prepare-readiness-v2/`.
The job is schema-validated and its transfer prerequisite is now approved.
Verify source/input staging before submission; admission must recheck GPU
ownership and capacity.
Source v9 has now been copied with rsync to all four approved hosts. Independent
read-only checks verify its 68 file digests and the exact teacher, std3 source,
training corpus and both held-out corpora on each host. The monitoring agent
confirmed loulou CUDA availability, 32,149 MiB free, no compute process/lease,
and the live shared controller. Prepare-v2 is ready for shared-queue admission;
it retains the existing 300 s cap and all readiness charges remain in the
original 24 GPU-hour allowance.
Remote preparation completed on fifi: attempt `81c5742849f447128ec428cc4dc78649`,
both execution and validator exit 0, 2.347 s native and 6.072 s charged. Mapped
artifact `results/prepare-readiness-v2/run/checkpoints/mapped.pt` on fifi is
840,764,555 bytes with SHA-256
`340f1402757744c9054f8d19ba4ae6e5bc2272a3032414d2d409f3eb13f7ec1a`.
Development KL 0.159108409 and perplexity 11.4943964 agree with the earlier local
smoke; all six mappings have positive denominators. This exact remote mapping
will initialize all subsequent matched remote arms. Total charged readiness is
77.115 s. The completed bundle is being collected to local
`results/opt125m-cmo-drn-hwa-first-v1/remote-prepare-readiness-v2/`.

Full-cohort HWA timing is submitted as
`opt-cmo-drn-first-hwa-selection-readiness-v1` on fifi, using v9 and that exact
mapped checkpoint. Spec: `20261016-opt125m-cmo-drn-hwa-first.hwa-selection-job.json`;
remote run: `/home/filip/analog-research/opt-cmo-drn-first-v1/results/hwa-selection-readiness-v1/run`.
It performs one effective-batch-32 update and initial/final selection on all
256 development sequences, with a 300 s queue cap and 240 s inner guard.
The shared queue owns execution/validation; `/root/run_monitor` checks bounded
semantic progress and root owns fixes. Next: use measured timings for tuning,
then deploy its explicit final checkpoint and exercise 16 physical updates.

Automatic approval review rejected both a broad source transfer and the
reduced Python dependency closure because it required explicit authorization
for the payload and destination. The subsequent user approval resolves this
blocker. The local payload is under
`results/opt125m-cmo-drn-hwa-first-v1/source-readiness-v9/`; it contains only
the execution's Python dependencies, seven exact configs and a hash receipt.

## Local verification

The selected test command is:

```bash
/home/filip/miniconda3/envs/py312/bin/python -m pytest -q \
  tests/test_opt_cmo_drn.py tests/test_opt_cmo_drn_tuning.py \
  tests/test_opt_cmo_drn_recovery.py \
  tests/test_ebl_cli.py tests/test_campaign_schema.py \
  tests/test_core_engine_stages_contract.py tests/test_campaign_runner.py \
  tests/test_example_configs.py
```

105 tests passed after adding physical recovery, public CLI coverage and lossless
physical-checkpoint encoding. Coverage includes all five
matched controls, exact physical/writer continuation, pinned upstream write-Adam
goldens, held-state validation and permanent faults, plus
four-device signs and both rail layouts, rejection of artifacts without the
new initialization contract, dense resistive-equation parity, input and
conductance finite-difference gradients, gradients through all six blocks,
held views during checkpoint recomputation, projection gradients, pinned CPU
device goldens, source reconstruction, tiny preparation/clean/std3 lifecycles,
native artifact validation, reuse invalidation, and interrupted-run exact
continuation. Tuning tests enforce complete phase coverage, the 1% tie rule,
the measured-cost tier decision and the no-improvement stopping condition.

The real pinned teacher and std3 source also load and reconstruct locally.
The full-size structure has 113,246,208 conductance parameters, 32,268
calibration parameters and 96,895,488 remaining digital parameters.
`results/opt125m-cmo-drn-hwa-first-v1/local-structure.json` records that check.
It performed no full-size forward or training and is not throughput evidence.

## Continuation commands

`workflow plan` emits a native campaign from an HWA config and explicit
`--teacher-weights`, `--source`, `--output`, and optional completed `--reuse`
preparation bundle. Reuse verifies native artifact hashes, source identities,
solver settings and the mapping's training cohort dependencies. `workflow
collect` takes explicit native run paths and distinguishes partial, failed,
invalid, missing and duplicate evidence.

After readiness, put measured update, selection, checkpoint and charged
readiness seconds into a timing JSON and use:

```bash
python -m workflow.tuning size --timings TIMINGS.json --output BUDGET.json
python -m workflow.tuning screen --base-config BASE.json \
  --budget-decision BUDGET.json --output SCREENS
```

The `refine`, `confirm` and `final` subcommands additionally require the prior
`--previous ladder.json` and `--runs` JSON mapping every candidate ID to its
explicit completed native bundle. They verify hashes and matched initialization
before generating the next phase's strict configs. No helper launches jobs.
The final stage produces no long-run configs if neither confirmation improves
on mapped initialization.

Scientific coverage is currently incomplete. Full-cohort HWA selection,
physical learning, calibration and rewrite readiness now pass on fifi, and the
measured full tuning tier is admitted below. No optimizer has yet been selected;
neither five-pass arm nor held-out evaluation has run. The user's standing
transfer approval has unblocked remote execution. Local CUDA failure remains
isolated to the unused local execution path. Full HWA, matched physical recovery
and scientific interpretation remain incomplete.

CMO is a fitted model-based control, not this worktree's measured synapse
dataset. These results cannot establish that on-chip recovery is needed.

### Remote readiness continuation, 2026-10-17

Full-development HWA readiness completed on fifi with native artifact validation.
One batch32 update took 1.813055 s, full256 selection took19.503072/19.560497 s,
and the checkpoint took1.564870 s. Queue charge was50.746037 s. Preparation
charge was6.072026 s. All future tuning and long arms use mapped artifact
SHA340f1402757744c9054f8d19ba4ae6e5bc2272a3032414d2d409f3eb13f7ec1a.
Deployment readiness passed, charge16.689320 s; canonical device-state SHA
54e847ec3330af181860a0ceada0a43346ffbd36f953b7637f4209d157dab3ea.
Initial raw/rewrite development KL was5.901298/0.180289; this is programming,
not learned recovery. Learning16 and calibration1 full-development checks are
queued as opt-cmo-drn-first-recover-readiness-v1 and
opt-cmo-drn-first-calibration-readiness-v1. Both restore this exact deployment.
Queue caps300/180 s, inner guards240 s; expected120/60 s. Shared queue owns
process supervision; /root/run_monitor checks bounded admission and completion.

Production transport will use the existing shared queue with bounded CPU-only
completion callbacks for collection and deterministic native-plan progression.
There is no second GPU scheduler. All phases require complete validated coverage.
Canonical completed bundles are retained locally. Once all declared remote
consumers of an artifact have completed, a hash-verified remote replica may be
removed with a migration receipt; original manifests and failed/partial evidence
remain intact. This replica-retirement policy avoids storing two complete studies
inside the aggregate100GiB allowance. Active or uncollected artifacts are never
pruned. Source v9 remains immutable; transport changes do not alter numerics.

### Production admission and progression

All full-size readiness controls completed with validated artifacts. Charged
readiness including failed attempts is234.880120 s. The measured full ladder
projects19687.829709 s (5.469GPUh), below the19.2GPUh planning limit; select
256/512/2048 updates. Eight screens are prepared as queue jobs
`opt-cmo-drn-first-screen-{A,B}-{0,1,2,3}`. Expected~8.8 minutes each with
~16-minute queue caps; all four approved hosts are eligible, one queue-owned
worker per GPU. Trex's unrelated run is preserved; scheduler admission chooses
available capacity. All hosts use PyTorch2.11.0+cu128 and numerical snapshotv9.

Native screen results: remote `.../opt-cmo-drn-first-v1/results/screen-*/run`,
canonical local `results/opt125m-cmo-drn-hwa-first-v1/execution/runs/screen-*/run`.
The immutable local `controller-v1` runs CPU-only callbacks through the existing
queue. Settings, prepared job specs and progression state live under
`results/opt125m-cmo-drn-hwa-first-v1/execution/`; `workflow/queue_execution.py`
is its versioned implementation. The callbacks require full validated phase
coverage and matching mapped/teacher identities. They generate the existing
strict tuning configs, then the two five-pass arms only on confirmed improvement.
Expected full HWA duration is~15.8 hours per arm; recovery learning~1.1 hours
per case, calibration~51 minutes per control, rewrite~2 minutes. Queue runtime
caps include startup, validation and conservative measured-throughput margins.
Tuning worst-case allowances remain below the original24GPUh cap.

After both HWA arms finish, existing native recovery plans define four deployments,
20 controls and20 physical held-out evaluations, plus selected/final HWA held-out
evaluations. Every physical child consumes the exact saved parent; learning-plus-
calibration consumes learning. No callback ranks held-out scores. Callbacks collect
and hash-validate canonical bundles, retire only unneeded verified replicas with
audit receipts, and defer submissions when observed storage plus reservations
would exceed100GiB. Missing/failed cases block their dependent decisions.
The shared queue is the durable monitor; bounded incident recovery is enabled
for production, with no automatic scientific changes or blind replay. Root owns
review; numerical completion leaves scientific_review pending.

The16-update learning smoke finished with34,118,379 write attempts, including
1,028,787 stuck attempts. Its development KL worsened from0.180289 to4.251389;
this is a cautionary readiness observation, not selected-source recovery evidence.
The fixed recovery protocol remains unchanged. Calibration1 used zero writes,
took15.117198 s/update and reachedKL0.177839. Two full-device rewrite opportunities
took0.027858/0.011605 s and validated successfully. These timings size the
predeclared controls; they do not select their hyperparameters.

All eight screen submissions were acknowledged by the shared queue. Each has
a528s expected duration and958s total attempt allowance. No screen was deferred
for storage: initial observed use23,270,359,013bytes plus54,902,665,772bytes
conservative active-job/input reservations fits107,374,182,400bytes. This
includes readiness replicas pending validated migration. Source/controller
snapshots are immutable. The13 transport/tuning tests and14 example-config
checks passed; existing numerical tests remain unchanged and previously passed.
`/root/run_monitor` owns one bounded live admission/progress check; the shared
queue remains the sole durable process monitor.

Initial semantic progression is verified: A-0 on fifi, worker3023278, attempt
aaf50fcdfa9a476ea1584266c6431f24, reached12/256; A-1 on loulou, worker3534538,
attempt164f28f3af97412ebcce811eee57c070, reached10/256; A-2 on riri, worker3122540,
attempt39862c415ba54ea78f650f3b464855fc, reached9/256. Each has exactly one worker,
fresh metrics, finite gradients, zero surrogate projections and initial selection
KL0.1716474940653379. Actual allocations are6348MiB; updates take1.76–1.79s.
Five cases remain queued. ControllerPID2529587 is alive with admission enabled;
its60s observations and native metric growth establish durable supervision.
Next automatic checks are the queue's next observation and each completed
screen's collection callback; phase promotion waits for all8 validated screens.
Use `python ~/.codex/skills/run-watch/scripts/gpu_queue.py status JOB_ID` and
`execution/execution-state.json` under this study root for operational status.
The bounded monitoring-agent check is complete; no interactive polling owner
is required while the shared queue is healthy. Incident recovery remains within
existing budgets and preserves failed attempts.

The additional full-progression transport test now passes:8screens→8refinements
→2confirmations→2full HWA arms→44native recovery stages+4HWAheld-out stages,
68jobs total. It checks every canonical deployment dependency and the shared
learning trajectory. The focused transport/ladder suite now has14passing tests.
This adds validation only; running controllerv1 and numerical snapshotv9 are
unchanged. Scientific interpretation remains pending actual completed coverage.

Controllerv2 is now frozen for future phase submissions; existing screen hooks
remain on unchangedv1 and read the atomic settings pointer when preparing the
next phase. Numerical snapshotv9 and all scientific configs are unchanged.
The transport update retires unused remote checkpoint replicas by artifact kind,
so selected weights remain available while redundant resumable/final copies
can migrate independently. Storage reservations cover measured temporary-save
overlap:11GiB/full HWA,8GiB/confirmation,6GiB/screen or refinement,9GiB/learning.
Native plan regeneration now compares canonical JSON hashes, allowing safe
callback retries after partial plan writing. The25 focused transport, tuning
and recovery tests pass, including full-DAG regeneration. No jobs were restarted.
Five verified remote readiness checkpoint replicas (9,420,583,365bytes) were
retired with migration receipts; complete canonical bundles remain local.

All six remote readiness bundles are now collected and locally validated with
`ebl runs inspect --verify-artifacts --require-complete`. Their checkpoint
replicas were migrated from fifi after all readiness consumers completed;
`readiness-migration.json` records11,676,400,849bytes retired. Canonical mapped
and HWA readiness weights, deployment, learning, calibration and rewrite states
remain intact locally. Production uses separate digest-pinned mapped copies.
Latest queue check still shows three healthy running screens and five queued;
there is no completed tuning coverage or final scientific conclusion yet.

### Status, 2026-10-17 approximately18:20 Europe/Paris

All18 tuning candidates completed, were collected and passed native artifact
validation. Total charged readiness+tuning is5.543934GPUh of the24GPUh cap.
At2048 confirmation updates, recipe A scored0.0827046014 and recipe B scored
0.0818004687, versus mapped initialization0.1716474941. These are development
teacher-KL means after programming and partner rewrite over nominal/random3%
and two assignment seeds. The predeclared rule selected B, with conductance and
calibration LR1e-4 and digital LR3e-6. This establishes a confirmed optimizer
choice; it does not yet compare final clean adaptation against std3 HWA.

The matched five-pass arms automatically started: full-clean on fifi, attempt
4a53706c293d4fbe9808aff375416ba3; full-std3 on loulou, attempt
94564b83f74d4d1281cfb84f5be843db. Both reached69/30505 updates at the live check.
Initial programmed selection matches exactly; metrics are less than2seconds
old, all gradient groups are finite, and no surrogate cells are projected.
Recent mean update times are1.8133s clean and1.7679s std3. Including selection
and checkpoint overhead, completion is estimated for Sunday morning18October
Europe/Paris, roughly15–16hours after launch. The shared controller remains
live with fresh observations; no study failures were reported.

Recovery and held-out evaluation have not started. They remain automatic
dependents of the completed matched HWA sources, under the existing exact
state, coverage, runtime and storage checks. Final scientific review is pending.

### Full HWA results and collection incident, 2026-10-18

Both five-pass runs completed30,505updates and passed remote native validation.
Clean: selected programmed developmentKL0.0787545492 at step1536; fixed-final
KL0.0921026181. Std3: selected/finalKL0.0719846552 at step30505. On selected
checkpoints, nominal/random3% means are0.0166138274/0.1408952711 for clean and
0.0228794511/0.1210898594 for std3. Thus std3 improves the selection objective
and corrupt-device performance while nominal fidelity is worse. These are
development observations under the declared fitted CMO control; held-out
evaluation and the matched physical-recovery conclusions remain pending.

Full-clean compute attempt4a53706c293d4fbe9808aff375416ba3 completed successfully
in57,169.919126chargedseconds. Its controller collection
c42ee91080484ceb8bce188db8863568 failed with rsync exit11 because the shared
/home filesystem filled; source artifacts remain intact on fifi. The bounded
incident agent preserved evidence and marked attention. This is an operational
collection failure, not failed training. Std3 collected successfully after
55,857.510824chargedseconds.

Root is migrating only this study's42.678GB tree to
`/tmp/filip-opt125m-cmo-drn-hwa-first-v1` on the separate root filesystem, which
has137.7GiB free. Every file, including partial/failed evidence, is byte-hash
verified before removing its original replica. The original results path is
preserved as a symlink; storage accounting uses the physical target. The
overflow-migration receipt records exact copied files. This is temporary
overflow storage: retain it until a verified move back to persistent project
storage is possible. No unrelated files are removed. Retry only collection
through the existing shared queue after migration; never repeat successful GPU
training. Recovery is waiting for both source bundles to finish collection.

The local collection capacity check has also been corrected in the versioned
transport code: future admissions require both the100GiB aggregate study cap
and sufficient free space on the actual collection filesystem, including
reserved output sizes and2GiB headroom. The new regression reproduces a nearly
full collection filesystem while the study is below its aggregate cap; all17
focused transport/tuning tests pass. This changes admission only, not numerical
settings, source identity, optimizer choice or physical controls. A new frozen
controller snapshot will apply to future submissions after the storage move.

Overflow migration completed:1375files totaling42,678,055,572bytes were SHA-256
verified. The original results path now resolves to the overflow directory.
Original operational settings are retained byte-for-byte in
`execution/settings.before-overflow.json`; the migration receipt records every
file. Home now has42,708,058,112bytes free; overflow has105,171,271,680bytes free.
Future phase submissions use immutable controllerv3 with the filesystem-capacity
guard. Existing successful compute and sourcev9 remain unchanged.

The supported `gpu_queue.py collect opt-cmo-drn-first-full-clean` command
acknowledged the safe collection retry. The shared queue owns the worker;
/root/run_monitor is performing a bounded progress check. Recovery/held-out
progression waits for local validation of this bundle. No training was rerun.

The retry subsequently collected and validated full-clean. Execution state now
records all 20 tuning/full runs complete and expands to 68 declared jobs,
including recovery and held-out evaluation. The shared queue has started
`clean-deploy-none` on fifi, `clean-deploy-random` on loulou, and
`clean-best_weights-held-out` on riri. The completion hook is still staging the
remaining eligible cases. The bounded monitor check is complete; the shared
queue remains the durable monitoring owner. Recovery and held-out measurements
are not yet available, and scientific review remains pending.

### Partial recovery evidence, 2026-10-18

At this check, 37 of 68 declared stages are collected and validated. For the
selected clean source on nominal devices, hold/calibration/physical-learning
development KL is 0.0166351617 / 0.0140767236 / 0.0813399773. Physical learning
completed the fixed 2048 updates and attempted 437,123,720 writes, averaging
3.859941 per device against the four-attempt cap. It worsens this nominal case;
calibration uses zero device writes and improves it. These are fixed-protocol
observations, not a reason to tune or rerun the arm.

For the selected clean source with random 3% faults, hold/calibration/rewrite
development KL is 0.1444786212 / 0.0846459973 / 5.7707249324. Generated held-out
KL is 0.1421088009 / 0.0847838677 / 5.7053199718; WikiText KL is
0.2040160031 / 0.1336655509 / 5.1872575204. Calibration improves both held-out
cohorts without conductance writes. The uncapped frozen-original-target rewrite
control is substantially worse than holding the partner-compensated deployment;
it is distinct from the initial partner rewrite. Physical learning for this
fault case is not yet run.

HWA nominal calibration is live on fifi (child PID 3275005), with fresh metrics
at update 12/200. Nominal clean learning-plus-calibration and several held-out
checks are queued or collecting. Remaining admissions are constrained by the
declared aggregate 100 GiB allowance, with pending collections and verified
replica retirement releasing reservations. The shared queue retains monitoring
ownership. HWA physical learning and all matched learning-plus-calibration
comparisons remain pending; no conclusion that on-chip learning is needed is
supported by this partial fitted-CMO evidence.

### Recovery learning-rate diagnosis, 2026-10-18

The optimizer ladder tuned off-chip HWA only. Physical learning has used one
fixed LR, 0.003; no physical-learning LR range has been tested. Its negative
nominal result therefore does not exclude a better physical optimizer setting.
Read-only replay of the recorded metrics shows no writes through update 9;
update 10 issues 249,666 writes. Pre-update training-minibatch KL is about
0.009-0.011 before those writes and 0.471661 at update 11, then falls to
0.065920 at update 2048. Batches differ, so this is an early degradation signal,
not a paired development-cohort comparison. Final development KL remains
0.0813399773 versus the deployed 0.0166351617.

Read-only inspection of the final writer state finds attempt-count frequencies
for 0/1/2/3/4 writes of 889,928 / 1,514,197 / 2,248,062 / 3,262,685 /
105,331,336 cells: 93.0109% of devices exhaust the four-attempt cap. The strict
write threshold is 0.0282790843 in normalized conductance, or 1.1198516946 uS.
Lowering LR changes when accumulated commands cross that threshold; it does
not remove the minimum issued command magnitude or the programming noise.
An LR/threshold/write-cap interaction is a plausible explanation, not an
identified cause.

Proposed follow-up, not launched or added to the active protocol: compare
physical LR 1e-4, 3e-4, 1e-3 and the existing 3e-3 from identical saved
deployments, with matched batches, RNG initialization, 2048 updates, threshold
and cap. Record development KL, first-write behavior and budget saturation;
select on development only. Distinguish improvement from merely avoiding
writes. If smaller LR only delays the first degradation, a separately declared
threshold/write-noise diagnostic would be needed. Preserve the current fixed
protocol as evidence while testing this hypothesis in a new comparison.

### Deeper recovery diagnosis and revised next steps, 2026-10-18

Read-only checkpoint analysis and a bounded CPU sensitivity diagnostic are
implemented in `labs/opt_cmo_drn_recovery_diagnose.py`. The final report is
`results/opt125m-cmo-drn-hwa-first-v1/analysis/recovery-diagnosis-20261018-noise.json`;
the adjacent `recovery-diagnosis-20261018-source.json` records code/report hashes
and confirms the numerical modules match immutable source-readiness-v9.
Earlier, narrower reports are retained. No new training arm or protocol change
was launched. The diagnostic modifies conductance views only in memory.

Seven focused checks passed: conductance/input gradcheck, checkpoint-held-view
parity, held-forward independence from masters, pinned writer golden, two
physical-state/RNG replay cases, and stuck-attempt accounting. Source weights
and every stored digital/master tensor exactly match the completed learning
checkpoint; changes are in the held physical bank. No sign, unit-conversion,
hidden-forward or frozen-parameter error was found in these checked paths.

Across the twelve initial nominal arrays, sampled median conductance is
9.244-9.289 uS. Thus the fixed 1.11985 uS threshold, specified as two sigma at
midrange, is approximately 9.6 sigma at the typical cell, and 8.05-8.80 sigma
when averaging the ratio over all cells within each array. Final physical
changes have RMS 1.827-2.044 uS by array. These observations suggest a coarse
update scale as well as the already observed 93.0109% write-cap exhaustion.

The CPU diagnostic uses the first 64 tokens of development sequences 0 and 1
(126 scored tokens), with the exact selected clean source and canonical
nominal deployment. It is a local sensitivity probe, not checkpoint selection
or full-cohort evidence. Evaluation KL is 0.0141345253; training-forward KL is
0.0141345244. The analytic directional derivative is -344.6306 and the central
finite-difference estimate is -343.1977. A noiseless first-Adam-direction step
of 1e-5 lowers KL to 0.0112642972; 1e-4 raises it to 0.0329157785, and 3e-3
raises it to 3.1309032440. This supports useful gradient direction with strong
sensitivity to collective update size on this cohort.

A second proxy selects the 249,666 largest current-gradient cells, matching
the first production burst's count but NOT its identities, batch or accumulated
commands. All selected cells receive commands at 1%, 10% or 100% of the current
threshold. With/without the declared CMO programming noise, KL is respectively
0.0112609714/0.0111976331, 0.1078051999/0.1089572757, and
2.1131625175/2.1189417839. Noise uses a fixed diagnostic RNG; this is not an
exact replay of the production trajectory. Within this proxy, command magnitude
dominates the immediate degradation, while much smaller commands still help
with noise present. The evidence does not establish the optimum physical LR or
threshold, or generalize to corrupt and HWA sources without matched checks.

The corrupt rewrite failure has a separate, directly supported mechanism.
Initial partner-compensating cells are RMS 42.67-44.60 uS away from their
original source targets by array. Frozen-original-target rewriting brings
them back to RMS 0.135-0.169 uS from those targets, removing compensation for
stuck partners. Development KL returns from compensated 0.1444786 to
5.7707249, near raw deployment 5.7722121. This is expected behavior of that
declared destructive-to-compensation control, not failed gradient learning.

The earlier LR-only follow-up suggestion is insufficient and is superseded by
this diagnostic order: (1) capture and replay the actual first write burst with
identical cells and commands, measuring same-cohort KL immediately before/after;
compare noiseless/noisy and scaled-command ablations; (2) screen LR jointly
with threshold and write-batch size, tracking command-gradient alignment,
per-layer conductance changes and cap exhaustion, and distinguish fewer writes
from beneficial writes; (3) add a compensation-preserving rewrite control if
rewrite is intended as recovery, explicitly retaining compensated targets or
reapplying apparent-verify partner compensation; (4) use development-selected
settings in a new matched clean/HWA, nominal/corrupt comparison against hold,
calibration-only and learning-plus-calibration, then evaluate held-out data.
The ongoing fixed-protocol controls remain evidence and keep their settings.
These follow-ups are recommendations, not newly authorized long GPU sweeps.

Calibration's benefit is condition-dependent: nominal clean held-out generated
KL is 0.0158905 after calibration versus hold 0.0157659, and WikiText is
0.0389227 versus 0.0388335, despite better development KL. The previously
reported corrupt-case held-out gains remain supported. Full physical-learning
coverage is still incomplete; all conclusions remain scoped to the fitted CMO
model and the tested controls.

### User removes the per-device write cap, 2026-10-18

The user requested removal of the write budget after the capped-run diagnosis.
Interpretation: remove the four-attempt cap per physical device, retaining the
2048 update opportunities, LR .003, strict nominal-midrange two-sigma threshold,
programming noise, initial deployments and device identities. This tests cap
exhaustion separately from the coarse-command hypothesis. The aggregate storage
and operational runtime limits remain in force. A full noiseless trajectory has
not been executed; only the explicitly labelled CPU sensitivity probes exist.
Whether to add full noiseless learning is an open optional scientific choice.

Explicit `recovery.max_write_attempts: null` now means unlimited attempts. Counts,
held states, RNG and optimizer history remain serialized; loading across capped
and uncapped controller contracts is rejected. Capped numerical goldens remain
bit-exact. Thirty focused recovery/transport/tuning checks pass, including
uncapped writes beyond four attempts, stuck-device accounting, exact native
resume, and revision of only unsubmitted plans.

Prepare four uncapped learning cases (clean/std3 sources, nominal/random3%
devices), each followed by calibration and held-out evaluation. Every learning
case restarts from its explicit original saved deployment. Started capped arms
and their downstream controls remain references; wholly unsubmitted capped
learning branches are superseded with their definitions retained in the
execution state. Public ebl runs and the existing shared queue remain the
execution and monitoring surfaces. Expected learning duration is about 1.1 GPU
hours per case, calibration about 51 minutes, plus brief held-out checks; queue
per-case allowances retain the existing measured bounds. The only additional
completed-case repeat currently required is clean/nominal learning, because its
capped trajectory is already finished. Results use distinct
`execution/runs/<source>-<condition>-uncapped-*` paths under this study root.
Stage immutable numerical source v10 and controller v4 before applying the
revision; existing frozen sources/configs and evidence remain intact.

Revision applied: source `source-uncapped-v10` contains 68 files, 746,135 bytes,
receipt SHA-256 `2da1bdb8135c1a19755bfcf6e546defcc8ada4c4f24b4a113241a5095f47f217`.
All four authorized hosts verified every file and parsed the null-cap config.
The live settings now use controller-v4 and the v10 job template; original
settings are retained in `execution/settings.before-uncapped.json`. Twelve
unsubmitted capped branch definitions are preserved under `superseded_jobs`;
the active plan has 72 stages, with 42 collected at transition. The old capped
clean/nominal trajectory and its already submitted downstream controls remain
intact. All sixteen uncapped learning/calibration/held-out stages are registered.

Uncapped learning admission is currently deferred by the separate 100 GiB
aggregate storage allowance: observed 87,820,547,588 bytes plus 19,394,461,696
reserved bytes. This is distinct from the removed per-device write limit and
from filesystem free capacity. The shared queue will reconsider admission on
completion/collection callbacks. Current compute is clean/nominal calibration
after the capped learning trajectory on fifi; std3 nominal calibration has
completed. Queue-managed artifact progress and controller ownership were checked.
No full noiseless learning arm has been registered or launched pending the
optional choice presented to the user.

### Clean/random uncapped learning incident, 2026-10-18

Queue job `opt-cmo-drn-first-clean-random-uncapped-learning`, loulou attempt
`3be35600f77f4cd5b5ca42d2785cdd80`, failed with exit 1 after
130.650494 charged seconds. Remote native status confirms a `FloatingPointError`
in `decoder.layers.11.fc2`: invalid physical conductances after programming.
The last recorded completed update is 75 (87,766,210 attempted writes); the
run's checkpoint directory is empty. Evidence remains at
`/home/filip/analog-research/opt-cmo-drn-first-v1/results/clean-random-uncapped-learning/run`
on loulou.

Incident decision: blocked. The declared physical contract rejects negative or
nonfinite realized conductances without clipping or resampling; the exception
does not distinguish those two conditions. No understood operational repair is
established. Safe replay is not declared and no prepared resume command exists.
Changing the physical law or optimizer requires a scientific decision. No
compute or collection was launched, and no queue state, scientific settings or
budget was changed. Preserve this partial failed trajectory; it is not completed
recovery evidence.

### Collection and independent progress after a terminal failure, 2026-10-18

Root identified an operational admission bug: the stopped failed learning case
still retained its full 9 GiB future-output reservation, deferring every other
uncapped case. Controller-v5 now collects terminal failed bundles separately,
records `failed` without adding `completed`, and excludes their future-output
reservation only after a terminal nonzero receipt, dead worker and released
queue lease are verified. Actual failed artifact bytes remain in measured
storage. Failed cases and descendants are never retried or treated as completed;
independent declared cases can proceed. Fifteen focused queue tests pass,
including stopped-worker/lease checks and dependent-versus-independent admission.
The numerical source, scientific configs, physical law and 100 GiB limit are
unchanged.

For subsequent stopped failures, the existing incident owner may run the
ordinary CPU `advance` action from the controller named in execution/settings.json:
`python -m workflow.queue_execution advance --settings
/home/filip/analog-research/lora/results/opt125m-cmo-drn-hwa-first-v1/execution/settings.json`.
This preserves failed evidence, reconciles reservations and admits only already
declared independent jobs. It does not retry failed compute or authorize a
scientific parameter change. Keep failed branches visibly unresolved for review.

Collected evidence now covers 56/72 stages. Capped clean/nominal learning's
generated/WikiText held-out KL is 0.0775599539/0.1636811675; following it with
calibration gives 0.0772555829/0.1622080724, still worse than holding the original
deployment at 0.0157658626/0.0388334762. Calibration does not repair the damage
from this learning rule. For HWA with random 3% corruption, calibration-only
improves generated KL from 0.1192717379 to 0.0567572341 and WikiText KL from
0.1856539585 to 0.1018037327, with zero recovery device writes. Its development
KL is 0.0570466909. Full noiseless learning remains unexecuted.

Controller-v5 is active. The failed clean/random bundle is collected under
`execution/failed-runs/clean-random-uncapped-learning/3be35600f77f4cd5b5ca42d2785cdd80/run`
and passes native artifact inspection as a valid failed bundle, not complete
recovery. Its 130.650494 seconds remain recorded. The stale 9 GiB reservation
is released without retrying its compute. The exact clean/nominal deployment
was restaged to fifi, and `opt-cmo-drn-first-clean-none-uncapped-learning` is now
submitted and queued under the verified existing shared controller (PID 2529587).
Both HWA uncapped cases remain deferred by the aggregate storage bound. The
completed count remains 56; failed branches remain excluded from completion.

### Clean/nominal uncapped learning incident, 2026-10-18

Queue job `opt-cmo-drn-first-clean-none-uncapped-learning`, fifi attempt
`e5b7d0f7eaaa43f78e2844c12f40a6cb`, failed with run exit 1 after
168.711126 charged seconds. Read-only remote inspection confirms native failed
status at 14:35:11 UTC with invalid physical conductances in
`decoder.layers.11.fc2`. Last completed update: 97, with 58,843,564 attempted
writes and teacher KL 0.1394902. The checkpoint directory is empty. Preserve
the partial bundle at
`/home/filip/analog-research/opt-cmo-drn-first-v1/results/clean-none-uncapped-learning/run`
on fifi.

Incident decision: blocked. The declared negative/nonfinite conductance guard
fired; the exception does not distinguish these conditions. No understood
operational defect is established, safe replay is not declared, and this job
has no prepared resume command or checkpoint. Changing the physical law or
optimizer requires a scientific decision. No compute, collection, controller
advance, queue mutation or budget change was performed. This is failed partial
evidence, not completed recovery or a collection failure after successful compute.

### Assigned noiseless physical learning and LR check, 2026-10-18

The user now explicitly assigns noiseless on-chip learning and a check of the
physical learning rate. Start with four matched **clean/nominal** trajectories
at LR `1e-4, 3e-4, 1e-3, 3e-3`, each with 2048 effective-batch-32 updates.
All restore selected clean weights and the same canonical deployment:
source SHA-256 `ca2066689da576c0e8193b13aac339a0722e5a64495e0e066129ef67ec260372`,
deployment SHA-256 `e872679349710bc54233096a5b0b8808b4a32c6b8798dad59480e9c4ea51e676`.
Keep assignment 271002, full cohorts, frozen digital parameters, Adam betas
(.9,.999), epsilon 1e-8, signed accumulation and the strict nominal-midrange
2-sigma threshold. Writes remain uncapped. Set only **recovery** write noise
to zero; initial programming errors and held deployed conductances stay intact.
The population noise scale used to calibrate the threshold stays unchanged.
Consume the same write RNG draws even when noise is disabled. Stuck cells
remain stuck in the implementation; this first LR screen itself is nominal.
This is a fitted-CMO-model control, not measured on-chip evidence.

Record fixed full-development KL at step zero, immediately after the first
actual write burst and every 128 updates, as well as final KL, actual write
counts and pending-controller diagnostics. Preserve the first burst's exact
cell indices, normalized commands and pre-write RNG states. All scoring and
gradients use held apparent conductances. The same fixed initial state gives
the exact pre-first-write development baseline; minibatch KL is secondary.
An invalid physical state stops the case with explicit negative/nonfinite
counts and minimum finite conductance; no clipping, resampling or retry with
different scientific settings is authorized.

The decision is whether a lower LR yields useful physical learning under the
existing threshold without write noise. Compare valid fixed-final scores to
the baseline; zero-write trajectories are holds, not recovered models. Require
at least 1% lower full-development KL for a provisional improving candidate;
within 1% prefer the smaller LR. Keep failed cases as excluded/partial evidence,
never rank them as completed trajectories. If no candidate improves, report
that this LR range is insufficient under the current threshold; do not infer
that all noiseless learning or all LR choices fail. The previous noisy .003
nominal trajectory is a matched initial-condition reference with partial
coverage through update 97. It is not a complete paired LR-by-noise grid.
Generalization to HWA sources and corrupt deployments, a noisy LR confirmation,
and threshold changes follow review of this first screen; they are not silently
substituted for one of these four cases. Pause only unsubmitted old uncapped
branches pending this review; preserve completed/failed/submitted cases.

Expected compute is approximately 4–5 GPU-hours across these cases, using
measured update and development-scoring timings. Per-case queue allowances are
`1.8 * expected_seconds + 120`, approximately 2.2 hours each, including all
attempts; total operational allowance is below 9 GPU-hours. Use the existing
authorized hosts and shared queue, retaining physical-parent placement on fifi
for this nominal screen. Cases may serialize under memory/storage admission.
Outputs are `execution/runs/clean-none-noiseless-lr{3e-3,1e-3,3e-4,1e-4}/run`
under the existing study root. Exact configs live in `execution/configs/`.
The 100 GiB aggregate storage limit is unchanged. Keep full resumable partial
checkpoints every 128 updates. Completed finals retain exact model weights,
physical bank, RNG and scalar controller diagnostics, omitting optimizer
moment tensors that cannot be used to resume a completed case. The existing
latest-to-final retention receipt remains required. Old completed/failed
bundles are not changed. Unneeded remote input-cache replicas may be retired
only after verifying the local canonical copy and checking future consumers
on that specific host; keep audited receipts and restage by exact digest if
a future explicitly declared consumer needs the copy again.

Monitoring stays with the existing shared queue, with 60-second process/artifact
checks and bounded incident dispatch. The current controller's ordinary
`advance` command is authorized after a verified stopped failure to collect
the failed bundle, release future-output reservations and admit another of
these four declared independent cases. Numerical failures are scientific
outcomes, not permission to retry or change settings. Root reviews fixed-cohort
scores and terminal coverage in this note. A source/controller freeze and live
launch handles will be recorded below after readiness checks.

The four exact versioned configs are `examples/opt_cmo_drn/noiseless_lr3e-3.json`,
`noiseless_lr1e-3.json`, `noiseless_lr3e-4.json` and `noiseless_lr1e-4.json`;
their generated execution configs match exactly. The main CPU checks passed
53 tests, with 31 recovery/transport checks rerun after the final validation and
cache-retirement changes. Frozen numerical source `source-noiseless-v11` contains
68 files / 750,881 bytes; its receipt SHA-256 is
`2b932d0d4fae91f062f48efafdadd1a5fdd91c2cd4c8db7cba40af9cdd8231a3`.
All four authorized hosts verified every source file and the new config parser
under PyTorch 2.11.0+cu128. Original numerical snapshots remain immutable.

The .003 case is running on fifi as queue job
`opt-cmo-drn-first-clean-none-noiseless-lr3e-3`, attempt
`81b3ad341b054e3bb17224cb35619909`, supervisor PID 3324614 / worker PID 3324615.
The shared controller remains PID 2529587, with native manifest and metric
growth observed. Its transient `admission_enabled:false` status during the
observation part of a cycle is not a global queue hold; normal admission
started this case. Queue checks remain every 60 seconds. Each new LR case has
expected duration 4339 seconds and total queue allowance 7929 seconds; the
native wall-time guard is 7809 seconds. The other three are registered and
admitted sequentially as storage permits. Current controller settings own
submissions and collection; use the controller directory named in
`execution/settings.json`, not a remembered older version.

Storage retention additionally permits removal of **remote replicas** for
unsubmitted paused branches, after verifying their canonical local bundles.
No already prepared/submitted job is paused this way. Any later resumed branch
must restage the exact canonical inputs through the existing resolver. A
stopped failed case's checkpoint replica may likewise migrate to its verified
canonical failed bundle after collection; preserve the complete partial state,
metrics, errors and a replica-migration receipt. These changes avoid counting
both large copies indefinitely and do not delete canonical scientific evidence
or raise the 100 GiB limit. Controller-v7 implements this retention extension;
the numerical source and scientific settings are unchanged.

First observed result: the full 65,280-token development baseline is exactly
0.016635161651042746 KL, matching the saved deployment. At update 10 the first
249,666 **noiseless** writes raise that same-cohort KL to 0.3437085472485598
(20.6616 times baseline). The 3,006,813-byte `artifacts/first-write.pt` stores
the actual commands and identities, including 153,159 writes in layer 11 fc2.
This establishes immediate degradation without recovery write noise under
the current .003/2-sigma update rule. It does not establish the best LR or
the final recovery outcome. At the next direct observation, update 66 was
complete with 43,554,222 writes; the trajectory was still running and no
partial checkpoint had yet been due. Smaller-LR outcomes remain pending.

The .003 noiseless case subsequently failed during update **97**, after 96
completed updates and 58,353,319 recorded write attempts. The guard identifies
one negative conductance, zero nonfinite values, minimum -0.2600735426 uS, in
`decoder.layers.11.fc2`. Charged runtime is 175.323704 seconds. No checkpoint
was due before failure. The failed native bundle is collected at
`execution/failed-runs/clean-none-noiseless-lr3e-3/81b3ad341b054e3bb17224cb35619909/run`;
native inspection reports valid structure with failed process state. Its
first-write artifact has SHA-256
`a6d854b63be52adfcfc4f28409d046826e91b772cc5fe529d88d2125c0e6c82b`.
The failed-bundle inspector declares zero terminal artifacts; the additional
first-write file is separately hashed in the analysis receipt, rather than
being presented as a complete run's verified artifact manifest.

Actual first-write magnitudes have minimum/median/maximum
1.1198521 / 1.1386740 / 1.1998321 uS. The first ten pre-write minibatch KL values
and gradient norms are bit-identical to the matched noisy .003 run. The noisy
control failed after 97 completed updates; the noiseless control failed after
96. Removing recovery write noise therefore does not remove this rule's early
damage or negative-conductance failure. The current .003 setting is unsuitable
under the tested threshold; LR versus threshold remains unresolved until the
smaller-rate trajectories are observed. Compact evidence is retained in
`analysis/noiseless-lr-initial-20261018.json`.

Controller-v7 identity SHA-256 is
`5464db1dc048f61e328b30827087c25f311c62ade65a312e45f308321f01b65c`;
all 18 queue tests pass after the retention extension. At its first advance,
actual aggregate storage was 86,339,721,924 bytes with 19,327,352,832 reserved,
below 107,374,182,400. The .001 and .0003 cases are now submitted; .0001 remains
registered and storage-deferred. No failed case is retried. Current scientific
coverage is one failed partial trajectory out of four planned LR cases.

The .001 case has now launched on fifi, attempt
`1991cbebecc04846b107fde39ec27bc0`, worker PID 3326335. A second direct check
confirms update 29 and fixed-development artifacts. Its first write occurs at
update 28 with only 26 cells: development KL is 0.01663376947536188 versus the
same 0.016635161651042746 baseline. This negligible early change is below the
declared improvement criterion and is not a final result. The .0003 case is
queued behind it; .0001 remains storage-deferred. The earlier failed case's
first-write SHA was also independently matched between fifi and the collected
local copy. The existing shared owner has acknowledged the new .001 process;
the next deterministic observation is due on its next 60-second cycle, with
the 7929-second total per-case allowance unchanged. Root's next scientific
review is at a terminal case or an explicit user update, using the fixed
development series; healthy runs need no additional polling agent.

### Noiseless LR .001 incident, 2026-10-18

Attempt `1991cbebecc04846b107fde39ec27bc0` failed with exit 1 after
1548.110182 charged seconds. Read-only inspection on fifi confirms native
failed status and 888 completed updates, with 63,482,801 recorded write
attempts. The subsequent write failed the physical guard in
`decoder.layers.10.fc2`: one negative conductance, zero nonfinite values,
minimum -0.13896608352661133 uS. The frozen v11 implementation applies
noiseless commands without clipping, as declared. Evidence remains in
`/home/filip/analog-research/opt-cmo-drn-first-v1/results/clean-none-noiseless-lr1e-3/run`
on fifi; coverage is failed/partial, not a completed LR candidate.

Incident decision: blocked. This is the predeclared invalid-physical-state
stop, with no established operational defect. Safe compute replay is not
declared and no prepared resume command exists; changing the physical law
or scientific settings requires a new decision. No compute, collection,
controller advance, cancellation or queue mutation was performed. The current
incident assignment prohibits submissions/state changes, so the earlier
general advance authorization was not exercised. Failed evidence and the
original budget remain intact; failed-bundle collection belongs to the
existing controller, not a successful-compute collection retry.

### Experiment review at the user's pause, 2026-10-18

Scope: this OPT125M six-DRN campaign, using the fitted CMO model. These are
simulation controls, not measured on-chip recovery results. Teacher KL is the
primary comparison below; lower is better. The 56 completed execution stages
include validation and deployment stages and are not 56 independent training
experiments. There are 18 completed optimizer candidates, two five-pass arms,
four deployments, four hold controls, four calibrations, four original-target
rewrites, one capped physical-learning trajectory, one calibration of that
trajectory, and 18 held-out evaluations. Readiness checks are separate.

Both five-pass arms start from the same full-span four-device DRN mapping of
the pinned std3 digital master. The clean arm disables HWA during this phase;
it is not a never-HWA-trained digital teacher. Mapping and numerical readiness
passed, but this campaign does not compare alternative initializations or
conductance spans. Programmed development selection KL at initialization is
0.1716474941. Off-chip tuning selects recipe B with conductance/calibration LR
1e-4 and remaining-digital LR 3e-6 (confirmation KL 0.0818004687 versus recipe
A 0.0827046014). This does not tune the physical-learning LR.

Both full arms complete 30,505 updates. Selected programmed development KL is
0.0787545492 for clean (step 1536) and 0.0719846552 for std3 (step 30505), an
8.60% reduction in the declared nominal/random-3% mixture. Their selected
generated/WikiText programmed mixture KL is 0.0770813094/0.1198100776 and
0.0706334241/0.1180604937 respectively. Thus the held-out mixture improvement
is smaller on WikiText. Clean fixed-final mixture development KL is
0.0921026181: longer clean adaptation improves unprogrammed fidelity but
worsens the deployment objective after its selected checkpoint. These are
selected-model comparisons, not equal-step checkpoint comparisons.

On independent deployment assignment 271002, raw versus partner-compensated
development KL is 0.0166352685 -> 0.0166351617 (clean nominal),
5.7722121314 -> 0.1444786212 (clean random-3%),
0.0229465116 -> 0.0229464755 (HWA nominal), and
5.1850668748 -> 0.1214751912 (HWA random-3%). Partner compensation is essential
in these corrupt controls. HWA reduces the corrupt gap but worsens nominal
fidelity relative to clean adaptation.

Calibration updates 32,268 digital parameters and performs zero recovery
conductance writes. All four calibration controls and their held-out
evaluations complete:

| Source/devices | Development hold -> calibration KL | WikiText hold -> calibration KL |
| --- | --- | --- |
| Clean/nominal | 0.01663516 -> 0.01407672 | 0.03883348 -> 0.03892273 |
| Clean/random-3% | 0.14447862 -> 0.08464600 | 0.20401600 -> 0.13366555 |
| HWA/nominal | 0.02294648 -> 0.01861742 | 0.04715658 -> 0.04435108 |
| HWA/random-3% | 0.12147519 -> 0.05704669 | 0.18565396 -> 0.10180373 |

The corrupt-case gains also hold on generated text. Nominal-clean calibration
slightly worsens both held-out cohorts despite improving development KL.
HWA plus digital calibration is the strongest completed corrupt recovery
control; its benefit does not establish a need for physical learning.

Capped physical learning completes only for clean/nominal at LR .003:
development KL 0.01663516 -> 0.08133998, WikiText 0.03883348 -> 0.16368117.
Following with calibration gives 0.07457971 and 0.16220807 respectively.
93.0109% of cells exhaust the four-attempt cap. This tested rule is harmful;
cap saturation alone is not an identified cause. Removing the cap at the same
LR with write noise yields two partial failures (nominal after 97 updates,
random-3% after 75) at the invalid-conductance guard. Those incomplete runs do
not establish the eventual outcome with a bounded physical write rule.

All four frozen-original-target rewrite controls complete. Nominal scores stay
near hold, while corrupt development KL returns to 5.77072493 (clean) and
5.18538409 (HWA). This undoes the compensating partner conductances, matching
the raw deployment failure. It is distinct from the successful initial partner
rewrite and is not a test of gradient learning.

Read-only gradient checks find no sign, unit-conversion, held-forward or
frozen-parameter error in the checked paths; analytic versus finite-difference
directional derivatives agree within about 0.4%. The tiny diagnostic cohort
benefits from a much smaller ideal command. Actual .003 noiseless learning
confirms the first 249,666 writes raise fixed full-development KL from
0.0166351617 to 0.3437085472. Recovery write noise is therefore not required
for that early damage. The fixed minimum issued command is about 1.12 uS;
LR also changes write timing, count and accumulated directions.

New read-only extraction from the stopped fifi native metrics distinguishes
the remaining noiseless LR outcomes. All use the same initial deployed state;
only recovery write noise is disabled, not the initial programming error.

| LR | Coverage | Fixed development evidence |
| --- | --- | --- |
| .003 | Failed after 96 updates; one negative cell | First-write step 10 KL 0.3437085472 |
| .001 | Failed after 888 updates; one negative cell | Step 128 KL 0.0389170428; step 768 KL 0.0429554837 |
| .0003 | Cancelled at 646 for the clipping change; no numerical failure | Step 128 KL 0.0182139874; 256: 0.0168262135; 384: 0.0164089900; 512: 0.0163068786; 640: 0.0162486072 |
| .0001 | Unsubmitted, paused | No evidence |

At step 640 the .0003 case has 666,612 actual writes and 2.3237% lower full
development KL than its initial 0.0166351617; at the last completed update
646 it has 678,864 writes. This is evidence of modest beneficial physical
updates in a partial noiseless trajectory, not merely a zero-write hold.
It supersedes any suggestion that reducing LR can only postpone degradation.
It has neither completed the predeclared 2048 updates nor received a held-out
evaluation, so it does not yet satisfy the fixed-final selection rule. Native
evidence remains at `fifi:/home/filip/analog-research/opt-cmo-drn-first-v1/results/clean-none-noiseless-lr3e-4/run`;
the latest resumable checkpoint is at step 640. The cancellation receipt is
authoritative for process state even if native status still says running.

No Gmin-clipped recovery result exists. The correct unresolved comparison is
whether bounded, appropriately tuned physical learning gives a reproducible
held-out gain beyond matched HWA-only and digital-calibration controls.
The present evidence supports HWA and calibration for random faults, identifies
a destructive rewrite control and harmful coarse updates, and leaves physical
learning useful/necessary as an open question. Execution stays paused while
the user reviews this breakdown.

### Requested DRN checkpoint collection, 2026-10-18

The user requested the DRNs after initialization, after clean/HWA KL training,
and after deployment with program-and-verify. The local collection is
`results/opt125m-cmo-drn-hwa-first-v1/exports/drn-stages-20261018/README.md`;
its `manifest.json` records exact native paths, SHA-256 values, paired configs,
selection steps, existing cohort-labelled metrics and verification results.
These are relative links to retained native artifacts, not duplicate tensors
or a standalone archive. The study still resolves to temporary overflow storage.

The collection has initialized weights, clean selected/final weights, HWA
selected/final weights, and four deployed model snapshots (clean/HWA crossed
with nominal/random-3%). Selected clean is step 1536; both fixed finals and
selected HWA are step 30505. HWA selected/final tensors are bit-identical.
Saved deployments derive from selected weights. The deployed model snapshots
are existing zero-update hold artifacts containing both weights and physical
bank; original deployment-only artifacts are linked alongside them. For all
four cases, every model tensor equals the selected source, all twelve held
arrays equal the original canonical deployed arrays, and recovery updates and
writes are zero. Device-facing evaluation must use the saved held arrays.

Public `ebl runs inspect --verify-artifacts --require-complete` passed for all
11 source bundles (preparation, two full training runs, four deployments and
four holds). No numerical training/evaluation was launched, no Gmin clipping
was applied, and the execution pause remains in effect. The initialization
clean KL in the collection is the original four-sequence readiness measurement;
it is explicitly not presented as comparable full-development evidence.
