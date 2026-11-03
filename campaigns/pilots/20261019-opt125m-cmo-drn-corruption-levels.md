# Fixed-LR DRN recovery across random corruption levels

## Question and assignment

Does the recovery selected at 3% random permanent corruption also improve the
other levels, holding the learning rates and update horizon fixed? On 19 October
the user assigned scheduling this extension on Jean Zay's V100 development queue.
This extends the completed [3% LR pilot](20261018-opt125m-cmo-bounded-recovery.md).

## Frozen comparison

Five new random-fault fractions: 1%, 2%, 4%, 5%, 6%, assignment seed 271002.
For each, program and partner-compensate the same HWA source once, save the held
physical state, and use it for a hold control and both learning laws. The existing
3% results remain an explicitly labelled RTX 5090 reference; no new 3% LR tuning.
Different GPU/PyTorch programming draws can differ, so comparisons to the old 3%
point are not exact cross-host replications. Within a rate both arms share one
saved state and one fault map.

HWA input: `results/opt125m-cmo-drn-hwa-first-v1/execution/runs/full-std3/run/checkpoints/best.pt`,
SHA-256 `dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9`.
Keep the pinned teacher and all corpus hashes from the previous pilot.
Six perfect-diode DRNs, four settling/four gradient sweeps, effective batch 32,
microbatch 1, 256-token sequences, fixed digital parameters and held apparent
forwards throughout. The original target was 2,048 updates per learning case.
On 20 October the user accepted the existing 1,954–2,002-update endpoints
instead of further training; the exact counts below supersede that horizon.

- Noiseless: direct Adam, constant LR 1e-4, bounded targets, no write noise.
- Noisy: accumulated bounded Adam, constant LR 3e-4, minimum issued command
  2ρ using target-dependent write-noise SD before writing.
- Both: unlimited device writes, target clipping to [9, 88.199997] µS before
  programming, unchanged permanent faults, no post-noise endpoint clipping.

No intermediate checkpoint selection or further tuning. Evaluate each fixed-final
endpoint and its hold on development, generated held-out and WikiText held-out.
Report KL, perplexity, attempted/healthy/state-changing writes and the fraction of
hold KL removed at each rate. Useful recovery requires lower development and both
held-out KLs plus nonzero physical state changes. One assignment does not establish
replication or a necessity for on-chip learning.

## Execution and readiness

Verified project: `umg` (default active allocation), account `umg@v100`, partition
`gpu_p13`, QOS `qos_gpu-dev`, constraint `v100-32g`, one GPU and ten CPUs per task,
two-hour allocations. Module `pytorch-gpu/py3/2.5.0` supplies Python 3.12.7 and
PyTorch 2.5.0. Existing Jean Zay teacher/corpus files passed exact hash checks.

The initial admission is one singleton canary, covering both laws at 6% with
32 updates each and 60-second inner checkpoint boundaries. It uses full-size
models/batches/cohorts, actual deployment, hold, training, held-out validation and
native artifact inspection. Each law must exercise full-state continuation.
It measures throughput and validates the immutable worker, wrapper, source,
resource contract and artifact path before production admission.

Production is ten cases, at most five concurrent tasks. The same worker gives
each continuation up to 6,000 training seconds within its two-hour allocation;
the remaining time covers initialization, save and evaluation. Native TimeoutError
bundles at this declared boundary are preserved, and only their exact full-state
checkpoints may continue with identical config, source and PyTorch. A real failure
blocks further automatic submissions. No reduction of update horizon, batch,
solver or model is an operational retry.

Provisional ceiling pending measured readiness: 60 V100 GPU-hours including
canary and continuation allocations, at most four production waves, 200 GiB of
new remote artifacts and 100 GiB local collection. These are upper limits, not
runtime predictions. The canary determines the expected cost before release;
an estimate outside these limits requires review. Queue waiting is additional.

Source/configs/wrapper are frozen under
`$WORK/opt125m-cmo-drn-rates-v1/source-v2/`. Current outputs live under
`$SCRATCH/opt125m-cmo-drn-rates-v1/retry-v2/`; the shared HWA input remains
under the parent study directory’s `inputs/`. Local evidence root:
`results/opt125m-cmo-drn-rates-jz-v1/` (backed by `/tmp/filip-opt125m-cmo-drn-rates-jz-v1/`).
Every numerical stage uses public `python -m ebl train/validate` commands.
`workflow.slurm_recovery` only transports and sequences those calls.

The frozen manifest records every config, source digest, Slurm command and input.
An exclusive scheduler lock and durable submission-intent record prevent duplicate
parents; ambiguous submission requires accounting reconciliation. Slurm accounting
and semantic case receipts must both complete before another continuation wave.
The persistent supervisor owns sequencing; one run-watch owner monitors
scheduler/resource/artifact/log progress and dispatch bounded incident handling.
Failed/partial bundles remain evidence. Final checkpoint states and native run
metadata are collected and validated before the study is called complete.

## Result and completed handoff

**Reviewed 20 October.** All ten user-accepted endpoints have completed
development, generated held-out and WikiText held-out evaluation. Canary
1332012 and all nine tasks of array 1332167 completed with exit 0:0; the
supervisor published the complete summary at **09:56:04 UTC** and exited.
No additional training updates or device writes were performed, and no remote
checkpoints were deleted.

| Random corruption | Noiseless / noisy updates | Hold development KL | Noiseless development KL | Noisy development KL | Hold WikiText KL | Noiseless WikiText KL | Noisy WikiText KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 2,002 / 1,958 | 0.03873214 | 0.01887781 | 0.02125319 | 0.06863509 | 0.04096365 | 0.04453576 |
| 2% | 1,973 / 1,969 | 0.07188399 | 0.02501169 | 0.02734330 | 0.11739973 | 0.05297006 | 0.05609041 |
| 4% | 1,996 / 1,985 | 0.18466940 | 0.04843297 | 0.04679823 | 0.28291658 | 0.09706706 | 0.09423995 |
| 5% | 1,971 / 1,967 | 0.25837827 | 0.06613509 | 0.06061265 | 0.38860192 | 0.12921428 | 0.11970395 |
| 6% | 1,954 / 1,957 | 0.35289586 | 0.08795493 | 0.07679190 | 0.48104934 | 0.16554122 | 0.14907960 |

**Decision:** both fixed-LR recovery policies benefit all five measured
deployments by the declared criterion: lower development and both held-out KLs
than the matched hold, with nonzero physical state changes. Development KL
reduction is 45.1–78.2%; WikiText KL reduction is 35.1–69.2%. These divide by
the hold KL itself. The noisy policy has lower development and WikiText KL at
4–6%; noiseless is lower at 1–2%. The policies differ in LR, accumulation and
noise, so this does not isolate a beneficial noise effect.

WikiText PPL improves in nine cases. At 1% corruption, noisy recovery slightly
worsens PPL (49.44022 versus 49.41795) despite lower KL. This exception is
reported in the presentation. Noisy recovery uses 747–974 times fewer physical
write attempts than the corresponding noiseless policy; counts are modeled
events rather than measured energy, pulse or latency costs.

Coverage is **10/10 accepted endpoints**, with **0/10 original 2,048-update
training completions**. Unequal actual update counts and the retrospective
user-approved stopping amendment remain explicit. One seed per rate and the
different GPU/PyTorch environment at the existing 3% reference limit generality.
This establishes benefit over these matched HWA/P&V holds, not necessity of
on-chip learning or a controlled cross-architecture advantage. No further
training or tuning is scheduled.

All 20 accepted/baseline checkpoints and the native metadata were collected
locally. Ten accepted checkpoint hashes match their frozen identities and the
validation inputs. Local native inspection verified 25 complete bundles
(15 baseline stages and 10 endpoint evaluations), including all declared
artifact hashes; 20 original interrupted training bundles remain structurally
valid, explicitly incomplete evidence. Evaluation also verified unchanged
held apparent values, device RNG and write counters.

Total charged compute: **35.469 V100 GPU-hours** (34.9867 before evaluation plus
1,737 GPU-seconds of evaluation), within the 60-hour ceiling. Local collection
is approximately 57.16 GiB, within the 100 GiB ceiling. Remote storage did not
require pruning. The prior training watcher stays stopped; the completed
evaluation supervisor has no remaining cases. A brief SSH connection failure
after evaluation delayed collection only; read-only reconnection recovered it.

- [All scores and write costs in the presentation](../../../hwa/docs/meeting_2026_10_22.md#8-drn-recovery-at-other-corruption-levels--accepted-endpoints).
- [Collected native summary](../../results/opt125m-cmo-drn-rates-jz-v1/collected/accepted-endpoint-evaluation/summary.json), SHA-256 `e2b749128a1b6b9d2afc2f49b32638cc9646e3ce69cf61f8d37c1fc1e40f6f1a`.
- [Numeric CSV](../../results/opt125m-cmo-drn-rates-jz-v1/collected/accepted-endpoint-evaluation/presentation.csv) and [criterion/coverage review](../../results/opt125m-cmo-drn-rates-jz-v1/collected/accepted-endpoint-evaluation/review.json).
- [Local integrity receipt](../../results/opt125m-cmo-drn-rates-jz-v1/evidence/collection-verified.json).

## Accepted-horizon execution record (completed)

**20 October: user accepts the saved endpoints.** Stop training at the existing
`attempt-001/run/checkpoints/latest.pt` states. The user said “ok, i am fine with
1950 updates as well”; this accepts the available endpoints, not a claim that
each ran exactly 1,950 updates. This is a retrospective horizon amendment,
independent of held-out evaluation. Preserve every checkpoint. The original
training bundles remain truthful wall-time interruptions.

| Random corruption | Noiseless updates | Noisy updates |
| --- | ---: | ---: |
| 1% | 2,002 | 1,958 |
| 2% | 1,973 | 1,969 |
| 4% | 1,996 | 1,985 |
| 5% | 1,971 | 1,967 |
| 6% | 1,954 | 1,957 |

The old training supervisor is dead and its local run-watch has been stopped;
no training restart or checkpoint pruning is part of this amendment. The
interactive Codifier owns the short evaluation/collection phase. Ten read-only
`ebl validate` runs use the unchanged training configs and exact accepted
checkpoint paths/hashes in `evidence/accepted-endpoints.json`. Their native
results distinguish completed evaluation from incomplete original training.
Evaluation checks that held apparent values, device RNG and write counters do
not change; completed-training checkpoints retain exact development replay.

Evaluation source is frozen at `$WORK/opt125m-cmo-drn-rates-v1/source-v3-eval/`,
manifest SHA-256
`c550825d249bbd813c0e3de378687e90dab1ff578f9fc02f21b86564ccf51080`.
Outputs: `$SCRATCH/opt125m-cmo-drn-rates-v1/retry-v2/accepted-endpoint-evaluation/`.
The same V100 account/partition/QOS/constraint, module, ten CPUs and thread
settings apply. Each evaluation has a 20-minute ceiling, at most five concurrent.
One actual accepted endpoint is the singleton canary and counts toward coverage;
the remaining nine release only after the new static/live gate. Expected
evaluation is a few minutes per case plus queue delay, to be measured by the
canary. New artifacts are small JSON/log bundles, with 1 GiB reserved in total.
Prior allocation cost reconciled directly with `sacct`: **34.9867 GPU-hours**,
including both old canaries and both training waves. The ten evaluation ceilings
add at most 3.3334 GPU-hours, within the unchanged 60-hour study ceiling.

The source also fixes the retired-parent `squeue` failure by falling back to
authoritative `sacct` accounting; other scheduler errors remain fatal.
Old monitoring budgets/history are preserved. Earlier incidents and observations
below are historical, not instructions to restart training or delete checkpoints.

Evaluation canary **1332012**, singleton index 0 (`random01-noiseless`), started
20 October at 09:43:46 UTC (11:43:46 Paris) on V100 node `r6i5n1`. Resource inspection matches the
frozen contract and the native validation manifest/status have been published.
Static checks: shell parsing and Python compilation passed; **38 applicable
tests passed** on the staged Jean Zay source, with the previously documented
unused legacy `ThresholdedWriteAdam` CPU golden excluded (PyTorch rounding
difference; golden unchanged). Both sbatch commands passed `--test-only`.
The executing agent checks completion/resources/artifacts before releasing indices
1–9; expected canary runtime is a few minutes. Checkpoint collection is running
in parallel: ten accepted endpoints plus five deployment and five hold states,
57.14 GiB total, within both local free space and the 100 GiB collection ceiling.

The canary completed successfully in **180 allocated seconds** with native
artifact inspection and checkpoint immutability checks. Gate passed at
09:48 UTC; array **1332167**, indices 1–9 with concurrency five, was released
under the same immutable manifest. Supervisor PID **3743337** on **jean-zay3**
owns accounting/summary publication; the interactive agent retains monitoring
and local collection/review ownership. This measures about three minutes per
evaluation; the remaining two groups should take roughly six minutes plus
queue delay. Full completion requires ten evaluated endpoints and local
bundle/checkpoint verification.

## Historical execution notes

**Bounded controller incident — 19 October, approximately 20:24 UTC.**
The supervisor exited on the declared artifact-reservation guard. Live inspection
on its recorded host `jean-zay2` confirms PID 736505 is absent and no study jobs
remain queued/running; unrelated CIFAR allocations were left untouched. The ten
production cases remain checkpointed, with final coverage still incomplete.
The existing gate matches immutable manifest
`e2cf122858edd655fda2c6120b321cc7e05e95ca1322178ec9e3b1abd919265b`.

The declared audited retention was prepared, but automatic approval review
rejected execution because it would irreversibly delete ten remote production
checkpoints and judged the trusted execution/transfer authorization insufficient
for that deletion. The retention command did not execute: no files were removed,
no retention receipts were created, and no supervisor or jobs were relaunched.
Explicit approval for deletion of the superseded `attempt-000/run/checkpoints/latest.pt`
files after exact successor lineage and both-hash verification is required to
resolve this approval block. Preserve all latest checkpoints and all metadata;
then recheck storage plus the unchanged 80 GiB wave reservation, scheduler
ownership, budget and gate before restarting the same supervisor. Automatic
incident retries are blocked; the existing Python watcher retains passive monitoring.


**Latest check: 19 October, 19:33:33 UTC / 21:33:33 Paris.** All ten cases
have started. Overall progress is 17,659/20,480 updates (86.2%); no final learning
endpoint has completed yet. Five cases have checkpointed at 1,958–2,002 updates;
the remaining five are active at 1,544–1,566. First production array 1295964
completed successfully and second array **1305227** is running. All native pauses
match the declared TimeoutError; no unexpected Slurm or training errors are
present. Supervisor heartbeat 19:33:07 UTC and watcher observation 19:30:50 UTC
are fresh. Expected remaining elapsed time is about 1–2 hours, plus queue delay.

At the common 1,536-update evaluation, development KL is:

| Random corruption | Matched hold | Noiseless at 1,536 | Noisy at 1,536 |
| --- | ---: | ---: | ---: |
| 1% | 0.038732 | 0.019309 | 0.021488 |
| 2% | 0.071884 | 0.025907 | 0.027985 |
| 4% | 0.184669 | 0.051345 | 0.049784 |
| 5% | 0.258378 | 0.070804 | 0.065013 |
| 6% | 0.352896 | 0.094377 | 0.083514 |

All ten interim points reduce matched hold KL by 44–76%. Noisy currently leads
noiseless at 4–6%; final held-out results remain pending. These are partial results.
The presentation document uses this common-step table. Evidence:
`results/opt125m-cmo-drn-rates-jz-v1/evidence/progress-20261019T193333Z.json`.

Storage at this check is 141.52 GiB. The unchanged third-wave reservation is
80 GiB, so the declared audited production-checkpoint retention is required
before that wave. Both checkpoints for the first five cases are present; their
successor input manifests identify the exact old checkpoint hashes and their
state receipts identify newer checkpoints at 1,958–2,002. The controller incident
owner must perform the already declared retention when the storage guard pauses
admission, then restart the same supervisor under the single-owner lock. Leave
active/necessary checkpoints and all metadata intact. No source or gate changes.
The following earlier checks are historical.


**Latest check: 19 October, 14:34:25 UTC / 16:34:25 Paris.** Array 1295964
has five healthy running tasks and five waiting on `JobArrayTaskLimit`; no full
case is complete yet. Progress is 694/2048 (1% noiseless), 678/2048 (1% noisy),
679/2048 (2% noiseless), 681/2048 (2% noisy), 694/2048 (4% noiseless). Recent
training steps take 5.80–5.94 seconds. No unexpected native or Slurm errors; both
supervisor and run-watch observations are fresh. First checkpoint boundaries are
about 30 minutes away, followed by admission of the other five cases subject to
queue availability. Continuations remain under the same supervisor.

At the common **640-update interim evaluation**, development KL is:

| Random corruption | Matched hold | Noiseless | Noisy |
| --- | ---: | ---: | ---: |
| 1% | 0.038732 | 0.021007 | 0.023070 |
| 2% | 0.071884 | 0.028875 | 0.031430 |
| 4% | 0.184669 | 0.061540 | Not started |

These are partial development results, not final or held-out endpoints. The
reduction versus each matched hold is 40–67% among these five interim cases.
Evidence: `results/opt125m-cmo-drn-rates-jz-v1/evidence/progress-20261019T143425Z.json`.
The following earlier observations are historical.


**Production released at 13:19:19 UTC / 15:19:19 Paris on 19 October.**
Parent **1295964**, indices 0–9, concurrency five. Tasks 0–4 are RUNNING;
5–9 are PENDING with `JobArrayTaskLimit`. Native deployment manifests are
published for rates 1%, 2%, 4%; the corresponding noisy tasks wait on the shared
baseline locks. Account, partition, QOS, V100-32G constraint, one GPU per task,
ten CPUs and two-hour limit match the frozen contract. Exactly one production
parent exists. Supervisor PID 736505 has a fresh heartbeat and holds the
exclusive lock. Local watcher PID 1525087 is alive; its production inventory
has no probe errors or unexpected native errors. The full gate is collected at
`results/opt125m-cmo-drn-rates-jz-v1/evidence/gate.json`. Expected completion is
8–10 hours of elapsed execution at concurrency five, plus queue waiting.
The following earlier submission observations are retained as history.


The 22 exact configs and Slurm worker/supervisor are prepared. The first live
canary, Slurm **1291902**, failed after 12 seconds on `r7i2n2` before any native
scientific bundle was created. Slurm executes a spool copy of its batch script;
the original wrapper incorrectly resolved the source from that copy’s location.
The supervisor blocked production. The original `source-v1/` and failed output
ledger/logs are preserved.

The wrapper now resolves the source from its explicit absolute manifest argument.
An additional regression test executes a spool copy and verifies the intended
working directory. All 18 local Slurm transport tests pass. The corrected,
immutable candidate is `source-v2/`; it uses a fresh `retry-v2/` output directory.
The remaining GPU allowance is 215,988 seconds, accounting for the first failure.
Local manifest: `results/opt125m-cmo-drn-rates-jz-v1/source-v2/launch/manifest.json`,
SHA-256 `e2cf122858edd655fda2c6120b321cc7e05e95ca1322178ec9e3b1abd919265b`.

**Resolved transfer approval:** automatic approval review initially rejected the selected
0.783 GiB HWA checkpoint to Jean Zay, stating that scheduling authorization did
not specifically authorize sending the model payload. It subsequently rejected
the final source/config update for the same egress-authorization reason. No
alternate transfer path was used. The user then explicitly replied **"yes, go on"**
to transferring the selected HWA checkpoint and final source/config bundle to
the isolated Jean Zay study directories. Both approved transfers succeeded;
remote teacher/HWA hashes match their declared identities.

Jean Zay's installed `tests` package initially shadowed the staged test helpers.
A local staged `tests/__init__.py` fixes that import collision; it changes no
scientific code. The original v1 manifest was
`e03e4291d869a5337fecf5dff34c6393c21f693d825a87fdfd1fe7e30bb1beab`.
Its local testing passed 54 tests; on the cluster 53 passed, and the legacy
`test_thresholded_controller_matches_pinned_upstream_golden` has a maximum
3.725290298461914e-9 CPU command difference on PyTorch 2.5.0 AVX2/AVX512
(7.450580596923828e-9 without AVX), versus local PyTorch 2.5.1 AVX2. The original
golden and controller remain unchanged. This reference is for the unused
`ThresholdedWriteAdam`, not either selected bounded writer. The applicable
submission suite therefore explicitly excludes that legacy bit-exact reference;
all direct/bounded writer, native continuation, and Slurm checks remain required.

The corrected candidate passed **54 applicable tests, one deselected**, on Jean
Zay, plus source-digest verification, shell syntax and both `sbatch --test-only`
commands. Canary **1292244** was submitted at 12:17 UTC on 19 October. The
exclusive supervisor is PID **736505** on **jean-zay2**, with a verified initial
heartbeat and durable `supervisor-owner.json`. Its queue state was `PENDING`
(reason `Priority`); production has not yet been released. Inspect its native
bundles, Slurm accounting and logs, record measured V100 cost and issue `gate.json`
only after all checks pass. Then release the ten cases under the already active exclusive supervisor and
run-watch owner. At this handoff the canary remains queued and production is
prepared but **not yet submitted**; neither scientific completion nor a V100
runtime estimate is claimed.

### Live check — 19 October, 12:50:58 UTC

Canary `1292244_0` is running on V100 node `r6i7n7` under the declared
account/partition/QOS/constraint, with 29m12s elapsed. The noiseless 6% smoke
completed 32/32 updates, six native attempts and held-out evaluation. The noisy
smoke reached 20/32 updates with positive physical writes and is continuing
from its fourth planned checkpoint boundary. Both laws show about 5.95 seconds
per training update; this excludes evaluation, checkpointing and process setup.
Slurm stderr contains module loading only. The supervisor heartbeat is fresh,
and run-watch observed actual native artifact progress at 12:50:08 UTC without
an incident. Production remains unsubmitted pending the completed canary review.
This supersedes the earlier queued-only observation; it is partial readiness
coverage, not completion of the ten-case study.

### Production admission and retention — declared before production submission

The primary agent is completing production admission after finding that the
canary incident agent interpreted its scope as validation only. That left the
controller `awaiting_gate` without an actionable watch incident. Canary validation
was successful; the full ten-case release remains authorized.

Measured mean update times are 6.242 s noiseless and 6.214 s noisy, including
canary restart warmups; steady medians are about 5.95 s. Use 6.25 s/update plus
1,500 seconds per case for deployment/paired waiting, evaluation, checkpointing,
process startup and inspections: approximately 39.7 production GPU-hours, or
7.9 hours at concurrency five. A conservative 8 s/update estimate plus the same
overhead is 49.7 GPU-hours, or 9.9 hours. Queue waits are additional. Three
production allocations per case are expected, within the four-wave ceiling.
The 1,200-second gap between inner boundary and allocation limit covers measured
setup/save/validation overhead. Peak allocated tensor memory is 5.97 GiB
noiseless and 6.39 GiB noisy on 32-GiB V100s.

Before launching production, declare its retention as **latest necessary resume
state plus completed final state**, with all native metadata/logs/metrics and
explicit removal receipts retained. The existing canary checkpoints remain intact.
For production only, an older `attempt-NNN/run/checkpoints/latest.pt` may be
removed after its immediate successor demonstrably loaded that exact checkpoint
and published a verified later full-state checkpoint or a validated final state.
Require matching config/source, input path/hash, positive later step and hash
verification of both the superseded and retained checkpoints. Write an atomic
retention receipt with paths, hashes, sizes, successor evidence and reason before
unlinking only the named superseded file; record completion afterward. Never
remove the newest unfinished checkpoint, final/deployed states, unmatched files,
or original manifests. Failed attempts with unrecognized errors are excluded.
This is storage maintenance, not a change to scientific runner checkpoint behavior.

The fixed supervisor reserves 8 GiB per remaining case before every wave.
Current canary/baseline storage is 47.79 GiB; preserving every production intermediate
would fit in about 170 GiB, but the conservative third-wave reservation could
exceed 200 GiB. Before continuing that wave, prune only eligible superseded
production checkpoints under the policy above. If the guard stops the supervisor,
the controller incident owner must perform the audited retention, verify remaining
storage plus the unchanged reservation, and restart the **same** frozen supervisor
under its exclusive lock. Do not raise the budget, modify the manifest/source, or
rerun science. Original partial bundles and retained resume/final evidence remain
available. Full collection still excludes superseded continuation payloads.

## Canary review and production release handoff

When the canary completes, the monitoring owner must finish this review and
release the already assigned cases; no further permission is needed. Use the
immutable v2 manifest and output root above. Preserve the static receipt,
Slurm logs and all failed/partial/native bundles. Do not change the frozen source.

1. Require `canary/complete.json` with the current manifest digest, both laws
   complete, and more than one native attempt for each law. Check every partial
   status is the exact declared wall-boundary TimeoutError with a positive saved
   step; final native inspections and held-out results must be valid. Confirm
   nonzero writes/state changes for both laws. A 32-update smoke need not improve
   every metric; its purpose is readiness, not scientific selection.
2. From the frozen source run `python3 launch/verify_canary.py --job-id 1292244
   --expected-account umg@v100 --expected-partition gpu_p13 --expected-qos
   qos_gpu-dev --expected-constraint v100-32g --expected-tasks 1 --require-file
   OUTPUT/canary/complete.json --receipt OUTPUT/canary-gate.json`. The result
   must be `gate_passed: true`, with every allocation `COMPLETED`, exit `0:0`.
   Review the Slurm and native logs for unexpected errors; planned native
   TimeoutErrors are expected. Verify source hashes again.
3. Read `training_step_seconds`, evaluation/save timing and peak memory from
   both canary laws. Project 2,048 updates per case, validation every 128 updates,
   initialization/checkpoint/held-out overhead and continuation count. Record
   expected GPU-hours and wall time at concurrency five, separately from queue
   waiting. Require expected total within the remaining 215,988 GPU-second
   allowance and at most four production waves. Do not silently enlarge limits.
4. Only after these checks, write `OUTPUT/gate.json` containing the passed
   verifier receipt plus `manifest_sha256`, the static receipt hash, log-review
   evidence and throughput/cost estimate. Run `python3 -m workflow.slurm_recovery
   release --manifest ABSOLUTE_MANIFEST`; this submits the ten cases once and the
   exclusive supervisor sequences unchanged continuations. Record the parent ID.
5. Observe the actual first production native artifact, matched baseline receipts
   and allocation contract. Update this note and local `readiness.json` with
   evidence and time estimate. The same run-watch configuration already covers
   all production cases and the controller; do not start a second owner.

If the canary fails, preserve it, diagnose the exact operational cause and account
its GPU time. Source/config/wrapper/resource changes require a fresh immutable
candidate and new static/live gate. Never release production after a failed gate.
The first v1 wrapper failure is already deducted from the v2 budget. A repeat of
that issue or an unrecognized scientific failure requires diagnosis, not blind
resubmission. A stale PID on a different Jean Zay login frontend does not establish
that the original supervisor is dead; check its recorded host or shared heartbeat.

## Monitoring and collection procedure

### Bounded canary validation — 19 October, 13:02 UTC

Canary `1292244_0` completed on `r6i7n7`, exit `0:0`, with **2,153
allocated GPU-seconds** including startup and teardown. Together with the charged
12-second v1 failure, this leaves **213,835 GPU-seconds** of the original ceiling
before production. The immutable v2 source verification passed again; the
completion receipt has the expected manifest hash. `canary-gate.json` now records
the passed scheduler-contract verifier (13:01:01 UTC).

Both laws completed exactly steps 1–32 across six attempts. The five partial
attempts per law have only the declared wall-boundary TimeoutError and positive
saved steps: noiseless 4/11/16/23/28, noisy 4/10/16/20/26. Existing native full-artifact-hash
inspections passed for deployment, hold, baseline held-out, both final states and
both final held-out evaluations. Collected final result hashes match those
inspection receipts. State-changing writes are 3,189,313,905 noiseless and
14,628,120 noisy. Both held-out evaluations scored 97,920 generated and 261,120
WikiText tokens; development replay scored 65,280. Slurm stderr contains module
loading only. This validates smoke execution and continuation, not the full study's
scientific outcome.

Small canary metadata/logs and the scheduler verifier receipt are collected under
the local evidence root's `canary/` and `canary-gate.json`; large checkpoints remain
preserved remotely for later collection. Training steps are approximately six
seconds, final development evaluation 12.7 seconds, checkpoint saves about
7.6 seconds noiseless / 8.4 seconds noisy, and held-out native evaluation about
83 seconds, excluding its process setup.

This incident was scoped solely to `drn-rates-canary-v2`, with no `pending_work`.
No production release or `gate.json` was issued. At that historical handoff, the controller's next scoped
decision still needed to finish the throughput/overhead projection against the remaining
budget and four-wave limit, then perform the existing gated release. Production
and controller monitoring remain owned by the existing Python watcher.

The active local watch config is
`campaigns/pilots/20261019-opt125m-cmo-drn-corruption-levels.watch.json`.
It was armed with bounded dispatch as local PID **1525087**. The detached
process and first observation at **2026-10-19T12:20:04.961520+00:00** were verified: all twelve
entries were queued, with no probe errors. No numerical bundle is expected until
Slurm allocates the canary. The local host must remain awake and connected;
reboot persistence and external notifications are not configured.

Its custom read-only adapter is `workflow.slurm_recovery_watch`; seven adapter
tests pass, including the corrected controller handoff: a completed canary with
a waiting gate now raises an actionable controller incident instead of remaining
quietly queued. This local observation change does not modify frozen remote code. It includes the singleton canary, all ten production cases and the
supervisor. Prepared production cases remain queued for review without claiming
to have Slurm allocations. It checks every case separately through live Slurm allocation,
accounting, native artifact progress and errors, and also checks the sequencing
supervisor's heartbeat. Poll every five minutes, with a 45-minute case stall
limit and a three-minute supervisor heartbeat allowance. Planned inner-runner
TimeoutErrors with verified continuation checkpoints are not scientific failures.
Never infer completion from a parent exit alone or silently relaunch a failed case.

The immutable remote manifest is
`/lustre/fswork/projects/rech/umg/ucy17uy/opt125m-cmo-drn-rates-v1/source-v2/launch/manifest.json`.
From that source directory, `python -m workflow.slurm_recovery probe --manifest PATH`
reads scheduler and semantic progress. `supervise --manifest PATH` owns the single
scheduler lock and may submit only unchanged, gated continuation waves. `release`
requires the completed live gate and authorizes the ten prepared production cases.
An uncertain `submitting` receipt must be reconciled against Slurm; never erase it
and submit again. Any launcher/source/resource change requires a fresh canary.

After terminal coverage, copy remote outputs to the local study's `evidence/`,
preserving native directory layout, final/deployed checkpoints and all metadata.
Exclude `checkpoints/latest.pt` from local copy-back to stay within 100 GiB;
the newest necessary unfinished checkpoint remains on Jean Zay; superseded
production payloads follow the declared audited retention policy above. All canary
checkpoints and all failed/partial metadata remain preserved. Verify
every completed native bundle locally with `ebl runs inspect --verify-artifacts
--require-complete`, reconcile ten learning cases and five holds plus their held-out
evaluations, record per-rate results/interpretation here, and add the reviewed
results to `/home/filip/analog-research/hwa/docs/meeting_2026_10_22.md` as the user
requested. Source/data transfer and completion collection for this exact study
are authorized; no new fault kinds, learning rates or extra arrays are authorized.

### Bounded case incident — 19 October, 20:35 UTC dispatch

Scoped to `drn-rates-random01-noiseless`; no pending cases were assigned.
SSH succeeds, but the frozen probe raises `CalledProcessError` querying
`squeue -j 1305227`. Read-only accounting confirms all ten tasks of that
parent are `COMPLETED`, exit `0:0`; the live user queue contains no rates
study allocation. The assigned case remains `checkpointed` at update
2,002/2,048 in `attempt-001`, with resume SHA-256
`5b6abaac96ea0204193f1e6bb4273871f6c045e0d98e26eca871657300a53ef7`.
Checkpoint payload integrity and final artifacts were not validated in this
incident; scheduler completion is not scientific completion. No remote state,
source, queue or budget was changed. Automatic case recovery is blocked:
the controller owner must reconcile the retired parent and restore its
probe/continuation handoff under the existing gate and retention policy.
This case agent did not restart the study-wide supervisor or submit a parent;
those actions could affect cases outside this incident scope. The existing
Python watcher retains future checks.

### Bounded case incident — 19 October, 20:40 UTC dispatch

Scoped to `drn-rates-random01-noisy`; no pending cases were assigned.
A fresh read-only SSH check succeeds. `squeue -j 1305227` returns
`Invalid job id specified`, while `sacct` reports its ten tasks completed
with exit `0:0`; task `1305227_1` consumed 6,101 allocated seconds.
The live user queue contains no rates-study allocation. This is a retired-parent
probe failure, not evidence of an unreachable host. The assigned case receipt
remains `checkpointed` at 1,958/2,048 in `attempt-001`, with recorded resume
SHA-256 `1ad86e5ebcc3918a6e362f06e513a5b6771c7c7cda710b943d39ab53ea9ce96e`.
The checkpoint payload and final coverage were not validated in this incident.
Automatic case recovery is blocked pending the controller owner's reconciliation
of the retired parent and restoration of the existing continuation handoff.
Restarting the study-wide supervisor or changing its shared scheduler receipts
could affect cases outside this assignment. No remote source, state, queue,
scientific settings or budget was changed; no submission was made. The Python
watcher retains passive monitoring ownership.
### Bounded case incident — 19 October, 20:45 UTC dispatch

Scoped to `drn-rates-random02-noiseless`, with no pending work. Fresh read-only
SSH succeeds; `squeue -j 1305227` fails with `Invalid job id specified`.
Accounting reports `1305227_2` completed, exit `0:0`, after 6,100 allocated
seconds on `r6i7n0`; the live user queue contains no rates-study allocation.
The assigned case remains `checkpointed` at 1,973/2,048 updates in `attempt-001`,
with recorded resume SHA-256
`f3473acb9f68765d4436129243de51b25b72deac96dc752daa97443baa66d259`.
Checkpoint payload integrity and final coverage were not validated here.
Automatic case recovery is blocked pending controller-owner reconciliation of
the retired scheduler parent and restoration of the existing continuation handoff.
Repairing shared scheduler receipts or restarting the study-wide supervisor
could affect cases outside this assignment. No remote state, source, queue or
budget was changed; no submission was made. The Python watcher retains passive
monitoring ownership. This is a retired-parent probe error, not an unreachable host.

### Bounded case incident — 19 October, 20:50 UTC dispatch

Scoped to `drn-rates-random02-noisy`, with no pending work. Fresh read-only
SSH succeeds; the frozen probe's parent query `squeue -j 1305227` returns
`Invalid job id specified`. Accounting reports `1305227_3` completed, exit
`0:0`, after 6,101 allocated seconds on `r7i1n1`; the live user queue contains
no rates-study allocation. The assigned case remains `checkpointed` at
1,969/2,048 updates in `attempt-001`, with recorded resume SHA-256
`8ecc5c1c76f4609d02158eda069de612d5ca2c5cdf15b664596e30144a2a028d`.
Checkpoint payload integrity and final coverage were not validated here.
Automatic case recovery is blocked pending controller-owner reconciliation of
the retired scheduler parent and restoration of the existing continuation handoff.
Repairing shared scheduler receipts or restarting the study-wide supervisor
could affect cases outside this assignment. No remote state, source, queue or
budget was changed; no submission was made. The Python watcher retains passive
monitoring ownership. This is a retired-parent probe error, not an unreachable host.

### Bounded case incident — 19 October, 20:55 UTC dispatch

Scoped to `drn-rates-random04-noiseless`, with no pending work. Fresh read-only
SSH succeeds; `squeue -j 1305227` returns `Invalid job id specified`.
Accounting reports `1305227_4` completed, exit `0:0`, after 6,100 allocated
seconds on `r7i1n1`; the live user queue contains no rates-study allocation.
The assigned case remains `checkpointed` at 1,996/2,048 updates in `attempt-001`,
with recorded resume SHA-256
`de3f7007187e046bcc875c844701512ffa1d3e96c46f17dcece794106db7c45c`.
Checkpoint payload integrity and final coverage were not validated here.
Automatic case recovery is blocked pending controller-owner reconciliation of
the retired scheduler parent and restoration of the existing continuation handoff,
including the previously recorded storage-retention approval block. Repairing
shared scheduler receipts or restarting the study-wide supervisor could affect
cases outside this assignment. No remote state, source, queue or budget was
changed; no submission was made. The Python watcher retains passive monitoring
ownership. This is a retired-parent probe error, not an unreachable host.

### Bounded case incident — 19 October, 21:00 UTC dispatch

Scoped to `drn-rates-random04-noisy`, with no pending work. Fresh read-only
SSH succeeds; `squeue -j 1305227` returns `Invalid job id specified`.
Accounting reports `1305227_5` completed, exit `0:0`, after 6,106 allocated
seconds on `r6i3n8`; the live user queue contains no rates-study allocation.
The assigned case remains `checkpointed` at 1,985/2,048 updates in `attempt-001`,
with recorded resume SHA-256
`247ab0e673781d917ca1b9afcae1fb251f8cc4efa57eb72a31f0ce8fa2a5c094`.
Its state receipt records job ID `1307948`; the accounting above describes the
retired parent that breaks the probe, not that latest recorded job.
Checkpoint payload integrity and final coverage were not validated here.
Automatic case recovery is blocked pending controller-owner reconciliation of
the retired scheduler parent and restoration of the existing continuation handoff.
Repairing shared scheduler receipts or restarting the study-wide supervisor could
affect cases outside this assignment. No remote state, source, queue or budget was
changed; no submission was made. The Python watcher retains passive monitoring
ownership. This is a retired-parent probe error, not an unreachable host.

### Bounded case incident — 19 October, 21:05 UTC dispatch

Scoped to `drn-rates-random05-noiseless`, with no pending work. Fresh read-only
SSH succeeds; `squeue -j 1305227` returns `Invalid job id specified`.
The live user queue contains no rates-study allocation. Accounting for both
`1305227_6` and the case receipt's raw job ID `1307949` resolves to task
`1305227_6`, completed with exit `0:0`, 6,106 allocated seconds on `r6i4n2`.
The case remains `checkpointed` at 1,971/2,048 updates in `attempt-001`, with
recorded resume SHA-256
`cd5f74d481e26c6a8e8c6e977d142b836aad1af6a09b66e3ca65abec200faa66`.
Checkpoint payload integrity and final coverage were not validated here.
Automatic case recovery is blocked pending controller-owner reconciliation of
the retired scheduler parent and restoration of the existing continuation handoff.
Repairing shared scheduler receipts or restarting the study-wide supervisor could
affect cases outside this assignment. No remote state, source, queue or budget
was changed; no submission was made. The Python watcher retains passive
monitoring ownership. This is a retired-parent probe error, not an unreachable host.

### Bounded case incident — 19 October, 21:11 UTC dispatch

Scoped to `drn-rates-random05-noisy`, with no pending work. Fresh read-only
SSH succeeds; `squeue -j 1305227` returns `Invalid job id specified`.
The live user queue contains no rates-study allocation. Accounting resolves the
case receipt's raw job ID `1307950` to `1305227_7`, completed with exit `0:0`,
6,109 allocated seconds on `r6i4n2`. The case remains `checkpointed` at
1,967/2,048 updates in `attempt-001`, with recorded resume SHA-256
`da081211bdce9aff0437952123938ef64da35d630b99d8b53d82d4a6aa57384d`.
Checkpoint payload integrity and final coverage were not validated here.
Automatic case recovery is blocked pending controller-owner reconciliation of
the retired scheduler parent and restoration of the existing continuation handoff.
Repairing shared scheduler receipts or restarting the study-wide supervisor could
affect cases outside this assignment. No remote state, source, queue or budget
was changed; no submission was made. The Python watcher retains passive
monitoring ownership. This is a retired-parent probe error, not an unreachable host.

### Bounded case incident — 19 October, 21:16 UTC dispatch

Scoped to `drn-rates-random06-noiseless`, with no pending work. Fresh read-only
SSH succeeds; `squeue -j 1305227` returns `Invalid job id specified`.
The live user queue contains no rates-study allocation. Accounting resolves the
case receipt's raw job ID `1307951` to `1305227_8`, completed with exit `0:0`,
6,106 allocated seconds on `r6i4n2`. The case remains `checkpointed` at
1,954/2,048 updates in `attempt-001`, with recorded resume SHA-256
`a584a696eaec25b948229b693a20c20dccd53c636fa51ffdfc0e0240d4570cd3`.
Checkpoint payload integrity and final coverage were not validated here.
Automatic case recovery is blocked pending controller-owner reconciliation of
the retired scheduler parent and restoration of the existing continuation handoff.
Repairing shared scheduler receipts or restarting the study-wide supervisor could
affect cases outside this assignment. No remote state, source, queue or budget
was changed; no submission was made. The Python watcher retains passive
monitoring ownership. This is a retired-parent probe error, not an unreachable host.
