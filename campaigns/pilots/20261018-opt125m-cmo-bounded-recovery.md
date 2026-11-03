# Bounded recovery from HWA + P&V at 3% corruption

## Question and assignment

Does appropriately tuned conductance learning improve the compensated HWA
deployment, and how do noiseless direct writes and noisy 2-rho-gated writes
behave? The user approved implementation and execution of the planned pilot,
using two RTX 5090s, then amended noisy recovery to require a minimum 2-rho
command before writing. Rho is the standard deviation of the device write noise.

The first stage tunes only at 3% corruption. Review its evidence before any
replication or the proposed 1%, 2%, 3%, 4%, 5%, 6% series. No jobs from the
older paused pilot are resumed or advanced by this assignment.

## Frozen computation

- Six OPT-125M perfect-diode DRNs, original topology and four settling/four
  gradient sweeps, amplification A=5/B=0.1, effective batch 32, microbatch 1.
- Frozen teacher and digital parameters, including gains, scales and biases.
  Optimize teacher KL using gradients of the held apparent conductances.
- HWA source: `results/opt125m-cmo-drn-hwa-first-v1/execution/runs/full-std3/run/checkpoints/best.pt`,
  SHA-256 `dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9`.
- Canonical device state: `results/opt125m-cmo-drn-hwa-first-v1/execution/runs/std3-deploy-random/run/checkpoints/deployed.pt`,
  SHA-256 `47c3f5b79ed1002c0bb1d5c9dc39c96ef099a09f1ee056b2651697d9d36074f7`.
  Assignment 271002, random 3% permanent defects, after initial P&V and partner
  compensation. Keep the original programming errors, fault map and saved RNG.
- Teacher SHA-256 `2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9`.
  Data identities and ordering remain those of `examples/opt_cmo_drn/recover.json`.
- Adam betas (.9,.999), epsilon 1e-8, constant LR in normalized conductance
  coordinates; no weight decay, warm-up or device-write cap.
- Before a write, form `G_target = clip(G_held + HALF_RANGE_US * command,
  9, 88.199997)` in microsiemens. No target projection occurs merely on loading
  a checkpoint. Unselected cells and permanent defects retain their exact state.
- Noiseless: apply every nonzero Adam command, skipping projected no-ops;
  healthy written cells reach the bounded target exactly.
- Noisy amendment: accumulate signed Adam commands. Compute
  `rho(G_target) = 0.0112185391 + 0.01129027418 * G_target` in microsiemens,
  using the bounded proposed target before the random write. Issue when
  `abs(HALF_RANGE_US * accumulated_command) >= 2 * rho(G_target)`.
  This gate applies to the accumulated command before target projection;
  a boundary write can therefore have a smaller projected change. Saturated
  zero-change targets are skipped. Consume issued commands even for stuck
  devices and projected no-ops; preserve unissued signed residuals without a
  final flush. This follows HWA's accumulation/consumption order, with the
  user's per-write rho replacing the old fixed nominal-midrange reference.
- Noisy healthy endpoints are `G_target + rho(G_target) * normal(0,1)`;
  there is no post-noise clipping. Negative/nonfinite physical states remain
  explicit failures. Both arms advance the same full-array write RNG stream
  at each update, including when no cells are written. No inference noise.

CMO is a hardware-derived fitted-model control, not measured-trace replay.
Noiseless writes are an idealized control. Since only the noisy arm uses the
2-rho gate, their difference includes both noise and update gating.

## Cases, selection and interpretation

Twelve exact configs under `examples/opt_cmo_drn/bounded_recovery/`:
`{noiseless,noisy}-lr{1e-5,3e-5,1e-4,3e-4,1e-3,3e-3}.json`.
Every candidate restarts from the same deployment with fresh Adam and receives
2,048 updates (16,777,216 input-token presentations). The initial, first-update,
128-update intervals and final full-development scores are reported. First
physical-write scores are additionally recorded when they occur later.

Select one LR per arm by lowest fixed-final development KL; exact ties prefer
the smaller LR. No best-intermediate selection. Require complete, finite
candidate coverage before declaring a winner. Mark boundary winners; do not
expand the grid automatically. Preserve failed/partial evidence.

Re-evaluate the unchanged deployment as the zero-write baseline. After locking
winners, evaluate their final states and the baseline on the pinned generated
and WikiText cohorts. Report KL, cross-entropy, perplexity, write counts,
stuck attempts, target clipping/no-ops, conductance statistics and trajectories.
Beneficial recovery requires improvement over hold on final development and
both held-out KLs, with nonzero state-changing writes. A negative or incomplete
result is valid evidence. One assignment does not establish reproducibility or
the necessity of on-chip training. Older digital-calibration results are
context only, not a newly matched control in this pilot.

## Resources, readiness and evidence

Use only the RTX 5090s on loulou and riri, one job per GPU; preserve unrelated
work. Total allowance is 24 GPU-hours including readiness, validation and
understood retries, and 100 GiB of new aggregate local/remote artifacts.
Inputs are shared by content hash. Keep final physical states and metrics;
keep full Adam/residual/RNG continuation in unfinished checkpoints. Completed
runs may remove redundant latest checkpoints with the native retention receipt.
First-write diagnostics sample at most 64 fixed positions per matrix.

Two full-size two-update readiness runs use the same 3% state, one per noise
arm at LR .003. Both must pass native artifact validation. Measure update,
full-development evaluation, full-resume save and final-save costs. Admit all
twelve candidates only if the measured projection plus 20% runtime headroom
fits each 6,750-second queue allowance and the aggregate budget. Each full
runner checkpoints and exits at 6,600 seconds; no horizon or batch reduction
is an operational retry. Failed readiness requires diagnosis/review.

Numerical work uses `python -m ebl train/validate` exclusively. Existing
`workflow.queue_execution.Execution` provides staging, queue submission and
collection; `workflow.bounded_recovery` adds the pilot's frozen cases, cost
gate and final selection. The shared queue owns process/resource/artifact
monitoring with bounded incident dispatch. Completion hooks collect native
bundles, verify hashes, admit eligible declared cases and record the frozen
comparison. They do not expand the study.

Output root: `results/opt125m-cmo-bounded-recovery-v1/` (ignored, backed by
`/tmp/filip-opt125m-cmo-bounded-recovery-v1/`). Source/config snapshots are
immutable. The executable settings are `execution/settings.json`; inspect with
`python -m workflow.bounded_recovery report --settings SETTINGS` and
`python ~/.codex/skills/run-watch/scripts/gpu_queue.py status JOB_ID`.
Resume requires the identical config, source/environment, physical state,
Adam moments, accumulated commands, counters and RNG. An understood operational
retry consumes the same allowance; no LR/threshold/noise changes are permitted.

## Implementation checks

CPU checks cover Adam parity, target clipping before noise, target-dependent
2-rho gating and its inclusive boundary, signed accumulation, no final flush,
stuck/untouched cells, saturated no-ops, held-state gradients, frozen digital
parameters, bounded diagnostic size and exact continuation. Existing writer
goldens and legacy recovery behavior are retained. The public-CLI GPU readiness
provides the real lifecycle/throughput check before substantial execution.

Validation before staging: 76 focused CPU tests passed, covering OPT DRN,
recovery, bounded writers, pilot selection/admission, queue transport and tuning.
Both source input hashes were recomputed and matched. Frozen numerical and
controller snapshot `source-v2` contains 76 Python/JSON files (823,535 bytes),
receipt SHA-256 `b92cf948a78f904f2978ca899e7a637a1c7c3f9a7064a1375b52d3116e16786c`.
Both hosts verified every snapshot file; both use Python 3.12.13 and
PyTorch 2.11.0+cu128. Local CPU checks used PyTorch 2.5.1+cu121.

Both public-CLI GPU readiness runs completed, were collected, and passed native
artifact/hash validation. Queue-charged durations were 28.5076 seconds for
noiseless (loulou, attempt `65af5821910f4668a8edc68315142c21`) and 33.3168 seconds
for noisy (riri, attempt `c0113a58c9fc480e923e550f5f0b9128`). They exactly replayed
initial development KL `0.12147519123904846`. Noisy step 1 issued zero writes;
step 2 issued 15,697,800 attempts after accumulating commands across 2-rho.
Noiseless step 1 issued 113,037,239 attempts. These smoke observations validate
operation, not LR quality.

Worst observed training-step durations were 1.9081/1.9346 seconds and CUDA
peaks 6,457,999,872/6,910,984,704 bytes. Including full-development evaluations,
checkpoint saves and overhead, the frozen cost model projects 4,091.86/4,151.15
seconds per full candidate and 51,558.04 seconds (14.32 GPU-hours) for the pilot,
before the required 20% margin. Both per-case and aggregate gates passed.
Expected elapsed training time on two GPUs is roughly 7–9 hours, subject to
queue availability and collection overhead. Shared controller PID 2529587 owns
monitoring and completion callbacks. Full-run admission has begun automatically.

Launch verification at approximately 17:36 UTC: the unchanged hold completed
remotely and entered native collection. Full candidate `noiseless-lr1e-3` on
riri (attempt `18e930c3a4314fa88b2a0faa71c0ccbe`, worker PID 3679238) reached
update 58; `noiseless-lr1e-4` on loulou (attempt
`baf1de35c74d4505a617122cceb3337c`, worker PID 4032727) reached update 5.
Both emitted finite training metrics and declared `held_apparent` forwards.
The shared service heartbeat was fresh, with live worker observations and
completion/recovery hooks armed. Its next observation is due within 60 seconds;
it remains the sole monitoring owner. Some prepared cases are initially deferred
by aggregate storage reservations and are admitted by completion hooks as space
is released. This is execution progress, not a final LR comparison. Remote riri
wall timestamps lag the controller by two hours; use queue elapsed receipts for
budget accounting.

Remote destination is the isolated subdirectory
`/home/filip/analog-research/opt-cmo-drn-first-v1/bounded-recovery-v1/` on each
host. Automatic approval initially rejected riri's source transfer; the user
then explicitly answered **yes** to transferring this pilot's Python/JSON
source/configs and pinned HWA/deployment checkpoints to loulou and riri at this
path. The subsequent transfers succeeded. One unused source-only staging copy
(under 1 MiB) remains at loulou's earlier `opt-cmo-drn-bounded-recovery-v1/`
destination; it owns no jobs or checkpoints. The old study remains paused.

## Microbatch throughput amendment

The user authorized benchmarking microbatches 4, 8 and 16 against 1 while
retaining effective batch 32. New sweep admissions are held. The loulou
`noiseless-lr1e-4` attempt was cancelled through its queue supervisor to free
one GPU; its partial artifacts and latest checkpoint remain preserved. It
consumed 601.9268 seconds. The riri candidate continues unchanged.

The bounded public-CLI `recovery_benchmark` stage uses the pinned HWA and
canonical 3% deployment. Compare gradients on the first two identical shuffled
32-sequence training batches without writes. Require every tensor's relative
L2 difference <=1e-3 and cosine >=0.999999, and loss difference
<=1e-6 + 1e-4 times the reference loss. Time eight real updates for each of
four microbatches and both noise arms at LR .003, restarting the deployment,
optimizer and write RNG each time; exclude the first two warmup steps.
Keep the 256-token DRN chunk size, checkpointing, solver and precision fixed.
Select a larger microbatch only for >=10% median speedup in both arms and
peak allocated memory <=28 GiB; choose the best summed relative step time.
These are diagnostic trajectories, never LR-selection candidates.

Benchmark budget: at most 600 GPU-seconds within the original 24 GPU-hours;
expected 3–5 minutes on loulou, under 50 MiB of new artifacts. Native output:
`/home/filip/analog-research/opt-cmo-drn-first-v1/bounded-recovery-v1/results/microbatch-benchmark/run`.
The shared queue owns the job; the executing agent owns collection and review.
Only after review will the continuation use a selected microbatch. Changing a
microbatch is not exact checkpoint replay; preserve original configuration and
source identities for all existing attempts.

### Benchmark evidence and decision

The public-CLI benchmark completed and its native bundle passed local artifact
hash validation. Native duration: 123.6144 seconds; queue charge: 129.2766 seconds.
Source snapshot: `source-microbatch-v1`, manifest SHA-256
`705240dd21259062f8df30b98daede73eac95ef35b99a8a7bf46e7956b2d5d25`.
Local bundle: `results/opt125m-cmo-bounded-recovery-v1/execution/runs/microbatch-benchmark/run/`.

| Microbatch | Noiseless seconds/update | Noisy seconds/update | Maximum GPU allocation (GiB) | Worst per-tensor relative L2 difference |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.6335 | 1.6454 | 6.44 | 0 |
| 4 | 1.4245 | 1.4411 | 7.56 | 0.0015313 |
| 8 | 1.3995 | 1.4157 | 9.05 | 0.0014021 |
| 16 | 1.3926 | 1.4074 | 12.02 | 0.0017147 |

All larger microbatches meet the memory and speed criteria but fail the
predeclared per-tensor gradient agreement criterion on at least one batch.
Global gradient differences are 0.000687–0.001025 relative L2, with negligible
loss differences (maximum 4.05e-8). These observations do not demonstrate bad
training; they do reject automatic substitution under this benchmark's rule.
The likely source is floating-point batching/reduction differences propagated
through the DRN's piecewise-linear solver; this explanation is an inference,
not a separately isolated numerical diagnosis. Keep microbatch 1. The fixed
256-token DRN chunks plausibly limit the throughput benefit; microbatch 16
reduces measured step time by only about 15%.

Restore the interrupted candidate from its exact saved configuration, Adam,
physical state and RNG via the public CLI's `--resume`; retain its original
partial bundle under `execution/interrupted-runs/noiseless-lr1e-4/run/`.
Use queue case `resume-noiseless-lr1e-4`, with the original 6,750-second allowance
reduced by its already charged 601.9268 seconds. Its completion maps back to
the original scientific candidate and charges both attempts exactly once.
All numerical training continues with frozen `source-v2`. The local controller
snapshot `source-controller-v3` adds this completion mapping; manifest SHA-256
`437db23bfdcdf5aa97a8f5ab72bef18a1ccf53e47856f6de40fa65d89ebad402`.
The benchmark is included in aggregate budget accounting. Focused checks passed
(50 recovery/config/numerical tests, followed by 28 controller/benchmark tests).

Continuation was prepared from update **256**, checkpoint SHA-256
`5ec241fc4a6b4d612eb000858a65d13d516665ae449ee08d2b85799bebb76aca`,
with exact configuration and PyTorch-version checks. The stopped run's updates
after that checkpoint remain partial evidence and are replayed from update 256.
The remote immutable input is a hard link to the preserved original checkpoint.
Automatic approval rejected deletion of remote checkpoint replicas; no replica
deletion was performed. Instead, the never-started, held `noisy-lr1e-5` queue
entry was withdrawn and re-registered as `deferred-noisy-lr1e-5`, releasing its
reservation until space permits admission. It still maps to the same scientific
candidate and has zero prior GPU charge. Every existing checkpoint is preserved.
Measured noiseless checkpoints are 4,067,769,419 bytes, so an 8 GiB reservation
covers their two-file atomic-save peak; noisy reservations remain 9 GiB.
After continuation admission, observed storage was 30,261,793,357 bytes and
reserved storage 70,866,960,384 bytes, below 100 GiB combined. Charged plus
reserved GPU time was 83,003.91 seconds, below the original 86,400-second cap.

Resume launch verified: loulou attempt `7cc8fa6ef1784fbea33d59958c10e656`
restored update 256 and advanced to update 262 with finite KL, held-apparent
forwards and a 1.632-second training step. Other admitted pending cases were
released; the original riri run remains active. Shared controller PID 2529587
is healthy (fresh 20-second-old observation at verification), with the existing
60-second monitoring cadence and completion hooks. A restricted local process
probe incorrectly reported the controller absent; the unrestricted service
check confirmed it had remained active, so no restart was performed.

## Completed pilot review — 2026-10-19

All 12 fixed-final 2,048-update LR cases and all three held-out validators
completed, were collected, and passed the declared native validation. The
interrupted noiseless case completed through its exact-state continuation;
the deferred noisy case completed under its replacement queue identity. No
scientific candidate is missing or failed. Development replay in each held-out
bundle exactly matches its selection score. No follow-up stage is scheduled:
the requested choice between longer 3% training and the 1–6% series remains open.

| State | Selected LR | Development KL | Generated held-out KL | WikiText held-out KL | WikiText perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged deployment | — | 0.12147519 | 0.11927174 | 0.18565396 | 53.77868 |
| Noiseless recovery | 1e-4 | 0.03496502 | 0.03314241 | 0.07174752 | 50.57618 |
| Noisy 2-rho recovery | 3e-4 | 0.03573988 | 0.03400891 | 0.07273585 | 51.03237 |

Both selected LRs are interior grid points, and both arms satisfy the
predeclared beneficial-recovery rule: lower final development and both held-out
KLs than the matched unchanged deployment, with nonzero physical state changes.
Noiseless/noisy development KL fell by 71.2%/70.6%; WikiText KL fell by
61.4%/60.8%. The noisy winner is close to the noiseless winner on these cohorts.
This demonstrates useful on-chip recovery for this one fixed 3% CMO deployment;
it does not establish replication or that all HWA deployments need recovery.

Both arms had no device-write cap. Selected noiseless/noisy trajectories made
224,152,665,876 / 256,210,235 attempted writes, respectively: approximately
875 times fewer attempts in the noisy arm. This is a comparison of the two
complete update policies; the noisy arm alone accumulates commands until 2-rho,
so the write-count difference must not be attributed to noise alone. Large LR
.003 worsened development KL in both arms (0.12778 noiseless, 0.15594 noisy).

Total charged execution was 43,351.76 GPU-seconds (12.04 GPU-hours), including
readiness, benchmarking, the interrupted attempt and final evaluations, within
the 24 GPU-hour limit. Observed aggregate artifacts were 80,377,539,267 bytes,
with no outstanding storage reservations, within the 100 GiB limit. The shared
queue reports all intended cases complete; the two historical cancelled queue
entries have completed replacements and are retained as operational evidence.

<!-- generated-evidence -->

## Current evidence and handoff

Updated 2026-10-18T23:56:12.447731+00:00. Completed LR cases: 12/12.
Shared queue owns submitted cases; job prefix `opt-cmo-drn-bounded-v1-`. Hosts: loulou, riri.
Native bundles: `/tmp/filip-opt125m-cmo-bounded-recovery-v1/execution/runs/`; execution state: `/tmp/filip-opt125m-cmo-bounded-recovery-v1/execution/execution-state.json`; summary: `/tmp/filip-opt125m-cmo-bounded-recovery-v1/execution/summary.json`.
Queue checks every 60 seconds; semantic-stall limit 45 minutes. Collection and selection use the existing completion hook.
Readiness admitted: True. Blocker: none.

| Case | Final development KL | Writes |
| --- | ---: | ---: |
| noiseless-lr1e-5 | 0.04898594968 | 222045354704 |
| noiseless-lr3e-5 | 0.04080222468 | 223556121410 |
| noiseless-lr1e-4 | 0.03496501503 | 224152665876 |
| noiseless-lr3e-4 | 0.03987184092 | 224453261211 |
| noiseless-lr1e-3 | 0.06877336256 | 224192370004 |
| noiseless-lr3e-3 | 0.1277820725 | 223308997583 |
| noisy-lr1e-5 | 0.06708377948 | 7800531 |
| noisy-lr3e-5 | 0.05154769287 | 22167760 |
| noisy-lr1e-4 | 0.03889130529 | 70161815 |
| noisy-lr3e-4 | 0.0357398799 | 256210235 |
| noisy-lr1e-3 | 0.08254628612 | 2065252671 |
| noisy-lr3e-3 | 0.1559430989 | 7109662261 |

Zero-write development KL: 0.12147519123904846.
```json
{
  "predeclared_rule": {
    "noiseless": {
      "development_improved": true,
      "held_out_improved": {
        "generated": true,
        "wikitext": true
      },
      "boundary_lr": false,
      "beneficial_recovery": true
    },
    "noisy": {
      "development_improved": true,
      "held_out_improved": {
        "generated": true,
        "wikitext": true
      },
      "boundary_lr": false,
      "beneficial_recovery": true
    }
  },
  "scope": "One fixed 3% deployment; CMO fitted-model evidence. Noise and write-gating differ between arms.",
  "next_decision": "Review before replication or the 1\u20136% corruption series; no automatic expansion."
}
```

Pending/failed coverage is not successful recovery. No result is claimed from an unfinished trajectory.
