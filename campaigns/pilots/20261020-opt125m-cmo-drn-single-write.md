# One write per device: noisy versus noise-free DRN recovery at matched LR

## Question and assignment

The fixed-LR series ([3% pilot](20261018-opt125m-cmo-bounded-recovery.md),
[1–6% extension](20261019-opt125m-cmo-drn-corruption-levels.md)) found only a
small difference between noiseless and noisy 2ρ recovery. Those arms differed in
LR (1e-4 / 3e-4), gating and noise, and both allowed unlimited writes per device.

Under a physically motivated budget of **one write attempt per device**, how much
does write noise alone cost recovery, and how much do the gate and cap cost
relative to dense noise-free learning? On 20 October the user chose one shared
LR (3e-4), the crossbar nominal write threshold, one attempt per device, a dense
threshold-free noise-free control, and Jean Zay V100/A100 development queues
with two-hour allocations on project `umg`. **Status: planned. Implementation and
launch have not been approved yet.**

Expected outcome, stated before running: with one write the next gradient can no
longer correct a device's own write noise, so A (noisy) should fall further behind
B (noise-free) than the previous noisy/noiseless gap; B should be worse than C
because the strict nominal gate and cap limit how many devices ever move.

## Frozen comparison

Corruption rates 1%, 2%, 3%, 4%, 5%, 6% random permanent faults, assignment
271002. Within a rate every arm starts from the same saved HWA + P&V +
partner-compensated device state and uses the same data order.

| Arm | Update rule | Gate | Write cap | Write noise |
| --- | --- | --- | --- | --- |
| A. noisy single-write | accumulated bounded Adam | nominal 2σ, `>` | 1 attempt/device | yes |
| B. noise-free single-write | same | same | 1 attempt/device | no |
| C. dense noise-free | existing `direct_adam` | none | none | no |
| Reference: noisy 2ρ | existing `bounded_thresholded_adam` | 2ρ(target), `>=` | none | yes |

- LR **3e-4** constant for A, B and C; Adam betas (.9, .999), epsilon 1e-8.
- **2,048 updates** for every new run; effective batch 32, microbatch 1, 256-token
  sequences, fixed digital parameters, held apparent-state forwards throughout.
- Nominal gate copied from the crossbar writer (`hwa/workflow/onchip.py:216–219`):
  `threshold = 2 · σ(MID_US, 2% acceptance) = 2 · 0.55993 µS = 1.11985 µS`,
  q = 0.0282791. It does not depend on the write-noise factor, so A and B share
  the identical gate. Strict `>` as in the crossbar.
- Targets `clip(G_held + HALF_RANGE_US · command, 9, 88.199997)` µS. Noisy
  endpoints `G_target + ρ(G_target) · N(0,1)`, sampled once and held, no
  post-noise clipping; noise-free endpoints equal the target.
- The cap counts **attempts**, not successful writes: an issued command consumes
  the device's only attempt even if the device is stuck or the projected target
  is a no-op. Afterwards the device never receives another command. Residual
  commands are never flushed. The controller sees no fault map.
- The 3% C endpoint already exists (RTX 5090, 2,048 updates); it gets held-out
  evaluation only. The noisy 2ρ references at 1/2/4/5/6% are the accepted
  1,954–1,985-update V100 endpoints; C therefore has up to 4.6% more updates
  than its reference, a mismatch that favors C.

Typical held conductances are near gmin (3% C endpoint: layer means 13.2–15.1 µS).
There 2ρ ≈ 0.34 µS, so the nominal 1.12 µS gate is about three times stricter
for most devices than the earlier 2ρ gate.

## Inputs

HWA source `results/opt125m-cmo-drn-hwa-first-v1/execution/runs/full-std3/run/checkpoints/best.pt`,
SHA-256 `dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9`.
Teacher SHA-256 `2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9`.
Corpus and cohort identities unchanged from the previous pilots.

| Rate | Starting device state | SHA-256 |
| --- | --- | --- |
| 1% | `$SCRATCH/opt125m-cmo-drn-rates-v1/retry-v2/baselines/01/deploy/run/checkpoints/deployed.pt` | `ec3b59f2dd3c0e4d088e54fe2fbfb69520b2006683694b426a5e5473611c1c61` |
| 2% | `…/baselines/02/deploy/run/checkpoints/deployed.pt` | `69ba9210c4c55c42a50b3e7821c864f0ea4ec2c0bae2c8bc171e404b052cb887` |
| 3% | local `results/opt125m-cmo-drn-hwa-first-v1/execution/runs/std3-deploy-random/run/checkpoints/deployed.pt`, to be staged | `47c3f5b79ed1002c0bb1d5c9dc39c96ef099a09f1ee056b2651697d9d36074f7` |
| 4% | `…/baselines/04/deploy/run/checkpoints/deployed.pt` | `e6c80a5672960a05ea6f04d0bc731bdee8de0aad2308b5bd49b5a160b0d746ca` |
| 5% | `…/baselines/05/deploy/run/checkpoints/deployed.pt` | `1d56330e33a0fc8eda8dcb4156a3cc74cf578dc126ffe55d9fc1d2530a27f845` |
| 6% | `…/baselines/06/deploy/run/checkpoints/deployed.pt` | `3742ad3d7bbe2a15af27e6195ca9fc64e9f61cf63b57a36586cb21165f2c56c6` |

Existing 3% C endpoint: `results/opt125m-cmo-bounded-recovery-v1/execution/runs/noiseless-lr3e-4/run/checkpoints/final.pt`,
SHA-256 `802371306accd2135578a07f7d9682e56994e3235894a0a5b31f5d8d73811b36`, to be staged.

## Measurements and interpretation rule

For every endpoint and its matched hold: development, generated held-out and
WikiText held-out KL and PPL; write attempts, healthy writes, state-changing
writes, stuck attempts; for A/B also the fraction of devices that ever crossed
the gate and the residual command RMS. Fixed final endpoints only, no selection.

- **A vs B** (same GPU type) measures write noise under one write per device.
- **B vs C** measures the gate plus cap without noise (crosses GPU type).
- **C vs reference** is the matched-LR noise-free/noisy comparison with
  unlimited writes, differing in noise and the 2ρ gate.

Report differences per rate; one assignment per rate gives no variance estimate,
so small differences are not claimed as effects. Beneficial recovery still means
lower development and both held-out KLs than the matched hold with nonzero state
changes. These are fitted-model simulations; they do not establish necessity of
on-chip learning.

## Execution plan

**Code** (before any submission):

1. A bounded nominal-gate Adam writer with `max_write_attempts=1`, sharing the
   bounded-target, attempt-consumption and RNG-stream conventions of
   `BoundedThresholdedAdam`, accepting `write_noise_factor` 0 or 1 with an
   unchanged gate.
2. Config validation in `experiments/opt_cmo_drn/config.py` admitting exactly
   this rule with cap 1; existing rules and their contracts unchanged.
3. Tests: cap never exceeded; stuck/no-op attempts consume the cap; identical
   gate with noise on/off; A/B identical up to the first write when no noise is
   drawn; full-state checkpoint/resume restores attempt counts and residuals.
4. An A100 profile for `workflow.slurm_recovery`: separate manifest/supervisor.

**Placement**, keeping each directly compared pair on one GPU type:

| GPU | Account / partition / QOS | Cases |
| --- | --- | --- |
| A100 | `umg@a100` / `gpu_p5` / `qos_gpu_a100-dev`, `-C a100`, 8 CPUs, `module load arch/a100` | A and B at all six rates: 12 runs |
| V100 | `umg@v100` / `gpu_p13` / `qos_gpu-dev`, `-C v100-32g`, 10 CPUs | C at 1, 2, 4, 5, 6%: 5 runs; held-out evaluation of 3% C |

Both QOS have a two-hour maximum and 10 jobs per user each (verified with
`sacctmgr` on 20 October). Module `pytorch-gpu/py3/2.5.0`; TF32 is disabled in
`experiments/opt_cmo_drn/runtime.py`. Inner training boundary 6,000 s per
allocation with native full-state continuation, as in the previous pilot.

**Readiness:** static tests on the staged source; A100 canary with A and B at 3%,
32 updates each including one continuation, measuring throughput and validating
the A100 module/profile; a short V100 C canary on the new source.

**Production:** release both supervisors after the canaries, no scientific gate.
V100 measured 6.25 s/update, so each C case needs three allocations. A100
throughput is unknown; two allocations per case are assumed until the canary.
Every case runs held-out evaluation after training.

| Resource | Expected | Ceiling |
| --- | --- | --- |
| V100 | ~22 GPU-h | 35 GPU-h |
| A100 | ~25 GPU-h, revised after canary | 55 GPU-h |
| Remote storage | ~120 GiB | 200 GiB, previous pilot's audited pruning rule |
| Local collection | < 1 GiB | 5 GiB, metadata/metrics/hashes only |

Wall-clock about 6 h per GPU type plus queue waits; `umg@v100` and `umg@a100`
are both in fairshare over-consumption, so queue time is uncertain. An estimate
outside the ceilings after the canary requires review.

Paths: source/configs `$WORK/opt125m-cmo-drn-single-write-v1/source-v1/`;
outputs `$SCRATCH/opt125m-cmo-drn-single-write-v1/`; local evidence
`results/opt125m-cmo-drn-single-write-v1/`. Checkpoints stay remote.
All numerical stages use `python -m ebl train/validate`.

## Current handoff

**20 October, approved by the user; implementation staged.**

- Writer `CappedNominalWriteAdam` in `training/cmo_direct_adam.py`
  (`update_rule: bounded_capped_nominal_adam`, `max_write_attempts: 1`);
  port `nominal_write_std_q()` in `experiments/opt_cmo_drn/recovery.py`;
  config validation in `experiments/opt_cmo_drn/config.py`. The launcher loads
  `arch/a100` on `gpu_p5` and evaluates an explicitly declared completed run
  (`trained`) without retraining. Tests: `tests/test_opt_cmo_drn_single_write.py`.
- Configs `examples/opt_cmo_drn/single_write/`; the 3% C evaluation uses its
  original `examples/opt_cmo_drn/bounded_recovery/noiseless-lr3e-4.json`.
- Device noise scale is 1.0 in both the 1% and 3% saved banks, so the gate is
  exactly 1.11985 µS.
- Static checks on the staged Jean Zay source (PyTorch 2.5.0, CPU): **73 passed**,
  2 deselected as in the previous pilot (legacy unbounded `ThresholdedWriteAdam`
  golden rounding difference; a test needing a module outside the snapshot).
  Shell syntax checked; both canary `sbatch --test-only` commands accepted.
- Source `/lustre/fswork/projects/rech/umg/ucy17uy/opt125m-cmo-drn-single-write-v1/source-v1`
  (97 verified files). Manifests `launch/manifest-a100.json` SHA-256
  `70f28f4c39825831769958612426d3e4a4d3e4a664fa9fd1ecba7af598de8fd6` and
  `launch/manifest-v100.json` SHA-256
  `c67f621b60718ecf84315cf80738ab66423bbdb15b0801b231ecb5a6283d3733`;
  builder `results/opt125m-cmo-drn-single-write-v1/build_manifests.py`.
- Outputs `/lustre/fsn1/projects/rech/umg/ucy17uy/opt125m-cmo-drn-single-write-v1/{a100,v100}/`,
  each with pre-created baseline receipts pointing at the existing deployments,
  holds and hold held-out runs. 3% inputs staged under `.../inputs/03/` and
  remote hashes verified (`47c3f5b7…`, `802371306…`).
- Manifest limits: A100 concurrency 10, max 3 waves, 55 GPU-h, 130 GiB;
  V100 concurrency 6, max 4 waves, 35 GPU-h, 70 GiB; 5 GiB reserve per case.

**User amendment, 20 October:** add noisy and noise-free arms with **four attempts
per device** (crossbar cap), same gate and LR, all six rates, on V100 after the C
runs. Source-v2 (`.../source-v2`, 89 files, 74 tests passed) admits cap 1 or 4;
manifest `launch/manifest-v100-cap4.json` SHA-256
`1b16f255ba07738be4ca1517b80f3d9652f33532eae7eb90cb96bd556efdd4cb`, output
`.../v100-cap4/`, 12 cases, 75 GPU-h, 140 GiB. Not submitted.

**Canaries, 20 October.** A100 1337576 (A100-SXM4-80GB) and V100 1337579 completed
0:0 in 16:48 and 16:58; every case continued through 4–6 attempts and finished
held-out validation. Median training step: **4.11 s A100**, 5.81 s V100; peak
6.49 GiB; resume checkpoints 4.32 GiB (capped) and 3.80 GiB (dense). In 32
updates the capped arms issued **no writes**: Adam moves a command by at most
about LR = 3e-4 per update, so the earliest possible write is near update 95.
The canary therefore validated setup, resume and evaluation, not a real write.

**Pre-launch concern, production held.** In the existing dense 3% C run at the
same LR, only **1.49%** of devices have a net held change above 1.12 µS after
2,048 updates (0.07% above 2.24 µS, 0.01% above 4.48 µS). A gated device's
accumulated command approximates that net movement (plus commands lost to gmin
clipping in the dense run), so the capped arms will probably write only a few
percent of devices once, and the four-attempt cap would rarely bind. The crossbar
paired this gate with LR 3e-3 (about 9 aligned updates per threshold, versus about
94 here). Awaiting the user's choice before release.

**User decision, 20 October: run the 3% single-write pair first**, then decide on
the other rates, the dense C runs and cap 4 from its measured write coverage and
recovery. Nothing else is released.

- Manifest `launch/manifest-a100-03.json` (source-v1), SHA-256
  `37b872735a48a5c3efafcb242a7a1706bc5531d63be08a17b60b73eeee1e1276`: the two
  3% cases only, 12 GPU-h, at most 3 waves, same output root `.../a100/`, so the
  completed cases count toward the full sweep later. Gate `.../a100/gate.json`
  ties the A100 canary 1337576 to this manifest and records the no-write limitation.
- Released 20 October 14:42 Paris: array **1338950** tasks 0 (`random03-single-noisy`)
  and 1 (`random03-single-noisefree`). Supervisor on login node **jean-zay2**, PID
  1117441, log `.../a100/supervisor-03.log`, heartbeat `.../a100/supervisor.json`.
- Expected: about 1,400 updates in the first allocation, the rest plus held-out
  evaluation in the second; about 3.5 h of compute plus queue.
- Monitor: local poll every 10 minutes via `python -m workflow.slurm_recovery probe
  --manifest launch/manifest-a100-03.json` from the source directory; it stops on
  completion, any failed case or a supervisor heartbeat older than 10 minutes.
  Check first writes near update 95 and the issued-device fraction at each
  128-update development point.

## 3% single-write result (20 October)

Both cases completed 2,048 updates in two A100 allocations (first 1,394 / 1,300
updates, then continuation array 1344133), with held-out evaluation; summary
`.../a100/summary.json`. Total charged 19,471 A100 GPU-seconds (5.4 GPU-h),
including the canary.

| 3% random | Dev KL | Generated KL / PPL | WikiText KL / PPL | Write attempts | Devices written |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hold | 0.12148 | 0.11927 / 7.792 | 0.18565 / 53.779 | 0 | 0% |
| Single-write, noise-free | 0.05500 | 0.05240 / 7.472 | 0.10288 / 52.819 | 29,763,579 | 26.3% |
| Single-write, noisy | 0.05525 | 0.05267 / 7.475 | 0.10332 / 52.928 | 29,794,313 | 26.3% |
| Earlier dense noise-free, 3e-4, unlimited (RTX 5090) | 0.03987 | not evaluated | not evaluated | 224,453,261,211 | — |
| Earlier noisy 2ρ, 3e-4, unlimited (RTX 5090) | 0.03574 | 0.03401 / 7.339 | 0.07274 / 51.032 | 256,210,235 | — |

Writes began at update 90. Development KL rose to 0.2637 at update 128 when the
first 3.2% of devices committed, then fell monotonically; about 1.4% more devices
wrote per 128 updates throughout. Stuck-device attempts were 3.3% of attempts.
My pre-launch coverage estimate of about 1.5% from net dense movement was wrong by
more than tenfold: gated residuals keep accumulating, including commands the
dense run lost to gmin clipping.

**Reading.** Both single-write arms remove 55% of the hold's development KL and
45% of its WikiText KL. Write noise costs only 0.4–0.5% relative KL: the arms
make identical write decisions because the gate ignores noise, and each issued
move is at least 1.12 µS against a write SD of about 0.17 µS at typical held
conductances. The remaining gap to unlimited-write learning mixes the larger
fixed gate with the one-attempt cap; this pair does not separate them. Different
GPUs (A100 versus RTX 5090) also limit the comparison with the earlier references.
One assignment and one write-noise draw; no variance estimate.

## Best-case on-chip learning (E1, E3), user decision 20 October

Question: how far can dense, exact, uncapped, bounded on-chip learning go, and
can it improve even the healthy array? The constant-LR 3% runs were still improving
at 2,048 updates (1e-4: 0.0357 → 0.0350 over the last 256) and constant 3e-4
fluctuated late (0.0385 at 1,792, 0.0399 at 2,048). The user chose E1 and E3
only; the 8,192-update E2 and digital co-training are not part of this decision.

- **E1:** random 3%, dense noise-free `direct_adam`, bounded targets, peak LR 3e-4,
  linear warm-up 102 updates, cosine to zero at 2,048 updates.
- **E3:** the same recipe on the healthy HWA deployment
  (`std3-deploy-none`, SHA-256 `25cd96952589852079bd203ca5914955cce7c0fdb44c7c8a539cafd2c9523b92`;
  hold development / generated / WikiText KL 0.02295 / 0.02209 / 0.04716).
- Compare E1 with constant 1e-4 (0.03496) and 3e-4 (0.03987) at the same budget,
  and E3 with its hold. Fixed final endpoints; held-out generated and WikiText.

Implementation: `DirectWriteAdam` optional `warmup_cosine` schedule; config fields
`recovery.lr_schedule` / `recovery.warmup_updates` (constant configs keep their
old hashes and contracts). Tests `tests/test_opt_cmo_drn_best_case.py`; 79 passed
on staged source-v3 (2 deselected as before). Configs `examples/opt_cmo_drn/best_case/`.
Manifest `.../source-v3/launch/manifest-a100-best.json` SHA-256
`49fa0040dc38cd2fc9ca52e12fb1ef223a5b876fb62b850cfcb74c579c2db905`, output
`.../a100-best/`, 8 A100 GPU-h, 3 waves, 40 GiB. Healthy inputs staged under
`.../inputs/00/` (hash verified). Canary 1347896 (E1 smoke, 32 updates with
continuation) completed 0:0 in 7:42 on A100-SXM4-80GB: 4 attempts, median step
4.10 s, resume checkpoint 3.79 GiB, LR 1.5e-4 at update 1 (2-update smoke warm-up),
dense writes issued, held-out validation complete. Gate `.../a100-best/gate.json`.
Released 20 October 18:06 Paris as array **1348623** (0: `random03-dense-cosine`,
1: `healthy-dense-cosine`); supervisor jean-zay2 PID 404594, log
`.../a100-best/supervisor.log`. Local 10-minute probe watcher as before.

### E1/E3 result (20 October)

Both completed 2,048 updates in two A100 allocations (1348623, continuation
1352470) plus held-out evaluation; `.../a100-best/summary.json`. Charged 18,402
A100 GPU-seconds (5.1 GPU-h) including the canary.

| Case | Dev KL | Generated KL / PPL | WikiText KL / PPL |
| --- | ---: | ---: | ---: |
| Healthy hold | 0.02295 | 0.02209 / 7.230 | 0.04716 / 49.479 |
| **E3** healthy, cosine | **0.01527** | 0.01463 / 7.204 | 0.03449 / 48.848 |
| 3% hold | 0.12148 | 0.11927 / 7.792 | 0.18565 / 53.779 |
| **E1** 3%, cosine | **0.03336** | 0.03155 / 7.319 | 0.06953 / 50.464 |
| 3%, constant 1e-4 (RTX 5090) | 0.03497 | 0.03314 / 7.335 | 0.07175 / 50.576 |
| 3%, constant 3e-4 (RTX 5090) | 0.03987 | not evaluated | not evaluated |

Development KL (E1 / E3): 256: 0.0492 / 0.0220; 1,024: 0.0379 / 0.0177;
1,536: 0.0343 / 0.0158; 1,920: 0.0334 / 0.0153; 2,048: 0.0334 / 0.0153. The
decayed schedule flattens both curves at the end. E1 made 2.22e11 write attempts
(6.4e9 below-gmin target clips, 9.8e9 projected no-ops); E3 2.24e11 (3.7e9, 7.1e9).

**Reading.** The schedule improves the 3% endpoint over the best constant LR by
4.6% development and 3.1% WikiText KL at the same budget; this is one draw on a
different GPU from the constant-LR references. Best-case learning also improves the
healthy array by 33% development and 27% WikiText KL, so the healthy hold is not a
floor. With faults, the best case remains well above its healthy counterpart:
0.0334 versus 0.0153 development KL and 50.46 versus 48.85 WikiText PPL. Relative
to learning on the healthy array, it closes 83% of the 3% hold's development-KL
gap. The healthy DRN result (49.48 → 48.85 WikiText PPL) is close to the crossbar's
healthy noiseless result in exp-022 (49.59 → 48.87), while the crossbar's random-3%
noiseless endpoint is 49.09; in this comparison, permanent faults leave a larger
residual in the DRN. Exp-022 used a different update budget and four-attempt
noisy cap, so this is descriptive.

## Dense noise-free LR 1e-4 to 4,096 updates, all rates (user decision, 20 October)

Question: do more updates close the remaining fault gap for the existing best
constant-LR dense noise-free rule? The user chose: dense every-update writes,
constant LR 1e-4, **4,096 updates**, all six random-fault rates, V100; continue the
existing 1/2/4/5/6% runs rather than restart.

- 1, 2, 4, 5, 6%: continue the accepted unfinished noiseless 1e-4 states of the
  [corruption-level pilot](20261019-opt125m-cmo-drn-corruption-levels.md) from
  updates 2,002 / 1,973 / 1,996 / 1,971 / 1,954 (resume SHA-256 prefixes
  5b6abaac96ea / f3473acb9f68 / de3f7007187e / cd5f74d481e2 / a584a696eaec).
  Those files are read only; new attempts write under `.../v100-longer/`.
- 3%: fresh 4,096-update run from the 3% deployment (the earlier 3% 1e-4 run was
  on an RTX 5090 and kept no optimizer state). Its 2,048-update development point
  is logged but is a V100 replicate, not the RTX 5090 run.
- **Horizon extension rule.** New config field `recovery.extends_updates` (here
  2,048) lets a constant-LR learning config resume states saved under the original
  horizon; only `recovery.updates` may differ, so continuation equals one longer run.
  A fresh start with an extension is refused, and endpoint validation still requires
  the exact config. Implemented in `experiments/opt_cmo_drn/config.py`
  (`resume_config_hashes`) and `recovery.py`; the launcher's `continue_from` case
  field verifies the checkpoint digest and its starting device state. Tests
  `tests/test_opt_cmo_drn_extension.py`.
- Configs `examples/opt_cmo_drn/longer/` (byte-for-byte the original configs plus
  the horizon fields). Source-v4: 82 tests passed (2 deselected as before).
  Manifest `.../source-v4/launch/manifest-v100-longer.json` SHA-256
  `c3e05d8dd22df90f3267c30840d51436e194e88d576db740a4ab28347a5ffb2f`; 42 V100
  GPU-h ceiling (about 30 expected), 6 waves, 130 GiB.
- Readiness: on the login node every continued checkpoint loaded with the resume
  schema, an accepted config hash, PyTorch 2.5.0, matching step and starting device
  state, and the LR 1e-4 writer contract. No GPU canary for this manifest (declared);
  the first wave is checked for correct resumption.
- Released 20 October 22:59 Paris as array **1359190** (indices 0–5 = 1–6%); all six
  started immediately. Supervisor jean-zay3 PID 1422582, log `.../v100-longer/supervisor.log`.
  Expected: continued cases about 3 allocations (~6–7 h), 3% about 5 allocations
  (~10–11 h) plus queue waits.
- Resume verified at 23:07 Paris: each continued case started at its declared step,
  and its development KL at that step equals the accepted endpoint
  (0.01888 / 0.02501 / 0.04843 / 0.06614 / 0.08795). Training steps 5.8–6.0 s on V100.
  Local 10-minute probe watcher (up to 15 h) stops on completion, any failed case
  or a supervisor heartbeat older than 10 minutes.

### 4,096-update result (21 October)

All six completed with held-out evaluation; the supervisor finished at 06:06 Paris
(waves 1359190, 1362386, 1364233, then 1365944 and 1367047 for the fresh 3% run).
Charged **91,898 V100 GPU-seconds (25.5 GPU-h)** of the 42 GPU-h ceiling. Summary
`.../v100-longer/summary.json`. The local watcher was interrupted when the previous
session ended at about 02:30; the remote supervisor completed independently.

| Faults | Hold dev / WikiText KL | ~2,000 updates dev / WikiText KL (PPL) | **4,096 updates** dev / WikiText KL (PPL) | Dev KL change | Noisy 2ρ 3e-4, ~1,960 updates, dev / WikiText KL (PPL) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 0.03873 / 0.06864 | 0.01888 / 0.04096 (49.192) | **0.01731 / 0.03807 (49.066)** | −8.3% | 0.02125 / 0.04454 (49.440) |
| 2% | 0.07188 / 0.11740 | 0.02501 / 0.05297 (49.757) | **0.02264 / 0.04808 (49.355)** | −9.5% | 0.02734 / 0.05609 (49.958) |
| 3% | 0.12148 / 0.18565 | 0.03496 / 0.07175 (50.576)* | **0.03096 / 0.06519 (50.101)** | −11.4% | 0.03574 / 0.07274 (51.032) |
| 4% | 0.18467 / 0.28292 | 0.04843 / 0.09707 (52.346) | **0.04163 / 0.08544 (51.115)** | −14.0% | 0.04680 / 0.09424 (52.008) |
| 5% | 0.25838 / 0.38860 | 0.06614 / 0.12921 (54.076) | **0.05498 / 0.11138 (52.860)** | −16.9% | 0.06061 / 0.11970 (53.643) |
| 6% | 0.35290 / 0.48105 | 0.08795 / 0.16554 (56.301) | **0.07166 / 0.14179 (54.729)** | −18.5% | 0.07679 / 0.14908 (55.395) |

*3% ~2,000-update value: development KL at update 2,048 of the fresh V100 run
(0.03496), which reproduces the earlier RTX 5090 endpoint (0.03497) to within
0.03%; its held-out values are the RTX 5090 endpoint's. The ~2,000-update rows for
other rates are the accepted 1,954–2,002-update endpoints that these runs continued.

Generated held-out KL / PPL at 4,096: 0.01636 / 7.215, 0.02137 / 7.245,
0.02913 / 7.308, 0.03947 / 7.382, 0.05300 / 7.479, 0.06740 / 7.580.
Development KL over the final 256 updates fell by 0.3% (1%), 0.9% (3%) and 1.2% (6%):
slowing but not flat. Write attempts doubled to 4.46–4.54e11 per run.

**Reading.** Doubling the horizon helps every rate, and helps more at higher
corruption (−8% development KL at 1% to −19% at 6%; WikiText KL −7% to −14%).
At 4,096 updates dense noise-free recovery is below the noisy 2ρ endpoints at every
rate, reversing the 4–6% ordering seen at ~2,000 updates, but this compares about
twice as many updates and a different LR, so it is not a matched noise comparison.
At 1% the endpoint (0.0173) approaches the healthy-array best case (E3: 0.0153,
cosine 3e-4, A100); at 3% constant 1e-4 for 4,096 updates (0.0310) beats the
2,048-update cosine run (0.0334). One assignment per rate; the 1/2/4/5/6% runs are
continuations of one trajectory each.

## Naive P&V held-out scores (21 October)

User request: add the naive P&V line to the DRN version of the crossbar meeting
figure. New recovery arm `naive_hold` reinstates each array's raw programmed state
from before partner compensation (the deployment's `raw` evidence, permanent faults
included), makes no update or write, and is scored by the unchanged `ebl validate`
path. Test `tests/test_opt_cmo_drn_naive.py` (end-to-end deploy, naive hold and
validation); 84 tests passed on staged source-v5. Configs `examples/opt_cmo_drn/naive/`
(the hold configs with only the arm changed). Manifest
`.../source-v5/launch/manifest-v100-naive.json` SHA-256
`b278169ead71abf2b2cac36f386652d824438c13372dd7c0c82d870425917298`, array 1375591,
seven tasks completed 0:0 in 3:39–3:46 each on 21 October. Every development KL equals
the deployment's own recorded raw development score (exactly, or within 3e-7 for the
RTX 5090 healthy and 3% deployments evaluated on V100).

| Faults | Naive dev KL | Naive generated KL / PPL | Naive WikiText KL / PPL | After partner compensation, WikiText PPL |
| --- | ---: | ---: | ---: | ---: |
| 0% | 0.02295 | 0.0221 / 7.230 | 0.0472 / 49.48 | 49.48 |
| 1% | 1.63621 | 1.5556 / 38.44 | 1.8300 / 313.2 | 49.42 |
| 2% | 4.17894 | 4.0866 / 586.0 | 3.9305 / 2,526 | 50.88 |
| 3% | 5.18507 | 5.1118 / 1,733 | 4.7569 / 5,765 | 53.78 |
| 4% | 5.59567 | 5.5272 / 2,686 | 5.1425 / 8,638 | 58.09 |
| 5% | 5.61583 | 5.5532 / 2,756 | 5.1971 / 8,994 | 63.62 |
| 6% | 5.77008 | 5.7034 / 3,310 | 5.2508 / 8,887 | 70.89 |

Naive P&V is far more damaging on the DRN than on the crossbar (crossbar gmin/gmax
WikiText PPL 55–206 at 1–6%); partner compensation removes nearly all of it.
Figure: `results/opt125m-cmo-drn-single-write-v1/figures/drn_wikitext_ppl.png`
(script `drn_wikitext_ppl.py`, values `naive_pv.csv` and `noisy_noiseless.csv`).
