# CMO drift recovery — one-shot methods at release (crossbar XB6/XB12, CMO ×{1,4,10})

Shared record: on-chip-learning
`campaigns/cmo-drift-recovery/series/001-one-shot/experiments/exp-001-one-shot-methods.md`
(all scientific choices, user, 18 November 2026), issue
[#67](https://github.com/filip137/on-chip-learning/issues/67) (follow-up of #42). DRN counterpart:
drn `refactor/training-core`. Prior art: [drift token budget](20261116-drift-token-budget.md)
(exp-005), whose runner this extends. Branch `exp/20261118-cmo-drift-recovery`.

**Question.** With CMO devices, the AIHWKit CMO drift law ×M (M = 1, 4, 10), the two-sided
crossbar mapping and reset-absorb writes, how much of the 24-h drift damage does each one-shot
recovery at release remove on XB6 and XB12, against untreated drift (A0)?

**Frozen comparison (transcribed from exp-001; not chosen here).**

| exp-001 item | Lifecycle field |
| --- | --- |
| 1. Mapping, P&V | `fault_tuning.drift.mapping` = two_sided, unit gain, margin 0: G± = 9 + 79.2·max(±q, 0), q per output channel (abs-max scale stays digital). In-job deployment: naive Gaussian-endpoint P&V (2 % CMO fit) of the mapped targets, then the verify-based 5σ pair rewrite at 0 % faults (`fault_rewrite` seed 241016), assignment 271001; t = 0 is release |
| 2. Data | pool 8,192 × 256 `69f79f1e…`, 8-sample targets file `9cba6cf3…` (tensor `51467dd7…`), dev 64 sequences `7411cf50…`; sources XB6 `86522e23…`, XB12 `ed53b6cb…` (unchanged from exp-005) |
| 3. Optimizer, gate | Adam (0.9, 0.999, 1e-8), batch 8, order seed 20261104, 2,400 updates cosine to zero, update k reads at 1.5·k s; gate `write_law.threshold_sigma` 2 at mid-range = 2·σ(48.6 µS) = 1.1198 µS of conductance (0.01414 in q under the 79.2 µS span), accumulate, no cap, `>=` |
| 4. Drift law | `aihwkit_cmo_log_independent.v1`, coefficients as exp-005, clip 5–150 µS, draw seed 391103, M ∈ {1, 4, 10} (one lifecycle each; never pooled) |
| 5. Writes | `write_clock: reset_absorb` |
| 6. Baseline | one in-job deployment per lifecycle; recorded: source ideal KL, clean mapped readout KL, deployed KL at release (A0-twin) |
| 7. Controls | A0, A0-twin (`no_drift`), CAL, CT-nodrift (×4 lifecycle only) |

Arms (LRs reused untuned): A0 `hold_fault_aware`; A0-twin `hold_fault_aware` + `no_drift`;
CAL `calibration` 1e-3; LN `layernorm` 3e-3; CT `child_tuning_d` reserve 0.1, XB6 1e-3 / XB12 3e-3;
L1 `onchip_noisy` 1e-3; LoRA `digital_lora` rank 16 (α = r), XB6 3e-4 / XB12 1e-3; CT-nodrift
`child_tuning_d` + `no_drift` (as CT). Grid 0, 1 min, 15 min, 30 min, 1, 2, 4, 8, 16, 24 h; dev
every 100 updates; 24-h projections of the writing arms at 300/600/1,200 updates (2,400 is the
24-h grid point), as exp-005.

**Decision rule** (exp-001): per model × M cell, the 24-h trapezoid-mean dev KL of each arm
against A0; LR fallback (÷3/×3 on that cell) only if an arm's dev KL at the end of the window is
not below A0's at 1 h, or it diverges. One chip per cell, exploratory.

## What changed (code)

- `methods/drift_tuning.py`: `write_clock: reset_absorb`; optional `mapping` (omitted = the
  standard pair, so every earlier lifecycle digest is unchanged, 70/70 checked); `Mapping`
  (reference / two_sided targets and readout); `drifted()` with per-device write time —
  unwritten devices evaluate exactly the release-clock expression, written ones τ = t − t_w
  with the same Y_t, τ ≤ 0 reads the held value; `DriftPlant` (reads, bounds, the
  reset-absorb write: held ← drifted read + step + σ(G)ξ, t_w ← t for written devices; the
  two-sided sign-side and zero-crossing rule); `GatedDriftAdam` (Adam, signed accumulation,
  fixed conductance gate `|79.2·pending| >= 1.1198 µS`, optional fixed mask, LR 0 never
  writes). `continues` and its learner path are unchanged.
- `hardware/devices/cmo.py`, `hardware/arrays.py`: `program_conductances` (the same endpoint
  law and stream as `program`, arbitrary (2, size) targets).
- `methods/programming/lifecycle.py`: `fault_rewrite` / `detect_stuck_devices` /
  `pair_rewrite_plan` take the mapping's expected targets, span and readout (defaults: the
  standard pair, bit for bit).
- `methods/fault_tuning.py`: `layernorm` (drift block only), arm flag `no_drift`.
- `ebl/schema/lifecycle.py`: drift arms = one drifted hold, at most one drift-off twin and
  learners in {onchip_noisy, child_tuning_d, calibration, layernorm, digital_lora}; `no_drift`
  only on the hold twin and writing learners; `continues` keeps the exp-005 arm set.
- `ebl/runtime/fault_tuning.py`: `deploy_mapped` (two-sided in-job deployment);
  `run_drift_arm` for every arm, all device-facing reads through the plant (Fisher at the
  released state, shared by CT and CT-nodrift); checks: zero-write replay of A0 (or of the twin
  for `no_drift`), every arm's release read equals A0's, digital arms leave the deployment
  unchanged, LayerNorms restored after each arm; per-job `summary.json` `drift.records`.
- `campaigns/fault-recovery-paper/launch/worker.py`: drift-aware row validation (the Jean Zay
  worker otherwise rejects drift rows).
- `campaigns/pilots/20261118-cmo-drift-recovery/`: `make_configs.py` (from the exp-005
  lifecycles), 6 full + 2 canary lifecycles, `run_lab.sh` (lab-host plan → run → inspect →
  validate, refuses a busy GPU).
- Tests: `tests/test_cmo_drift_recovery.py` (15), worker drift validation (8).

## Lifecycles

`campaigns/pilots/20261118-cmo-drift-recovery/lifecycles/`:
`full/cmo-drift-recovery-20261118-{xb6,xb12}-m{1,4,10}.json`,
`canary/cmo-drift-recovery-20261118-{xb6,xb12}-m4-canary.json` (32 updates, projection at 16,
all eight arms). `lifecycle describe`: stages prepare, devices, hwa (frozen source),
fault_tuning; XB6 28,311,552 / XB12 56,623,104 analog weights.

## Readiness

- **Tests.** `tests/test_cmo_drift_recovery.py` 15 pass: forced write (held = drifted read +
  step + σ(G)ξ, τ restarts only for written devices, unwritten devices bit-identical to A0);
  write only at the age of the learner read; τ = 0 reads the held value; `continues` reads equal
  `drifted_weights` and its law round-trips; two-sided targets, clean readout = q (and q·c with
  channel scales), sign-side and zero-crossing writes (noise-free result q + dq; σ(9 µS) on the
  floor write); mapped verify rewrite; gate `>=` exactly at 1.1198 µS (0.01414 q); mask; LR 0
  replays A0 at all 10 ages; drift-off plant reads the release state at every age; lifecycles
  (inputs identical to exp-005 except drift and arms; LRs per model) and rejected arm sets.
  Worker drift validation 8 pass. Neighbouring suites (`test_drift_tuning`, `test_fault_*`,
  `test_workflow_{cmo,fault_rewrite,partner_training,threshold,write_noise,lifecycle,stages,
  runtime,opt,analog_lora,lora}`): the combined run has the same 22 order-dependent failures as
  trunk `dc36fe9`, and every one of those files passes alone. Every existing `fault_tuning`
  lifecycle digest is unchanged (70/70).
- **CLI.** `python -m ebl describe` lists `crossbar_lifecycle.v1`; `ebl lifecycle check` ok for
  all 8 lifecycles; `ebl lifecycle plan` + `campaign run` exercised by the smoke below.
- **Local smoke** (nom-cool-2 RTX 3090, idle but for the desktop; XB6 ×4, 8 updates, all eight
  arms plus forced-write arms Lhot = `onchip_noisy` LR 0.05 and CThot = `child_tuning_d`
  LR 0.05; `--allow-dirty`): campaign complete, `runs inspect --verify-artifacts
  --require-complete` valid, wall 3 min 51 s.

  | Record | Value |
  | --- | ---: |
  | Source ideal-weight KL | 0.04297 |
  | Clean mapped readout KL (max \|readout − q\| 6e-8) | 0.04297 |
  | Deployed KL at release (A0-twin) | 0.04321 |
  | Verify rewrite (5σ, 0 %) | 39 devices (17 G+, 22 G−), 9 clipped |
  | A0 24-h mean / 24-h KL | 0.0569 / 0.0590 |
  | Devices at 5 µS at 24 h (A0) | 26.8 % |

  8-update 24-h means: CAL 0.0580, LN 0.0564, LoRA 0.0564 (737,280 digital parameters;
  CAL and LN 9,216 each); CT and L1 wrote nothing and replay A0 exactly at all ages;
  CT-nodrift wrote nothing and replays the twin (0.04321); every arm's release read equals A0's.
  Lhot: 1.23e8 device writes, 3.1e6 zero-crossing updates, 55 % of devices written; CThot:
  1.26e7 writes, none outside the 2,831,160-weight mask. For context only (different mapping,
  write clock and deployment numerics): exp-005's standard-pair XB6 ×4 smoke had release 0.0455
  and A0 24-h mean 0.1236.

- **`continues` bit for bit.** The exp-005 smoke lifecycle (`drift-token-budget-smoke2-xb6`,
  digest `3a09998b…`, arms A0, L1, Lhot LR 1.0), first run with hwa `61eab77` on this GPU,
  rerun with this branch: every grid KL, projection, training loss, write count (1.91e8) and
  final apparent hash is identical.
- **Timing probe** (XB6 ×4, 200 updates, arms A0, CAL, LN, CT, L1, LoRA; valid bundle). It
  shared the 3090 with a DRN smoke, which roughly doubled every cost against the smoke alone
  (evaluation 1.79 s vs 0.87 s, Fisher 191 s vs 97 s). Shared-GPU rates: digital arms
  (CAL/LN/LoRA) 0.16–0.18 s/update, writing arms (CT/L1, writes firing) 0.32–0.34 s/update.
  Halved to an idle 3090: ~0.09 and ~0.17 s/update, 0.87 s/evaluation, Fisher ~100 s.
- **Wall estimate** (idle 3090, ~37 evaluations per trained arm): XB6 ×4 job (6 trained arms
  incl. CT-nodrift) ≈ 38 min, XB6 ×1/×10 ≈ 31 min; XB12 ≈ 2× (≈ 75 / 62 min); canary ≈ 5 min
  (XB12 ≈ 10). A V100 ran exp-005's XB6 learner at about the same rate as this 3090, so XB12 ×4
  should fit `qos_gpu-dev` 2 h; a 5090 should be ~1.5–2× faster (DRN12 exp-005: 3.8× V100).
  XB total ≈ 5 3090-h-equivalent for the six fulls plus canaries, inside the 30 GPU-h cap
  shared with DRN; the canaries measure the real rate on the target GPU.

## Launch (prepared, not run)

`$SHA` is the trunk commit that merged this branch (pushed). Canary gate (exp-001): complete,
valid bundle; zero-write learners replay A0 (or writes explained); every arm's release read
equals A0's (checked in the runtime); pins equal; finite KL; measured time per update.

**RTX 5090 (trex/loulou/riri/fifi), lab path** (`run_lab.sh`: refuses a busy GPU or dirty
checkout, verifies the pins, `lifecycle plan` → `campaign run --resume` → `runs inspect
--verify-artifacts --require-complete` → drift row validation):

```bash
H=trex; R=analog-research/cmo-drift-recovery-20261118      # on the 5090 host, under ~
git -C /home/filip/analog-research/hwa bundle create /tmp/hwa-$SHA.bundle main
ssh $H "mkdir -p $R/inputs"
scp /tmp/hwa-$SHA.bundle $H:$R/
ssh $H "cd $R && git clone -q hwa-$SHA.bundle source-$SHA && git -C source-$SHA checkout -q --detach $SHA"
scp /tmp/fault-token-targets/teacher-samples8-n8192-51467dd7edb2.pt $H:$R/inputs/
scp /home/filip/analog-research/drn/data/opt_cmo_drn/std3-source.pt $H:$R/inputs/source-xb6-86522e23.pt
scp /home/filip/analog-research/opt-cmo-exp1/inputs/corpora/{wikitext-2-raw-v1,opt125m-synthetic-50m}.pt $H:$R/inputs/
# XB12: copy source-l12-ed53b6cb.pt (Jean Zay $SCRATCH/drift-all-mlps-20261111/inputs/) to $R/inputs/.
ssh $H nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader   # must be empty
ssh $H "tmux new -d -s cmo-xb6-canary 'cd $R/source-$SHA && bash campaigns/pilots/20261118-cmo-drift-recovery/run_lab.sh \
  campaigns/pilots/20261118-cmo-drift-recovery/lifecycles/canary/cmo-drift-recovery-20261118-xb6-m4-canary.json \
  ~/$R 2>&1 | tee ~/$R/xb6-m4-canary.log'"
```

Teacher: `~/analog-research/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin`
(`2d74da66…`, checked); interpreter `~/miniconda3/envs/py312/bin/python` (override with
`PYTHON=`/`TEACHER_WEIGHTS=`). Outputs `~/$R/{plans,runs}/<lifecycle id>`.

**Jean Zay V100 (`umg@v100`, `qos_gpu-dev`)**, the fault/exp-005 path: `python
campaigns/fault-recovery-paper/launch/stage_source.py <clean pinned checkout> main <out>`,
copy the bundle and receipt, `bash jz_setup_source.sh <root> $SHA`; a `fault_tuning.jz_manifest.v1`
like exp-005's `manifest-xb6.json` (same inputs and pins) with units
`xb{6,12}-m{1,4,10}` and `xb{6,12}-m4-canary` pointing at these lifecycles; `bash -l
jz_prepare.sh <launch> <manifest> xb6-m4-canary` (login-node plan + dry run), then `bash -l
jz_submit.sh <launch> <manifest> 00:45:00 now xb6-m4-canary`. The worker's drift validation is
part of this commit.

## Handoff

Both XB models passed native 5090 and V100 canaries. XB6 full ×4,
Jean Zay 387978, completed all eight arms and used 2,523 s. XB12 full ×4
is job 387963. Native XB6 JSON, measurements and validation are collected
under `results/cmo-drift-recovery-20261118/jean-zay/xb6-m4/`.

On XB6 the 24-h trapezoid-mean KL was 0.057628 for A0, 0.049467 CAL,
0.050334 LN, 0.058437 CT, 0.079147 L1 and 0.045737 LoRA. The
drift-free CT control reached 0.031654. Only L1 met the frozen 1-h LR
fallback criterion. The two native lifecycles under
`20261118-cmo-drift-recovery/fallback/` (commit 693eec0) run L1-low
(0.001/3) and L1-high (0.003) with paired A0 and A0-twin. The full run
uses 2,400 updates; the target canary uses 32 and validates execution.
The V100 backend, worker, source receipt and input pins stay at fe55904.
Compare against the original only after verifying the same initial apparent
state, deployment/control grids and cohort. Select among the three rates
by full 24-h mean KL, with the costs and all candidate results retained.
Job IDs, live watch and the 30 GPU-h shared balance are maintained in
[issue 67](https://github.com/filip137/on-chip-learning/issues/67).

**XB12 full and XB6 fallback gate.** Job 387963 completed all eight arms,
ten ages and 2,400 updates per trained arm on the V100 (3,140 s).
The native JSON and validation are in
`results/cmo-drift-recovery-20261118/jean-zay/xb12-m4/`. Mean KL was
0.221404 untreated, 0.170106 CAL, 0.172479 LN, 0.234053 CT, 0.172210 L1
and 0.147420 LoRA. CT used 109,068,405 writes and L1 used 176,092,341.
No XB12 arm triggered the 1-h fallback. Separately, the XB6 L1 bracket
gate 390133 passed all four arms and matched the original full XB6
deployment, cohort and A0/twin age grids. Full bracket job 390386 is under
the shared study watchdog and will be ranked against the original L1.
