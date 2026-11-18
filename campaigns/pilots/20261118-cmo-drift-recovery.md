# CMO drift recovery exp-001 — one-shot methods on DRN6/DRN12 ratio3 under CMO ×{1, 4, 10}

State: implemented, smoke passed, **ready to canary** (no remote run launched).
Branch: `drn/20261118-cmo-drift-recovery`. Authorization: user, 18 November 2026 (approved
goal, 30 GPU-h cap shared with the crossbar part, RTX 5090s).

Shared record (owns the question, decisions, fixed choices 1–7, handoff and budget):
on-chip-learning [`campaigns/cmo-drift-recovery/series/001-one-shot/experiments/exp-001-one-shot-methods.md`](https://github.com/filip137/on-chip-learning/blob/main/campaigns/cmo-drift-recovery/series/001-one-shot/experiments/exp-001-one-shot-methods.md),
issue [#67](https://github.com/filip137/on-chip-learning/issues/67). Predecessor: drift
token budget ([pilot](20261116-drift-token-budget.md), exp-005), whose stage this extends.
The crossbar counterpart is owned by hwa.

## Question

With the AIHWKit CMO law ×M (M ∈ {1, 4, 10}), ratio3 mapping and reset-absorb writes, how
much of the 24-h drift damage does each one-shot recovery at release remove on DRN6 and
DRN12: Child-Tuning_D (CT), the dense noisy learner (L1), LayerNorm adaptation (LN) and
zero-write output calibration (CAL), against untreated drift (A0), the drift-free release
(A0-twin) and CT with drift off (CT-nodrift, ×4 only)?

## Frozen comparison (from the shared note; not changed here)

- Source DRN6 `dfb9f22f…` / DRN12 `6805b629…`, ratio3 (G′ = min(3G, 88.2), bias current ×3),
  in-job `program_rewritten` naive P&V + 5σ partner rewrite at 0 % faults, realization
  271001; one shared deployment per job; t = 0 is release.
- Law `aihwkit_cmo_log_independent.v1`, clip 5–150 µS, draws keyed on (391103, array, read
  time in ms), M ∈ {1, 4, 10}; results at different M are not pooled.
- **Writes `reset_absorb`**: per-device write time t_w (0 at release), τ = t − t_w; a write at
  t sets held ← drifted read at t (the learner's draw) + step + CMO write noise (target
  clamp 9–88.2), then t_w ← t for written devices only.
- **Gate `fixed_mid`**: a device is written when |pending| ≥ 2σ(48.6 µS) = 1.11985 µS of
  physical conductance (under ratio3 the bank holds physical µS); no cap; remainder kept.
- Arms, each alone from release: A0, A0-twin, CAL (1e-2), LN (1e-2), CT (child_d, Fisher
  top-10 % per matrix at the released state, 1e-3), L1 (noisy, 3e-4), CT-nodrift (×4).
  Adam, batch 8, order seed 20261104, 2,400 updates cosine to zero, update k reads at
  1.5·k s, then frozen. Pool `69f79f1e…`, targets `9cba6cf3…` (tensor `51467dd7…`), dev
  `7411cf50…` (64 sequences, teacher KL).
- Grid 0, 1 min, 15/30 min, 1, 2, 4, 8, 16, 24 h; dev every 100 updates; 24-h projections
  at updates 300/600/1,200/2,400 for CT and L1.
- Per job: source ideal-weight KL, mapped-target (clean readout) KL, deployed KL at release
  (A0-twin), writes per device, fraction written, fraction at 5 µS at 24 h.

## Implementation (commit on this branch, merged to `refactor/training-core`)

| Semantic | Where |
| --- | --- |
| `reset_absorb` read law, per-device τ, bit-exact path for unwritten devices | `methods/drift_tuning.py` `drifted(…, write_ms)`, `WriteClock` |
| Absorbing write (drifted read → bounded noisy write → t_w ← t, no-ops restored) | `methods/drift_tuning.py` `AbsorbWritePort` |
| Optional `write_gate` / `drift_off_arms` (old configs keep their resolved form) | `methods/drift_tuning.py` `DriftTuning.parse`, `CampaignDriftTuning` |
| Fixed gate (`gate='fixed_mid'`, `threshold_q = 2·nominal_write_std_q`) | `methods/cmo_direct_adam.py` `ThresholdedBoundedWriteAdam` |
| Gate/port options of the physical arms; `LayerNormArm`; drifted digital steps | `methods/fault_tuning.py` |
| Arm rules (campaign form only), `layernorm` drift-only | `ebl/schema/opt_cmo_drn.py`, `methods/config/fault_tuning.py` |
| Stage: arms, Fisher once at release, clocks, drift-off arms, clean references, checks | `ebl/runtime/opt_cmo_drn/drift_tuning.py` |
| Configs | `make_configs.py` → `examples/opt_cmo_drn/cmo_drift_recovery/{full,canary}/` |
| Bundle check | `verify_unit.py`; lab launcher `run_lab.sh` |

Runtime guards: every arm's release KL equals A0's (identity corrections, same state);
zero-write drifting learners replay A0 at every age; clock resets equal written devices;
CT writes only inside its mask; digital arms leave devices and the model unchanged after
closing (model digest).

## Readiness evidence (18 November 2026)

**Tests.** `tests/test_cmo_drift_recovery.py` (10): forced absorbing write (held = drifted
read + step + noise exactly, clock restarts only on written devices, unwritten devices read
bit-identically to A0 at later ages), relaxation with τ = t − t_w, fixed gate fires exactly
at |pending| ≥ 1.11985 µS under ratio3 (not one ulp below), default 2ρ contract unchanged,
LR = 0 absorbing learner replays A0 at every age, CT writes only inside its mask, LN/CAL
restore the model, configs current and differing from exp-005 only where declared, invalid
configs rejected, and a CPU end-to-end run of the stage with all seven arms followed by
`verify_unit.py`. Neighbouring: `tests/test_drift_tuning.py` and `tests/test_opt_cmo_drn*.py`
(193 pass), fault/drift/architecture tests (95 pass); `test_fault_paper_launcher.py::
test_exact_endpoint_coverage_and_native_canaries` fails identically on the trunk (unrelated).

**Local smoke** (nom-cool-2 RTX 3090, commit `0dd2431`, DRN6 ×4 canary config with 8 updates
and two extra forced-write arms L1-force/CT-force at LR 0.1): `ebl train` rc 0 in 30 min,
`runs inspect --verify-artifacts --require-complete` valid, `verify_unit.py` verified.
- Clean references: source ideal 0.0477, mapped targets 0.0527, deployed release 0.0531
  (A0-twin, constant over the grid). A0 release and 60-s KL equal the exp-005 smoke on this
  GPU bit for bit (0.053053, 0.068279); A0 24-h mean 0.1239, 24-h KL 0.1330.
- CT, L1 (no writes in 8 updates under the 1.12 µS gate) and CT-nodrift replay A0 / A0-twin
  exactly. LN 24-h mean 0.1129 and CAL 0.1373 after 8 updates (not evidence).
- Forced arms: L1-force wrote 99.6 % of 113 M devices (4.35 writes per device), CT-force
  exactly 10.0 % (the mask); clock resets equal written devices; 2.1 % / 0.2 % of devices at
  the 5 µS floor at 24 h (A0 9e-8). ratio3 saturates 0.21 % of targets.
- Time on the 3090: Fisher pass 1,380 s (1,024 batches); update 0.78 s (CAL), 0.85 s (LN),
  1.2–1.3 s (physical arms); evaluation 2.9 s; peak 7.6 GB.

**5090 estimate** (exp-005: the 5090 ran DRN12 3.8× faster than the V100, which matches this
3090 per update): DRN6 ≈ 0.2–0.35 s/update, Fisher ≈ 6 min; a ×4 job (5 trained arms) ≈ 65
min, ×1/×10 ≈ 55 min; DRN12 about twice that (≈ 130 / 110 min). All six full jobs ≈ 8.5
5090-h plus canaries ≈ 0.6 h. Wall caps in the configs are about 2–3× these. The canaries
measure the real rate before the full jobs.

## Budget and execution

Cases (one job each; all arms share one in-job deployment):

| Case | Config | Host |
| --- | --- | --- |
| DRN6 canary ×4 (32 updates) | `examples/opt_cmo_drn/cmo_drift_recovery/canary/drn6-ratio3-m4-canary.json` | riri |
| DRN12 canary ×4 (32 updates) | `examples/opt_cmo_drn/cmo_drift_recovery/canary/drn12-ratio3-m4-canary.json` | fifi |
| DRN6 ×1, ×4, ×10 | `examples/opt_cmo_drn/cmo_drift_recovery/full/drn6-ratio3-m{1,4,10}.json` | riri |
| DRN12 ×1, ×4, ×10 | `examples/opt_cmo_drn/cmo_drift_recovery/full/drn12-ratio3-m{1,4,10}.json` | fifi |

Each model keeps all its jobs on one 5090 (the deployment depends on GPU numerics).
Launcher `run_lab.sh SOURCE OUTPUT CONFIG CORPUS_DIR WEIGHTS TEACHER LIMIT_SECONDS`:
GPU lock, refusal of an occupied GPU, source/teacher/receipt hash checks, `ebl train`,
`runs inspect --verify-artifacts --require-complete`, `verify_unit.py`. Storage per job
< 1 GB (masks, small adapters, JSON; no device-state checkpoint). Outputs under
`~/analog-research/cmo-drift-recovery-20261118/runs-v1/<config name>/run` on each host.

## Current execution and monitoring handoff

All DRN canaries passed. DRN6 ×4 completed on riri (launcher PID 2406112,
worker 2406154). DRN12 ×4 on fifi remains with its full-run watcher.
The [shared study record](https://github.com/filip137/on-chip-learning/issues/67)
owns subsequent phase admissions and shared accounting.

## Evidence and coverage

`results/cmo-drift-recovery-20261118/riri/drn6-ratio3-m4/run` contains
the locally verified native bundle for chip 271001. The source run and local
copy passed native hash inspection. The unit verifier accepted seven arms,
ten ages each and all five 2,400-update learners. Time to terminal validation
was 5,362.679 s (1.489633 GPU-h). All drifting learners beat untreated at 1 h.

Mean 24-h KL was 0.120391 untreated, 0.077072 CAL, 0.079059 LN,
0.081238 CT and 0.077338 L1. The release twin was 0.053020 and
drift-free CT was 0.038504. CAL recovered 64.3% of the mean gap with zero
analog writes. CT supplied 90.9% of L1's gain but used 9,876,580 modeled
writes versus 5,780,982 for L1. Their absolute KL gains over A0 persisted
through 24 h (24-h gain divided by 1-h gain: CT 1.58, L1 1.57).
