# Drift token budget — one-shot recovery at deployment (DRN6, ratio3, CMO ×4)

Shared record: on-chip-learning `campaigns/drift-regimes/series/003-cmo-recovery/experiments/exp-005-one-shot-recovery-token-budget.md`,
issue [#42](https://github.com/filip137/on-chip-learning/issues/42). Authorized by the user on 16 November 2026
(phase 1a: XB6 + DRN6, 4 V100-h, +3 V100-h only for the DRN LR fallback). The crossbar
counterpart is hwa `exp/20261116-drift-token-budget`.

## Question

Under AIHWKit CMO drift ×4, does the fault-tuning noisy learner (2,400 updates on the
fixed 2.1M-token pool during the first hour after release) lower the 24-h mean teacher
KL versus untreated drift, and how much of the gain remains at 24 h?

## Frozen comparison

- Source DRN6 `dfb9f22f…`, mapping **ratio3** (G′ = min(3G, G_max), bias current ×3),
  fault-free deployment of assignment 271001 by the native `program_rewritten` (in-job).
- Law `aihwkit_cmo_log_independent.v1`, M = 4, clip 5–150 µS, fresh draw per
  (draw_seed 391103, array, age in ms), clock from release. **Writes shift the held
  programmed value and do not restart the clock** (user, 16 November 2026).
- Arms: A0 (hold) and L1 (`noisy`, LR 3e-4, 2ρ gate, cosine to zero over 2,400, batch 8,
  order seed 20261104, pool `69f79f1e…`, samples `51467dd7…`, dev `7411cf50…`).
  Update k reads at 1.5·k s. Inherited unchanged from fault token-budget `drn6-n8192`.
- Grid 0, 60 s, 15/30 min, 1, 2, 4, 8, 16, 24 h; dev every 100 updates; 24-h projections
  of the held state at updates 300/600/1,200 (and 2,400 = the 24-h grid point).
- Configs: `examples/opt_cmo_drn/drift_token_budget/{full,canary}/`, made by
  `campaigns/pilots/20261116-drift-token-budget/make_configs.py`.

**Decision rule.** Primary: L1 24-h mean KL < A0 24-h mean KL on this single chip
(exploratory, one realization). Report the retained fraction
(A0(24 h) − L1(24 h)) / (A0(1 h) − L1(1 h)) and the projection curve.
**LR fallback** (pre-declared): if L1's KL at 1 h is not below A0's or the loss diverges,
run L1 at 1e-4 and 1e-3 (same profile) and select by the same 24-h mean.

## Readiness evidence (local, nom-cool-2 RTX 3090, dirty worktree)

8-update smoke with an extra LR 0.1 arm: run complete and valid; A0 24-h mean 0.1239
(clean deployment 0.0531; drift-mapping exp-006 ratio3 ×4 reported 0.122); L1 made no
writes and replayed A0 exactly at all ten ages; the LR 0.1 arm wrote 5.4e8 devices
through the drift path. ~1.6 s/update, ~4 s/evaluation, 6.7 GB peak; 0.21 % of devices
saturate under ratio3; no device at the 5 µS floor through 24 h for A0. Tests:
`tests/test_drift_tuning.py` (8) plus 66 neighbouring DRN tests pass.

## Budget and handoff

Jean Zay V100 `umg@v100`, `qos_gpu-dev`, wrapper `run.slurm` (in-job deploy, native
`ebl train`, `runs inspect --verify-artifacts --require-complete`, `verify_unit.py`).
Canary 25 min wall, full 110 min wall. Storage < 1 GB (no device-state checkpoint;
the deployment is reproducible from seeds and its held hashes are recorded).
Handles and observations: see the shared experiment note.
