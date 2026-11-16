# Drift token budget — one-shot recovery at deployment (XB6, CMO ×4)

Shared record: on-chip-learning `campaigns/drift-regimes/series/003-cmo-recovery/experiments/exp-005-one-shot-recovery-token-budget.md`,
issue [#42](https://github.com/filip137/on-chip-learning/issues/42). Authorized by the user on
16 November 2026 (phase 1a, XB6 + DRN6, 4 V100-h). DRN counterpart: drn `drn/20261116-drift-token-budget`.

**Question.** Under AIHWKit CMO drift ×4, does the fault-tuning noisy learner (2,400
updates, LR 1e-3, 2σ gate, on the fixed 2.1M-token pool during the first hour after
release) lower the 24-h mean teacher KL relative to untreated drift on XB6?

**Frozen comparison.** XB6 std3 `86522e23…`, fault-free case, in-job deployment (naive P&V
+ verify rewrite, assignment 271001). `fault_tuning.drift`: law
`aihwkit_cmo_log_independent.v1`, M = 4, clip 5–150 µS, draw seed 391103, clock from
release, writes shift the held value without restarting the clock. Arms A0
(`hold_fault_aware`) and L1 (`onchip_noisy`). Update k reads at 1.5·k s. Grid 0–24 h
(10 ages), dev every 100 updates, 24-h projections at updates 300/600/1,200. Everything
else is inherited from `fault-token-budget-20261116-xb6-n8192`, except analog-LoRA tiles
(none). Lifecycles: `campaigns/pilots/20261116-drift-token-budget/lifecycles/`.

**Decision rule.** L1 24-h mean KL < A0's on this chip (one realization, exploratory);
report the retained fraction and the projection curve.

**Readiness.** Local smoke (nom-cool-2, 8 updates plus an LR 1.0 arm): campaign complete;
A0 24-h mean 0.1236 (release 0.0455); L1 without writes replays A0 at all ages; the hot arm
wrote 1.9e8 devices. Tests: `tests/test_drift_tuning.py` (6) and 48 neighbouring pass.

**Handoff.** Jean Zay V100 dev, source `source-hwa-61eab77`, Codex's token-budget `worker.py`
(verbatim). Job IDs and observations: shared experiment note.
