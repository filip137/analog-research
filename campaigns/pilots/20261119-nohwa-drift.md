# Pilot: HWA vs no-HWA crossbar sources under CMO ×4 drift (series003 exp-002 part b)

State: implementing. Branch: `exp/20261119-nohwa-drift`. Authorization: user, 19 November 2026
(issue [#33](https://github.com/filip137/on-chip-learning/issues/33)); crossbar share at most
1.2 V100 GPU-h including the canary; Jean Zay V100 dev QoS.

Owning note (all scientific choices): on-chip-learning
`campaigns/fault-recovery-paper/series/003-mapping-deployment/experiments/exp-002-nohwa-drn-and-drift.md`.

## Frozen comparison (transcribed)

Six cells, XB6 and XB12 × {HWA base, no-HWA base, no-HWA two-sided}. Each copies the CMO
drift-recovery exp-001 ×4 lifecycle of its depth (`campaigns/pilots/20261118-cmo-drift-recovery/
lifecycles/full/cmo-drift-recovery-20261118-<model>-m4.json`): law `aihwkit_cmo_log_independent.v1`
×4, clip 5–150 µS, draw seed 391103, reset_absorb, ten ages 0 … 24 h, realization 271001, 0 % faults,
naive Gaussian-endpoint P&V plus 5σ verify pair rewrite (seed 241016), dev 64 `7411cf50…`, pool
8,192 and target file `9cba6cf3…`, 2,400 updates. Changes (generator
`campaigns/fault-recovery-paper/make_nohwa_drift.py`): arms A0 (`hold_fault_aware`), A0-twin
(`no_drift`), CAL (`calibration`, LR 1e-3) only; base = the standard pair (no mapping key, as the
drift token budget / drift-regimes base); two-sided unchanged; no-HWA = HWA arm `pretrained` with the
pinned sources of [20261119-mapping-faults](20261119-mapping-faults.md) (XB6 `3442d7d2…`, XB12
`480e1323…`). Reused: HWA two-sided A0 / A0-twin / CAL from CMO exp-001 (XB6 job 387978, XB12
387963).

No code change was needed: the schema accepts the drift block with CAL as the learner, and
reset_absorb accepts both an omitted and an explicit mapping. Tests
`tests/test_nohwa_drift_lifecycles.py`: each cell differs from exp-001 only in arms, mapping and source;
the pretrained sources' provenance sections and arm digest fit; the HWA base cell equals the drift
token-budget ×4 base lifecycle (job 289103, A0 24-h mean 0.1263992838) in every deployment and drift
setting except the write clock, which does not affect unwritten devices.

Canary (one job): `xb6-hwa-base` must replay token-budget XB6 ×4 (apparent hashes `dd9b1f7c…` /
`227de134…`, A0 grid); `xb6-hwa-two-replay` (exp-001 XB6 lifecycle restricted to these arms) must
replay exp-001's deployment hashes (`46abcee9…` / `b3c81347…`) and A0 / A0-twin grids (A0 0.057628).
A passing replay validates reusing exp-001's HWA two-sided controls for both depths (same code path).

## Budget and execution

Expected per cell (exp-001 arm times): XB6 ~6 min, XB12 ~6.5 min plus ~2 min stage overhead;
7 units ≈ 0.95 GPU-h. One job per unit, the remaining five in parallel after the canary.

## Current execution and monitoring handoff

19 Nov: lifecycles committed; nothing launched.
