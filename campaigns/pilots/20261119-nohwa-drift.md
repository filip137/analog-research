# Pilot: HWA vs no-HWA crossbar sources under CMO ×4 drift (series003 exp-002 part b)

State: executed, collected and reviewed; the user decides source and mapping. Branch: `exp/20261119-nohwa-drift`. Authorization: user, 19 November 2026
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

Launch commit `4d9a066` (main, pushed); Jean Zay source `/lustre/fswork/projects/rech/umg/ucy17uy/
nohwa-drift-20261119/source-4d9a066/hwa` (2,553 files, 0 mismatches), launch dir `…/launch-4d9a066/`,
outputs `/lustre/fsn1/projects/rech/umg/ucy17uy/nohwa-drift-20261119/runs-v1/`. Jobs: canary 445217
(xb6-hwa-base + xb6-hwa-two-replay, 15 min 04 s); 445807 xb6-nohwa-base 6:29, 445810 xb6-nohwa-two
6:32, 445811 xb12-nohwa-base 8:14, 445812 xb12-nohwa-two 8:15, 445813 xb12-hwa-base 8:17. All
COMPLETED, no retries; 52 min 51 s = 0.88 V100 GPU-h (cap 1.2). Monitoring ended.

## Evidence and coverage

7/7 units complete; every bundle `runs inspect --verify-artifacts --require-complete` complete/valid
(`results/nohwa-drift-20261119/inspect.txt`); worker drift validation passed. Bundles (no
checkpoints) `results/nohwa-drift-20261119/bundles/`, table `results/nohwa-drift-20261119/table.{csv,json}`.

Canary passed bit for bit: XB6 HWA base reproduces the drift token-budget ×4 deployment hashes and
A0 grid (24-h mean 0.1263992838); the XB6 HWA two-sided replay reproduces CMO exp-001's deployment
hashes and the A0, A0-twin and CAL grids (0.057628 / 0.043107 / 0.049467). The reused HWA two-sided
controls are therefore valid for XB6 and, by the same code path and pins, XB12.

24-h trapezoid-mean dev KL; damage = A0 − A0-twin; clip = device fraction at the 5 µS / 150 µS
bounds at 24 h (A0).

| Model | Source | Mapping | A0 | A0-twin | CAL | Damage | Clip 5 µS / 150 µS |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| XB6 | HWA | base | 0.12640 | 0.04486 | 0.10785 | 0.08154 | 0.0081 / 0 |
| XB6 | HWA | two-sided (exp-001) | 0.05763 | 0.04311 | 0.04947 | 0.01452 | 0.2682 / 0 |
| XB6 | no-HWA | base | 2.05472 | 0.02670 | 0.65207 | 2.02802 | 0.0002 / 0 |
| XB6 | no-HWA | two-sided | 0.17648 | 0.00070 | 0.12586 | 0.17578 | 0.2895 / 0 |
| XB12 | HWA | base | 0.31965 | 0.20690 | 0.25074 | 0.11274 | 0.0076 / 0 |
| XB12 | HWA | two-sided (exp-001) | 0.22140 | 0.20411 | 0.17011 | 0.01729 | 0.2705 / 0 |
| XB12 | no-HWA | base | 4.80746 | 0.06031 | 1.81305 | 4.74715 | 0.0003 / 0 |
| XB12 | no-HWA | two-sided | 0.69760 | 0.00145 | 0.26392 | 0.69615 | 0.2912 / 0 |

## Interpretation and next decision

The frozen expectation holds: under ×4 drift the no-HWA damage far exceeds HWA's, and no-HWA
loses its release advantage on total 24-h KL in every cell. Best per model is HWA two-sided (XB6 A0
0.058, CAL 0.049; XB12 A0 0.221, CAL 0.170). Two-sided reduces drift damage 5–12× against base for
both sources. Caveats: one realization; the XB12 no-HWA two-sided A0 grid is erratic across ages
(0.39–1.16, fresh draws per age), so its mean is noisy; under two-sided ~27–29 % of devices sit at
the 5 µS clip at 24 h (the 9 µS floor devices drift down), none at 150 µS. The user decides.
