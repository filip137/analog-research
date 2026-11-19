# No-HWA DRN source under faults; HWA vs no-HWA under CMO ×4 drift (exp-002)

State: **complete** (19 Nov 2026): 2 prepares, 24 fault cells, 6 drift units, 2 control replays, all verified. Branch `drn/20261119-nohwa-drift`. Authorization: user, 19 Nov
2026 (issue #33). DRN cap 2.7 RTX-5090 GPU-h including canaries (raised from 1.8 by the user
the same day). Hosts fifi (all DRN12), riri and loulou (DRN6).

Shared record (owns the question, conditions and handoff):
on-chip-learning [`campaigns/fault-recovery-paper/series/003-mapping-deployment/experiments/exp-002-nohwa-drn-and-drift.md`](https://github.com/filip137/on-chip-learning/blob/main/campaigns/fault-recovery-paper/series/003-mapping-deployment/experiments/exp-002-nohwa-drn-and-drift.md).
Predecessor: [mapping-faults pilot](20261119-mapping-faults.md) (HWA cells, same pipeline).

## Source

`20261119-nohwa-drift/build_source.py`: a model-scope CMO payload = the crossbar no-HWA
source (hwa `pretrained`, method none; XB6 `3442d7d2…`, XB12 `480e1323…`: per-output-channel
q and channel scales, unit output gains, teacher MLP biases and norms, checked equal) plus the
teacher's digital weights (`digital.*`, same key set as the HWA std3 payload), scope `model`.
q·scale reproduces the teacher weights to 5e-8 relative; reconstructed source vs teacher KL
~1e-7 (random tokens, CPU). Payloads: DRN6 `b1bb8705…f98ee9`, DRN12 `75347a8f…0954ea`
(`results/nohwa-drift-20261119/sources/`). Then the native `prepare` stage (readout-gain fit,
no DRN training) → mapped DRN state, pinned in `pins.json`.

## Cases

- prepare: `nohwa_drift/prepare/drn{6,12}-nohwa-prepare.json` (DRN6 riri, DRN12 fifi).
- (a) `nohwa_drift/faults/`: the 24 mapping-faults cells with the no-HWA source only.
- (b) `nohwa_drift/drift/drn{6,12}-{hwa-base,nohwa-base,nohwa-ratio3}-m4.json`: CMO exp-001 ×4
  conditions, arms A0, A0-twin (drift off), CAL (1e-2, 2,400 updates). HWA ratio3 is reused from
  exp-001; `-replay` (32 updates) re-checks its deployment hashes and A0 grid on the same host.

Launcher `run_lab.sh MODE …` (prepare | cell | drift) with the locks, input checks
(`check_inputs.py`, `pins.json`), `runs inspect --verify-artifacts --require-complete` and
`verify.py` (the exp-001 verifiers with this pilot's pins).

Prepare (launch commit `d506572`): DRN6 on riri, mapped `48475a1f…a8e347`, clean synthetic-dev
KL 0.00519 (per-layer relative squared readout error 2.6e-4–1.2e-3); DRN12 on fifi, mapped
`47374e33…f97021c`, clean KL 0.02116 (1.8e-4–2.5e-3). Not ≈ 0 as on the crossbar (1e-10): the
T = 4 perfect-diode DRN reproduces each ReLU MLP to ~0.1 % squared error, not exactly. Still
9× (DRN6) below the HWA DRN6 source ideal on WikiText dev (0.0477). Copies in
`results/nohwa-drift-20261119/sources/`.

## Handoff

- Launch commits `d506572` (prepare, DRN6 HWA base drift on loulou) and the pins commit
  (all other units), each merged and pushed; staged by `git archive` on fifi, riri, loulou
  under `~/analog-research/nohwa-drift-20261119/source-drn-<sha>`.
- Launch note: stale October tmux sessions on riri/loulou leave `RUN_WATCH_ATTEMPT` in the tmux
  global environment; the launcher refused (rc 75, nothing ran) until started with
  `env -u RUN_WATCH_ATTEMPT -u RUN_WATCH_WORKER_ROOT -u CUDA_VISIBLE_DEVICES`. The legacy GPU
  lock and the occupancy check still apply.
- Units (tmux; logs under `~/analog-research/nohwa-drift-20261119/launch/`, runs under `runs-v1/`):
  fifi `x2-f2` (`fifi-chain.sh`): DRN12 replay → 12 no-HWA fault cells → drift nohwa-ratio3,
  nohwa-base, then hwa-base (cap raised to 2.7 GPU-h by the user, 19 Nov). riri `x2-r2`: DRN6
  replay → 12 no-HWA fault cells → drift nohwa-ratio3. loulou `x2-l1` drift hwa-base (staged
  `d506572`), then `x2-l2` drift nohwa-base.
- Replays (32 updates) of the HWA ratio3 controls on their exp-001 hosts: deployment hashes,
  clean references, A0 grid, A0 and twin means equal exp-001 exactly (DRN6 riri, DRN12 fifi);
  the exp-001 full controls are reused.

## Completion

All units rc 0, every bundle `complete valid=True` under `runs inspect --verify-artifacts
--require-complete` and accepted by `verify.py` (24 `cell-verified.json`; 6 + 2 `unit-verified.json`).
GPU time 2.52 RTX-5090 h of unit wall time (fifi 1.66, riri 0.44, loulou 0.43; prepare
< 1 min each), under the 2.7 h cap. No retries; the rc-75 refusals ran nothing. Host clocks
differ by hours (riri, loulou); durations are measured within each host. Compact copies:
`results/nohwa-drift-20261119/` (`faults/cells.{csv,json}`, `drift6.json`, `drift12.json`).

## Evidence

(a) Fault-aware WikiText-test KL, no-HWA (HWA exp-001 in brackets):

| Model, mapping | 0 % | 1 % | 3 % | 5 % |
| --- | --- | --- | --- | --- |
| DRN6 base | 0.0128 (0.0472) | 0.247 (0.069) | 3.58 (0.186) | 4.88 (0.388) |
| DRN6 ratio2 | 0.0139 (0.0484) | 0.033 (0.054) | 0.297 (0.077) | 1.34 (0.116) |
| DRN6 ratio3 | 0.0208 (0.0523) | 0.026 (0.055) | 0.079 (0.065) | 0.252 (0.078) |
| DRN12 base | 0.0550 (0.1920) | 3.33 (0.214) | 4.96 (0.447) | 5.24 (1.104) |
| DRN12 ratio2 | 0.0621 (0.2215) | 0.173 (0.227) | 2.68 (0.256) | 4.77 (0.320) |
| DRN12 ratio3 | 0.405 (0.279) | 0.472 (0.285) | 1.08 (0.301) | 3.48 (0.328) |

Clean dev KL no-HWA (ideal / mapped targets): DRN6 0.0069 / 0.0069, 0.0085, 0.0164; DRN12
0.0413 / 0.0413, 0.0467, 0.329. The DRN12 ratio3 mapped-target loss comes from 0.009 % of
saturated targets: every output channel's largest weight maps to 88.2 µS (q = ±1), so ×3
clips exactly the per-channel peak weights; the HWA sources lose less (0.198 → 0.288).
Denominator change at 3 % (median of layer medians, FA): 0.22 / 0.08 / 0.035 (DRN6),
0.23 / 0.08 / 0.036 (DRN12), as for HWA.

(b) CMO ×4 drift, 24-h trapezoid-mean dev KL (HWA ratio3 = exp-001, replay-checked):

| Unit | release | A0 | CAL | drift damage A0 − twin | at 5 µS at 24 h | at 150 µS |
| --- | --- | --- | --- | --- | --- | --- |
| DRN6 HWA ratio3 | 0.0530 | 0.120 | 0.077 | 0.067 | 7e-8 | 0 |
| DRN6 HWA base | 0.0480 | 0.708 | 0.300 | 0.660 | 0.329 | 0 |
| DRN6 no-HWA base | 0.0128 | 5.60 | 1.28 | 5.59 | 0.401 | 0 |
| DRN6 no-HWA ratio3 | 0.0210 | 1.64 | 0.380 | 1.62 | 1.6e-7 | 0 |
| DRN12 HWA ratio3 | 0.290 | 0.418 | 0.217 | 0.129 | 4e-8 | 0 |
| DRN12 HWA base | 0.199 | 2.12 | 0.744 | 1.92 | 0.341 | 0 |
| DRN12 no-HWA base | 0.0587 | 5.06 | 4.21 | 5.00 | 0.410 | 0 |
| DRN12 no-HWA ratio3 | 0.355 | 5.44 | 1.11 | 5.09 | 1.3e-7 | 0 |

## Review (no automatic rule; the user decides)

- Frozen expectations hold. No-HWA is 3.5–4× better at 0 % (except DRN12 ratio3, whose
  mapping clips the channel peaks) and far worse with faults from 1 % on, and ratio
  mappings reduce but do not remove that. Under ×4 drift, no-HWA damage is 8–40× HWA's
  and its 24-h mean KL never beats HWA's. Base mappings lose ~33–41 % of devices to the 5 µS floor;
  ratio3 avoids the floor. HWA ratio3 remains the best drift cell on both depths (CAL 0.077 / 0.217).
