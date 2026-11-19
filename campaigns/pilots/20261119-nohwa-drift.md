# No-HWA DRN source under faults; HWA vs no-HWA under CMO ×4 drift (exp-002)

State: implemented; launching. Branch `drn/20261119-nohwa-drift`. Authorization: user, 19 Nov
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
