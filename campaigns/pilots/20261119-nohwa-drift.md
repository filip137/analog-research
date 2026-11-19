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

## Handoff

Pending launch.
