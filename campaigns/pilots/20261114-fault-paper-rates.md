# Pilot 20261114 — DRN fault-paper LR screens at 0, 1 and 5 % (6 MLPs)

Tracking: on-chip-learning [#33](https://github.com/filip137/on-chip-learning/issues/33);
board `campaigns/fault-recovery-paper/board.md` there.

**Authorization:** user, 14 November 2026: tune every arm at 0, 1, 3 and 5 % faults,
with a fault-free (0 %) twin of every arm; "yes to everything, on jean-zay".

## What it is

The 20261113 screen ([pilot note](20261113-fault-paper-lr-tuning.md)) repeated at
three more fault rates on tuning map 271001, plus the 3 % edge extensions. Settings,
methods, recovery set, samples `df73ce66`, schedule and selection rule are unchanged.

| Rate | Deployment | Arms |
| --- | --- | --- |
| 0 % (fault-free) | new, `fault_kind = none` | holds; noisy, Child-D at 1e-4…1e-2; noiseless at 3e-5…1e-2; LoRA r16/r64 at 1e-4…1e-2 |
| 1 % | new, random | same |
| 5 % | new, random | same |
| 3 % | existing `159848f7…` | noiseless 3e-5; LoRA r16/r64 at 1e-4, 3e-4; Child-D *p* = 0.3 at 1e-3 (tuned rate) |

The grids at 0/1/5 % already include the 3 % edge extensions. Child-D *p* = 0.3 at
0/1/5 % waits for those rates' tuned values.

## Code change

- `ebl/schema/opt_cmo_drn.py`: `fault_tuning` also accepts a fault-free deployment
  (`none`, 0). Programming already handles `none`: no stuck mask, and partner rewrite only
  acts on programming outliers. So at 0 % naive and fault-aware P&V can still differ
  slightly; both holds are kept.
- Configs: `examples/opt_cmo_drn/fault_paper_rates/`, written by
  [make_configs.py](20261114-fault-paper-rates/make_configs.py). It imports the 20261113
  generator, so shared settings cannot drift.
- Tests: two new tests in `tests/test_opt_cmo_drn_fault_tuning.py`. Each config equals the
  3 % noiseless config except deployment, arms and (for one) `child_fraction`; the schema
  accepts `none`/0 and rejects `random`/0.

## Choices

- **Deployments on Jean Zay (V100 dev),** not on the 3090 as at 3 %: nom-cool-2's `/home`
  has 7 GB free. Each rate's screen jobs depend (`afterok`) on its deploy job, and all
  arms of one rate share that deployment.
- **Samples:** the verified V100 file `runs-9efad55/samples-v100/` of the 20261113 screen;
  no redraw.
- **Jobs:** per rate five A100 dev jobs (holds + noisy; noiseless; Child-D; LoRA r16; LoRA
  r64), as at 3 %, plus two 3 % extension jobs: 20 jobs in all.

## Execution

- Local canary on the RTX 3090 (nom-cool-2): fault-free deployment, then all six methods at
  3 updates with the 3090 sample draw (`results/fault-paper-rates/local/`).
- Launch: [jz_submit.sh](20261114-fault-paper-rates/launch/jz_submit.sh) from the staged
  `git archive` source with `source_snapshot.json`.
