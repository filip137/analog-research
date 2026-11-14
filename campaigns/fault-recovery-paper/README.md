# Fault recovery for the paper (crossbar half)

Crossbar side of the shared study
[`on-chip-learning/campaigns/fault-recovery-paper`](../../../on-chip-learning/campaigns/fault-recovery-paper/README.md)
(issues #33, #34, #38, #39 on `filip137/on-chip-learning`): at 0–5 % permanently stuck
devices, how much does on-chip learning recover after fault-aware P&V, compared with
digital LoRA, analog LoRA and Child-Tuning_D?

- Code: the native `fault_tuning` lifecycle stage (`ebl/runtime/fault_tuning.py`,
  `methods/fault_tuning.py`), tests in `tests/test_fault_tuning.py`.
- Lifecycles: `make_lifecycles.py` → `lifecycles/`; pinned inputs in `pins.json`.
- Jean Zay launch: `launch/`.

## Series

- [001 — learning-rate tuning at 3 %](series/001-lr-tuning/experiments/exp-001-crossbar-lr-screen.md);
  [exp-002, all 12 MLPs analog](series/001-lr-tuning/experiments/exp-002-crossbar-l12-lr-screen.md)
