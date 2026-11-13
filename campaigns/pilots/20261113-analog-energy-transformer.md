# Analog energy attention: implementation handoff

13 November 2026 · [issue #32](https://github.com/filip137/on-chip-learning/issues/32).

The user requested a campaign, circuit design, proof that KCL decreases its
energy, then implementation. The scientific record is owned by the shared
study's [analog-energy-transformer campaign](https://github.com/filip137/on-chip-learning/tree/main/campaigns/analog-energy-transformer).
The circuit design and frozen decision rules are in `circuit-design.md` and
`experiments/exp-001.md` there (initially local).

Implement a new, explicitly ideal circuit family. Do not alter the existing
perfect-diode model or use a measured-device claim. Code: `networks/energy_transformer/`
for the circuit/solver/SPICE export; `methods/energy_transformer.py` for
EqProp; strict EBL schema and characterize runtime for native bundles.

Execution contract: three fixed signed circuits (mixed ReLU, all active,
softmax only), independent branch-current/KCL/energy checks, three starts,
SPICE transients and +/-0.005 equilibria, all five parameter/input gradient
groups against implicit derivatives and finite differences, beta sweep
0.04/0.02/0.01/0.005, one eta=0.1 update, invalid-compensation/RC controls.
Thresholds and exact scientific interpretation belong to the shared record.

Local CPU, one thread, at most ten minutes and 100 MB for evidence. Raw outputs:
`/home/filip/analog-research/drn/results/analog-energy-transformer/exp-001/`.
Branch `drn/20261113-analog-energy-transformer`. The executing agent owns direct
monitoring through semantic completion and bundle validation. No large-model,
GPU or remote training is authorized by this first implementation.

Status: design derived and specification frozen; implementation pending.
The launch checkpoint will record the exact config/command and pushed SHA.
