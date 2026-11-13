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

Status: implementation ready. Circuit/numerical/SPICE tests passed; 25 existing
CLI/architecture tests passed with one version-dependent skip. A one-case CLI
smoke and artifact-hash inspection passed. The full run will use a clean pushed
launch commit, with three native bundles under the declared result root.

The learning step uses the SPICE phase readouts, holds input a fixed and
checks the updated circuit with SPICE. This adds one transient per case to
the initial specification (18 total), within the same ten-minute/100-MB budget.
All original decision thresholds remain unchanged.

Exact launch command, from the pinned checkout:

```bash
python -m ebl characterize --config examples/analog_energy_transformer_circuit.json --output-dir /home/filip/analog-research/drn/results/analog-energy-transformer/exp-001/evidence
```

Tool: ngspice 39.3, built locally and retained at
`results/analog-energy-transformer/tools/ngspice-39.3/bin/ngspice` under the
DRN trunk. Its binary hash and version are captured in every bundle. Both
negative controls are reported separately: frozen compensation changes the
force, and unequal softmax RCs can make the nominal energy increase.
