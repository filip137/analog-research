# Analog energy attention: implementation handoff

13 November 2026 · [issue #32](https://github.com/filip137/on-chip-learning/issues/32).

The user requested a campaign, circuit design, proof that KCL decreases its
energy, then implementation. The scientific record is owned by the shared
study's [analog-energy-transformer campaign](https://github.com/filip137/on-chip-learning/tree/main/campaigns/analog-energy-transformer).
The circuit design and frozen decision rules are in `circuit-design.md` and
`series/001-ideal-circuit/experiments/exp-001-circuit.md` there (initially local).

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

Local verification result: source `841f499bd3566294439db0e1320f150a31002442`
was clean in all three readiness manifests. All cases pass, with 18 SPICE
transients, 24 numerical nudged equilibria, three programming steps, and all
five gradient groups checked (W/c are empty in softmax-only). Artifact hashing
and completion validation passed. Runs took 10.06 s and indexed 63.34 MB.
Worst SPICE transient error: 2.512e-6 V; centered SPICE gradient error: 5.148e-5
relative. The post-update loss falls in all three cases with input a fixed.
Readiness root: `results/analog-energy-transformer/exp-001/readiness/` under
the DRN trunk. Exact bundle names and interpretation are in the shared
campaign's `series/001-ideal-circuit/results/exp-001-readiness.md`.

Publication gate: automatic approval review rejected the shared-trunk merge
and push twice, including after the policy's standing permission was quoted.
It requires explicit confirmation in this conversation. The user was asked;
no approval is assumed. No active jobs remain. Preserve the local branch and
readiness evidence; do not call the experiment closed or promote these runs
to the pinned already-pushed evidence class. Next action is publication
approval, the pinned evidence run, and the closing git checkpoint.

The figure generator `experiments/energy_transformer/figures.py` reads completed
bundles without mutating them. Its plotting environment is separate from the
numerical runtime. The circuit still has ideal active macros and a software
programming controller; finite-bandwidth/device/writer hardware is not validated.


## Published-source verification — 13 November

The user explicitly approved publication and pinned CPU verification. The
implementation is published on `energy-transformer` and integrated into
`refactor/training-core`. Clean launch source:
`99143e7531612ec20780458e7d3c15837d0c9a00`. All three previously specified
cases passed the unchanged full characterization from a detached checkout
of that already-pushed source. Native completion and artifact-hash verification
passed for every bundle below. Original readiness runs remain unchanged.

| Case | Evidence bundle | Transient error (V) | SPICE EP relative error | Loss before / after |
| --- | --- | ---: | ---: | --- |
| mixed_relu | `20261113T150403.948501Z-363fe0cf-dad88653` | 2.51e-06 | 3.38e-05 | 0.18040144 / 0.16588997 |
| active_relu | `20261113T150407.691214Z-c2dc952f-445f8305` | 1.76e-06 | 5.15e-05 | 0.23679287 / 0.19972072 |
| softmax_only | `20261113T150411.204773Z-8fbbbca6-6b8b0709` | 1.76e-06 | 3.29e-05 | 0.19526474 / 0.18116940 |

Bundles are under `/home/filip/analog-research/drn/results/analog-energy-transformer/exp-001/evidence/`.
All three cases, 18 SPICE transients, eight numerical nudges per case, input
and parameter gradient groups, programming steps and both invalid controls
are present and pass. This completes the original circuit-design/KCL/EqProp
implementation milestone (#32). The larger convergent multi-token demonstration
continues in #36 on the user-requested `energy-transformer` branch.

Closing checkpoint: the original implementation and this result record are
merged into the published DRN trunk. The original merged branch/worktree and
the completed detached source checkout were removed after checking for local
changes; only generated Python caches existed in the original worktree. All
readiness, failed and published evidence is retained. The convergent model's
active worktree continues independently on `energy-transformer`.
