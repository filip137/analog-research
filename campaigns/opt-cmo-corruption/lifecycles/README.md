# Lifecycle definitions

Each file is one loop under the [lifecycle contract](../../../docs/crossbar_lifecycle.md).
The file name equals its `lifecycle_id`. The linked experiment note owns the
decision rule, budget and interpretation; a definition alone authorizes no
execution. Validate with `python -m workflow check`.

| Lifecycle | Role | Experiment |
| --- | --- | --- |
| [opt125m-cmo-l3-exp1-probe](opt125m-cmo-l3-exp1-probe.json) | Cost probe (not evidence): corruption-aware HWA arm, one array, 0 % and 5 %, 8 sequences | [exp-001](../series/001-hwa-robustness/experiments/exp-001-hwa-source-robustness.md) |
| [opt125m-cmo-l3-lrdev-e20-lr3e-5](opt125m-cmo-l3-lrdev-e20-lr3e-5.json), [-lr1e-4](opt125m-cmo-l3-lrdev-e20-lr1e-4.json), [-lr3e-4](opt125m-cmo-l3-lrdev-e20-lr3e-4.json), [-lr1e-3](opt125m-cmo-l3-lrdev-e20-lr1e-3.json) | Stage 1a: standard HWA for 20 epochs at each rate; only prepare, devices and HWA run | [exp-001](../series/001-hwa-robustness/experiments/exp-001-hwa-source-robustness.md) |
| [opt125m-cmo-l3-lrdev-e50-lr5e-3](opt125m-cmo-l3-lrdev-e50-lr5e-3.json) | Stage 1a: standard HWA for exactly 50 epochs at 5e-3; only prepare, devices and HWA run | [exp-001](../series/001-hwa-robustness/experiments/exp-001-hwa-source-robustness.md) |
| [opt125m-cmo-l3-lrdev-lr1e-5](opt125m-cmo-l3-lrdev-lr1e-5.json), [-lr3e-5](opt125m-cmo-l3-lrdev-lr3e-5.json), [-lr1e-4](opt125m-cmo-l3-lrdev-lr1e-4.json) | Superseded first Stage 1a (4 epochs, not converged); kept as a record of what ran | [exp-001](../series/001-hwa-robustness/experiments/exp-001-hwa-source-robustness.md) |

`opt125m-cmo-l3-exp1` (Experiment 1 proper) is added once Stage 1a has frozen
the learning rate. Every other field is fixed in the experiment note.
