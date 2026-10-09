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
| [opt125m-cmo-l3-exp1](opt125m-cmo-l3-exp1.json), [opt125m-cmo-l6-exp1](opt125m-cmo-l6-exp1.json) | Experiment 1 proper: pretrained, standard and corruption-aware HWA at L3 and L6, deployed at 0–5 % stuck devices | [exp-001](../series/001-hwa-robustness/experiments/exp-001-hwa-source-robustness.md) |
| [opt125m-cmo-l6-exp1b](opt125m-cmo-l6-exp1b.json) | Experiment 1b: L6 sources distilled on 50 M teacher-generated tokens (streamed epochs), with 6 % noise and 5 % corruption-aware arms | [exp-002](../series/001-hwa-robustness/experiments/exp-002-hwa-synthetic-data.md) |
| [opt125m-cmo-l6-exp1b-smoke](opt125m-cmo-l6-exp1b-smoke.json) | Smoke (not evidence): two streamed epochs of the 5 % corruption-aware arm | [exp-002](../series/001-hwa-robustness/experiments/exp-002-hwa-synthetic-data.md) |
