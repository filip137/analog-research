# OPT-125M on CMO crossbars with stuck devices

When, and at what physical write cost, does on-chip training of deployed CMO
ReRAM arrays recover OPT-125M from stuck devices beyond hardware-aware training,
rewriting and digital calibration?

The analog part is fc1/fc2 of the last decoder MLPs on IBM CMO/HfOx ReRAM:
- the 2 % program-and-verify fit;
- an active and a mid-range reference device per weight;
- hierarchical stuck devices (see the [device contract](../../docs/reram_cmo_device.md)).

Attention, embeddings, LayerNorms and the head stay digital. All evidence is
exploratory and model-based. Stuck-device rates and their spreads are
synthetic.

**Current direction.** The synthetic-data, six-MLP HWA experiments selected
separate analog/calibration and digital learning rates, now frozen by the user.
The authorized four-arm comparison keeps 3% and 5% corruption-aware HWA alongside
standard 3% and 9% Gaussian-noise HWA, all at the frozen split learning rates.
Two sources are reused and two trained, with deployment across 0–5% stuck devices.
This establishes HWA baselines and healthy/faulty tradeoffs for the later matched
on-chip learning, rewriting and digital-calibration comparison.

- [Ledger](ledger.md)
- [Series 001: HWA source robustness](series/001-hwa-robustness/README.md), [Experiment 1](series/001-hwa-robustness/experiments/exp-001-hwa-source-robustness.md)
- [Latest result](series/001-hwa-robustness/results/exp-006.md), [Four-arm comparison](series/001-hwa-robustness/experiments/exp-007-corruption-rate-sweep.md)
- [Lifecycle definitions](lifecycles/README.md)
- [Lifecycle contract](../../docs/crossbar_lifecycle.md), [CMO device contract](../../docs/reram_cmo_device.md)

Raw runs live under `results/lifecycles/<lifecycle_id>/`, which is not versioned.
The pinned corpus lives under `data/corpora/`.
