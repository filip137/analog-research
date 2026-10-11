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

**Current direction.** The completed four-arm, six-MLP comparison uses the frozen
split learning rates and covers five matched arrays at each of 0–5% faults.
Standard 3% Gaussian HWA preserves healthy performance; 9% Gaussian HWA is a
strong moderate-corruption control; explicit 3%-fault training adds a material
gain over 9% Gaussian in both domains at 5% actual faults. Training with 5%
temporary faults supplies no material improvement over 3% in this tested regime.
These one-seed, late-selected results establish baselines for the later matched
on-chip learning, rewriting and digital-calibration comparison.

- [Ledger](ledger.md)
- [Series 001: HWA source robustness](series/001-hwa-robustness/README.md), [Experiment 1](series/001-hwa-robustness/experiments/exp-001-hwa-source-robustness.md)
- [Latest result](series/001-hwa-robustness/results/exp-007.md), [Four-arm comparison](series/001-hwa-robustness/experiments/exp-007-corruption-rate-sweep.md)
- [Dense 50M-token on-chip pilot](series/002-onchip-recovery/experiments/exp-008-overnight-recovery.md): standard 3% Gaussian HWA, 3% actual faults, LR 5e-3; running on loulou after validated readiness checks.
- [Lifecycle definitions](lifecycles/README.md)
- [Lifecycle contract](../../docs/crossbar_lifecycle.md), [CMO device contract](../../docs/reram_cmo_device.md)

Raw runs live under `results/lifecycles/<lifecycle_id>/`, which is not versioned.
The pinned corpus lives under `data/corpora/`.
