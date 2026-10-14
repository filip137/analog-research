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
The completed recovery extension covers three of these sources, all 0–5% fault
rates and three independent arrays. At every tested nonzero rate, recovery
improves held-out teacher KL; four attempts per device preserve nearly all of
the corresponding uncapped gain. Healthy-chip benefit depends on the source and
cap. The completed healthy/random-3% partner-rewrite phase now closes
92.5–93.3% of std3's WikiText fault-KL gap across three arrays and beats the
tested recovery budgets without learning. Learning after that rewrite slightly
worsens std3's WikiText teacher KL. Other rates/kinds and digital calibration
remain unresolved; these results retain one-training-seed, model-based limits.

- [Ledger](ledger.md)
- [Series 001: HWA source robustness](series/001-hwa-robustness/README.md), [Experiment 1](series/001-hwa-robustness/experiments/exp-001-hwa-source-robustness.md)
- [HWA result](series/001-hwa-robustness/results/exp-007.md), [Four-arm comparison](series/001-hwa-robustness/experiments/exp-007-corruption-rate-sweep.md)
- [Recovery across regimes: final result](series/002-onchip-recovery/results/exp-013.md), [Frozen comparison](series/002-onchip-recovery/experiments/exp-013-recovery-regimes.md)
- [Partner rewrite: completed 3% phase](series/002-onchip-recovery/results/exp-016.md), [Broader comparison deferred](series/002-onchip-recovery/experiments/exp-016-pair-rewrite.md)
- [Dense 50M-token on-chip pilot](series/002-onchip-recovery/experiments/exp-008-overnight-recovery.md)
- [Lifecycle definitions](lifecycles/README.md)
- [Lifecycle contract](../../docs/crossbar_lifecycle.md), [CMO device contract](../../docs/reram_cmo_device.md)
- [Active comparison: training through the verified partner](series/002-onchip-recovery/experiments/exp-017-partner-training.md)

Raw runs live under `results/lifecycles/<lifecycle_id>/`, which is not versioned.
The pinned corpus lives under `data/corpora/`.
