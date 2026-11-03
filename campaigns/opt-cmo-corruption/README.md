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

**Standing default (user, October 15, 2026).** Every new deployment is followed
by the improved program-and-verify: exp-016's verify-based partner rewrite
(`fault_rewrite` with `detection: verify`, `roles: pair`, `threshold_sigma: 5`).
Every new arm, including calibration and learning, starts from the rewritten
array. The unrewritten deployment is reported only as a reference row.

**First positive on-chip learning result (October 15, 2026; [exp-016 phases 3–4](series/002-onchip-recovery/results/exp-016.md)).**
- **The verify partner rewrite holds up to 6% stuck devices.** Across three
  arrays it keeps std3 within about 2 WikiText perplexity points of healthy for
  gmin/gmax faults at 1–6%. At 6%, perplexity goes from about 200 unrewritten
  to 51.6, against 49.5 healthy, which removes 93–95.5% of the teacher-KL
  damage.
- **Its leftover grows linearly with the rate,** about 0.010 KL per percentage
  point. That leftover is mainly the wrong-sign weights that no partner setting
  can fix.
- **On-chip learning on top of the rewrite removes part of that leftover.**
  With the frozen exp-013 cap-4 recipe on array 271002:
  - it removes 25–32% (gmin) and 16–32% (gmax) of WikiText's leftover teacher
    KL at 4–6%, and 36–49% on generated text;
  - this contrasts with `random` 3%, where learning after the rewrite slightly
    hurt;
  - the gain is small or absent at 2%;
  - it costs about 37M device writes on top of the rewrite;
  - routing learning to the reference where the active device was detected
    stuck (`verified_partner`, October 16) removes 96–98% of the wasted stuck
    writes but changes teacher KL by under 2% of the residual, so which device
    is trained is not the limit.
- **Scope:** one array, one HWA source, a learning rate that was not re-tuned,
  and **no digital calibration control yet**. It is therefore not yet the
  AGENTS.md viability claim.
- **Next:** the same rewritten arrays with calibration only (no writes), then
  more arrays if learning still wins.

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
worsens std3's WikiText teacher KL. For gmin/gmax faults the picture changes
(see the result above). Digital calibration after the rewrite, cdt3/std9 under
other kinds, and `open`/random beyond 3% remain unresolved. These results retain
one-training-seed, model-based limits.

- [Ledger](ledger.md)
- [Series 001: HWA source robustness](series/001-hwa-robustness/README.md), [Experiment 1](series/001-hwa-robustness/experiments/exp-001-hwa-source-robustness.md)
- [HWA result](series/001-hwa-robustness/results/exp-007.md), [Four-arm comparison](series/001-hwa-robustness/experiments/exp-007-corruption-rate-sweep.md)
- [Recovery across regimes: final result](series/002-onchip-recovery/results/exp-013.md), [Frozen comparison](series/002-onchip-recovery/experiments/exp-013-recovery-regimes.md)
- [Partner rewrite and learning on top: result](series/002-onchip-recovery/results/exp-016.md) (random 3%; gmin/gmax/open 3%; gmin/gmax 1–6%; learning on top, one array), [experiment note](series/002-onchip-recovery/experiments/exp-016-pair-rewrite.md)
- [Dense 50M-token on-chip pilot](series/002-onchip-recovery/experiments/exp-008-overnight-recovery.md)
- [Lifecycle definitions](lifecycles/README.md)
- [Lifecycle contract](../../docs/crossbar_lifecycle.md), [CMO device contract](../../docs/reram_cmo_device.md)
- [Active comparison: training through the verified partner](series/002-onchip-recovery/experiments/exp-017-partner-training.md)
- [Series 003: task-fine-tuned OPT](series/003-finetuned-tasks/README.md), [exp-018: on-chip recovery of LoRA-fine-tuned HellaSwag/SST-2/PIQA teachers](series/003-finetuned-tasks/experiments/exp-018-finetuned-task-recovery.md), [exp-019: analog LoRA adapters on the std3 base, adapter vs base fault sensitivity](series/003-finetuned-tasks/experiments/exp-019-analog-lora.md)

Raw runs live under `results/lifecycles/<lifecycle_id>/`, which is not versioned.
The pinned corpus lives under `data/corpora/`.
