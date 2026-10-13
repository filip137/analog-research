# On-chip recovery after frozen HWA

Measure whether learning on a deployed faulty array improves over frozen HWA,
digital calibration and rewriting, and report the physical write cost.

- [Dense 50M-token on-chip pilot](experiments/exp-008-overnight-recovery.md)
- [Digital LoRA rank comparison on the same fixed array](experiments/exp-009-digital-lora.md)
- [Development-triggered LR follow-up](experiments/exp-010-lr-followup.md)
- [Recovery across HWA sources and fault rates](experiments/exp-013-recovery-regimes.md): three assignments, two/four/uncapped attempts and a frozen short recovery recipe.
- [Completed recovery context-length comparison](experiments/exp-014-context-length.md): 64–1,024 tokens, matched data and optimizer updates, with two/four/uncapped device-write budgets; [confirmed context-256 result](results/exp-014.md).
- [Predecessor: completed four-arm HWA comparison](../001-hwa-robustness/results/exp-007.md)

The first pilot uses standard 3% Gaussian HWA, one 3%-faulty array and dense
updates at LR 5e-3. Sparse writes and broader control comparisons follow review.
