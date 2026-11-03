# RESET initialization and objective controls

Raw measured RESET studies, corrected versus historical amplifier indexing, and different losses. Budgets and circuit semantics differ; historical mismatched-index runs are provenance controls.

[Campaign question and synthesis](../../README.md)

| Comparison | Specification | Evidence and interpretation |
|---|---|---|
| Learning from measured RESET | [exp-001](experiments/exp-001-reset.md) | [RESET training learns, but the teacher-KL accuracy gate fails.](results/exp-001-reset.md) |
| Bias, loss and historical amplifier indexing | [exp-002](experiments/exp-002-bias-and-indexing.md) | [The 95.97% historical result is reproduced under mismatched amplifier semantics.](results/exp-002-bias-and-indexing.md) |
| RESET loss × bias × indexing factorial | [exp-003](experiments/exp-003-reset-factorial.md) | [Corrected logical-index MSE reaches 93.90–93.94%; loss and indexing interact.](results/exp-003-reset-factorial.md) |
| Ten-epoch differential RESET | [exp-004](experiments/exp-004-differential-reset.md) | [Historical differential RESET reaches 93.94% test accuracy; its named raw root is absent.](results/exp-004-differential-reset.md) |

These records import historical work. Native completion, recorded validation, coverage and scientific interpretation are separate. Run directories remain at their original ignored locations; missing local evidence is explicit in the result notes.
