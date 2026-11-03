# Measured A/B transfer and write policies

The measured HDF5 physical-cell split and global-nearest projection. Preserve raw/isotonic, auto/fixed learning-rate and code-revision differences; test roots are evaluations, not replicates.

[Campaign question and synthesis](../../README.md)

| Comparison | Specification | Evidence and interpretation |
|---|---|---|
| Raw and isotonic cohort-A feasibility | [exp-001](experiments/exp-001-cohort-a.md) | [Cohort-A runs establish historical feasibility under their recorded circuit semantics.](results/exp-001-cohort-a.md) |
| Cohort-B transfer and learning-rate controls | [exp-002](experiments/exp-002-cohort-b-transfer.md) | [The historical auto-LR transfer reaches 95.75% test; fixed-rate controls use a different baseline.](results/exp-002-cohort-b-transfer.md) |
| Fixed-rate programming deadbands | [exp-003](experiments/exp-003-deadband.md) | [Deadbands reduce writes substantially without a resolved accuracy advantage.](results/exp-003-deadband.md) |
| Matched probabilistic write policies | [exp-004](experiments/exp-004-probabilistic-writing.md) | [Uniform p=0.85 saves 15% of requests with a −0.01-point test change in one seed.](results/exp-004-probabilistic-writing.md) |

These records import historical work. Native completion, recorded validation, coverage and scientific interpretation are separate. Run directories remain at their original ignored locations; missing local evidence is explicit in the result notes.
