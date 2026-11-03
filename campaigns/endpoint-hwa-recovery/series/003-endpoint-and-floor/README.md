# Endpoint noise and conductance floors

PCM/CMO/Wan endpoint fits, literal versus affine floors, and repeated noisy writes. Keep teacher-initialized and independently trained DRN regimes, ages and selection splits separate.

[Campaign question and synthesis](../../README.md)

| Comparison | Specification | Evidence and interpretation |
|---|---|---|
| PCM/CMO repeated noisy recovery | [exp-006](experiments/exp-006-pcm-cmo-noisy.md) | [Both recovery paths remain stable, but deployment loss is too small to rank them.](results/exp-006-pcm-cmo-noisy.md) |
| CMO literal-floor recovery | [exp-007](experiments/exp-007-literal-floor.md) | [Literal-floor deployment loses accuracy; full rewriting partly recovers it while passive LoRA fails.](results/exp-007-literal-floor.md) |
| Literal versus affine floor mechanisms | [exp-008](experiments/exp-008-floor-mechanism.md) | [Affine retained-floor mapping preserves accuracy while strongly attenuating voltages.](results/exp-008-floor-mechanism.md) |
| Independently initialized CMO/Wan comparison | [exp-009](experiments/exp-009-wan-cmo-independent.md) | [Noisy BPTT changes CMO by −0.63 points and Wan by +18.45 points in one realization.](results/exp-009-wan-cmo-independent.md) |
| Teacher-initialized CMO/Wan comparison | [exp-010](experiments/exp-010-wan-cmo-teacher.md) | [Noisy BPTT reaches 96.46% CMO and 91.84% Wan test accuracy from matched teacher initialization.](results/exp-010-wan-cmo-teacher.md) |

These records import historical work. Native completion, recorded validation, coverage and scientific interpretation are separate. Run directories remain at their original ignored locations; missing local evidence is explicit in the result notes.
