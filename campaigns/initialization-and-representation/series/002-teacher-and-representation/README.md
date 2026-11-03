# Teacher mapping and representation

Frozen ReLU teacher, single/differential encoding and common-window mappings. Keep matched ten-epoch comparisons distinct from 20-epoch screens; four/eight-device classification equality does not imply equal logits.

[Campaign question and synthesis](../../README.md)

| Comparison | Specification | Evidence and interpretation |
|---|---|---|
| Teacher-initialized measured DRN distillation | [exp-005](experiments/exp-005-teacher-distillation.md) | [Common-window differential mapping learns; independent single-conductance mapping stays near chance.](results/exp-005-teacher-distillation.md) |
| Ten-epoch teacher-initialized differential control | [exp-006](experiments/exp-006-differential-ten-epoch.md) | [The historical matched differential control reaches 97.48% test accuracy.](results/exp-006-differential-ten-epoch.md) |
| Ten-epoch teacher-initialized single-conductance control | [exp-007](experiments/exp-007-single-ten-epoch.md) | [The historical single-conductance replay remains near chance at 7.98% test accuracy.](results/exp-007-single-ten-epoch.md) |
| Pairwise common window versus bounded FP32 | [exp-008](experiments/exp-008-common-window-fp32.md) | [Common-window adaptation reaches 96.96% test versus 97.37% bounded FP32.](results/exp-008-common-window-fp32.md) |
| Four versus eight measured devices | [exp-009](experiments/exp-009-four-versus-eight.md) | [Four and eight devices reach 97.46% and 97.48% test accuracy after adaptation.](results/exp-009-four-versus-eight.md) |

These records import historical work. Native completion, recorded validation, coverage and scientific interpretation are separate. Run directories remain at their original ignored locations; missing local evidence is explicit in the result notes.
