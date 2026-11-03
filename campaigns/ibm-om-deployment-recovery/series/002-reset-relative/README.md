# RESET-relative quantization and pulse recovery

Nine logical levels from shared RESET-relative commissioning. Recovery compares one exact development deployment; pulse budgets, endpoint seeds and algorithm versions define separate comparisons.

[Campaign question and synthesis](../../README.md)

| Comparison | Specification | Evidence and interpretation |
|---|---|---|
| Shared RESET-relative nine-level QAT | [exp-004](experiments/exp-004-shared-quantization.md) | [QAT reaches 91.00% mean apparent accuracy, but its +0.71-point advantage misses the gate.](results/exp-004-shared-quantization.md) |
| Matched refresh, direct-pulse and Tiki-Taka recovery | [exp-005](experiments/exp-005-pulse-recovery.md) | [All twelve terminal recovery arms fall below the 92.74% initial apparent validation accuracy.](results/exp-005-pulse-recovery.md) |
| Direct-pulse gradient-threshold follow-up | [exp-006](experiments/exp-006-gradient-thresholds.md) | [p99/cap-0.1 ends at 92.78%: +3.42 points over direct pulses, +0.04 over initialization.](results/exp-006-gradient-thresholds.md) |
| Canonical-axis TTv2 repair | [exp-007](experiments/exp-007-ttv2-repair.md) | [The repaired ideal-fast arm ends at 60.02%, above old Tiki-Taka but below the 92.74% start.](results/exp-007-ttv2-repair.md) |

These records import historical work. Native completion, recorded validation, coverage and scientific interpretation are separate. Run directories remain at their original ignored locations; missing local evidence is explicit in the result notes.
