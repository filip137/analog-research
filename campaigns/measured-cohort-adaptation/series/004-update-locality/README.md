# Sign and pulse-constrained updates

SignSGD, monotone pulses and gradient thresholds. Raw versus isotonic initialization and selected versus terminal epochs must remain explicit.

[Campaign question and synthesis](../../README.md)

| Comparison | Specification | Evidence and interpretation |
|---|---|---|
| Four-device sign-only SGD | [exp-009](experiments/exp-009-sign-sgd.md) | [The archived signSGD comparison passes its declared feasibility gate.](results/exp-009-sign-sgd.md) |
| Unthresholded one-pulse-down control | [exp-010](experiments/exp-010-one-pulse-down.md) | [The archived monotone one-pulse hypothesis is refuted.](results/exp-010-one-pulse-down.md) |
| Four-device one-pulse gradient thresholds | [exp-011](experiments/exp-011-four-device-thresholds.md) | [The archived threshold sweep passes its selected-checkpoint gate, with terminal saturation limits.](results/exp-011-four-device-thresholds.md) |
| Eight-device gradient-threshold follow-up | [exp-012](experiments/exp-012-eight-device-thresholds.md) | [Local and Akib-collected threshold bundles are available, including a failed attempt.](results/exp-012-eight-device-thresholds.md) |

These records import historical work. Native completion, recorded validation, coverage and scientific interpretation are separate. Run directories remain at their original ignored locations; missing local evidence is explicit in the result notes.
