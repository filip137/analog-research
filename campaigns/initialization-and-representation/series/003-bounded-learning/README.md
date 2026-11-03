# Bounded FP32 learning and optimizer controls

Ideal bounded and heterogeneous-cell-bounds FP32 references. These test optimization and representability without pulse programming; uniform initialization and teacher mapping are separate interventions.

[Campaign question and synthesis](../../README.md)

| Comparison | Specification | Evidence and interpretation |
|---|---|---|
| Independently trained bounded DRN reference | [exp-010](experiments/exp-010-bounded-teacher.md) | [The independently trained bounded DRN reaches 94.29% held-out test accuracy.](results/exp-010-bounded-teacher.md) |
| Direct ReLU-to-bounded FP32 mapping | [exp-011](experiments/exp-011-relu-bounded-map.md) | [The recorded direct bounded mapping passes the one-point accuracy-loss gate.](results/exp-011-relu-bounded-map.md) |
| Full-range uniform bounded initialization | [exp-012](experiments/exp-012-uniform-initialization.md) | [Inherited rates yield 83.74% validation and 84.43% test, below the 90% gate.](results/exp-012-uniform-initialization.md) |
| Uniform-initialization learning-rate screen | [exp-013](experiments/exp-013-uniform-lr-screen.md) | [Three available rate candidates reach 93.06–94.10% validation; overall planned coverage is partial.](results/exp-013-uniform-lr-screen.md) |
| SGD and Adam within heterogeneous OM cell bounds | [exp-014](experiments/exp-014-cell-bounds-optimizers.md) | [Adam reaches 96.76% validation and 96.94% test; its 97% validation gate fails.](results/exp-014-cell-bounds-optimizers.md) |
| Expanded SGD learning-rate control | [exp-015](experiments/exp-015-expanded-sgd.md) | [Expanded SGD reaches 96.18% validation and 96.48% test, meeting its stated gates.](results/exp-015-expanded-sgd.md) |

These records import historical work. Native completion, recorded validation, coverage and scientific interpretation are separate. Run directories remain at their original ignored locations; missing local evidence is explicit in the result notes.
