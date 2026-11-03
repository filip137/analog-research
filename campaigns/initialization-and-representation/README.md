# Initialization and representation

Which initializations and conductance representations preserve a useful, trainable perfect-diode DRN?

Teacher mapping, measured common windows and direct bounded FP32 mapping isolate different losses. RESET factorials expose a loss-by-amplifier-index interaction; the historical 95.97% result is not a corrected-circuit baseline. Four and eight devices approach the same classification accuracy after adaptation in one seed, while initialization and teacher KL remain different. Full-range and cell-bounded learning require their own learning-rate controls.

## Evidence series

- [RESET initialization and objective controls](series/001-reset-and-objective/README.md).
- [Teacher mapping and representation](series/002-teacher-and-representation/README.md).
- [Bounded FP32 learning and optimizer controls](series/003-bounded-learning/README.md).
- [Circuit and amplifier verification](series/004-circuit-verification/README.md).

## Current direction

Use an explicitly matched logical starting point before attributing degradation to devices. Keep circuit, objective and initialization changes separate; additional seeds and matched budgets remain proposed work.

Reviewed from local records on 2026-10-02; no new execution is assigned. [Generated evidence ledger](ledger.md) provides navigation, not live job monitoring or automatic scientific judgments. [Workflow](../../docs/experiment_workflow.md).
