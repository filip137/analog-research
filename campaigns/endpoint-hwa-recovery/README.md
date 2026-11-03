# Endpoint-model HWA and recovery

When do programming endpoint errors defeat HWA, and when can full or low-rank recovery repair the resulting function?

The historical Wan rank study shows additional recovery from low-rank branches on one trained base. PCM/CMO endpoint studies sometimes have too little deployment loss to compare recovery methods. Literal target clipping and passive floor loading are different mechanisms. Teacher-initialized CMO/Wan results show substantial noisy-BPTT recovery, but lack a matched no-HWA recovery arm and independent-seed coverage.

## Evidence series

- [Early adapter and device controls](series/001-early-controls/README.md).
- [MNIST HWA and low-rank recovery](series/002-mnist-low-rank/README.md).
- [Endpoint noise and conductance floors](series/003-endpoint-and-floor/README.md).

## Current direction

Compare recovery against its own saved deployed state and retain the initialization, floor mapping, device age and selection split. A population-level necessity claim still needs matched controls.

Reviewed from local records on 2026-10-02; no new execution is assigned. [Generated evidence ledger](ledger.md) provides navigation, not live job monitoring or automatic scientific judgments. [Workflow](../../docs/experiment_workflow.md).

[Unassigned historical follow-ups](ideas.md).
