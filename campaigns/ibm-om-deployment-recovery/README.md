# IBM OM deployment and recovery

How do mapping, HWA, quantization and post-deployment pulses change network performance on fixed and held-out IBM OM arrays?

Fixed-array HWA has a narrow KL benefit. Shared RESET-relative QAT transfers at 91.00% mean apparent accuracy without meeting its QAT-advantage gate; raw-active seven-level QAT meets its +2-point gate but reaches only 52.66%. All twelve original recovery arms finish below their 92.74% initial apparent validation accuracy. Thresholding largely suppresses damage; the TTv2 repair remains below the starting network.

## Evidence series

- [Common-window and differential deployment](series/001-common-window/README.md).
- [RESET-relative quantization and pulse recovery](series/002-reset-relative/README.md).
- [Raw-active p90 mapped HWA](series/003-raw-active/README.md).

## Current direction

Preserve the exact array, apparent deployment and pulse semantics when comparing recovery. Do not infer a need for on-chip training from an unclosed HWA gap or pool RESET-relative and raw-active coordinates.

Reviewed from local records on 2026-10-02; no new execution is assigned. [Generated evidence ledger](ledger.md) provides navigation, not live job monitoring or automatic scientific judgments. [Workflow](../../docs/experiment_workflow.md).
