---
id: "exp-010"
title: "Migrate CIFAR evidence and reproduction code into HWA"
status: "complete"
hypotheses: []
---
# exp-010 — Migration and reproducibility audit

## Question and authorization

Can the HWA worktree independently retain the original CIFAR results and reproduce
their analysis and native run commands? On 2026-10-04 the user requested migration
of the old crossbar results into HWA, campaign records following the LoRA format,
and all code needed for reproduction. This is an engineering audit, not a new
scientific recovery experiment.

## Inputs and preservation contract

ResNet donor: `/home/filip/server_code/.codex/worktrees/cifar-resnet-suffix-recovery`,
HEAD `e3052ac1f469fa14187a8c72d773f3e940010505`, including its uncommitted source.
Dense donor: `/home/filip/server_code/.codex/worktrees/cifar10-crossbar-recovery-20260917`,
HEAD `db93217c69444f93761be6eadab0eae15ccbad1d`, clean.
Destination: `codex/revised-hwa-training`, starting at `9606b48`.

Preserve both originals and LoRA unchanged. Keep failed and superseded attempts.
Copy retained artifacts independently using filesystem reflinks and verify source
and destination SHA-256. Preserve immutable manifests, original runtime records,
exact resolved scientific configs and source snapshots. Keep large artifacts ignored.

## Execution and budget

Local source integration, copies, hashing, config/input resolution, proportionate
unit/numerical tests and read-only report regeneration. No full HWA/recovery training,
remote launches, tuning or new scientific seed budget. Audit root:
[`results/cifar-migration-20261004/`](../../../../../results/cifar-migration-20261004/).

Acceptance: all retained donor files match; all native configs parse and their
explicit inputs resolve by hash locally; report tables match historical values;
required tests and native device checks pass. Reconcile missing states against the
pre-existing September 25 retention inventory, separately from migration errors.

## Handoff

Local migration and engineering validation are complete. The
[result record](../results/exp-010.md) owns checks, pre-existing defects and limits;
[the receipt](../../../migration-receipt.json) pins audit evidence. No training or
remote monitoring is active, and no historical run is marked newly executed.
