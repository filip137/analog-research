---
id: "exp-001"
title: "Dense CIFAR-10 HWA and matched physical recovery"
status: "imported"
hypotheses: []
---
# exp-001 — Dense CIFAR-10 HWA and matched physical recovery

## Question and historical contract

Does apparent-feedback recovery add more benefit on faulty arrays than on healthy arrays after development-selected HWA?

Imported on 2026-10-04 from the September 2026 CIFAR worktrees. The
[frozen protocol](../../../../../docs/cifar10_crossbar_protocol.md) owns the original scientific settings.
No retrospective hypothesis is assigned and no new computation is authorized by this record.

## Cases, controls and budget

One digital pretraining run; four ten-epoch HWA noise arms; six deployments; twelve 30-epoch recoveries. One development array and two held-out arrays, paired healthy/faulted P0 and open/closed-loop writers. Epoch-zero recovery is eligible for validation-KL selection.

The primary comparisons use original-teacher KL and accuracy on paired arrays;
consult the protocol for source selection, write budgets, seeds and validity gates.

## Code, inputs and outputs

Public family: `cifar10_ibm_om_crossbar.v1`. [Configs](../../../../../examples/cifar10_crossbar) and each native
`manifest.json`/`config.resolved.json` specify the exact command, inputs and hashes.
Use the [reproduction guide](../../../../../docs/cifar_reproduction.md) to relocate them.

- [results/cifar10-digital-relu-pretrain-20260917-v1](../../../../../results/cifar10-digital-relu-pretrain-20260917-v1)
- [results/cifar10-hwa-recovery-20260917-v1](../../../../../results/cifar10-hwa-recovery-20260917-v1)

## Current handoff

Imported history; no process is active on behalf of this migration. 23 complete native runs: one teacher and all 22 hardware stages. No failed native attempts in these roots.
Source identities and preservation checks are in the
[migration audit](../../006-migration/results/exp-010.md).
The [result record](../results/exp-001.md) retains the historical interpretation and limits.
