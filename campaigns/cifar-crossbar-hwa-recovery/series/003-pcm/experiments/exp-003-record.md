---
id: "exp-003"
title: "One-pass PCM permanent-fault screen"
status: "imported"
hypotheses: []
---
# exp-003 — One-pass PCM permanent-fault screen

## Question and historical contract

Can one pass of array-specific recovery improve teacher KL beyond calibration and matched frozen-target rewriting?

Imported on 2026-10-04 from the September 2026 CIFAR worktrees. The
[frozen protocol](../../../../../docs/cifar_pcm_fault_recovery.md) owns the original scientific settings.
No retrospective hypothesis is assigned and no new computation is authorized by this record.

## Cases, controls and budget

Original digital source, nominal and 0.01/0.1/1% open/Gmax/random per-device faults, three arrays/dataset, five controls, one pass over 5,000 images (79 batches), full 10,000-image tests.

The primary comparisons use original-teacher KL and accuracy on paired arrays;
consult the protocol for source selection, write budgets, seeds and validity gates.

## Code, inputs and outputs

Public family: `cifar_pcm_fault_recovery.v1`. [Configs](../../../../../examples/cifar_crossbar) and each native
`manifest.json`/`config.resolved.json` specify the exact command, inputs and hashes.
Use the [reproduction guide](../../../../../docs/cifar_reproduction.md) to relocate them.

- [results/cifar10-pcm-faults-one-epoch-v1](../../../../../results/cifar10-pcm-faults-one-epoch-v1)
- [results/cifar100-pcm-faults-one-epoch-v1](../../../../../results/cifar100-pcm-faults-one-epoch-v1)
- [results/cifar-pcm-fault-canary](../../../../../results/cifar-pcm-fault-canary)
- [results/cifar-pcm-fault-analysis](../../../../../results/cifar-pcm-fault-analysis)

## Current handoff

Imported history; no process is active on behalf of this migration. Eight complete scientific stages, 300 control measurements across both datasets, plus one complete native canary.
Source identities and preservation checks are in the
[migration audit](../../006-migration/results/exp-010.md).
The [result record](../results/exp-003.md) retains the historical interpretation and limits.
