---
id: "exp-007"
title: "Four/eight-convolution fault-rate sweep"
status: "imported"
hypotheses: []
---
# exp-007 — Four/eight-convolution fault-rate sweep

## Question and historical contract

How do failure rate, analog depth and HWA source affect incremental array-specific recovery beyond calibrated HWA?

Imported on 2026-10-04 from the September 2026 CIFAR worktrees. The
[frozen protocol](../../../../../docs/cifar_fault_rate_sweep.md) owns the original scientific settings.
No retrospective hypothesis is assigned and no new computation is authorized by this record.

## Cases, controls and budget

Two datasets, OM/PCM, four/eight analog convolutions, original/standard/noise/CDT sources, nominal and 1/2/3/5% open/Gmax/random faults; three final arrays per condition; five full epochs and matched controls. V1 and V2 execution revisions remain separate.

The primary comparisons use original-teacher KL and accuracy on paired arrays;
consult the protocol for source selection, write budgets, seeds and validity gates.

## Code, inputs and outputs

Public family: `cifar_crossbar_fault_sweep.v1`. [Configs](../../../../../examples/cifar_crossbar) and each native
`manifest.json`/`config.resolved.json` specify the exact command, inputs and hashes.
Use the [reproduction guide](../../../../../docs/cifar_reproduction.md) to relocate them.

- [results/cifar-sweep-collected](../../../../../results/cifar-sweep-collected)
- [results/cifar-sweep-analysis](../../../../../results/cifar-sweep-analysis)
- [results/cifar-sweep-cifar10-analysis](../../../../../results/cifar-sweep-cifar10-analysis)
- [results/cifar-sweep-cifar100-analysis](../../../../../results/cifar-sweep-cifar100-analysis)

## Current handoff

Imported history; no process is active on behalf of this migration. 562 complete retained native attempts including earlier revisions and canaries. Final V2 evidence comprises 376 campaign nodes and 6,480 control outcomes, not 6,480 independent arrays. The historical audit recorded 192 selected endpoint replays.
Source identities and preservation checks are in the
[migration audit](../../006-migration/results/exp-010.md).
The [result record](../results/exp-007.md) retains the historical interpretation and limits.
