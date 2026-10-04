---
id: "exp-008"
title: "One-pulse open-loop recovery and RESET controls"
status: "imported"
hypotheses: []
---
# exp-008 — One-pulse open-loop recovery and RESET controls

## Question and historical contract

Does uncapped one-pulse open-loop learning change recovery from exact historical P0, and what changes under a separate RESET programming intervention?

Imported on 2026-10-04 from the September 2026 CIFAR worktrees. The
[frozen protocol](../../../../../docs/cifar_om_open_loop.md) owns the original scientific settings.
No retrospective hypothesis is assigned and no new computation is authorized by this record.

## Cases, controls and budget

Both datasets, four/eight convolutions, inherited sources/faults/arrays and five full epochs. Saved-P0 paired recovery and RESET-open-loop programming are separate arms; open loop has no verify feedback and no total recovery pulse cap.

The primary comparisons use original-teacher KL and accuracy on paired arrays;
consult the protocol for source selection, write budgets, seeds and validity gates.

## Code, inputs and outputs

Public family: `cifar_om_open_loop.v1`. [Configs](../../../../../examples/cifar_crossbar) and each native
`manifest.json`/`config.resolved.json` specify the exact command, inputs and hashes.
Use the [reproduction guide](../../../../../docs/cifar_reproduction.md) to relocate them.

- [results/cifar-om-openloop-collected](../../../../../results/cifar-om-openloop-collected)
- [results/cifar-om-openloop-analysis](../../../../../results/cifar-om-openloop-analysis)
- [results/cifar-om-openloop-cifar10-analysis](../../../../../results/cifar-om-openloop-cifar10-analysis)
- [results/cifar-om-openloop-cifar100-analysis](../../../../../results/cifar-om-openloop-cifar100-analysis)

## Current handoff

Imported history; no process is active on behalf of this migration. 32 complete native attempts. The final screen contains 5,184 controls across both datasets; each dataset has twelve full runs, 2,592 controls and 1,944 recorded endpoint replays.
Source identities and preservation checks are in the
[migration audit](../../006-migration/results/exp-010.md).
The [result record](../results/exp-008.md) retains the historical interpretation and limits.
