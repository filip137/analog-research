---
id: "exp-009"
title: "Development-selected closed-loop learning rates"
status: "imported"
hypotheses: []
---
# exp-009 — Development-selected closed-loop learning rates

## Question and historical contract

Can development-only rate selection improve uncapped closed-loop recovery on paired historical arrays?

Imported on 2026-10-04 from the September 2026 CIFAR worktrees. The
[frozen protocol](../../../../../docs/cifar_om_closed_loop_lr.md) owns the original scientific settings.
No retrospective hypothesis is assigned and no new computation is authorized by this record.

## Cases, controls and budget

Both datasets and depths; original and standard-HWA sources; nominal and 3/5% stuck high. Rates 1e-4/3e-4/1e-3 selected using two development arrays, then fixed for three-array confirmation. Fresh 1e-4 baseline, five full epochs, one pulse opportunity per update.

The primary comparisons use original-teacher KL and accuracy on paired arrays;
consult the protocol for source selection, write budgets, seeds and validity gates.

## Code, inputs and outputs

Public family: `cifar_om_closed_loop_lr.v1`. [Configs](../../../../../examples/cifar_crossbar) and each native
`manifest.json`/`config.resolved.json` specify the exact command, inputs and hashes.
Use the [reproduction guide](../../../../../docs/cifar_reproduction.md) to relocate them.

- [results/cifar-om-closedloop-lr-collected](../../../../../results/cifar-om-closedloop-lr-collected)
- [results/cifar-om-closedloop-lr-analysis](../../../../../results/cifar-om-closedloop-lr-analysis)

## Current handoff

Imported history; no process is active on behalf of this migration. 24 complete native runs: four canaries, eight development and twelve confirmation runs. 336 full-cohort and 96 smoke trajectories, with 432 historical epoch-five replays.
Source identities and preservation checks are in the
[migration audit](../../006-migration/results/exp-010.md).
The [result record](../results/exp-009.md) retains the historical interpretation and limits.
