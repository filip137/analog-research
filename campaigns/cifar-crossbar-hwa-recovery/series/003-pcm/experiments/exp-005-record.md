---
id: "exp-005"
title: "One to twenty passes on the PCM recovery cohort"
status: "imported"
hypotheses: []
---
# exp-005 — One to twenty passes on the PCM recovery cohort

## Question and historical contract

Does longer recovery change the comparison with equal-budget non-weight controls?

Imported on 2026-10-04 from the September 2026 CIFAR worktrees. The
[frozen protocol](../../../../../docs/cifar_pcm_recovery_epochs.md) owns the original scientific settings.
No retrospective hypothesis is assigned and no new computation is authorized by this record.

## Cases, controls and budget

Inherited digital/HWA/CDT sources; new paired arrays independent of prior finals; 1/2/5/10/20 passes over the same 5,000-image cohort; development-selected schedules and full-test milestones.

The primary comparisons use original-teacher KL and accuracy on paired arrays;
consult the protocol for source selection, write budgets, seeds and validity gates.

## Code, inputs and outputs

Public family: `cifar_pcm_recovery_epochs.v1`. [Configs](../../../../../examples/cifar_crossbar) and each native
`manifest.json`/`config.resolved.json` specify the exact command, inputs and hashes.
Use the [reproduction guide](../../../../../docs/cifar_reproduction.md) to relocate them.

- [results/cifar-epochs-collected](../../../../../results/cifar-epochs-collected)
- [results/cifar-epochs-analysis](../../../../../results/cifar-epochs-analysis)

## Current handoff

Imported history; no process is active on behalf of this migration. Ten complete retained native attempts including eight scientific stages. Twenty passes mean 100,000 image presentations and 1,580 updates.
Source identities and preservation checks are in the
[migration audit](../../006-migration/results/exp-010.md).
The [result record](../results/exp-005.md) retains the historical interpretation and limits.
