---
id: "exp-006"
title: "Five full recovery epochs with OM and PCM"
status: "imported"
hypotheses: []
---
# exp-006 — Five full recovery epochs with OM and PCM

## Question and historical contract

What remains of deployment loss after full-cohort recovery, compared with equally trained calibration and rewriting?

Imported on 2026-10-04 from the September 2026 CIFAR worktrees. The
[frozen protocol](../../../../../docs/cifar_full_epoch_recovery.md) owns the original scientific settings.
No retrospective hypothesis is assigned and no new computation is authorized by this record.

## Cases, controls and budget

CIFAR-10/100, four suffix convolutions, OM/PCM, original/standard-HWA sources, nominal and 1% faults, three paired arrays. Five 45,000-image epochs and all 10,000 test images at each endpoint.

The primary comparisons use original-teacher KL and accuracy on paired arrays;
consult the protocol for source selection, write budgets, seeds and validity gates.

## Code, inputs and outputs

Public family: `cifar_crossbar_full_epochs.v1`. [Configs](../../../../../examples/cifar_crossbar) and each native
`manifest.json`/`config.resolved.json` specify the exact command, inputs and hashes.
Use the [reproduction guide](../../../../../docs/cifar_reproduction.md) to relocate them.

- [results/cifar-full5-collected](../../../../../results/cifar-full5-collected)
- [results/cifar-full5-analysis](../../../../../results/cifar-full5-analysis)
- [results/cifar-full5-om-analysis](../../../../../results/cifar-full5-om-analysis)
- [results/cifar-full5-validation](../../../../../results/cifar-full5-validation)

## Current handoff

Imported history; no process is active on behalf of this migration. 32 complete retained native attempts including canaries/revisions. The final four studies have 20 stages, 480 controls and 1,920 post-recovery test endpoints.
Source identities and preservation checks are in the
[migration audit](../../006-migration/results/exp-010.md).
The [result record](../results/exp-006.md) retains the historical interpretation and limits.
