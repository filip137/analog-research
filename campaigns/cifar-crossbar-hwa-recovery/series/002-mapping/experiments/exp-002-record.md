---
id: "exp-002"
title: "ResNet-32 suffix mapping pilot and PCM reference"
status: "imported"
hypotheses: []
---
# exp-002 — ResNet-32 suffix mapping pilot and PCM reference

## Question and historical contract

Does the tiled suffix preserve the pretrained digital network, and are device-state and update semantics explicit before comparing adaptation?

Imported on 2026-10-04 from the September 2026 CIFAR worktrees. The
[frozen protocol](../../../../../docs/cifar_resnet_suffix_recovery.md) owns the original scientific settings.
No retrospective hypothesis is assigned and no new computation is authorized by this record.

## Cases, controls and budget

CIFAR-10/100 teacher audit, nominal OM and legacy PCM pilots, HWA/deploy/recovery stages, and a separately labelled PCM inference/drift reference. The suffix contains four analog convolutions plus classifier; mapping capacity is 512.

The primary comparisons use original-teacher KL and accuracy on paired arrays;
consult the protocol for source selection, write budgets, seeds and validity gates.

## Code, inputs and outputs

Public family: `cifar_resnet_suffix_recovery.v1`. [Configs](../../../../../examples/cifar_crossbar) and each native
`manifest.json`/`config.resolved.json` specify the exact command, inputs and hashes.
Use the [reproduction guide](../../../../../docs/cifar_reproduction.md) to relocate them.

- [results/cifar-pilot-collected](../../../../../results/cifar-pilot-collected)
- [results/cifar-reference-collected](../../../../../results/cifar-reference-collected)

## Current handoff

Imported history; no process is active on behalf of this migration. 38 complete pilot native stages plus one complete reference stage.
Source identities and preservation checks are in the
[migration audit](../../006-migration/results/exp-010.md).
The [result record](../results/exp-002.md) retains the historical interpretation and limits.
