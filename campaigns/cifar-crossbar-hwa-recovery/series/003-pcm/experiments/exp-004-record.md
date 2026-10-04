---
id: "exp-004"
title: "PCM HWA/CDT and one-pass fixed-array recovery"
status: "imported"
hypotheses: []
---
# exp-004 — PCM HWA/CDT and one-pass fixed-array recovery

## Question and historical contract

Does fixed-array learning add benefit beyond standard HWA, noise-selected HWA, fault-aware CDT and equally trained calibration/rewrite controls?

Imported on 2026-10-04 from the September 2026 CIFAR worktrees. The
[frozen protocol](../../../../../docs/cifar_pcm_hwa_comparison.md) owns the original scientific settings.
No retrospective hypothesis is assigned and no new computation is authorized by this record.

## Cases, controls and budget

Full-cohort HWA and development-only source selection; original, standard, noise and three CDT sources; paired Gaussian PCM arrays; 5,000-image recovery; three fresh arrays per condition and full test evaluation.

The primary comparisons use original-teacher KL and accuracy on paired arrays;
consult the protocol for source selection, write budgets, seeds and validity gates.

## Code, inputs and outputs

Public family: `cifar_pcm_hwa_comparison.v1`. [Configs](../../../../../examples/cifar_crossbar) and each native
`manifest.json`/`config.resolved.json` specify the exact command, inputs and hashes.
Use the [reproduction guide](../../../../../docs/cifar_reproduction.md) to relocate them.

- [results/cifar-hwa-collected](../../../../../results/cifar-hwa-collected)
- [results/cifar-hwa-cdt-analysis](../../../../../results/cifar-hwa-cdt-analysis)
- [results/cifar-hwa-cdt-analysis-cifar10](../../../../../results/cifar-hwa-cdt-analysis-cifar10)
- [results/cifar-hwa-development-analysis](../../../../../results/cifar-hwa-development-analysis)

## Current handoff

Imported history; no process is active on behalf of this migration. 22 complete and two failed native attempts are preserved. Final comparisons contain 1,260 control measurements. Each dataset attempted 20 HWA candidates: 19 completed and SGD at LR 0.5 was rejected as nonfinite.
Source identities and preservation checks are in the
[migration audit](../../006-migration/results/exp-010.md).
The [result record](../results/exp-004.md) retains the historical interpretation and limits.
