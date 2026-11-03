---
experiment: "exp-005"
evidence: "imported-summary"
summary: "The historical full-model control exceeds the low-rank recovery ceiling."
verdicts: {}
---

# Result: exp-005 — Full-model rewriting control

[Experiment and historical specification](../experiments/exp-005-full-finetune.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-hwa-reram-full-finetune](../../../../../results/mnist-hwa-reram-full-finetune) | complete: 10, failed: 1 | 10 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [analysis/summary.json](../../../../../results/mnist-hwa-reram-full-finetune/analysis/summary.json).

## Observations

The historical full-model control exceeds the low-rank recovery ceiling.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/baseline-validation/seeds_17_29/20260729T130343.254827Z-e7ec6cad-2ffc1e7b](../../../../../results/mnist-hwa-reram-full-finetune/runs/baseline-validation/seeds_17_29/20260729T130343.254827Z-e7ec6cad-2ffc1e7b/result.json) | complete | evaluation accuracy: 93.51% |
| [runs/baseline-validation/seeds_31_37/20260729T130353.400017Z-e7ec6cad-4eea8708](../../../../../results/mnist-hwa-reram-full-finetune/runs/baseline-validation/seeds_31_37/20260729T130353.400017Z-e7ec6cad-4eea8708/result.json) | complete | evaluation accuracy: 93.30% |
| [runs/baseline-validation/seeds_41_43/20260729T130402.822327Z-e7ec6cad-3831ae12](../../../../../results/mnist-hwa-reram-full-finetune/runs/baseline-validation/seeds_41_43/20260729T130402.822327Z-e7ec6cad-3831ae12/result.json) | complete | evaluation accuracy: 92.01% |
| [runs/baseline-validation/seeds_47_53/20260729T130419.512941Z-e7ec6cad-57934412](../../../../../results/mnist-hwa-reram-full-finetune/runs/baseline-validation/seeds_47_53/20260729T130419.512941Z-e7ec6cad-57934412/result.json) | complete | evaluation accuracy: 92.53% |
| [runs/baseline-validation/seeds_59_61/20260729T130526.577726Z-441225e1-df99fe9f](../../../../../results/mnist-hwa-reram-full-finetune/runs/baseline-validation/seeds_59_61/20260729T130526.577726Z-441225e1-df99fe9f/result.json) | complete | evaluation accuracy: 94.46% |
| [runs/full-finetune/seeds_17_29/20260729T130550.703597Z-2fc992ef-f16eec05](../../../../../results/mnist-hwa-reram-full-finetune/runs/full-finetune/seeds_17_29/20260729T130550.703597Z-2fc992ef-f16eec05/result.json) | complete | selected accuracy: 96.93%; terminal accuracy: 96.67% |
| [runs/full-finetune/seeds_31_37/20260729T130557.179457Z-2fc992ef-7b4d8a91](../../../../../results/mnist-hwa-reram-full-finetune/runs/full-finetune/seeds_31_37/20260729T130557.179457Z-2fc992ef-7b4d8a91/result.json) | complete | selected accuracy: 96.59%; terminal accuracy: 96.58% |
| [runs/full-finetune/seeds_41_43/20260729T130604.663459Z-2fc992ef-d2e4352d](../../../../../results/mnist-hwa-reram-full-finetune/runs/full-finetune/seeds_41_43/20260729T130604.663459Z-2fc992ef-d2e4352d/result.json) | complete | selected accuracy: 96.80%; terminal accuracy: 96.70% |
| [runs/full-finetune/seeds_47_53/20260729T130612.099064Z-2fc992ef-a2dd6f42](../../../../../results/mnist-hwa-reram-full-finetune/runs/full-finetune/seeds_47_53/20260729T130612.099064Z-2fc992ef-a2dd6f42/result.json) | complete | selected accuracy: 96.87%; terminal accuracy: 96.66% |
| [runs/full-finetune/seeds_59_61/20260729T130621.761431Z-2fc992ef-3b132e86](../../../../../results/mnist-hwa-reram-full-finetune/runs/full-finetune/seeds_59_61/20260729T130621.761431Z-2fc992ef-3b132e86/result.json) | complete | selected accuracy: 96.75%; terminal accuracy: 96.67% |

<a id="source-mnist_hwa_lora_experiment_setup-full-model-fine-tuning-control"></a>

## Full-model fine-tuning control

A fifth arm tests whether the deployed DRN contains additional recoverable
accuracy when the frozen-base constraint is removed:

```text
HWA -> ReRAM -> direct W1/W2/bias BPTT fine-tuning
```

It starts from the same five original HWA-to-ReRAM checkpoints, uses no
adapter or weight modifier, and trains for 20 epochs with the original base
learning rates:

```text
W1 = 0.08
W2 = 0.05
bias = 0.15
```

The dense arrays are independently bounded by `[1e-7, 1.1]`. The upper
bound is slightly above the nominal 1.0 device reference because one
AIHWKit-realized starting value is `1.05865`; using 1.0 would silently alter
that deployment before the comparison.

| W1/W2 seeds | HWA -> ReRAM | Rank-4 LoRA | Full-model fine-tuning |
| --- | ---: | ---: | ---: |
| 17 / 29 | 93.51% | 94.71% | 96.93% |
| 31 / 37 | 93.30% | 94.93% | 96.59% |
| 41 / 43 | 92.01% | 95.09% | 96.80% |
| 47 / 53 | 92.53% | 94.44% | 96.87% |
| 59 / 61 | 94.46% | 95.23% | 96.75% |
| **Mean** | **93.162%** | **94.880%** | **96.788%** |
| **Sample SD** | **0.942 pp** | **0.313 pp** | **0.130 pp** |

Full-model fine-tuning gains `+3.626 pp` over deployment and `+1.908 pp`
over rank-4 LoRA. It exceeds the clean HWA checkpoint by `+0.548 pp`, so it
is continuing classifier optimization rather than only reversing device
error.

That additional accuracy has a much larger rewrite footprint. On average,
85.81% of W1, 99.42% of W2, and 85.40% of the hidden bias change in the
cost-selected checkpoints: approximately 136,630 of 158,900 existing
values. Rank-4 LoRA adds 7,152 conductances but leaves all 158,900 existing
values bit-identical.

The full-model arm is an ideal upper control. Its continuous tensor updates
do not model physical ReRAM pulses, update asymmetry, write noise, endurance,
energy, or latency. It establishes available adaptation capacity, not yet
the feasibility of in-device fine-tuning.

Historical source: [mnist-hwa-reram-full-finetune](../../../../../docs/experimental_manifest.md#mnist-hwa-reram-full-finetune).

- **Outcome:** positive control
- **Interpretation:** The remaining post-LoRA accuracy gap is not an optimization ceiling of the deployed DRN. Direct access to all base parameters can recover and exceed clean-HWA accuracy, but it solves a different engineering problem: large-scale reprogramming instead of a small frozen-base adapter.

## Interpretation and limitations

Ideal access to all base parameters supplies an optimization control but rewrites a different set of conductances and omits realistic incremental pulse cost.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Compare rewrite cost and physically constrained updates before preferring full recovery over a frozen-base adapter.
