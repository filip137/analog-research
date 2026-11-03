---
experiment: "exp-004"
evidence: "imported-summary"
summary: "The five-pair comparison improves from 93.162% HWA deployment to 94.880% rank-4 recovery."
verdicts: {}
---

# Result: exp-004 — MNIST HWA and physical LoRA rank study

[Experiment and historical specification](../experiments/exp-004-rank-study.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

This record uses the historical sources linked below; no new root under results is assigned.

Earlier ignored evidence stays under labs/cases; follow the exact paths in the frozen manifest entry. No legacy directory was moved into results.

## Observations

The five-pair comparison improves from 93.162% HWA deployment to 94.880% rank-4 recovery.


<a id="source-mnist_hwa_lora_experiment_setup-primary-four-arm-results"></a>

## Primary four-arm results

| W1/W2 seeds | FP32 -> ReRAM | HWA -> ReRAM | HWA -> ReRAM -> rank-4 LoRA |
| --- | ---: | ---: | ---: |
| 17 / 29 | 90.46% | 93.51% | 94.71% |
| 31 / 37 | 93.43% | 93.30% | 94.93% |
| 41 / 43 | 91.95% | 92.01% | 95.09% |
| 47 / 53 | 92.57% | 92.53% | 94.44% |
| 59 / 61 | 93.80% | 94.46% | 95.23% |
| **Mean** | **92.442%** | **93.162%** | **94.880%** |
| **Sample SD** | **1.324 pp** | **0.942 pp** | **0.313 pp** |

The resulting comparison is:

| Measurement | Accuracy | Change |
| --- | ---: | ---: |
| Clean ordinary FP32 reference | 96.150% | — |
| Ordinary FP32 -> ReRAM | 92.442% | -3.708 pp from clean FP32 |
| HWA -> ReRAM | 93.162% | +0.720 pp over FP32 -> ReRAM |
| HWA -> ReRAM -> rank-4 LoRA | 94.880% | +1.718 pp over HWA -> ReRAM |

Therefore:

- **HWA improvement:** `+0.720` percentage points relative to the matched
  FP32-to-ReRAM deployment baseline.
- **LoRA improvement:** `+1.718` percentage points relative to the matched
  HWA-to-ReRAM deployment baseline.
- **Combined HWA and LoRA improvement:** `+2.438` percentage points relative
  to the ordinary FP32-to-ReRAM mean.
- **Remaining gap from clean ordinary FP32:** `1.270` percentage points.
- **LoRA recovery of the clean-HWA deployment gap:** `55.8%`.

The clean-HWA recovery calculation is:

```text
(94.880 - 93.162) / (96.240 - 93.162) = 55.8%
```

HWA and LoRA improve different stages of the workflow. The `+0.720` and
`+1.718` point improvements should not both be measured from clean FP32:
HWA is evaluated against arm b, while LoRA is evaluated against arm c.

<a id="source-mnist_hwa_lora_experiment_setup-expanded-rank-4-device-replication"></a>

## Expanded rank-4 device replication

Rank-4 LoRA was repeated on five additional Wan-2022 device seed pairs:

```text
67/71, 73/79, 83/89, 97/101, 103/107
```

The additional realizations were retained without filtering, including a
hard 87.79% deployment.

Across all 10 HWA-to-ReRAM realizations:

| Measurement | Mean accuracy | Sample SD |
| --- | ---: | ---: |
| HWA -> ReRAM baseline | 92.408% | 2.045 pp |
| HWA -> ReRAM -> rank-4 LoRA | 94.639% | 0.463 pp |

- Mean paired LoRA improvement: `+2.231 pp`
- 95% t-interval for the mean paired gain: `[+1.011, +3.451] pp`
- Improved realizations: 10 out of 10
- Mean clean-HWA deployment gap recovered: 58.2%
- Correlation between deployed accuracy and gain: `r = -0.986`

The negative correlation means that the more damaged deployments generally
received the largest recovery. It is exploratory and is also influenced by
the clean-accuracy ceiling.

The 10-device result is not a new measurement of the HWA improvement because
the ordinary FP32 checkpoint was not programmed into the five additional
device pairs. It expands only the HWA-to-ReRAM versus LoRA comparison.

<a id="source-mnist_hwa_lora_experiment_setup-rank-expansion"></a>

## Rank expansion

Ranks 1, 2, 4, and 8 were compared on the original five HWA-to-ReRAM device
pairs with all other settings fixed.

| Rank | Added factor conductances | Mean accuracy | Gain over HWA -> ReRAM |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 93.162% | 0.000 pp |
| 1 | 1,788 | 94.518% | +1.356 pp |
| 2 | 3,576 | 94.732% | +1.570 pp |
| 4 | 7,152 | 94.880% | +1.718 pp |
| 8 | 14,304 | 94.984% | +1.822 pp |

The additional factor count across both branches is:

```text
rank * (1568 + 100 + 100 + 20) = rank * 1788
```

The base dense edges contain 158,800 conductances. Ranks 1, 2, 4, and 8 add
approximately 1.13%, 2.25%, 4.50%, and 9.01% additional conductances.

Most of the recovery is already available at rank 2. Rank 8 doubles the
rank-4 factor conductance count but improves mean accuracy by only
`0.104 pp`. With five matched device pairs, the paired 95% intervals for
each incremental rank difference include zero.

<a id="source-mnist_hwa_lora_experiment_setup-physical-and-numerical-validation"></a>

## Physical and numerical validation

Across the original rank-4 runs and the 20-run expansion, 25 completed LoRA
trajectories were audited:

- every zero-output-factor initialization reproduced its deployed baseline;
- every run completed 20 epochs;
- the cost-selected checkpoint was epoch 20 for every run;
- W1, W2, and the hidden bias were bit-identical before and after recovery;
- A1 and A2 stayed within `[1e-7, 1.0]`;
- B1 and B2 stayed within `[0.0, 1.0]`;
- all checkpoints and validation metrics were finite.

Repository validation after the expansion:

```text
Focused tests: 62 passed, 1 skipped
Legacy labs:   42 passed, 1 skipped
```

<a id="source-mnist_hwa_lora_experiment_setup-interpretation"></a>

## Interpretation

The experiment supports the following claims:

1. HWA improves robustness to the tested Wan-2022 ReRAM deployment compared
   with ordinary FP32 training.
2. Physical low-rank conductance branches can recover additional accuracy
   while the deployed base conductances remain frozen.
3. Recovery is consistent across the 10 tested rank-4 device realizations.
4. Increasing rank gives diminishing returns; rank 2 captures most of the
   observed recovery, and rank 4 is a reasonable area/accuracy compromise.
5. LoRA also reduces the observed variance across device realizations.
6. Direct full-model fine-tuning provides a substantially higher ideal
   accuracy ceiling, but requires rewriting most existing conductances.

It does not yet establish that:

- the result generalizes across multiple independently trained base models;
- the reported accuracy generalizes to an untouched final test split;
- LoRA remains effective when its own A/B arrays receive device programming
  noise and retention drift;
- the additive Gaussian HWA modifier is the best match for Wan-2022 ReRAM;
- incremental rank differences are statistically distinct.
- the ideal full-model advantage survives a physical pulse-aware ReRAM
  update process.

<a id="source-mnist_hwa_lora_experiment_setup-limitations"></a>

## Limitations

- Only one ordinary/HWA base-training seed was used.
- The canonical MNIST test split was used for checkpoint selection and final
  reporting.
- The original four-arm and rank comparisons use five device pairs; only
  rank-4 LoRA recovery was expanded to 10.
- HWA uses additive Gaussian perturbations rather than the exact Wan-2022
  forward model during training.
- ReRAM effects are applied to the base DRN, while LoRA A/B device effects
  remain ideal.
- The conductance reference was calibrated from the selected checkpoints
  rather than preregistered.
- The current nonnegative passive LoRA branch can add conductance but cannot
  directly subtract excess base conductance.
- Full-model fine-tuning uses ideal continuous updates and omits physical
  write noise, asymmetry, endurance, energy, and latency.

Historical source: [mnist-hwa-lora-rank-study](../../../../../docs/experimental_manifest.md#mnist-hwa-lora-rank-study).

- **Outcome:** positive
- **Interpretation:** This is the strongest positive evidence in the current program, but device-seed replication does not replace independent base-training replication.

## Interpretation and limitations

The later ten-device-pair rank-4 replication is still one base-training seed. Test-based checkpoint selection and ideal factor-array assumptions limit generalization.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Independent base-training seeds, untouched test selection and noisy factor arrays are the remaining discriminating controls.
