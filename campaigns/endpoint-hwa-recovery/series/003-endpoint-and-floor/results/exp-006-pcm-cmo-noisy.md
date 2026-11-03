---
experiment: "exp-006"
evidence: "imported-summary"
summary: "Both recovery paths remain stable, but deployment loss is too small to rank them."
verdicts: {}
---

# Result: exp-006 — PCM/CMO repeated noisy recovery

[Experiment and historical specification](../experiments/exp-006-pcm-cmo-noisy.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-ibm-pcm-cmo-noisy-recovery](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery) | complete: 9 | 9 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [analysis/summary.json](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery/analysis/summary.json).

## Observations

Both recovery paths remain stable, but deployment loss is too small to rank them.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [cmo_full/20260729T190829.109718Z-d630eaba-2b9cb08a](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery/cmo_full/20260729T190829.109718Z-d630eaba-2b9cb08a/result.json) | complete | selected accuracy: 96.43%; terminal accuracy: 96.38% |
| [cmo_lora/20260729T190829.332002Z-8ee76053-2850fb64](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery/cmo_lora/20260729T190829.332002Z-8ee76053-2850fb64/result.json) | complete | selected accuracy: 96.40%; terminal accuracy: 96.40% |
| [fp32_cmo_probe/20260729T190621.768171Z-5ad34457-b9ff552c](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery/fp32_cmo_probe/20260729T190621.768171Z-5ad34457-b9ff552c/result.json) | complete | selected accuracy: 96.33%; terminal accuracy: 96.33% |
| [fp32_pcm_probe/20260729T190555.112744Z-1e90d8cf-fc706b23](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery/fp32_pcm_probe/20260729T190555.112744Z-1e90d8cf-fc706b23/result.json) | complete | selected accuracy: 95.83%; terminal accuracy: 95.83% |
| [hwa/20260729T190640.644636Z-b36af5a4-16c65c90](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery/hwa/20260729T190640.644636Z-b36af5a4-16c65c90/result.json) | complete | selected accuracy: 96.37%; terminal accuracy: 96.37% |
| [hwa_cmo_probe/20260729T190748.239167Z-5ad34457-d4eaf50a](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery/hwa_cmo_probe/20260729T190748.239167Z-5ad34457-d4eaf50a/result.json) | complete | selected accuracy: 96.51%; terminal accuracy: 96.51% |
| [hwa_pcm_probe/20260729T190748.238119Z-1e90d8cf-5d8eb219](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery/hwa_pcm_probe/20260729T190748.238119Z-1e90d8cf-5d8eb219/result.json) | complete | selected accuracy: 96.15%; terminal accuracy: 96.15% |
| [pcm_full/20260729T190828.621092Z-7800f3e6-bb200b3b](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery/pcm_full/20260729T190828.621092Z-7800f3e6-bb200b3b/result.json) | complete | selected accuracy: 96.43%; terminal accuracy: 96.45% |
| [pcm_lora/20260729T190828.851851Z-b338e938-d6f8aa66](../../../../../results/mnist-ibm-pcm-cmo-noisy-recovery/pcm_lora/20260729T190828.851851Z-b338e938-d6f8aa66/result.json) | complete | selected accuracy: 96.30%; terminal accuracy: 96.30% |

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-results"></a>

## Results

| Measurement | Accuracy |
| --- | ---: |
| Existing FP32 reference | 96.15% |
| Shared two-epoch HWA checkpoint | 96.37% |
| FP32 → PCM | 96.15% |
| HWA → PCM | 96.35% |
| FP32 → CMO | 96.21% |
| HWA → CMO | 96.38% |
| HWA → PCM → full noisy BPTT | 96.43% |
| HWA → PCM → noisy rank-4 LoRA | 96.30% |
| HWA → CMO → full noisy BPTT | 96.43% |
| HWA → CMO → noisy rank-4 LoRA | 96.40% |

These are selected-checkpoint accuracies for FP32/HWA and pre-update
`initial_validation` accuracies for device deployment. A one-seed
`+0.20 pp` PCM or `+0.17 pp` CMO HWA difference is descriptive only.
Recovery checkpoints are selected by cost on one persistent device state at
each epoch; they are not selected from a multi-realization noise average.
Relative to matched HWA deployment, the selected recovery changes are:

| Device | Full noisy BPTT | Noisy rank-4 LoRA |
| --- | ---: | ---: |
| PCM | +0.08 pp (epoch 7) | -0.05 pp (epoch 20) |
| CMO | +0.05 pp (epoch 17) | +0.02 pp (epoch 20) |

The endpoint writes create essentially no deployment loss at this seed:
PCM leaves the FP32 accuracy unchanged and lowers the clean HWA accuracy by
only `0.02 pp`, while CMO is slightly above both clean checkpoints. The HWA
deployment advantage therefore mostly follows the extra HWA-stage training,
not demonstrated robustness to a large device perturbation. The two recovery
arms should be interpreted primarily as tests of optimization with repeated
noisy writes, not as a high-headroom recovery benchmark.
For the HWA checkpoint, the first write has an RMSE of `0.00517` in logical
DRN units for PCM and `0.00477` for CMO, against a logical reference range
of `[0, 1]`.

The four 75,000-step trajectories remained finite and stable. In both LoRA
arms, reconstructed CUDA programming confirms that W1 and W2 in the selected
checkpoint are bit-exact copies of the initial programmed base
(`max_abs=0`); all four factor arrays are finite and within their configured
bounds. Both full-recovery hidden biases remain bit-exact to the HWA
checkpoint.

The single-seed differences are too small to establish a full-versus-LoRA
winner. The defensible result is narrower: repeated endpoint noise did not
destabilize either optimization path, while neither device supplied enough
deployment damage to measure substantial recovery.

Historical source: [mnist-ibm-pcm-cmo-noisy-recovery](../../../../../docs/experimental_manifest.md#mnist-ibm-pcm-cmo-noisy-recovery).

- **Outcome:** mixed, stability-positive
- **Interpretation:** The study validates stable repeated endpoint writes on both full and factor arrays, but the initial device gap was effectively zero. The small changes cannot establish recovery efficacy or a full-versus-LoRA winner.

## Interpretation and limitations

Small single-seed changes with near-zero initial headroom do not establish a full-versus-LoRA recovery winner.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Select a predeclared regime with meaningful deployment loss for a recovery-efficacy comparison.
