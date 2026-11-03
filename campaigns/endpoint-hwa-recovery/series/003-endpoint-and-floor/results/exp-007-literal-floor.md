---
experiment: "exp-007"
evidence: "imported-summary"
summary: "Literal-floor deployment loses accuracy; full rewriting partly recovers it while passive LoRA fails."
verdicts: {}
---

# Result: exp-007 — CMO literal-floor recovery

[Experiment and historical specification](../experiments/exp-007-literal-floor.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-cmo-literal-floor-noisy-recovery](../../../../../results/mnist-cmo-literal-floor-noisy-recovery) | complete: 4 | 4 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [analysis/summary.json](../../../../../results/mnist-cmo-literal-floor-noisy-recovery/analysis/summary.json).

## Observations

Literal-floor deployment loses accuracy; full rewriting partly recovers it while passive LoRA fails.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [cmo_full/20260729T203638.904407Z-331aba99-49fe37df](../../../../../results/mnist-cmo-literal-floor-noisy-recovery/cmo_full/20260729T203638.904407Z-331aba99-49fe37df/result.json) | complete | selected accuracy: 89.73%; terminal accuracy: 89.73% |
| [cmo_lora/20260729T203638.904426Z-1c8f6562-17f9fe43](../../../../../results/mnist-cmo-literal-floor-noisy-recovery/cmo_lora/20260729T203638.904426Z-1c8f6562-17f9fe43/result.json) | complete | selected accuracy: 80.56%; terminal accuracy: 80.54% |
| [fp32_cmo_probe/20260729T203608.335175Z-f51309ba-04df9ffd](../../../../../results/mnist-cmo-literal-floor-noisy-recovery/fp32_cmo_probe/20260729T203608.335175Z-f51309ba-04df9ffd/result.json) | complete | selected accuracy: 79.24%; terminal accuracy: 79.24% |
| [hwa_cmo_probe/20260729T203612.681104Z-f51309ba-cd449384](../../../../../results/mnist-cmo-literal-floor-noisy-recovery/hwa_cmo_probe/20260729T203612.681104Z-f51309ba-cd449384/result.json) | complete | selected accuracy: 80.34%; terminal accuracy: 80.34% |

<a id="source-mnist_ibm_pcm_cmo_noisy_recovery-literal-floor-cmo-follow-up"></a>

## Literal-floor CMO follow-up

The follow-up changes only the CMO mapping and reuses the exact FP32 and HWA
checkpoints, device seed, noise scales, training hyperparameters, and
cost-selection protocol. It retains the physical floor:

```text
G_target_uS = clamp(88.199997 * G_drn, 9, 88.199997)
G_drn_realized = G_realized_uS / 88.199997
```

The DRN therefore sees a minimum connection of
`9/88.199997 = 0.1020408`; the `9 µS` offset is not subtracted.

| Measurement | Accuracy | Matched change |
| --- | ---: | ---: |
| Existing FP32 reference | 96.15% | — |
| Shared clean HWA checkpoint | 96.37% | — |
| FP32 → literal-floor CMO | 79.51% | -16.64 pp vs FP32 |
| HWA → literal-floor CMO | 80.61% | -15.76 pp vs HWA |
| HWA → CMO → full noisy BPTT | 89.73% | +9.12 pp vs deployed HWA |
| HWA → CMO → noisy rank-4 LoRA | 80.56% | -0.05 pp vs base-only deployment |

The HWA write clips `153,862 / 156,800` W1 targets (`98.13%`) and
`1,762 / 2,000` W2 targets (`88.10%`) to the floor. Its programming RMSE is
`0.08966` in DRN units, compared with `0.00477` under the earlier
normalized-offset mapping. The gap is therefore dominated by range mismatch,
not the stochastic residual alone.

Full BPTT reaches a transient peak of `89.74%` at epoch 19 and the
cost-selected `89.73%` at epoch 20. It recovers `57.87%` of the clean-HWA
gap but remains `6.64 pp` below clean HWA. Its last write still clips
`94.89%` of W1 and `91.15%` of W2 targets; recovery comes from strengthening
a small subset of useful connections, including values at the `88.2 µS`
ceiling.

The LoRA run initializes at `80.55%` because the nominally zero factor arrays
also retain the floor. Its cost-selected epoch-17 result is `80.56%`, only
`+0.01 pp` relative to its own initialization. At the last write, every clean
target in A1, B1, A2, and B2 remains below the physical threshold, so all
factor targets program to `9 µS` and differ only through sampled endpoint
noise. The positive-only branch has no subtractive path for cancelling the
excess base conductance.

All native statuses and eight checkpoint hashes verify. Full and LoRA biases
remain bit-exact to HWA, and a CUDA reconstruction confirms that both frozen
LoRA base arrays are bit-exact to their initial programmed realization. The
raw follow-up, exact summary, and staged source bundle are local ignored
artifacts under `results/mnist-cmo-literal-floor-noisy-recovery/`.
The source bundle SHA-256 is
`a6d96d837d21f4347dd6843de4d3dba3ba57751ef9a5dc106a38e741dedc4d43`.

This is still a post-hoc deployment calibration. The DRN was not trained from
initialization with the `9 µS` floor, and the endpoint model does not include
individual verification pulses, endurance, energy, latency, converter
quantization, or wire IR drop. The result establishes that literal absolute
scaling is a severe clipping/range-mismatch test. It does not by itself show
that every floor-retaining calibration must produce the same gap.

Historical source: [mnist-cmo-literal-floor-noisy-recovery](../../../../../docs/experimental_manifest.md#mnist-cmo-literal-floor-noisy-recovery).

- **Outcome:** mixed; full-recovery positive, LoRA negative
- **Interpretation:** The earlier normalized-offset mapping both removed the floor from the effective equations and avoided literal many-to-one clipping. Under literal absolute scaling, full base access provides substantial but incomplete recovery. A positive-only LoRA branch cannot subtract the dense excess conductance and has no controllable factor state while its targets remain below `9 µS`.

## Interpretation and limitations

This regime merges small targets and keeps a physical floor. It must not be pooled with offset-removed or affine mapping, and failure of positive-only factors is not a rank-independent impossibility.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use the affine-floor mechanism control before attributing loss to the floor alone.
