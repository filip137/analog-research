---
experiment: "exp-005"
evidence: "imported-summary"
summary: "Corrected rank-4 passive LoRA remains nearly flat; selected test accuracy is 94.38%."
verdicts: {}
---

# Result: exp-005 — Corrected RESET passive LoRA recovery

[Experiment and historical specification](../experiments/exp-005-cohort-b-lora.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [measured-cohort-b-lora-rank4-fully-reset-20260815-v1](../../../../../results/measured-cohort-b-lora-rank4-fully-reset-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-lora-rank4-fully-reset-peak-20260815-v1](../../../../../results/measured-cohort-b-lora-rank4-fully-reset-peak-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-lora-rank4-fully-reset-peak-test-20260815-v1](../../../../../results/measured-cohort-b-lora-rank4-fully-reset-peak-test-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-lora-rank4-fully-reset-test-20260815-v1](../../../../../results/measured-cohort-b-lora-rank4-fully-reset-test-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-lora-rank4-reset-20260815-v1](../../../../../results/measured-cohort-b-lora-rank4-reset-20260815-v1) | running: 1 | 0 |
| [.launchers](../../../../../results/.launchers) | No native status records | 0 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

The shared results/.launchers directory also contains the cohort-B auto-LR reproduction and write-policy evaluation launchers. It is supporting operational evidence, not a scientific replicate.

## Observations

Corrected rank-4 passive LoRA remains nearly flat; selected test accuracy is 94.38%.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [20260815T171726.852231Z-1de7986f-c7e18779](../../../../../results/measured-cohort-b-lora-rank4-fully-reset-20260815-v1/20260815T171726.852231Z-1de7986f-c7e18779/result.json) | complete | selected accuracy: 93.68%; terminal accuracy: 93.68% |
| [20260815T172943.165131Z-1cd6ff86-dd3b7e94](../../../../../results/measured-cohort-b-lora-rank4-fully-reset-peak-20260815-v1/20260815T172943.165131Z-1cd6ff86-dd3b7e94/result.json) | complete | selected accuracy: 93.76%; terminal accuracy: 93.76% |
| [20260815T173102.117401Z-10ae1503-51d8c7ef](../../../../../results/measured-cohort-b-lora-rank4-fully-reset-peak-test-20260815-v1/20260815T173102.117401Z-10ae1503-51d8c7ef/result.json) | complete | evaluation accuracy: 94.42% |
| [20260815T172847.524690Z-10ae1503-d1665698](../../../../../results/measured-cohort-b-lora-rank4-fully-reset-test-20260815-v1/20260815T172847.524690Z-10ae1503-d1665698/result.json) | complete | evaluation accuracy: 94.38% |

<a id="source-measured_cohort_b_lora_recovery-exploratory-result"></a>

## Exploratory result

The corrected rank-4 MNIST run on 2026-08-15 did **not** produce meaningful
accuracy recovery:

| checkpoint | validation accuracy | held-out test accuracy |
| --- | ---: | ---: |
| deployed cohort-B base | 93.64% | not measured at initialization |
| fully reset LoRA initialization | 93.70% | not measured at initialization |
| epoch 1 (best validation accuracy) | 93.76% | 94.42% |
| epoch 20 (best validation cost) | 93.68% | 94.38% |

The corresponding selected full-network cohort-B fine-tuning reference reached
94.56% on the held-out test set. By epoch 20, 43.75--63.36% of the four LoRA
shadow arrays were at a conductance bound, and just 2.48--5.00% of their pulse
indices changed during the final epoch. This is consistent with a structural
limitation rather than an insufficient run length: two finite, nonnegative
conductance factors initialized at RESET do not form a zero-centered additive
correction, and the physical factors saturate while validation accuracy stays
flat. A signed or differential factor encoding is the appropriate next
experiment.

The run artifacts are under
`results/measured-cohort-b-lora-rank4-fully-reset-20260815-v1/`; the one-epoch
peak reproduction and both held-out evaluations use sibling result roots with
`peak` and `test` in their names.


## Interpretation and limitations

The nonnegative factors saturate while validation stays flat. This is consistent with a representation limitation but does not causally prove it. The earlier reset attempt has only a historical running status; peak reproductions are not independent seeds.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

A signed or differential factor control would discriminate the proposed representation limitation; longer repetition alone is not supported by this trajectory.
