---
experiment: "exp-002"
evidence: "imported-summary"
summary: "The historical auto-LR transfer reaches 95.75% test; fixed-rate controls use a different baseline."
verdicts: {}
---

# Result: exp-002 — Cohort-B transfer and learning-rate controls

[Experiment and historical specification](../experiments/exp-002-cohort-b-transfer.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [measured-cohort-b-raw-mnist-exploratory-20260814-v1](../../../../../results/measured-cohort-b-raw-mnist-exploratory-20260814-v1) | complete: 1 | 1 |
| [measured-cohort-b-raw-mnist-test-20260814-v1](../../../../../results/measured-cohort-b-raw-mnist-test-20260814-v1) | complete: 1 | 1 |
| [measured-cohort-b-transfer-20260814-v1](../../../../../results/measured-cohort-b-transfer-20260814-v1) | No native status records | 0 |
| [measured-cohort-b-auto-lr-reproduction-20260815-v1](../../../../../results/measured-cohort-b-auto-lr-reproduction-20260815-v1) | complete: 1 | 1 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [summary.json](../../../../../results/measured-cohort-b-transfer-20260814-v1/summary.json).

Auto-LR launcher sidecars remain in the shared results/.launchers directory linked by the cohort-B LoRA record.

## Observations

The historical auto-LR transfer reaches 95.75% test; fixed-rate controls use a different baseline.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [20260814T171118.215270Z-51b83d58-a59a029b](../../../../../results/measured-cohort-b-raw-mnist-exploratory-20260814-v1/20260814T171118.215270Z-51b83d58-a59a029b/result.json) | complete | selected accuracy: 95.22%; terminal accuracy: 94.74% |
| [20260814T172750.515301Z-ab03cb5d-629f9c95](../../../../../results/measured-cohort-b-raw-mnist-test-20260814-v1/20260814T172750.515301Z-ab03cb5d-629f9c95/result.json) | complete | evaluation accuracy: 95.75% |
| [20260815T164828.477308Z-58b59c2c-53cfc5d5](../../../../../results/measured-cohort-b-auto-lr-reproduction-20260815-v1/20260815T164828.477308Z-58b59c2c-53cfc5d5/result.json) | complete | selected accuracy: 95.22%; terminal accuracy: 94.74% |

<a id="source-measured_cohort_b_finetuning-exploratory-outcome-2026-08-14"></a>

## Exploratory outcome (2026-08-14)

The raw cohort-B run completed all nine LR candidates and 20 production
epochs. The named cohort-A checkpoint measured 95.10% validation accuracy;
the initial cohort-B write measured 93.64%. Fine-tuning selected epoch 19
(zero-based epoch 18) at 95.22% validation accuracy and reached 95.75% on all
10,000 official test examples, versus 95.97% for the source cohort-A
checkpoint. All automatic transfer gates passed.

The selected learning rates were `2.5922685e-7` for W1 and its bias and
`1.9080487e-8` for W2. The initial A-to-B projection had zero target clipping
and mean absolute conductance errors of 2.55 uS for W1 and 3.78 uS for W2.
Ongoing global-nearest fine-tuning still admitted rare near-full-range pulse
jumps, so this result establishes held-out-device feasibility but does not
validate a sequential local-pulse programming policy.


## Interpretation and limitations

The automatic-LR feasibility result is not the control for later fixed-rate write policies. Preserve revision and rate-selection differences instead of attributing their accuracy gap to deadband or write sparsity.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use the later fixed-rate control when interpreting deadband and probabilistic writing; retain auto-LR only as transfer feasibility evidence.
