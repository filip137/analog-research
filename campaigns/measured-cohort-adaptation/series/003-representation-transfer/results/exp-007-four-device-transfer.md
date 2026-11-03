---
experiment: "exp-007"
evidence: "imported-summary"
summary: "Four-device transfer falls to 35.79% test and recovers to 97.31% after adaptation."
verdicts: {}
---

# Result: exp-007 — Four-device A-to-B adaptation

[Experiment and historical specification](../experiments/exp-007-four-device-transfer.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1](../../../../../results/mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1) | complete: 4 | 4 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [aggregate/results.json](../../../../../results/mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1/aggregate/results.json).

## Observations

Four-device transfer falls to 35.79% test and recovers to 97.31% after adaptation.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [stages/four_device_cohort_b_quad_common_window_10ep/local/cohort_b_quad_test/attempt-001/runs/20260819T163948.483279Z-6807ef5b-f7253d5d](../../../../../results/mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1/stages/four_device_cohort_b_quad_common_window_10ep/local/cohort_b_quad_test/attempt-001/runs/20260819T163948.483279Z-6807ef5b-f7253d5d/result.json) | complete | evaluation accuracy: 97.31% |
| [stages/four_device_cohort_b_quad_common_window_10ep/local/cohort_b_quad_train/attempt-001/runs/20260819T163547.705092Z-21427a4f-7fd62c18](../../../../../results/mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1/stages/four_device_cohort_b_quad_common_window_10ep/local/cohort_b_quad_train/attempt-001/runs/20260819T163547.705092Z-21427a4f-7fd62c18/result.json) | complete | selected student accuracy: 97.14%; initial student accuracy: 35.68%; terminal student accuracy: 97.14% |
| [stages/four_device_cohort_b_quad_deployment_control/local/cohort_b_quad_deployment_control/attempt-001/runs/20260819T163541.599936Z-be7956e6-3b292b97](../../../../../results/mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1/stages/four_device_cohort_b_quad_deployment_control/local/cohort_b_quad_deployment_control/attempt-001/runs/20260819T163541.599936Z-be7956e6-3b292b97/result.json) | complete | selected student accuracy: 35.68%; initial student accuracy: 35.68%; terminal student accuracy: 35.68% |
| [stages/four_device_cohort_b_quad_deployment_control/local/cohort_b_quad_deployment_test/attempt-001/runs/20260819T163944.692899Z-6807ef5b-77f5a8b4](../../../../../results/mnist-four-device-reram-cohort-b-quad-common-window-10ep-20260819-v1/stages/four_device_cohort_b_quad_deployment_control/local/cohort_b_quad_deployment_test/attempt-001/runs/20260819T163944.692899Z-6807ef5b-77f5a8b4/result.json) | complete | evaluation accuracy: 35.79% |

<a id="source-mnist_four_device_cohort_b_adaptation-results"></a>

## Results

| Stage | Test accuracy | Teacher agreement | Teacher KL |
|---|---:|---:|---:|
| Cohort A, selected four-device source | 97.46% | 98.68% | 0.0183103 |
| Cohort B, immediate quad-common-window deployment | 35.79% | 35.72% | 1.761729 |
| Cohort B, after ten adaptation epochs | **97.31%** | **97.61%** | **0.0557048** |

The immediate deployment loses `61.67` percentage points relative to the
cohort-A source. Ten matched off-chip epochs recover `61.52` points, or
`99.76%` of that classification loss, and reduce test KL by `96.84%`. The
adapted checkpoint remains `0.15` points below its cohort-A source. The
selected checkpoint is the tenth completed epoch (zero-indexed epoch 9); its
validation accuracy, agreement, and KL are `97.14%`, `97.58%`, and
`0.0611194`. The deployment-control and adaptation arms have identical
initial cohort-B assignments and metrics (`35.68%` validation accuracy,
`35.86%` agreement, and `1.792735` KL), confirming deterministic
initialization before their update rules diverge.

For the matched eight-device paired-common-window experiment, immediate and
adapted cohort-B test accuracies were `42.96%` and `97.81%`, with KL values
`2.03767` and `0.03156`. Thus four devices are `7.17` accuracy points below
eight devices immediately after reassignment and `0.50` points below after
adaptation. The immediate four-device KL is nevertheless `13.54%` lower,
showing that classification and teacher-logit fidelity need not rank the two
initializations identically. After adaptation, four-device KL is `76.50%`
higher than eight-device KL.

<a id="source-mnist_four_device_cohort_b_adaptation-interpretation"></a>

## Interpretation

The four-device common-window representation remains expressive enough for a
high-accuracy solution on cohort B, despite having one conductance for each
collapsed pair sum and therefore less range and mismatch averaging than the
eight-device construction. It is not initialization-portable across device
cohorts. Differential initialization cancels the nominal common baseline,
but it does not make independently projected devices symmetric, preserve the
source signed scale, or preserve the passive load. The eight-device common-
window arm is more robust before adaptation and retains better logit fidelity
after adaptation; the four-device arm recovers almost all classification
performance while using half as many conductances per teacher weight.

This is a deterministic, single-seed comparison. It establishes feasibility
for this measured cohort and training budget, not universal robustness or an
in-situ programming result.

Historical source: [mnist-four-device-reram-cohort-b-quad-common-window-10ep](../../../../../docs/experimental_manifest.md#mnist-four-device-reram-cohort-b-quad-common-window-10ep).

- **Outcome:** completed; immediate cross-cohort transfer failed, but ten matched off-chip epochs recovered `99.76%` of the lost classification accuracy while retaining four conductances per teacher weight
- **Interpretation:** Differential initialization cancels a nominal common baseline but does not preserve symmetry, signed scale, or circuit denominators after independent measured-state projection. The four-device scheme is expressive after adaptation; the eight-device scheme remains better for immediate transfer and final logit fidelity in this single-seed comparison.

## Interpretation and limitations

The representation remains expressive after adaptation but is not portable across assignments at initialization. Eight-device teacher-logit fidelity remains better in the historical matched comparison.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Keep reassignment loss and recovery separate when choosing four versus eight devices.
