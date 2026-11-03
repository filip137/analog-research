---
experiment: "exp-012"
evidence: "imported-summary"
summary: "Local and Akib-collected threshold bundles are available, including a failed attempt."
verdicts: {}
---

# Result: exp-012 — Eight-device gradient-threshold follow-up

[Experiment and historical specification](../experiments/exp-012-eight-device-thresholds.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-eight-device-cohort-a-gradient-threshold-sweep-20260820-v1](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-20260820-v1) | complete: 6, failed: 1 | 6 |
| [mnist-eight-device-cohort-a-gradient-threshold-sweep-akib-20260820-v1](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-akib-20260820-v1) | complete: 4 | 4 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [analysis/four-vs-eight/report.md](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-20260820-v1/analysis/four-vs-eight/report.md), [aggregate/results.json](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-20260820-v1/aggregate/results.json).

Existing analyses and receipts: [aggregate/results.json](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-akib-20260820-v1/aggregate/results.json).

Historical subprocess manifests: campaigns/manifests/mnist_relu_drn_eight_device_one_pulse_down_threshold_10ep.json and campaigns/manifests/mnist_relu_drn_eight_device_one_pulse_down_threshold_akib.json. The local report at analysis/four-vs-eight/report.md owns the cross-topology comparison.

## Observations

Local and Akib-collected threshold bundles are available, including a failed attempt.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [analysis/remote-provenance/akib-eight-device-threshold-smoke-20260820-v1/20260820T102705.546578Z-79e9e6b1-7a3abb5c](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-20260820-v1/analysis/remote-provenance/akib-eight-device-threshold-smoke-20260820-v1/20260820T102705.546578Z-79e9e6b1-7a3abb5c/result.json) | complete | selected student accuracy: 43.75%; initial student accuracy: 31.25%; terminal student accuracy: 43.75% |
| [fresh-test/threshold_p90/20260820T103336.842316Z-df387013-ecf8b374](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-20260820-v1/fresh-test/threshold_p90/20260820T103336.842316Z-df387013-ecf8b374/result.json) | complete | evaluation accuracy: 84.04% |
| [fresh-test/threshold_zero/20260820T103324.926905Z-df387013-85216d5f](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-20260820-v1/fresh-test/threshold_zero/20260820T103324.926905Z-df387013-85216d5f/result.json) | complete | evaluation accuracy: 48.98% |
| [initialization-hold/20260820T101143.331986Z-6dbd0759-ffc758fc](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-20260820-v1/initialization-hold/20260820T101143.331986Z-6dbd0759-ffc758fc/result.json) | complete | selected student accuracy: 48.54%; initial student accuracy: 48.54%; terminal student accuracy: 48.54% |
| [stages/threshold_p90/local/threshold_p90_train/attempt-002/runs/20260820T102603.860361Z-c4f86644-f45c119a](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-20260820-v1/stages/threshold_p90/local/threshold_p90_train/attempt-002/runs/20260820T102603.860361Z-c4f86644-f45c119a/result.json) | complete | selected student accuracy: 82.40%; initial student accuracy: 48.54%; terminal student accuracy: 13.76% |
| [stages/threshold_zero/local/threshold_zero_train/attempt-002/runs/20260820T101929.189192Z-9c0245eb-02f2a162](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-20260820-v1/stages/threshold_zero/local/threshold_zero_train/attempt-002/runs/20260820T101929.189192Z-9c0245eb-02f2a162/result.json) | complete | selected student accuracy: 48.54%; initial student accuracy: 48.54%; terminal student accuracy: 8.76% |
| [stages/threshold_p99/akib/threshold_p99_test/attempt-001/runs/20260820T104045.294780Z-df387013-4c167d24](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-akib-20260820-v1/stages/threshold_p99/akib/threshold_p99_test/attempt-001/runs/20260820T104045.294780Z-df387013-4c167d24/result.json) | complete | evaluation accuracy: 86.72% |
| [stages/threshold_p99/akib/threshold_p99_train/attempt-001/runs/20260820T102840.720205Z-debe6db7-758155ef](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-akib-20260820-v1/stages/threshold_p99/akib/threshold_p99_train/attempt-001/runs/20260820T102840.720205Z-debe6db7-758155ef/result.json) | complete | selected student accuracy: 86.08%; initial student accuracy: 48.54%; terminal student accuracy: 78.34% |
| [stages/threshold_p99_9/akib/threshold_p99_9_test/attempt-001/runs/20260820T104049.263235Z-df387013-9524fdcb](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-akib-20260820-v1/stages/threshold_p99_9/akib/threshold_p99_9_test/attempt-001/runs/20260820T104049.263235Z-df387013-9524fdcb/result.json) | complete | evaluation accuracy: 82.96% |
| [stages/threshold_p99_9/akib/threshold_p99_9_train/attempt-001/runs/20260820T103334.909509Z-fc972c1e-2e61b857](../../../../../results/mnist-eight-device-cohort-a-gradient-threshold-sweep-akib-20260820-v1/stages/threshold_p99_9/akib/threshold_p99_9_train/attempt-001/runs/20260820T103334.909509Z-fc972c1e-2e61b857/result.json) | complete | selected student accuracy: 81.74%; initial student accuracy: 48.54%; terminal student accuracy: 81.74% |


<details>
<summary>Existing analysis: report.md</summary>

# Four- versus eight-device one-pulse-down comparison

| Gate | Four selected val. | Four test | Eight selected val. | Eight test | Selected epoch (4 / 8) |
|---|---:|---:|---:|---:|---:|
| zero | 38.32% | 38.40% | 48.54% | 48.98% | 0 / 0 |
| p90 | 84.18% | 85.91% | 82.40% | 84.04% | 1 / 1 |
| p99 | 82.22% | 83.65% | 86.08% | 86.72% | 7 / 6 |
| p99.9 | 72.98% | 74.38% | 81.74% | 82.96% | 10 / 10 |

All physical update invariants passed.

The schemes use separate per-parameter thresholds calibrated at their own shared isotonic initialization; threshold magnitudes are therefore not compared directly across parameterizations.

</details>


## Interpretation and limitations

Keep machine/shard records and shared controls explicit. A best selected epoch is not terminal recovery; these are historical same-protocol follow-ups, not new independent teacher seeds.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Keep topology-specific thresholds and selection epochs explicit; the recorded cross-topology table is a single-protocol comparison.
