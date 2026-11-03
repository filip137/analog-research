---
experiment: "exp-010"
evidence: "imported-summary"
summary: "The archived monotone one-pulse hypothesis is refuted."
verdicts: {"H-010": "contradicts"}
---

# Result: exp-010 — Unthresholded one-pulse-down control

[Experiment and historical specification](../experiments/exp-010-one-pulse-down.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| `results/mnist-four-device-cohort-a-one-pulse-down-20260819-v1` — unavailable locally | Unavailable locally | 0 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

The archived monotone one-pulse hypothesis is refuted.


Historical source: [mnist-four-device-cohort-a-one-pulse-down-20260819-v1](../../../../../docs/experimental_manifest.md#mnist-four-device-cohort-a-one-pulse-down-20260819-v1).

<details>

<summary>Historical record: mnist-four-device-cohort-a-one-pulse-down-20260819-v1</summary>


### mnist-four-device-cohort-a-one-pulse-down-20260819-v1 (`results/mnist-four-device-cohort-a-one-pulse-down-20260819-v1`; unavailable locally)

**Four-device cohort-A fine-tuning with one-pulse conductance decreases**

- **Finished:** 2026-08-19
- **Evidence class:** `exploratory`
- **Outcome:** refuted
- **Interpretation:** The predeclared hypothesis is refuted. The earlier raw-trace ordinary-SGD and signSGD accuracies are not matched controls because isotonic fitting also changed the initialization substrate: the present initialization was 38.32% validation accuracy versus 61.48% in those raw-trace studies.
- **Details:** local artifacts (`results/mnist-four-device-cohort-a-one-pulse-down-20260819-v1`; unavailable locally) · human report (`results/mnist-four-device-cohort-a-one-pulse-down-20260819-v1/analysis/report.md`; unavailable locally) · machine summary (`results/mnist-four-device-cohort-a-one-pulse-down-20260819-v1/analysis/summary.json`; unavailable locally)


</details>

## Interpretation and limitations

Raw-trace SGD/signSGD are not matched controls because this experiment also used isotonic initialization. Keep the failed scientific outcome even though the historical run completed.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use the threshold successor to investigate saturation; preserve the negative unconditional-pulse outcome.
