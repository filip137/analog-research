---
experiment: "exp-011"
evidence: "imported-summary"
summary: "The archived threshold sweep passes its selected-checkpoint gate, with terminal saturation limits."
verdicts: {"H-011": "supports"}
---

# Result: exp-011 — Four-device one-pulse gradient thresholds

[Experiment and historical specification](../experiments/exp-011-four-device-thresholds.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| `results/mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1` — unavailable locally | Unavailable locally | 0 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

The archived threshold sweep passes its selected-checkpoint gate, with terminal saturation limits.


Historical source: [mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1](../../../../../docs/experimental_manifest.md#mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1).

<details>

<summary>Historical record: mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1</summary>


### mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1 (`results/mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1`; unavailable locally)

**Four-device cohort-A one-pulse-down gradient-threshold sweep**

- **Finished:** 2026-08-20
- **Evidence class:** `exploratory`
- **Outcome:** supported
- **Interpretation:** The predeclared hypothesis is supported. The threshold therefore repairs the unconditional update rule, but its useful operating point depends strongly on whether early stopping is allowed.
- **Details:** local artifacts (`results/mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1`; unavailable locally) · human report (`results/mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1/analysis/report.md`; unavailable locally) · machine summary (`results/mnist-four-device-cohort-a-gradient-threshold-sweep-20260819-v1/analysis/summary.json`; unavailable locally)


</details>

## Interpretation and limitations

Thresholding repairs the unconditional update in the declared regime; its benefit depends on selection and stopping. The absent raw root prevents a fresh terminal-metric reconciliation.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

State whether early stopping is permitted when selecting a threshold; recover the raw bundles before fresh verification.
