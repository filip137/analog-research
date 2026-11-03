---
experiment: "exp-009"
evidence: "imported-summary"
summary: "The archived signSGD comparison passes its declared feasibility gate."
verdicts: {"H-009": "supports"}
---

# Result: exp-009 — Four-device sign-only SGD

[Experiment and historical specification](../experiments/exp-009-sign-sgd.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| `results/mnist-four-device-cohort-a-sign-sgd-20260819-v1` — unavailable locally | Unavailable locally | 0 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

The archived signSGD comparison passes its declared feasibility gate.


Historical source: [mnist-four-device-cohort-a-sign-sgd-20260819-v1](../../../../../docs/experimental_manifest.md#mnist-four-device-cohort-a-sign-sgd-20260819-v1).

<details>

<summary>Historical record: mnist-four-device-cohort-a-sign-sgd-20260819-v1</summary>


### mnist-four-device-cohort-a-sign-sgd-20260819-v1 (`results/mnist-four-device-cohort-a-sign-sgd-20260819-v1`; unavailable locally)

**Four-device cohort-A fine-tuning with sign-only SGD**

- **Finished:** 2026-08-19
- **Evidence class:** `exploratory`
- **Outcome:** supported
- **Interpretation:** The predeclared hypothesis is supported. Neither sign arm clipped its shadow or reached conductance bounds.
- **Details:** local artifacts (`results/mnist-four-device-cohort-a-sign-sgd-20260819-v1`; unavailable locally) · human report (`results/mnist-four-device-cohort-a-sign-sgd-20260819-v1/analysis/report.md`; unavailable locally) · machine summary (`results/mnist-four-device-cohort-a-sign-sgd-20260819-v1/analysis/summary.json`; unavailable locally)


</details>

## Interpretation and limitations

Direction-only global-nearest updates are not one-pulse updates. The named raw result root is absent; retain the original control and its limits.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use the pulse-local controls to test physical locality rather than extrapolating global-nearest signSGD.
