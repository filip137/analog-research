---
experiment: "exp-011"
evidence: "imported-summary"
summary: "The recorded direct bounded mapping passes the one-point accuracy-loss gate."
verdicts: {"H-011": "supports"}
---

# Result: exp-011 — Direct ReLU-to-bounded FP32 mapping

[Experiment and historical specification](../experiments/exp-011-relu-bounded-map.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-relu-to-bounded-fp32-map-20260824-v1](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1) | complete: 2 | 2 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-relu-to-bounded-fp32-map-20260824-v1

Saved summary: [summary.json](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1/analysis/summary.json). Generated `2026-08-24T13:45:35.773966+00:00`; recorded state `reviewed`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| bounded-map | 1 | 1 | 0 | 0 | 0 | True |
| bounded-map-test | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 2 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1/analysis/summary.json), [analysis/review.json](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1/analysis/review.json), [analysis/final.json](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1/analysis/final.json).

## Observations

The recorded direct bounded mapping passes the one-point accuracy-loss gate.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/bounded-map/20260824T134046.646663Z-d07d32ea-498b4429](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1/runs/bounded-map/20260824T134046.646663Z-d07d32ea-498b4429/result.json) | complete | selected student accuracy: 97.26%; initial student accuracy: 97.26%; terminal student accuracy: 97.26% |
| [runs/bounded-map-test/20260824T134120.827403Z-2a2be32a-84af9eef](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1/runs/bounded-map-test/20260824T134120.827403Z-2a2be32a-84af9eef/result.json) | complete | evaluation accuracy: 97.38% |


Historical source: [mnist-relu-to-bounded-fp32-map-20260824-v1](../../../../../docs/experimental_manifest.md#mnist-relu-to-bounded-fp32-map-20260824-v1).

<details>

<summary>Historical record: mnist-relu-to-bounded-fp32-map-20260824-v1</summary>


### [mnist-relu-to-bounded-fp32-map-20260824-v1](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1)

**Direct ReLU compression into the bounded FP32 DRN**

- **Finished:** 2026-08-24
- **Evidence class:** `exploratory`
- **Outcome:** supported
- **Interpretation:** The evidence supports the predeclared hypothesis. It does not establish measured-device deployability.
- **Details:** [local artifacts](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1) · [human report](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1/analysis/report.md) · [machine summary](../../../../../results/mnist-relu-to-bounded-fp32-map-20260824-v1/analysis/summary.json)


</details>

## Interpretation and limitations

The direct map supplies a logical initialization control without training or device programming. It does not establish measured-device deployability.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use the exact direct-map checkpoint as the logical control for downstream deployment studies.
