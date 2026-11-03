---
experiment: "exp-012"
evidence: "imported-summary"
summary: "Inherited rates yield 83.74% validation and 84.43% test, below the 90% gate."
verdicts: {"H-012": "contradicts"}
---

# Result: exp-012 — Full-range uniform bounded initialization

[Experiment and historical specification](../experiments/exp-012-uniform-initialization.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-bounded-drn-uniform-initialization-20260826-v1](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1) | complete: 2 | 2 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-bounded-drn-uniform-initialization-20260826-v1

Saved summary: [summary.json](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1/analysis/summary.json). Generated `2026-08-26T08:30:45.328583+00:00`; recorded state `reviewed`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| uniform-train-seed17 | 1 | 1 | 0 | 0 | 0 | True |
| uniform-selected-test-seed17 | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 2 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1/analysis/summary.json), [analysis/review.json](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1/analysis/review.json), [analysis/final.json](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1/analysis/final.json).

## Observations

Inherited rates yield 83.74% validation and 84.43% test, below the 90% gate.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/uniform-selected-test-seed17/20260826T073923.827607Z-12ac8ab1-85bfc09b](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1/runs/uniform-selected-test-seed17/20260826T073923.827607Z-12ac8ab1-85bfc09b/result.json) | complete | evaluation accuracy: 84.43% |
| [runs/uniform-train-seed17/20260826T073537.251122Z-15475b6e-d2050584](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1/runs/uniform-train-seed17/20260826T073537.251122Z-15475b6e-d2050584/result.json) | complete | selected accuracy: 83.74%; terminal accuracy: 83.74% |


Historical source: [mnist-bounded-drn-uniform-initialization-20260826-v1](../../../../../docs/experimental_manifest.md#mnist-bounded-drn-uniform-initialization-20260826-v1).

<details>

<summary>Historical record: mnist-bounded-drn-uniform-initialization-20260826-v1</summary>


### [mnist-bounded-drn-uniform-initialization-20260826-v1](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1)

**Bounded DRN training from a full-range uniform initialization**

- **Finished:** 2026-08-26
- **Evidence class:** `exploratory`
- **Outcome:** refuted
- **Interpretation:** The fixed ten-epoch hypothesis was refuted: full-range uniform initialization reached 83.74% validation and 84.43% test accuracy with the inherited [0.008,0.005] learning rates. Same-cohort replay indicates those rates are severely scale-mismatched, especially for W1, so this is not an accuracy ceiling for uniform initialization.
- **Details:** [human report](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1/analysis/report.md) · [machine summary](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1/analysis/summary.json) · [local study folder](../../../../../results/mnist-bounded-drn-uniform-initialization-20260826-v1)


</details>

## Interpretation and limitations

The declared ten-epoch hypothesis fails at these rates. Conductance-scale mismatch leaves an optimization explanation open; this is not an expressivity ceiling.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Interpret the rate-screen successor before claiming a limit from full-range uniform initialization.
