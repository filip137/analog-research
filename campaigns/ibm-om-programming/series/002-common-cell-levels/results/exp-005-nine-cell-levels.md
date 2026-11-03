---
experiment: "exp-005"
evidence: "imported-summary"
summary: "Held-out support is 90.16%, but persistent nearest-code correctness is only 28.5%."
verdicts: {"H-005": "contradicts"}
---

# Result: exp-005 — Array-wide nine-level physical-cell programming

[Experiment and historical specification](../experiments/exp-005-nine-cell-levels.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [ibm-om-raw-active-common-cell-9-canary-20260826-v1](../../../../../results/ibm-om-raw-active-common-cell-9-canary-20260826-v1) | complete: 1, failed: 1 | 1 |
| [ibm-om-raw-active-common-cell-9-program-verify-20260826-v1](../../../../../results/ibm-om-raw-active-common-cell-9-program-verify-20260826-v1) | complete: 2 | 2 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: ibm-om-raw-active-common-cell-9-canary-20260826-v1

Saved summary: [summary.json](../../../../../results/ibm-om-raw-active-common-cell-9-canary-20260826-v1/analysis/summary.json). Generated `2026-08-26T21:14:03.997659+00:00`; recorded state `ready_for_review`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| development-smoke | 1 | 1 | 1 | 0 | 0 | True |

Current metadata reconciliation: 1 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/ibm-om-raw-active-common-cell-9-canary-20260826-v1/analysis/report.md), [analysis/summary.json](../../../../../results/ibm-om-raw-active-common-cell-9-canary-20260826-v1/analysis/summary.json).

### Recorded study coverage: ibm-om-raw-active-common-cell-9-program-verify-20260826-v1

Saved summary: [summary.json](../../../../../results/ibm-om-raw-active-common-cell-9-program-verify-20260826-v1/analysis/summary.json). Generated `2026-08-26T21:15:27.565162+00:00`; recorded state `ready_for_review`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| development-assignment | 1 | 1 | 0 | 0 | 0 | True |
| heldout-assignment | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 2 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/ibm-om-raw-active-common-cell-9-program-verify-20260826-v1/analysis/report.md), [analysis/summary.json](../../../../../results/ibm-om-raw-active-common-cell-9-program-verify-20260826-v1/analysis/summary.json).

## Observations

Held-out support is 90.16%, but persistent nearest-code correctness is only 28.5%.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/development-smoke/20260826T211355.054834Z-dea314b9-893df325](../../../../../results/ibm-om-raw-active-common-cell-9-canary-20260826-v1/runs/development-smoke/20260826T211355.054834Z-dea314b9-893df325/result.json) | complete | persistent nearest-code correctness: 33.33%; exact-support apparent acceptance: 100.00%; all-nine population support: 90.00% |
| [runs/development-assignment/20260826T211422.941212Z-b9fe3912-3272c79c](../../../../../results/ibm-om-raw-active-common-cell-9-program-verify-20260826-v1/runs/development-assignment/20260826T211422.941212Z-b9fe3912-3272c79c/result.json) | complete | persistent nearest-code correctness: 28.62%; exact-support apparent acceptance: 99.40%; all-nine population support: 90.00% |
| [runs/heldout-assignment/20260826T211458.968593Z-27043ab0-934c3ea6](../../../../../results/ibm-om-raw-active-common-cell-9-program-verify-20260826-v1/runs/heldout-assignment/20260826T211458.968593Z-27043ab0-934c3ea6/result.json) | complete | persistent nearest-code correctness: 28.50%; exact-support apparent acceptance: 100.00%; all-nine population support: 90.16% |


## Interpretation and limitations

The ≥89% support gate passes; the ≥90% persistent correctness gate fails, so the conjunctive hypothesis is contradicted. Apparent acceptance is 100% on exact-support validation trajectories, not evidence that nine codes are distinguishable. Apparent acceptance ambiguity is 38.25%. These are fitted-model characterization outcomes, not network metrics.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Reject nine persistently distinguishable codes under the tested controller. Any changed controller or codebook is a new hypothesis.
