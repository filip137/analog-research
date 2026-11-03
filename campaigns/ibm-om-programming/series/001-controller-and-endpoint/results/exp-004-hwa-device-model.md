---
experiment: "exp-004"
evidence: "imported-summary"
summary: "Both local characterization arms complete; the saved summary used metadata-only validation."
verdicts: {"H-004": "inconclusive"}
---

# Result: exp-004 — Focused cap-128 HWA endpoint prerequisite

[Experiment and historical specification](../experiments/exp-004-hwa-device-model.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [ibm-reram-om-hwa-device-model-20260822-v1](../../../../../results/ibm-reram-om-hwa-device-model-20260822-v1) | complete: 2 | 2 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: ibm-reram-om-hwa-device-model-20260822-v1

Saved summary: [summary.json](../../../../../results/ibm-reram-om-hwa-device-model-20260822-v1/analysis/summary.json). Generated `2026-08-22T22:14:07.821854+00:00`; recorded state `ready_for_review`; recorded validation mode `metadata_only`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| om-continuous | 1 | 1 | 0 | 0 | 0 | True |
| om-published | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 2 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/ibm-reram-om-hwa-device-model-20260822-v1/analysis/report.md), [analysis/summary.json](../../../../../results/ibm-reram-om-hwa-device-model-20260822-v1/analysis/summary.json).

## Observations

Both local characterization arms complete; the saved summary used metadata-only validation.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/om-continuous/20260822T220823.176990Z-195d0e8e-eae91b6f](../../../../../results/ibm-reram-om-hwa-device-model-20260822-v1/runs/om-continuous/20260822T220823.176990Z-195d0e8e-eae91b6f/result.json) | complete | See terminal result (no common accuracy field). |
| [runs/om-published/20260822T220823.154271Z-515b82b4-c82e2dc2](../../../../../results/ibm-reram-om-hwa-device-model-20260822-v1/runs/om-published/20260822T220823.154271Z-515b82b4-c82e2dc2/result.json) | complete | See terminal result (no common accuracy field). |


## Interpretation and limitations

This focused lower-start model is an HWA prerequisite, not the full four-population characterization. Its narrower coverage and metadata-only recorded check must not be promoted to full artifact verification.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Keep the focused endpoint model tied to its covered regimes and metadata-only historical validation; it does not replace the broad characterization.
