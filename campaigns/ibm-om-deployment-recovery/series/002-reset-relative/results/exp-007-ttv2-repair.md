---
experiment: "exp-007"
evidence: "imported-summary"
summary: "The repaired ideal-fast arm ends at 60.02%, above old Tiki-Taka but below the 92.74% start."
verdicts: {"H-007": "inconclusive"}
---

# Result: exp-007 — Canonical-axis TTv2 repair

[Experiment and historical specification](../experiments/exp-007-ttv2-repair.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-ibm-om-reset-relative-ttv2-repair-20260825-v1](../../../../../results/mnist-ibm-om-reset-relative-ttv2-repair-20260825-v1) | complete: 1 | 1 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-ibm-om-reset-relative-ttv2-repair-20260825-v1

Saved summary: [summary.json](../../../../../results/mnist-ibm-om-reset-relative-ttv2-repair-20260825-v1/analysis/summary.json). Generated `2026-08-26T06:06:40.475870+00:00`; recorded state `ready_for_review`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| ttv2-ideal-budget-0p1 | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 1 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-ibm-om-reset-relative-ttv2-repair-20260825-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-ibm-om-reset-relative-ttv2-repair-20260825-v1/analysis/summary.json).

### Recorded recovery source identities

These are stored source identities, checked for agreement in the available terminal metadata; the source bytes were not rehashed.

- source_deployment_sha256: `0b99f4dd73841a891993708bdace0ab508864e2e60e56620c165e3df99f006cd`.
- source_weights_sha256: `537681598b887d4d63c6429c7f7bee3a7082041215c09d6cb3c25947452f311a`.

## Observations

The repaired ideal-fast arm ends at 60.02%, above old Tiki-Taka but below the 92.74% start.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/ttv2-ideal-budget-0p1/20260825T181012.724825Z-8342ae10-348cccee](../../../../../results/mnist-ibm-om-reset-relative-ttv2-repair-20260825-v1/runs/ttv2-ideal-budget-0p1/20260825T181012.724825Z-8342ae10-348cccee/result.json) | complete | terminal apparent validation: 60.02%; initial apparent validation: 92.74%; slow write requests: 15880; apparent logical sign flips: 496 |


## Interpretation and limitations

The repair improves on the historical 10.06% ideal-fast cap-0.1 outcome but still loses 32.72 points from deployment. Algorithm repair and useful scientific recovery are different claims.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

The repair still fails to recover the saved network. Diagnose its update behavior before assigning a larger recovery study.
