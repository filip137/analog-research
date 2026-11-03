---
experiment: "exp-006"
evidence: "imported-summary"
summary: "p99/cap-0.1 ends at 92.78%: +3.42 points over direct pulses, +0.04 over initialization."
verdicts: {"H-006": "supports"}
---

# Result: exp-006 — Direct-pulse gradient-threshold follow-up

[Experiment and historical specification](../experiments/exp-006-gradient-thresholds.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1) | complete: 9 | 9 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1

Saved summary: [summary.json](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/analysis/summary.json). Generated `2026-08-26T06:06:40.567451+00:00`; recorded state `ready_for_review`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| p90-budget-0p1 | 1 | 1 | 0 | 0 | 0 | True |
| p90-budget-1 | 1 | 1 | 0 | 0 | 0 | True |
| p90-budget-4 | 1 | 1 | 0 | 0 | 0 | True |
| p95-budget-0p1 | 1 | 1 | 0 | 0 | 0 | True |
| p95-budget-1 | 1 | 1 | 0 | 0 | 0 | True |
| p95-budget-4 | 1 | 1 | 0 | 0 | 0 | True |
| p99-budget-0p1 | 1 | 1 | 0 | 0 | 0 | True |
| p99-budget-1 | 1 | 1 | 0 | 0 | 0 | True |
| p99-budget-4 | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 9 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/analysis/summary.json).

### Recorded recovery source identities

These are stored source identities, checked for agreement in the available terminal metadata; the source bytes were not rehashed.

- source_deployment_sha256: `0b99f4dd73841a891993708bdace0ab508864e2e60e56620c165e3df99f006cd`.
- source_weights_sha256: `537681598b887d4d63c6429c7f7bee3a7082041215c09d6cb3c25947452f311a`.

## Observations

p99/cap-0.1 ends at 92.78%: +3.42 points over direct pulses, +0.04 over initialization.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/p90-budget-0p1/20260825T163034.315452Z-47fcc4be-d15827b9](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/runs/p90-budget-0p1/20260825T163034.315452Z-47fcc4be-d15827b9/result.json) | complete | terminal apparent validation: 89.12%; initial apparent validation: 92.74%; slow write requests: 9605; apparent logical sign flips: 683 |
| [runs/p90-budget-1/20260825T163504.390145Z-1e482d40-650a8c3a](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/runs/p90-budget-1/20260825T163504.390145Z-1e482d40-650a8c3a/result.json) | complete | terminal apparent validation: 85.26%; initial apparent validation: 92.74%; slow write requests: 95126; apparent logical sign flips: 2321 |
| [runs/p90-budget-4/20260825T163949.572643Z-a45b947b-91089aca](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/runs/p90-budget-4/20260825T163949.572643Z-a45b947b-91089aca/result.json) | complete | terminal apparent validation: 85.92%; initial apparent validation: 92.74%; slow write requests: 246455; apparent logical sign flips: 3039 |
| [runs/p95-budget-0p1/20260825T164434.805947Z-a167f4df-57d6afbf](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/runs/p95-budget-0p1/20260825T164434.805947Z-a167f4df-57d6afbf/result.json) | complete | terminal apparent validation: 90.88%; initial apparent validation: 92.74%; slow write requests: 6049; apparent logical sign flips: 494 |
| [runs/p95-budget-1/20260825T164849.971183Z-0916ef10-6ca52cae](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/runs/p95-budget-1/20260825T164849.971183Z-0916ef10-6ca52cae/result.json) | complete | terminal apparent validation: 85.62%; initial apparent validation: 92.74%; slow write requests: 63908; apparent logical sign flips: 1854 |
| [runs/p95-budget-4/20260825T165335.145512Z-9b2ee71d-7b73c7fb](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/runs/p95-budget-4/20260825T165335.145512Z-9b2ee71d-7b73c7fb/result.json) | complete | terminal apparent validation: 84.42%; initial apparent validation: 92.74%; slow write requests: 198046; apparent logical sign flips: 2613 |
| [runs/p99-budget-0p1/20260825T165820.323502Z-35cc41ab-e0f0e3d7](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/runs/p99-budget-0p1/20260825T165820.323502Z-35cc41ab-e0f0e3d7/result.json) | complete | terminal apparent validation: 92.78%; initial apparent validation: 92.74%; slow write requests: 2326; apparent logical sign flips: 261 |
| [runs/p99-budget-1/20260825T170235.503755Z-438fe9b0-1d58324b](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/runs/p99-budget-1/20260825T170235.503755Z-438fe9b0-1d58324b/result.json) | complete | terminal apparent validation: 84.76%; initial apparent validation: 92.74%; slow write requests: 32276; apparent logical sign flips: 1255 |
| [runs/p99-budget-4/20260825T170705.705005Z-72e7215d-9a712f87](../../../../../results/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1/runs/p99-budget-4/20260825T170705.705005Z-72e7215d-9a712f87/result.json) | complete | terminal apparent validation: 83.18%; initial apparent validation: 92.74%; slow write requests: 81132; apparent logical sign flips: 1723 |

<a id="source-ibm_om_reset_relative_direct_gradient_thresholds-grid-and-evidence-boundary"></a>

## Grid and evidence boundary

The one-seed grid crosses p90, p95, and p99 with caps 0.1, 1, and 4. Every arm
runs ten epochs from source deployment SHA-256
`0b99f4dd73841a891993708bdace0ab508864e2e60e56620c165e3df99f006cd`.
Epoch 10 is primary; the best intermediate epoch is descriptive only.

Each run retains the per-parameter thresholds, raw-gradient sample digests,
strict comparison rule, dynamic eligibility counts, frozen-scale provenance,
pulse ledgers, apparent and persistent endpoint states, and exact resume RNG
state. The experiment remains on the development array and validation split.
No additional seed or held-out deployment is released by this study.


## Interpretation and limitations

The p99/cap-0.1 arm improves terminal apparent validation from 89.36% to 92.78% relative to the matched direct-pulse control, with 2,326 versus 15,880 slow write requests and 261 versus 953 apparent logical sign flips. It meets all three directional parts of the declared hypothesis in this development seed. However, the gain over the unchanged 92.74% deployment is only two of 5,000 validation examples: this supports damage suppression, not substantial or replicated recovery. The stored deployment and weight digests and direct probability scale agree with the original control.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Retain p99 as damage suppression relative to unthresholded writes; independent seeds and meaningful recovery headroom are needed for a stronger claim.
