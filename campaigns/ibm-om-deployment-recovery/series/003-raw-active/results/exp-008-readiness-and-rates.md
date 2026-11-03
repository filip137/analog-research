---
experiment: "exp-008"
evidence: "imported-summary"
summary: "The local raw-active canary and six rate-screen training bundles are terminal."
verdicts: {"H-008": "supports"}
---

# Result: exp-008 — Raw-active canary and learning-rate screen

[Experiment and historical specification](../experiments/exp-008-readiness-and-rates.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-ibm-om-raw-active-p90-canary-20260826-v1](../../../../../results/mnist-ibm-om-raw-active-p90-canary-20260826-v1) | complete: 2 | 2 |
| [mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1) | complete: 6 | 6 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1

Saved summary: [summary.json](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1/analysis/summary.json). Generated `2026-08-26T18:56:14.638635+00:00`; recorded state `ready_for_review`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| continuous-x0p25 | 1 | 1 | 0 | 0 | 0 | True |
| continuous-x1 | 1 | 1 | 0 | 0 | 0 | True |
| continuous-x4 | 1 | 1 | 0 | 0 | 0 | True |
| quantized-x0p25 | 1 | 1 | 0 | 0 | 0 | True |
| quantized-x1 | 1 | 1 | 0 | 0 | 0 | True |
| quantized-x4 | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 6 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1/analysis/report.md), [analysis/selection_report.md](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1/analysis/selection_report.md), [analysis/summary.json](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1/analysis/summary.json).

## Observations

The local raw-active canary and six rate-screen training bundles are terminal.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/zero-update-quantized-canary/20260826T184256.749327Z-cc726bf6-5edea847](../../../../../results/mnist-ibm-om-raw-active-p90-canary-20260826-v1/runs/zero-update-quantized-canary/20260826T184256.749327Z-cc726bf6-5edea847/result.json) | complete | selected student accuracy: 25.00%; initial student accuracy: 25.00%; terminal student accuracy: 25.00% |
| [runs/zero-update-quantized-canary/20260826T184505.702546Z-cc726bf6-e3888f12](../../../../../results/mnist-ibm-om-raw-active-p90-canary-20260826-v1/runs/zero-update-quantized-canary/20260826T184505.702546Z-cc726bf6-e3888f12/result.json) | complete | selected student accuracy: 25.00%; initial student accuracy: 25.00%; terminal student accuracy: 25.00% |
| [runs/continuous-x0p25/20260826T185159.611428Z-9905f791-106941d1](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1/runs/continuous-x0p25/20260826T185159.611428Z-9905f791-106941d1/result.json) | complete | selected student accuracy: 48.83%; initial student accuracy: 29.10%; terminal student accuracy: 48.83% |
| [runs/continuous-x1/20260826T185159.613985Z-ff892c08-b2ad6f5f](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1/runs/continuous-x1/20260826T185159.613985Z-ff892c08-b2ad6f5f/result.json) | complete | selected student accuracy: 64.36%; initial student accuracy: 29.10%; terminal student accuracy: 64.36% |
| [runs/continuous-x4/20260826T185159.611322Z-9fc94c25-d4d50173](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1/runs/continuous-x4/20260826T185159.611322Z-9fc94c25-d4d50173/result.json) | complete | selected student accuracy: 79.25%; initial student accuracy: 29.10%; terminal student accuracy: 79.25% |
| [runs/quantized-x0p25/20260826T185159.610829Z-925fdd7a-918d882d](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1/runs/quantized-x0p25/20260826T185159.610829Z-925fdd7a-918d882d/result.json) | complete | selected student accuracy: 55.13%; initial student accuracy: 30.57%; terminal student accuracy: 55.13% |
| [runs/quantized-x1/20260826T185159.610829Z-c7973b86-5a7a0f43](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1/runs/quantized-x1/20260826T185159.610829Z-c7973b86-5a7a0f43/result.json) | complete | selected student accuracy: 72.31%; initial student accuracy: 30.57%; terminal student accuracy: 72.31% |
| [runs/quantized-x4/20260826T185159.587496Z-31efca39-773b92c2](../../../../../results/mnist-ibm-om-raw-active-p90-learning-rate-screen-20260826-v1/runs/quantized-x4/20260826T185159.587496Z-31efca39-773b92c2/result.json) | complete | selected student accuracy: 82.47%; initial student accuracy: 30.57%; terminal student accuracy: 82.47% |


## Interpretation and limitations

All six development rate-screen runs have positive recorded acceptance-gate improvements and complete local terminal bundles. This supports a viable mapped-HWA/QAT rate in the development screen. The canary, rate search and held-out successor are different stages, not additional independent replicates.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use the frozen rates and checkpoints recorded by the held-out successor; do not retune against held-out outcomes.
