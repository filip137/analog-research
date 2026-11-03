---
experiment: "exp-004"
evidence: "imported-summary"
summary: "QAT reaches 91.00% mean apparent accuracy, but its +0.71-point advantage misses the gate."
verdicts: {"H-004": "inconclusive"}
---

# Result: exp-004 — Shared RESET-relative nine-level QAT

[Experiment and historical specification](../experiments/exp-004-shared-quantization.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1) | complete: 24, failed: 2 | 24 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1

Saved summary: [summary.json](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/analysis/summary.json). Generated `2026-08-25T14:39:48.831600+00:00`; recorded state `reviewed`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| train-zero-update-quantized | 1 | 1 | 0 | 0 | 0 | True |
| train-clean-bptt-quantized-deploy | 1 | 1 | 0 | 0 | 0 | True |
| train-continuous-shared-target-hwa | 1 | 1 | 0 | 0 | 0 | True |
| train-quantized-qat | 1 | 1 | 2 | 0 | 0 | True |
| deploy-zero-update-heldout | 5 | 5 | 0 | 0 | 0 | True |
| deploy-clean-bptt-heldout | 5 | 5 | 0 | 0 | 0 | True |
| deploy-continuous-hwa-heldout | 5 | 5 | 0 | 0 | 0 | True |
| deploy-quantized-qat-heldout | 5 | 5 | 0 | 0 | 0 | True |

Current metadata reconciliation: 24 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/primary_analysis.md](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/analysis/primary_analysis.md), [analysis/report.md](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/analysis/summary.json), [analysis/review.json](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/analysis/review.json), [analysis/final.json](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/analysis/final.json).

## Observations

QAT reaches 91.00% mean apparent accuracy, but its +0.71-point advantage misses the gate.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/deploy-clean-bptt-heldout/20260825T122720.505193Z-cb19fb9a-888b18a3](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-clean-bptt-heldout/20260825T122720.505193Z-cb19fb9a-888b18a3/result.json) | complete | evaluation accuracy: 65.20% |
| [runs/deploy-clean-bptt-heldout/20260825T122746.670486Z-f932c19f-e0cb4c72](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-clean-bptt-heldout/20260825T122746.670486Z-f932c19f-e0cb4c72/result.json) | complete | evaluation accuracy: 84.97% |
| [runs/deploy-clean-bptt-heldout/20260825T122834.003598Z-803a365e-8eb869e9](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-clean-bptt-heldout/20260825T122834.003598Z-803a365e-8eb869e9/result.json) | complete | evaluation accuracy: 74.93% |
| [runs/deploy-clean-bptt-heldout/20260825T122858.975491Z-fdab35eb-cf18605d](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-clean-bptt-heldout/20260825T122858.975491Z-fdab35eb-cf18605d/result.json) | complete | evaluation accuracy: 75.84% |
| [runs/deploy-clean-bptt-heldout/20260825T122934.429753Z-0fcae610-6ee21882](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-clean-bptt-heldout/20260825T122934.429753Z-0fcae610-6ee21882/result.json) | complete | evaluation accuracy: 83.33% |
| [runs/deploy-continuous-hwa-heldout/20260825T122720.505192Z-f88239db-4cf4dfa2](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-continuous-hwa-heldout/20260825T122720.505192Z-f88239db-4cf4dfa2/result.json) | complete | evaluation accuracy: 90.17% |
| [runs/deploy-continuous-hwa-heldout/20260825T122746.711932Z-b4001e65-fd9d68df](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-continuous-hwa-heldout/20260825T122746.711932Z-b4001e65-fd9d68df/result.json) | complete | evaluation accuracy: 87.88% |
| [runs/deploy-continuous-hwa-heldout/20260825T122833.988662Z-9db36c79-391306b6](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-continuous-hwa-heldout/20260825T122833.988662Z-9db36c79-391306b6/result.json) | complete | evaluation accuracy: 90.09% |
| [runs/deploy-continuous-hwa-heldout/20260825T122858.949433Z-031b18d6-d8a65a1f](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-continuous-hwa-heldout/20260825T122858.949433Z-031b18d6-d8a65a1f/result.json) | complete | evaluation accuracy: 91.19% |
| [runs/deploy-continuous-hwa-heldout/20260825T122934.448332Z-66b24a95-69ae203a](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-continuous-hwa-heldout/20260825T122934.448332Z-66b24a95-69ae203a/result.json) | complete | evaluation accuracy: 92.12% |
| [runs/deploy-quantized-qat-heldout/20260825T122720.505194Z-cb19fb9a-ca9200f7](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-quantized-qat-heldout/20260825T122720.505194Z-cb19fb9a-ca9200f7/result.json) | complete | evaluation accuracy: 92.38% |
| [runs/deploy-quantized-qat-heldout/20260825T122746.695098Z-f932c19f-5e5f4260](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-quantized-qat-heldout/20260825T122746.695098Z-f932c19f-5e5f4260/result.json) | complete | evaluation accuracy: 93.23% |
| [runs/deploy-quantized-qat-heldout/20260825T122834.024678Z-803a365e-2c4dca24](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-quantized-qat-heldout/20260825T122834.024678Z-803a365e-2c4dca24/result.json) | complete | evaluation accuracy: 88.56% |
| [runs/deploy-quantized-qat-heldout/20260825T122858.975320Z-fdab35eb-10d80598](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-quantized-qat-heldout/20260825T122858.975320Z-fdab35eb-10d80598/result.json) | complete | evaluation accuracy: 87.34% |
| [runs/deploy-quantized-qat-heldout/20260825T122934.442853Z-0fcae610-b56f8269](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-quantized-qat-heldout/20260825T122934.442853Z-0fcae610-b56f8269/result.json) | complete | evaluation accuracy: 93.49% |
| [runs/deploy-zero-update-heldout/20260825T122720.495095Z-cb19fb9a-aa3c04ff](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-zero-update-heldout/20260825T122720.495095Z-cb19fb9a-aa3c04ff/result.json) | complete | evaluation accuracy: 63.61% |
| [runs/deploy-zero-update-heldout/20260825T122746.673270Z-f932c19f-2e393879](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-zero-update-heldout/20260825T122746.673270Z-f932c19f-2e393879/result.json) | complete | evaluation accuracy: 85.88% |
| [runs/deploy-zero-update-heldout/20260825T122833.949280Z-803a365e-7b0b38c8](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-zero-update-heldout/20260825T122833.949280Z-803a365e-7b0b38c8/result.json) | complete | evaluation accuracy: 75.14% |
| [runs/deploy-zero-update-heldout/20260825T122858.968904Z-fdab35eb-f649d4f5](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-zero-update-heldout/20260825T122858.968904Z-fdab35eb-f649d4f5/result.json) | complete | evaluation accuracy: 77.55% |
| [runs/deploy-zero-update-heldout/20260825T122934.429765Z-0fcae610-b5a7165a](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/deploy-zero-update-heldout/20260825T122934.429765Z-0fcae610-b5a7165a/result.json) | complete | evaluation accuracy: 84.14% |
| [runs/train-clean-bptt-quantized-deploy/20260825T082122.803993Z-bd3bf2c3-57fee846](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/train-clean-bptt-quantized-deploy/20260825T082122.803993Z-bd3bf2c3-57fee846/result.json) | complete | selected student accuracy: 75.43%; initial student accuracy: 74.90%; terminal student accuracy: 75.01% |
| [runs/train-continuous-shared-target-hwa/20260825T082122.854153Z-1905696f-06466d93](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/train-continuous-shared-target-hwa/20260825T082122.854153Z-1905696f-06466d93/result.json) | complete | selected student accuracy: 92.67%; initial student accuracy: 81.81%; terminal student accuracy: 89.89% |
| [runs/train-quantized-qat/20260825T092317.972756Z-52034a40-acebee3d](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/train-quantized-qat/20260825T092317.972756Z-52034a40-acebee3d/result.json) | complete | selected student accuracy: 92.38%; initial student accuracy: 71.21%; terminal student accuracy: 89.19% |
| [runs/train-zero-update-quantized/20260825T081932.627202Z-167e1c61-a08e2c8d](../../../../../results/mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1/runs/train-zero-update-quantized/20260825T081932.627202Z-167e1c61-a08e2c8d/result.json) | complete | selected student accuracy: 74.90%; initial student accuracy: 74.90%; terminal student accuracy: 74.90% |


Historical source: [mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1](../../../../../docs/experimental_manifest.md#mnist-ibm-om-shared-reset-relative-quantized-hwa-20260824-v1).

- **Outcome:** mixed
- **Interpretation:** Shared RESET-relative deployment transferred strongly: nine-level QAT reached 91.00% mean apparent accuracy. Its +0.71-point paired advantage over continuous HWA did not meet the predeclared +2-point gate and its 95% interval crossed zero, so a QAT-specific benefit remains unresolved.

## Interpretation and limitations

The paired interval crosses zero and the declared +2-point gate fails. Strong RESET-relative transfer is supported; a QAT-specific advantage remains unresolved. Persistent endpoints remain diagnostic.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Retain strong shared-coordinate transfer while leaving QAT-specific advantage unresolved.
