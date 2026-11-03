---
experiment: "exp-014"
evidence: "imported-summary"
summary: "Adam reaches 96.76% validation and 96.94% test; its 97% validation gate fails."
verdicts: {"H-014": "contradicts"}
---

# Result: exp-014 — SGD and Adam within heterogeneous OM cell bounds

[Experiment and historical specification](../experiments/exp-014-cell-bounds-optimizers.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1) | complete: 12 | 12 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1

Saved summary: [summary.json](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/analysis/summary.json). Generated `2026-08-26T10:10:02.584292+00:00`; recorded state `ready_for_review`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| sgd-w1-0p128-w2-0p01 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-0p128-w2-0p02 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-0p256-w2-0p01 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-0p256-w2-0p02 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-0p512-w2-0p01 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-0p512-w2-0p02 | 1 | 1 | 0 | 0 | 0 | True |
| adam-lr-1e4 | 1 | 1 | 0 | 0 | 0 | True |
| adam-lr-3e4 | 1 | 1 | 0 | 0 | 0 | True |
| adam-lr-1e3 | 1 | 1 | 0 | 0 | 0 | True |
| adam-lr-3e3 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-validation-selected-test | 1 | 1 | 0 | 0 | 0 | True |
| adam-validation-selected-test | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 12 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/analysis/summary.json).

## Observations

Adam reaches 96.76% validation and 96.94% test; its 97% validation gate fails.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/adam-lr-1e3/20260826T095456.715363Z-da82543e-26518ad1](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/adam-lr-1e3/20260826T095456.715363Z-da82543e-26518ad1/result.json) | complete | selected accuracy: 96.76%; terminal accuracy: 96.76% |
| [runs/adam-lr-1e4/20260826T094746.805220Z-de6dfb9c-72256431](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/adam-lr-1e4/20260826T094746.805220Z-de6dfb9c-72256431/result.json) | complete | selected accuracy: 94.90%; terminal accuracy: 94.90% |
| [runs/adam-lr-3e3/20260826T095831.730373Z-53ec153f-a66e4404](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/adam-lr-3e3/20260826T095831.730373Z-53ec153f-a66e4404/result.json) | complete | selected accuracy: 96.66%; terminal accuracy: 96.18% |
| [runs/adam-lr-3e4/20260826T095120.812841Z-592eed7f-9d27dddc](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/adam-lr-3e4/20260826T095120.812841Z-592eed7f-9d27dddc/result.json) | complete | selected accuracy: 96.26%; terminal accuracy: 96.26% |
| [runs/adam-validation-selected-test/20260826T100459.962611Z-98cc1853-97e3004a](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/adam-validation-selected-test/20260826T100459.962611Z-98cc1853-97e3004a/result.json) | complete | evaluation accuracy: 96.94% |
| [runs/sgd-validation-selected-test/20260826T100438.841446Z-98cc1853-9d504e87](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/sgd-validation-selected-test/20260826T100438.841446Z-98cc1853-9d504e87/result.json) | complete | evaluation accuracy: 95.33% |
| [runs/sgd-w1-0p128-w2-0p01/20260826T092627.285222Z-393e223a-8916bcc9](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/sgd-w1-0p128-w2-0p01/20260826T092627.285222Z-393e223a-8916bcc9/result.json) | complete | selected accuracy: 92.92%; terminal accuracy: 92.92% |
| [runs/sgd-w1-0p128-w2-0p02/20260826T093001.574437Z-c432aad8-6932d438](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/sgd-w1-0p128-w2-0p02/20260826T093001.574437Z-c432aad8-6932d438/result.json) | complete | selected accuracy: 93.18%; terminal accuracy: 93.18% |
| [runs/sgd-w1-0p256-w2-0p01/20260826T093334.562972Z-80b9eb26-c7847df0](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/sgd-w1-0p256-w2-0p01/20260826T093334.562972Z-80b9eb26-c7847df0/result.json) | complete | selected accuracy: 93.98%; terminal accuracy: 93.98% |
| [runs/sgd-w1-0p256-w2-0p02/20260826T093705.909601Z-2aa3ef3f-d244043a](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/sgd-w1-0p256-w2-0p02/20260826T093705.909601Z-2aa3ef3f-d244043a/result.json) | complete | selected accuracy: 94.14%; terminal accuracy: 94.14% |
| [runs/sgd-w1-0p512-w2-0p01/20260826T094040.208720Z-3701c68a-e72ecb30](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/sgd-w1-0p512-w2-0p01/20260826T094040.208720Z-3701c68a-e72ecb30/result.json) | complete | selected accuracy: 94.56%; terminal accuracy: 94.56% |
| [runs/sgd-w1-0p512-w2-0p02/20260826T094414.695046Z-263cda6d-25be4ea7](../../../../../results/mnist-om-cell-bounds-uniform-fp32-optimizer-reference-20260826-v1/runs/sgd-w1-0p512-w2-0p02/20260826T094414.695046Z-263cda6d-25be4ea7/result.json) | complete | selected accuracy: 94.82%; terminal accuracy: 94.82% |


## Interpretation and limitations

Adam outperforms the original limited SGD grid (94.82% validation, 95.33% test), but the boundary optimum motivates the expanded SGD control. These ideal bounds-only references do not include programming loss.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Compare with the expanded SGD grid before treating the initial Adam advantage as algorithmic.
