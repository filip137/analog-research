---
experiment: "exp-015"
evidence: "imported-summary"
summary: "Expanded SGD reaches 96.18% validation and 96.48% test, meeting its stated gates."
verdicts: {"H-015": "supports"}
---

# Result: exp-015 — Expanded SGD learning-rate control

[Experiment and historical specification](../experiments/exp-015-expanded-sgd.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1) | complete: 13 | 13 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1

Saved summary: [summary.json](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/analysis/summary.json). Generated `2026-08-26T12:46:43.723628+00:00`; recorded state `ready_for_review`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| sgd-w1-0p512-w2-0p02 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-0p512-w2-0p04 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-0p512-w2-0p08 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-1p024-w2-0p02 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-1p024-w2-0p04 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-1p024-w2-0p08 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-2p048-w2-0p02 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-2p048-w2-0p04 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-2p048-w2-0p08 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-4p096-w2-0p02 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-4p096-w2-0p04 | 1 | 1 | 0 | 0 | 0 | True |
| sgd-w1-4p096-w2-0p08 | 1 | 1 | 0 | 0 | 0 | True |
| validation-selected-test | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 13 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/analysis/summary.json).

## Observations

Expanded SGD reaches 96.18% validation and 96.48% test, meeting its stated gates.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/sgd-w1-0p512-w2-0p02/20260826T120321.863224Z-263cda6d-b73dddb5](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-0p512-w2-0p02/20260826T120321.863224Z-263cda6d-b73dddb5/result.json) | complete | selected accuracy: 94.82%; terminal accuracy: 94.82% |
| [runs/sgd-w1-0p512-w2-0p04/20260826T120654.325143Z-ebf3dfbf-e4065141](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-0p512-w2-0p04/20260826T120654.325143Z-ebf3dfbf-e4065141/result.json) | complete | selected accuracy: 94.86%; terminal accuracy: 94.86% |
| [runs/sgd-w1-0p512-w2-0p08/20260826T121026.623182Z-523596c4-d8ec76d0](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-0p512-w2-0p08/20260826T121026.623182Z-523596c4-d8ec76d0/result.json) | complete | selected accuracy: 95.08%; terminal accuracy: 95.08% |
| [runs/sgd-w1-1p024-w2-0p02/20260826T121400.646316Z-2b9ea733-f9693661](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-1p024-w2-0p02/20260826T121400.646316Z-2b9ea733-f9693661/result.json) | complete | selected accuracy: 95.30%; terminal accuracy: 95.30% |
| [runs/sgd-w1-1p024-w2-0p04/20260826T121732.983285Z-de20ef9f-407b0d6b](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-1p024-w2-0p04/20260826T121732.983285Z-de20ef9f-407b0d6b/result.json) | complete | selected accuracy: 95.46%; terminal accuracy: 95.46% |
| [runs/sgd-w1-1p024-w2-0p08/20260826T122106.286135Z-443d3935-ca7e89f7](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-1p024-w2-0p08/20260826T122106.286135Z-443d3935-ca7e89f7/result.json) | complete | selected accuracy: 95.80%; terminal accuracy: 95.80% |
| [runs/sgd-w1-2p048-w2-0p02/20260826T122440.232965Z-41a8b3f1-50afaf12](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-2p048-w2-0p02/20260826T122440.232965Z-41a8b3f1-50afaf12/result.json) | complete | selected accuracy: 95.64%; terminal accuracy: 95.64% |
| [runs/sgd-w1-2p048-w2-0p04/20260826T122811.585916Z-04346a98-682b3c44](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-2p048-w2-0p04/20260826T122811.585916Z-04346a98-682b3c44/result.json) | complete | selected accuracy: 95.92%; terminal accuracy: 95.92% |
| [runs/sgd-w1-2p048-w2-0p08/20260826T123142.845379Z-4f4f24d8-b9c29ad4](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-2p048-w2-0p08/20260826T123142.845379Z-4f4f24d8-b9c29ad4/result.json) | complete | selected accuracy: 96.18%; terminal accuracy: 96.18% |
| [runs/sgd-w1-4p096-w2-0p02/20260826T123515.610964Z-54cffe59-a64104db](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-4p096-w2-0p02/20260826T123515.610964Z-54cffe59-a64104db/result.json) | complete | selected accuracy: 95.84%; terminal accuracy: 95.84% |
| [runs/sgd-w1-4p096-w2-0p04/20260826T123847.500084Z-6647f070-637f0b04](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-4p096-w2-0p04/20260826T123847.500084Z-6647f070-637f0b04/result.json) | complete | selected accuracy: 96.08%; terminal accuracy: 96.08% |
| [runs/sgd-w1-4p096-w2-0p08/20260826T115818.748294Z-d29bbf27-3997d958](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/sgd-w1-4p096-w2-0p08/20260826T115818.748294Z-d29bbf27-3997d958/result.json) | complete | selected accuracy: 96.04%; terminal accuracy: 96.06% |
| [runs/validation-selected-test/20260826T124612.779828Z-98cc1853-4a68f443](../../../../../results/mnist-om-cell-bounds-uniform-fp32-sgd-expanded-lr-20260826-v1/runs/validation-selected-test/20260826T124612.779828Z-98cc1853-4a68f443/result.json) | complete | evaluation accuracy: 96.48% |


## Interpretation and limitations

The validation gap to the prior Adam optimum shrinks from 1.94 to 0.58 points, a 1.36-point reduction. The original optimizer gap partly reflected grid truncation; the remaining difference is single-seed and not a population conclusion.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Retain the narrowed optimizer gap; multiple matched seeds are needed to establish a persistent advantage.
