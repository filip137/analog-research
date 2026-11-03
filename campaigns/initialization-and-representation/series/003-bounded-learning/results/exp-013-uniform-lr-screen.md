---
experiment: "exp-013"
evidence: "imported-summary"
summary: "Three available rate candidates reach 93.06–94.10% validation; overall planned coverage is partial."
verdicts: {"H-013": "supports"}
---

# Result: exp-013 — Uniform-initialization learning-rate screen

[Experiment and historical specification](../experiments/exp-013-uniform-lr-screen.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-bounded-drn-uniform-learning-rate-screen-20260826-v1](../../../../../results/mnist-bounded-drn-uniform-learning-rate-screen-20260826-v1) | complete: 3 | 3 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

Three available rate candidates reach 93.06–94.10% validation; overall planned coverage is partial.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/w1-0p128-w2-0p01/20260826T083754.523249Z-54b5d0c4-073b4a5c](../../../../../results/mnist-bounded-drn-uniform-learning-rate-screen-20260826-v1/runs/w1-0p128-w2-0p01/20260826T083754.523249Z-54b5d0c4-073b4a5c/result.json) | complete | selected accuracy: 93.06%; terminal accuracy: 93.06% |
| [runs/w1-0p128-w2-0p02/20260826T084134.012659Z-d1b1de73-ba920aff](../../../../../results/mnist-bounded-drn-uniform-learning-rate-screen-20260826-v1/runs/w1-0p128-w2-0p02/20260826T084134.012659Z-d1b1de73-ba920aff/result.json) | complete | selected accuracy: 93.38%; terminal accuracy: 93.38% |
| [runs/w1-0p256-w2-0p01/20260826T084516.021838Z-019ff1e7-40696089](../../../../../results/mnist-bounded-drn-uniform-learning-rate-screen-20260826-v1/runs/w1-0p256-w2-0p01/20260826T084516.021838Z-019ff1e7-40696089/result.json) | complete | selected accuracy: 94.10%; terminal accuracy: 94.10% |


## Interpretation and limitations

All three observed candidates exceed 90% validation and improve over the 83.74% inherited-rate reference by more than five points, supporting the stated feasibility claim within observed coverage. Three other training candidates and the selected-test run are absent locally. The full-grid optimum and test performance remain unresolved.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Retain the observed feasibility result and partial coverage. Completing the remaining screen/test cases requires a separate execution assignment.
