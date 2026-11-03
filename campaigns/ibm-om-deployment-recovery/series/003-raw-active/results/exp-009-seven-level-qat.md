---
experiment: "exp-009"
evidence: "imported-summary"
summary: "QAT beats continuous HWA by +2.35 points, yet mean held-out apparent accuracy is 52.66%."
verdicts: {"H-009": "supports"}
---

# Result: exp-009 — Held-out raw-active seven-level QAT

[Experiment and historical specification](../experiments/exp-009-seven-level-qat.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1) | complete: 24 | 24 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1

Saved summary: [summary.json](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/analysis/summary.json). Generated `2026-08-26T19:43:13.568678+00:00`; recorded state `ready_for_review`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| train-zero-update-quantized | 1 | 1 | 0 | 0 | 0 | True |
| train-clean-bptt-quantized-deploy | 1 | 1 | 0 | 0 | 0 | True |
| train-continuous-mapped-hwa | 1 | 1 | 0 | 0 | 0 | True |
| train-quantized-qat | 1 | 1 | 0 | 0 | 0 | True |
| deploy-zero-update-heldout | 5 | 5 | 0 | 0 | 0 | True |
| deploy-clean-bptt-heldout | 5 | 5 | 0 | 0 | 0 | True |
| deploy-continuous-hwa-heldout | 5 | 5 | 0 | 0 | 0 | True |
| deploy-quantized-qat-heldout | 5 | 5 | 0 | 0 | 0 | True |

Current metadata reconciliation: 24 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/primary_analysis.md](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/analysis/primary_analysis.md), [analysis/report.md](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/analysis/summary.json).

## Observations

QAT beats continuous HWA by +2.35 points, yet mean held-out apparent accuracy is 52.66%.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/deploy-clean-bptt-heldout/20260826T193415.926742Z-dfac1a30-f4b692f2](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-clean-bptt-heldout/20260826T193415.926742Z-dfac1a30-f4b692f2/result.json) | complete | evaluation accuracy: 17.51% |
| [runs/deploy-clean-bptt-heldout/20260826T193447.702630Z-f1082bf2-0bf7e22a](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-clean-bptt-heldout/20260826T193447.702630Z-f1082bf2-0bf7e22a/result.json) | complete | evaluation accuracy: 18.61% |
| [runs/deploy-clean-bptt-heldout/20260826T193518.597564Z-6278f510-26aee031](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-clean-bptt-heldout/20260826T193518.597564Z-6278f510-26aee031/result.json) | complete | evaluation accuracy: 19.84% |
| [runs/deploy-clean-bptt-heldout/20260826T193548.514756Z-140441ca-970ed24f](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-clean-bptt-heldout/20260826T193548.514756Z-140441ca-970ed24f/result.json) | complete | evaluation accuracy: 16.88% |
| [runs/deploy-clean-bptt-heldout/20260826T193618.619556Z-c1eb60e0-d29d39fb](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-clean-bptt-heldout/20260826T193618.619556Z-c1eb60e0-d29d39fb/result.json) | complete | evaluation accuracy: 18.08% |
| [runs/deploy-continuous-hwa-heldout/20260826T193415.926745Z-d474d671-1e1b70ae](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-continuous-hwa-heldout/20260826T193415.926745Z-d474d671-1e1b70ae/result.json) | complete | evaluation accuracy: 50.58% |
| [runs/deploy-continuous-hwa-heldout/20260826T193447.705932Z-0dd4d506-f714d247](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-continuous-hwa-heldout/20260826T193447.705932Z-0dd4d506-f714d247/result.json) | complete | evaluation accuracy: 50.62% |
| [runs/deploy-continuous-hwa-heldout/20260826T193518.585762Z-39226968-2603ef44](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-continuous-hwa-heldout/20260826T193518.585762Z-39226968-2603ef44/result.json) | complete | evaluation accuracy: 48.88% |
| [runs/deploy-continuous-hwa-heldout/20260826T193548.514758Z-81f8ff74-114675c6](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-continuous-hwa-heldout/20260826T193548.514758Z-81f8ff74-114675c6/result.json) | complete | evaluation accuracy: 47.18% |
| [runs/deploy-continuous-hwa-heldout/20260826T193618.618296Z-ae8784f1-32fcd9ab](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-continuous-hwa-heldout/20260826T193618.618296Z-ae8784f1-32fcd9ab/result.json) | complete | evaluation accuracy: 54.29% |
| [runs/deploy-quantized-qat-heldout/20260826T193415.926750Z-dfac1a30-a1acfabd](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-quantized-qat-heldout/20260826T193415.926750Z-dfac1a30-a1acfabd/result.json) | complete | evaluation accuracy: 51.85% |
| [runs/deploy-quantized-qat-heldout/20260826T193447.692060Z-f1082bf2-1c837484](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-quantized-qat-heldout/20260826T193447.692060Z-f1082bf2-1c837484/result.json) | complete | evaluation accuracy: 52.97% |
| [runs/deploy-quantized-qat-heldout/20260826T193518.585762Z-6278f510-9f66b9e8](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-quantized-qat-heldout/20260826T193518.585762Z-6278f510-9f66b9e8/result.json) | complete | evaluation accuracy: 52.56% |
| [runs/deploy-quantized-qat-heldout/20260826T193548.514759Z-140441ca-4640da7c](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-quantized-qat-heldout/20260826T193548.514759Z-140441ca-4640da7c/result.json) | complete | evaluation accuracy: 51.28% |
| [runs/deploy-quantized-qat-heldout/20260826T193618.618295Z-c1eb60e0-1bd33578](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-quantized-qat-heldout/20260826T193618.618295Z-c1eb60e0-1bd33578/result.json) | complete | evaluation accuracy: 54.66% |
| [runs/deploy-zero-update-heldout/20260826T193415.926993Z-dfac1a30-3540b101](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-zero-update-heldout/20260826T193415.926993Z-dfac1a30-3540b101/result.json) | complete | evaluation accuracy: 17.65% |
| [runs/deploy-zero-update-heldout/20260826T193447.702647Z-f1082bf2-d911ec9d](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-zero-update-heldout/20260826T193447.702647Z-f1082bf2-d911ec9d/result.json) | complete | evaluation accuracy: 18.87% |
| [runs/deploy-zero-update-heldout/20260826T193518.585794Z-6278f510-dda6f621](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-zero-update-heldout/20260826T193518.585794Z-6278f510-dda6f621/result.json) | complete | evaluation accuracy: 20.25% |
| [runs/deploy-zero-update-heldout/20260826T193548.514756Z-140441ca-ac319857](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-zero-update-heldout/20260826T193548.514756Z-140441ca-ac319857/result.json) | complete | evaluation accuracy: 16.66% |
| [runs/deploy-zero-update-heldout/20260826T193618.618479Z-c1eb60e0-24ea73db](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/deploy-zero-update-heldout/20260826T193618.618479Z-c1eb60e0-24ea73db/result.json) | complete | evaluation accuracy: 18.17% |
| [runs/train-clean-bptt-quantized-deploy/20260826T190255.830365Z-54923aae-3e42452c](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/train-clean-bptt-quantized-deploy/20260826T190255.830365Z-54923aae-3e42452c/result.json) | complete | selected student accuracy: 30.15%; initial student accuracy: 29.25%; terminal student accuracy: 30.15% |
| [runs/train-continuous-mapped-hwa/20260826T190255.837865Z-326fd400-1ab2c53a](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/train-continuous-mapped-hwa/20260826T190255.837865Z-326fd400-1ab2c53a/result.json) | complete | selected student accuracy: 88.82%; initial student accuracy: 27.98%; terminal student accuracy: 88.82% |
| [runs/train-quantized-qat/20260826T190255.826495Z-0bc0915e-90498562](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/train-quantized-qat/20260826T190255.826495Z-0bc0915e-90498562/result.json) | complete | selected student accuracy: 90.18%; initial student accuracy: 29.25%; terminal student accuracy: 90.18% |
| [runs/train-zero-update-quantized/20260826T190255.801610Z-4e1a2871-654f89a3](../../../../../results/mnist-ibm-om-raw-active-p90-quantized-hwa-20260826-v1/runs/train-zero-update-quantized/20260826T190255.801610Z-4e1a2871-654f89a3/result.json) | complete | selected student accuracy: 29.25%; initial student accuracy: 29.25%; terminal student accuracy: 29.25% |


<details>
<summary>Existing analysis: primary_analysis.md</summary>

# Evidence-only primary analysis

The formal study has four valid development runs, all twenty held-out deployments, and twenty read-only endpoint decompositions. Each held-out deployment evaluated all 10,000 test examples. Artifact verification reports `ready_for_review`. This document records measurements; scientific closeout remains a human review decision.

## Primary comparison

| Endpoint seed | Seven-level QAT | Continuous HWA | QAT minus continuous |
| --- | ---: | ---: | ---: |
| 85101 | 51.85% | 50.58% | +1.27 pp |
| 85102 | 52.97% | 50.62% | +2.35 pp |
| 85103 | 52.56% | 48.88% | +3.68 pp |
| 85104 | 51.28% | 47.18% | +4.10 pp |
| 85105 | 54.66% | 54.29% | +0.37 pp |
| **Mean** | **52.66%** | **50.31%** | **+2.35 pp** |

The paired-difference sample standard deviation is 1.57 points. Exact enumeration of all 3,125 paired bootstrap resamples gives a 95% interval of `[+1.13 pp, +3.58 pp]`. The predeclared +2.00-point mean gate was met.

## Pipeline decomposition

| Pipeline | Development selected | Clean selected | Ideal mapped | Apparent mean | Persistent diagnostic | Ideal-to-apparent flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-update quantized | 29.25% | 96.88% | 91.11% | 18.32% | 19.02% | 80.46% |
| Clean BPTT, quantized deploy | 30.15% | 97.40% | 91.62% | 18.18% | 18.96% | 80.58% |
| Continuous mapped HWA | 88.82% | 67.06% | 76.98% | 50.31% | 42.32% | 47.59% |
| Seven-level QAT | 90.18% | 50.33% | 76.36% | 52.66% | 44.65% | 46.97% |

The read-only development decompositions separate ideal mapping from the saved programmed endpoint. Seven-level QAT is 75.90% at the development ideal target and 89.66% at its saved development endpoint; on held-out data it is 76.36% ideal and only 52.66% apparent. Continuous HWA similarly changes from 76.78% ideal / 88.50% apparent in development to 76.98% ideal / 50.31% apparent when held out. Thus the large transfer loss occurs in the programmed endpoint response, not in ideal mapping accuracy. Repeated checkpoint selection on fixed development endpoint streams is a plausible exposure mechanism and is recorded as a limitation rather than asserted as a causal proof.

The held-out structural mask is not the development mask. For W1, 3,401 quads that were eligible during development become structural failures on assignment 85001, while 3,580 move in the opposite direction. Only 350 W1 quads fail in both assignments. Baseline and eligibility tensor hashes also change, although the array-wide coordinate and D90 remain frozen. Because ideal mapped accuracy stays within half a point, this mask turnover does not explain the gross accuracy collapse by itself.

## Differential-signal mechanism

| Pipeline/layer | Eligible D RMS | Min nonzero D | Apparent D-error RMS, all | Eligible only | Accepted-cell only | Persistent eligible | Eligible apparent SNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous mapped HWA, W1 | 0.020247 | 2.384e-07 | 0.034789 | 0.006997 | 0.004318 | 0.018994 | 9.23 dB |
| Continuous mapped HWA, W2 | 0.031191 | 9.298e-06 | 0.038821 | 0.004564 | 0.004304 | 0.017985 | 16.70 dB |
| Seven-level QAT, W1 | 0.026148 | 3.451e-02 | 0.035076 | 0.006957 | 0.004312 | 0.018944 | 11.50 dB |
| Seven-level QAT, W2 | 0.032699 | 3.451e-02 | 0.038822 | 0.004906 | 0.004403 | 0.018234 | 16.49 dB |

For QAT, the eligible minimum nonzero D is the frozen seven-level spacing. It exceeds the accepted-cell-only D-error RMS in both layers. The much larger all-quad error is concentrated in the explicit structural-failure quads; those quads are retained, not donor-reassigned or clipped.

## Programming cost

| Pipeline | Conditioning pulses | Target pulses/cell | Eligible acceptance | Accepted persistent within tolerance | Exhausted cells | Structural-failure cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-update quantized | 32,328,665 | 11.745 | 99.85% | 43,186.2 | 216.6 | 15,208 |
| Clean BPTT, quantized deploy | 32,328,665 | 11.748 | 99.85% | 43,177.2 | 216.4 | 15,208 |
| Continuous mapped HWA | 32,328,665 | 11.734 | 99.86% | 43,184.4 | 206.0 | 15,208 |
| Seven-level QAT | 32,328,665 | 11.755 | 99.85% | 43,165.2 | 210.4 | 15,208 |

Boundary conditioning is separate from the 128-pulse target budget. Apparent cell acceptance is about 99.86% among programming-eligible cells, but only about 30% of accepted cells remain within tolerance in persistent state. Active corrupt and non-finite counts are zero; published-corrupt identities remain reported provenance for the counterfactually repaired array.

## Scope

The comparison with the earlier RESET-relative study is descriptive only: coordinate, baseline policy, codebook, conditioning/controller path, gains, and learning rates all changed. This study contains no Tiki-Taka, LoRA, or direct pulse-recovery arm. A remaining deployment gap therefore cannot establish that on-chip training is necessary.

</details>


## Interpretation and limitations

The declared +2-point mean gate passes (paired 95% bootstrap interval +1.13 to +3.58 points). The large development-to-held-out programmed-endpoint loss remains. Fixed endpoint-stream exposure is a plausible confound, not a demonstrated cause. No recovery arm establishes that on-chip training is necessary, and the RESET-relative comparison changes too many factors for causal attribution.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Investigate held-out programmed-endpoint degradation separately from the supported QAT increment; no on-chip necessity claim follows.
