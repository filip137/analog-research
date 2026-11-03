---
experiment: "exp-004"
evidence: "imported-summary"
summary: "Uniform p=0.85 saves 15% of requests with a −0.01-point test change in one seed."
verdicts: {}
---

# Result: exp-004 — Matched probabilistic write policies

[Experiment and historical specification](../experiments/exp-004-probabilistic-writing.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [measured-cohort-b-write-policy-comparison-20260815-v1](../../../../../results/measured-cohort-b-write-policy-comparison-20260815-v1) | No native status records | 0 |
| [measured-cohort-b-policy-deadband-0p0003-20260815-v1](../../../../../results/measured-cohort-b-policy-deadband-0p0003-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-policy-deadband-0p0003-test-20260815-v1](../../../../../results/measured-cohort-b-policy-deadband-0p0003-test-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-policy-displacement-0p0005-20260815-v1](../../../../../results/measured-cohort-b-policy-displacement-0p0005-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-policy-displacement-0p0005-test-20260815-v1](../../../../../results/measured-cohort-b-policy-displacement-0p0005-test-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-policy-uniform-p0p85-20260815-v1](../../../../../results/measured-cohort-b-policy-uniform-p0p85-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-policy-uniform-p0p85-test-20260815-v1](../../../../../results/measured-cohort-b-policy-uniform-p0p85-test-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-policy-zero-20260815-v1](../../../../../results/measured-cohort-b-policy-zero-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-policy-zero-test-20260815-v1](../../../../../results/measured-cohort-b-policy-zero-test-20260815-v1) | complete: 1 | 1 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [summary.json](../../../../../results/measured-cohort-b-write-policy-comparison-20260815-v1/summary.json).

Write-policy evaluation launcher sidecars remain in the shared results/.launchers directory linked by the cohort-B LoRA record.

## Observations

Uniform p=0.85 saves 15% of requests with a −0.01-point test change in one seed.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [20260815T145009.451715Z-782e9796-b68a3436](../../../../../results/measured-cohort-b-policy-deadband-0p0003-20260815-v1/20260815T145009.451715Z-782e9796-b68a3436/result.json) | complete | selected accuracy: 93.70%; terminal accuracy: 93.70% |
| [20260815T151116.080783Z-ab03cb5d-aea83b27](../../../../../results/measured-cohort-b-policy-deadband-0p0003-test-20260815-v1/20260815T151116.080783Z-ab03cb5d-aea83b27/result.json) | complete | evaluation accuracy: 94.60% |
| [20260815T150240.182361Z-d882967b-8d4f0a09](../../../../../results/measured-cohort-b-policy-displacement-0p0005-20260815-v1/20260815T150240.182361Z-d882967b-8d4f0a09/result.json) | complete | selected accuracy: 93.72%; terminal accuracy: 93.68% |
| [20260815T151123.484903Z-ab03cb5d-d0c567a1](../../../../../results/measured-cohort-b-policy-displacement-0p0005-test-20260815-v1/20260815T151123.484903Z-ab03cb5d-d0c567a1/result.json) | complete | evaluation accuracy: 94.55% |
| [20260815T145628.627360Z-6339ecbf-8f43e13e](../../../../../results/measured-cohort-b-policy-uniform-p0p85-20260815-v1/20260815T145628.627360Z-6339ecbf-8f43e13e/result.json) | complete | selected accuracy: 93.80%; terminal accuracy: 93.70% |
| [20260815T151119.800950Z-ab03cb5d-bf2cb358](../../../../../results/measured-cohort-b-policy-uniform-p0p85-test-20260815-v1/20260815T151119.800950Z-ab03cb5d-bf2cb358/result.json) | complete | evaluation accuracy: 94.55% |
| [20260815T144357.859255Z-b91d9a7c-a9eb49a8](../../../../../results/measured-cohort-b-policy-zero-20260815-v1/20260815T144357.859255Z-b91d9a7c-a9eb49a8/result.json) | complete | selected accuracy: 93.82%; terminal accuracy: 93.68% |
| [20260815T151112.246076Z-ab03cb5d-ced42a82](../../../../../results/measured-cohort-b-policy-zero-test-20260815-v1/20260815T151112.246076Z-ab03cb5d-ced42a82/result.json) | complete | evaluation accuracy: 94.56% |

<a id="source-measured_cohort_b_finetuning-matched-policy-outcome-2026-08-15"></a>

### Matched policy outcome (2026-08-15)

Four 20-epoch arms used identical fixed learning rates, initialization,
cohort-B assignments, data order, source checkpoint, and software revision.
Only the write policy changed. Each selected checkpoint was then evaluated on
all 10,000 held-out test examples.

| Policy | Selected validation acc./cost | Test acc./cost | Write requests | Pulse changes |
| --- | ---: | ---: | ---: | ---: |
| Every update | 93.82% / 0.092204 | 94.56% / 0.086236 | 100.00% | 20.34% |
| Deadband, 3e-4 RMS | 93.70% / 0.092434 | 94.60% / 0.086232 | 85.34% | 18.33% |
| Uniform Bernoulli, p=0.85 | 93.80% / 0.091962 | 94.55% / 0.085801 | 85.00% | 17.38% |
| Displacement proportional, scale=5e-4 RMS | 93.72% / 0.092222 | 94.55% / 0.086125 | 87.56% | 18.71% |

Uniform Bernoulli is the most promising tested probabilistic strategy. It
suppressed 15.0% of programming requests and 14.6% of pulse changes while test
accuracy changed by only -0.01 percentage points, one example. It also gave
the lowest selected-validation cost and lowered held-out test cost by 0.50%.
Displacement-proportional writing saved fewer writes and did not improve on
uniform accuracy or cost. The deadband's +0.04-point test change is only four
examples and coincides with lower selected-validation accuracy, so it is not
evidence of an accuracy benefit.

This is a single-training-seed exploratory comparison under global-nearest
programming. It supports uniform p=0.85 as the next policy to replicate across
seeds; it does not yet establish a statistically reliable accuracy gain. The
controlled summary and plot are in
`results/measured-cohort-b-write-policy-comparison-20260815-v1/`.


## Interpretation and limitations

The comparison supports write-cost reduction under global-nearest programming. Neither the tiny positive deadband change nor the uniform result establishes improved population accuracy.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Uniform p=0.85 is a candidate for replication, not an established accuracy-improving policy.
