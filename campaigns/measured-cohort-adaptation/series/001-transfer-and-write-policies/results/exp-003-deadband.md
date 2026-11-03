---
experiment: "exp-003"
evidence: "imported-summary"
summary: "Deadbands reduce writes substantially without a resolved accuracy advantage."
verdicts: {}
---

# Result: exp-003 — Fixed-rate programming deadbands

[Experiment and historical specification](../experiments/exp-003-deadband.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [measured-cohort-b-deadband-fixed-zero-20260815-v1](../../../../../results/measured-cohort-b-deadband-fixed-zero-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-threshold-sweep-20260815-v1](../../../../../results/measured-cohort-b-threshold-sweep-20260815-v1) | No native status records | 0 |
| [measured-cohort-b-threshold-rms-0p0001-20260815-v1](../../../../../results/measured-cohort-b-threshold-rms-0p0001-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-threshold-rms-0p0003-20260815-v1](../../../../../results/measured-cohort-b-threshold-rms-0p0003-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-threshold-rms-0p001-20260815-v1](../../../../../results/measured-cohort-b-threshold-rms-0p001-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-threshold-rms-0p001-test-20260815-v1](../../../../../results/measured-cohort-b-threshold-rms-0p001-test-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-threshold-rms-0p003-20260815-v1](../../../../../results/measured-cohort-b-threshold-rms-0p003-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-threshold-rms-0p003-test-20260815-v1](../../../../../results/measured-cohort-b-threshold-rms-0p003-test-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-threshold-rms-0p01-20260815-v1](../../../../../results/measured-cohort-b-threshold-rms-0p01-20260815-v1) | complete: 1 | 1 |
| [measured-cohort-b-threshold-rms-0p01-test-20260815-v1](../../../../../results/measured-cohort-b-threshold-rms-0p01-test-20260815-v1) | complete: 1 | 1 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [summary.json](../../../../../results/measured-cohort-b-threshold-sweep-20260815-v1/summary.json).

## Observations

Deadbands reduce writes substantially without a resolved accuracy advantage.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [20260815T141540.321281Z-8f27774e-bf8524d0](../../../../../results/measured-cohort-b-deadband-fixed-zero-20260815-v1/20260815T141540.321281Z-8f27774e-bf8524d0/result.json) | complete | selected accuracy: 93.82%; terminal accuracy: 93.68% |
| [20260815T142151.624873Z-b5283049-97f68588](../../../../../results/measured-cohort-b-threshold-rms-0p0001-20260815-v1/20260815T142151.624873Z-b5283049-97f68588/result.json) | complete | selected accuracy: 93.76%; terminal accuracy: 93.84% |
| [20260815T142805.003809Z-81154e73-ac2db84c](../../../../../results/measured-cohort-b-threshold-rms-0p0003-20260815-v1/20260815T142805.003809Z-81154e73-ac2db84c/result.json) | complete | selected accuracy: 93.70%; terminal accuracy: 93.70% |
| [20260815T135244.560538Z-a1764441-1b5f2228](../../../../../results/measured-cohort-b-threshold-rms-0p001-20260815-v1/20260815T135244.560538Z-a1764441-1b5f2228/result.json) | complete | selected accuracy: 93.82%; terminal accuracy: 93.62% |
| [20260815T141227.778052Z-ab03cb5d-01ceb292](../../../../../results/measured-cohort-b-threshold-rms-0p001-test-20260815-v1/20260815T141227.778052Z-ab03cb5d-01ceb292/result.json) | complete | evaluation accuracy: 94.52% |
| [20260815T135855.696485Z-1bfc5195-9e1a8a03](../../../../../results/measured-cohort-b-threshold-rms-0p003-20260815-v1/20260815T135855.696485Z-1bfc5195-9e1a8a03/result.json) | complete | selected accuracy: 93.80%; terminal accuracy: 93.60% |
| [20260815T141231.548194Z-ab03cb5d-f265504d](../../../../../results/measured-cohort-b-threshold-rms-0p003-test-20260815-v1/20260815T141231.548194Z-ab03cb5d-f265504d/result.json) | complete | evaluation accuracy: 94.52% |
| [20260815T140505.206714Z-e0aa1e34-2b97e46a](../../../../../results/measured-cohort-b-threshold-rms-0p01-20260815-v1/20260815T140505.206714Z-e0aa1e34-2b97e46a/result.json) | complete | selected accuracy: 93.72%; terminal accuracy: 93.58% |
| [20260815T141235.235972Z-ab03cb5d-89c3cecd](../../../../../results/measured-cohort-b-threshold-rms-0p01-test-20260815-v1/20260815T141235.235972Z-ab03cb5d-89c3cecd/result.json) | complete | evaluation accuracy: 94.59% |

<a id="source-measured_cohort_b_finetuning-deadband-sweep-outcome-2026-08-15"></a>

### Deadband sweep outcome (2026-08-15)

A fixed-learning-rate sweep at relative thresholds 0.001, 0.003, and 0.01
completed 20 epochs per arm from the identical cohort-A checkpoint and
deterministic cohort-B deployment. The correct matched no-deadband control is
the fixed-rate run at 93.82% selected validation and 94.56% held-out test
accuracy. The earlier summary incorrectly used the 2026-08-14 automatic-LR
run at 95.22%/95.75% as its control. That older run remains a valid transfer
feasibility result, but its earlier software and LR-selection path make it an
invalid control for the fixed-rate write-policy intervention.

Against the matched control, thresholds 0.001, 0.003, and 0.01 produced
93.82%, 93.80%, and 93.72% selected validation accuracy and 94.52%, 94.52%,
and 94.59% held-out test accuracy. No threshold improved selected validation;
the largest test change was only +0.03 percentage points, or three examples.
Weighted across dense weights, the thresholds reduced programming requests by
39.1%, 72.0%, and 93.3% and actual pulse-index changes by 37.1%, 73.9%, and
94.3%. The corrected machine-readable summary and plot are in
`results/measured-cohort-b-threshold-sweep-20260815-v1/`.


## Interpretation and limitations

Use the 93.82% validation / 94.56% test fixed-rate control, not the older 95.75% test auto-LR run. Test changes of only a few examples are descriptive, not significance claims.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Evaluate write reduction at matched rates; seed replication is needed before an accuracy claim.
