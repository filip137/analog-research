---
experiment: "exp-008"
evidence: "imported-summary"
summary: "Common-window adaptation reaches 96.96% test versus 97.37% bounded FP32."
verdicts: {}
---

# Result: exp-008 — Pairwise common window versus bounded FP32

[Experiment and historical specification](../experiments/exp-008-common-window-fp32.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-single-rail-common-window-vs-fp32-10ep-20260818-v1](../../../../../results/mnist-single-rail-common-window-vs-fp32-10ep-20260818-v1) | complete: 4 | 4 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [aggregate/results.json](../../../../../results/mnist-single-rail-common-window-vs-fp32-10ep-20260818-v1/aggregate/results.json).

## Observations

Common-window adaptation reaches 96.96% test versus 97.37% bounded FP32.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [stages/cohort_a_single_pairwise_common_window_10ep/local/measured_pairwise_test/attempt-001/runs/20260818T083811.167280Z-6807ef5b-eeb310a4](../../../../../results/mnist-single-rail-common-window-vs-fp32-10ep-20260818-v1/stages/cohort_a_single_pairwise_common_window_10ep/local/measured_pairwise_test/attempt-001/runs/20260818T083811.167280Z-6807ef5b-eeb310a4/result.json) | complete | evaluation accuracy: 96.96% |
| [stages/cohort_a_single_pairwise_common_window_10ep/local/measured_pairwise_train/attempt-001/runs/20260818T083239.247955Z-81d5359d-66faa328](../../../../../results/mnist-single-rail-common-window-vs-fp32-10ep-20260818-v1/stages/cohort_a_single_pairwise_common_window_10ep/local/measured_pairwise_train/attempt-001/runs/20260818T083239.247955Z-81d5359d-66faa328/result.json) | complete | selected student accuracy: 96.20%; initial student accuracy: 34.20%; terminal student accuracy: 96.20% |
| [stages/ideal_single_centered_70_90us_10ep/local/fp32_bounded_test/attempt-001/runs/20260818T083814.095094Z-6c07a7fa-ad10ad6c](../../../../../results/mnist-single-rail-common-window-vs-fp32-10ep-20260818-v1/stages/ideal_single_centered_70_90us_10ep/local/fp32_bounded_test/attempt-001/runs/20260818T083814.095094Z-6c07a7fa-ad10ad6c/result.json) | complete | evaluation accuracy: 97.37% |
| [stages/ideal_single_centered_70_90us_10ep/local/fp32_bounded_train/attempt-001/runs/20260818T083538.888711Z-0d701ba6-d0965c58](../../../../../results/mnist-single-rail-common-window-vs-fp32-10ep-20260818-v1/stages/ideal_single_centered_70_90us_10ep/local/fp32_bounded_train/attempt-001/runs/20260818T083538.888711Z-0d701ba6-d0965c58/result.json) | complete | selected student accuracy: 97.26%; initial student accuracy: 97.26%; terminal student accuracy: 97.44% |

<a id="source-mnist_single_rail_common_window_vs_fp32-result"></a>

## Result

All four stages completed from clean commit `7e144ace8b573149057bf55d9e3fc63eda20b623`.
An immediate `--resume` replay reused the completed stages after validating
their fingerprints and artifact hashes.

| Arm | Selected layer scales | Initial validation | Epoch-10 validation | Selected checkpoint | Fresh test |
| --- | --- | ---: | ---: | --- | ---: |
| Measured cohort-A common window | `(1.0, 0.5)` | 34.20% | 96.20% | epoch 10 | 96.96% |
| Bounded FP32, 70--90 uS | `(1.0, 1.0)` | 97.26% | 97.44% | initialization | 97.37% |

Selection used validation teacher KL, not accuracy.  The measured arm's
selected epoch-10 checkpoint has validation KL 0.050072 and fresh-test KL
0.041213.  The FP32 initialization exactly preserves the teacher closely
enough that none of its trained checkpoints improves on its initial
validation KL of 7.804e-6; its selected fresh-test KL is 8.180e-6.  The FP32
epoch-10 validation accuracy is reported as an endpoint diagnostic, while its
fresh test uses the protocol-selected initialization checkpoint.

The measured initialization used mean common-window baselines of 70.799 and
71.209 uS and mean spans of 18.323 and 17.934 uS for the two layers.  Empty
pairwise overlaps affected 1.367% and 1.200% of pairs.  Despite small
per-device nearest-state projection errors (0.245 and 0.148 uS RMS), the
independently projected complementary rails accumulated enough mismatch to
reduce initial validation accuracy to 34.20%.

Ten epochs of measured on-chip-style updates recovered 62.00 validation
percentage points and 97.16% of the initial teacher-KL gap.  Its selected
fresh-test accuracy is only 0.41 percentage points below the bounded FP32
control.  Thus the narrow conductance range itself is not the limiting factor:
device-specific projection mismatch damages the initial function, while
analog adaptation recovers nearly all of the classification-accuracy loss in
this single-seed screen.


## Interpretation and limitations

The common-window arm recovers after a poor initialization, while bounded FP32 selects its initialization. Narrow range alone does not explain the measured projection loss in this seed.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Keep bounded FP32 as the no-device reference and report initialization loss separately from adaptation.
