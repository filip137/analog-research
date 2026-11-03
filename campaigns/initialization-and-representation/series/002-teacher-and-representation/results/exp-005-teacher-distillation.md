---
experiment: "exp-005"
evidence: "imported-summary"
summary: "Common-window differential mapping learns; independent single-conductance mapping stays near chance."
verdicts: {}
---

# Result: exp-005 — Teacher-initialized measured DRN distillation

[Experiment and historical specification](../experiments/exp-005-teacher-distillation.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-relu-drn-kd-exploratory-20260816](../../../../../results/mnist-relu-drn-kd-exploratory-20260816) | complete: 15, failed: 4, running: 2 | 15 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [analysis/summary.json](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/analysis/summary.json).

## Observations

Common-window differential mapping learns; independent single-conductance mapping stays near chance.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [ideal_differential_finetune/20260816T135214.689196Z-9ef10aa2-642ca991](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/ideal_differential_finetune/20260816T135214.689196Z-9ef10aa2-642ca991/result.json) | complete | selected student accuracy: 97.16%; initial student accuracy: 96.74%; terminal student accuracy: 97.16% |
| [ideal_differential_test/20260816T141748.133282Z-5f2770a2-d0390615](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/ideal_differential_test/20260816T141748.133282Z-5f2770a2-d0390615/result.json) | complete | evaluation accuracy: 97.42% |
| [ideal_single_finetune/20260816T134436.350896Z-6613a14b-ed8c93cb](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/ideal_single_finetune/20260816T134436.350896Z-6613a14b-ed8c93cb/result.json) | complete | selected student accuracy: 97.16%; initial student accuracy: 96.74%; terminal student accuracy: 97.16% |
| [ideal_single_lr1000_probe/20260816T134215.735319Z-fb189170-03a42e89](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/ideal_single_lr1000_probe/20260816T134215.735319Z-fb189170-03a42e89/result.json) | complete | selected student accuracy: 97.22%; initial student accuracy: 96.74%; terminal student accuracy: 97.22% |
| [ideal_single_lr100_probe/20260816T134036.783940Z-41a659aa-7b1131f5](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/ideal_single_lr100_probe/20260816T134036.783940Z-41a659aa-7b1131f5/result.json) | complete | selected student accuracy: 97.20%; initial student accuracy: 96.74%; terminal student accuracy: 97.20% |
| [ideal_single_test/20260816T141748.148778Z-2afe68a0-aae3240b](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/ideal_single_test/20260816T141748.148778Z-2afe68a0-aae3240b/result.json) | complete | evaluation accuracy: 97.42% |
| [ideal_single_v2/20260816T133250.739654Z-4c208bf3-430bd2fc](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/ideal_single_v2/20260816T133250.739654Z-4c208bf3-430bd2fc/result.json) | complete | selected student accuracy: 96.74%; initial student accuracy: 96.74%; terminal student accuracy: 97.52% |
| [measured_differential_affine_v2/20260816T140455.983641Z-629b69f5-7db6a620](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/measured_differential_affine_v2/20260816T140455.983641Z-629b69f5-7db6a620/result.json) | complete | selected student accuracy: 97.36%; initial student accuracy: 89.12%; terminal student accuracy: 97.50% |
| [measured_differential_test/20260816T141748.097520Z-5e1aec0e-b9019ef8](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/measured_differential_test/20260816T141748.097520Z-5e1aec0e-b9019ef8/result.json) | complete | evaluation accuracy: 97.30% |
| [measured_single_affine/20260816T140549.752032Z-92c3322d-62fb07e6](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/measured_single_affine/20260816T140549.752032Z-92c3322d-62fb07e6/result.json) | complete | selected student accuracy: 8.20%; initial student accuracy: 8.10%; terminal student accuracy: 8.20% |
| [measured_single_finetune/20260816T135329.096214Z-b70fa901-1f05d90a](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/measured_single_finetune/20260816T135329.096214Z-b70fa901-1f05d90a/result.json) | complete | selected student accuracy: 8.38%; initial student accuracy: 8.40%; terminal student accuracy: 8.40% |
| [measured_single_test/20260816T141748.097551Z-9aeb9402-f574f722](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/measured_single_test/20260816T141748.097551Z-9aeb9402-f574f722/result.json) | complete | evaluation accuracy: 7.99% |
| [teacher/20260816T132258.976966Z-fbff3c26-556558ea](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/teacher/20260816T132258.976966Z-fbff3c26-556558ea/result.json) | complete | selected accuracy: 96.30%; terminal accuracy: 96.38% |
| [teacher_fixed_init/20260816T132557.806720Z-fbff3c26-6f6867f1](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/teacher_fixed_init/20260816T132557.806720Z-fbff3c26-6f6867f1/result.json) | complete | selected accuracy: 97.26%; terminal accuracy: 96.98% |
| [teacher_test_gpu/20260816T132750.222354Z-d3640c04-7801f4dc](../../../../../results/mnist-relu-drn-kd-exploratory-20260816/teacher_test_gpu/20260816T132750.222354Z-d3640c04-7801f4dc/result.json) | complete | evaluation accuracy: 97.36% |

<a id="source-mnist_relu_drn_kd-exploratory-result"></a>

## Exploratory result

The bias-free ReLU teacher selected at epoch 11 reached 97.26% validation and
97.36% held-out test accuracy. The four matched DRN arms then produced:

| Backend and encoding | Initial validation KL | Selected validation KL | Validation accuracy | Test KL | Test accuracy | Test agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal, one device | 0.010683 | 0.000685 | 97.16% | 0.000712 | 97.42% | 99.83% |
| ideal, differential pair | 0.010683 | 0.000685 | 97.16% | 0.000712 | 97.42% | 99.83% |
| cohort A, one device | 2.242688 | 2.242688 | 8.20% | 2.246097 | 7.99% | 7.82% |
| cohort A, differential pair | 0.231839 | 0.010153 | 97.36% | 0.010119 | 97.30% | 98.98% |

The ideal arms are equal to numerical precision, as expected when doubling the
differential learning rate makes its effective `G+ - G-` update match the
single bounded parameter update. Two devices therefore provide no intrinsic
optimization benefit without device variation.

With measured cohort-A traces, the differential arm lowers selected validation
KL by 99.55% relative to the one-device arm and passes the predefined comparison
gate. Its own KL falls by 95.62% from its programmed initialization. The
one-device arm changes negligibly despite small target-projection errors. This
supports the interpretation that independently varying nonzero device offsets
destroy the dual-rail signal, while the common-window differential mapping
cancels each pair's baseline before fine-tuning. The result is specific to this
single seed and deterministic virtual-device assignment.

A clean ten-epoch follow-up exactly reproduced the first ten epochs of the
20-epoch measured differential arm. It selected epoch 9 and reached `97.48%`
held-out accuracy, `98.76%` teacher agreement, and test KL `0.0140905`. The
20-epoch budget reduced test KL to `0.0101188` and raised agreement to `98.98%`,
although its test accuracy was `0.18` percentage points lower. Selection was by
validation KL, so this single-seed result supports ten epochs as a useful
screening budget, not as a more accurate final budget. See the
[ten-epoch initialized-differential report](../experiments/exp-006-differential-ten-epoch.md#source-mnist_relu_drn_initialized_differential_10ep-teacher-initialized-differential-reram-mnist-fine-tuning-10-epochs).

The corresponding clean ten-epoch one-conductance replay also exactly matched
the first ten epochs of its earlier 20-epoch arm, but it remained at chance:
`7.98%` held-out accuracy, `7.81%` agreement, and test KL `2.246097`. Against
this budget-matched arm, the common-window differential pair gains `89.50`
accuracy points and lowers test KL by `99.37%`. See the
[ten-epoch initialized one-conductance report](../experiments/exp-007-single-ten-epoch.md#source-mnist_relu_drn_initialized_single_10ep-teacher-initialized-single-conductance-reram-mnist-fine-tuning-10-epochs).

Machine-readable metrics and the matplotlib learning curves are in
`results/mnist-relu-drn-kd-exploratory-20260816/analysis/summary.json` and
`validation_comparison.png`.

<a id="source-mnist_relu_drn_kd-evidence-boundary"></a>

## Evidence boundary

These runs are exploratory, single-seed evidence. They establish numerical
feasibility and define a reproducible comparison, but they do not estimate
uncertainty across teacher seeds, device splits, or virtual-device assignments.


## Interpretation and limitations

Ideal arms coincide after matching effective learning rates. The measured contrast implicates representation and device offsets for one assignment; unfinished native attempts do not invalidate or complete other arms.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use the matched single/differential records as the representation control; retain the current initialization before varying device physics.
