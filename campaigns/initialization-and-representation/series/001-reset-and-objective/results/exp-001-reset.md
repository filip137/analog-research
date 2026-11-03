---
experiment: "exp-001"
evidence: "imported-summary"
summary: "RESET training learns, but the teacher-KL accuracy gate fails."
verdicts: {}
---

# Result: exp-001 — Learning from measured RESET

[Experiment and historical specification](../experiments/exp-001-reset.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-relu-drn-reset-exploratory-20260816](../../../../../results/mnist-relu-drn-reset-exploratory-20260816) | complete: 6 | 6 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [analysis/summary.json](../../../../../results/mnist-relu-drn-reset-exploratory-20260816/analysis/summary.json).

## Observations

RESET training learns, but the teacher-KL accuracy gate fails.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [mnist-relu-drn-reset-cohort-a-20260816-v1/stages/cohort_a_cross_entropy/local/reset_ce_test/attempt-001/runs/20260816T162825.698022Z-79d120bb-f958fe35](../../../../../results/mnist-relu-drn-reset-exploratory-20260816/mnist-relu-drn-reset-cohort-a-20260816-v1/stages/cohort_a_cross_entropy/local/reset_ce_test/attempt-001/runs/20260816T162825.698022Z-79d120bb-f958fe35/result.json) | complete | evaluation accuracy: 86.88% |
| [mnist-relu-drn-reset-cohort-a-20260816-v1/stages/cohort_a_cross_entropy/local/reset_ce_train/attempt-001/runs/20260816T161333.024929Z-9ced4ff0-6f7b282c](../../../../../results/mnist-relu-drn-reset-exploratory-20260816/mnist-relu-drn-reset-cohort-a-20260816-v1/stages/cohort_a_cross_entropy/local/reset_ce_train/attempt-001/runs/20260816T161333.024929Z-9ced4ff0-6f7b282c/result.json) | complete | selected student accuracy: 85.44%; initial student accuracy: 9.96%; terminal student accuracy: 59.48% |
| [mnist-relu-drn-reset-cohort-a-20260816-v1/stages/cohort_a_kl/local/reset_kl_test/attempt-001/runs/20260816T162822.147195Z-0a316d95-382feb56](../../../../../results/mnist-relu-drn-reset-exploratory-20260816/mnist-relu-drn-reset-cohort-a-20260816-v1/stages/cohort_a_kl/local/reset_kl_test/attempt-001/runs/20260816T162822.147195Z-0a316d95-382feb56/result.json) | complete | evaluation accuracy: 85.52% |
| [mnist-relu-drn-reset-cohort-a-20260816-v1/stages/cohort_a_kl/local/reset_kl_train/attempt-001/runs/20260816T155845.304277Z-6e525266-9f1904a4](../../../../../results/mnist-relu-drn-reset-exploratory-20260816/mnist-relu-drn-reset-cohort-a-20260816-v1/stages/cohort_a_kl/local/reset_kl_train/attempt-001/runs/20260816T155845.304277Z-6e525266-9f1904a4/result.json) | complete | selected student accuracy: 83.60%; initial student accuracy: 9.96%; terminal student accuracy: 75.78% |
| [mnist-relu-drn-reset-cohort-a-20260816-v1/stages/teacher/local/teacher_test/attempt-001/runs/20260816T155843.090833Z-d3640c04-6a7eadf2](../../../../../results/mnist-relu-drn-reset-exploratory-20260816/mnist-relu-drn-reset-cohort-a-20260816-v1/stages/teacher/local/teacher_test/attempt-001/runs/20260816T155843.090833Z-d3640c04-6a7eadf2/result.json) | complete | evaluation accuracy: 97.36% |
| [mnist-relu-drn-reset-cohort-a-20260816-v1/stages/teacher/local/teacher_train/attempt-001/runs/20260816T155716.266364Z-fbff3c26-30ca4e0d](../../../../../results/mnist-relu-drn-reset-exploratory-20260816/mnist-relu-drn-reset-cohort-a-20260816-v1/stages/teacher/local/teacher_train/attempt-001/runs/20260816T155716.266364Z-fbff3c26-30ca4e0d/result.json) | complete | selected accuracy: 97.26%; terminal accuracy: 96.98% |

<a id="source-mnist_relu_drn_reset-exploratory-result"></a>

## Exploratory result

The six-stage local-GPU campaign completed successfully. The teacher reached
`97.36%` held-out test accuracy. Both DRN arms began at `9.96%` validation
accuracy from the identical pulse-zero assignments, and all nine learning-rate
cells passed their safety canaries.

| Objective | Selected learning rates (hidden, output) | Selected validation accuracy | Test accuracy | Test teacher agreement | Raw test KL | Report-only calibrated test KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `teacher_kl` | `(6.824e-7, 8.823e-8)` | `83.60%` | `85.52%` | `86.35%` | `0.68973` | `0.38257` |
| `cross_entropy` | `(2.260e-7, 8.707e-8)` | `85.44%` | `86.88%` | `87.61%` | `0.67903` | `0.35383` |

The predefined hypothesis gate did not pass: KL accuracy is below `94%` and
is `1.36` percentage points below the cross-entropy control, just outside the
one-point tolerance. This is nevertheless real learning from RESET rather than
an unchanged initialization: both layers changed pulse states, and KL improved
validation accuracy by `73.64` points over the RESET operating point.
Training was strongly non-monotonic under global-nearest raw-trace projection,
so validation-selected checkpoints were essential.

The matched control indicates that KL is not the main source of the shortfall:
hard-label cross-entropy improves test accuracy by only `1.36` points. This
experiment is also not an exact reproduction of the earlier approximately
`95%` measured-cohort-A from-scratch run. That historical `small_drn.v1` run
trained `base.bias.0` and used `SquaredErrorPairedOutputs`; the present student
is deliberately bias-free and uses KL or cross-entropy. A completed exact
historical-control replay subsequently found an additional confound: production
training used process-global amplifier indices 3/4/5, while LR selection and
fresh-process testing used 0/1/2. That replay recovered the archived `95.97%`
only with paired squared error; cross-entropy reached `92.37%` and KL `83.72%`.
See [`mnist_relu_drn_reset_bias.md`](../experiments/exp-002-bias-and-indexing.md#source-mnist_relu_drn_reset_bias-bias-inclusive-reset-loss-control). The archived
result is therefore not a valid target for the corrected logical-index circuit.

Machine-readable metrics and the matplotlib comparison are in
`results/mnist-relu-drn-reset-exploratory-20260816/analysis/summary.json` and
`validation_comparison.png`.


## Interpretation and limitations

KL and cross-entropy both learn from pulse zero; they do not reproduce the historical biased, differently indexed squared-error circuit.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use the corrected loss/indexing factorial to interpret the historical gap; do not tune a new RESET run merely to recover the archived number.
