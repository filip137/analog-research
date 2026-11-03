---
experiment: "exp-002"
evidence: "imported-summary"
summary: "The 95.97% historical result is reproduced under mismatched amplifier semantics."
verdicts: {}
---

# Result: exp-002 — Bias, loss and historical amplifier indexing

[Experiment and historical specification](../experiments/exp-002-bias-and-indexing.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1) | complete: 8 | 8 |
| [mnist-relu-drn-reset-bias-loss-control-20260816-v1](../../../../../results/mnist-relu-drn-reset-bias-loss-control-20260816-v1) | complete: 2, failed: 1 | 2 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [analysis/summary.json](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/analysis/summary.json), [aggregate/results.json](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/aggregate/results.json).

Supporting historical launcher logs: results/mnist-reset-bias-legacy-loss-campaign.log and results/mnist-reset-bias-loss-campaign.log.

## Observations

The 95.97% historical result is reproduced under mismatched amplifier semantics.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [stages/cross_entropy/local/cross_entropy_test/attempt-002/runs/20260816T180523.077241Z-d3c672f0-0d78c3ac](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/stages/cross_entropy/local/cross_entropy_test/attempt-002/runs/20260816T180523.077241Z-d3c672f0-0d78c3ac/result.json) | complete | evaluation accuracy: 92.37% |
| [stages/cross_entropy/local/cross_entropy_train/attempt-002/runs/20260816T173341.968634Z-a6206cb3-ba7b55b3](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/stages/cross_entropy/local/cross_entropy_train/attempt-002/runs/20260816T173341.968634Z-a6206cb3-ba7b55b3/result.json) | complete | selected student accuracy: 91.16%; initial student accuracy: 9.96%; terminal student accuracy: 65.12% |
| [stages/paired_squared_error/local/paired_squared_error_test/attempt-002/runs/20260816T180519.587274Z-9a087cf2-3361b5ea](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/stages/paired_squared_error/local/paired_squared_error_test/attempt-002/runs/20260816T180519.587274Z-9a087cf2-3361b5ea/result.json) | complete | evaluation accuracy: 95.97% |
| [stages/paired_squared_error/local/paired_squared_error_train/attempt-002/runs/20260816T171706.808736Z-a1e48d3f-77ad67a0](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/stages/paired_squared_error/local/paired_squared_error_train/attempt-002/runs/20260816T171706.808736Z-a1e48d3f-77ad67a0/result.json) | complete | selected student accuracy: 95.10%; initial student accuracy: 9.96%; terminal student accuracy: 94.94% |
| [stages/teacher/local/teacher_test/attempt-002/runs/20260816T171704.537198Z-d3640c04-8e548d33](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/stages/teacher/local/teacher_test/attempt-002/runs/20260816T171704.537198Z-d3640c04-8e548d33/result.json) | complete | evaluation accuracy: 97.36% |
| [stages/teacher/local/teacher_train/attempt-002/runs/20260816T171537.485897Z-fbff3c26-eb83a18d](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/stages/teacher/local/teacher_train/attempt-002/runs/20260816T171537.485897Z-fbff3c26-eb83a18d/result.json) | complete | selected accuracy: 97.26%; terminal accuracy: 96.98% |
| [stages/teacher_kl/local/teacher_kl_test/attempt-002/runs/20260816T180526.624213Z-484ed92f-a9954f35](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/stages/teacher_kl/local/teacher_kl_test/attempt-002/runs/20260816T180526.624213Z-484ed92f-a9954f35/result.json) | complete | evaluation accuracy: 83.72% |
| [stages/teacher_kl/local/teacher_kl_train/attempt-002/runs/20260816T174949.872962Z-8d44ae18-147bfc14](../../../../../results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/stages/teacher_kl/local/teacher_kl_train/attempt-002/runs/20260816T174949.872962Z-8d44ae18-147bfc14/result.json) | complete | selected student accuracy: 82.78%; initial student accuracy: 9.96%; terminal student accuracy: 56.96% |
| [stages/teacher/local/teacher_test/attempt-001/runs/20260816T165138.722009Z-d3640c04-5529b76e](../../../../../results/mnist-relu-drn-reset-bias-loss-control-20260816-v1/stages/teacher/local/teacher_test/attempt-001/runs/20260816T165138.722009Z-d3640c04-5529b76e/result.json) | complete | evaluation accuracy: 97.36% |
| [stages/teacher/local/teacher_train/attempt-001/runs/20260816T165010.065077Z-fbff3c26-0f1d263d](../../../../../results/mnist-relu-drn-reset-bias-loss-control-20260816-v1/stages/teacher/local/teacher_train/attempt-001/runs/20260816T165010.065077Z-fbff3c26-0f1d263d/result.json) | complete | selected accuracy: 97.26%; terminal accuracy: 96.98% |

<a id="source-mnist_relu_drn_reset_bias-results"></a>

## Results

The eight-stage local RTX 3090 campaign completed successfully. Training
artifacts report edges 3→4 and 4→5; fresh-process student tests report 0→1 and
1→2.

| Objective | Selected validation | Held-out test | Test gap vs paired | Selected LR vector (dense 0, dense 1, bias) |
|---|---:|---:|---:|---|
| Paired squared error | 95.10% | **95.97%** | — | `[1.84898e-7, 6.42698e-8, 1.84898e-7]` |
| Cross-entropy | 91.16% | **92.37%** | −3.60 pp | `[2.26008e-7, 8.70685e-8, 2.26008e-7]` |
| Teacher KL | 82.78% | **83.72%** | −12.25 pp | `[6.82423e-7, 8.82274e-8, 6.82423e-7]` |

The ReLU teacher reached 97.36% held-out accuracy. The paired arm reproduced
the archived selected validation accuracy, selected loss, epoch-1 trajectory,
and 95.97% test accuracy exactly. This validates the provenance replay.

The scientific interpretation is narrower: under the historical train/test
amplifier mismatch, the objective is material. Paired squared error succeeds,
cross-entropy loses 3.60 percentage points, and KL loses 12.25 points. Thus the
archived 95% run was not “the same run with KL”; changing only the loss does
not preserve its recovery. The replay does not establish that the intended
logical 0→1 / 1→2 circuit can reach 95%.

Canonical configs are in `examples/mnist_relu_drn_reset_bias_legacy/`, and the
campaign manifest is
`campaigns/manifests/mnist_relu_drn_reset_bias_legacy.json`. Raw results are
under
`results/mnist-relu-drn-reset-bias-legacy-loss-control-20260816-v1/`.
`labs/tools/plot_mnist_relu_drn_reset_bias_comparison.py` produces the checked
summary and validation/test comparison in the campaign's `analysis/` folder.

This remains a single-seed measured-device simulation. It establishes exact
historical provenance and a controlled objective comparison, not robustness
across seeds or fabricated on-chip behavior.


## Interpretation and limitations

This replay establishes provenance, not accuracy for the corrected logical-index circuit. The unfinished corrected bias-control attempts remain separate from the complete legacy replay.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use the corrected logical-index circuit for future claims. Keep the legacy replay solely as provenance.
