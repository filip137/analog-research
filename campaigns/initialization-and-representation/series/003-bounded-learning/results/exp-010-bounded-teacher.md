---
experiment: "exp-010"
evidence: "imported-summary"
summary: "The independently trained bounded DRN reaches 94.29% held-out test accuracy."
verdicts: {}
---

# Result: exp-010 — Independently trained bounded DRN reference

[Experiment and historical specification](../experiments/exp-010-bounded-teacher.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist_bounded_drn_teacher_seed17_10ep_6aa32237](../../../../../results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237) | complete: 1 | 1 |
| [mnist_bounded_drn_teacher_seed17_10ep_6aa32237.launcher](../../../../../results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237.launcher) | No native status records | 0 |
| [mnist_bounded_drn_teacher_seed17_10ep_6aa32237_test](../../../../../results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237_test) | complete: 1 | 1 |
| [mnist_bounded_drn_teacher_seed17_10ep_6aa32237_test.launcher](../../../../../results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237_test.launcher) | No native status records | 0 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

The independently trained bounded DRN reaches 94.29% held-out test accuracy.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [20260820T140222.931115Z-809aec60-8da7bc60](../../../../../results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237/20260820T140222.931115Z-809aec60-8da7bc60/result.json) | complete | selected accuracy: 93.58%; terminal accuracy: 93.58% |
| [20260820T140600.942755Z-c40671d3-e2c7d4b0](../../../../../results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237_test/20260820T140600.942755Z-c40671d3-e2c7d4b0/result.json) | complete | evaluation accuracy: 94.29% |

<a id="source-mnist_bounded_drn_teacher-terminal-result"></a>

## Terminal result

- Training run ID: `20260820T140222.931115Z-809aec60-8da7bc60`.
- Launcher: `nohup`, PID `281272`, exit code `0`; repository-native status is
  `complete`. Runtime was 149.214 s.
- All 10 epochs completed (34,380 minibatch updates). The selected checkpoint
  is the final epoch: 93.58% validation accuracy and validation cost
  0.08011558. Final-epoch training accuracy was 93.9964%.
- Independent test run ID:
  `20260820T140600.942755Z-c40671d3-e2c7d4b0`; launcher exit code `0` and
  repository-native status `complete`.
- Untouched 10,000-example MNIST test accuracy: **94.29%**; mean paired
  squared-error cost: `0.07546469`. All evaluations used exactly four
  coordinate iterations.
- Selected named-weights SHA-256:
  `f0036b36cebe970c5105c22ebe703d53fec99b7716215cca0e55ee09cfaf70cf`.
  Exact-resume SHA-256:
  `aceb970c287430e641f1de0d2c32565959ba3c6f77be3a391ccab6b0a2a4ee24`.
- First layer realized logical range: `0.10204082--0.23940003`; CMO mapping:
  `9.0--21.1151 uS`; Wan mapping: `4.08163--9.57600 uS`.
- Second layer realized logical range: `0.10204082--0.56101835`; CMO mapping:
  `9.0--49.4818 uS`; Wan mapping: `4.08163--22.4407 uS`.
- No value reached the upper bound. The fraction at the finite lower floor was
  1.007% in the first layer and 4.95% in the second layer.
- Remote and collected local copies match file-by-file SHA-256 for the
  training run, test run, both checkpoints, validation artifacts, and both
  launcher sidecars. Local artifacts are under
  `results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237*`.


## Interpretation and limitations

Retain this checkpoint as a bounded-learning reference. It is not the common ReLU-mapped initial state needed for later matched endpoint comparisons.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Keep this independently trained checkpoint as a reference, while using direct ReLU mapping for matched deployment comparisons.
