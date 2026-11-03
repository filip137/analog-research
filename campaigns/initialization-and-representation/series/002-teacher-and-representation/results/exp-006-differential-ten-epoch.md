---
experiment: "exp-006"
evidence: "imported-summary"
summary: "The historical matched differential control reaches 97.48% test accuracy."
verdicts: {}
---

# Result: exp-006 — Ten-epoch teacher-initialized differential control

[Experiment and historical specification](../experiments/exp-006-differential-ten-epoch.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| `results/mnist-differential-reram-initialized-finetune-10ep-20260817-v1` — unavailable locally | Unavailable locally | 0 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

The historical matched differential control reaches 97.48% test accuracy.


<a id="source-mnist_relu_drn_initialized_differential_10ep-interpretation-boundary"></a>

## Interpretation boundary

This run is directly comparable to the earlier 20-epoch initialized
differential arm as an epoch-budget screen. It is not an initialization-only
ablation against the literal-RESET run: the initialized path uses teacher KL
and a fitted readout gain, whereas the literal-RESET run used hard-label paired
squared error and fixed gain one.

<a id="source-mnist_relu_drn_initialized_differential_10ep-result"></a>

## Result

The campaign completed from clean source commit
`00bc68760f7cd0018e8bf62c1c7c98ad57bd7553`. Both stages exited zero, the
training stream contains one initialization record followed by exactly ten
completed epochs, and neither stage contains a non-finite metric. The
common-window map selected teacher scale fractions `[1.0, 0.125]` and fitted
readout gain `7943.282347`.

| Evaluation point | KL | Accuracy | Teacher agreement |
| --- | ---: | ---: | ---: |
| mapped initialization, validation | 0.231839 | 89.12% | 90.06% |
| selected epoch 9, validation | 0.0138965 | 97.42% | 98.94% |
| selected checkpoint, held-out test | 0.0140905 | 97.48% | 98.76% |

The selected validation KL recovered `94.01%` of the initialization gap. The
tenth epoch regressed slightly to validation KL `0.0156730`, accuracy `97.30%`,
and agreement `98.72%`, so selection correctly retained epoch 9. Training took
`222.72 s`; fresh-process validation of all 10,000 held-out examples took
`1.49 s`.

The selected checkpoint records the expected model-local stages
`[(0, 1, 1, 1), (1, 2, 4, 0.0625)]`, all four stable `G+`/`G-` conductance
keys, and the measured-device assignment hashes. Its SHA-256 is
`82924b43399920ad3df342c1cf12dd72fe3ea0e9b34908b8396dcac7f210f031`.
The exact epoch-10 resume state has SHA-256
`b1c843b6d8000b6272052db6625f83dc1c4a26abc32fe3eb62c96868731ca4b6`.
Fresh validation reproduced the checkpoint's topology and device provenance.

<a id="source-mnist_relu_drn_initialized_differential_10ep-epoch-budget-comparison"></a>

## Epoch-budget comparison

All first-ten epoch metrics match the earlier 20-epoch initialized-
differential run exactly. This makes the comparison a deterministic epoch-
budget screen rather than a separate stochastic replicate. The 20-epoch run
ultimately selected validation KL `0.0101526` and produced test KL `0.0101188`,
test accuracy `97.30%`, and agreement `98.98%`. Thus ten epochs produced
slightly higher test accuracy (`+0.18` percentage points, 18 examples), while
the longer run produced lower KL and higher teacher agreement. Because
selection minimizes validation KL and this is one seed, the accuracy
difference is not evidence that ten epochs generalize better.

The initialized ten-epoch result is also `3.54` percentage points above the
literal-RESET differential run's `93.94%` test accuracy. That gap is not a
clean initialization ablation: the literal-RESET run used paired hard-label
squared error with fixed gain one, while this run used teacher KL and a fitted
gain.

<a id="source-mnist_relu_drn_initialized_differential_10ep-matched-one-conductance-comparison"></a>

## Matched one-conductance comparison

The clean ten-epoch one-conductance control remained at `7.98%` test accuracy,
`7.81%` agreement, and KL `2.246097`. This differential pair is therefore
`89.50` accuracy points higher and has `99.37%` lower test KL under the same
teacher, data, seed, amplification, solver, objective, calibration set, and
minibatch budget. The nominal mappings chose the same scale fractions and
pre-programming gain, but independent per-device affine floors destroyed the
one-conductance score direction. The pair's shared reachable window preserved
direction through local common-component cancellation. See the
[initialized one-conductance report](../experiments/exp-007-single-ten-epoch.md#source-mnist_relu_drn_initialized_single_10ep-teacher-initialized-single-conductance-reram-mnist-fine-tuning-10-epochs).

Historical source: [mnist-differential-reram-initialized-finetune-10ep](../../../../../docs/experimental_manifest.md#mnist-differential-reram-initialized-finetune-10ep).

- **Outcome:** completed; ten epochs retained strong accuracy while the longer budget improved KL and teacher agreement
- **Interpretation:** completed; ten epochs retained strong accuracy while the longer budget improved KL and teacher agreement

## Interpretation and limitations

Ten epochs retain strong classification while 20 epochs improve KL. This is not an initialization-only RESET ablation because objective and gain differ. Its named local root is absent.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use ten epochs for a bounded screening comparison while retaining KL and agreement; do not treat its test-accuracy advantage over 20 epochs as established.
