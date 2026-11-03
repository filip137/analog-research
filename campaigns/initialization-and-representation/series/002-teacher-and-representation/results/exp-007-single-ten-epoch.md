---
experiment: "exp-007"
evidence: "imported-summary"
summary: "The historical single-conductance replay remains near chance at 7.98% test accuracy."
verdicts: {}
---

# Result: exp-007 — Ten-epoch teacher-initialized single-conductance control

[Experiment and historical specification](../experiments/exp-007-single-ten-epoch.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| `results/mnist-single-reram-initialized-finetune-10ep-20260817-v1` — unavailable locally | Unavailable locally | 0 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

The historical single-conductance replay remains near chance at 7.98% test accuracy.


<a id="source-mnist_relu_drn_initialized_single_10ep-interpretation-boundary"></a>

## Interpretation boundary

This is the explicit ten-epoch replay of the measured one-conductance arm in
the earlier 20-epoch KD study. It is a matched control for the initialized
`G+ - G-` run in teacher, data, seed, solver, objective, calibration, device
cohort, amplification, and minibatch budget. The architecture-appropriate
initial mappings differ: one conductance uses its device-local affine trace
range, while a differential pair uses the two traces' shared reachable window.

<a id="source-mnist_relu_drn_initialized_single_10ep-result"></a>

## Result

The campaign completed from clean source commit
`c507dd9d13825721aa4e25c05f2303302c2a050c`. Both stages exited zero, the
training stream contains one initialization record followed by exactly ten
completed epochs, and no non-finite value or error marker was present.

The nominal teacher-map search selected scale fractions `[1.0, 0.125]`, with
calibration KL `0.00970462` and gain `0.630957` before measured programming.
After projection onto independently assigned device-local affine ranges, the
calibration KL became `2.24954` and the fitted gain hit its lower grid boundary
at `0.001`. The programming step therefore destroyed the useful direction of
the nominal teacher scores; this was not merely a uniform attenuation that a
positive gain could undo.

| Evaluation point | KL | Accuracy | Teacher agreement |
| --- | ---: | ---: | ---: |
| mapped initialization, validation | 2.242688 | 8.10% | 8.12% |
| selected epoch 10, validation | 2.242688 | 8.16% | 8.16% |
| selected checkpoint, held-out test | 2.246097 | 7.98% | 7.81% |

The selected validation KL improved by only `1.23e-7` in absolute terms, a
relative recovery of `5.48e-8`; the declared 10% recovery gate failed. The
output-layer aggregate conductance statistics did not change, while the hidden
layer's mean conductance shifted by only `1.95e-11 S`. The clipped-small
readout gain also scales the KL gradient reaching the circuit, and almost all
shadow updates remained below the raw-trace projection resolution. Ten more
epochs in the earlier study did not alter that conclusion.

Training took `182.01 s`, and fresh-process evaluation of all 10,000 held-out
examples took `1.42 s`. The teacher itself reached `97.36%` test accuracy.

<a id="source-mnist_relu_drn_initialized_single_10ep-matched-differential-comparison"></a>

## Matched differential comparison

| Evaluation | One conductance | `G+ - G-` pair |
| --- | ---: | ---: |
| initial validation accuracy | 8.10% | 89.12% |
| initial validation KL | 2.242688 | 0.231839 |
| selected validation accuracy | 8.16% | 97.42% |
| selected validation KL | 2.242688 | 0.0138965 |
| test accuracy | 7.98% | 97.48% |
| test agreement | 7.81% | 98.76% |
| test KL | 2.246097 | 0.0140905 |

The differential pair gains `89.50` percentage points of test accuracy and
reduces test KL by `99.37%`. Both arms use the same teacher, data, seed,
amplification, objective, solver, calibration set, minibatch count, and
gradient-equivalent learning rates. Their architecture-appropriate mapping is
the material distinction: per-device affine floors remain as unrelated terms
in the one-conductance dual rails, whereas the pair's shared reachable window
allows a local common component to cancel in `G+ - G-`.

This does not mean that one-conductance DRNs cannot learn. The corrected
from-RESET paired-MSE controls reached about `93.9%`; it means this particular
ReLU-to-independent-affine-device programming scheme destroys the teacher
signal and then locks KL fine-tuning near the uniform-output solution.

<a id="source-mnist_relu_drn_initialized_single_10ep-artifact-integrity"></a>

## Artifact integrity

Fresh validation reproduced model-local stages
`[(0, 1, 1, 1), (1, 2, 4, 0.0625)]`, teacher and device-source hashes, and the
two stable keys `base.dense_weight.0` and `base.dense_weight.1`. The selected
weights SHA-256 is
`1096560c3ce5d06b299948d9264caf32c6ea9d9e1dbfe8b6ba47a416047a5bfd`;
the exact epoch-10 resume SHA-256 is
`d90785dc326d5f78283b7e4df6b5ae25e6366277d18f9773c32223cbe857f806`.
The complete initialization and first-ten trajectory exactly match the earlier
20-epoch measured one-conductance run.

Historical source: [mnist-single-reram-initialized-finetune-10ep](../../../../../docs/experimental_manifest.md#mnist-single-reram-initialized-finetune-10ep).

- **Outcome:** completed; ReLU programming and ten KL epochs did not rescue the measured one-conductance scheme
- **Interpretation:** completed; ReLU programming and ten KL epochs did not rescue the measured one-conductance scheme

## Interpretation and limitations

The matched mapping loses useful score direction and calibrated gradient scale. This supports a failure of this measured representation, not a generic failure of DRN learning. Its named local root is absent.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Separate mapping cancellation and projection-resolution effects before proposing another longer run.
