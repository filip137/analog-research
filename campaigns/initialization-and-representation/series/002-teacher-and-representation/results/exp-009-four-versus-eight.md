---
experiment: "exp-009"
evidence: "imported-summary"
summary: "Four and eight devices reach 97.46% and 97.48% test accuracy after adaptation."
verdicts: {}
---

# Result: exp-009 — Four versus eight measured devices

[Experiment and historical specification](../experiments/exp-009-four-versus-eight.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1](../../../../../results/mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1) | complete: 4 | 4 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [aggregate/results.json](../../../../../results/mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1/aggregate/results.json).

The existing presentation and figures remain at docs/mnist_four_vs_eight_device_validation_accuracy.tex and its referenced PNGs. They combine topology and IBM OM range illustrations and are not a new experiment.

## Observations

Four and eight devices reach 97.46% and 97.48% test accuracy after adaptation.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [stages/cohort_a_eight_device_differential_10ep/local/eight_device_differential_test/attempt-001/runs/20260819T140631.757216Z-df387013-025ced95](../../../../../results/mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1/stages/cohort_a_eight_device_differential_10ep/local/eight_device_differential_test/attempt-001/runs/20260819T140631.757216Z-df387013-025ced95/result.json) | complete | evaluation accuracy: 97.48% |
| [stages/cohort_a_eight_device_differential_10ep/local/eight_device_differential_train/attempt-001/runs/20260819T140137.302900Z-9c1d452d-cbbbaaa2](../../../../../results/mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1/stages/cohort_a_eight_device_differential_10ep/local/eight_device_differential_train/attempt-001/runs/20260819T140137.302900Z-9c1d452d-cbbbaaa2/result.json) | complete | selected student accuracy: 97.42%; initial student accuracy: 89.12%; terminal student accuracy: 97.30% |
| [stages/cohort_a_four_device_quad_common_window_10ep/local/four_device_quad_test/attempt-001/runs/20260819T140628.147513Z-6807ef5b-fc3a08e8](../../../../../results/mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1/stages/cohort_a_four_device_quad_common_window_10ep/local/four_device_quad_test/attempt-001/runs/20260819T140628.147513Z-6807ef5b-fc3a08e8/result.json) | complete | evaluation accuracy: 97.46% |
| [stages/cohort_a_four_device_quad_common_window_10ep/local/four_device_quad_train/attempt-001/runs/20260819T135747.947900Z-2385a4ba-aa544edb](../../../../../results/mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1/stages/cohort_a_four_device_quad_common_window_10ep/local/four_device_quad_train/attempt-001/runs/20260819T135747.947900Z-2385a4ba-aa544edb/result.json) | complete | selected student accuracy: 97.26%; initial student accuracy: 61.48%; terminal student accuracy: 97.26% |

<a id="source-dual_rail_input_four_vs_eight_devices-result"></a>

## Result

All four stages completed from clean commit
`bb88b007fa424f88bec86f8602690dac988b022a`. An immediate `--resume` replay
reused all four stages after validating their campaign fingerprints and
artifact hashes; it created no additional attempts. The 12 focused mapping
and circuit-equivalence tests also passed before launch.

| Arm | Devices per teacher weight | Selected layer scales | Initial validation | Selected validation | Selected checkpoint | Fresh test |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| Four-device quad window | 4 | `(1.0, 1.0)` | 61.48% | 97.26% | epoch 10 | **97.46%** |
| Eight-device differential | 8 | `(1.0, 0.125)` | 89.12% | 97.42% | epoch 9 | **97.48%** |

Selection minimized validation teacher KL, not accuracy. The corresponding
function-matching diagnostics are:

| Arm | Selected validation KL | Validation agreement | Fresh-test KL | Test agreement |
| --- | ---: | ---: | ---: | ---: |
| Four-device quad window | 0.0197817 | 98.74% | 0.0183103 | 98.68% |
| Eight-device differential | 0.0138965 | 98.94% | 0.0140905 | 98.76% |

The four-device arm passes the preregistered retention screen: its held-out
accuracy is only `0.02` percentage points (two of 10,000 examples) below the
eight-device arm, well inside the `1.0`-point threshold. Its test agreement is
`0.08` points lower. Its test KL is nevertheless `0.004220`, or `29.95%`,
higher, so equal classification accuracy should not be read as equal logit-
level approximation.

Requiring one intersection across all four conductance curves makes the
initialization more restrictive than the eight-device pairwise construction:

| Arm and layer | Window group | Empty windows | Mean baseline | Mean span | Initial projection RMS error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-device, layer 1 | 4 | 5.837% | 76.332 uS | 11.145 uS | 0.350 uS |
| Four-device, layer 2 | 4 | 6.000% | 76.303 uS | 11.244 uS | 0.272 uS |
| Eight-device, layer 1 | 2 | 1.351% | 70.730 uS | 18.383 uS | 0.238 uS |
| Eight-device, layer 2 | 2 | 1.550% | 71.066 uS | 18.068 uS | 0.287 uS |

Thus the four-cell intersection has only `61%`--`62%` of the pairwise mean
span and about four times the empty-window rate. No nominal targets were
clipped, but independently projecting the four symmetry-related targets onto
their measured pulse states still perturbs the cancellation. This explains
why the four-device initialization is materially below the eight-device
initialization (`61.48%` versus `89.12%`).

The strict quad window is still a large improvement over the earlier
four-device *pairwise*-window control: initial accuracy rises from `34.20%`
to `61.48%`; selected test accuracy rises from `96.96%` to `97.46%`; and test
KL falls from `0.041213` to `0.018310`, a `55.57%` reduction. Ten epochs then
recover `97.94%` of the quad arm's initial validation-KL gap. The eight-device
arm recovers `94.01%` and retains the better initialization and KL.

The exploratory conclusion is therefore narrow but positive: under this
frozen single-seed protocol, measured adaptation retains essentially all of
the eight-device classification benefit with four conductances per teacher
weight. The eight-device realization remains better if initialization-only
fidelity or close teacher-logit matching is the goal.

Historical source: [mnist-dual-rail-four-vs-eight-common-window-10ep](../../../../../docs/experimental_manifest.md#mnist-dual-rail-four-vs-eight-common-window-10ep).

- **Outcome:** completed; the four-device arm passed the preregistered post-adaptation accuracy-retention screen, while the eight-device arm retained better initialization and teacher-logit fidelity
- **Interpretation:** Four measured conductances per teacher weight retain essentially all classification benefit after adaptation while halving the conductance count relative to the eight-device realization. This is an exploratory single-seed result, not a universal equivalence claim.

## Interpretation and limitations

The single-seed retention gate passes; initialization and teacher KL favor eight devices. Exact reduction at a clamped input does not establish equivalent hidden-layer circuits or identical physical programming.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Choose four devices only against the declared area/accuracy target; retain eight-device initialization and KL advantages as tradeoffs.
