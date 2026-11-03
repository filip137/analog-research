---
experiment: "exp-004"
evidence: "imported-summary"
summary: "Historical differential RESET reaches 93.94% test accuracy; its named raw root is absent."
verdicts: {}
---

# Result: exp-004 — Ten-epoch differential RESET

[Experiment and historical specification](../experiments/exp-004-differential-reset.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| `results/mnist-differential-reram-reset-10ep-20260817-v1` — unavailable locally | Unavailable locally | 0 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

Historical differential RESET reaches 93.94% test accuracy; its named raw root is absent.


<a id="source-mnist_relu_drn_reset_differential_10ep-result"></a>

## Result

The campaign completed on 2026-08-17 from clean source commit
`187441fd11d27cdcc021aea35609703d5d4d6f8a`. The launcher exited zero, and both
the training and fresh-process test stages completed without non-finite values
or error markers. Learning-rate selection chose grid cell `cell_00`, with
rates `5.529865262288025e-7` and `1.7375886495148892e-7` for the first and
second differential pairs. This was the low/low edge of the bounded grid; the
declared protocol did not expand the grid, so this run does not establish that
either rate is globally optimal.

| Metric | Value |
| --- | ---: |
| Initial validation accuracy | 9.32% |
| Selected validation accuracy | 92.98% |
| Selected validation teacher agreement | 93.58% |
| Test accuracy | 93.94% |
| Test teacher agreement | 94.70% |
| Test paired squared error | 0.0933816 |
| Test raw teacher-to-student KL | 1.60862 |
| Diagnostic post-hoc calibrated test KL | 0.138329 |
| Teacher test accuracy | 97.36% |

The selected checkpoint is the tenth completed production epoch (zero-based
selection index `9`). The epoch-boundary resume checkpoint records epoch `10`.
The metrics stream contains exactly ten production records, with completed
epochs `1` through `10`; the training stage, including bounded learning-rate
selection, took 914.1 seconds.

Fresh-process validation reproduced the recorded model-local differential
semantics: resolved edges `(0, 1)` and `(1, 2)`, forward gains `1` and `4`, and
post-layer metrics `1` and `0.0625`. Generated diagnostic layer names differed
between the training and validation processes, as expected, but did not alter
the resolved equations. The selected named-weights catalog contains exactly
`base.conductance_plus.0`, `base.conductance_minus.0`,
`base.conductance_plus.1`, and `base.conductance_minus.1`.

Artifact integrity checks reproduced these hashes:

- selected weights:
  `9f73bfb5783ed1fce0d13a31df4eb1115da76ef1bff4b005de1c440a1047bb6b`;
- epoch-10 resume state:
  `21f32f510f3cbf199a5b4c5fb96875ec39199d459b64be78996841bce7e068e2`;
- learning-rate-selection record:
  `471be7bc391e203d074f793856582ffc0a7dae2033c896ca785e56154effdd43`.

The result demonstrates that the literal dual-RESET differential model can be
trained stably to 93.94% test accuracy under the declared ten-epoch protocol.
It does not yet estimate a differential-architecture benefit: the closest
single-device factorial arms used a 20-epoch budget, and a matched ten-epoch
single-device replay was not part of this campaign. It also does not test the
shared-reachable-zero initialization needed to distinguish independent RESET
baseline mismatch from common-floor cancellation. Those matched controls are
required before making a comparative architecture claim.

Machine-readable artifacts are under
`results/mnist-differential-reram-reset-10ep-20260817-v1/`. In particular,
`aggregate/results.json` records both completed stages, while each stage's
`result.json` records the numerical metrics and checkpoint provenance.

Historical source: [mnist-differential-reram-reset-10ep](../../../../../docs/experimental_manifest.md#mnist-differential-reram-reset-10ep).

- **Outcome:** completed; stable learning demonstrated, comparative benefit still open
- **Interpretation:** Literal dual-RESET differential training is numerically viable under this protocol. This run does not establish an architecture uplift because it lacks a matched ten-epoch one-device arm, and it does not isolate common-floor cancellation because independently assigned devices do not share an exactly matched RESET baseline. The selector also chose the low/low boundary cell of its declared non-expanding grid.

## Interpretation and limitations

Stable learning is reported, but the 20-epoch single-device factorial is not a matched architecture control. The raw run cannot be freshly checked locally.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

A matched ten-epoch single-device control is needed before attributing an architecture benefit; recover the named historical bundle before any new audit.
