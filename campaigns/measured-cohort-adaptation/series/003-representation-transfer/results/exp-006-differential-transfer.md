---
experiment: "exp-006"
evidence: "imported-summary"
summary: "Historical common-window differential adaptation reaches 97.81% test after severe transfer loss."
verdicts: {}
---

# Result: exp-006 — Differential A-to-B adaptation

[Experiment and historical specification](../experiments/exp-006-differential-transfer.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| `results/mnist-differential-reram-cohort-ab-adaptation-10ep-20260817-v1` — unavailable locally | Unavailable locally | 0 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

Historical common-window differential adaptation reaches 97.81% test after severe transfer loss.


<a id="source-mnist_differential_cohort_ab_adaptation-results"></a>

## Results

| Hardware state | Test accuracy | Teacher agreement | Test KL |
|---|---:|---:|---:|
| Cohort A, selected after 10 epochs | 97.48% | 98.76% | 0.01409 |
| Cohort B literal transfer, no adaptation | 9.45% | 9.45% | 2.23586 |
| Cohort B literal transfer, after 10 epochs | 92.90% | 93.95% | 0.18439 |
| Cohort B common-window transfer, no adaptation | 42.96% | 43.32% | 2.03767 |
| Cohort B common-window transfer, after 10 epochs | **97.81%** | 98.12% | 0.03156 |

Measured adaptation effects on the test set:

- Literal transfer: **+83.45 percentage points** accuracy and **91.75%** KL
  reduction. It recovers 94.80% of the classification-accuracy loss relative
  to the cohort-A source, but remains 4.58 points below it.
- Common-window transfer: **+54.85 percentage points** accuracy and **98.45%**
  KL reduction. Its final accuracy is 0.33 points above the cohort-A source,
  although its KL is still 2.24 times larger and its teacher agreement is 0.64
  points lower.

The selected checkpoint is the tenth completed B epoch in both adaptation
arms. On validation, literal transfer moves from 9.94% / KL 2.22985 to 91.44%
/ KL 0.22053; common-window transfer moves from 42.52% / KL 2.03568 to 97.50%
/ KL 0.03037.

<a id="source-mnist_differential_cohort_ab_adaptation-interpretation"></a>

## Interpretation

There is strong value in **adapting the physical conductances after hardware
transfer** under this measured-device model. A fixed scalar readout calibration
cannot repair either deployment, while device-constrained updates recover most
or all classification performance.

The transfer parameterization still matters substantially:

- Literal projection discards the pair relationship. The source differential
  RMS values are 3.68 uS and 1.06 uS by layer, while immediate B values become
  7.71 uS and 7.61 uS. This branch mismatch dominates the intended signed
  signal; the fitted gain collapses from 7,943 to 56.2.
- Common-window transfer preserves a shared baseline, but each edge receives a
  different reachable span. The mean overlap spans are only about 18.9 uS out
  of the nominal 110 uS range, 0.85--1.18% of pairs have no overlap, and the
  immediate differential RMS contracts to 0.836 uS and 0.528 uS. One scalar
  gain cannot undo this edge- and layer-dependent distortion.
- B adaptation reshapes the conductance differences. In the common-window arm,
  final differential RMS recovers to 3.32 uS and 0.799 uS, close enough to the
  source operating scale to restore accuracy.

The 0.33-point common-window accuracy advantage over cohort A should not be
read as evidence that B is intrinsically better: this is one deterministic
seed, no paired significance test was run, and the worse KL/agreement show that
the adapted B logits reproduce the teacher less faithfully despite slightly
more correct class decisions.

<a id="source-mnist_differential_cohort_ab_adaptation-scope-limits"></a>

## Scope limits

This experiment demonstrates the value of **analog/device-constrained weight
adaptation**, not a complete autonomous on-chip learning system. Gradients and
teacher targets are computed digitally, adaptation uses the full MNIST
training set for 10 epochs, and the measured backend uses a digital shadow with
global-nearest access to any point on a trace. It is not a sequential-pulse,
write-energy, endurance, latency, or few-shot adaptation model and may
overestimate what a physical programming circuit can recover economically.


## Interpretation and limitations

The result supports global-nearest measured adaptation on one assignment. Its named local result root is absent; it is not an autonomous pulse-local on-chip learning demonstration.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Recover the named raw bundle before a fresh audit; keep global-nearest feasibility distinct from local-pulse realizability.
