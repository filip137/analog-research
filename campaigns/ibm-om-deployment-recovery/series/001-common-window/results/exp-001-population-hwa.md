---
experiment: "exp-001"
evidence: "imported-summary"
summary: "Four training arms complete; sixteen planned held-out evaluations are not covered."
verdicts: {"H-001": "inconclusive"}
---

# Result: exp-001 — Initial population HWA/programming pilot

[Experiment and historical specification](../experiments/exp-001-population-hwa.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-ibm-om-hwa-program-verify-pilot-20260822-v1](../../../../../results/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1) | complete: 4 | 4 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-ibm-om-hwa-program-verify-pilot-20260822-v1

Saved summary: [summary.json](../../../../../results/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1/analysis/summary.json). Generated `2026-08-22T23:36:55.392834+00:00`; recorded state `incomplete`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| train-clean | 1 | 1 | 0 | 0 | 0 | True |
| train-gaussian-3pct | 1 | 1 | 0 | 0 | 0 | True |
| train-om-repaired | 1 | 1 | 0 | 0 | 0 | True |
| train-om-published | 1 | 1 | 0 | 0 | 0 | True |
| test-clean-83101 | 1 | 0 | 0 | 0 | 0 | False |
| test-gaussian-3pct-83101 | 1 | 0 | 0 | 0 | 0 | False |
| test-om-repaired-83101 | 1 | 0 | 0 | 0 | 0 | False |
| test-om-published-83101 | 1 | 0 | 0 | 0 | 0 | False |
| test-clean-83102 | 1 | 0 | 0 | 0 | 0 | False |
| test-gaussian-3pct-83102 | 1 | 0 | 0 | 0 | 0 | False |
| test-om-repaired-83102 | 1 | 0 | 0 | 0 | 0 | False |
| test-om-published-83102 | 1 | 0 | 0 | 0 | 0 | False |
| test-clean-83103 | 1 | 0 | 0 | 0 | 0 | False |
| test-gaussian-3pct-83103 | 1 | 0 | 0 | 0 | 0 | False |
| test-om-repaired-83103 | 1 | 0 | 0 | 0 | 0 | False |
| test-om-published-83103 | 1 | 0 | 0 | 0 | 0 | False |
| test-clean-83104 | 1 | 0 | 0 | 0 | 0 | False |
| test-gaussian-3pct-83104 | 1 | 0 | 0 | 0 | 0 | False |
| test-om-repaired-83104 | 1 | 0 | 0 | 0 | 0 | False |
| test-om-published-83104 | 1 | 0 | 0 | 0 | 0 | False |

Current metadata reconciliation: 4 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1/analysis/summary.json).

## Observations

Four training arms complete; sixteen planned held-out evaluations are not covered.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/train-clean/20260822T223356.607607Z-ac195488-7724abed](../../../../../results/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1/runs/train-clean/20260822T223356.607607Z-ac195488-7724abed/result.json) | complete | selected student accuracy: 11.24%; initial student accuracy: 11.24%; terminal student accuracy: 11.24% |
| [runs/train-gaussian-3pct/20260822T223356.632496Z-dde03641-cc537b8b](../../../../../results/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1/runs/train-gaussian-3pct/20260822T223356.632496Z-dde03641-cc537b8b/result.json) | complete | selected student accuracy: 11.88%; initial student accuracy: 11.24%; terminal student accuracy: 11.96% |
| [runs/train-om-published/20260822T223356.597251Z-2d7bc722-6e0a64e8](../../../../../results/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1/runs/train-om-published/20260822T223356.597251Z-2d7bc722-6e0a64e8/result.json) | complete | selected student accuracy: 12.86%; initial student accuracy: 11.24%; terminal student accuracy: 12.86% |
| [runs/train-om-repaired/20260822T223356.627723Z-0e5f6c9d-9356d8f7](../../../../../results/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1/runs/train-om-repaired/20260822T223356.627723Z-0e5f6c9d-9356d8f7/result.json) | complete | selected student accuracy: 15.62%; initial student accuracy: 11.24%; terminal student accuracy: 15.62% |

<a id="source-ibm_om_hwa_program_verify_pilot-device-and-pulse-interpretation"></a>

## Device and pulse interpretation

The OM preset is IBM's optimized-material 14 nm ReRAM array fit. Its nominal
state is `w in [-1,1]` and its nominal minimum update is `dw_min=0.0949`, or
`0.04745` of the normalized full span after

```text
x = (w + 1) / 2.
```

Thus the nominal full-span arithmetic is about 21 minimum pulses. That is not
a programming-time guarantee. Sampled cell bounds, nonlinear soft-bound
response, cycle-to-cycle variation, verify reversals, and corrupt/stuck cells
change both success and cost. The HfO2 preset has a much larger nominal
minimum step (`dw_min=0.4622`, about 4.3 nominal full-span updates), which is
why HfO2 target pulse counts are much smaller. HfO2 remains a characterization
stress control and is not part of this first network pilot.

The program-and-verify controller starts directly from each sampled cell's
lower persistent bound, representing a fully RESET device after boundary
conditioning. Conditioning pulses are outside the 128 target-programming
pulse budget. Starting from logical zero would mix an arbitrary simulator
initial state into the programming result; starting from RESET instead gives
the physical procedure a declared and reproducible initial boundary.

At each verify read, the controller observes the apparent state:

```text
accept if abs(x_apparent - x_target) <= tau
SET if x_apparent < x_target - tau
RESET if x_apparent > x_target + tau.
```

Here `tau = 0.5 * dw_min / 2 = 0.023725` in normalized OM units. The adaptive
controller predicts how many identical pulses to batch, applies them
sequentially, verifies again, and reverses polarity only after the apparent
state crosses the acceptance window. It does not RESET after every rejected
SET. A trajectory stops on acceptance, a non-finite verify value, or 128
target-programming pulses.

The **apparent endpoint residual** is what the verify circuit admits and what
the DRN forward pass receives. The **persistent endpoint residual** is the
underlying programmed state after apparent write variation is removed. An
accepted residual is therefore effective endpoint programming error, not
pure write noise: it contains the acceptance window, pulse resolution,
stochastic pulse response, sampled bounds, and controller behavior. The
Wan-2022 comparator is an endpoint programming-noise model and does not expose
this pulse trajectory or the same failure/cost decomposition.

<a id="source-ibm_om_hwa_program_verify_pilot-training-phase-evidence"></a>

## Training-phase evidence

The four formal training arms completed on 2026-08-23. Each arm performed 10
epochs and exactly 34,380 optimizer updates. The artifact-verified study state
is intentionally `incomplete`: all four training arms have one valid complete
run and all 16 held-out validation arms remain untouched.

| Training arm | Selected epoch | Common-assignment KL | Student accuracy | Programming success | Pulses, mean / median | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | initial (-1) | 0.17078 | 11.24% | 56.07% | 68.05 / 63 | 10.2 min |
| Gaussian 3% | 8 | 0.16697 | 11.88% | 56.07% | 68.04 / 63 | 10.2 min |
| OM, repaired training population | 9 | 0.10565 | 15.62% | 56.81% | 67.31 / 59 | 62.7 min |
| OM, published corruption | 9 | 0.10538 | 12.86% | 56.86% | 67.11 / 58 | 62.7 min |

These are selection results on the single predeclared full-corrupt assignment
`83003`, not held-out test results. Relative to clean BPTT, Gaussian HWA
reduced selection KL by 2.23%, while repaired and published OM HWA reduced it
by 38.13% and 38.29%, respectively. The low absolute physical accuracies and
the reversal of the repaired-versus-published ordering between KL and accuracy
make the four held-out assignments necessary before interpreting robustness or
claiming a need for on-chip recovery.

All four selection-population NPZ files have SHA-256
`945cc8e62266c097d847982470d4869db9778b5b68240e5e16a8602c0ca93010`
and population fingerprint
`879f08d1474e6e9d2c1cf0f96ff22d1753b77b1968225ca4bd8e951fd50cf55e`.
For training assignment `83001`, the repaired and published populations have
byte-identical healthy-cell tensors and the same 21,466-site published-corrupt
mask; the repaired population activates zero corrupt cells while the published
population retains all 21,466. Every selected checkpoint is digest-linked to
its exact `ibm_om_deployment.pt` persistent state.


## Interpretation and limitations

Training completion does not establish the population deployment hypothesis. Retain model prerequisite and repaired/corrupt regimes; no recovery was assigned in this pilot.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Reconcile or collect the missing held-out cases before claiming population-level deployment benefit.
