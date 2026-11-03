---
experiment: "exp-002"
evidence: "imported-summary"
summary: "Corrected HWA lowers apparent validation KL by 0.7874% relative to matched clean training."
verdicts: {"H-002": "supports"}
---

# Result: exp-002 — Corrected fixed-array common-window HWA

[Experiment and historical specification](../experiments/exp-002-fixed-array.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v1](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v1) | failed: 2 | 0 |
| [mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2) | complete: 2 | 2 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v1

Saved summary: [summary.json](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v1/analysis/summary.json). Generated `2026-08-24T08:56:31.738543+00:00`; recorded state `incomplete`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| train-clean-common-window | 1 | 0 | 1 | 0 | 0 | False |
| train-exact-bounds-hwa | 1 | 0 | 1 | 0 | 0 | False |

Current metadata reconciliation: 0 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v1/analysis/summary.json).

### Recorded study coverage: mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2

Saved summary: [summary.json](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2/analysis/summary.json). Generated `2026-08-24T14:05:57.497854+00:00`; recorded state `reviewed`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| train-clean-common-window | 1 | 1 | 0 | 0 | 0 | True |
| train-exact-bounds-hwa | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 2 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2/analysis/summary.json), [analysis/review.json](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2/analysis/review.json), [analysis/final.json](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2/analysis/final.json).

## Observations

Corrected HWA lowers apparent validation KL by 0.7874% relative to matched clean training.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/train-clean-common-window/20260824T092611.618400Z-4e49cb59-31086fe6](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2/runs/train-clean-common-window/20260824T092611.618400Z-4e49cb59-31086fe6/result.json) | complete | selected student accuracy: 51.80%; initial student accuracy: 51.80%; terminal student accuracy: 51.80% |
| [runs/train-exact-bounds-hwa/20260824T092611.622367Z-d7ba7637-7565051f](../../../../../results/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2/runs/train-exact-bounds-hwa/20260824T092611.622367Z-d7ba7637-7565051f/result.json) | complete | selected student accuracy: 62.98%; initial student accuracy: 51.80%; terminal student accuracy: 62.98% |

<a id="source-ibm_om_hwa_program_verify_pilot-corrected-fixed-array-common-window-pilot"></a>

## Corrected fixed-array common-window pilot

The corrected two-arm study
[`mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2.json`](../../../../../studies/mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2.json)
completed on 2026-08-24 and is artifact-verified as `ready_for_review`. It
tests one fixed counterfactually repaired OM array, not the published-corrupt
or held-out-array cases. Both arms use the same assignment, four-cell
dual-rail grouping, pulse-resolved selection stream, and apparent-forward
endpoint policy. The HWA arm is the only arm whose minibatch forward pass uses
the fixed-array compact endpoint model.

For each logical synapse, the mapper intersects the four assigned device
ranges and places the clean nominal targets in the central 50 percent of that
common range. The fixed assignment has 39,700 quads. Four W1 quads have an
empty intersection; all other quads have zero mapped cells outside their own
bounds. Compact HWA therefore uses fitted endpoints for 158,792 cells and the
predeclared exact pulse controller for the eight cells in those four empty
quads. All eight fallback cells programmed successfully, with 10.125 mean,
7 median, and 26 maximum pulses; none exhausted the budget or became
non-finite. These fallback statistics describe the final compact-HWA
realization recorded with the selected checkpoint, not an aggregate over all
minibatches.

| Training arm | Selected epoch | Clean accuracy | Pulse-deployed accuracy | Teacher agreement | Deployment KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | initial (-1) | 93.58% | 51.80% | 53.20% | 0.054706 |
| fixed-array HWA | 10 (stored index 9) | 86.64% | 62.98% | 64.60% | 0.054275 |

Relative to the clean-training deployment control, fixed-array HWA improves
apparent-forward accuracy by 11.18 percentage points and teacher agreement by
11.40 points. It reduces KL by 0.000431, or 0.787 percent, while reducing
clean logical accuracy by 6.94 points. The best physical accuracy during HWA
was 63.54 percent after epoch 6, but the predeclared KL selector chose the
final epoch. Thus HWA closes 26.8 percent of the initial 41.78-point
teacher-to-deployment accuracy gap and leaves a 30.60-point gap.

The matched deployment programming statistics are almost unchanged: the
clean control accepts 157,673 of 158,800 cells (99.290 percent), while the HWA
checkpoint accepts 157,657 (99.280 percent). The selected HWA deployment uses
21.063 mean and 11 median pulses, with 1,143 cells exhausting the 128-pulse
budget and no non-finite endpoints. Accepted apparent residual RMSE is
0.01369, versus 0.05655 for the hidden persistent state. Forward inference
uses the apparent endpoint; both states and the RNG continuation are retained
in the deployment sidecar for a later two-state update experiment. A recovery
run must explicitly load that sidecar, verify its selected-weight digest, and
prove the two-state restore/update semantics; the generic deployment field is
not the handoff.

This result supports the narrow conclusion that, with quad common-window
placement held fixed, exact fixed-array HWA materially improves deployment
accuracy on this one repaired array, but does not close the gap. It does not
establish that
on-chip training is required: that claim needs a predeclared matched recovery
arm starting from this exact two-state deployment bundle. It also does not
establish generalization to new assignments, behavior with published corrupt
devices, donor realism, or measured silicon beyond the AIHWKit OM fit.

Historical source: [mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2](../../../../../docs/experimental_manifest.md#mnist-ibm-om-common-window-hwa-program-verify-pilot-20260824-v2).

- **Outcome:** supported
- **Interpretation:** The evidence narrowly supports the predeclared hypothesis on this one fixed counterfactually repaired IBM OM array: array-specific compact-endpoint HWA reduced selected pulse-resolved apparent-forward validation KL from 0.0547058969 for the matched clean control to 0.0542751399, an absolute reduction of 0.0004307570 (0.7874 percent relative). The result supports the narrow HWA-versus-clean KL hypothesis, but it neither establishes satisfactory deployability nor demonstrates that on-chip recovery is required.

## Interpretation and limitations

The narrow fixed-array KL hypothesis is supported in the historical review; absolute deployability remains poor. The failed v1 and corrected v2 are attempts, not independent replications.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Retain the narrow fixed-array KL benefit and evaluate deployment loss mechanisms separately; recovery necessity is untested here.
