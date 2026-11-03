---
experiment: "exp-003"
evidence: "imported-summary"
summary: "The clean arm completes; HWA has seven recorded epochs and no terminal result."
verdicts: {"H-003": "inconclusive"}
---

# Result: exp-003 — Interrupted eight-device differential-pair pilot

[Experiment and historical specification](../experiments/exp-003-eight-device.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1](../../../../../results/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1) | complete: 1, running: 1 | 1 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1

Saved summary: [summary.json](../../../../../results/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1/analysis/summary.json). Generated `2026-08-24T14:01:27.060739+00:00`; recorded state `running`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| train-clean-differential-pair | 1 | 1 | 0 | 0 | 0 | True |
| train-exact-bounds-pair-hwa | 1 | 0 | 0 | 1 | 0 | False |

Current metadata reconciliation: 1 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1/analysis/summary.json).

## Observations

The clean arm completes; HWA has seven recorded epochs and no terminal result.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/train-clean-differential-pair/20260824T120938.552401Z-c36c7dea-3f26e53a](../../../../../results/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1/runs/train-clean-differential-pair/20260824T120938.552401Z-c36c7dea-3f26e53a/result.json) | complete | selected student accuracy: 49.94%; initial student accuracy: 49.94%; terminal student accuracy: 46.84% |

<a id="source-ibm_om_hwa_program_verify_pilot-interrupted-v1-observations"></a>

### Interrupted v1 observations

The formal launcher passed every declared preflight and completed the clean
arm. The HWA arm emitted seven durable epoch records through epoch index `6`
(`24,066` updates) before a user-requested stop. The worker exited by SIGTERM;
there was no numerical error, but there is consequently no terminal HWA
`result.json`, selected deployment sidecar, or post-run artifact audit. These
measurements are partial evidence and must not be entered into the finalized
study ledger.

| State | Accuracy |
| --- | ---: |
| Clean initial network | 97.16% |
| Clean-control apparent programmed deployment | 49.94% |
| Partial HWA KL-selected clean checkpoint, epoch index 5 | 95.72% |
| Exact pair-window mapping of that selected checkpoint | 95.16% |
| Apparent programmed validation at selected epoch index 5 | 55.72% |
| Post-hoc best apparent accuracy, epoch index 6 | 60.36% |

The exact selected-checkpoint mapping replay used the saved seed-`84001`
population and the ordered 5,000-example validation cohort. Mapping changed
122 predictions (2.44%) and cost only 0.56 accuracy points, but compressed
calibrated score RMS from `10.25447` to `0.008294`, approximately 1,236-fold.
The subsequent selected pulse evaluation lost 39.44 points relative to the
ideal mapped state. The declared KL selector retained epoch index `5` even
though epoch index `6` had 4.64 points higher accuracy; its KL was worse by
only `0.0000868`.

A read-only tensor decomposition of the completed clean-control deployment
isolates why the programmed signal is fragile. Pair-window mapping reduced
the per-edge plus/minus difference RMS from `0.13669` to `0.04975` in W1 and
from `0.03746` to `0.01377` in W2. At the same time, the normalized pair-sum
mean rose from `0.10422` to `0.74341` in W1 and from `0.03043` to `0.71689` in
W2. The shared baseline cancels algebraically in `G+ - G-`, but it remains as
passive conductance loading. Apparent programming then produced logical
contrast SNR `2.97` (`9.46 dB`) in W1 and `0.882` (`-1.09 dB`) in W2, with
8.44% and 22.0% sign flips respectively. Accepted per-cell residual RMS was
approximately `0.01367` normalized (`1.50` microSiemens), while 2,292 total
budget failures supplied a much larger tail.

The reproducible read-only receipts are:

```text
results/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1/
  analysis/ideal_pair_window_mapped_selected_epoch5.json
  analysis/deployment_decomposition_clean_signal_noise.json
```

Do not pool this result with four-device v2 or label their difference an
eight-versus-four effect. The studies differ in teacher and base checkpoint,
encoding, conductance units, learning rates, physical population size and
fingerprint, and common-window grouping. A descriptive mechanistic table is
permitted only with those mismatches visible. This eight-device pilot has no
Tiki-Taka, LoRA, or soft SET/RESET arm; any claim that on-chip recovery is
needed still requires a later matched study starting from one frozen
persistent deployment bundle.


## Interpretation and limitations

The source records a user-requested SIGTERM. The remaining native running marker is historical, not a live monitoring assertion. Partial selected and post-hoc-best observations cannot replace a complete matched result.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Preserve the user-stopped partial arm; any continuation needs an explicit assignment and valid resume contract.
