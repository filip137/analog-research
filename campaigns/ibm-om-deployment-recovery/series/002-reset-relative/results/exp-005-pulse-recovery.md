---
experiment: "exp-005"
evidence: "imported-summary"
summary: "All twelve terminal recovery arms fall below the 92.74% initial apparent validation accuracy."
verdicts: {"H-005": "contradicts"}
---

# Result: exp-005 — Matched refresh, direct-pulse and Tiki-Taka recovery

[Experiment and historical specification](../experiments/exp-005-pulse-recovery.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1) | complete: 13, failed: 1 | 13 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

### Recorded study coverage: mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1

Saved summary: [summary.json](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/analysis/summary.json). Generated `2026-08-26T06:06:40.786433+00:00`; recorded state `ready_for_review`; recorded validation mode `full_artifact_hashes`.

| Arm | Expected | Complete | Failed | Running | Invalid | Recorded covered |
|---|---:|---:|---:|---:|---:|---|
| deploy-source-development | 1 | 1 | 0 | 0 | 0 | True |
| rail-refresh-budget-0p1 | 1 | 1 | 1 | 0 | 0 | True |
| rail-refresh-budget-1 | 1 | 1 | 0 | 0 | 0 | True |
| rail-refresh-budget-4 | 1 | 1 | 0 | 0 | 0 | True |
| direct-pulse-budget-0p1 | 1 | 1 | 0 | 0 | 0 | True |
| direct-pulse-budget-1 | 1 | 1 | 0 | 0 | 0 | True |
| direct-pulse-budget-4 | 1 | 1 | 0 | 0 | 0 | True |
| tiki-taka-ideal-budget-0p1 | 1 | 1 | 0 | 0 | 0 | True |
| tiki-taka-ideal-budget-1 | 1 | 1 | 0 | 0 | 0 | True |
| tiki-taka-ideal-budget-4 | 1 | 1 | 0 | 0 | 0 | True |
| tiki-taka-physical-budget-0p1 | 1 | 1 | 0 | 0 | 0 | True |
| tiki-taka-physical-budget-1 | 1 | 1 | 0 | 0 | 0 | True |
| tiki-taka-physical-budget-4 | 1 | 1 | 0 | 0 | 0 | True |

Current metadata reconciliation: 13 saved-summary terminal metric payloads equal their local native result payloads. No artifact hashes were recomputed.

Existing analyses and receipts: [analysis/report.md](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/analysis/report.md), [analysis/summary.json](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/analysis/summary.json).

### Recorded recovery source identities

These are stored source identities, checked for agreement in the available terminal metadata; the source bytes were not rehashed.

- source_deployment_sha256: `0b99f4dd73841a891993708bdace0ab508864e2e60e56620c165e3df99f006cd`.
- source_weights_sha256: `537681598b887d4d63c6429c7f7bee3a7082041215c09d6cb3c25947452f311a`.

## Observations

All twelve terminal recovery arms fall below the 92.74% initial apparent validation accuracy.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [runs/deploy-source-development/20260825T152340.475476Z-b5f2dd50-29c0e6b0](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/deploy-source-development/20260825T152340.475476Z-b5f2dd50-29c0e6b0/result.json) | complete | evaluation accuracy: 92.74% |
| [runs/direct-pulse-budget-0p1/20260825T153322.292937Z-8359c5d9-f0e44179](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/direct-pulse-budget-0p1/20260825T153322.292937Z-8359c5d9-f0e44179/result.json) | complete | terminal apparent validation: 89.36%; initial apparent validation: 92.74%; slow write requests: 15880; apparent logical sign flips: 953 |
| [runs/direct-pulse-budget-1/20260825T153752.454492Z-65cf024a-7cd1158a](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/direct-pulse-budget-1/20260825T153752.454492Z-65cf024a-7cd1158a/result.json) | complete | terminal apparent validation: 83.54%; initial apparent validation: 92.74%; slow write requests: 154705; apparent logical sign flips: 3236 |
| [runs/direct-pulse-budget-4/20260825T154252.550237Z-8efc9f83-c581ab36](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/direct-pulse-budget-4/20260825T154252.550237Z-8efc9f83-c581ab36/result.json) | complete | terminal apparent validation: 86.36%; initial apparent validation: 92.74%; slow write requests: 510880; apparent logical sign flips: 4084 |
| [runs/rail-refresh-budget-0p1/20260825T152537.029051Z-f8c3fa3b-67657d74](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/rail-refresh-budget-0p1/20260825T152537.029051Z-f8c3fa3b-67657d74/result.json) | complete | terminal apparent validation: 79.44%; initial apparent validation: 92.74%; slow write requests: 15880; apparent logical sign flips: 314 |
| [runs/rail-refresh-budget-1/20260825T152752.134882Z-b2f0a794-1d5b6f61](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/rail-refresh-budget-1/20260825T152752.134882Z-b2f0a794-1d5b6f61/result.json) | complete | terminal apparent validation: 78.58%; initial apparent validation: 92.74%; slow write requests: 158800; apparent logical sign flips: 536 |
| [runs/rail-refresh-budget-4/20260825T153037.273956Z-33898f33-9facda44](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/rail-refresh-budget-4/20260825T153037.273956Z-33898f33-9facda44/result.json) | complete | terminal apparent validation: 87.10%; initial apparent validation: 92.74%; slow write requests: 635200; apparent logical sign flips: 8 |
| [runs/tiki-taka-ideal-budget-0p1/20260825T154752.730813Z-e8cca17a-720bc473](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/tiki-taka-ideal-budget-0p1/20260825T154752.730813Z-e8cca17a-720bc473/result.json) | complete | terminal apparent validation: 10.06%; initial apparent validation: 92.74%; slow write requests: 12588; apparent logical sign flips: 622 |
| [runs/tiki-taka-ideal-budget-1/20260825T155207.909132Z-2c452c84-419cb9ce](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/tiki-taka-ideal-budget-1/20260825T155207.909132Z-2c452c84-419cb9ce/result.json) | complete | terminal apparent validation: 10.00%; initial apparent validation: 92.74%; slow write requests: 130195; apparent logical sign flips: 3708 |
| [runs/tiki-taka-ideal-budget-4/20260825T155623.136486Z-afed9c07-cb404207](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/tiki-taka-ideal-budget-4/20260825T155623.136486Z-afed9c07-cb404207/result.json) | complete | terminal apparent validation: 39.60%; initial apparent validation: 92.74%; slow write requests: 194899; apparent logical sign flips: 3768 |
| [runs/tiki-taka-physical-budget-0p1/20260825T160038.249123Z-004d1dc8-ebc3790c](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/tiki-taka-physical-budget-0p1/20260825T160038.249123Z-004d1dc8-ebc3790c/result.json) | complete | terminal apparent validation: 38.86%; initial apparent validation: 92.74%; slow write requests: 14252; apparent logical sign flips: 709 |
| [runs/tiki-taka-physical-budget-1/20260825T160538.405140Z-56eff422-3fdafabb](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/tiki-taka-physical-budget-1/20260825T160538.405140Z-56eff422-3fdafabb/result.json) | complete | terminal apparent validation: 31.46%; initial apparent validation: 92.74%; slow write requests: 137228; apparent logical sign flips: 3787 |
| [runs/tiki-taka-physical-budget-4/20260825T161138.653570Z-f5b181c6-f775fa84](../../../../../results/mnist-ibm-om-reset-relative-on-chip-recovery-pilot-20260825-v1/runs/tiki-taka-physical-budget-4/20260825T161138.653570Z-f5b181c6-f775fa84/result.json) | complete | terminal apparent validation: 50.44%; initial apparent validation: 92.74%; slow write requests: 425206; apparent logical sign flips: 4429 |

<a id="source-ibm_om_reset_relative_on_chip_recovery-evidence-boundary"></a>

## Evidence boundary

This is a one-seed development pilot. It uses the fitted AIHWKit OM preset and
host-computed gradients, not fabricated on-chip gradient circuitry. No test
array is released. A viable method and budget must be frozen before two more
development repeats and a separately declared formal deployment study.

The fixed p90/p95/p99 absolute-gradient follow-up is specified separately in
[`ibm_om_reset_relative_direct_gradient_thresholds.md`](../experiments/exp-006-gradient-thresholds.md#source-ibm_om_reset_relative_direct_gradient_thresholds-reset-relative-direct-gradient-thresholding).
It reuses this pilot's exact direct-pulse scales and does not redistribute
suppressed writes.


## Interpretation and limitations

At ten epochs the best original arm is direct pulses at cap 0.1 (89.36%). Refresh and both historical Tiki-Taka variants also degrade the apparent network. This contradicts useful terminal recovery in this pilot; it does not rule out other pulse algorithms. Failures/replacements and the later repaired TTv2 algorithm are distinct.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. The frontmatter verdict applies only to the imported hypothesis and original decision rule. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Do not recommend the original pulse rules as effective recovery on this deployment; inspect threshold and algorithm-repair follow-ups.
