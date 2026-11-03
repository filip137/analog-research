---
experiment: "exp-001"
evidence: "imported-summary"
summary: "Cohort-A runs establish historical feasibility under their recorded circuit semantics."
verdicts: {}
---

# Result: exp-001 — Raw and isotonic cohort-A feasibility

[Experiment and historical specification](../experiments/exp-001-cohort-a.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [measured-cohort-a-comparison-20260814-v1](../../../../../results/measured-cohort-a-comparison-20260814-v1) | No native status records | 0 |
| [measured-cohort-a-isotonic-mnist-exploratory-20260814-v1](../../../../../results/measured-cohort-a-isotonic-mnist-exploratory-20260814-v1) | complete: 1 | 1 |
| [measured-cohort-a-isotonic-mnist-test-20260814-v1](../../../../../results/measured-cohort-a-isotonic-mnist-test-20260814-v1) | complete: 1 | 1 |
| [measured-cohort-a-raw-mnist-exploratory-20260814-v1](../../../../../results/measured-cohort-a-raw-mnist-exploratory-20260814-v1) | complete: 1 | 1 |
| [measured-cohort-a-raw-mnist-test-20260814-v1](../../../../../results/measured-cohort-a-raw-mnist-test-20260814-v1) | complete: 1 | 1 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [summary.json](../../../../../results/measured-cohort-a-comparison-20260814-v1/summary.json).

## Observations

Cohort-A runs establish historical feasibility under their recorded circuit semantics.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [20260814T161241.592159Z-5d59ec26-6587ee5a](../../../../../results/measured-cohort-a-isotonic-mnist-exploratory-20260814-v1/20260814T161241.592159Z-5d59ec26-6587ee5a/result.json) | complete | selected accuracy: 94.26%; terminal accuracy: 94.16% |
| [20260814T163738.553827Z-ab03cb5d-232797b8](../../../../../results/measured-cohort-a-isotonic-mnist-test-20260814-v1/20260814T163738.553827Z-ab03cb5d-232797b8/result.json) | complete | evaluation accuracy: 94.91% |
| [20260814T155503.075028Z-7851be69-570dcc06](../../../../../results/measured-cohort-a-raw-mnist-exploratory-20260814-v1/20260814T155503.075028Z-7851be69-570dcc06/result.json) | complete | selected accuracy: 95.10%; terminal accuracy: 94.94% |
| [20260814T161053.429239Z-ab03cb5d-b6455c49](../../../../../results/measured-cohort-a-raw-mnist-test-20260814-v1/20260814T161053.429239Z-ab03cb5d-b6455c49/result.json) | complete | evaluation accuracy: 95.97% |


## Interpretation and limitations

The historical 95.97% raw-trace reference carries the amplifier-index confound resolved by the RESET factorial. Raw versus isotonic projection and fresh tests remain separate cases.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Carry the historical amplifier-index qualification into every downstream comparison using this source checkpoint.
