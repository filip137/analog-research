# Research questions and campaigns

The central question is when deployment defeats HWA and whether a matched post-deployment update closes the remaining gap. These campaigns cover measured-device evidence and the explicitly declared generic drift simulation study.

| Campaign | Research question |
|---|---|
| [HWA and programming for reliable inference under drift](../../on-chip-learning/campaigns/drift-mitigation/README.md) | Current priority: which device model and HWA/P&V/maintenance schemes should precede evaluating on-chip learning? |
| [OPT drift dispersion and on-chip recovery](opt-drift-recovery/README.md) | Does large drift dispersion require analog updates beyond calibration, refresh and digital-only recovery? |
| [Initialization and representation](initialization-and-representation/README.md) | Which initializations and conductance representations preserve a useful, trainable perfect-diode DRN? |
| [Measured-cohort adaptation](measured-cohort-adaptation/README.md) | What survives reassignment to held-out measured devices, and which updates recover performance at a useful write cost? |
| [Endpoint-model HWA and recovery](endpoint-hwa-recovery/README.md) | When do programming endpoint errors defeat HWA, and when can full or low-rank recovery repair the resulting function? |
| [IBM OM programming](ibm-om-programming/README.md) | How accurately and economically can the IBM fitted pulse plants program targets and retain distinguishable cell states? |
| [IBM OM deployment and recovery](ibm-om-deployment-recovery/README.md) | How do mapping, HWA, quantization and post-deployment pulses change network performance on fixed and held-out IBM OM arrays? |

## Reading and extending the evidence

Start with a campaign README, then its series and result notes. Experiment notes own specifications, dependencies and run locations; result notes own observations, interpretation, uncertainty and the next decision. Each raw comparison has one primary owner; cross-campaign dependencies link to that record.

The 2026-10-02 migration reconciles historical reports with local summaries and native metadata. Imported records remain labelled imported-summary. Neither old running statuses nor historical launch approvals assign current execution. Unavailable raw roots and partial coverage remain explicit. The frozen global indexes are provenance, not current queues.

For a new bounded question use one [pilot note](pilots/20260925-lora-bounded-learning.md) and the [pilot template](../.agents/skills/experiment-loop/references/pilot.md). Sustained work belongs in the appropriate campaign series using the [record conventions](../.agents/skills/experiment-loop/references/artifacts.md). Generate that campaign’s ledger after changing its notes.

The existing runner.py, schema.py and [executable manifests](manifests/README.md) remain subprocess orchestration. Historical JSON plans remain in studies; large bundles remain in [results](../results/README.md). The workflow-validation pilot retains its own synthetic-control evidence and is not counted as measured-device science.

[Technical documentation and historical sources](../docs/README.md) · [Execution and review policy](../docs/experiment_workflow.md).
