---
id: "exp-003"
title: "128-pulse successor specification"
status: "imported"
hypotheses: ["H-003"]
---

# exp-003 — 128-pulse successor specification

## Question and provenance

128-pulse successor specification. Reducing the maximum target-programming budget from 512 to 128 pulses removes long soft-bound chases and noise-selected late admissions while retaining the endpoint distribution of persistently reachable targets well enough to fit the deployment noise, failure, saturation, and cost models.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-003-cap128-successor.md).

Dependencies and related controls: [Short-production programming and endpoint model](../results/exp-002-short-production.md).

## Frozen plans

- [IBM ReRAM endpoint model with a 128-pulse write-verify cap](../../../../../studies/ibm-reram-program-verify-noise-20260822-v3.json) — `ibm-reram-program-verify-noise-20260822-v3`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.

Historical document locations (now references/forwarding pages): [ibm_reram_short_production_review.md](../../../../../docs/ibm_reram_short_production_review.md).

<a id="source-ibm_reram_short_production_review-post-completion-128-pulse-successor-decision"></a>

## Post-completion 128-pulse successor decision

This decision was made after v2 completed and therefore does not change the
v2 protocol, configurations, or interpretation. The completed validation
ledger showed that all one-pulse failures and 99.96% of adaptive failures at
the 512-pulse cap had target windows outside the paired persistent reachable
interval. Failed trajectories averaged only 0.03 reversals for one-pulse and
0.12 for adaptive control, so repeated overshoot correction was not the
dominant source of the long tail.

The table below is a retrospective prefix calculation on the v2 held-out
validation trajectories. `Success @128` counts endpoints first accepted by
pulse 128; pulse-work reduction replaces every observed count above 128 by
128. It is not an exact counterfactual adaptive replay because truncating an
adaptive batch at the new cap can add a final verify that was absent from the
512-pulse trajectory.

| Population | Controller | Success @128 | Success @512 | Estimated pulse-work reduction |
|---|---:|---:|---:|---:|
| OM continuous | adaptive | 87.03% | 91.54% | 48.68% |
| OM continuous | one-pulse | 93.85% | 95.44% | 46.46% |
| OM corrupt | adaptive | 79.27% | 84.03% | 58.65% |
| OM corrupt | one-pulse | 87.10% | 89.25% | 59.33% |
| HfO2 continuous | adaptive | 96.80% | 98.73% | 39.07% |
| HfO2 continuous | one-pulse | 99.16% | 99.74% | 21.97% |
| HfO2 corrupt | adaptive | 91.86% | 95.24% | 56.51% |
| HfO2 corrupt | one-pulse | 97.70% | 99.34% | 34.38% |

A 64-pulse cap was rejected as the first successor because it reduced OM
adaptive success to 77.83% without corruption and 70.78% with corruption.
Among validation endpoints accepted only after pulse 128, 7,021 of 9,746
adaptive cases and 3,315 of 3,989 one-pulse cases were outside the diagnostic
persistent reachable interval. The remaining reachable tail is why 128 is a
new comparison rather than being declared equivalent to 512 from prefix data.

The successor contract is
`studies/ibm-reram-program-verify-noise-20260822-v3.json`. It preserves the v2
population, seeds, partitions, targets, tolerance, starts, controllers,
conditioning, and analysis, while selecting the strict
`production_short_cap128` profile. Only an exact v3 rerun may provide the
128-pulse endpoint and failure models used for deployment.

After preparing that study, the reviewed launcher selects it explicitly:

```bash
python -m experiments.reram_program_verify.local_short_launcher \
  --study-profile v3-cap128 \
  --aihwkit-python /home/filip/miniconda3/envs/aihwkit/bin/python
```

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
