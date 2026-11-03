---
id: "exp-019"
title: "Analog LoRA adapters on the frozen std3 HWA base: do stuck devices hurt adapters more than base MLPs?"
status: "complete"
hypotheses: []
---
# exp-019: analog LoRA adapters vs base MLP arrays under stuck devices

The user redirected the work on October 15, 2026, at about 20:25 (after
[exp-018](exp-018-finetuned-task-recovery.md) had started).

## Question and decision

Take the existing HWA-trained OPT-125M (the frozen std3 source of
exp-007/exp1g). Train per-task LoRA adapters on the six analog MLPs only, and
simulate those adapters on crossbar arrays too.

**Question:** at the same per-device stuck rate, do faults in the adapter
arrays hurt task accuracy more than faults in the base MLP arrays?

**Decision it informs:** whether task adapters on analog hardware need extra
protection (redundancy, digital adapters, fault-aware training), or whether
the base arrays remain the dominant vulnerability.

Possible outcomes:
- **Adapters more fragile.** The adapter-only accuracy loss exceeds the
  base-only loss, even though adapters hold about 1/38 of the devices.
- **Base dominates.**
- **Both negligible after the verify rewrite.**

## Cases and controls

User-confirmed choices are marked **(C)**; the rest are proposed and frozen
here before any run.

- **Base (C).** The std3 whole-model HWA source:
  - lifecycle `opt125m-cmo-l6-exp1e-digital-lr1em5`, arm `standard_hwa_model`,
    selected epoch 98;
  - `source.pt` SHA-256 `86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028`;
  - pinned teacher OPT-125M `27dcfa74`.

  The base's analog targets, digital copy and calibration stay frozen (checked
  bitwise).
- **Adapters (C).** Rank 16, α 32 on fc1/fc2 of layers 6–11 (`network.analog_lora`).
  - Each base matrix W gets A (16 × in) and B (out × 16) as additional analog
    matrices: 737,280 adapter weights, against 28,311,552 base weights.
  - The adapter path is x → A → B, digitised between the two arrays and scaled
    by α/r, then added before the frozen output gain and bias.
  - Adapters use per-output-channel abs-max scaling and the same
    active/reference CMO pairs, P&V and fault model as the base.
  - Each adapted matrix's A and B form their own fault group, so base fault maps
    are identical with or without adapters.
- **Adapter training (C: off-chip, HWA-noise-aware).**
  - Adam on logical adapter masters (A uniform ±1/√in, B = 0).
  - Each step: std3 Gaussian noise (0.03 × output-channel abs-max) on every base
    and adapter weight, straight through.
  - Objective: multiple-choice cross entropy over per-token mean continuation
    log-likelihoods, on all task-train examples (37,905 / 64,349 / 14,612).
  - Batch 32 examples; epochs 2 / 2 / 4 (HellaSwag / SST-2 / PIQA); linear
    warm-up over 6% of updates, then cosine to zero; seed 41.
  - Three arms per task: LR 1e-4, 3e-4 and 1e-3.
  - Checkpoint selection (epoch 0 = base without adapters, then each epoch):
    mean development acc_norm on the healthy selection arrays 211001 and 211002
    after real P&V of base and adapters.
  - **Arm selection:** the arm with the highest selected development score
    (ties: lower LR). Only that arm is deployed on the evaluation array.
- **Tasks (C).** HellaSwag, SST-2 and PIQA. Same task files, splits and
  lm-eval templates as exp-018; official validation is the test split.
- **Technology.** CMO, 2% `gaussian_endpoint` P&V, no relaxation or read
  noise, tiles 256, 48 arrays per die.
- **Faults (C: base-only vs adapter-only vs both).** Hierarchical, permanent,
  no concentrations. Cases:
  - `none`;
  - random 1%, random 3% and gmax 3%, each with targets all, `base` and
    `adapter`.

  That gives 10 cases. A targeted case fails exactly the devices of that subset
  in the matching all-array case.
- **Arms.**
  - `none`: raw deployment, reference only.
  - `pair_rewrite_verify`: the standing default verify partner rewrite, 5σ,
    applied to every array including the adapters.
  - No on-chip learning.
- **Array.** Assignment 271002 only (user preference: one array per task).

## Metrics and decision

**Primary metric:** test `accuracy_norm_percent` (paired over test examples).

Companions:
- agreement with the clean digital adapted model, from per-example
  predictions;
- the adapter gain over the base (epoch 0);
- choice KL against the pinned teacher (context only);
- failed devices per subset.

```text
Unit: task × fault kind/rate × arm (raw none | rewrite), on array 271002. Never pooled.
loss(target) = acc_norm(healthy deployment, same arm) − acc_norm(target case, same arm), paired.
  SE = sqrt(b + c − (b − c)²/n) / n, where b and c are the discordant counts.
Adapters more fragile at (kind, rate, arm):
  loss(adapter) − loss(base) ≥ 0.5 pt and ≥ 2·SE of their paired difference.
Base more fragile: the mirror condition. Otherwise comparable / inconclusive.
Per-device fragility (reported, not decisive): loss per 1,000 failed devices for each subset.
Additivity check: loss(all) against loss(base) + loss(adapter).
The verdict is stated for the rewrite arm (deployed default) and the raw arm (reference).
```

**Evidence class:** exploratory, model-based, single array; one HWA base and
one adapter seed per task.

**Not claimed:** generalisation across arrays, ranks or seeds; on-chip
adapter training (a possible follow-up).

## Implementation

The whole experiment runs inside the lifecycle (contract updated):
- `network.analog_lora` (`workflow/networks/opt_mlp.py`).
- `defects.cases[].target` (`workflow/devices.py`).
- HWA arm `method: lora` with `base_source`, `learning_rate` and
  `warmup_fraction` (`workflow/hwa.py`, `train_lora`), plus a frozen base input
  `--lora-base ARM=PATH` that must match the arm's SHA-256.
- `task.training_examples` cohort and the HWA objective `task_choice_token_mean`.
- A LoRA source's provenance includes the `task` section.

Tests: `tests/test_workflow_analog_lora.py` passes, including a tiny end-to-end
run (base HWA → LoRA → targeted faults → rewrite → collect).

**Lifecycles:** `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp19-analog-lora-{hellaswag,sst2,piqa}.json`,
written by `launch/exp19_lifecycles.py`.

## Budget, stop and expected outputs

- **Compute:** at most 8 GPU-hours on loulou and fifi, plus idle 5090s after
  22:00. Estimate:
  - LoRA training: 3 arms × (≈15 / 8 / 6 min);
  - deployment and rewrite evaluation of 10 cases: ≈50 min HellaSwag, ≈10 min
    each for the others.
- **Storage:** at most 100 GB on the workers.
- **Stop rules:**
  - a nonfinite LoRA objective or a failed export or base-unchanged gate →
    escalate;
  - a selected adapter that does not beat the base (epoch 0) on development
    data is reported as is; its "adapter" faults then test an unused path.
- **Outputs:** raw runs under `results/lifecycles/opt125m-cmo-l6-exp19-*/` on
  the workers, and the result note `results/exp-019.md`.

## Execution and monitoring handoff

**Source.** Staged snapshot v5 `…/opt-cmo-exp1/hwa-exp19-analog-lora-20261015`
on loulou and fifi: base `b0c1f06` plus the working tree, receipt `aa94b238…`,
with a local throwaway git commit for the runner.
- Base source: `inputs/exp1g-frozen-sources/std3/…/checkpoints/source.pt`
  (`86522e23…`), on both workers.
- Inputs: `inputs/exp18/` (task files and streams).
- Driver: [`launch/exp19_drive.py`](../../../launch/exp19_drive.py), one per
  worker:
  - plan with `--lora-base`;
  - run prepare, devices and the three LoRA arms;
  - select the arm by development score, recorded in
    `results/lifecycles/<id>/arm-selection.json`;
  - run deploy and on-chip for that arm on 271002 (trimmed manifests, resume).

**Runs.**
- **HellaSwag:** loulou, tmux `exp19-loulou`, started 20:47.
- **SST-2, fifi:**
  - The first start at 20:39 began early beside an exp-018 job on the same GPU,
    because the wait condition was wrong. The operator stopped it in its prepare
    stage, and the attempt is kept.
  - The requeued driver then failed at re-planning, because the interrupted run
    had left a trimmed manifest in the plan directory.
  - Rerun queued (tmux `exp19-fifi-sst2`).
- **PIQA, fifi:**
  - Started 20:50. Arms `lora_lr1em4` and `lora_lr1em3` completed: about 4.6 min
    each, 8.6 GB peak, base unchanged.
  - `lora_lr3em4` failed with signal −7 (SIGBUS), the fifi driver instability
    seen before.
  - Resume retry queued after SST-2 (tmux `exp19-fifi-piqa-retry`); it reruns
    only that arm.

**First observation (development only):** PIQA development acc_norm on the
healthy P&V selection arrays went from 61.33 (base, epoch 0) to 64.40
(`lora_lr1em3`, epoch 2), and to 62.30 for `lora_lr1em4`.

**Arm selection (development only).** Mean development acc_norm on the P&V
selection arrays, at each arm's selected epoch; `arm-selection.json` holds the
record.

| Task | LR 1e-4 | LR 3e-4 | LR 1e-3 | Selected |
| --- | --- | --- | --- | --- |
| HellaSwag | 39.13 (epoch 2) | 42.85 (2) | **44.38** (2) | `lora_lr1em3` |
| SST-2 | 91.20 (2) | 92.97 (2) | **93.93** (2) | `lora_lr1em3` |
| PIQA | 62.30 (4) | **64.53** (3) | 64.40 (2) | `lora_lr3em4` |

Every selected epoch beats epoch 0 (the base without adapters).

**Deployment.** The runner keeps one resolved contract per campaign ID and
output root, so the driver's in-place deploy phase was refused (exit 2). Each
selected arm's deploy and on-chip stages therefore run as campaign
`<lifecycle_id>-deploy`, built by
[`launch/exp19_deploy.py`](../../../launch/exp19_deploy.py):
- the same stage configs and worktree;
- the upstream cache, device bundle and LoRA source given as explicit paths.

Status:
- **SST-2 and PIQA:** complete on fifi (20 rows each).
- **HellaSwag:** loulou, tmux `exp19-hs-deploy`, started 21:55. A first waiting
  session hung for about 30 minutes, because `tmux has-session -t exp19-loulou`
  prefix-matched its own name.
- **Review:** `results/lifecycles/exp19-collected/review.json` (local),
  produced by [`launch/exp19_report.py`](../../../launch/exp19_report.py).
