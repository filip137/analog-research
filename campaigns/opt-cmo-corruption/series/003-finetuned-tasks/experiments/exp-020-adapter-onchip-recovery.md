---
id: "exp-020"
title: "On-chip recovery of analog-LoRA deployments: adapter-only vs all-weight learning vs calibration"
status: "complete"
hypotheses: []
---
# exp-020: on-chip recovery of the exp-019 analog-LoRA deployments

The user assigned this follow-up to [exp-019](exp-019-analog-lora.md) on
October 15, 2026, at about 22:50.

## Question and decision

**Question:** after the verify partner rewrite, on the exp-019 deployments
(frozen std3 base plus rank-16 analog LoRA on array 271002), which recovers
more task accuracy, and at what write cost?
- digital calibration;
- on-chip learning of **only the adapter arrays** (737k weights);
- on-chip learning of **all analog weights** (29.0M).

**Decision it informs:** whether small analog adapters are a practical on-chip
recovery handle, cheap to write, compared with whole-array learning and
no-write calibration.

**Expected outcome:** little room at 3% (exp-019 left at most 0.8 pt after the
rewrite); possible gains at 6%.

## Cases and controls

User-confirmed choices are marked **(C)**; the rest are proposed and frozen
here.

- **Sources.** The development-selected exp-019 LoRA sources, reused unchanged
  through `--hwa-source`:
  - HellaSwag `lora_lr1em3`;
  - SST-2 `lora_lr1em3`;
  - PIQA `lora_lr3em4`.

  Network, data, task, devices, deployment, P&V and HWA arm are identical to
  exp-019.
- **Faults (C: harsher cases added).** Healthy, plus gmax 3%, gmax 6% and
  random 6%, each in all arrays, `base` only and `adapter` only: 10 cases. The
  rates and maps follow exp-019's construction.
- **Objective (C).** `source_kl`: KL to the **clean deployed source** (intended
  base and adapter weights, deployed calibration), recomputed by a frozen copy
  of the source on task text. Labels are unused.
- **Arms.** Everything after the rewrite (detection by verify read, 5σ):
  - R0 `none` and R0c `calibration`: unrewritten references;
  - R1 rewrite hold;
  - R2 rewrite + calibration;
  - RA rewrite + adapter-only learning (`write_scope: adapter`; writes are
    eligible only on adapter devices, and zero base writes are asserted);
  - RW rewrite + all-weight learning.
- **On-chip law.** `cmo_thresholded_write_adam`: threshold 2σ, accumulated
  updates, at most 4 write attempts per device, 2,048 recovery sequences × 4
  epochs (256 updates, batch 32), seed 53. Intermediate epochs score
  development cohorts only; the final epoch is complete.
- **Learning rates.**
  - Three development screens per task on array 271002 at gmax 6% (all arrays),
    using development data only. The test split is never read for the choice.
  - Weight LR {1e-3, 3e-3, 1e-2}, selected **separately** for RA and RW
    (per-arm `learning_rate`). Calibration LR {1e-4, 3e-4, 1e-3} for R2.
  - Each is chosen by the highest final `task_development` agreement with the
    clean adapted model; ties go to the lower LR.
  - The screens share the evaluation array, a deviation from exp-018's separate
    screen array: exp-019's sources fix the device section.

## Metrics and decision

**Primary metric:** test acc_norm, paired.

**Companions:**
- agreement with the clean adapted model;
- write cost: issued attempts, adapter and base device writes, touched
  devices;
- the rewrite's programs.

```text
Unit: task × fault case, on array 271002, rows after the rewrite. Never pooled.
Δ_X = acc_norm(X) − acc_norm(R1), paired, with the discordant-pair SE, for X in {R2, RA, RW}.
X helps if Δ_X ≥ 0.5 pt and ≥ 2·SE. X hurts if Δ_X ≤ −2·SE. Otherwise no detectable effect.
Adapter-only vs calibration and all-weight: acc_norm(RA) − acc_norm(R2) and acc_norm(RA) − acc_norm(RW),
  same thresholds.
Report the write cost of RA and RW, and RA's base_device_writes, which must be 0.
On nominal, any Δ is adaptation without faults (guard).
```

**Evidence class:** exploratory, model-based, single array; one source per
task.

## Budget, stop and expected outputs

- **Compute:** fifi only. The other 5090s were busy with another user's jobs
  (riri and trex), and on loulou another user had just started an MPS server.
  Cap 8 GPU-hours, expected to finish on the morning of October 16.
- **Stop rules:** nonfinite values, any base write by RA, or failed export
  gates → escalate. An operational fifi crash is retried once with resume.
- **Outputs:**
  - lifecycles `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp20-*.json`;
  - frozen LRs `launch/exp20_frozen.json`;
  - the result note `results/exp-020.md`.

## Execution and monitoring handoff

**Implementation.**
- `onchip.objective: source_kl`, using a frozen deep copy of the deployed
  source as the reference.
- Per-arm `write_scope: adapter`, through the thresholded writer's eligibility
  mask, with `base_device_writes` asserted 0.
- Per-arm `learning_rate`.
- `onchip.epoch_evaluation: development`.
- Tests: `tests/test_workflow_analog_lora.py`; 488 workflow tests pass.

**Launch.** October 16, 00:23, on fifi: tmux `exp20-fifi`, staged snapshot v6
`…/hwa-exp20-adapter-recovery-20261016` (receipt `40bcd939…`).
- Pipeline `results/lifecycles/jobs/run_exp20.sh`:
  - the 9 screens through `launch/exp18_drive.py`, reusing
    `inputs/exp19-sources/<task>/source.pt`;
  - [`launch/exp20_select.py`](../../../launch/exp20_select.py), writing
    `launch/exp20_frozen.json`;
  - `launch/exp20_lifecycles.py main`;
  - the 3 main runs;
  - one resume pass if a job fails.
- Logs: `results/lifecycles/logs/pipeline-exp20.log` and
  `driver-jobs-{screens,main}.log`.
- A first launch at 00:22 failed at planning, before any computation: the
  driver hard-coded the HWA arm name. It was fixed and relaunched; the failed
  planning directories are kept under `results/failed-planning-20261016T0022/`.
- Source SHA-256: HellaSwag `9c14abc7…`, SST-2 `0e5c7e98…`, PIQA `b766f18b…`.
