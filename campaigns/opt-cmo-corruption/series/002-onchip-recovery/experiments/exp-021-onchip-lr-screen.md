---
id: "exp-021"
title: "On-chip learning-rate and schedule development screen"
status: "complete"
hypotheses: []
---
# exp-021: on-chip learning-rate and schedule screen

## Question and authorization

On October 16, 2026 the user approved a development-only screen of the on-chip
learning rate and schedule, run in parallel on the Jean Zay V100 dev queue
(`umg@v100`, `qos_gpu-dev`). It follows the exp-015 outcome. Planning and launch are
both authorized within the budget below.

Question: on std3 array 271002, does a lower and/or decaying learning rate stop
on-chip learning from degrading the healthy array while keeping or improving
recovery at 3% random faults? This is tested for exact digital Adam and for the
physical noisy thresholded law.

Decision informed: which recipe replaces the frozen exp-013 recipe (constant LR
0.003) before the learning-on-top-of-rewrite and calibration comparisons. No
follow-up launches automatically.

Expected outcome: a lower peak LR with cosine decay keeps the healthy development
KL at or below the held deployment and recovers at least as much at 3%.
Alternatives that change the decision:
- a lower LR fixes the healthy array but recovers less at 3%, a trade-off with
  no single recipe;
- nothing in the grid is healthy-safe, so the degradation is not an LR effect
  (look at the objective, the adaptation data or the batch size instead).

## Prior evidence

- [exp-015](exp-015-noise-free-writes.md) (collected, result note pending): on
  std3, exact thresholded writes (B) are within 1% of noisy writes (A) on
  WikiText teacher KL at every rate. Exact dense digital Adam (C) worsens the
  healthy array: 271002 development KL goes from 0.0208 to 0.0277 and rises at
  every checkpoint. Under exp-015's frozen rule, C is therefore LR-confounded,
  and that rule's next step is to "revisit the learning rate and schedule with a
  development lifecycle". The thresholded arm spikes to 0.064 at update 64 and
  is still descending at update 256.
- [exp-013](../results/exp-013.md): the constant 0.003 recipe; recovery
  improves every 1–5% rate but worsens healthy std3 by 14%.
- [exp-010](../results/exp-010.md): the source of LR 0.003, chosen on
  unrewritten noisy arrays.

## Frozen comparison

Confirmed by the user (October 16): std3 only, array 271002, healthy and random
3%, exact dense and physical noisy arms, LR {3e-4, 1e-3, 3e-3} with warm-up and
cosine decay, selection on development KL only, V100 dev queue. The constant-LR
rows and the schedule details are Explorer proposals made at launch. They add no
case, source or array.

- **Network and source:** OPT-125M, the last six MLPs (layers 6–11): 28,311,552
  logical weights, 56,623,104 devices. Frozen exp-007 std3 whole-model source
  (`standard_hwa_model`, source SHA-256 `86522e23…`).
- **Starting state:** the exact exp-013 held P0 of std3/271002,
  `reused/std3-271002/portable.pt` (`3e35ce9b…`, parent deployment `abcd1c46…`).
  It covers cases `nominal` and `random_30000ppm` and is selected with
  `onchip.case_labels`; upstream defect declarations and provenance are unchanged.
- **Laws:**
  - `noisy`: exp-013 `uncapped` = exp-015 arm A. `cmo_thresholded_write_adam`,
    accumulated, nominal 2σ threshold, physical write noise, uncapped.
  - `exact`: exp-015 arm C. `cmo_idealized_write_adam` with
    `write_noise_scale: 0` (dense exact writes). This is a non-physical oracle.
- **Grid per law:** LR {3e-3, 1e-3, 3e-4} × {constant, cosine}.
  - **Cosine (proposed):** `cosine_warmup_decay`, 256 updates, 13 linear
    warm-up updates (5%), decaying to zero (`min_factor` 0).
  - **Constant 3e-3:** the frozen recipe rerun on V100. It is the host-matched
    reference.
  - That makes 12 lifecycles, one job each, with arms `none` and
    `onchip_weights`.
- **Fixed recipe:** objective `teacher_kl` on 8,192 recovery sequences of 256
  tokens (2,097,152 tokens), batch 32, one pass (256 updates). Data seed
  20261008, on-chip seed 53. Development every 64 updates. Fixed final endpoint.
- **Evaluation:** the unchanged development (256), generated-test (384) and
  WikiText-test (1,024) cohorts at length 256, all on held apparent state.

### Selection rule (frozen before results; thresholds proposed)

```text
Unit: law x config, array 271002. Metric: development teacher KL at update 256.
D0, D3 = development KL of the `none` arm (healthy, 3%) in the same job.
Healthy-safe: healthy final KL <= D0.
Selected config per law: the healthy-safe config with the lowest 3% final KL;
  if none is healthy-safe, the lowest 3% final KL, flagged "no healthy-safe config".
Noisy law, adoption for follow-ups: the selected config is adopted if it is
  healthy-safe and its 3% KL <= that of constant 3e-3 (same law);
  "material" if it is >= 10% lower. Otherwise the frozen recipe stays.
Exact law, oracle validity: the selected config is a valid oracle if it is
  healthy-safe and its KL does not rise between updates 192 and 256 in either
  case. Only then is noisy minus exact (3%) reported as the noise+gating cost.
Held-out generated and WikiText KL are reported for every row after selection
  and never used to select. A disagreement with development is reported.
```

Validity checks:
- **Cross-host anchor:** compare the V100 `noisy-const-3em3` endpoint with
  exp-013 uncapped std3/271002 (5090, PyTorch 2.11). It is not a gate; a
  difference above 2% relative development KL is reported as a host confound
  between this screen and earlier results. Within the screen every row shares
  one host type.
- **Inspection:** every bundle passes native inspection and exact replay.

Evidence class: exploratory, model-based. Limits: one array, one source, one
seed, 256 development sequences; KL differences of a few percent are within
unmeasured run-to-run noise.

## Implementation

`onchip.lr_schedule` (CMO laws only) reuses the HWA schedule kinds. Each update u
(one-based) uses `learning_rate × factor(u)`. `total_steps` must equal the full
on-chip budget. Omitted, every existing lifecycle digest and trajectory is
unchanged; a unit factor is bitwise identical to the constant law. Development
records include `lr_factor`. Tests are in
`tests/test_workflow_onchip_lr_schedule.py`. All workflow tests passed (487 passed,
1 skipped). Lifecycles come from `launch/exp21_lr_screen.py generate`.

## Budget and execution

- **Target:** Jean Zay V100-32G, `umg@v100`, `qos_gpu-dev`. One job per
  lifecycle: 12 jobs, at most 10 queued at a time.
- **Per-job wall:** 1 h 30 requested. Expected 20–40 min, unmeasured on V100.
  The first jobs serve as the cost probe.
- **Caps:** 15 V100-hours including failures. One retry per understood
  operational failure; scientific outcomes never trigger a retry. 70 GB on
  Lustre. Deadline October 16, 23:59 Paris.
- **Copy-back:** only `metrics.jsonl`, `result.json` and inspection receipts
  come back to `results/lifecycles/opt125m-cmo-l6-exp21-lr-screen/`.
  Checkpoints stay on Lustre.

## Current execution and monitoring handoff

**Complete (October 16, 16:14–17:10).** All 12 jobs completed and passed inspection; no retries. Result: [exp-021](../results/exp-021.md). Nothing is queued.

Owner: the launching session, which was also the Monitor; no delegation.

- **Source:** branch `codex/exp021-onchip-lr-screen` at `7b3c95c`, cloned clean on
  Jean Zay at `$JZ/hwa-7b3c95c`, where
  `JZ=/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1`.
- **Inputs:** under `$JZ/exp21/inputs/`, SHA-256 verified after the copy:
  - `std3/cache.pt` `b7438a37…`;
  - `std3/devices.pt` `7bc1bd14…`;
  - `std3-271002/portable.pt` `3e35ce9b…`.

  The teacher and corpora are the existing `$JZ/inputs/` copies, with matching
  pinned digests.
- **Plans:** `$JZ/hwa-7b3c95c/results/lifecycles/<id>/plan`. The only stage run is
  `onchip__standard_hwa_model__271002`, with `--input` overrides for the three
  frozen inputs (`$JZ/exp21/submit.sh LABEL…`).
- **Runs:** `$JZ/hwa-7b3c95c/results/lifecycles/<id>/runs/`. Logs:
  `$JZ/logs/exp21-<label>-<jobid>.out`.
- **Jobs (submitted October 16):** 1181879–1181888 for the first ten
  configurations. `noisy-const-3em4` and `exact-const-3em4` are submitted
  automatically as dev-queue slots free.
- **Inspect and collect:**
  `python -m ebl runs inspect <run> --verify-artifacts --require-complete` on
  Jean Zay, then copy back `metrics.jsonl`/`result.json` and run
  `launch/exp21_lr_screen.py report`.
