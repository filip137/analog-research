---
id: "exp-018"
title: "On-chip recovery of LoRA-fine-tuned OPT-125M on HellaSwag, SST-2 and PIQA"
status: "partial"
hypotheses: []
---
# exp-018: does on-chip training gain more for a task-fine-tuned OPT?

The user assigned this experiment on October 15, 2026, including its
implementation and execution.

## Question and dependencies

The pretrained-LM regime is close to settled.
- The verify partner rewrite closes about 93% of std3's WikiText fault-KL gap
  at 3% random faults: KL 0.188 → 0.054, against 0.044 healthy
  ([exp-016](../../002-onchip-recovery/results/exp-016.md)).
- On-chip learning after the rewrite slightly worsens KL
  ([exp-016](../../002-onchip-recovery/results/exp-016.md),
  [exp-017](../../002-onchip-recovery/results/exp-017.md)).
- Learning also worsens KL on healthy arrays
  ([exp-013](../../002-onchip-recovery/results/exp-013.md)).
- [exp-004](../../001-hwa-robustness/results/exp-004.md) showed that HWA
  specialises strongly to its training text.

**Question:** does a network fine-tuned on a narrow task leave a larger residual
after the rewrite, one that on-chip learning on in-task data can repair?

**Decision it informs:** whether to pursue on-chip training as a
post-deployment strategy for task-specialised models (scale to more arrays and
tasks), or to close this direction as well.

**Expected outcome:** a small or no gain, as in the pretrained regime. The
alternative that would change the decision is a clear accuracy gain for the
fine-tuned teacher at a matched write cost, absent for the pretrained control
and on healthy arrays.

## Cases and controls

User-confirmed choices are marked **(C)**. Everything else is proposed in the
approved plan and frozen here before any result exists.

- **Deployment (C).** LoRA adapters are merged into the FP32 weights. The merged
  fc1/fc2 of layers 6–11 (28,311,552 weights, 56,623,104 devices) sit on CMO
  arrays; the rest of the network stays digital.
  - LoRA is used only to prepare teachers. It is never a deployed digital
    adapter, and the exp-009 adapter arm is not used.
- **Tasks (C).** HellaSwag, SST-2 and PIQA, with lm-eval-harness v0.4
  templates. The one deviation: every scored row starts with BOS (id 2).

  | Task | Source | Context → choices | K | Development / test examples |
  | --- | --- | --- | ---: | --- |
  | HellaSwag | `Rowan/hellaswag` | lm-eval preprocess of `activity_label: ctx_a Ctx_b` → `" " + ending` | 4 | 2,000 / 10,042 |
  | SST-2 | `stanfordnlp/sst2` | `"{sentence}\nQuestion: Is this sentence positive or negative?\nAnswer:"` → `" negative"`, `" positive"` | 2 | 3,000 / 872 |
  | PIQA | `ybisk/piqa` | `"Question: {goal}\nAnswer:"` → `" {sol1}"`, `" {sol2}"` | 2 | 1,500 / 1,838 |

  - **Splits:**
    - The official validation set is the test split.
    - The development split is a seeded hold-out from train, disjoint by group
      (HellaSwag `source_id`, SST-2 sentence, PIQA goal).
    - All source files are pinned by SHA-256.
- **Teachers.** Each task has two:
  - **FT (fine-tuned):** LoRA rank 16, α 32 on q/k/v/out_proj/fc1/fc2 of all 12
    layers. Embeddings, LayerNorms and the tied head stay frozen.
    - Loss: multiple-choice cross-entropy over the **per-token mean**
      continuation log-likelihoods (`training.score: token_mean`). This is a
      user-approved revision of October 15, 20:07.
      - The originally frozen summed objective failed G1 for HellaSwag and
        PIQA (see the handoff).
      - For SST-2's single-token answers the two objectives are identical, so
        its teacher is unchanged.
    - Training: AdamW with cosine decay, batch 32 examples, seed 61. Epochs: 2
      for HellaSwag and SST-2, 4 for PIQA.
    - Learning rate: screened over {1e-4, 3e-4, 1e-3} and selected on
      development acc_norm.
    - The merged FP32 export is pinned by SHA-256.
    - **Gate G1:** development acc_norm at least the pretrained zero-shot score
      + 3 points. Otherwise the task stops.
  - **PT (control, C):** the pinned pretrained OPT-125M. Its lifecycle is
    identical to FT's except `network.teacher`.
- **Data per task.** One token stream of BOS + context + gold continuation per
  example.
  - Train, validation and test come from the train-adapt, development and test
    examples. Sequence length 256.
  - HWA and recovery cohorts come only from train-adapt.
  - Development: 128 stream sequences, plus every development example for task
    scoring.
  - Test: 64 stream sequences, plus every test example.
  - WikiText-2 transfer: test split, 256 sequences.
  - **Recovery:** 2,048 sequences × 4 epochs, batch 32 (256 updates). Labels
    shape the stream only through the gold continuation; no objective uses
    them after fine-tuning.
- **Technology.** Campaign standard: CMO with the AIHWKit 1.1.0 fit,
  `gaussian_endpoint` P&V at 2%, no relaxation, read noise 0,
  per-output-channel abs-max scaling, active/reference pair, tile 256, 48
  arrays per die.
- **Defects.** `hierarchical_die_array`, permanent, no concentrations.
  - Cases: `none`, `random` 3% and `gmax` 3%, never pooled.
  - gmax leaves about 3× random's post-rewrite residual (exp-016).
- **Arrays.** The user's standing preference is one array per task for
  exploratory OPT/CMO runs.
  - Evaluation: assignment **271002** only.
  - HWA selection: 211001 and 211002.
  - Learning-rate screens: 271001, a development array rather than a
    replicate.
- **HWA (same for FT and PT).**
  - Noise: std3 additive Gaussian, 0.03 × each output channel's abs-max. This
    follows Büchel §3.1's convention; the backward pass goes through the noisy
    weights, as in Rasch and AIHWKit.
  - Training: model scope, digital lr 1e-5, Adam 5e-3 with cosine warm-up decay
    to 0.2×, batch 32, seed 41. Teacher-KL distillation on the task train
    stream, starting from that lifecycle's teacher.
  - **Budget:** 3,200 updates (about 26M tokens).
  - Selection: development teacher KL on the selection arrays (cases `none` and
    `random` 3%) every 512 updates.
  - **Deviations:**
    - Task text replaces Büchel's synthetic data (B.1) because the task corpora
      are small.
    - The budget is ⅛ of the pretrained std3 budget.
    - No α·std clipping or DAC/ADC quantisation (not implemented).
- **Arms.** Every arm except the unrewritten references follows the verify
  partner rewrite (`detection verify`, `roles pair`, 5σ, seed 241016), the
  standing default.
  - R0 `none`: unrewritten reference.
  - R0c `calibration`: unrewritten reference with calibration.
  - R1 rewrite hold.
  - R2 rewrite + digital calibration.
  - R3 rewrite + learn, writing original active devices only (no `write_roles`).
  - R4 rewrite + learn + calibration.
- **On-chip law.** `cmo_thresholded_write_adam` with threshold 2σ, accumulated
  updates, at most 4 attempts per device, seed 53, and a development check
  every 64 steps.
- **Learning rates.** Screened per (task, teacher) as separate development
  lifecycles on array 271001 at `random` 3%, using development data only:
  - weight lr over {1e-3, 3e-3, 1e-2};
  - calibration lr over {1e-4, 3e-4, 1e-3}.

  Each is selected on development `choice_kl`, then frozen here before array
  271002 runs. Calibration has never run with a nonzero lr for OPT.

## Execution contract

Implementation follows the extension points of the
[lifecycle contract](../../../../../docs/crossbar_lifecycle.md):
- `network.teacher` for a fine-tuned FP32 teacher pinned by SHA-256;
- an optional `task` section for the labelled multiple-choice evaluation
  (`crossbar_lifecycle.choice_task.v1`, built by `python -m workflow task-data`);
- task metrics on the held apparent state;
- the native teacher fine-tuning experiment `opt_task_teacher.v1`.

The Codifier records exact configs, commands and hashes below before running.

## Metrics and decision

**Primary metric:** `accuracy_norm_percent` on the task test split. The
companions are:
- `choice_kl`: KL against the teacher over the softmax of the summed choice
  log-likelihoods;
- `choice_kl_norm`, accuracy, agreement and continuation-token teacher KL;
- stream teacher KL and perplexity, and WikiText-2 transfer;
- write cost: attempts, touched devices, maximum attempts per device, and
  rewrite verify reads and programs.

Every result records the analog layers and the logical weight count.

```text
Unit: task × teacher (FT | PT) × defect case, on array 271002. Tasks and cases are never pooled.
Rows, chosen on development data only:
  L = better of R3, R4 by dev choice_kl
  A = better of R1, R2 by dev choice_kl
Δ = acc_norm(L) − acc_norm(A) on the test examples, paired.
  SE = sqrt(b + c − (b − c)²/n) / n, where b and c are the discordant counts.
On-chip helps for (task, teacher, case): Δ ≥ 0.5 pt and Δ ≥ 2·SE.
  Harmful: Δ ≤ −2·SE. Otherwise inconclusive.
Main question, per task × faulty case:
  D = Δ_FT − Δ_PT.
  SE_D: paired bootstrap over test examples, 10,000 resamples, seed 18.
  "Larger on-chip gain for the fine-tuned model" if D ≥ 0.5 pt and D ≥ 2·SE_D,
    and the choice_kl residual closed by L beyond A, (KL_A − KL_L) / KL_A, is larger for FT than for PT.
  "Not larger" if D ≤ 0. Otherwise inconclusive.
  Where PT zero-shot accuracy is within 3 pt of chance, its accuracy Δ is uninformative:
    only the choice_kl comparison is reported, and the main verdict is inconclusive on accuracy.
Repair-versus-adaptation guard: report Δ on `nominal`.
  A faulty-case gain counts as fault repair only beyond the nominal Δ.
Report write cost for every learn row.
```

**Evidence class:** exploratory, model-based, single array (one programming
realization of 271002).

**Excluded claims:**
- fabricated-hardware behaviour;
- generalisation across arrays, fine-tuning seeds or HWA seeds;
- digital-adapter deployment;
- generative or few-shot tasks.

## Budget, stop and expected outputs

- **Compute:** at most 14 GPU-hours (5090-equivalent), under the standing
  compute authorization of October 9.
  - Teacher fine-tuning runs on nom-cool-2's RTX 3090 when idle.
  - Lifecycles run on at most two RTX 5090s from 08:00 to 22:00, and on any
    idle 5090 at night.
- **Storage:** at most 200 GB on the worker hosts. The local `/home` is full,
  so it keeps only the task data and the three merged teachers.
- **Stop rules:**
  - G1 fails → drop that task.
  - The cost probe (HellaSwag FT, array 271002, all cases and arms)
    extrapolates past either cap → stop and report.
  - Nonfinite values or invalid artifacts → escalate.
  - Retries are not replicates.
- **Outputs:**
  - lifecycles `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp18-*.json`;
  - raw runs under `results/lifecycles/opt125m-cmo-l6-exp18-*/`;
  - result note [results/exp-018.md](../results/exp-018.md).

## Execution and monitoring handoff

**Implementation (October 15).**
- `workflow/tasks.py`: templates, task files and the scorer.
- `workflow/teacher_lora.py`: `opt_task_teacher.v1`.
- `network.teacher` and the `task` section in `workflow/lifecycle.py`.
- Task cohorts and metrics in `workflow/networks/opt_mlp.py`.
- `python -m workflow task-data`.
- Task columns in `collect` and `campaigns/lifecycle_collect.py`.
- Contract: [crossbar_lifecycle.md](../../../../../docs/crossbar_lifecycle.md#labelled-multiple-choice-tasks-and-fine-tuned-teachers-opt).

**Tests.** `tests/test_workflow_tasks.py` (19 tests) passes. All existing
campaign lifecycles validate with unchanged digests; for example, exp-017 is
still `4305f258…`. The full suite has 5 failures and 19 errors, all in
subprocess tests of `test_lifecycle_collect.py` and `test_workflow_lora.py`,
caused by an MKL threading-layer clash that depends on test order. A clean HEAD
(`b0c1f06`) checkout gives the identical counts, so they predate this change.

**Task sources.** Hugging Face parquet, pinned by revision and SHA-256; the
receipt is in `data/tasks/sources/receipt.json`.

| Task | Source | Train SHA-256 | Validation SHA-256 |
| --- | --- | --- | --- |
| HellaSwag | `Rowan/hellaswag@382ffb52` (refs/convert/parquet) | `cacb1258…` | `89981307…` |
| SST-2 | `stanfordnlp/sst2@8d51e7e4` | `c7921283…` | `fb00fe00…` |
| PIQA | `ybisk/piqa@142c5123` (refs/convert/parquet) | `29da7ec1…` | `40e1677b…` |

**Task files.** Split seed 20261015. A rebuild reproduces the SHA-256 values.

| Task | Task file | Stream | Train / development / test examples | Stream train / validation / test blocks of 256 |
| --- | --- | --- | --- | --- |
| HellaSwag | `hellaswag-opt125m-v1` `a4e4e977…` | `bfa4e8dd…` | 37,905 / 2,000 / 10,042 | 11,522 / 604 / 3,153 |
| SST-2 | `sst2-opt125m-v1` `d0e4e957…` | `ace4201b…` | 64,349 / 3,000 / 872 | 6,875 / 322 / 132 |
| PIQA | `piqa-opt125m-v1` `767bfbf1…` | `25670635…` | 14,612 / 1,500 / 1,838 (1 excluded) | 2,142 / 217 / 268 |

Overlap receipts:
- Development and train never share a group.
- 14 HellaSwag test contexts also occur in train.
- 135 PIQA validation goals also occur in train, a property of the dataset.

**Codifier engineering choices** (within the frozen fields):
- HWA cohorts: 10,240 / 6,144 / 2,048 sequences, streamed 2,048 per epoch over
  50 epochs (3,200 updates). Warm-up: 26 steps.
- The recovery cohort is a nested prefix of the HWA cohort.
- Every assignment-dependent lifecycle declares assignments [271001, 271002], so
  one HWA source serves the screens and the main comparison; each manifest is
  trimmed to its own assignment's stages.
- Screens score 100 test examples and 8 test stream sequences, with no
  WikiText. They never use test data.
- No `development_every_steps`: every 64-update recovery epoch ends with a full
  evaluation that includes `task_development`.
- Teacher screens check development every half epoch.

**Execution.**
- Worker source: staged snapshots of base `b0c1f06` plus the working-tree diff
  (`git diff` SHA-256 `7cfeae81…`), each verified by an `EBL_SOURCE_RECEIPT`
  hash list:
  - loulou: `…/opt-cmo-exp1/hwa-exp18-finetuned-tasks-20261015`, receipt
    `42aa4bf4…`;
  - trex: `…/hwa-exp18-finetuned-tasks-20261015b`, receipt `9c5815d4…`, with a
    local throwaway git commit so the campaign runner accepts it.
- Inputs: `…/opt-cmo-exp1/inputs/exp18/`.
- Teacher LR screen: loulou, tmux `exp18-teachers`, nine runs under
  `results/teachers/`.
- PT HWA sources: trex, tmux `exp18-pt-sources`, under
  `results/lifecycles/opt125m-cmo-l6-exp18-*-pt-source/`.
- Owner: the executing agent.

**Teacher screen.** Complete on loulou, 17:12–18:00; 9/9 runs complete and
every export gate passed. Values are development acc_norm (%); the selected LR
is in bold. Record: [`teachers/screen.json`](../../../teachers/screen.json).

| Task | Zero-shot | lr 1e-4 | lr 3e-4 | lr 1e-3 | Gain | G1 (+3) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SST-2 | 58.50 | 94.03 | **94.50** | 91.37 | +36.0 | **passed** → teacher `exp18-sst2-lora-r16-lr3e-4`, SHA-256 `f8e351cb…` |
| PIQA | 61.73 | 63.13 | **64.00** | 57.67 | +2.27 | **failed** |
| HellaSwag | 32.65 | 29.05 | **30.10** | 29.45 | −2.55 | **failed** |

Under the frozen rule, HellaSwag and PIQA stop at G1. The HellaSwag failure
points at the frozen teacher objective rather than the task:
- Cross entropy over *summed* continuation log-likelihoods favours short
  endings, while the decision normalises by length.
- The development choice loss is about 10.5, and every LR lowers acc_norm below
  zero-shot.

Changing the objective is a material choice, escalated to the user. As evidence
for that decision only, four development-only diagnostic fits run at lr 3e-4,
using the new declared `training.score` options:
- `token_mean`: cross entropy over per-token mean scores;
- `gold_lm`: next-token cross entropy on the gold continuation.

They do not alter the frozen screen, and no lifecycle uses them unless the user
approves.

**PT HWA sources.** Complete on trex, about 13–14 min each, all native stages
complete:

| Task | HWA source SHA-256 |
| --- | --- |
| PIQA | `2503f898…` |
| SST-2 | `de7273aa…` |
| HellaSwag | `b3e6d54f…` |

Each selected at or near epoch 50 of 50, so `convergence_review_required` is
set: HWA was still improving when the frozen 3,200-update budget ended. On the
selection arrays, random-3% teacher KL fell from 2.77 to 0.21 for SST-2. The
pretrained PIQA clean test score is 62.0% acc_norm and 62.8% acc, consistent
with the published lm-eval OPT-125M PIQA figures.

**Incident (operational, 19:23).** The first PT screens (snapshot v3, fifi)
failed at their deploy stage. A screen had shrunk `data.test_sequences`, so its
`data` section no longer matched the HWA source, and source reuse was refused.
- The screens now keep `data` identical; only `task.evaluation_examples` and the
  WikiText transfer test differ, and the source does not depend on those.
- The failed attempts are kept under snapshot v3 (`…-20261015c`).
- Relaunched from snapshot v4 (`…-20261015d`, receipt `520610bc…`).

A first attempt to run the diagnostic teacher fits on nom-cool-2 failed at CUDA
initialisation, because the agent sandbox blocks the GPU. They now run on
loulou instead.

**SST-2 learning-rate screens.** Complete: array 271001, `random` 3%,
development only.
- FT source: snapshot v3 on loulou, complete 19:36.
- FT screens on loulou and PT screens on fifi, from snapshot v4.
- Record: [`launch/exp18_frozen.json`](../../../launch/exp18_frozen.json).

Final `task_development` choice KL:

| Teacher | rewrite hold | rewrite+learn, LR 1e-3 / 3e-3 / 1e-2 | rewrite+calibration, LR 1e-4 / 3e-4 / 1e-3 | Frozen weight LR / calibration LR |
| --- | ---: | --- | --- | --- |
| FT | 0.00183 | 0.00272 / 0.00492 / 0.01211 | 0.00156 / 0.00152 / 0.00150 | **1e-3 / 1e-3** |
| PT | 0.00223 | 0.00241 / 0.03535 / 0.01994 | 0.00131 / 0.00120 / 0.00119 | **1e-3 / 1e-3** |

Observations:
- For both teachers, every screened weight LR worsens development choice KL
  relative to rewrite hold.
- The weight LR therefore selects at the lowest grid value. A lower LR would
  only approach hold, but this is a grid-boundary choice.
- Calibration improves development KL at every screened LR. Its selection also
  sits at the grid edge (1e-3), with a flat optimum.

These values are frozen before any array-271002 run.

**Teacher-objective diagnostics.** Development only, lr 3e-4, loulou, every
gate passed:

| Objective | HellaSwag dev acc_norm (zero-shot 32.65) | PIQA dev acc_norm (zero-shot 61.73) |
| --- | ---: | ---: |
| `sum` (originally frozen, best LR) | 30.10 | 64.00 |
| `token_mean` | **47.50** (+14.85) | **66.60** (+4.87) |
| `gold_lm` | 34.80 | 63.87 |

**User decision (October 15, 20:07).**
- Adopt `token_mean`, and rerun the frozen LR screen {1e-4, 3e-4, 1e-3} for
  HellaSwag and PIQA; the lr 3e-4 fits above are its middle point.
- Select on development acc_norm, re-check G1, then run the FT sources,
  screens and main comparisons.
- The screen runs on loulou (tmux `exp18-tokenmean`, after the FT SST-2 main).

**Other learning-rate decisions.**
- **PT PIQA** (screens complete, fifi): weight LR **1e-3** (development choice
  KL 0.02008 / 0.02704 / 0.03858, against hold 0.01816) and calibration LR
  **1e-3** (0.01712 / 0.01696 / 0.01684). Frozen.
- **Incident:** the first attempt of HellaSwag PT screen c failed at deploy
  with exit −11, the known fifi `libcuda` segfault. Its prepare and devices
  stages were complete, so it is queued as a resume retry after the PT SST-2
  main; the failed attempt is kept.

**`token_mean` teacher LR screen.** Complete on loulou at 20:47; every gate
passed. Values are development acc_norm (%):

| Task | Zero-shot | lr 1e-4 | lr 3e-4 | lr 1e-3 | Selected / G1 |
| --- | ---: | ---: | ---: | ---: | --- |
| HellaSwag | 32.65 | 43.85 | **47.50** | 28.55 | lr 3e-4 (`8ca97f1f…`), +14.85, passed |
| PIQA | 61.73 | 65.47 | **66.60** | 51.60 | lr 3e-4 (`aad4b470…`), +4.87, passed |

**Scope change (user, October 15, about 20:25).** The user redirected the work
to [exp-019](exp-019-analog-lora.md): analog LoRA adapters on the frozen std3
base. Running exp-018 jobs were allowed to finish, and no further exp-018
crossbar runs are launched.

The HellaSwag and PIQA FT teachers above are selected but have no FT source,
screens or main comparison. Final exp-018 coverage:
- SST-2: FT and PT main comparisons;
- PIQA: PT screens and main;
- HellaSwag: PT screens.

**SST-2 main comparison** (`opt125m-cmo-l6-exp18-sst2-{ft,pt}`, array 271002
only):
- FT: queued on loulou after the teacher diagnostics, using the FT source from
  snapshot v3.
- PT: queued on fifi after the HellaSwag PT screens, using the PT source
  `de7273aa…`.
- FT and PT differ only in `network.teacher` and `question`.
- The SST-2 main runs also serve as the cost probe. HellaSwag stays on hold
  pending the teacher decision.
