---
id: "exp-005"
title: "Experiment 1e: whole-model versus analog-only HWA (L6, generated text, LR decay, both test domains)"
status: "ready"
hypotheses: []
---
# exp-005: Experiment 1e, whole-model HWA

## Question and dependencies

**Question.** Every HWA source so far trained only the analog fc1/fc2 weights
and their calibration parameters: 22.6 % of OPT-125M at L6. Everything else
stayed frozen at the teacher. Both reference papers fine-tune the whole
pretrained model. When HWA also trains every digital parameter (embeddings,
attention, LayerNorms, the tied head), how do the deployed teacher-KL curves
change on each test domain? And do the earlier conclusions survive?

The earlier conclusions are:
- the corruption-aware crossover at about 3 % in-domain ([exp-004](../results/exp-004.md));
- 9 % noise in the best set at 1–5 % ([exp-003](../results/exp-003.md));
- domain specialization ([exp-004](../results/exp-004.md)).

**Origin.** Your review of 2026-10-10: "in the HWA we should normally train the
whole model, so these results might all be biased". You chose to train
everything (as in both papers), and to implement and test only, then stop for
review.

**Decision informed.**
- Which HWA scope and recipe the on-chip recovery sources (Experiment 2)
  should use.
- Which earlier conclusions must be re-stated.

**Expected outcome (stated before running).**
- Whole-model HWA lowers KL most on healthy hardware and in-domain, because
  the digital layers co-adapt.
- It specializes more: a larger out-of-domain penalty.
- The ordering of the noise and corruption-aware arms stays the same.

## Contract extensions (implemented, tested; not yet used)

**Per-arm `scope`.**
- `analog` (default, omitted) keeps today's behaviour.
- `model` adds a trainable copy of every digital parameter, trained by Adam at
  the arm's `digital_learning_rate` under the shared schedule.
- Only the analog weights see noise and corruption, and the frozen teacher
  stays the KL target.
- On-chip arms never train the digital copy.
- Existing arm and lifecycle digests are unchanged. Every campaign lifecycle
  was checked.

**Optional `transfer_tests` section.**
- Second test cohorts from other pinned corpora, scored at every evaluation
  and collected as `*_transfer_<label>_*` columns.
- Sources do not depend on it, so the decayed analog-only sources of
  [exp-003](exp-003-hwa-overnight-extensions.md) parts B and F are reused
  unchanged. Their source sections and arm digests were checked equal.

**Code:**
- `workflow/lifecycle.py`, `workflow/hwa.py` (two Adam parameter groups);
- `workflow/networks/opt_mlp.py` (`StudentInput`, the digital copy);
- `workflow/runtime.py`, `workflow/onchip.py`, `workflow/__main__.py`;
- `docs/crossbar_lifecycle.md`.

**Tests** in `tests/test_workflow_cmo.py`:
- schema and digest stability;
- the whole-model student starts identical to the analog one;
- whole-model HWA trains the digital weights, deploys them and reports
  transfer scores;
- on-chip recovery holds them.

The workflow suites pass: 173 passed, 1 skipped.

## Design

| Field | Setting |
| --- | --- |
| Base | `opt125m-cmo-l6-exp1c-decay`, unchanged: L6, CMO 2 % P&V, per-channel mapping. Corpus `opt125m-synthetic-50m` (50 M generated tokens, 2.1 passes, 12,800 steps). Adam 5e-3 on the analog weights, ×0.1 after epoch 40 of 50. Selection on development KL at 0 % and 3 % stuck (generated validation). Arrays 271001–271005, rates 0–5 %, WikiText-2 test (1,024 sequences) |
| Transfer test | `generated`: the first 384 sequences of `test_100k.txt`, through corpus `wikitext2-train-synthtest` (`8e73b6b4…`), split `test`. No shared 200-character window with the 50 M training text (0/500, checked 2026-10-09) |
| Analog-only arms (reused, no training) | `standard_hwa` and `cdt_random_3pct` (exp-003 part B), `standard_hwa_9pct` (part F) |
| Whole-model arms (new) | the same three recipes with `scope: model`, at the digital learning rate frozen by Stage 1e-a |
| `pretrained` | deployed again, to reproduce Experiment 1b's rows and to give generated-test values |

**Stage 1e-a (digital learning rate).**
- Lifecycle `opt125m-cmo-l6-exp1e-lrdev` (`0cf4f02ecd12`): three whole-model
  standard-HWA arms at digital learning rate 1e-5, 3e-5 and 1e-4
  (`standard_hwa_model_dlr1em5`, `…dlr3em5`, `…dlr1em4`).
- Only the `prepare`, `devices` and HWA stages run.
- **Rule:**
  - freeze the rate with the lowest selected development score;
  - within 1 % relative, take the smaller rate;
  - at a grid edge, add one rate a factor of 3 beyond before freezing;
  - if every rate is worse than the analog-only `standard_hwa` (0.0935),
    report that and stop for review.
- **Why these rates.** Büchel 2025 uses about 1e-4 (AdamW) at batches of
  about 393 K tokens; ours are 8 K tokens. The usual OPT-125M fine-tuning range
  is 1e-5 to 1e-4.
- The main lifecycle `opt125m-cmo-l6-exp1e` is written after the freeze. It
  reuses the frozen arm's source by its exact arm ID.

**Decision rule (frozen before results).** "A beats B" means at least 4 of 5
arrays and a mean relative KL reduction of at least 10 %. It is applied per
rate and per test domain, never pooled:
1. **Scope effect:** each whole-model arm against its analog-only
   counterpart, paired per array.
2. **Recipe ordering under whole-model scope:**
   - corruption-aware 3 % against standard 3 % (the crossover);
   - 9 % noise against corruption-aware 3 %;
   - compared with the analog-only ordering.
3. **Specialization:** for each source, generated-test KL ÷ WikiText KL at
   0 %, compared with the analog-only source.
4. **Per-rate best set** over all seven sources, per domain.

**Guards.**
- The `prepare` gate.
- **Reproduction:** `pretrained` and the reused analog-only sources must
  reproduce their exp-003 WikiText rows exactly.
- The teacher is unchanged (the existing frozen-prefix check).
- No nonfinite values.
- Five complete assignments.

## Deviations from the reference papers

[`hwa_papers/`](../../../../../hwa_papers/README.md):

| Choice | Here | Rasch 2023 | Büchel 2025 |
| --- | --- | --- | --- |
| Trained parameters | the whole model (new). The analog-only arms are kept as a control | whole model | whole model |
| Analog subset | MLP fc1/fc2 of the last 6 layers only | all linear layers | all linear layers |
| Noise | per-channel Gaussian 3 % or 9 %, through the backward pass | conductance-dependent, through the backward pass | per-channel Gaussian 2–3 %, noise-free backward |
| Clipping, quantization, input ranges | none | clipping, learned ranges | α·std clipping, static DAC/ADC ranges |
| Optimizer | Adam, no weight decay, no gradient clipping; digital learning rate scanned | SGD or Adam per network | AdamW (weight decay 0.01, gradient norm 1.0) |
| Schedule | constant, then ×0.1 for the last 10 of 50 epochs | per network, with decay | polynomial with warmup |
| Data | 50 M generated tokens (temperature 0.9, top-k 50) × 2.1 passes | task data | 20 B tokens sampled at temperature 1 |
| Selection | best of every 2 epochs on development KL | last checkpoint | last checkpoint |

## Budget, placement and outputs

- **Training:** 5 whole-model trainings (3 in Stage 1e-a, then 2) on RTX
  5090s with PyTorch 2.11: at most two by day, any idle one at night.
  - Time per training is unmeasured. The estimate is 45–60 min, against
    36 min analog-only.
  - Peak memory is unmeasured. The estimate is 25–28 GB on a 32 GB GPU,
    against 18 GB analog-only.
  - A 2-epoch smoke of one whole-model arm measures both first. If memory
    does not fit, stop and report.
- **Deployments:** Jean Zay V100 dev jobs.
  - Stage 1e-a deploys nothing.
  - Experiment 1e: 7 sources × 5 arrays, about 7 jobs of 15–20 min, since the
    transfer test adds about 40 % evaluation time.
- **Compute cap:** 8 GPU-hours.
- **Storage:**
  - whole-model sources about 0.5 GB each;
  - their deployment bundles about 0.4 GB each (15 bundles, about 6 GB on Jean
    Zay `$SCRATCH`);
  - cap 10 GB.
- **Outputs:** `results/lifecycles/opt125m-cmo-l6-exp1e{-lrdev,}/` and
  `results/exp-005.md`.

## Execution and monitoring handoff

**State:** ready, not launched. You asked to implement and test only, then
review before any launch (2026-10-10).

**To launch after review:**
1. Commit, then stage the bundle and both corpora (`opt125m-synthetic-50m`,
   `wikitext2-train-synthtest`) on the chosen 5090 hosts and Jean Zay.
2. Run the smoke.
3. Run Stage 1e-a (three trainings), and freeze the digital learning rate by
   the rule.
4. Write `opt125m-cmo-l6-exp1e`.
5. Train the remaining two whole-model arms.
6. Deploy all seven sources on Jean Zay.
7. Inspect, collect, write the result.
