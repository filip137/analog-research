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
3. **Specialization:** for each source, out-of-domain ÷ in-domain KL at 0 %
   (WikiText ÷ generated, since every source here trains on generated
   text), compared with the analog-only source. The wording was clarified
   before any results.
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

**State:** ready, not launched. You asked (2026-10-10, about 08:00) for "a
handoff file for another agent to implement and monitor this".

**What that assigns to the receiving agent**, within this note's scope and
budget:
- staging, the smoke, Stage 1e-a, the freeze;
- writing `opt125m-cmo-l6-exp1e`;
- the two remaining trainings, deployments, monitoring, collection, and the
  result note.

**Defaults in force.** You had not yet answered these review points, so the
plan's defaults apply unless you change them:
- the digital learning-rate grid {1e-5, 3e-5, 1e-4} and its freeze rule;
- Adam without weight decay or gradient clipping;
- a single ×0.1 step after epoch 40;
- best-checkpoint selection.

**Escalate to the user, and do not decide alone:**
- any change to arms, data, metric, rule or budget;
- a smoke that does not fit in memory, or whose extrapolation exceeds the
  8 GPU-hour cap;
- Stage 1e-a falling under the stop rule;
- any reproduction-guard failure.

### Inputs (verified 2026-10-10)

**Code.**
- Commit `3ea25e8` plus the commit that adds this handoff, the launch kit
  (`campaigns/opt-cmo-corruption/launch/`) and
  `opt125m-cmo-l6-exp1e-smoke`. Use the newest commit on
  `codex/revised-hwa-training` that contains this section. The other
  session's uncommitted `AGENTS.md`, `explorer.md` and `hwa_papers/` edits do
  not affect execution.
- **Lifecycles:**
  - `opt125m-cmo-l6-exp1e-smoke` (`9e0484104cfe`; not evidence);
  - `opt125m-cmo-l6-exp1e-lrdev` (`0cf4f02ecd12`);
  - `opt125m-cmo-l6-exp1e`, written after the freeze by
    `launch/make_exp1e_lifecycle.py <rate>`. Tested for all three rates:
    every reused source passes provenance.

**Teacher.** `opt-125m-27dcfa74/pytorch_model.bin`, SHA-256 `2d74da66…`. It is
in `<root>/inputs/` on `fifi`, `riri`, `loulou`, `trex`, `nom-cool-1` and
Jean Zay.

**Corpora**, in `<root>/inputs/corpora/`:

| Corpus | SHA-256 | Present on |
| --- | --- | --- |
| `opt125m-synthetic-50m.pt` | `ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984` | every host and Jean Zay |
| `wikitext2-train-synthtest.pt` (the transfer cohort) | `8e73b6b43aafb92b8b52e11cdd060d573331fae410350cf2427f3eeba640e224` | only `riri` and Jean Zay |

Copy the second from local `data/corpora/` to any other training host and
verify its SHA-256.

**Reused analog-only sources on Jean Zay** (`<root>/sources/`). Each was hash
checked against its run record when copied there:

| File | SHA-256 | Origin |
| --- | --- | --- |
| `opt125m-cmo-l6-exp1c-decay-standard_hwa-source.pt` | `5ddfad0a…` | exp-003 part B, `fifi` |
| `opt125m-cmo-l6-exp1c-decay-cdt_random_3pct-source.pt` | `7c000693…` | exp-003 part B, `fifi` |
| `opt125m-cmo-l6-exp1c-decay-9pct-standard_hwa_9pct-source.pt` | `c58b03ef…` | exp-003 part F, `riri` |

### Run order

The launch kit is in `campaigns/opt-cmo-corruption/launch/`; its README lists
the pitfalls and the compute rules. All training runs on RTX 5090s with
PyTorch 2.11, the platform of every reused source.
- Daytime: at most two of our 5090s at once.
- Only start on a GPU with no compute process (`nvidia-smi
  --query-compute-apps=pid`).

1. **Stage.**
   - `git bundle create hwa-<C>.bundle codex/revised-hwa-training`; copy it to
     `<root>/stage/` on the chosen 5090s and Jean Zay.
   - `git clone -q stage/hwa-<C>.bundle hwa-<C> -b codex/revised-hwa-training`,
     then check `git status --porcelain` is empty.
   - Copy the launch kit to `<root>/launch/`, and both corpora where missing.
     Verify every SHA-256.
   - Run `pytest -q tests/test_workflow_cmo.py tests/test_workflow_opt.py
     tests/test_workflow_lifecycle.py` in one host clone (CPU,
     `CUDA_VISIBLE_DEVICES=`).
2. **Smoke** (5090 A).
   - In tmux, run `SRC=<root>/hwa-<C> launch/train_stages.sh
     opt125m-cmo-l6-exp1e-smoke prepare devices
     hwa__standard_hwa_model_dlr3em5 > logs/exp1e-smoke-<host>.log 2>&1`.
   - **Record:**
     - seconds per epoch: the `elapsed_seconds` difference between the
       epoch-0 and epoch-2 `hwa_selection` rows of the run's `metrics.jsonl`;
     - peak GPU memory: the HWA result's `metrics.peak_gpu_memory_bytes`
       (analog-only L6 used 15.6 GB);
     - `fit.scope` (whole-model scope, its digital learning rate and the
       digital parameter count).
   - **Stop and escalate** if peak memory exceeds about 30 GB, or if five
     trainings at the measured rate exceed the cap. Expected: 45–60 min per
     training and 25–28 GB.
3. **Stage 1e-a**, two 5090s in parallel.
   - A: `launch/train_stages.sh opt125m-cmo-l6-exp1e-lrdev prepare devices
     hwa__standard_hwa_model_dlr1em5 hwa__standard_hwa_model_dlr1em4`.
   - B: `… hwa__standard_hwa_model_dlr3em5`.
   - Nothing deploys yet.
4. **Freeze** by the rule above. Read each source's selected score:
   ```bash
   python3 -c 'import glob,json; [print(p.split("/runs/")[1].split("/")[0], json.load(open(p))["metrics"]["fit"]["development_kl"], json.load(open(p))["metrics"]["fit"]["selected_epoch"]) for p in sorted(glob.glob("hwa-<C>/results/lifecycles/opt125m-cmo-l6-exp1e-lrdev/runs/hwa__*/*/result.json"))]'
   ```
   - The analog-only reference score is 0.0935 (exp-003 part B
     `standard_hwa`).
   - At a grid edge, add the arm a factor of 3 beyond the edge, e.g.
     `standard_hwa_model_dlr3em4`, to the lrdev lifecycle. Commit, ship, and
     train it on its own before freezing.
   - Record the decision in this note.
5. **Write the main lifecycle.**
   - `python3 campaigns/opt-cmo-corruption/launch/make_exp1e_lifecycle.py
     <rate>`, then `python -m workflow check` and commit.
   - Ship the commit to the training hosts and Jean Zay. Plan it on Jean Zay
     (`. /etc/profile; module load pytorch-gpu/py3/2.5.0; export
     EBL_CORPUS_ROOT=<root>/inputs/corpora; python -m workflow plan … --device
     cuda:0`).
6. **Train the two new arms** on the two 5090s:
   `train_stages.sh opt125m-cmo-l6-exp1e prepare devices
   hwa__standard_hwa_9pct_model_<tag>` on one, and
   `hwa__cdt_random_3pct_model_<tag>` on the other.
7. **Deploy on Jean Zay**, every job with `sbatch --export=ALL,SRC=<root>/hwa-<C>`
   and `--time=01:00:00`:
   - **Submit now:** shared `prepare devices hwa__pretrained`; then, each with
     `--dependency=afterok:<shared>`:
     - `deploy__pretrained__2710{01..05}`;
     - one job per reused analog source, with `--input
       deploy__<arm>__<seed>:weights=<root>/sources/<file>` for each seed.
   - **The frozen Stage 1e-a source:** print its path and hash with
     `launch/source_record.py <lrdev runs> hwa__standard_hwa_model_<tag>`.
     Copy it to `<root>/sources/` and check the SHA-256 on Jean Zay. Submit
     its five deployments the same way.
   - **The two new whole-model sources:** one `launch/pipeline.sh` per
     training host. For example:
     ```bash
     HOST=<host> CLONE=hwa-<C> JCLONE=hwa-<C> ID=opt125m-cmo-l6-exp1e \
       ALIVE='tmux has-session -t <session>' SHARED=<shared job> \
       EXTRA=<all manual job ids> \
       launch/pipeline.sh <arm> > pipeline-<arm>.log 2>&1
     ```
     Run it in the background so the controlling agent is notified on exit.
   - When both pipelines report `DONE`, run `launch/inspect_collect.sh
     opt125m-cmo-l6-exp1e <root>/hwa-<C>` once more on Jean Zay. Copy
     `collected/deployments.{json,csv}` and the inspect output to local
     `results/lifecycles/opt125m-cmo-l6-exp1e/collected/`. That gives one
     complete collection: 7 sources × 5 arrays × 6 rates = 210 rows.
8. **Analyse:**
   `python3 campaigns/opt-cmo-corruption/series/001-hwa-robustness/analysis/exp1e_decision.py`.
   It applies the frozen rule on both test domains and checks the
   reproduction guard: `pretrained` and the reused analog sources must
   reproduce their earlier WikiText rows exactly.
   - Copy each training's `metrics.jsonl` to `hwa-metrics/` for the
     convergence report.
   - Write `results/exp-005.md` in the format of exp-003 and exp-004:
     - evidence and coverage (hosts, job ids, inspect, rows, hashes);
     - measurements;
     - the rule outcomes;
     - interpretation, with the paper-deviation table;
     - the decision.
   - Then set this note's status, regenerate the ledger with `--check`, and
     commit.

### Monitoring

**Owner:** the receiving agent, from launch to result.
- Check each running stage at about 10–15 min intervals: log tail,
  `status.json`, the latest `hwa_selection` row, GPU state.
- Watch the pipeline logs for `FAIL`. A pipeline that ends with `FAIL` stops
  only itself. Diagnose before retrying, and retry only understood
  operational failures within the budget. Preserve failed attempts.
- **Escalate:** nonfinite objectives, a failed `prepare` gate, memory errors,
  a reproduction-guard failure, or the budget at risk.
- Keep this handoff's handle table current: hosts, tmux sessions, log paths,
  job ids, states with times, and the next check.

| Work | Where | Handle | State |
| --- | --- | --- | --- |
| Smoke | — | — | not started |
| Stage 1e-a (3 trainings) | — | — | not started |
| Freeze | — | — | — |
| `opt125m-cmo-l6-exp1e` trainings (2) | — | — | not started |
| Jean Zay shared, `pretrained`, reused-source deployments | — | — | not started |
| Collection, analysis, result | — | — | not started |
