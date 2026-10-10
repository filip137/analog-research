---
id: "exp-003"
title: "Experiment 1c: overnight extensions of Experiment 1b (9 % noise, mixed-rate corruption, LR decay, L3, training seed)"
status: "running"
hypotheses: []
---
# exp-003: Experiment 1c, overnight extensions of Experiment 1b

## Question and dependencies

**Authority.** On 2026-10-09, about 22:50, you wrote: "I give you authority
to launch experiments that you'd find interesting but we did not launch now
due to limited resources", on 5090s and 3090s that free up during the night.

Every part reuses the [Experiment 1b](exp-002-hwa-synthetic-data.md) setup
unchanged except where stated:
- the corpus `opt125m-synthetic-50m` and 50 streamed epochs of 8,192
  sequences (2.1 passes, 12,800 steps);
- Adam at 5e-3, batch 32;
- the selection rule (development KL at `none` and `random` 3 %);
- CMO devices, arrays 271001–271005 and the 0–5 % random stuck-device grid;
- the WikiText-2 test cohort.

Arrays and test tokens are shared with Experiments 1 and 1b, so every
comparison below is paired per array.

| Part | Question | Lifecycle (digest) | Arms |
| --- | --- | --- | --- |
| A | Can stronger generic noise, or corruption-aware training over a range of rates, give one source that covers 0–5 %? | `opt125m-cmo-l6-exp1c` (`c28b57a93ffd`) | `standard_hwa_9pct` (`noise_strength` 3.0, i.e. 9 %). `cdt_random_mixed`: 3 % noise plus a random stuck rate drawn per minibatch from {1, 2, 3, 4, 5} % |
| B | Was the constant-rate plateau a noise floor? The learning rate drops ×0.1 after epoch 40 (`lr_milestones` [40], `lr_decay` 0.1) | `opt125m-cmo-l6-exp1c-decay` (`67a56244d2af`) | `standard_hwa`, `cdt_random_3pct` |
| C | Do the Experiment 1b conclusions hold at L3? | `opt125m-cmo-l3-exp1b` (`01b205dacca1`) | the five Experiment 1b arms at L3 |
| D | How large is training-run variance? HWA seed 42 and data seed 20261009, so a new stream order and new noise draws | `opt125m-cmo-l6-exp1c-seed2` (`1b476b99d62c`) | `standard_hwa`, `cdt_random_3pct` |

**Expected outcomes (stated before running).**
- A: 9 % noise lies between 6 % noise and 3 % corruption-aware HWA on
  both fidelity and robustness. Mixed-rate training sits between 3 % and 5 %
  corruption-aware HWA, with a smaller cost at 0–1 %.
- B: decay lowers development KL a few per cent for both arms, without
  reordering them.
- C: same direction as L6, with smaller gaps.
- D: a training-run spread well below the 10 % materiality threshold.

## Decision rule (frozen before results)

The unit, metric and "A beats B" rule are as in Experiment 1b: at least 4 of 5
arrays and a mean relative KL reduction of at least 10 %, per rate, never
pooled.
- **A.**
  - `standard_hwa_9pct` against `standard_hwa_6pct` and `standard_hwa`.
  - `cdt_random_mixed` against `cdt_random_3pct` and `cdt_random_5pct`.
  - The Experiment 1b per-rate best set, recomputed with A's two sources
    added.
  - A source "covers 0–5 %" when it is in the best set at every rate from 1 %
    to 5 % and its 0 % KL is not beaten by `standard_hwa`.
- **B.** Decayed against constant (Experiment 1b) for each arm, at every rate.
  Also each arm's development KL at epoch 50, against its epoch-40 value.
- **C.** The Experiment 1b rule (exp-002) at L3. The data effect is paired
  against Experiment 1 L3.
- **D.** For each arm, per rate: the relative difference between the
  seed-2 and Experiment 1b means, set against the 10 % threshold and the
  array-to-array spread. With one replicate this is only a variance
  indication.

**Guards.**
- The `prepare` gate.
- No nonfinite values.
- Five complete assignments per source; anything less is reported as
  partial.
- Convergence reported as in exp-002.

**Coverage.**
- Parts A, B and C are scheduled on GPUs that are available now.
- Part D runs only if a 5090 becomes idle early enough to finish before
  08:00.
- An unlaunched part is reported as not run, not as a negative result.

## Part E (added 2026-10-10, 02:40, before its results)

**Motivation.** Part A found standard HWA at 9 % noise in the best set at
every rate from 1 % to 5 %. Read directly against 3 % corruption-aware HWA
(an exploratory pairing, not predeclared), it is materially better at 0–1 %
and within 5.3 % at 2–5 %.

**Lifecycle:** `opt125m-cmo-l6-exp1c-e`, with the same shared HWA settings as
Experiment 1b. Two arms:
- `standard_hwa_12pct`: `noise_strength` 4.0, i.e. 12 %;
- `cdt_random_3pct_noise9`: 9 % noise plus 3 % random stuck devices.

**Rule (frozen before results):**
- `standard_hwa_12pct` against `standard_hwa_9pct` and `cdt_random_3pct`.
- `cdt_random_3pct_noise9` against `standard_hwa_9pct` and
  `cdt_random_3pct`.
- The per-rate best set over every L6 generated-text source.

**Expected outcome.** 12 % noise costs more at 0 % (more than 10 % against
9 %) and gains little at 4–5 %. The combination sits close to 9 % noise.

**Placement.** On `fifi`, after part B, starting about 04:15 and taking about
75 min. Deployments on Jean Zay.

## Placement, budget and stopping

- **A then B** on `fifi`, chained after Experiment 1b's four trainings. That
  is the RTX 5090 and PyTorch 2.11 environment of every Experiment 1b source.
  About 37 min per arm, roughly 01:10 to 03:50.
- **C** on `nom-cool-1` (RTX 3090, PyTorch 2.5.1), all four L3 trainings in
  one chain, starting about 23:00. Expected about 75 min per arm, done about
  04:00.
  - **Recorded deviation:** the L3 Experiment 1 sources trained on 5090s, so
    C's data-effect comparison crosses environments. This affects
    floating-point rounding only.
- **D** on any RTX 5090 among `riri`, `loulou` and `trex` that is idle (no
  compute process) when checked.
  - It starts only before 07:00 local time, so it finishes by about 07:45.
    This respects the rule that more than two 5090s must be done by 08:00.
  - It never runs beside another process.
- **Deployments:** Jean Zay V100s (`umg@v100`, `qos_gpu-dev`), one job per
  source. Each lifecycle gets its own shared job, and `pretrained` is
  deployed only for C.
- **Compute cap:** 14 GPU-hours. Expected about 11:
  - about 2.6 h of 5090 time for A and B;
  - about 5 h on the 3090 for C;
  - about 1.3 h of 5090 time for D;
  - about 2.6 h on V100s for deployments.
- **Storage cap:** 8 GB.
- **Stop on:** a failed `prepare` gate, a nonfinite objective, a failed
  deployment, or an exceeded cap. Retries only for understood operational
  failures.
- **Outputs:** `results/lifecycles/<id>/{collected,hwa-metrics}` locally.
  One result note, `results/exp-003.md`, covering whatever parts completed.

## Execution and monitoring handoff

**State:** running, from about 23:00 on 2026-10-09.

**Monitoring owner:** the executing agent, through local pipeline watchers
(one per training chain) and an opportunistic dispatcher for D. All live in
the job's temp directory. Each watcher:
- waits for each source;
- copies it to Jean Zay and verifies its hash against the run's artifact
  record;
- submits its deployments, retrying while the dev QoS submit limit is full;
- waits for its jobs;
- inspects and collects on Jean Zay, inspects on the training host;
- copies the rows and training metrics back.

Each stops with `FAIL` on an unexpected state.

### Current handoff (2026-10-09, 23:00)

**Source commit:** `69f6b47`, shipped as a git bundle. Clean clones
`hwa-69f6b47` are staged on `fifi`, `nom-cool-1`, `riri`, `loulou`, `trex`
and Jean Zay. On every host the corpus (`ce58085a…`) and the teacher
(`2d74da66…`) were verified by SHA-256. All four lifecycles are planned on
Jean Zay.

| Work | Where | Handle | State at 23:00 |
| --- | --- | --- | --- |
| C: four L3 trainings (`standard_hwa`, `standard_hwa_6pct`, `cdt_random_3pct`, `cdt_random_5pct`) | `nom-cool-1` (RTX 3090) | detached PID 3984068 (`logs/exp1c-l3-nomcool1.pid`), log `logs/exp1c-l3-nomcool1.log` | running from 22:54 |
| A then B (`standard_hwa_9pct`, `cdt_random_mixed`; then decay `standard_hwa`, `cdt_random_3pct`) | `fifi` (RTX 5090), `launch/chain_exp1c_fifi.sh`. It waits for the Experiment 1b chain and an idle GPU | tmux `opt-exp1c-l6`, log `logs/exp1c-l6-fifi.log` | waiting for Experiment 1b (about 01:10) |
| D: two seed-2 trainings | the first idle RTX 5090 among `fifi`, `riri`, `loulou`, `trex`, between 22:00 and 06:15 | local `dispatch_d.sh` (log `exp1b/dispatch-d.log`); on launch, tmux `opt-exp1c-seed2` | polling every 5 min; nothing idle at 23:00 |
| Jean Zay shared jobs | A 844412, B 844413, C 844414 (with `hwa__pretrained`), D 844415 | `logs/x1c-*-shared-*.out` | all complete |
| C `pretrained` deployments | Jean Zay 844416 | `logs/x1c-c-pre-844416.out` | running |
| Deploy pipelines | local `pipeline.sh`, one per lifecycle (logs `exp1b/pipeline-{a,b,c}.log`); D's runs inside the dispatcher | background tasks | waiting for sources |

**Updates.**
- **D, complete (02:23).** The dispatcher found `riri` idle at 00:56 and
  launched both seed-2 trainings there (RTX 5090, PyTorch 2.11).
  - Deployments: Jean Zay 849073 and 849901, both COMPLETED.
  - Inspect passed for all 12 Jean Zay bundles and the `riri` bundles.
  - 60 rows collected, no duplicates.
- **C, in progress.** `standard_hwa` was deployed as Jean Zay 846442.
- **Experiment 1b's watcher fault** affected C as well. Its first collection
  failed silently because the helper sourced `/etc/profile` under `set -u`.
  Jean Zay's `launch/inspect_collect.sh` was fixed at 01:20, before any
  Experiment 1c pipeline reached it.

**Next:**
- When each pipeline reports `DONE`, check the inspect outputs under
  `results/lifecycles/<id>/collected/`.
- Run [`analysis/exp1c_decision.py`](../analysis/exp1c_decision.py).
- Write `results/exp-003.md` with every part that completed. A part that
  never launched is labelled not run.
