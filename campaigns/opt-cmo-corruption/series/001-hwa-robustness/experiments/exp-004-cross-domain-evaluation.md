---
id: "exp-004"
title: "Experiment 1d: cross-domain evaluation of WikiText-trained and generated-text-trained L6 HWA sources"
status: "complete"
hypotheses: []
---
# exp-004: Experiment 1d, training corpus × test domain (L6)

## Question and dependencies

**Question.** On the WikiText-2 test, [Experiment 1b](../results/exp-002.md)
found HWA distilled on 50 M generated tokens much worse than HWA distilled on
WikiText-2. That test is in-distribution only for the WikiText-trained
sources. On held-out teacher-generated text, which source is better?

The answer separates domain specialization from a genuine training-data
effect:
- if each source wins on its own domain, it is specialization;
- if one source wins on both domains, it is a genuine data effect.

**Authority.** Launched under your overnight authority of 2026-10-09 ("launch
experiments that you'd find interesting"), at 02:30 on 2026-10-10, after
Experiment 1b's result. It needs no code change.

**Design: retrain identically, with a different test split.** A source cannot
be scored on another corpus yet, because deployments verify the source's data
digest. So each source is retrained in a lifecycle whose corpus keeps the
original train and validation splits but whose test split is the held-out
generated text.
- `test_100k.txt` (SHA-256 `f188d390…`) comes from the same generator as the
  50 M corpus and has 390 blocks. It shares no 200-character window with the
  50 M training file (0/500 checked on 2026-10-09).
- Training never reads the test split, and runs are deterministic
  (`deterministic=True`, TF32 off) on the RTX 5090 / PyTorch 2.11 platform.
  So the retrained sources should be tensor-identical to Experiment 1 L6
  `cdt_random_3pct` (trained on `riri`) and to the Experiment 1b sources
  (`fifi`).
- Experiment 1 L6 `standard_hwa` was trained on a 3090, so its retrain is
  not expected to match.

| Lifecycle (digest) | Corpus | Training | Arms |
| --- | --- | --- | --- |
| `opt125m-cmo-l6-exp1d-wikitrain` (`4c32a81cb01d`) | `wikitext2-train-synthtest` (`8e73b6b4…`): WikiText-2 train and validation, generated test | as Experiment 1 L6 (8,192 sequences, 50 epochs) | `pretrained`, `standard_hwa`, `cdt_random_3pct` |
| `opt125m-cmo-l6-exp1d-synthtrain` (`e63c6926…`) | `opt125m-synthetic-50m-synthtest` (`55c11c96…`): 50 M train, `val_100k` validation, generated test | as Experiment 1b (streamed, 2.1 passes) | `pretrained`, `standard_hwa`, `cdt_random_3pct` |

Both use test sequences 384 of 390, the same arrays (271001–271005) and the
same 0–5 % grid. With Experiments 1 and 1b, this fills the 2×2 for both
source types.

## Decision rule (frozen before results)

The "A beats B at p" rule is unchanged: at least 4 of 5 arrays, and a mean
relative KL reduction of at least 10 %. For each source type (standard,
corruption-aware 3 %) and each rate, compare the WikiText-trained source with
the generated-text-trained one on the generated test:

- **Specialization:** each wins on its own test domain. On the generated test
  the generated-text source wins, and on the WikiText test Experiment 1 won.
- **WikiText genuinely better:** the WikiText-trained source also wins, or
  ties, on the generated test.
- **Generated text genuinely better:** the generated-text source wins on the
  generated test by more than it lost on WikiText. Report the asymmetry.

**Guards.**
- **Tensor identity:** check each retrained source against its Experiment 1 /
  1b counterpart where the platform matches. A mismatch is reported, but it
  does not invalidate the run.
- The `prepare` gate.
- Five complete assignments.

## Placement, budget and handoff

- **Training:** `riri` (RTX 5090, PyTorch 2.11), idle at 02:27, runs the four
  trainings back to back: WikiText-trained standard and corruption-aware, then
  generated-text-trained standard and corruption-aware. About 2.4 h, done
  about 05:00. With `fifi` that is two 5090s, which is within the daytime
  rule.
- **Deployments:** Jean Zay V100 dev jobs, one shared job and one
  `pretrained` job per lifecycle, plus one job per trained source.
- **Compute cap:** 4 GPU-hours.
- **Storage:** about 1 GB.
- **Monitoring:** the local pipeline watchers, as in exp-003.

### Current handoff (2026-10-10, 02:35)

**Source commit:** `59f8170`. Clean clones `hwa-59f8170` are on `riri` and
Jean Zay. Both corpora were verified by SHA-256 on both hosts, and both
lifecycles are planned on Jean Zay.

| Work | Where | Handle | State at 02:35 |
| --- | --- | --- | --- |
| Four trainings (WikiText `standard_hwa`, `cdt_random_3pct`; then generated-text `standard_hwa`, `cdt_random_3pct`) | `riri` | tmux `opt-exp1d`, logs `logs/exp1d-{wiki,synth}-riri.log` | running from 02:33. Epoch-0 development KL 1.4964 equals Experiment 1 L6 |
| Shared jobs | Jean Zay 850483 (WikiText-trained), 850484 (generated-text-trained) | `logs/x1d-*-shared-*.out` | submitted |
| `pretrained` deployments | Jean Zay 850485, 850486 | `logs/x1d-*-pre-*.out` | dependent on the shared jobs |
| Deploy pipelines | local `pipeline2.sh` (logs `exp1b/pipeline-1d-{wiki,synth}.log`) | background tasks | waiting for sources |

**Updates.**
- **03:40.** Both WikiText-trained sources finished on `riri`, about
  33 min each, and were deployed as Jean Zay 851215 and 851970.
- **Tensor-identity guard:** the retrained `cdt_random_3pct` is
  tensor-identical to Experiment 1 L6 `cdt_random_3pct` (compared on `riri`,
  every tensor `torch.equal`). Its generated-text test scores are therefore
  those of the exact Experiment 1 weights.
- `standard_hwa` has no same-platform original: Experiment 1's was trained on
  a 3090.

- **05:04, complete.** Generated-text-trained sources deployed as Jean Zay
  852590 and 853334. Both are tensor-identical to Experiment 1b's.
  - Inspect passed for both lifecycles: 18 + 18 Jean Zay bundles and the
    `riri` bundles.
  - Collect: 90 + 90 rows, no duplicates.
  - Result: [exp-004](../results/exp-004.md).
