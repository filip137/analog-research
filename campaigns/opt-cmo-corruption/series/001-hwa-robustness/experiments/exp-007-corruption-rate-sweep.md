---
id: "exp-007"
title: "Four-arm HWA comparison with frozen whole-model learning rates"
status: "ready"
hypotheses: []
---
# exp-007: four-arm HWA robustness

## Question and authorization

On 2026-10-11 the user explicitly replaced the proposed five-training-rate sweep
with four recipes: standard 3% Gaussian HWA, 3% Gaussian plus 3% temporary faults,
3% Gaussian plus 5% temporary faults, and standard 9% Gaussian HWA. Reuse the
first two fitted sources and train only the latter two. The user authorized
immediate execution on **two 5090s for training and a third for existing-source
evaluation**, prioritizing completion speed. Prior authorization permits raising
the compute allowance. No 1%, 2% or 4% training fit or seed-replication grid is
part of this phase.

The questions are whether explicit corruption masks improve robustness beyond
Gaussian noise, whether assuming 5% instead of 3% faults helps on more defective
chips and costs performance on healthier chips, and whether stronger Gaussian
noise supplies comparable robustness without corruption masks. This establishes
strong HWA baselines for later on-chip recovery, not on-chip recovery itself.

Prior evidence: [exp-005](../results/exp-005.md) selected digital peak 1e-5 for
standard 3% HWA; [exp-006](../results/exp-006.md) selected the same rate for
corruption-aware 3%. The standard-9% sources from the old cosine comparison used
analog-only or equal-rate whole-model training, so neither fills the missing
split-LR whole-model cell. Historical [exp-002](exp-002-hwa-synthetic-data.md)
addressed fault-rate overestimation with a different training recipe.

## Frozen comparison

| Label | Gaussian weight noise | Temporary stuck-device rate | Source | Execution |
|---|---:|---:|---|---|
| std3 | 3% | 0% | Existing selected whole-model source | riri evaluation |
| cdt3 | 3% | 3% | Existing selected whole-model source | riri evaluation |
| cdt5 | 3% | 5% | New whole-model fit from the teacher | fifi training then deployment |
| std9 | 9% | 0% | New whole-model fit from the teacher | loulou training then deployment |

Zero faults means no temporary stuck devices, with Gaussian noise still present.
Temporary HWA masks are resampled during training; deployed faults are permanent.
Existing-source evaluation on riri repeats the existing matched test grid as the
user requested. It is an evaluation replay, not a new independent array or
training replicate; preserve earlier evaluations as reference evidence.

All settings below retain the completed split-LR protocol:

- Pinned FP32 OPT-125M, six analog MLPs (28,311,552 logical weights); attention
  and other components remain digital. All new fits start from the teacher.
- Synthetic 50M-token corpus, SHA-256
  `ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984`,
  195,226 training sequences, length 256; HWA seed 41, data seed 20261008.
- Whole-model HWA; analog weights and existing calibration parameters use
  **5e-3 to 1e-3**, additional digital parameters **1e-5 to 2e-6**.
  Warm up for 205 updates, then cosine decay over **25,600 total updates**,
  batch 32, approximately **4.196 corpus passes**. No automatic extension.
- Teacher-KL objective, Adam defaults, original mapping/q clamp and calibration
  group; no changes to clipping, optimizer, quantization or noise ramp.
- Identical checkpoint rule for all four: minimum mean healthy/3%-fault
  development KL, on 256 synthetic sequences and selection arrays 211001/211002,
  at initialization and every 512 updates. Retain late-convergence flags. In
  particular, the 5%-training arm does not silently select on a 5% specialist rule.
- CMO 2% endpoint fit, two active/reference devices per weight; permanent random
  faults at 0%, 1%, 2%, 3%, 4%, 5%, with exact counts per MLP and no die/array
  concentration variation. No drift or read noise.
- Five matched assignments 271001–271005, endpoint offset 10000. All four recipes
  use the same identities/fault maps/programming streams. Evaluate separately
  on pinned WikiText (1,024 sequences) and generated text (384), length 256.

Coverage is **120 deployment rows**, four recipes by five assignments by six
fault rates, with two domains per row. There are exactly two new full fits.
The 512-update cdt5 canary is separate readiness evidence, with its own shortened
cosine horizon and the same 205-update warm-up; it never selects a scientific rate.

## Decision and limits

Report both development components, selected update and late-convergence flag.
Report teacher KL and perplexity separately per source, array, rate and domain.
Compare cdt3/cdt5/std9 with std3, cdt5 with cdt3, and both corruption-aware sources
with std9. Retain the existing material-effect rule: at least 4/5 paired array
wins and at least 10% mean paired relative KL reduction. Report healthy costs.
No pooled fault-rate claim, test-driven retuning or further automatic dispatch.

One training seed, reused text cohorts and adaptively revisited arrays keep the
result exploratory and model-based. The LR is frozen in common, not separately
optimized for each new recipe. Distinguish HWA noise amplitude from fault fraction;
these synthetic rates are not measured fabrication statistics. This phase does
not establish training-seed reproducibility, spatial-clustering robustness or
on-chip viability/write costs.

The pinned Rasch and Büchel PDFs were consulted during planning. Per-channel
Gaussian noise and synthetic distillation retain Büchel-inspired choices;
noisy-weight backpropagation retains the Rasch-style choice. Corruption masks,
split LR, cosine floor, Adam defaults, smaller token budget and best-development
selection remain campaign-specific choices; neither paper establishes this ratio.

## Execution, budget and ownership

Strict definitions: `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1g-*.json`.
Preparation/completion: `campaigns/opt-cmo-corruption/launch/exp1g_four_arm.py`.
Exact reused source catalog: `campaigns/opt-cmo-corruption/launch/exp1g_four_arm_sources.json`.
Native training/deployment/inspection and collection use the existing public CLIs;
no workflow, optimizer, physical model or dispatcher runtime is changed.

Local evidence: `results/lifecycles/opt125m-cmo-l6-exp1g-four-arm/`.
Remote checkout/results on fifi/loulou/riri:
`/home/filip/analog-research/opt-cmo-exp1/hwa-exp1g-four-arm-20261011/`.
The exact old source bundles are copied, without changes, to riri under
`/home/filip/analog-research/opt-cmo-exp1/inputs/exp1g-frozen-sources/` and checked
against their source/result hashes and original full native artifact records.

Phase ID: `opt-cmo-exp1g-four-arm-20261011`. Cap **7 incremental GPU-hours,
24 GB retained remote storage, six hours from first admission**, with no worker
past **2026-10-12 08:00 Paris**. Recent full fits took 1.70–1.73 GPU-hours and a
30-row evaluation took 0.0914; expect roughly 4 GPU-hours overall and **about
2–2.5 hours elapsed from readiness/launch**. Local staging of the two existing
source bundles is approximately 1 GB. Full workers have 9,000-second hard
allowances; evaluation workers 1,200 s; the canary 900 s. Reservations include
21,500 MiB plus 2,200 MiB headroom for training, 17,000 plus 2,200 for evaluation.
One worker per GPU; no unrelated workload or GPU settings may be changed.

The sole owner is the existing persistent `run-watch-queue.service`, with a
300-second campaign poll, durable ledger, idempotent submissions, locks and
bounded read-only incident investigation. Its systemd heartbeat supervision,
restart/recovery, investigator and duplicate-prevention evidence is reused for
the unchanged dispatcher. New completion logic and coverage checks are tested.
Expected native stage durations: prepare/devices 120 s, HWA 6,600 s (canary 300),
deployment 180 s per array, validation/collection 600 s; startup 600, stale 900.
No automatic compute retries or scientific modifications. Failed mandatory gates
or inadequate remaining budget block dependent work.

All jobs are registered held. After pinned-input, native dry-run and service
preflight checks, release the canary and both riri evaluations. The validated,
collected canary opens the immutable handoff gate and releases both full fits
automatically. Each full native campaign trains then deploys its own selected
checkpoint, with source-fit validation before deployment. The riri manifests
contain no HWA stage. Completion hooks collect exact bundles and update the
four-arm paired report; no interactive coordinator is needed.

## Current handoff

Preparation is in progress. Frozen revision, manifest hash, readiness evidence,
worker attempts/tmux handles and first semantic observations will be recorded
here after staging and admission. Existing source identities are pinned; the
new runs have not yet been admitted.
