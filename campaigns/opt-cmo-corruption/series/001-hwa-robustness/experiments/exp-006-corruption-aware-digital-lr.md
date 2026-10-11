---
id: "exp-006"
title: "Corruption-aware HWA with smaller digital learning rates"
status: "ready"
hypotheses: []
---
# exp-006: corruption-aware HWA with split learning rates

## Question and authorization

On 2026-10-11 the user approved the proposed next comparison and requested prompt
execution while the two RTX 5090s remain free until the following morning.
Does corruption-aware HWA benefit from the smaller digital learning rates that
improved standard HWA, with analog training and the selection rule held fixed?
Run two full fits on fifi/loulou, then deploy only the development-selected
checkpoint. The two-additional-training-seed confirmation is a later phase with
its own cases and budget; this phase does not dispatch it or on-chip learning.

Evidence: [exp-005](../results/exp-005.md#completed-frozen-checkpoint-transfer-comparison-2026-10-11).
Standard HWA's digital peak 1e-5 gave modest held-out gains under corruption,
below the 10% material-effect threshold, with a small healthy WikiText cost.
Existing corruption-aware analog HWA is a stronger faulty-array control but has
a larger healthy-performance penalty. Its selected mean development KL is
0.07576591790306803 at update 25,088. These references are one training seed,
not new independent replicates.

## Frozen comparison

| Setting | Value |
|---|---|
| New training cases | Corruption-aware HWA, model scope, digital peaks 1e-5 and 1e-4 |
| Reused zero endpoint | Existing `cdt_random_3pct` analog-only source from exp-005 |
| Analog/calibration LR | 5e-3 peak, cosine decay to 1e-3 |
| Digital schedule | Same factor: finals 2e-6 and 2e-5 |
| Training budget | 205 warm-up updates, 25,600 updates, batch 32, 4.196 corpus passes |
| Initialization | Original pinned FP32 OPT-125M teacher, not a prior trained source |
| Network | Last six analog MLPs, 28,311,552 logical analog weights; other components digital |
| Data | Pinned synthetic 50M-token corpus, 195,226 sequences of length 256; original seed/order and HWA seed 41 |
| HWA noise | Per-output-channel abs-max Gaussian 3%, plus temporary i.i.d. 3% random stuck active/reference masks |
| Selection data | Original 256 development sequences and arrays 211001/211002, healthy and 3% permanent random faults |
| Selection rule | Minimum mean healthy/3%-fault KL; compare peaks 0, 1e-5, 1e-4, within 1% of the minimum prefer smaller LR |
| Unchanged optimizer | Adam defaults; no weight decay/gradient clipping; q clamp [-1,1]; original calibration group |
| Selected-source deployment | Five assignments 271001–271005; endpoint offset 10000; permanent random faults 0–5% separately |
| Test domains | Original pinned WikiText 1,024 sequences and generated 384 sequences, length 256 |

The zero endpoint is the existing analog implementation, supported by the prior
zero-digital equivalence checks. It is not silently relabelled as a new model-scope
fit. Source and comparison receipts are pinned in
`campaigns/opt-cmo-corruption/launch/exp1f_references.json`. The existing standard
grid is reused; each method chooses its own digital rate from the same three
values. Standard 9% analog remains an additional robustness reference.

This retains one joint healthy/faulty checkpoint per method. No separate healthy
or 3%-specialist checkpoint rule is introduced. The original best-development
selection and late-convergence flag remain. Test metrics never affect selection.
If zero wins, reuse its existing complete test deployments; do not duplicate them.
Otherwise run exactly one new 30-row deployment job, with both domains per row.

Both HWA reference papers were consulted. Per-channel Gaussian noise and synthetic
distillation follow the Büchel-inspired choices; backward propagation through
noisy weights follows the Rasch-style choice. Corruption masks, the split LR grid,
cosine floor, Adam settings, token budget and best-development checkpoint remain
campaign-specific choices, not a paper-matched recipe. No clipping, noise ramp,
quantization, objective or optimizer changes are bundled into this comparison.

## Decision and limits

Report both development components and clean KL for every candidate. On test
arrays, report KL and perplexity and paired differences/wins separately for each
fault rate and domain, against corruption-aware analog, selected standard 3% and
standard 9% analog. Retain the descriptive material-effect rule: at least 4/5
array wins and at least 10% mean paired relative KL reduction. Healthy costs
remain explicit. A smaller or negative gain is valid evidence, not a retry cause.

This is exploratory, model-based and adaptive, with one training seed and reused
test cohorts/arrays. The result does not establish optimal rates, independent
training reproducibility or on-chip recovery. Freeze candidate recipes for later
seed confirmation only after this comparison is reviewed.

## Execution, budget and ownership

Strict lifecycle files: `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp1f-cdt-*.json`.
Preparation/completion: `campaigns/opt-cmo-corruption/launch/exp1f_cdt_lr.py`.
Evidence root: `results/lifecycles/opt125m-cmo-l6-exp1f-cdt-lr/`.
Remote source/results: `/home/filip/analog-research/opt-cmo-exp1/hwa-exp1f-cdt-lr-20261011/`.
Execution phase: `opt-cmo-exp1f-cdt-lr-20261011`.

Incremental cap: **6 GPU-hours, 16 GB retained storage, six hours from first
admission**, and no worker beyond **2026-10-12 08:00 Paris**. Prior whole-model
corruption-aware training took 6,113 s; expect about **2–2.5 h on the two 5090s**
including readiness, deployment and collection, around 4 GPU-hours. Each full
worker has a 9,000 s hard allowance; the sole 512-update canary has 900 s; each
conditional deployment has 1,200 s. Training reserves 21,500 MiB plus 2,200 MiB
headroom; deployment reserves 17,000 MiB plus 2,200 MiB. One worker per GPU.

One short fifi canary combines the temporary corruption masks with digital peak
1e-5. It uses 512 updates and a separately declared 512-update cosine horizon,
retaining 205 warm-up updates. Its scores are readiness evidence only. Both
full fits remain held until native validation/collection and the immutable
handoff gate pass. Fifi then trains 1e-5, loulou 1e-4 concurrently. Both fits
must validate before the frozen development rule can release one deployment.
Nonselected deployment alternatives are cancelled before any attempt, with their
history preserved. No automatic compute retry or scientific configuration change.

The existing `run-watch-queue.service` is the sole dispatcher/monitor, with its
durable SQLite ledger, locks, idempotent submissions, bounded read-only incident
investigator and self-heartbeat supervision. It polls campaign semantics every
five minutes and distinguishes queued work from active workers. Expected native
stage durations: prepare/devices 120 s, HWA 6,600 s (canary 300 s), deployment
180 s per array, validation and collection 600 s. Startup 600 s and stale 900 s.
Reuse prior verified dispatcher restart, investigator and duplicate-prevention
receipts for the unchanged service; test the new conditional completion logic.
Any mandatory failed gate or insufficient remaining budget stops downstream work.

Native execution uses `workflow plan` and `ebl campaign run` in detached tmux.
Training collection explicitly requires source-only coverage; selected deployment
requires deployment-only coverage, full original-source artifact validation and
exact source lineage. Completion hooks run independently of this interactive
session, collect compact native evidence and produce development/transfer reports.
Large artifacts and failed/partial evidence remain on their execution hosts.
Exact prepared specs, hashes, attempts and first semantic observations will be
recorded in the handoff below before the interactive owner leaves.
