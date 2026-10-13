---
id: "exp-014"
title: "Recovery context length after frozen HWA at matched tokens and updates"
status: "complete"
hypotheses: []
---
# exp-014: context length for recovery after HWA

## Question and preparation scope

What is the shortest recovery context that retains the improvement from longer
contexts after frozen HWA, under two, four or uncapped write attempts per device?
The decision is which context and write budget to carry into subsequent recovery
comparisons, including whether a longer context justifies its memory and time cost.

On October 12, 2026 the user requested preparation and clarified that **sequence
length means context length**, not total adaptation data. On the same day the user explicitly instructed **“launch this”**, confirming the
prepared grid, conditional confirmation phase, metric contexts and five-GPU-hour
budget. This authorizes implementation, readiness, launch and monitoring within
the frozen scope. Experiment 013 is a separate recovery-regime execution task.

## Prior evidence and expectations

- [exp-011](../results/exp-011.md): signed accumulation with a nominal 2-sigma
  write threshold and LR 3e-3 halved development KL on one 3%-faulty array.
  About 50M tokens gave little improvement over 2.1M, at 26.5 times the writes.
- [exp-012](../results/exp-012.md): four attempts retained 99.46% of the short
  uncapped development gain; two retained 94.08%. Both used context 256 and
  assignment 271001. These observations motivate comparing both caps with
  uncapped recovery; they do not answer the context question.
- [exp-007](../../001-hwa-robustness/results/exp-007.md): stronger Gaussian and
  corruption-aware HWA can improve the deployment baseline. Standard 3% HWA is
  retained here to isolate the context intervention and connect to the recovery
  pilots, not treated as the best possible HWA source.

Expectation: 128–256 tokens may retain most of the 1,024-token recovery while
reducing attention cost. Alternatives are that long contexts materially help,
that write caps change the preferred context, or that none of these contexts
produces reliable recovery. These are prospective possibilities, not findings.

## Frozen comparison

The user confirmed this prepared comparison by assigning its launch on October 12.

| Factor | Frozen setting |
| --- | --- |
| Network | Pinned FP32 OPT-125M; fc1/fc2 in decoder layers 6–11, 28,311,552 logical weights and 56,623,104 physical devices |
| Frozen source | Existing `standard_hwa_model`: whole-model HWA, 3% per-output-channel Gaussian noise, no temporary faults; no new HWA fit |
| Device regime | CMO 2% endpoint fit; active/reference pairs; random permanent faults at 3%; exact total per MLP; both concentrations null; 256-by-256 physical arrays, 48 arrays/die; no drift/read noise |
| Learning | `cmo_thresholded_write_adam`, LR 3e-3, nominal threshold 2 sigma, signed accumulation, seed 53; all digital parameters and references frozen |
| Write budgets | `max_write_attempts_per_device` = 2, 4, or null; attempts include stuck cells and exclude initial programming |
| Contexts | 64, 128, 256, 512, 1,024 input tokens; both teacher and student receive the same context within a training example |
| Training budget | 2,097,152 input tokens, 2,064,384 common scored positions, 256 optimizer updates, one pass, fresh optimizer/accumulator/counters from P0 |
| Controls | Matched held P0 (`none`) in every native recovery run; the same-cap 1,024-token run is the context reference |
| Primary outcome | Final teacher KL on synthetic development data at fixed evaluation context 1,024; context 256 is a required secondary check |
| Other outcomes | Teacher KL and perplexity on generated and WikiText held-out data at both fixed contexts; writes, touched cells, time and peak memory |
| Checkpoint rule | Fixed final state after 256 updates; development measurements every 64 updates are diagnostic only |

“Uncapped” means no additional per-device attempt cap **within the same 256
updates**, not unlimited training time or tokens. Each healthy active cell can
receive at most one write per update. Do not replace the thresholded law with
the previously unsuccessful dense law. Writes use held apparent state, preserve
permanent faults, introduce no verify/retry and do not flush endpoint residuals.

Stage A comprises **15 learning fits**: five contexts by three write budgets on
assignment 271001. Each starts from the identical portable P0 used in exp-012.
Its held control is re-evaluated on the new fixed evaluation cohorts.

Stage B contains **at most 18 learning fits**: for each write budget, the
development-selected context and 1,024-token reference on assignments
271002, 271003 and 271004, all at 3% faults. If the selected context is 1,024,
run it once per assignment, not twice. These are distinct physical assignments
already used in HWA evaluation, not new HWA-training seeds or pristine unseen
test conditions. Keep endpoint offset 10000 and original selection seeds
211001/211002. Pin one P0 per assignment before branching into recovery arms.
The number of physical devices, fault maps, initial apparent states and update
random-stream seeds must match across contexts and caps for each assignment.

Stage B is conditional on the development rule below and adequate measured
resources; it is authorized within the shared five-GPU-hour budget.
The maximum is 33 learning fits and 33 native held-control trajectories.

## Matched text, batching and evaluation

Reuse the pinned synthetic corpus, SHA-256
`ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984`.
Its training stream has 49,977,869 tokens. Cut it into complete, non-overlapping
1,024-token parent windows. Select the first 2,048 windows of a permutation
seeded by 20261008; shuffle these parents once with seed 53. Every update consumes
the next eight parents, with each parent split into the declared context length.
Do not independently shuffle the child sequences: all lengths must see identical
raw token identities at the same update. Reset positions at each context boundary;
add no new BOS/EOS and never join unrelated parents.

| Recovery context | Sequences per update | Total sequences | Input tokens | Optimizer updates |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 32,768 | 2,097,152 | 256 |
| 128 | 64 | 16,384 | 2,097,152 | 256 |
| 256 | 32 | 8,192 | 2,097,152 | 256 |
| 512 | 16 | 4,096 | 2,097,152 | 256 |
| 1,024 | 8 | 2,048 | 2,097,152 | 256 |

Ordinary next-token scoring loses a target at each context boundary. To keep
the supervised positions identical, exclude predictor offsets 63, 127, ...,
1,023 within every parent at **all** lengths. Score the remaining 1,008
positions per parent: 8,064 per update and 2,064,384 per run. Average KL over
these positions. Teacher distributions are recomputed with each candidate's
context; their change with available context is part of the intervention.
This mask and new parent grouping mean even the length-256 row is a new fit;
exp-012 is background evidence, not a substitute for that row.

Effective batch sizes in the table are fixed. If a microbatch implementation is
needed, sum token-weighted gradients across an entire eight-parent update and
call Adam/the physical writer exactly once. Do not shrink effective batches,
change the number of updates or write once per microbatch to solve an OOM.

All models are evaluated on the same fixed cohorts at **both** 256 and 1,024,
independent of training length. No training boundary mask is applied to evaluation:
use the ordinary token-weighted next-token metric and report `scored_tokens`.

| Cohort | Pinned source/split | Context 256 | Context 1,024 | Same raw prefix |
| --- | --- | ---: | ---: | ---: |
| Development | `opt125m-synthetic-50m`, validation | 256 sequences | 64 sequences | 65,536 tokens |
| Generated reporting | `wikitext2-train-synthtest`, test | 384 | 96 | 98,304 |
| WikiText reporting | `wikitext-2-raw-v1`, test | 1,024 | 256 | 262,144 |

Generated reporting corpus SHA-256:
`8e73b6b43aafb92b8b52e11cdd060d573331fae410350cf2427f3eeba640e224`.
WikiText corpus SHA-256:
`33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01`.
Use only their declared test splits. Training, development and reporting spans
are disjoint by the original corpus lineage; retain its split audit and verify
the new span receipts before running. The two evaluation contexts share text and
are repeated measurements, not independent cohorts. Never compare KL values
across evaluation contexts as if their teacher distributions were identical.

## Development selection and confirmation rule

The thresholds below were frozen before any result exists.
For write budget c, evaluation context e and recovery length L, let K0_e be
held-P0 development KL and K_c,L,e the final KL. Define
G_c,e = K0_e - min_L K_c,L,e.

For a budget to qualify for confirmation, its best observed gain must be at
least 2% of K0_e at both evaluation contexts. Select the **shortest** length
satisfying `K0_e - K_c,L,e >= 0.95 * G_c,e` at both contexts. If no single
length satisfies both, report a context-dependent tradeoff and do not silently
select a compromise or dispatch that budget's confirmation fits. Report all
15 outcomes regardless of eligibility.

Freeze these selections before reading Stage B or held-out reporting metrics.
For each budget, evaluate the selected length against the same-cap 1,024-token
reference separately on the three confirmation assignments. A shorter length
is supported in this narrow regime only if, at both evaluation contexts and
on all three assignments, both fits improve over P0 and the shorter fit retains
at least 95% of the reference's improvement. Nonpositive reference improvement
makes retained-gain ratios undefined; report the raw K values and call that
comparison inconclusive. Report paired differences and their range, not a
significance claim from three arrays. If the chosen length is 1,024, Stage B
tests recovery reproducibility only; it cannot establish context shortening.

Compare caps with uncapped recovery **at the same length** in Stage A, including
gain retained, issued attempts and successful writes. Confirmation supports only
the declared candidate/reference pairs; do not imply a full confirmed cap-by-length
grid. Generated/WikiText endpoints are reporting only and do not choose context,
LR, threshold, cap, checkpoint or additional runs.

Report total and maximum attempts, successful writes, stuck attempts, distinct
touched cells, cap-exhausted cells, pending commands, zero verify reads and
out-of-range states. Verify attempts = successful writes + stuck attempts and
per-cell successful writes <= attempts <= cap where capped. Report idealized
CMO write events; no measured pulse count, energy or endurance claim follows.

## HWA literature alignment and scope limits

The locally pinned PDFs in [hwa_papers](../../../../../hwa_papers/README.md)
were consulted and their SHA-256 values matched the catalog during preparation.
[Büchel et al. 2025](https://arxiv.org/abs/2505.09663v3), section 3.1 and appendix I,
uses per-channel Gaussian noise, synthetic distillation, iterative clipping and
long-context training. Its 4,096-token contexts concern different LLMs and
off-chip HWA; they do not establish the context needed for CMO recovery here.
[Rasch et al. 2023](https://arxiv.org/abs/2302.08469v1), Methods, motivates noisy
forward/backward HWA, range clipping and teacher distillation. Its BERT example
uses 128-token contexts for a different task, not an OPT recovery prescription.

The frozen source follows Büchel's 3% per-channel noise and synthetic-distillation
direction, and Rasch's noisy-weight backward pass and mapped-range clipping.
It retains campaign deviations: CMO/MLP-only hardware, no ADC/DAC quantization,
no learned input range, no noise ramp or alpha-times-standard-deviation clipping;
Adam, pure teacher KL at temperature 1, split analog/digital peak LRs 5e-3/1e-5,
205-step warmup and cosine decay to 0.2 of peak over 25,600 updates; minimum
development KL selection every 512 updates on separate selection arrays.
The smaller corpus, repeated HWA passes and context 256 are inherited source
choices. No source retraining or recipe correction is hidden in this experiment.
Recovery's threshold, caps, constant LR and context grid are campaign-specific
interventions motivated by exp-011/012, not prescriptions from either paper.

This experiment answers recovery-context sufficiency for one HWA recipe at
3% random faults and a short adaptation budget. It does not establish the best
HWA recipe, HWA-training context, a universal optimum through OPT's 2,048-token
limit, other fault rates/clustering, fresh-array transfer or on-chip viability.
The full viability claim additionally requires the best HWA-only, matched digital
calibration and budget-matched frozen-target rewrite controls across fault rates.
Those are not replaced by the held-P0 control here; the broader recovery-regime
study is separate. All conclusions remain exploratory and model-based.

## Frozen-source identities and implementation handoff

The teacher revision is `27dcfa74d334bc871f3234de431e71c6eeba5dd6`, checkpoint
SHA-256 `2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9`.
The standard-HWA checkpoint is recorded in
[the source catalog](../../../launch/exp1g_four_arm_sources.json), `std3`, SHA-256
`86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028`.
Stage A's portable P0 is the explicit `device_state` in
`results/lifecycles/opt125m-cmo-l6-exp6-writecap/prepared/loulou.json`:
SHA-256 `8068397182b3ffe3b0a92298fb36d0205d2e717c420abacb0c061e975c523e4f`,
parent P0 `f80a9447469697cf9e4ff657cb17545d2824e60ae4f12d4694f01f3e4ebaf873`,
held apparent-state hash
`34578c8c1f1a87bcd38597d7808e34eb4f0859a77af0a41183986f454a583db6`.
All four original deployment, device-model and teacher hashes were verified
on loulou before launch; the copied Stage A deployment and device model were
also verified on trex. Host preparation checked lifecycle provenance against
the new pinned caches for every placement.

**Implemented shared data/metric extension; GPU readiness passed.** The original
`data.sequence_length` controls HWA, recovery and all evaluation cohorts together.
`build_cache` independently shuffles blocks at each length, and both HWA-source
and deployment provenance include `data`. Editing that field would therefore
change text, evaluation context and frozen-source compatibility at once.

Extend the existing lifecycle data/metric path, not a per-experiment runtime:

1. In `workflow/lifecycle.py`, define strict optional recovery/evaluation-view
   settings with their own identities. Keep the original HWA/source data contract
   unchanged; preserve all historical digests when the settings are absent.
2. In `workflow/networks/opt_mlp.py` and `workflow/data.py`, construct the shared
   parents, grouped batches, loss mask and separately named evaluation views.
   Bind raw corpus spans, order, masks and view settings into cache provenance.
   Include view settings in cache/on-chip identity without rewriting frozen P0
   provenance or accepting arbitrary mismatched source sections.
3. In `workflow/onchip.py` and the native runtime/collector, preserve one physical
   update per effective batch, report scored/input tokens and replay each fixed
   evaluation view from the saved final state. Extend collection checks for the
   cache/view contract. Existing cap/write semantics remain unchanged.
4. Test common token and target positions, identical parent membership per
   update, independent evaluation context, frozen-source acceptance and rejection,
   legacy behavior, no-write P0 equality, one write opportunity per update,
   cap accounting and exact replay. Run the required `tests/test_workflow_*.py`
   suite after implementation, plus affected native collector checks.

Strict definitions are now generated under
`campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp014-context-*.json`,
use `python -m workflow plan` with explicit pinned inputs, and inspect a native
`python -m ebl campaign run --dry-run`. Recovery manifests must omit HWA training
and must not accidentally launch the generic full lifecycle. The definitions include 15 screens, 45 held confirmation candidates and three
readiness configurations; at most 18 confirmation candidates may execute.

## Budget, stop rules and outputs

Authorized target: at most two idle RTX 5090s from the standing allowed hosts,
one worker per GPU, following the repository's time-of-day/resource policy.
Frozen caps: **2 GPU-hours for Stage A/readiness**, **3 additional GPU-hours for
Stage B**, **120 GB aggregate incremental storage**, and **6 hours elapsed from
first launch**. Resolve the absolute Paris deadline and live host reservations
when execution is assigned. Failed attempts and probes consume these same caps.

Exp-012's context-256 fits took about 102–112 seconds each including validation
and used about 20 GB peak reserved memory. Those measurements are not a forecast
for context 1,024 or the additional evaluation views. Expected wall time is
unknown until probes; 5 GPU-hours is a cap, not a measured requirement.

Read-only `workflow describe --json` on six temporary, explicitly labelled
legacy-compatible storage envelopes gave 28,311,552 weights and 2 trajectories
per screen fit, at 2.43 GB capped / 2.21 GB uncapped. Three-assignment envelopes
give 6 trajectories, 7.30 / 6.62 GB. The resulting conservative state envelope
is **35.35 GB for Stage A plus at most 42.44 GB for Stage B** before shared inputs,
probes, logs and collection. The generic descriptions each include one prepare,
devices and HWA stage, plus one deploy/onchip stage per assignment; the planned
recovery-only execution omits HWA and reuses pinned P0. These scratch descriptions
estimate storage only; they do not validate the new context design.

Under the assigned execution authority, first measure context-64/batch-128 and
context-1,024/batch-8 readiness probes on one or two analog layers, then a bounded
six-layer canary using the frozen source/P0. Mark probes as non-evidence. Measure
GPU memory, tokens/second, evaluation time and saved state size; recompute total
time/storage before releasing Stage A and again before Stage B. An OOM or budget
risk returns to the Codifier; do not change the scientific batch or cohorts.

Stop affected work on nonfinite values, changed P0/fault hashes, unmatched spans,
cap/accounting failure or invalid replay. Preserve failures and all fixed final
states. No automatic LR search, longer training, extra contexts, HWA refit or
scientific retry. A supported persistent monitor must own each launched case;
the existing persistent service will own admitted cases after verified registration.

Expected evidence root: `results/lifecycles/opt125m-cmo-l6-exp014-context/`.
Expected result note: `series/002-onchip-recovery/results/exp-014.md` in this
campaign. Report per-assignment context-versus-KL curves for each cap, KL-versus-
attempts/touched-cells plots, and a time/memory table. Keep evaluation context,
training context, cap and assignment as separate columns. Standard plotting
tools should produce exportable figures from collected native rows.

## Preparation checks (before launch authorization)

The original preparation was read-only; the following checks preceded launch authorization.
Local corpus inspection confirms all proposed spans fit their existing splits.
A bounded CPU tensor check verified identical flattened inputs, predictor tokens
and next-token targets across all five lengths under the common mask, and all
five budgets equal 256 updates / 2,097,152 inputs / 2,064,384 scored positions.
Using the repository's `tensor_hash`, the proposed receipts are:

- parent indices: `d126fb40b097eff02360387b28ee0a523a261d30f077bb5f602eee4f7bd54895`;
- parent tokens: `041b49b70e37ea98bd9ac7aba0b8161553531013ce1372544927894d65eb2fdc`;
- parent order: `9933f043effb82b6e71bc64f184be7fe3535474bf8ce9925933c355bbcb86d5f`.

The synthetic training corpus digest and both reference-PDF digests were checked
locally. This does not substitute for native runtime tests, full input validation
on execution hosts, memory probes or independent-array scientific evidence.


## Final execution handoff

Reviewed October 13, 2026. All **15 screen fits and 18 selected confirmation
fits** completed native validation and local collection. Context 256 was
selected independently for cap two, cap four and uncapped recovery. It passed
all three confirmation assignments at both evaluation contexts under every
budget. The [final result](../results/exp-014.md) owns measurements,
interpretation, limitations and exportable figures. No further runs are queued.

The screen finished October 12 at 22:47:39 Paris and final confirmation collection
finished October 13 at **00:29:58**, before the original **02:54:37 Paris** cutoff.
Charged compute was **2.030 GPU-hours** including readiness; monitored experiment
roots occupied approximately **71.30 GB**. Both are within the original five-hour
and 120 GB caps. The 27 unselected held candidates had no attempts and were closed
at final review. All scientific trajectories succeeded. The cancelled original
readiness registration had no attempt; its corrected replacement succeeded.

Exact execution/evidence locations are retained under
`results/lifecycles/opt125m-cmo-l6-exp014-context/`:

- `hosts/{trex,loulou}.json` and `prepared/{screen,confirm}/` bind placements,
  manifests, jobs, inputs, gates and budgets; `readiness/` preserves original
  resource probes, the registration correction and interim checks.
- `collected/<case>/<attempt>/` holds validated native exports and receipts;
  `reviews/` holds all 33 reports, selection and coverage. `analysis/decision.json`
  records final receipt verification and the scientific rule; CSVs and plots
  preserve per-array metrics and costs. `analysis/unselected-candidates.json`
  records the 27 prospectively excluded, never-run candidates.
- Both remote workers remain at
  `/home/filip/analog-research/opt-cmo-exp1/hwa-exp014-context-20261012`.
  Full native runs/checkpoints are beneath `results/lifecycles/<case>/runs/<case>/`.
  Main receipt/log roots are
  `/home/filip/analog-research/opt-cmo-exp1/worker-receipts/d9885dc65b7f67a8664742198f85cdbf466334c0/<case>/<attempt>/`.

GPU/runtime code was frozen at `defd54b45e79a69055b9e8c7af0457d2ebdb3686`
for readiness. Main workers and collector used
`d9885dc65b7f67a8664742198f85cdbf466334c0`; the intervening change corrected
phase storage accounting and its test without changing training. Both hosts used
PyTorch 2.11.0+cu128, CUDA 12.8 and Transformers 5.8.0. The four original
teacher/device/P0 identities were byte-verified on loulou, with copied Stage A
inputs also checked on trex. All 63 lifecycle definitions and native placement
dry runs passed. Scientific training code and the source checkpoints remain frozen.

The existing `run-watch-queue.service` was the sole scheduling/monitoring owner.
It ran bounded native tmux workers, validated and collected each result, applied
the frozen development rule, and released exactly 18 confirmations. Main jobs
had priority 90 behind exp-013. No competing scheduler, shared service/config
change, or scientific retry was introduced. No exp-014 case remains active;
the shared service continues to own other experiments.

The original readiness registration was cancelled before launch because storage
roots lacked prepared execution/collection paths. The replacement retained the
same computation and absolute deadline. Readiness reserved 0.5 GPU-hours,
screens 1.5 and confirmation 3. Confirmation checked completed trex storage
(27.60 GB, below 35 GB) before reserving at most 80 GB for loulou and local
evidence, leaving 5 GB for staging/older receipts within the total allowance.
The historical registration incident remains preserved; it is not an unresolved
scientific execution failure.

The analysis command is
`python campaigns/opt-cmo-corruption/series/002-onchip-recovery/analysis/exp014_review.py`.
It rechecks compact native evidence and applies the unchanged decision rule;
it does not launch jobs. Final review is complete. Any further contexts, HWA
recipes, cohorts, fault regimes or repetitions require a new scientific plan.
