# Pilot: does fault placement change OPT recovery?

State: user-narrowed array-variation, rewrite-only pilot complete and reviewed.
All declared coverage passed; no further execution is scheduled by this pilot.
Assigned October 16, 2026: the user asked whether variation in
where corrupt devices occur changes performance, selected the existing OPT-125M
experiment, and said "go". This authorizes the bounded comparison below.

## Current scope: user revision on October 16

The user cancelled the fresh uniform comparison, deferred die/combined variation
and all learning, and requested array variation with verified partner rewriting
only. They explicitly authorized cancelling our job on loulou and using that GPU,
then asked for a monitor agent after launch. This section supersedes the broader
comparison below, which is retained as preparation history.

- The four original OPT jobs were cancelled before any attempt (zero GPU time).
  Cancellation of `cifar-native-loulou-baseline_c1_h2-20261016-v1` was requested
  through its owning queue; preserve its partial artifacts and confirm the worker
  stops before launching OPT.
- Same six-MLP std3 model, teacher, data, geometry and fault totals. Only array
  variation (`die_concentration: null`; array concentration 48/95/140 at 2/4/6%).
  Gmin and gmax separately; three recovery assignments 271002–271004; healthy
  checks retained. No newly simulated uniform/die/both condition and no learning.
- Only verified partner rewriting changes devices. The native no-write `none`
  arm remains as a deployment measurement, required by the lifecycle's matched
  control contract. There are 21 rewrite endpoints and 21 no-write baseline
  endpoints. Five deployment-only source-provenance assignments yield 35 rows.
- Compare rewritten KL/PPL and cost across the three assignments. Reuse existing
  uniform evidence on assignment 271002 only; do not apply the original all-three
  placement effect rule or claim a learning result. Other assignments have no new
  matched uniform control. Report synthetic realized spread and both-stuck pairs.
- New lifecycle: [opt125m-cmo-l6-exp21-array-rewrite.json](../opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp21-array-rewrite.json).
  New evidence root: `results/lifecycles/opt125m-cmo-l6-exp21-array-rewrite/`.
  Existing frozen snapshots and receipts remain unchanged.
- Target: idle loulou RTX 5090. Expected 20 minutes; cap one GPU-hour including
  validation, within the original six-hour allowance. Worker timeout 45 minutes.
  Storage cap 40 GB: about 21.4 GB for 21 pairs of hold/rewrite states, plus
  deployment bundles, compact collection and source. Original deadline and dated
  admission windows retained. No unattended scope expansion.
- The existing queue remains the sole launcher and process supervisor. After
  launch, a requested monitor agent will own progress review and scientific
  interpretation through that controller, without a second launcher.

Revised execution handoff: job `opt125m-cmo-l6-exp21-array-rewrite`, campaign
`opt-cmo-exp21-array-rewrite-20261016`, manifest SHA-256
`ebbdf04c0be1f5fcc6eb4f83ce751e27b64e4b28096c6f409d110c5ded1e3f33`.
Worker checkout: `loulou:/home/filip/analog-research/opt-cmo-exp1/hwa-exp21-array-rewrite-20261016`,
source commit `1c1cec1896764305ccfc9f2c1e8691aad7d7c546`.
The immutable source and input hashes, successful native dry run, report fixture
(including missing-endpoint rejection), queue registration and cancellation
receipts are in the revised evidence root's `prepared/` directory. Existing
hierarchy/rewrite tests remain applicable; `workflow check` passed for the new
lifecycle. Callback watch `9095cebd39316163dd27e0a0` targets this chat and
`reviews/summary.json`. Collection validates all native bundles before reporting
21 rewrites and 21 baseline measurements.

The CIFAR queue started `cifar-native-loulou-ours_c1_h1-20261016-v1` immediately
after the first cancellation. That continuation was also cancelled through its
verified supervisor; the seven remaining queued cases in the same CIFAR run were
held to honor the requested GPU reassignment. Completed CIFAR results and partial
states remain intact. These holds are not an authorization to resume that study
automatically after OPT.

At 16:33 Paris both cancelled CIFAR workers had confirmed terminal cancellation,
and a direct GPU query found no remaining compute process on loulou. The reduced
OPT job was admitted at 16:35:16 Paris, priority 110. Direct worker status confirmed
both worker and supervisor alive. Attempt `09d80e15b4b744c49cb457bb0dcb30d6`;
supervisor PID 3049428; worker PID 3049456; tmux session
`opt125m-cmo-l6-exp21-array-rewrite-09d80e15b4b744c49cb457bb0dcb30d6`.
Log: `<worker>/results/lifecycles/opt125m-cmo-l6-exp21-array-rewrite/worker/09d80e15b4b744c49cb457bb0dcb30d6/run.log`.
Native artifacts: `<worker>/results/lifecycles/opt125m-cmo-l6-exp21-array-rewrite/runs/opt125m-cmo-l6-exp21-array-rewrite/`.
The user-requested agent `/root/monitor_array_rewrite` verified first semantic
progress: preparation and devices complete, digital teacher KL −1.5175×10⁻⁹
(numerical zero, within the 10⁻⁶ gate), finite deployment metrics and live worker
and supervisor. The existing queue then handled five-minute observations and
collection without an idle agent polling. Completion watch
`9095cebd39316163dd27e0a0` delivered the terminal callback to originating thread
`01a14502-a883-7d82-b873-ba37ec79e98b`, which acknowledged it and resumed the same
agent for review. The single attempt completed in 1,145.109 seconds including
validation (19.1 minutes); no retry or second launcher was used. Native remote
results occupy 24,918,649,277 bytes. Preparation/review time and token usage were
not separately metered. The seven CIFAR holds remain outside this pilot's scope.

## Revised result: completed October 16

Verified partner rewriting substantially reduces corruption damage for both stuck
kinds at all three rates and assignments. Array variation leaves a small,
rate-dependent residual above healthy deployment. The six comparisons with
historical uniform placement show small, mixed differences; they do not establish
a consistent placement penalty or equivalence. This is exploratory CMO model
evidence for six analog MLPs, decoder layers 6–11, containing 28,311,552 logical
weights. No learning was performed.

**Validity and coverage.** The native collector validated all ten stages and
their full artifact hashes, pinned lineage and evaluation cohorts. Independent
review rechecked the retained collection-file hashes, 35 deployment rows,
21 rewrite plus 21 no-write endpoints, and 42 exact saved-state replays.
Every hold/rewrite pair shares its deployment hash, initial apparent state,
initial metrics and defect identity. All 21 no-write controls preserve state and
metrics exactly; all endpoints report zero learning presentations and no
calibration. Exact fault totals per MLP are 188,744 / 377,487 / 566,231 at
2 / 4 / 6%, unchanged between arms. The CMO programming implementation masks
stuck cells from updates; stuck attempts are accounted separately. No exclusions,
failed attempts or replacement assignments enter this result.

Evidence: [native collection](../../results/lifecycles/opt125m-cmo-l6-exp21-array-rewrite/collected/index.json),
[summary](../../results/lifecycles/opt125m-cmo-l6-exp21-array-rewrite/reviews/summary.json),
[independent audit](../../results/lifecycles/opt125m-cmo-l6-exp21-array-rewrite/reviews/review-audit.json),
[figure](../../results/lifecycles/opt125m-cmo-l6-exp21-array-rewrite/reviews/array-rewrite-results.png)
([PDF](../../results/lifecycles/opt125m-cmo-l6-exp21-array-rewrite/reviews/array-rewrite-results.pdf)).
The [review script](../../results/lifecycles/opt125m-cmo-l6-exp21-array-rewrite/reviews/review.py)
reproduces the audit and figure without modifying native evidence. Summary SHA-256:
`98e847193dd09c28fd842f4d2376778f0d7d076915c2d461fe921ddd9ce79345`.

**Held-out performance.** Each KL triplet below is assignments 271002, 271003,
271004, respectively; PPL ranges show those same three assignments. Domains and
stuck kinds remain separate. Exact per-assignment values are in the linked audit.

| Fault | WikiText KL after rewrite | WikiText PPL range | Generated KL after rewrite | Generated PPL range |
| --- | --- | --- | --- | --- |
| gmin 2% | .062430, .062068, .061876 | 49.770–50.213 | .035288, .035334, .035194 | 7.290–7.313 |
| gmin 4% | .082681, .083354, .081447 | 50.234–50.976 | .051844, .052446, .051969 | 7.399–7.411 |
| gmin 6% | .105591, .104689, .105334 | 51.490–52.132 | .069893, .070727, .070366 | 7.512–7.547 |
| gmax 2% | .062956, .062535, .061401 | 49.847–50.345 | .035248, .035372, .034737 | 7.297–7.329 |
| gmax 4% | .083534, .083127, .084546 | 50.548–51.126 | .052243, .051887, .053169 | 7.405–7.425 |
| gmax 6% | .103474, .103952, .105202 | 51.351–51.642 | .071153, .070480, .071681 | 7.502–7.510 |

Every faulty endpoint improves KL in both domains. On WikiText, rewriting removes
93.20–95.56% of the initial KL excess above its matched rewritten-healthy
reference; this range describes the 18 individual results, not a pooled estimate.
Healthy rewritten WikiText KL is .044283 / .043804 / .043694 and PPL
49.586 / 49.503 / 49.493. The healthy detector makes 26 / 28 / 42 false-positive
programs, so rewriting is not exactly a no-op: the largest absolute healthy
WikiText KL change is 7.54×10⁻⁷. The plot includes these changed healthy references.

**Physical cost and geometry.** Every rewrite endpoint makes one full verify scan:
56,623,104 device reads (two per logical weight). The table gives successful
device programming events; each selected healthy device is programmed once,
so these are also distinct successfully touched devices. Add the reported stuck
attempts for all addressed devices. No incremental learning writes occur, and
endpoint programs must not be described as measured pulses. Initial deployment
cost is separate from these additional rewrite costs.

| Fault | Device programs across assignments | Stuck attempts | Clipped rewrite targets |
| --- | --- | --- | --- |
| gmin 2% | 1,078,391–1,079,854 | 353–404 | 544,408–546,069 |
| gmin 4% | 2,132,985–2,134,685 | 891–1,002 | 1,076,510–1,077,398 |
| gmin 6% | 3,142,635–3,145,205 | 1,951–1,983 | 1,586,014–1,589,006 |
| gmax 2% | 1,064,267–1,066,077 | 881–980 | 544,015–545,809 |
| gmax 4% | 2,106,007–2,107,597 | 2,374–2,441 | 1,077,665–1,079,917 |
| gmax 6% | 3,102,244–3,105,614 | 4,713–4,849 | 1,588,969–1,591,469 |

Realized array-rate SD ranges are 1.905–2.000 / 1.967–2.003 / 1.930–2.014
percentage points at 2 / 4 / 6%. Corresponding realized die SDs are
.158–.257 / .173–.227 / .199–.226 percentage points, despite no sampled die
variation. Arrays range from 0–16.376%, .444–15.721%, and 1.404–16.083%,
respectively. These are synthetic placements with exact totals, not measured
process statistics. Both-stuck weight pairs are 22,597 / 21,727 / 21,757 at 2%,
56,641 / 56,592 / 55,839 at 4%, and 112,391 / 113,838 / 113,094 at 6%
(assignment order as above). Gmin and gmax reuse these same placements.

**Historical placement comparison and limits.** On assignment 271002, array
minus historical-uniform rewritten WikiText KL is +.000734 / −.000127 / +.000264
for gmin 2 / 4 / 6%, and +.000230 / +.000797 / −.002176 for gmax. Relative
total-KL changes range from −2.06% to +1.19%; the generated-domain changes also
have mixed signs. Historical both-stuck counts were 11,410 / 45,429 / 101,773,
so array variation increases this mechanism, most strongly at 2%, while clipped
rewrite targets decrease slightly. More both-stuck pairs alone therefore do not
predict a clear increase in this pilot's final KL; no causal decomposition was run.

The historical full native measurements match their pinned hash. Old and new
network/data/device/deployment/P&V/transfer settings, teacher and mapping hashes,
endpoint and rewrite streams, clean-source metrics and runtime math match.
Software commits differ (`b0c1f06e…` versus `1c1cec18…`); the user cancelled fresh
uniform reproduction, so historical software differences were not experimentally
excluded. Only assignment 271002 has a uniform comparison. The original
three-assignment placement rule does not apply, and three clustered assignments
cannot establish placement equivalence. This run cannot answer whether learning
helps, whether die variation matters, or whether on-chip training beats the best
no-weight-learning alternative. The next scientific decision is whether any
replicated placement comparison or learning follow-up is warranted; neither is
launched or authorized by this result.

## Superseded full comparison

## Question and decision

At a fixed number and kind of stuck devices, does variation between physical
arrays or dies change the damage left by verified partner rewriting, or the
additional improvement from the existing noisy on-chip learning recipe?

The starting evidence is [exp-016, phases 3–5](../opt-cmo-corruption/series/002-onchip-recovery/results/exp-016.md):
rewriting removes most of the gmin/gmax damage; learning removes part of the
remaining teacher KL at 4–6% on assignment 271002. Random faults at 3% gave a
different result. The positive learning result has one assignment and no
calibration control. No existing result answers the placement question.

Expected outcome: clustering could increase the residual, including weights
whose two devices are both stuck, and change how much healthy weights can
compensate. Alternatively, the total fault count may dominate and placement
may have little effect. Either outcome determines whether subsequent recovery
experiments need placement as a separate axis.

## Frozen comparison

Confirmed scope: existing pretrained OPT-125M, six analog MLPs (decoder layers
6–11, fc1/fc2 only), frozen std3 source, previous data and learning recipe;
placement is the intervention. The following bounded implementation was made
visible before launch, following the user's execution instruction.

- **Physical scale:** 28,311,552 logical weights, 56,623,104 devices; 256×256
  physical arrays, 48 arrays per die, 144 arrays and three dies per MLP.
- **Source:** exp-007 std3, SHA-256
  `86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028`.
  Teacher revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`, SHA-256
  `2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9`.
  There is no HWA fitting or checkpoint reselection.
- **Faults:** gmin and gmax, separately, at 2%, 4%, 6% per physical device;
  exact total per MLP, plus one healthy case. Four placements per kind/rate:
  uniform (`null`, `null`); array variation (`null`, κ); die variation
  (κ, `null`); both (κ, κ). κ is respectively 48, 95, 140 at 2%, 4%, 6%,
  from κ = μ(1−μ)/0.02² − 1. These give nominal 2-percentage-point SD at
  a level whose parent mean is μ. Combined variation is larger; exact-total
  allocation changes realized spreads. Report the actual array and die rates,
  their SD/range and the fault allocation between fc1/fc2. These synthetic
  spreads are not measured CMO statistics. No within-array row/column faults
  or movement of faults between MLPs is implied.
- **Assignments:** 271002, 271003, 271004 for the matched recovery comparison.
  The source's five declared assignment seeds are retained for provenance;
  271001 and 271005 receive deployment-only reference evaluation. Those
  auxiliary rows are not substitutes for the three matched recovery assignments.
  Endpoint offset 10000; the same assignment, endpoint, rewrite and learning
  streams are reused across placement scenarios. One endpoint realization per
  assignment; no inference that endpoint seeds are independent arrays.
- **Arms:** unrewritten `none` reference; `pair_rewrite_verify` (5σ detector,
  both device roles, seed 241016); `pair_rewrite_verify_recovery`, starting
  from that identical rewritten state. No new calibration or adapter arm.
  This is a placement comparison, not a claim that learning beats every
  no-weight-learning alternative.
- **Learning:** frozen exp-013/016 active-only thresholded CMO Adam, LR .003,
  accumulated 2σ commands, at most four attempts per active device, 256
  updates, batch 32, 2,097,152 recovery tokens, on-chip seed 53. The physical
  noise remains enabled; errors persist in the held post-write state. Final
  checkpoint only; development measurements every 64 steps do not select it.
- **Data:** unchanged exp-016 synthetic corpus, 256-token sequences, seed
  20261008, 8,192 recovery and 256 development sequences. Report the same
  1,024 WikiText-test and 384 generated-test sequences; recovery and test
  tokens are disjoint. All three corpus hashes are in the lifecycle.
- **Metrics:** teacher KL primary, WikiText as the leading held-out domain;
  generated teacher KL and perplexity in both domains reported separately.
  No pooling of rates, stuck kinds or domains.
- **HWA lineage:** the frozen source retains the per-output-channel 3%
  Gaussian noise and synthetic distillation discussed in the
  [reference-paper record](../../hwa_papers/README.md), including the existing
  deviations (no α·std clipping or converter quantization, noisy backward
  pass, Adam/cosine schedule and development selection). Placement does not
  alter any HWA choice.

For each assignment, kind, rate and domain, report:

1. Initial KL/PPL, rewritten KL/PPL and learned KL/PPL, with the matched
   healthy reference.
2. Placement effect on each arm: clustered minus uniform KL, and relative
   change in the rewrite residual above healthy. Near-zero denominators
   (≤1e-6) are left undefined.
3. Learning benefit `rewrite KL − learned KL`, and its change from the
   uniform placement. Positive means learning helps.
4. Rewrite programs/reads, learning writes, stuck attempts and distinct
   touched devices; both-stuck pairs and clipped rewrite targets when present
   in native receipts. Modeled programs/write events are not measured pulses.

**Predeclared exploratory rule:** a placement effect is consistent when all
three paired assignment differences have the same sign. Call it material in
this pilot when all three change the rewrite residual by at least 10% in that
direction. Report absolute KL/PPL alongside this threshold. Smaller or mixed
effects remain inconclusive. Also report whether the sign of learning benefit
changes. These are effect-size and replication rules, not a p-value or a
population-level significance claim. Three assignments cannot establish
equivalence or broad viability. Do not tune or extend the run after seeing results.

## Budget and execution

- Target: an idle RTX 5090 on trex or riri, one GPU per executable slice. The user
  explicitly offered trex or the V100 development queue; trex preserves the
  existing 5090 programming-noise realization and passed the same preparation.
  The existing
  queue enforces fleet admission and the launcher requires an idle device.
  Other users' processes are left alone. No new controller or global settings.
- Compute cap: six GPU-hours including validation and operational retry;
  expected about three hours, extrapolated from the identical exp-016
  six-case run (684 s) plus deployment, state hashing and collection overhead.
- Storage cap: 250 GB for new local/remote evidence, snapshots and retained
  checkpoints. Keep native final states and failed evidence; no pruning.
  Prior measured state sizes distinguish held and learned arms when refining
  the conservative `workflow describe` estimate.
- Queue window: until October 18, 16:11 Paris; no automatic extension. At most
  one understood operational retry within the same total allowance.
- Admission windows: October 16 until 18:00, then October 16 and 17 from
  22:00 to 08:00 the next morning, Paris time. These use the queue's existing
  weekday two-GPU limit during the first window and the repository's overnight
  permission. Each worker is bounded by its window. The queue's broader
  weekend/evening policy is not imported into this experiment.
- Coverage: 25 unique defect cases and 225 unique recovery endpoints over
  the three selected assignments (75 case×assignment starts). Four executable
  placement slices each include the same healthy control: 252 executed
  endpoints and 140 deployment reference rows in total. Repeated healthy
  controls must agree and are not replicates. Each slice has ten native
  stages: prepare, devices, five deployments and three on-chip stages. No HWA
  stage. Slicing keeps native compact exports within their transport limit.
- Lifecycle: [opt125m-cmo-l6-exp21-fault-placement.json](../opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp21-fault-placement.json).
  This file describes the whole comparison; executable siblings end in
  `-uniform`, `-array`, `-die`, and `-both`.
  Prepare with `python -m workflow plan --hwa-source standard_hwa_model=<pinned-source>`;
  execute with `python -m ebl campaign run`; collect with the existing native
  lifecycle collector and `python -m workflow collect`.
- Local evidence: `results/lifecycles/opt125m-cmo-l6-exp21-fault-placement/`.
  The preparation receipt records immutable source, input, plan and job hashes.

## Current execution and monitoring handoff

All four dry runs passed on both riri and trex; commands and all pinned files
match exactly across hosts. All four jobs were released after adding trex by
an audited operational amendment. The existing campaign
execution service owns monitoring and collection; its completion callback
returns evidence to this originating chat for review.

Latest handoff, October 16 around 16:24 Paris: the controller's campaign check
at 16:21:47 verified storage (39,589,705 bytes), accepted the uniform case's
gates, and left the other cases waiting for its validation. Trex was idle during
preparation, but another user's job started before admission; riri is also
occupied. No worker was launched and no compute budget was consumed. The jobs
remain released for the first idle permitted host within their dated windows.
Controller PID 2529587 and its process identity were verified. The compact
receipt is `<evidence>/readiness/monitor-handoff.json`; normal campaign checks
continue every 300 seconds, next due around 16:27. This is a monitored queue
handoff, not evidence of execution or performance.

- Immutable worker source on both `riri` and `trex`:
  `/home/filip/analog-research/opt-cmo-exp1/hwa-exp21-placement-20261016`,
  snapshot commit `7fc81ca1b23495a5de5ae7e8bb4086506001bd18`. The exact
  controller copy, source bundle, file hashes and preparation receipts are in
  `<evidence>/prepared/`; original worktree changes were preserved.
- Trex placement amendment: `opt-cmo-exp21-trex-placement-20261016`, SHA-256
  `67334c4b910accbb330da755cf8c42ff7e7a95c6c2a6fed77163bece5a48fe07`.
  The four job IDs, commands, dependencies, windows and budgets are unchanged;
  the queue chooses one of the two prepared hosts per job. Receipt:
  `<evidence>/prepared/hosts/trex.json`.
- Campaign: `opt-cmo-exp21-fault-placement-20261016`.
  Jobs: `opt125m-cmo-l6-exp21-fault-placement-{uniform,array,die,both}`.
  Uniform runs first; all three interventions depend on its successful
  collection and reproduction of the 18 exp-016 endpoints on assignment
  271002 (apparent-state hashes and write costs exact, KL/PPL within 1e-6
  relative). The native run supplies exact replay and stuck-state checks.
- Each worker: expected 3,000 seconds, total allowance 5,400 seconds including
  validation, 21,500 MiB GPU memory plus 2,048 MiB headroom. One job at a time
  per GPU. Native campaign processes are supervised by the existing tmux
  bridge, with 900-second leases and a 4,800-second execution timeout.
- Remote paths: `<worker>/results/lifecycles/<job>/`; `worker/` contains
  attempt-specific handles/logs/receipts; `runs/<job>/` contains native
  campaign stages. Local collection: `<evidence>/collected/`; geometry and
  paired effects: `<evidence>/reviews/`.
- Cadence: existing service's 300-second campaign observation; 600-second
  startup allowance and 900-second stale-artifact limit. Check native metric
  growth and process state. One understood operational retry only; preserve
  failed artifacts and all budgets. No scientific changes during recovery.
- Completion callback: watch `1cee12c11641fe2037817104`, originating thread
  `01a14502-a883-7d82-b873-ba37ec79e98b`, all four jobs, with
  `reviews/progress.json` and `reviews/summary.json` attached. The existing
  controller owns observation, validation, collection and delivery; scientific
  review remains pending. Read-only incident investigation is enabled globally;
  automatic code repair is not enrolled for this pilot.
- Review commands: the prepared jobs contain exact native validation and
  collection commands. Read `reviews/summary.json`, all per-scenario
  `*-mechanisms.json`, `collected/rows.json` and native measurement receipts;
  finish interpretation here before reporting the study complete. Plot paired
  rewrite residuals and learning benefits by placement, preserving assignment,
  fault kind, rate and evaluation domain. Report realized placement spreads
  and both-stuck pairs beside the performance comparison.

## Evidence and coverage

No new performance evidence yet. Readiness: eight hierarchical-fault tests and
23 fault-rewrite tests passed; `workflow check` passed; eight remote plans and
dry runs passed; the report fixture checks paired effects, identical healthy
controls and rejection of missing endpoints. Pinned input hashes were checked
on both hosts, whose interpreters are PyTorch 2.11.0+cu128 as in exp-016.

`workflow describe` gives a deliberately broad 456.52 GB estimate for the
unsliced five-assignment, 375-trajectory template. The actual execution selects
three recovery assignments. The prior native artifact sizes are 0.339742 GB
for an unrewritten hold, 0.679491 GB for rewrite hold and 1.132478 GB for
learning. Applied conservatively to all 84 executed starts, those give
**180.744 GB**, plus approximately 12.3 GB for 20 deployment bundles and small
caches/source/metadata. This measured estimate fits the 250 GB cap. No new
technology, layer count or batch size was introduced; the uniform job serves
as the real reproduction check before the interventions.

## Interpretation and next decision

Pending the declared comparison. Any conclusion is restricted to this frozen
HWA source, learning recipe, synthetic defect hierarchy and tested rates.
