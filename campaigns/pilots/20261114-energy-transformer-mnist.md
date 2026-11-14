# FP32 energy Transformer versus digital attention on MNIST

Issue: https://github.com/filip137/on-chip-learning/issues/36
Branch: `energy-transformer`. Exploratory GPU simulation, one seed.
User authorized implementation and the six-GPU-hour plan on 14 November 2026.

## Frozen question and comparison

Can the convergent energy model learn MNIST completion with FP32 centered EP,
and how do reconstruction quality, settling, GPU memory and wall time scale
against a small digital attention model trained with BP?

Eight runs: energy_ep and digital_bp for small (16 tokens, D32), reference
(16,D64), wider (16,D128), more_tokens (49,D64). All have four heads and
6D² core weights: 6,144 / 24,576 / 98,304 / 24,576. Energy memory width is 4D;
digital FF width D matches the independent Q/K/V/output attention parameter
cost. Input encoder (2P→D), output decoder (D→P) and position (N×D) have
matching shapes and identical initial tensors. Total counts are 11,360 /
35,008 / 119,168 / 30,784. Both norms use epsilon .5, with no affine gains.

The energy model retains Q/K, both reciprocal returns, scalar energy,
persistent inputs and output feedback; it has no independent V. Memory is the
minimum of .5r²−rUg + I[r≥0]: a quasistatic perfect diode, not a newly added
RC memory state. Four matched ideal LEVEL1 triode currents reduce to xy;
the matched fixed-drain weak-inversion MOS bank reduces to softmax. Operand
limit ±4 is enforced. The prior generic BSIM3 circuit is historical evidence,
not the model simulated here. Normalization, signed projections, isolation and
supplies remain ideal macros; this is not a transistor-level validation.

States, phase statistics and Adam are FP32; AMP and TF32 are disabled. Solver
starts free states at zero, initializes ± phases from the same free state,
uses dt .5 with per-example energy backtracking (20 halvings, roundoff
allowance 8 eps*(1+|E|)), tau_z=1, tau_y=1.3, residual <=1e-6, maximum 4096
steps. No trajectory differentiation. Centered beta .005; normalize gradient
by masked-pixel count after subtraction. Failed phases never update weights.
Scientific convergence/domain stops stop the arm and remain evidence.

Digital baseline: one pre-LN bidirectional attention block, Q/K/V/output,
two residual additions, ReLU FF, bias-free maps, no dropout. All maps use
normal initialization std .15/sqrt(fan-in), positions .2, seed 20261114 with
name-local generator offsets documented in `gpu.initialize`. Adam .003,
default betas/epsilon, no decay, batch64, max2000 updates; no tuning.

## Task, metrics and gates

Local MNIST: `/home/filip/datasets/mnist`; hash all four raw IDX files.
Hold out 500 examples per digit for 5k validation (split seed3100), use
remaining55k training, original10k test. Reproducible sampling with replacement
(3101) and varying train masks (3102); fixed val/test masks (3103/3104).
Hide8 of16 image-space7×7 regions; identical masks for 7×7 and4×4 tokenization.
Pixels map to [-.4,.4], missing pixels zero, observation indicator supplied.
Masked half-MSE in voltage units trains both arms; report MSE and PSNR in
original [0,1] units without clipping predictions.

Primary evidence: final test against its exact paired initialization and
training-set mean image. Record completed evaluation counts; partial prefixes
are not full test results. Compare training curves at common update and wall
budgets, plus fixed256-image validation probes. Zero/uniform attention replays
are diagnostic ablations, not independently trained causal controls.

Readiness: diode complementarity and switch, explicit currents vs reductions,
energy-force identity/descent, FP32 EP vs independent FP64 activation-coordinate
implicit derivative and finite differences; cosine>.99 and relative error<5%
globally and per parameter group. Test identical masks/parameter counts and
optimizer/RNG restoration. No FP64 training fallback. A useful result needs
valid numerics and improvement over initialization and mean image; digital
parity/superiority is measured, never assumed.

## Execution and handoff

Backend `/home/filip/analog-research/drn`, interpreter
`/home/filip/miniconda3/envs/py312/bin/python`, local RTX3090 only.
Results `/home/filip/analog-research/drn/results/energy-transformer-mnist/`.
Configs `examples/energy_transformer_mnist/<case>-<arm>.json` select the public
`energy_transformer_mnist.v1` train interface. Each case gets its own native
RunStore bundle. The serial composition is
`python -m experiments.run_energy_transformer_mnist --output-root <root>/benchmark-001 --readiness <root>/readiness/gradient-device-checks.json --charged-readiness-seconds <measured>`.

Budget: readiness<=30min, each arm<=35min including25min training and10min
evaluation, final profiling/analysis allowance50min; absolute aggregate6h.
Launcher enforces the remaining wall ceiling, including subprocess startup.
No new SPICE, remote host, mixed precision, hyperparameter sweep or replicas.
Storage ceiling5GB total,500MB per arm; retain initial, each250-update and final
checkpoints with optimizer/RNG, metrics, masks, input identities and examples.
No automatic pruning. Resume only identical config, scientific stops are not
retryable. Operational retries must debit the original allowance and retain
failed attempts. Source must be committed, integrated/pushed and pinned before
evidence launch. Readiness smokes may use the feature worktree.

Monitor: one run-watch owner, five-minute polls,45min stall threshold. Existing
launcher writes worker/child PIDs, status/coverage and per-case logs; native
metrics/checkpoints establish progress. Watcher must observe child metrics,
not just the launcher's heartbeat. Completion requires8 accounted-for cases,
native artifact checks and explicit partial/failure reconciliation. No remote
placement or scientific changes in recovery. The watcher may dispatch bounded
agents for operational incidents or final evidence review; no routine agent
polling. The user is informed if monitoring/notification is unavailable.

## Current evidence

Implementation completed; targeted regressions passed57 distinct tests with4 environment/SPICE skips. Both public-CLI real-MNIST two-update smokes passed. One digital smoke failed before training with transient CUDA driver initialization failure; the identical retry passed. Original failed bundle retained under readiness/digital_bp. No scientific settings changed.
GPU gradient gate: relative error0.0008392, cosine0.999999859, worst parameter
group error0.0028623; all independent FD errors<3.3e-9. Explicit ideal current
identities and perfect-diode complementarity passed. No scale result yet.

Readiness accounting: conservatively debit300seconds for tests, small CUDA checks,
startup and smokes (including the failed attempt). Detailed native durations are
in `results/energy-transformer-mnist/readiness/accounting.json`; this debit is
part of, not additional to, the six-hour ceiling.

## Launch and operational recovery (14 November)

Published initial implementation2e766d2. Automatic approval review rejected
merging/pushing shared `refactor/training-core` because it considered that
broader than explicit branch authorization. The safe alternative is evidence
from a clean pinned checkout of the published `energy-transformer` commit;
shared integration remains pending. Do not retry that rejected action.

First sweep `benchmark-001` started11:42:58UTC, absolute deadline17:37:58UTC.
Native bundles confirm clean source2e766d2. Six cases failed before any update
with CUDA driver initialization failure. Detached/foreground CUDA probes then
passed; the49-token EP case started and reached382 updates with accepted
residuals and falling training MSE. This is not a held-out result. Preserve its
execution and the subsequent digital case. The original watcher recorded the
incident and was stopped while the root agent owned diagnosis.

Recovery adds a tested, narrowly bounded path: only a CUDA initialization
failure with zero recorded training updates may retry. Scientific/domain/
settling failures never retry. `--retry-from <prior-root>` waits for the live
predecessor, reuses its terminal science, preserves all failed bundles and
debits prior wall time and each case's already consumed allowance. It retains
the original absolute deadline. At most two attempts in the recovery sweep;
no new science or extra GPU allowance. The new watchdog retains the original
watch-state dispatch history and budget; it owns the queued recovery and final
review.58 distinct targeted tests passed,4 SPICE/environment checks skipped.

The49-token EP case completed its10k test and5k validation after stopping at
update382 (nudged residual1.54972e-6). TestMSE0.0545565 versus initialization
0.2366731 and training-mean image0.0673463. Uniform-attention validation replay
was almost unchanged (0.0548616 vs0.0548131); zero attention worsened it to
0.1067171. Thus this case demonstrates learning and dependence on attention
coupling, but does not establish useful content-dependent routing.

Sweep002 reuses this case and retries seven startup failures. Small digital
completed2000updates; small EP again failed before training, and reference EP
started. CUDA startup probes passed6/6 when their supervisor retained a CUDA
context. The launcher now retains one four-byte GPU tensor/context between
workers as an operational guard; it changes no science. Driver/context memory
is additional to reported model tensor peaks. A subsequent recovery may reuse
all terminal cases and retry remaining initialization failures within the
original17:37:58UTC deadline. All attempts/dispatch history remain preserved.

## Current durable handoff (11:57 UTC)

Scientific implementation commit2e766d2; bounded recovery399a2d2; CUDA-context
guard83a00fb. All are published on `energy-transformer`. Shared trunk remains
unchanged because automatic approval review rejected its merge/push.

Active sweep002: PID2184836 (authoritative PID in its worker.pid), execution
session35034; clean source399a2d2. Final recovery003: PID2189095, session54673;
clean source83a00fb, queued until002 exits. Check current handles rather than
assuming these remain live. Root003 reuses all terminal science from002,
which already reuses the49-token EP result from001.

Canonical watcher:
`/home/filip/analog-research/drn/results/energy-transformer-mnist/benchmark-001/watch.json`
now watches `energy-transformer-mnist-003`. Watcher PID2189788, dispatch enabled,
five-minute polls; its first real observation saw queued process2189095 and
advancing predecessor metric bytes. Prior incident/dispatch history retained
in benchmark-001/watch-state; first configuration archived. No pending incident
agent at handoff. Next scheduled check is approximately12:02UTC.

Read current progress/status in benchmark-003/status.json and the detailed
monitor/recovery/review contract in benchmark-003/handoff.md. Expected final
comparison: benchmark-003/analysis/comparison.md and comparison.json. Original
absolute deadline remains14Nov17:37:58UTC (18:37:58Paris), all startup failures
and retries debit the original allowance. Only zero-update CUDA startup errors
may retry; stopped scientific runs remain final evidence. A review must state
actual update/test counts and compare matched cases, not different widths.

Small digital completed2000updates, full testMSE0.0524059. This is not a matched
comparison against the49-token EP result. Reference EP is still active; wider
and remaining paired coverage are not yet reported. No automatic chat callback
is configured; durable status/artifacts are the notification surface.

The implementing agent may yield with this verified monitoring/review handoff.
The completion agent records findings in this note and pushes only the feature
branch. Keep issue36 open for the broader analog-transformer objective.

Analysis also extracts fixed256-image validation probes at shared update
counts and plots their recorded wall times; incomplete probes and full5k
validation scores are excluded from that matched cohort. Sparse checkpoint
times are not advertised as identical wall budgets.59 distinct targeted tests
passed,4 SPICE/environment checks skipped. At closeout rerender the derived
comparison with the current feature-branch `experiments.plot_energy_transformer_mnist`
module; the pinned runtime source and raw native bundles remain unchanged.


## Terminal review — 14 November 2026

Benchmark003 accounts for all eight terminal arms. Fresh public-CLI inspection
with `--verify-artifacts --require-complete` passed all eight explicit native
bundles with full artifact hashes and no errors. Worker2189095 and child2204303
are absent. The historical startup failures remain preserved; the reused small
BP and reference EP bundles are from002, and49-token EP remains from001.
No further training or scientific retries are authorized by this review.
Scientific coverage remains partial (`coverage_complete=false`), despite valid
terminal artifacts. Derived comparison.json/md, learning.png and validation.png
were regenerated from feature commit `af7b9a3`; raw evidence was unchanged.

| Case | EP updates | EP test count | EP test MSE | BP updates | BP test count | BP test MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 1050 | 5568 | 0.0635810 | 2000 | 10000 | 0.0524059 |
| reference | 1450 | 0 | unavailable | 2000 | 10000 | 0.0461110 |
| wider | 6 | 0 | unavailable | 2000 | 10000 | 0.0466076 |
| more_tokens | 382 | 10000 | 0.0545565 | 2000 | 10000 | 0.0419638 |

Every EP arm reached the predeclared convergence stop. Small failed the positive
nudged phase at residual0.000316203; reference and wider failed free phases at
2.86102e-6 and1.90735e-6;49-token failed the positive nudged phase at1.54972e-6.
Small test evaluation separately stopped at residual2.74926e-5. Reference and
wider have no final validation/test scores. Their incomplete ablation phases
remain recorded; wider zero-attention replay alone completed. Last accepted
training residuals were below1e-6 and operand maxima remained below4; positive
triode margins were0.474/0.301/0.565/0.583 in case order. Recorded small positive
energy increases are numerical diagnostics, not proof of strict descent.

The49-token EP result demonstrates learning: full-test MSE0.0545565 beats its
initial0.2366731 and paired mean-image0.0673463. Small EP beats its own initial
and mean image on its5568-example prefix (0.236536 and0.0647197), but that prefix
cannot be compared directly to BP's full test. All BP arms beat their paired
initialization and mean image. Uniform attention scarcely changes49-token EP
MSE while zero attention worsens it, so content-dependent routing is not
established. No hardware performance claim follows from these simulations.

On the same256-image validation cohort at the largest shared trained checkpoint,
EP/BP MSE is0.0676298/0.0655144 at1000updates (small),
0.0662456/0.0519687 at1250 (reference), and0.0560462/0.0552632 at250
(49tokens). Wider has only a shared initialization probe. Digital BP is lower
on these matched checkpoints; this is one seed, without statistical replication.
Training curves show early improvement then EP plateaus, with reference
instability near its stop. Final scores have unequal update counts and are not
matched-training comparisons. Probe timestamps show much greater simulator
cost for EP, but sparse probes do not establish identical wall budgets.
Model tensor peak memory EP/BP in MiB:21.875/19.849,29.876/21.750,
46.730/26.783,99.775/31.714; supervisor CUDA-context overhead is additional.

Launcher receipts charge609.0467seconds for003 plus1061.9854 prior seconds and
300 readiness seconds:1971.0320seconds total, within21600seconds. Original
absolute deadline17:37:58UTC and all failed attempts remain unchanged. No budget
extension or remaining-case launch is needed. The sandbox GPU query could not
access the driver; this does not invalidate the completed artifact checks and
no GPU action was taken.

Verdict: terminal execution and artifact review are complete, with explicitly
partial scientific coverage. Energy EP learns in the49-token case but robustness
across widths and digital parity are not established. Any further investigation
requires a new assignment; no retuning or rerun follows from these stops.
Issue36 stays open. Shared trunk integration remains prohibited by the earlier
review rejection; only this feature-branch review is published. The experiment
worktree and pinned source remain retained rather than claiming trunk closeout.

## New assignment: iteration/gradient fidelity on trained models

The user requested a cap around128iterations, choosing a smaller cap if EP/BPTT
gradient cosine reaches about0.9, measured after some training. The user also
explicitly requested GPT-6.1Sol to run and monitor the simulations after the
implementation and to return first results. This is new authorization beyond
the completed benchmark review; shared-trunk publication remains unapproved.

Use the existing250-update EP checkpoints for small, reference and49-token
models. Wider has only6updates and is excluded from this matched early-training
diagnostic. Use identical first24saved validation examples/masks, three batches
of8; hash checkpoints, cohorts and raw IDX files. Read-only: no optimizer writes
or additional MNIST training. FP32, beta.005, tau1/1.3, dt.5 with the same
energy-backtracking rule and ideal device-domain guards.

Candidates T=8,16,32,64,96,128, applied to each phase: T free plus T positive
and T negative steps (3T serial). Finite-time EP uses both nudges from the same
free state and explicitly permits non-settled terminal states. BPTT differentiates
the exact same T-step free dynamics from zero; discrete accepted line-search
choices are treated as locally constant. Initial states are not detached in
BPTT's free trajectory. Existing strict settled mode remains unchanged.

The comparison reference is1024-step BPTT, qualified against512steps: global
cosine>=.999, relative L2<=.01 and final residual<=1e-5 (FP32 reference floor).
These longer runs are diagnostic references, never production iteration caps.
If reference qualification fails, do not select a cap. Report per-group cosine,
relative L2 and norm ratios for all six parameter groups, residuals, domain
margins, energy-roundoff increases, GPU time and tensor memory. Global agreement
can hide errors in small parameter groups; inspect Q/K separately in the review.

Select the smallest tested T with global EP/BPTT_T, EP/reference and
BPTT_T/reference cosine all>=.9 on every batch. Select a common cap only from
the intersection of passing caps across all three cases, not from initialization
or an average that hides a failed batch. No qualifying cap<=128 is a valid
negative outcome; do not expand candidate caps or change beta silently.

Public interface: `python -m ebl validate --config
examples/energy_transformer_iterations/<case>.json --weights <explicit
step-0250.pt> --output-dir <result-root>/<case>`. Source from a clean pinned
published feature commit. Root:
`/home/filip/analog-research/drn/results/energy-transformer-iterations/20261114-001/`.
Budget<=30localRTX3090minutes,600seconds per case, including understood startup
retries; retain all attempts. No remote host or new SPICE. Outputs are native
RunStore bundles, checkpoint/cohort identities, saved gradients, curves, plots
and cap-selection JSON. Each qualifying case emits a prospective training
configuration with `phase_mode=fixed_steps`, `max_steps<=128`; do not train it
automatically during this diagnostic.

The executing Sol agent owns launch verification, progress, completion inspection
and first-results reporting. Short runs remain directly supervised; if a durable
handoff is needed, use run-watch with bounded dispatch and preserve one owner.
Scientific failures are evidence; only understood operational failures may retry
within the allowance. Commit code/tests/configs before delegation and record
handles/results below. Shared trunk remains untouched.

### Iteration diagnostic results — 14 November 2026

GPT-6.1Sol executed and monitored the three cases from clean published
commit9fc5477. All completed first attempt; aggregate supervisor wall54.653s,
within1800s. All native completeness/hash inspections passed, the saved24-image
cohorts were tensor-identical, and all input checkpoint hashes remained unchanged.
The43targeted implementation/regression tests passed before launch. No optimizer
updates occurred. Results/receipts:
`/home/filip/analog-research/drn/results/energy-transformer-iterations/20261114-001/`.
Native runs are small/20261114T134047.470377Z-af10027e-83d8a08c,
reference/20261114T134105.771512Z-3385e7fd-648ec0a8 and
more_tokens/20261114T134123.609858Z-7d41d030-4029804f.

The long-unroll references qualified on every batch:512/1024gradient cosine
was essentially1, relativeL2<=7.62e-8 and residual<=2.384e-7. Common passing
caps under the predeclared GLOBAL criterion are32,64,96,128. Thus32 is the
smallest tested global selection, preserved in native selected-config artifacts.

| Case | T | Min EP/BPTT | Min EP/reference | Min BPTT/reference |
| --- | ---: | ---: | ---: | ---: |
| small |32|0.98297|0.99771|0.96956|
| reference |32|0.99592|0.99993|0.99524|
| more_tokens |32|0.99011|0.99990|0.98873|
| small |64|0.99968|0.99992|0.99940|
| reference |64|0.99999|1.00000|0.99999|
| more_tokens |64|0.99997|0.99999|0.99998|

**Recommendation:64iterations per phase**, using the stronger group-level
check on already measured caps. At32, small-model encoder EP/reference cosine
falls to0.80528 (relativeL2up to0.59290), and encoder BPTT/reference falls to
0.05752 despite high whole-gradient cosine. More-token key BPTT/reference also
falls to0.87815. At64, every group in every batch and every comparison exceeds
0.9: worst0.98036 (small encoder BPTT/reference). EP/reference itself exceeds
0.9984 for every group; small encoder relativeL2is<=0.06421. This recommendation
is separate from, and does not rewrite, the predeclared global32selection.

Prospective configs: `examples/energy_transformer_mnist/capped64/` for small,
reference and more_tokens. They use phase_mode=fixed_steps,max_steps=64. The
schema enforces a128ceiling in fixed-step mode.64means192serial centered-EP
iterations (64free,+64positive,+64negative), a64fold reduction in the former
4096per-phase maximum; actual speedup versus adaptive convergence was not
measured by this diagnostic. EP32time was0.107–0.108s per8-example batch, EP64
about0.21s. Peak tensor memory EP/BPTT32(MiB):19.09/29.39,21.18/39.85,
32.84/90.33. These are GPU simulator costs, not physical circuit timing.

Very short trajectories can mislead: at8steps the more-token EP/reference
cosine is−0.41674 while EP/same-lengthBPTTis0.97564. Finite-length agreement
alone is insufficient. The initial target0.9 was therefore checked against a
stable longer reference as well. Inspect `analysis/selection.json`,
`recommendation.json`, `comparison.md` and `gradient-agreement.png`/`.svg`.
Solid curves show worst batch; faint dashed curves show mean batch.

All measured triode margins remained positive, max operand<=1.368. Small
positive FP32 energy increments remain roundoff diagnostics, not a strict
descent claim. Evidence covers these three checkpoints after250updates and
24validation examples, not all later training states or the unqualified wider
model. Further full training with the64step recipes is not part of this read-only
diagnostic. Issue36 remains open for the larger analog-transformer objective.


## Authorized continuation: 64-step training — 14 November 2026

The user explicitly instructed us to launch the prepared training continuation.
Run fresh training from the declared initialization for small, reference and
more_tokens, using the published capped64 recipes. All free and positive/negative
phases run exactly 64 iterations; this is finite-time approximate EP, with 192
serial state updates per training update. The 1e-6 residual remains a diagnostic,
not a stopping gate. Preserve all other scientific settings: GPU FP32, seed
20261114, batch 64, beta 0.005, Adam learning rate 0.003, ideal perfect-diode/MOS
model, data splits and masks. No new sweep or retuning is authorized by recovery.

Question: does the shortened solver train reliably across the three qualified
sizes, and how does its masked-image error compare with matched digital attention?
Target 2,000 updates per case, 5,000-example final validation and 10,000-example
test evaluation, with the existing 256-example fixed validation probes and
attention ablations. Report actual update/evaluation coverage; budget/domain
stops remain scientific results. Compare fixed-cohort scores at common update
counts, and final test scores only with the update counts stated.

Reuse the three completed digital_bp native bundles referenced in
`results/energy-transformer-mnist/benchmark-003/coverage.json`, after checking
case/config identities and native artifact integrity. Reuse is explicit, with
no mutation or new BP training. Wider is excluded because its early-training
checkpoint was not qualified by the cap diagnostic. The original settled runs
remain separate historical evidence.

Execution: local RTX 3090 only, through the public EBL train CLI, in the clean
pinned checkout of the next published feature-branch launch commit. Composition:
`python -m experiments.run_energy_transformer_mnist --capped64-from
/home/filip/analog-research/drn/results/energy-transformer-mnist/benchmark-003/coverage.json
--output-root /home/filip/analog-research/drn/results/energy-transformer-mnist/capped64-001
--readiness /home/filip/analog-research/drn/results/energy-transformer-mnist/readiness/gradient-device-checks.json --charged-readiness-seconds 0`.
Interpreter: `/home/filip/miniconda3/envs/py312/bin/python`. Existing numerical
readiness and the 43 passing finite-time/diagnostic tests are reused unchanged;
only the new composition gets proportional focused verification. A retained
supervisor CUDA context prevents the previously observed cold-worker issue.

Budget: at most 7,200 seconds from this launch, at most 2,100 seconds per case,
including 1,500 seconds training and all startup/retry costs. Storage allowance
500 MB per case plus plots. Keep native checkpoints and every attempt. Permit
only the existing bounded retry for a CUDA initialization failure before any
training metric; no silent scientific retry. The supervisor enforces deadlines.

Root: `results/energy-transformer-mnist/capped64-001/`, containing worker.pid,
child.pid, status.json, coverage.json, metrics.jsonl, run.log and result.json;
native cases retain hashes, receipts, summaries and checkpoints. Comparison plots
and tables go in analysis/. Sol owns launch and two initial semantic progress
checks; the verified run-watch service then owns 300-second checks and bounded
incident dispatch, with the coordinator responsible for interpretation. The
45-minute stall limit and explicit hard deadline apply. No remote target,
shared-trunk publication or GPU-wide settings change is included. Feature-branch
publication and pinned execution follow the recorded shared-trunk review exception.


### Capped64 launch and monitoring handoff

Launched 14 November at 16:46:57 UTC from clean published feature commit
`d3222895d75af837df6628ccfa67d7bd22e90b8e` (composition commit `eed01ee`).
Pinned checkout:
`/home/filip/analog-research/drn/results/energy-transformer-mnist/source-capped64`.
The launch contract is stored in capped64-001/launch.json, including the exact
command, PID and hard deadline **18:46:57 UTC**. Supervisor PID 2390137; first
child PID 2390156. The actual supervisor log is `supervisor.log`. Initial small
native bundle:
`small-energy_ep/attempt-01/20261114T164659.941086Z-0718d744-5989ea79`.

Sol verified metrics growth twice (steps 52 to 82; 20,954 to 38,858 bytes), then
continued growth to 100,658 bytes. The coordinator independently verified both
GPU processes, growing native metrics, exactly [64,64,64] phase iterations and
roughly 0.211 seconds per update (303 images/s). Domain margin was positive,
about 0.611. Residuals around 2.3e-4 and nonsettled flags are permitted finite-time
diagnostics; they are not relabelled converged. No startup failure or retry.
18 focused tests passed before launch. All three reused BP native bundles pass
independent artifact-hash/completeness inspection; their saved configurations
exactly match the expected BP recipes, including seed, data, precision, batch,
optimizer settings and 2,000 updates.

Persistent monitoring is armed: watchdog PID **2391340**, configuration
`capped64-001/watch.json`, state `capped64-001/watch-state/`, operational handoff
`capped64-001/handoff.md`. First real observations at 16:48 UTC confirmed a live
supervisor, GPU worker, growing metric bytes (207,271 to 213,853), no probe/log
errors and no incident. Sole monitoring owner is now run-watch, with 300-second
checks, 2,700-second stall threshold and bounded GPT-6.1 Sol incident/completion
review (300 seconds per invocation, at most two per incident and six per day).
The root's initial terminal_cases=3 counts the reused BP rows; completion needs
all six rows and terminal result/analysis, not this count alone. Expected duration
from measured early throughput is approximately 25 minutes for three cases and
evaluation, subject to later solver cost; the hard two-hour ceiling is unchanged.

The completion handoff requires native evidence and coverage review, fixed-cohort
matched-update comparisons, final-test and attention-ablation interpretation,
`analysis/review.json`, and a result append/commit on this feature branch. It must
preserve all attempts, budgets and shared-trunk restrictions. Chat notification
is not configured; watcher status and result records are the delivery surface.

### Capped64 terminal review — 14 November 2026

The supervisor finished at 17:08:54 UTC, charging 1,316.31 seconds of the
7,200-second allowance. Supervisor and last child are absent. Exactly six
terminal rows are preserved: three explicitly reused digital BP bundles and
three new EP bundles, one attempt each. Independent `ebl runs inspect
--verify-artifacts --require-complete --json` passes all six bundles, including
resolved-config digests and declared artifact hashes. Saved recipes, declared
science, identical cohort artifact hash, native summaries, update counts and
retained initial/final/intermediate checkpoints were reconciled. Native manifests
record clean EP source `d3222895d75af837df6628ccfa67d7bd22e90b8e`.
The full review and inspection receipts are in
`results/energy-transformer-mnist/capped64-001/analysis/review.json`.
All original outputs and source checkouts are retained.

Small and reference EP both reach 2,000 updates and complete 5,000 validation,
10,000 test and 256 examples per attention probe. At the same 2,000 updates,
test MSE is respectively **0.0638694 / 0.0640147**, versus digital BP
**0.0524059 / 0.0461110**. EP improves over initialization (0.236340 / 0.236224)
and the mean-image baseline (0.0673463), but trails BP. On the common fixed
256-example validation cohort at update 2,000, EP/BP MSE is
0.0645230/0.0533840 (small) and 0.0651446/0.0472646 (reference).

More_tokens stops scientifically at **1,633 updates** because a finite-time
phase leaves the declared triode domain. Final validation/test and normal
attention probe have zero completed examples; coverage remains explicitly
partial. At the latest common fixed-cohort checkpoint, update 1,500, EP MSE
0.0520096 trails BP 0.0450571. BP reaches 2,000 updates with test MSE 0.0419638;
there is no EP final-test score or matched final-checkpoint comparison.
Uniform/zero attention probes retain 256 examples each (MSE 0.0515984/0.162909),
but the failed normal probe prevents interpreting that pair as a normal-attention
advantage. No retry or scientific adjustment was made.

Small/reference normal, uniform and zero attention probe MSEs are respectively
0.064523/0.072262/0.111892 and 0.065145/0.074465/0.140456. These interventions
support a limited benefit from normal attention at those two checkpoints;
they do not establish useful attention generally or at transformer scale.
Every recorded successful EP training update uses exactly [64,64,64] phase
iterations; minimum recorded training triode margins are 0.3041/0.3744/0.3078
for small/reference/more_tokens. The later domain failure remains valid despite
positive margins on preceding completed updates. Residuals remain nonzero and
phases are not relabelled settled.

EP training seconds / peak allocated MiB are 427.37/23.49, 454.77/32.57 and
358.22/112.94; reused BP costs are 6.86/19.85, 7.50/21.75 and 6.91/31.71.
Measured launcher case receipts are 459.59, 487.32 and 363.49 seconds, all below
2,100 seconds, with no retry or budget correction. Simulator timing and memory
are not analog circuit latency, power or hardware-efficiency evidence. Sparse
probe times do not imply identical wall-time budgets.

Verdict: this exploratory one-seed pilot trains small/reference but underperforms
matched BP; reliable training across all three sizes is not established because
more_tokens leaves the domain. All six terminal outcomes are reviewed, while
scientific coverage remains five fully evaluated rows out of six. No measured
device, SPICE, PDK, silicon or mid-scale performance claim follows. Further
seeds, solver changes or domain investigations require a new scientific
assignment. This closes monitoring for capped64-001 only; broader issue 36
remains open. Feature-branch publication only; shared trunk is untouched and
no chat notification delivery is claimed.


## Active repair objective — 14 November 2026

The user explicitly instructed us to keep the goal active, resolve failures and
continue until the model works. This supersedes the preceding pilot's instruction
to stop and seek a new scientific assignment after failure. It authorizes focused
cause analysis, justified circuit/solver/training corrections, new validation and
retraining within this same analog-energy-transformer objective. The failure is
retained as evidence; completion of that subprocess is not completion of the goal.

Success requires the three declared sizes to finish training and full held-out
evaluation, useful improvement over initialization/mean-image, explicit matched
digital comparisons, and approximately 0.9 or better EP/BPTT gradient agreement
on trained checkpoints with production phases capped at 128. Preserve a stated
scalar energy and reciprocal forces, ideal diode memory, physically valid ideal
MOS mapping, GPU FP32 and honest simulator-versus-circuit limits. Do not disable a
domain check or clip state without deriving the corresponding model. Retain the
historical fixed64 runs and unchanged baselines. The broader small/mid-scale
objective remains active; no completion claim follows from a failure report.

First diagnostic: replay small/reference final2000 and more_tokens step1500/final1633
on the identical saved24 validation examples (3 batches of8), read-only. Trace the
first rejected q/k values and actual MOS voltage/margin equations; distinguish the
conservative operand bound4 from true device-law failure. Compare late EP/BPTT at
64 and128, with qualified long references where valid. Sol owns this bounded
10-minute local-GPU diagnostic under results/energy-transformer-repair/diagnostic-001.
The coordinator owns code/model repair and interpretation. Input/checkpoint hashes
are preserved and checked; any trace beyond the old guard is labelled diagnostic,
not valid historical-model evaluation.

Working resource allowance for focused repairs and evaluation is at most six
local RTX3090 hours, with individual diagnostics/runs bounded and all attempts
charged. Expected first repair validation is minutes; three2000-update training
cases are tens of minutes. No remote hosts or shared-trunk mutation. Before each
new evidence run, record exact changed scientific choices, config/source and the
acceptance checks here; publish the authorized feature branch and use its pinned
checkout. Understood failures trigger diagnosis and continuation within this
objective, rather than asking again for routine corrective work.


### Failure diagnosis and first supported repair

Read-only replay identifies a conservative software cutoff, not a demonstrated
physical transistor failure. At final1633, first64 validation crosses |k|=4 at
free iteration63 (zero-based), sample35 / MNIST trainID50082, head3/token32/
component10: k=4.003088. First64 test crosses at iteration38, sample/testID13,
head3/token39/component10: k=4.020101. Final maxima are4.026628/4.173181.
Tracing these full64 trajectories gives positive physical score/return domain
margins: validation minimumtriode0.32341V, drain0.09867V, weak-inversion0.11472V;
test0.30336V,0.09134V,0.12077V. The summed P-transpose-Q return current reaches
11.97, but is not an individual product-cell operand and must not be bounded as
one. Each return device multiplies one probability by one Q/K component.

The new explicit `device_domain=physical` policy uses the existing device biases
and gains, with NO changes to logical energy, forces, weights or state values.
For the four-device score cell, Vg=1.5 +/-0.05q, Vd=0.3 +/-0.05k and Vth=0.5.
All token pairs within each head/component require
`0.7-0.05*(max_token(abs(q))+max_token(abs(k)))>0`. Individual probability-Q/K
return products additionally require the conservative sum bound
`0.7-0.05*(max(abs(p))+max(abs(q),abs(k)))>0`. Both return directions require
`0.3-0.05*max(abs(q),abs(k))>0`. The weak-inversion bank uses its exact stable
log-source equation and requires positive threshold margin. All accepted states,
including final states, are checked; failures now record phase/iteration, sample,
actual operand and every margin. No domain guard is disabled and no state is
clipped. Historical `declared4` remains the default for historical configs.

A separate late-gradient problem is confirmed. On identical first24 validation
examples (3x8), old64/.5 minimum global EP/BPTT cosines are0.99431(small2000),
0.99849(reference2000),0.98930(more_tokens1500), but0.47555(more_tokens1633).
Old128/.5 improves the last to0.90159 but BPTT/reference falls to0.89456 and one
512/1024 reference fails qualification. This is finite-trajectory disagreement,
not evidence of zero/vanished gradients.

The focused corrective probe uses nominal Euler step1.0 instead of0.5, retaining
per-example energy backtracking, the same reciprocal dynamics and tau_y=1.3.
At128iterations on final1633 it passes the predeclared global rule on all3batches:
minimum EP/BPTT0.975025, EP/reference0.992244, BPTT/reference0.955585. All512/1024
references qualify (cosine>=0.999999976, relativeL2<=0.00031768, residual<=4.768e-7).
Q/K cosines exceed0.99038 across all comparisons. Every EP/BPTT group exceeds
0.91597 and every EP/reference group0.96102; BPTT/reference position has a0.80091
exception, so this is not an all-groups/all-comparisons claim.64steps atdt1 still
fails the global rule. Therefore select128/dt1 for the repair; the user ceiling128
is respected. Finite-time phases remain approximate, not relabelled equilibria.

Diagnostic provenance, unchanged checkpoint hashes, original/copied solver source,
saved gradients and exact traces are retained under
`results/energy-transformer-repair/diagnostic-001/`; original diagnostic took63.6s
and the dt1follow-up15.73s, plus short traces, within the declared allowance.

Repair evidence: fresh more_tokens training from the original initialization,
2,000updates, beta0.005, Adam0.003, FP32,128iterations/phase, nominaldt1,
physicaldevicechecks. Config:
`examples/energy_transformer_mnist/repaired128/more_tokens-energy_ep.json`.
Reuse small/reference capped64 EP and all three digital arms after their verified
completion; these are separate explicitly labelled solver configurations, not a
claim that all arms use128. Neither successful run needs replacement to repair49
-token coverage. Evaluate all10,000test and5,000validation examples and attention
probes; then run the native trained-checkpoint gradient diagnostic on the repaired
final model. A failed evaluation or fidelity check keeps the overall goal active
and triggers the next focused correction within the authorized resource allowance.

Composition command from the next clean published pinned feature commit:
`python -m experiments.run_energy_transformer_mnist --repair-from
/home/filip/analog-research/drn/results/energy-transformer-mnist/capped64-001/coverage.json
--output-root /home/filip/analog-research/drn/results/energy-transformer-repair/training-001
--readiness /home/filip/analog-research/drn/results/energy-transformer-mnist/readiness/gradient-device-checks.json
--charged-readiness-seconds 0`. Training cap1,500seconds; case wall2,100seconds;
composition ceiling7,200seconds remains within the six-hour repair allowance.
Retain250-update checkpoints and failed attempts. Native schema now accepts any
explicit trained checkpoint for replay and restores its declared solver/device
policy; initialization remains forbidden in that diagnostic.


Production repair gate: all3minibatch EP128 and BPTT128 gradient vectors are
bit-identical to the saved dt1diagnostic (maximum absolute difference0). Read-only
replay of old final1633 under physical policy, originaldt0.5/T64, now completes
5,000validation and10,000test examples: MSE0.05207266/0.05144100, maxoperand
4.24529/4.31232, minimumtriode0.29804/0.29142V, drain0.08774/0.08438V and weak
inversion0.09712/0.09582V. Thus the earlier missing evaluation was caused by the
conservative bound, not a violation anywhere in these complete replay cohorts.
Worst residuals0.43008/0.65369 remain finite-time diagnostics. This corrected-policy
replay is separately labelled and does not rewrite the failed historical bundle.
The checkpoint hash and parameters remain unchanged. Gate duration25.40seconds;
source hashes and full evidence are in diagnostic-001/production-repair-gate.json.
