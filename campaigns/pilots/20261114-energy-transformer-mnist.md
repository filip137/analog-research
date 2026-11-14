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
