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
