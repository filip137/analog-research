# Sparse equilibrium adaptation first pilots

User authorization: 9 November 2026, test the Explorer proposal, first runs at
23:00 Europe/Paris on Jean Zay V100 dev queue. Earliest start is
2026-11-09T23:00:00+01:00. This is exploratory simulation evidence.

Question: do sparse conductance updates provide useful task adaptation beyond
matched sparse explicit updates, and does the difference depend on nonlinear
constraints? The shared campaign and task exploration are in
`on-chip-learning/campaigns/sparse-equilibrium-adaptation/`.

## Frozen first screen

One seed (271109), ideal writes, no faults, drift or read noise. Report ideal
device assignment commands and unique cells; no pulse or measured energy claims.

1. Small graph: 4x4 grid, grounded quadratic energy, linear and nonnegative
   variants. A target domain changes four edge conductances. Paired input cohorts
   of 512 train/128 development/256 test. Compare frozen, random/Fisher/known-edge
   sparse graph updates (K=4), full graph and explicit MLP controls after source
   fitting. Source fitting and adaptation are separate. This circuit-generated
   target is a mechanistic positive control, not broad superiority evidence.
2. OPT-125M: existing crossbar std3 and DRN std3 sources; adapt MLP indices 6–11.
   Counterfactually augmented sentiment at repository commit
   `6f232a1d2a11462a30ce08fb4825b734ab30828e`, paired original/revised data.
   Fit source on originals for 128 updates, then clone that state for adaptation.
   Adapt 128 updates at effective batch 8, context cap 128 tokens, microbatch 1.
   Use training pairs only; development pairs select among rates 3e-4 and 1e-3.
   Test stays unopened by the training worker in this first development screen.
   Full analog, random sparse and empirical-Fisher sparse use normalized physical
   device coordinates and Adam; K=46,080, fixed equal quotas over 12 matrices.
   Fisher uses squared individual-example gradients on 16 fixed training rows.
   Digital controls: input/output boundary affine, scoped MLP bias-only and
   rank-5 parallel linear LoRA. Each gets the same two-rate allowance. LoRA has
   46,080 scalars; boundary/bias counts are reported at their actual sizes.
   Frozen attention, embeddings, norms and LM head throughout. Binary label-token
   logit scoring uses the same prompt and positive/negative tokens for both models.

The OPT source-fit rate is 3e-4 in normalized conductance coordinates. This is
a first rate screen, not a claim of converged capacity. The short context may
truncate a causal edit: record identical-token pairs and exclude them from paired
adaptation/evaluation, preserving exclusions in the data receipt. Keep standard
dataset split assignments and check original/revision grouping and duplicates.
Original identity uses exact text after whitespace normalization. Pairs without
one unambiguous original are excluded with an explicit reason; no fuzzy matching.
Development pairs overlapping training text after whitespace normalization are
excluded as whole pairs; retain their IDs and reasons in the data receipt.

DRN forward uses 24 total sweeps in both training and evaluation (16 settling
plus 8 differentiated in training). Canary compares against longer relaxation
and a longer gradient tail; failures block the batch, without silent retuning.
Sparse physical writes affect only declared devices; crossbar references remain
fixed. Source fits and each method's starting metrics are retained separately.

## Decisions and budgets

Primary graph endpoint: held-out MSE vs ideal write commands, with active-set
and response-rank diagnostics. Primary OPT endpoint: development original,
revised and both-correct pair accuracy, alongside source-fit headroom and writes.
Null/negative comparisons are valid. Single-seed development results do not
support a confirmed architecture advantage.

One same-wrapper canary (array 0-0, <=20 min), then array 0-1%2: task 0 runs
graph + MLP sentiment; task 1 runs DRN sentiment. Each full allocation is bounded
by 2 h on one V100-32GB, 10 CPUs, account umg@v100, partition gpu_p13,
qos_gpu-dev; PyTorch module pytorch-gpu/py3/2.5.0. Maximum allocated allowance
4 h 20 min. Source/data/results combined <=12 GB incremental storage.
Stop at the wall-time guard and preserve completed cases as partial coverage;
no automatic extra allocation. A failed canary prevents full submission.
Require completion by 06:00 Paris on 10 November; Slurm's deadline bounds late
starts. The worker stops after 6,600 seconds (canary 1,050), leaving teardown
time inside the allocation. GPU memory is V100-32GB. IDRIS assigns host memory
from the allocation and forbids explicit `--mem` options; record the actual grant.

Data freeze: 1,630 train and 231 development pairs. Training exclusions: four
ambiguous original identities and 73 identical truncated prompts. Development:
one ambiguous identity, 12 identical prompts, one training-text overlap. The
prepared data preserves original/revised labels, token IDs, pair IDs, text hashes,
truncation flags, excluded IDs/reasons and pinned source-file hashes.

## Handoff

Branch: drn/sparse-equilibrium-pilot. Numerical work uses the public
`python -m ebl train --config <config.json> --output-dir <destination>` and native RunStore
bundles. The exact staged commit, commands, tests, Slurm IDs and durable watch
are filled before scheduling. Scheduling/resume must never duplicate a parent.
The watcher owns gating, observation, collection and the terminal review handoff.

Implementation: `experiments/sparse_equilibrium/launch.py` reuses the native Slurm
accounting and locking helpers. Launch configs are in `examples/sparse_equilibrium/`.
The same `recovery.slurm` wrapper handles the explicit canary mode and production.
`launch/manifest.json` in the staged source records both exact commands, all
effective source/config hashes and the passing static receipt. The release command
checks successful singleton accounting, native artifact hashes, coverage and logs,
then submits one array. A persisted uncertain submission blocks duplicate attempts.

Static checks: `bash -n experiments/sparse_equilibrium/recovery.slurm`, compilation
of changed Python, and the sparse-equilibrium, architecture-boundary, public-CLI
and native Slurm tests. Source and configs are immutable after the live canary.

### Overnight owner and exact handoff

Current state (9 November, 19:15 Paris): canary parent **2198883**, task 0,
`PENDING (BeginTime)`, eligible/start estimate 23:00 Paris, 20-minute limit.
Exactly one parent is recorded in `scheduler.json`; production is prepared and
has not been submitted. `scontrol` confirms `umg@v100`, `gpu_p13`, `qos_gpu-dev`,
`v100-32g`, one GPU, 10 CPUs and scheduler-derived `mem=40000M`. Queue availability
can delay execution. The final candidate passed 67 tests, Python compilation,
shell syntax and both exact `sbatch --test-only` commands.

Watcher PID **2119883**, dispatch enabled; its real initial probe confirms the
canary queue state and both gated production tasks. Local state is
`<local results>/watch-state/`, with `monitor-armed.json` recording ownership and
observation. Next routine check is every five minutes, with a poll at 23:00;
deadline 10 November 06:00 Paris. The next scientific handoff is the live-canary
completion review, followed by exact gated release. No numerical result exists yet.

Launch source: `9a15c50c` (full SHA in `launch-receipt.json`); numerical source
unchanged from `c173efed`. The first large staging attempt hit the WORK quota.
Only that attempt's new files were removed. The reduced runtime archive is retained
under `$WORK/sparse-equilibrium-20261109/runtime-source.tar.gz`; its execution copy
is `$SCRATCH/sparse-equilibrium-20261109/source-9a15c50c`. Historical lab artifacts
are omitted. The reduced tree passes the same checks. No canary preceded this fix.

Remote root: `/lustre/fsn1/projects/rech/umg/ucy17uy/sparse-equilibrium-20261109`.
Manifest: `<remote root>/source-9a15c50c/launch/manifest.json`.
Local receipts and collected evidence:
`/home/filip/analog-research/drn/results/sparse-equilibrium-20261109/`.
Native outputs are `<remote root>/{canary,mlp,drn}/run`; Slurm logs are in
`<remote root>/slurm/`. Per-stage `completion.json`, native `result.json`, metrics,
source-fit states, per-arm JSON and trainable-parameter checkpoints form evidence.
Source-fit writes are separate from adaptation writes; these are ideal assignment
commands, not measured programming pulses or energy. The K-cell mask budget counts
addressed cells; the four-rail DRN has more eligible cells than the crossbar.

The sole overnight owner is the installed `run-watch` service, using the adjacent
`20261109-sparse-equilibrium.watch.json`. It polls every 300 seconds and holds the
shared per-run owner locks. It checks queue/accounting state, native artifacts,
metric growth and bounded logs. Its bounded incident agent is Astra/medium,
900 seconds maximum, four invocations per 24 hours, at most two per incident.
The local host must remain awake and connected. No external messaging or automatic
chat callback is configured; decisions and evidence are written to local status.
Read this worktree's `AGENTS.md` and `.agents/skills/experiment-loop/SKILL.md`
before editing the backend handoff or collecting/reviewing native results.

The canary completion incident owns the already-authorized batch release. Read
its stdout/stderr and both baseline/audit JSON files. Check all 30 records (18
graph, six MLP, six DRN), finite metrics, gradient/convergence audits, physical
mask isolation, actual timing and memory. Then use the exact prepared release:

```bash
ssh -o BatchMode=yes jean-zay 'bash -l -s' <<'REMOTE'
set -euo pipefail
cd /lustre/fsn1/projects/rech/umg/ucy17uy/sparse-equilibrium-20261109/source-9a15c50c
PYTHONDONTWRITEBYTECODE=1 /lustre/fshomisc/sup/hpe/pub/miniforge/24.9.0/envs/pytorch-gpu-2.5.0+py3.12.7/bin/python -m experiments.sparse_equilibrium.launch release --manifest "$PWD/launch/manifest.json"
REMOTE
```

The command verifies the frozen candidate, static receipt, singleton canary
accounting, logs and every native artifact hash, writes the gate receipt and
submits exactly one production parent. The same command safely returns an already
recorded parent on resume; an uncertain submission requires scheduler reconciliation,
never a blind retry. Preserve any partial/failed evidence. A failed canary blocks
production. No new scientific settings, extra allocation or alternate host is
authorized by this first screen. Do not touch unrelated jobs.

Immediately verify the new parent has indices 0 and 1, concurrency two, the declared
account/QOS/constraint/resources, and a live run-watch PID/owner lock. Watch the
first running task through `environment.json` and its native manifest/first metric;
if it stays queued, retain the pending reason and return to the durable watcher.
Record parent IDs and gate paths below and in the shared campaign handoff.

On each completion, collect its native bundle locally with `rsync -a` from that
explicit path and use `python -m ebl runs inspect <local run> --verify-artifacts
--require-complete --json`. Require 30 records for MLP (18 graph plus 12 sentiment)
and 12 for DRN. Collect the scheduler/gate receipts and Slurm logs too. A wall-budget
stop is partial coverage: collect completed per-arm artifacts and label missing
cases, without new allocations or an invented complete result.

Review against the question: report graph linear/constrained outcomes, source-fit
mismatch/headroom, dev original/revised/both-correct accuracy and loss, parameter
and physical-write counts, each rate and timings. Compare random/Fisher against
full analog and the zero-write digital controls within each architecture before
cross-architecture conclusions. This is one-seed development evidence; do not claim
implicitness, EP or noisy-hardware superiority. Save the interpretation and next
decision here and link it from the shared campaign. Commit the handoff/review on
`drn/sparse-equilibrium-pilot`; integrate/push at closeout per the experiment-loop
skill. Do not remove this worktree or the pinned local runtime while any watcher
needs it. The final completion incident owns the scientific review, not just a
scheduler-success check.

### Canary review and gated release — 9 November, 23:08 Paris

Canary parent `2198883` completed with `0:0` in 85 seconds of allocation wall
time (78 seconds for the worker step). All 30 records are present: 18 graph,
six MLP and six DRN. The collected native bundle passed `ebl runs inspect
--verify-artifacts --require-complete --json`: 32 artifacts, no errors. All
JSON artifact numeric values are finite. Both DRN audits have zero forward and
gradient relative error and cosine approximately 1; the completed sparse arms
passed the runner's unchanged-physical-base assertions. Logs contain only the
expected module loading and successful input checks. V100 memory capacity is
32,768 MiB; host grant is 40,000 MiB. Peak GPU memory was not recorded, so no
peak-memory claim is made. Canary update timings were 0.020–0.024 s for MLP
and 0.107–0.147 s for DRN at smoke batch 1; these do not forecast full-batch time.

The two-pair, two-update smoke is readiness evidence only. Source fitting raised
development loss from 0.527 to 1.338 (MLP) and 0.611 to 4.541 (DRN), which must
remain visible when reviewing production source headroom. Graph MLP source fits
are also poor after two smoke updates. Neither result justifies retuning or an
architecture conclusion. The next decision remains the frozen development screen.

The exact unchanged release command passed the source/static/native/scheduler
checks and submitted **one production parent `9694`** at 23:07:17 Paris.
`canary-gate.json`, `gate.json`, `scheduler.json`, `production-submission.json`
and Slurm logs are collected under the local results root. Array `0-1%2` started
at 23:07:24: `9694_0` (raw ID `9703`, graph + MLP) and `9694_1` (raw ID `9694`,
DRN), both on `r7i5n3`, each with one scheduler-assigned V100, 10 CPUs per task,
40,000 MiB host memory, `umg@v100`, `gpu_p13`, `qos_gpu-dev`, `v100-32g`, and a
two-hour limit. Physical GPU indices are managed by Slurm. Both environment
receipts name the frozen source and Python. No duplicate parent or extra
allocation was created; the original 4 h 20 min allowance is unchanged.

Watcher PID `2119883` and its watcher lock plus all three per-run ownership locks
were verified live outside the sandbox process namespace. It retains the next
five-minute checks, production collection and scientific review. No source,
scientific configuration, checkpoint or runtime allowance was changed.

Immediate post-release verification also confirmed both native manifests and
first metrics: task 0 published graph frozen/step-0 metrics; task 1 published
DRN loaded with 113,246,208 eligible devices. Future checks belong to run-watch.
