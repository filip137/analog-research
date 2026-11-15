# Larger pseudoconductance attention–DRN on CIFAR-10

User assignment: scale the previous shallow pilot substantially and train longer.
Explicit allowance: up to12hours on the local RTX3090, including readiness,
diagnostics, controls and retries. On16November the user additionally authorized
nom-cool-1 as fallback ("launch otherwise nom cool1"). This is an exploratory scaling
experiment, not a paper benchmark or a physical circuit/energy measurement.

Question: can the direct-exp L1 attention + passive perfect-diode DRN hybrid
learn useful CIFAR features at depth and reach substantially higher accuracy?
Digital CIFAR reference points are94–96% (Jordan2024, arXiv:2404.00498);90% is
a useful first aspiration, not a promised outcome or a retry-until-pass gate.
The earlier one-block56,960parameter model reached45.31% after2,000updates.

Target: six width256 blocks, eight32-component heads,64patch tokens (4x4 RGB),
DRN hidden1024, dual input/output rails and positive conductances with unit
shunts. About8M parameters, measured exactly at readiness. Keep the direct-exp
negative-L1 controls, value bound0.75, null anchor0.01 and capped32GS sweeps
per free/nudged phase. EP remains local to the separately buffered attention
and DRN blocks; scorer/projections/normalizers/classifier/AdamW remain digital.
DRN differential gain scales from32 at width64 to128 at width256, preserving
the initialization scale; no claim this ideal gain has a designed circuit.

Two cases: primary large DRN/EP, and same-depth/width/scorer digital-MLP/BP
architecture control. Initial shared tensors and data/RNG states match.
The digital control has fewer parameters and is not an EP estimator control;
same-checkpoint EP/BPTT probes provide that comparison. No pretraining.
Split45k/5k/10k with priorseed4100; train random crop/flip plus mixup;
validation selects the best checkpoint, test only after selection. Standard
pre-norm residual blocks, trainable LayerNorm, final norm, AdamW with warmup
and cosine decay. Exact recipe and epoch count frozen after a bounded
throughput/memory/gradient readiness measurement, before evidence launch.
At most200epochs; primary allocation at most10hours, control at most1hour,
readiness/final diagnostics reserve1hour, all subject to a shared12-hour
deadline established at first successful GPU readiness execution, less charged
failed-startup time. A time-limited result
is honestly labelled truncated. No scientific changes to an active case.

Optimization: eliminate value-node updates algebraically from attention GS
to reduce temporary memory, preserving the capped finite-step map and nudges.
Verify against the existing explicit-state implementation, autograd and finite
differences. Do not replace the DRN with a feedforward surrogate or an exact
uncapped solve. Use FP32 withoutTF32 for finite-nudge sensitivity.

Gradient checks: fixed validation cohort at initialization, selected milestones
and best/final checkpoints; compare EP and identical32-step BPTT, then64-step
BPTT reference. Report all parameter groups; cosine>=.99 and relL2<=5% for
non-negligible reference norms. Detect finite-state/gradient failures.
Readiness includes a full optimizer-state/RNG resume replay.

Retention: input hashes, initial weights, rolling atomic latest full-state
checkpoint, validation-best full-state checkpoint, final state, metrics and
gradient diagnostics. Rolling latest/best files are explicitly replaceable;
no per-epoch archive. Cap storage8GB in/tmp/pseudo-drn-large-20261115 (211GB
free at intake; home has88MB free). Records and source versioned in backend.
Native EBL CLI/bundles; launch source pinned after merge/push. Failures retained.
Recovery may resume the same settings from a complete checkpoint within the
remaining original deadline; no implicit allowance reset. Nom-cool-1 is now
explicitly authorized as fallback; other remote hosts remain outside scope.

Monitoring: executing agent through readiness and first training progress,
then one persistent run-watch owner with300s checks and bounded incident
dispatch. Metrics every25updates and each epoch, checkpoint everyepoch plus
mid-epoch periodic saves,45-minute semantic stall gate. Launcher enforces
hard shared deadline without relying on agent availability. Exact handles,
configs, epochs, throughput, deadline and handoff added before transfer.

## Readiness status

Tracking [issue53](https://github.com/filip137/on-chip-learning/issues/53).
Implementation contains7,902,474primary parameters (138.74times the previous
56,960). Six blocks, width256, hidden1024, heads8. Eight numerical tests pass,
including explicit/reduced capped attention and both nudged phases at1/3/8/32
sweeps, EP/BPTT derivatives, shared initialization and optimizer/RNG replay.
A full-size CPU gradient check on two examples passed: worst relative L2
0.418%, minimum cosine0.99999127, zero32/64-step forward discrepancy.
Raw result:/tmp/pseudo-drn-large-20261115/cpu-initial-gradients.json.

GPU readiness is currently blocked before any model execution. PyTorch2.5.1
and an independent ctypes driver call both fail with CUDA initialization
error3. Device files open and driver ioctls return without OS permission
errors; NVIDIA575.57.08 sees the idle3090. nvidia_uvm has zero users, but
sudo requires a password. Xorg uses the GPU, so no device reset or display
driver reload was attempted. Repeated same-process and NVML-retention
diagnostics did not initialize CUDA. Failed logs:/tmp/pdl-benchmark.log,
/tmp/pdl-benchmark-attempt1.log,/tmp/pdl-benchmark-nvml.log;
syscall trace:/tmp/pdl-cuinit.trace. User asked to restore CUDA; no secrets
requested and no GPU-wide settings changed.

Budget accounting correction before evidence launch: the user authorized
12GPU hours, with no absolute calendar deadline. No CUDA context succeeded,
so idle time awaiting administrator action must not consume GPU training time.
Conservatively charge300seconds for all failed startups/diagnostics so far;
42,900seconds remain. The initial attempted clock is preserved in
/tmp/pseudo-drn-large-20261115/budget.first-attempt.json. Active budget.json
records awaiting_working_cuda with deadline unset; the benchmark arms the
remaining clock once CUDA actually initializes. After that, the deadline is
immutable through smokes, runs and retries. This preserves, rather than
renews, the user's allowance. GPU throughput, public-CLI smoke and full-size
resume replay remain pending. No full training run or durable monitoring
handoff is claimed yet.

## GPU readiness and frozen launch recipe (16 November)

Local cuInit briefly returned zero, but PyTorch still failed to allocate. The
user-authorized fallback nom-cool-1 passes actual CUDA allocation. It has an
idle RTX3090 UUID GPU-2262d56b-2c1d-1546-3e93-2e441399cd4f, PyTorch2.5.1+cu121,
and667GB home/195GB tmp free at placement. No other compute users were present.
Copied identical CIFAR batches to the same absolute data path; native manifests
verify all six hashes. Source c4f0843 was shallow-cloned with its original Git
identity and staged under/home/filip/analog-research/pseudo-drn-large-20261115/
source-c4f0843. No other worktree or GPU setting was modified.

Measured batch64 update time: DRN0.53045s, digital0.10335s. Peak allocated memory
2,412/2,157MiB respectively. After12readiness updates, worst EP/BPTT/reference
relative L2 is0.2723%/0.0223%; all groups pass, with zero32/64-step forward
discrepancy. JSON/logs remain in the remote/tmp/pseudo-drn-large-20261115 root.

Freeze80epochs for both cases before training, preserving all other recipe
settings. There are704updates/epoch and56,320updates/case. Predicted training
time is8.30h DRN and1.62h digital, plus validation/checkpoint overhead. Allocate
at most9h DRN and2h digital, leaving the original shared allowance for readiness
and diagnostics. This replaces the earlier provisional100epoch/10h/1h split
before any evidence run, to retain equal epoch coverage within12hours.

The remaining42,900-second clock started2026-11-15T23:03:13.801678Z and ends
2026-11-16T10:58:13.801678Z (11:58Paris). It is fixed through readiness,
staging, training and retries; no fresh budget from host migration. Remote
context keeper PID2206375 prevents cold starts while native CLI smokes and
resume replay are prepared, then exits before queue admission. Main and
control will be one serial prepared job owned by the existing shared GPU queue;
no duplicate per-study watcher. Launcher forwards native step/epoch progress
and keeps children in the queue's process group for enforceable hard stops.

## Prepared queue handoff

Native GPU readiness passed on pinned source
745ebc97c53728b029f053a21adcd8c5a6a3cf85: smoke, fixed-cohort gradients,
artifact verification and full-state interruption/resume. Resumed update4 equals
uninterrupted update4 exactly (model/optimizer/RNG maximum difference0), from
interrupted update2. Shared initialization has55identical common tensors.
Remote receipt:/tmp/pseudo-drn-large-20261115/readiness.json.

Queue job:pseudo-drn-large-cifar-20261115-v1, permitted target nom-cool-1 only.
Prepared transport and main study record are in on-chip-learning/
explorations/pseudo-drn-large-20261115/. Both80epoch cases run serially from
/home/filip/analog-research/pseudo-drn-large-20261115/source-745ebc9 using
/home/filip/miniconda3/envs/py312/bin/python. Source remains pinned despite
later collector/documentation commits. Budget stop remains10:58:13UTC16November.

Shared queue ownership replaces the provisional separate300s watcher: service
poll60s, semantic progress through root/supervisor.json, startup grace300s,
stall threshold2700s. The serial wrapper stays in the queue process group.
No blind replay: retry_safe=false. Native full-state resume is verified, but an
incident must first prove old descendants stopped, retain bundles and prepare
compatible resume commands under the original deadline. No scientific changes
or allowance reset. Other remote hosts remain unauthorized.

Remote validator:/tmp/pdl-collect.py; controller collector is published in
731931e. Collection target:/tmp/pseudo-drn-large-collected-20261115. It checks
native artifact hashes and evaluation counts, plots validation curves and queues
this originating task for review (thread01a1e07c-65dd-7c91-98a8-0a489f574997).
A successful execution is not a completed scientific closeout; review achieved
coverage, truncation, gradient gates and both final accuracies before concluding.

Admission confirmed2026-11-15T23:16:27Z. Queue attempt
200f00b2d3914b909baf635634aea528; remote queue supervisor2209675,
serial wrapper2209676, native DRN worker2209722. Native DRN bundle:
/tmp/pseudo-drn-large-20261115/drn/20261115T231629.027932Z-d6d8d737-504334b3.
At23:17:11Z, update75 is recorded with loss2.24055, gradnorm1.27332,
0.54284s/update and2,412MiB allocated. Initial checkpoint and gradient
probe artifact exist. Actual NVIDIA process memory including caches/contexts
is3.8GiB on the exclusively reserved24GiB GPU. The completion validator also
passes CPU verification of the completed readiness bundles.
