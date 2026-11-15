# Larger pseudoconductance attention–DRN on CIFAR-10

User assignment: scale the previous shallow pilot substantially and train longer.
Explicit allowance: up to12hours on the local RTX3090, including readiness,
diagnostics, controls and retries; no other host. This is an exploratory scaling
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
deadline established at first GPU readiness execution. A time-limited result
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
remaining original deadline; no implicit allowance reset or remote failover.

Monitoring: executing agent through readiness and first training progress,
then one persistent run-watch owner with300s checks and bounded incident
dispatch. Metrics every25updates and each epoch, checkpoint everyepoch plus
mid-epoch periodic saves,45-minute semantic stall gate. Launcher enforces
hard shared deadline without relying on agent availability. Exact handles,+configs, epochs, throughput, deadline and handoff added before transfer.
