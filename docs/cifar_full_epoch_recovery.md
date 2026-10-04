# Five full CIFAR recovery epochs, PCM and OM

User request: repeat recovery for about five epochs and include OM devices.
This new exploratory study leaves the previous 20-pass/5000-image experiment
unchanged. One epoch now means the entire fixed **45000-image adaptation
split**, with 5000 images reserved for development (the existing balanced
1000-image development prefix is evaluated). No recovery augmentation.
Five epochs = 225000 image presentations and 3520 minibatch updates at batch
64. Evaluate all 10000 test images at deployment and after every epoch;
there is no test-based selection or early stopping.

Use the original public CIFAR-10/100 ReLU ResNet-32 teachers, original split
seed 20260917, four final convolutions plus classifier, frozen prefix and BN
running statistics, fixed mapping scales, and original digital-teacher KL
at temperature one. The teacher previously saw all 50000 training images;
development is held out from adaptation, not from teacher pretraining.

Primary sources: direct digital and standard device-specific HWA. The six
source families of the previous PCM study are not all repeated: noise-enhanced
HWA and per-fault CDT remain prior evidence, not new five-epoch arms.
PCM reuses the exact prior development-selected standard HWA weights.
OM trains a separate standard HWA model from the same digital checkpoint:
30 full augmented training epochs, Adam teacher-KL 1e-4, learning rate reduced
tenfold after epochs 15 and 25, all suffix weights and digital calibration
trainable. Sample fresh residuals each minibatch from the existing 41x512
target-conditioned OM endpoint kernel, fitted on independent literal device
identities. The independent held-out kernel must pass the existing mean-error
adequacy gate. Select the epoch by mean development KL over two fixed held-out
kernel draws, including epoch zero. Flag late improvement and the single
limited recipe; do not call this an exhaustively tuned or converged OM oracle.

For every dataset/backend: nominal, 1% stuck-low, 1% stuck-high, 1% random
stuck; two sources; three fresh arrays (151001–151003, endpoint seeds
161001–161003). Five controls restore each exact P0/model/RNG:
no recovery, digital calibration only, unchanged-target rewrite plus
calibration, weights only, and weights plus calibration. Each adapting arm
runs all five epochs; persistent Adam state; identical epoch-specific shuffles.
Fixed recovery LR 1e-4; calibration LR 3e-4 CIFAR-10 and 1e-3 CIFAR-100.
These are inherited base rates, not newly optimized full-cohort rates.

PCM retains the existing differential-pair Gaussian programming endpoint law,
25 uS nominal scale, per-bank Bernoulli faults with permanent zero/25 uS/
Uniform[0,25] stuck states. Learned targets are fully reprogrammed after each
minibatch; rewriting has the same schedule. No added pulse/P&V loop. This is
digital gradient/optimizer plus endpoint reprogramming, not pulse training.

OM uses literal AIHWKit 1.1.0 ReRamArrayOMPresetDevice identities, with the
existing counterfactual repair of native corrupt identities before independently
injecting the declared Bernoulli failures per active cell. Keep every sampled
reference exactly; transfer uses apparent q=a-r in a standard MVM, not a passive
DRN solve. Each logical weight has one abstract active state plus its reference;
this is not a claim about a fabricated device count or four/eight-device DRN
encoding. Stuck-low means the cell's own lower bound, stuck-high its own upper
bound, and random a fixed uniform draw between those bounds. It does **not**
mean PCM's zero/25 uS endpoints. Persistent stuck states remain fixed; native
apparent write noise can change on attempted writes, as in the existing plant.
Evidence class is `model_based_aihwkit_preset`, without absolute conductance
or physical energy claims.

OM initialization starts at sampled RESET and permits at most 128 apparent
feedback pulses/cell, tolerance 0.5 nominal dw_min. Recovery uses the existing
closed-loop PulseWriter Adam with an accumulated target, one verify/pulse
opportunity per minibatch and a total cap of 640 pulses/cell across all epochs.
The rewrite control uses the same opportunity and total cap toward frozen source
targets; actual pulse counts are reported rather than assumed equal. Healthy
and failed cells both consume commanded pulses. Calibration never writes.
Report apparent and persistent evaluation, cap saturation and verify reads.
The controller receives observable reads and gradients, never identity masks.
OM pulses and PCM full programming events are different costs, not comparable
energy units. Existing standard-crossbar mapping is reused; a new DRN encoding
or codebook/quantization claim is outside this experiment.

Expected coverage: four dataset/backend studies; each cache, sources and three
screen stages; 40 controls per screen (32 learning trajectories), 120 controls
and 480 post-recovery test endpoints per study. Canary uses a labelled 128-image
cohort, one HWA epoch and one array but exercises all five recovery epochs,
fault cases, controls, native artifacts and replay before full launch.

Native runs enter through EBL and reuse the dependency campaign executor.
Record input hashes, population identities, P0, all epoch checkpoints, optimizer
states, RNG and prefix invariance. Report mean/sample-SD and paired differences
across three arrays, including comparisons against equally trained calibration
and rewrite. Accuracy, KL and write costs are complementary outcomes. This
study cannot establish that recovery is necessary if only an undertuned HWA
baseline is beaten. Retain negative outcomes and limited HWA convergence.

Checkpoint interpretation: PCM's `master_target` is its learned digital target.
For OM the authoritative accumulated controller target is `writer.target`;
the generic `master_target` field retains the unused initial source vector.
Replay uses the saved physical array and calibration state for both devices.

Compute: one authorized 5090 per dataset (loulou/fifi); nom-cool-1 samples CPU
AIHWKit identities. Existing unrelated jobs are untouched. The native run ledger
records launch handles, logs and receipt paths. Verify process and semantic
progress immediately, inspect at least every 30 minutes, diagnose no progress
for 45 minutes, and collect/artifact-verify all declared coverage before closeout.
