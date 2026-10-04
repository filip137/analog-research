# CIFAR mixed-rate stuck-device sweep

Approved plan: complete CIFAR-10 first, then CIFAR-100; pretrained ReLU ResNet-32,
last four or eight convolutions plus classifier; Gaussian endpoint PCM and
AIHWKit 1.1.0 OM. This protocol governs only `cifar_crossbar_fault_sweep.v1`.
The prior experiments remain unchanged. Evidence is exploratory/model-based.

Fault rates are nominal plus 1%, 2%, 3%, 5%; types low, high, random-stuck.
PCM probabilities are per differential-bank device (0/25/u[0,25] microSiemens).
OM uses repaired literal identities, one abstract active cell/logical weight,
exact intrinsic references, native per-cell low/high/uniform-between-bounds
stuck states. There is no calibrated OM absolute-conductance or energy claim.
No passive DRN or four/eight-device encoding is introduced.

Layer-name-bound identities and RNGs retain common devices across analog depths.
Independent uniform masks produce nested rates and matched masks across fault
kinds and sources. Every control restores the same source-specific P0 and RNG.
Final arrays 251001–251003, endpoints +10000; development 211001–211003;
canary 271001; kernel fit/heldout 231001/231002. All are independent of prior
studies. Three physical repetitions do not constitute three HWA training seeds.

Sources: digital, standard HWA, noise HWA, CDT low/high/random. Generic sources
cover all 13 cases; each CDT covers nominal and its own type at four rates.
54 source/case combinations x 3 arrays x 5 controls per dataset/backend/depth.
Four studies/dataset = 3240 controls, 2592 learning trajectories and 12960
post-adaptation full-test endpoints. Both datasets double those counts.
Controls are none, calibration, frozen-target rewrite+calibration, weights only,
weights+calibration. Complete all outcomes, including degradation.

Every fit starts from the original teacher. Adam teacher KL, fresh crop/flip,
45000 training images, batch256, constant LR grid 3e-5/1e-4/3e-4. Nine generic
fits cross LR with error strengths 1/2/5. Standard selects nominal KL from
strength1; noise HWA selects mean nominal plus all 2/3/5% case KL. CDT uses
that selected strength, three LRs/type; equal-probability resampled 2/3/5%
each minibatch and fresh hypothetical faults. Nominal and own 2/3/5% cases
select CDT. The 1% reference is never used for HWA selection. Original teacher
KL at T1 on balanced1000 development images, three held-out physical arrays.
Evaluate epochs0,5,...30; extend to60 when any applicable best epoch is in the
last20%; keep late flags at60. Canary1 epoch, no extension. Fail nonfinite
candidates explicitly; select only completed finite candidates. 18 fits/study.

OM healthy and per-kind fully-failed target-conditioned endpoint tables use
41x512 independent literal identities, RESET plus128-pulse P0, held-out mean
error gate abs(meanfit-meanheldout)<=max(.02,4*SE) in every bin and table.
CDT samples their mixture; healthy residuals alone get optional strength
inflation. Strength>1 is an off-chip robustness intervention, not a changed
physical deployment law. Final and development scoring use the actual plant.

Recovery: five full45000-image epochs, no augmentation, batch64, 3520updates,
constant weightsLR1e-4; calibration3e-4 C10/1e-3 C100. Freeze digital prefix,
BN running stats and mapping scales. Digital calibration is gains, suffix BN
affine parameters and classifier bias. Original pretrained teacher fixed; it
saw all50000 original training images, so development is adaptation-held-out.
All10000 test images at P0 and each epoch, with no test selection. PCM full
Gaussian endpoint reprogramming per minibatch, no additional P&V. OM RESET
and at most128 P0 pulses, one pulse opportunity/update,640 total recovery
pulses/cell, tolerance .5nominal dw_min; existing pulse Adam with accumulated
target. Rewrite has the same opportunities/cap. No hidden-mask controller.
Report apparent/persistent OM metrics, write/verify/cap counts separately from
PCM full-write costs. Recovery gradients and optimizers remain digital.

Native EBL runs only; immutable contracts/inputs/sources and exclusive arm
ownership. Canaries exercise all source families, types, nominal/5%, both
datasets/devices/depths and all five epochs on128 images. Numerical tests cover
all rates and cross-depth identities. Validate replay and cross-GPU tolerance.
Deduplicate immutable checkpoint contents; retain P0 and each epoch's physical,
optimizer and RNG state. Benchmark1/2/4 workers per5090, choose best aggregate
throughput under85% memory; initially one worker per3090/3080. No unrelated
jobs disturbed. CIFAR-100 starts after complete CIFAR-10 artifact validation.
Monitor every30min;45min without semantic progress requires diagnosis. Preserve
failed attempts. Follow native scientific review before formal finalization.

Execution revision2: the initial full-data fits exposed a slow legacy digital
convolution path. A timing fixture measured0.7302 versus0.0480 seconds/batch
with deterministic FP32 cuDNN on a5090, with maximum teacher-logit difference
1.82e-5 inside the predeclared tolerance. V1 canaries and interrupted full fits
are preserved. V2 study/config directories are separate; no old plan is edited.
The compatible execution policy enables cuDNN on compute capability>=12 workers,
keeps TF32 disabled and deterministic algorithms, and retains ATen on the Ampere
hosts with their older library stacks. HWA/cache jobs use5090s; all four GPUs can
run cached recovery. No dataset, architecture, objective, optimizer, fault law,
selection rule or training budget changes. Record actual math policy per run.
