# OM closed-loop learning-rate investigation

The user approved learning rates 1e-4, 3e-4 and 1e-3 to test whether the
historical closed-loop Adam command is too small relative to the unchanged
OM verify tolerance, 0.04745. This exploratory controller-tuning family is
`cifar_om_closed_loop_lr.v1`. The user subsequently specified loulou as the
sole GPU host. Preserve unrelated jobs and every earlier frozen experiment.

Use both CIFAR datasets, pretrained ResNet-32 with four/eight suffix
convolutions plus classifier mapped to OM, frozen digital and standard-HWA
source checkpoints, and nominal/3%/5% stuck-high cases. No HWA is refitted.
The frozen prefix, mapping scales, permanent-fault identities and digital
teacher remain unchanged. This focused grid diagnoses the learning-rate
explanation; corruption-aware HWA and other fault kinds are outside its scope.

For development, use the existing independently sampled identities 211001
and 211002, endpoint seeds +10000, and legacy deployment programming with
tolerance 0.04745 and up to 128 pulses. Save one exact physical P0 per
source/case/array, and restore it including device RNGs for every candidate
and calibration-only control. The initial programming budget is historical;
the recovery controller has no cumulative pulse cap. It retains the same
accumulated digital target, Adam moments, observable verify read, fixed
tolerance and at most one physical pulse per cell per minibatch update.

Recovery always lasts five full 45,000-image epochs, batch size 64, original
teacher KL at temperature one. Keep calibration Adam LR at 3e-4 for CIFAR-10
and 1e-3 for CIFAR-100. All three candidate weight LRs include the same
calibration parameters. Calibration-only is run once per development P0.
This is the joint-recovery comparison underlying the reported open-loop
advantage. It does not claim weight-only tuning or physical-gradient training.

Selection uses only the frozen 1000-image development cohort and both
development arrays. The test cohort is removed before development execution.
For each dataset/depth, select a single shared weight LR by the smallest
mean epoch-five development KL across two sources, three cases and two
arrays (12 equally weighted endpoints); break exact ties toward smaller LR.
This is a mean of raw KL values, so severe failures may dominate selection.
Keep all individual LR curves and clean-array regressions in the report.
Do not select checkpoints, rates or epochs on test labels or final arrays.

Freeze a selection receipt with source and development-result hashes before
confirmation. Confirmation uses identities 251001–251003 and the exact
historical saved deployment state/RNG, with selected LR and uncapped 1e-4
baseline sharing that P0. If selected LR is 1e-4, run the identical condition
once. Historical calibration and open-loop joint-recovery controls are valid
external references only after exact inherited P0 hashes agree. Do not use
the fresh-RESET open-loop initialization as a matched comparator.

Four 128-image native smoke studies, using separate array 271001, verify all
datasets/depths, source/case/rate branches and final-state replay. They cannot
select scientific settings. Full development has eight native array runs,
192 trajectories, 960 full epochs. Confirmation has twelve native array
runs and 72–144 trajectories (360–720 full epochs). Native execution enters
only through `python -m ebl`; every trained epoch has a lossless physical,
optimizer and controller checkpoint, and epoch five is replayed exactly.

Run canaries conservatively on loulou, measure one/two/four-worker timing and
memory under its current unrelated workload, and freeze the concurrency
choice before full execution. Process ownership, GPU occupancy, recent
metrics, logs, terminal coverage and checkpoint collection are checked by the
existing durable coordinator plus a passive 30-minute watchdog. A missing
process, nonzero exit or 45 minutes without progress requires diagnosis.
Numerical failures retain their native artifacts; retries require a new
source receipt when source changes. No silent scientific changes are allowed.

Report accuracy, teacher KL, pulse coverage, mean and maximum cell pulses,
verify reads, accumulated-target displacement and cells beyond the old
640-pulse cap, alongside paired source/case/array outcomes. Findings concern
this observable controller and modeled AIHWKit OM preset. They do not establish
inherent superiority of open-loop or closed-loop training. Scientific review
and manifest finalization remain pending after artifact-verified collection.
