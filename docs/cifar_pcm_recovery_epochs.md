# CIFAR recovery over 20 passes

The user approved extending recovery on CIFAR-10 and CIFAR-100, starting from
digital or HWA deployment. This is a new exploratory study; the completed
one-pass study and its frozen results remain unchanged.

## Fixed contract

- Full pretrained CIFAR ResNet-32 and the same four suffix convolutions plus
  classifier on nine logical 512x512 tiles. Freeze the prefix and BN running
  statistics. Reuse the exact cached 5000-image recovery, 1000-image development
  and 10000-image official test cohorts and original digital teacher.
- Reuse all six selected source checkpoints: digital, standard HWA,
  noise-selected HWA and open/Gmax/random CDT. The first three cover nominal
  hardware and 1% of each physical failure type. Each CDT covers nominal and
  1% of its own type: 18 source/case combinations per physical-array seed.
- Keep Gaussian PCM endpoints at nominal noise scale, 25 microSiemens, fixed
  per-bank failure identities and stuck conductances. No additional P&V loop,
  read noise, drift or peripheral quantization. Adam/backprop are digital;
  each learning minibatch requests a complete Gaussian endpoint rewrite.
  This remains an endpoint-model adaptation experiment, not an incremental
  pulse-law simulation.
- Continuous Adam state across 20 passes over the same 5000 images, batch 64:
  1580 updates and at most 1580 rewrites per trajectory. Reshuffle each pass
  with data_seed+epoch; pass one matches the old adapter. No new augmentation.
  Report all milestones 1,2,5,10,20, including negative results.
- Retain development-selected base rates from the prior recovery bundle:
  weight 1e-4; calibration 3e-4 on CIFAR-10 and 1e-3 on CIFAR-100.
- Compare constant rates against multiplication by 0.3 after passes 5 and 10.
  Select schedules separately for calibration, weight-only and joint recovery
  using mean final/initial development KL at pass 20. Tasks: standard HWA on
  nominal arrays, and each CDT on its own 1% fault. Two independent arrays per
  task (121001/121002, endpoints 131001/131002), two schedules and three methods
  give 48 trajectories per dataset. Select a secondary reporting epoch among
  the five milestones on these same development trajectories. Final arrays
  and test data never choose schedules or reporting epochs.
- Final arrays: 101001–101003, endpoints 111001–111003, separate from all prior
  final arrays and development identities. For each source/case restore exact
  P0/model/RNG for none, calibration, rewrite+calibration, weight-only and joint
  recovery. Rewrite uses calibration's selected schedule and unchanged targets.
  All adapting methods receive 100000 image presentations; writing methods
  make 1580 requests. Do not infer pulses, energy or latency from request counts.
- Evaluate held-out development KL and accuracy every pass. Evaluate all 10000
  test images at the five fixed milestones, never for stopping or selection.
  Save P0 and compact milestone states (all changing conductances, target
  weights, affine parameters, array RNG and Adam state) for replay.

## Coverage and interpretation

Per dataset: one development stage and three final screens, 270 source/case/
control records, of which 216 contain 20-pass learning curves. Across both:
96 development trajectories, 540 final control records, 432 final learning
trajectories and 2160 post-adaptation test endpoints. The no-update baseline is
evaluated once per source/case/array, not counted as repeated training.

Primary metric is mean KL(p_teacher || p_student), nats at temperature 1;
accuracy is secondary. Compare equal-budget controls and one-pass versus later
milestones on the same arrays. Show mean and sample SD across three arrays,
paired differences, clean source fidelity and gains/losses. No significance
claim or universal necessity claim is planned.

This extends recovery only. Reused HWA/CDT fits retain their earlier late-
improvement flags and one-training-seed limitation. Performance beyond those
sources is conditional on their fixed HWA budgets. Full-data HWA convergence
and a physical incremental training algorithm are separate follow-ups.

## Execution and validation

Native experiment: cifar_pcm_recovery_epochs.v1, using the existing EBL campaign
executor. Input paths/digests come from completed v3 campaign receipts, never
from newest-file discovery. Plans/configs and staged source are frozen before
launch. Original artifacts are read-only inputs.

Check one-pass adapter parity, persistent optimizer state, fault permanence,
zero-write calibration, exact rewrite budgets and source/test selection
isolation. Run a native two-pass GPU canary before the full studies. Monitor
native metrics, processes and receipts through completion. Collect and verify
coverage and replay selected saved endpoints before reporting numerical results.
