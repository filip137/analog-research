# Uncapped open-loop OM comparison

User request, 2026-09-19: repeat OM recovery with open-loop updates, and run
open-loop programming from scratch in parallel. The user clarified that
one pulse per cell per update stays; only the cumulative pulse cap is removed.
This separate experiment family is `cifar_om_open_loop.v1`; the completed
V2 sweep and its capped closed-loop results remain immutable references.
“From scratch” here means programming the pretrained/HWA target weights
onto a fresh RESET array. It does not change the pretrained digital prefix.

Two initialization arms run concurrently for each dataset/depth:

- `saved_p0`: load the exact physical state and device RNG at the V2
  deployment endpoint. Historical initialization used up to 128 pulses;
  no new programming is performed in this arm.
- `reset_open_loop`: retain the same device identities and fault mask,
  start at each physical cell's RESET/lower state, and program the same
  logical source with a pulse schedule derived from the public nominal
  soft-bounds response `q(n)=1-2*(1-dw_min)^n`. Use the closer integer pulse
  count after inverse mapping, with nominal reference zero. For the q=1
  asymptote, use the closest interior FP32 target. These counts have no
  imposed upper limit. Per-cell variations and actual write noise still
  apply. No deployment read or hidden identity changes the schedule.

Both arms then use digital Adam gradients of KL(original teacher || student)
to command open-loop pulses. For command delta, draw one pulse with probability
`min(abs(delta)/dw_min, 1)`, using command sign for direction. Preserve the
one-pulse-per-cell-per-update limit and log probability clipping. There is no
cumulative cap, target accumulator, or write-verification loop.
The physical device bounds and stuck states remain effective. Forward reads
are still required to compute the network output and digital gradients.

Keep CIFAR-10 and CIFAR-100 ResNet-32, four/eight suffix convolutions plus
classifier, and the V2 54 source/case combinations: digital, standard HWA,
noise HWA, kind-specific CDT; nominal plus 1/2/3/5% low/high/random failures.
Use the exact frozen V2 source checkpoints, caches, mapping scales, data split,
array identities 251001–251003, and endpoint seeds +10000. Do not refit HWA
or select a new checkpoint using these arrays. Keep its convergence flags.

Four controls per source/case are P0, calibration only, open-loop weights
only, and open-loop weights plus calibration. Historical V2 closed-loop and
rewrite controls are additional external comparisons for the saved-P0 arm;
they do not share the RESET-programmed P0 of the second arm. Five full
45,000-image epochs, batch 64, weight LR 1e-4, calibration LR 3e-4 on CIFAR-10
and 1e-3 on CIFAR-100, fixed epoch-five reporting, and 10,000-image tests at
P0/every epoch are unchanged. Calibration means digital gains, suffix BN
affines and classifier bias; prefix and running BN statistics remain frozen.

Eight full studies, three native array arms each, contain 5,184 controls
and 3,888 learning trajectories. Before full launches, eight native canaries
use the first 128 cohort images, array 251001, nominal/5% faults, all source
families and both initializations: 576 controls total. These are execution
checks and cannot select hyperparameters. Verify fixed faults, prefix hashes,
zero write-verification reads, actual pulse costs, exact physical checkpoint
replay at epoch five for every learning trajectory, and full native artifacts.
Run CIFAR-10 full studies first, then CIFAR-100; both initialization schemes
run alongside each other. A numerical failure is retained and diagnosed;
poor accuracy is evidence and does not trigger scientific changes.

Record source/input hashes and native EBL handles. Use the existing durable
campaign coordinator and 30-minute read-only watchdog with a separate ledger.
Check resources and actual metric progress after launch; a missing process
or 45 minutes without progress requires diagnosis. At most two RTX 5090s,
nom-cool-1, and the previously authorized other hosts are available; never
disturb unrelated jobs. Concurrency follows measured throughput and memory.

Analysis compares accuracy and temperature-one teacher KL, per-array paired
changes, and physical pulse/verification costs. Report the nominal open-loop
programming approximation and digital-gradient implementation. This remains
exploratory AIHWKit-preset evidence, not measured hardware or an energy claim.
Human scientific review remains necessary before manifest finalization.
