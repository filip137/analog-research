# CIFAR fault sweep execution

Protocol: [cifar_fault_rate_sweep.md](cifar_fault_rate_sweep.md).

Current status (2026-09-19): CIFAR-10 and CIFAR-100 numerical runs and
artifact verification are complete; all eight V2 studies are
`ready_for_review`. See the [verified measurements](cifar_fault_rate_sweep_results.md).
Scientific interpretation and manifest finalization await human review.
The dated checkpoints below record the execution history.

Implemented in `experiments/cifar_crossbar/sweep_{config,devices,runtime,campaign,report}.py`,
registered as native experiment `cifar_crossbar_fault_sweep.v1`.
Versioned run plans are in `examples/cifar_crossbar/sweep_v2_*`.
Tests: 79 passed across the sweep, reporting and existing CIFAR suites;
seven additional report/numerical-validation regressions subsequently passed.

Execution V1 completed all eight small canaries: 180 native stages, 720 control
outcomes, 864 exact saved-state evaluations, and cross-GPU logit agreement.
Those checks validate the pipeline; they are not the full-data comparison.
The first full-data attempt reproduced 93.53% CIFAR-10 digital accuracy for
both suffix depths and validated both OM kernels. Its unfinished HWA fits were
interrupted to enable the measured faster deterministic FP32 convolution path.
No complete full-data HWA/recovery result came from that V1 attempt.

V2 retains the same scientific settings, uses separate immutable study plans,
and validates the updated execution path. Its source identity is
`1b4d137195d451882d3e6f5b84ede9434ed37aff79dc639d55d8bf5ed6dde992`.
All historical records remain intact. Numerical stages use `python -m ebl train`.

The local coordinator and read-only watchdog live in
`artifacts/cifar_fault_sweep_sources/`. The detailed handoff is its `README.md`.
Live status is `full-progress.json`; per-node ownership, completion and source
paths are in `execution-v2/campaign-execution.json`. Native remote handles,
logs and terminal receipts are under each staged source root's `launch/`.
Watchdog resource/artifact snapshots are recorded every 30 minutes in
`execution-v2/watchdog-latest.json` and `watchdog-history.jsonl`.

The gated runner completes and verifies CIFAR-10 before CIFAR-100. Expected
coverage is 3,240 controls per dataset, across four/eight analog convolutions,
PCM/OM, three physical arrays and the declared source/fault/control matrix.
Collected runs will be in `results/cifar-sweep-collected/v2/`. Per-dataset
reports will be in `results/cifar-sweep-{cifar10,cifar100}-analysis/`; combined
tables/figures will be in `results/cifar-sweep-analysis/`.

The execution status at the time of the plan was active validation. The
current status is recorded above; human interpretation and native study
finalization remain pending.

## Interim HWA development check, 2026-09-18 15:35 UTC

This is development-array evidence during the active CIFAR-10 run, before any
full-data fresh-array recovery screen. It must not be treated as a deployment
or on-chip result. All 36 generic HWA candidates and 18 PCM fault-specific
candidates finished with finite results. Selection chose noise strength 5 and
learning rate 3e-4 for the generic noise HWA source in both depths/devices.
The selected generic sources reached their best development scores at epochs
55/60 (PCM four convolutions) or 60 (the other three conditions), so their
convergence remains open under the frozen 60-epoch cap.

For OM four convolutions, the best mixed-fault generic candidate reduced mean
teacher KL from 22.30 at initialization to 0.425 at epoch 60. For OM eight
convolutions it reduced 17053 to 50.65 at epoch 60. These large initial OM
values are from the declared mixed 2/3/5% stress cases; they are not test
accuracy. For fault-specific PCM HWA, best development KL was 0.0117 for stuck
high and 0.0049 for random stuck with four convolutions, versus 0.153 and
0.0417 with eight. The 1% cases are excluded from source selection and await
the final-array screen.

The live `execution-v2/campaign-execution.json` and native candidate artifacts
are the source of these numbers. Results may change in interpretation after
fresh-array deployment, on-chip recovery, and full report checks. A recovery
advantage against an HWA source still improving at epoch 60 needs that limit
reported explicitly.

## Provisional physical-array observation, 2026-09-18 16:28 UTC

One active CIFAR-10 PCM four-convolution screen (array 251002, **digital**
source) has completed five full recovery epochs for stuck-low and stuck-high
cases. Its 5% stuck-low case had 93.09% / 0.0294 KL at P0, 93.47% / 0.0051
after calibration, and 93.34% / 0.0103 after joint recovery. At 5% stuck-high,
P0 was 11.28% / 5.913 KL; calibration reached 90.91% / 0.1358, and weights-only
on-chip recovery reached 91.85% / 0.0990. Weights-only also beat calibration
on this array at 1%, 2%, and 3% stuck-high. These are full 10000-image tests
on one array, read from completed method rows during an active native run. They
remain provisional until all three arrays, standard HWA, noise HWA, CDT,
artifact checks and exact replay are complete. The physical write costs of
weights-only recovery exceed calibration and must be compared in the final
report.

At 16:36 UTC, the same array completed the corresponding **standard-HWA**
stuck-high cases. At 5%, deployment was 11.59% / 5.8763 KL, calibration after
five epochs 90.94% / 0.1348, and weights-only on-chip recovery 91.82% /
0.1009. For 1/2/3% stuck-high, the HWA-deployed accuracy was
91.19/80.07/58.05%; five-epoch calibration reached 93.05/92.58/92.26%,
and five-epoch weights-only recovery reached 93.25/92.97/92.69%. These are
matched on array 251002 with the same physical fault identities used above.
The held-out arrays and fault-specific HWA comparisons remain outstanding;
none of these one-array test observations selected a model or setting.

At 16:49 UTC the same array also completed generic noise-aware HWA for all
stuck-high rates. At 5%, its P0 was 52.27% / 1.6656 KL, calibration reached
92.21% / 0.0975, and joint on-chip recovery reached 92.38% / 0.0824. Thus,
on this array, noise-aware HWA plus calibration already exceeds recovery from
the digital or standard-HWA starts; the incremental joint-recovery accuracy
gain after noise-aware HWA is 0.17 percentage points. The corresponding
2% and 3% noise-HWA calibration accuracies were 93.05% and 92.85%, versus
92.97% and 92.73% for digital-start weights-only recovery. This is a
one-array interim contrast, not the predeclared three-array conclusion.

By 16:55 UTC all three PCM four-convolution digital-source arrays had
completed the stuck-low and stuck-high five-epoch controls. For 5% stuck-high,
the three-array mean ± sample SD test accuracy was 47.80 ± 31.63% at P0,
90.957 ± 0.050% after calibration, and 92.137 ± 0.248% after weights-only
recovery. Corresponding teacher KL was 2.7018 ± 2.7817, 0.14031 ± 0.00402,
and 0.09051 ± 0.00738. The paired weight-recovery gain over calibration was
1.180 percentage points and 0.04980 lower KL, positive on all three arrays.
At 1/2/3% stuck-high, paired accuracy gains were 0.180/0.443/0.437 points
and KL improvements 0.00914/0.02584/0.03300, again positive on every array.
For 5% stuck-low, calibration was 93.380% / 0.00575 KL versus 93.347% /
0.00753 after weights-only recovery; calibration had lower KL on all three
arrays at every tested stuck-low rate. These completed method rows are still
part of active native screens, awaiting artifact verification and exact replay.

At 17:05 UTC, the first full native screen (PCM, four convolutions, array
251002) finished and the controller collected its 270 complete controls with
a result hash. The RTX 5090 slot immediately advanced to PCM with eight
convolutions. The other three screens were still active; exact checkpoint
replay and the three-array HWA/OM comparison remain pending.

On that first completed array, fault-specific stuck-high HWA at 5% failed
devices deployed at 90.26% / 0.2644 KL, then reached 92.71% / 0.0671 with
calibration and 92.86% / 0.0603 with joint on-chip recovery. Its incremental
joint-recovery accuracy gain was only 0.15 points, versus 0.17 points for
generic noise-aware HWA and 0.75 points for standard HWA on this same array
(joint recovery compared with calibration in each source). Fault-specific
HWA plus calibration here already beat digital-start recovery, but a
one-array test comparison cannot establish the broader conclusion. The
fault-specific PCM four-convolution HWA fit was selected at epoch 55 of 60,
so its convergence remains uncertain under this cap.

## Three-array CIFAR-10 interim checkpoint, 2026-09-18 20:12 UTC

Nine of twelve full CIFAR-10 screens are collected and pass the declared
per-screen validation: all 2,430 controls for PCM four/eight convolutions and
OM four convolutions have five full recovery epochs and 10,000-image test
endpoints. All three OM eight-convolution screens are active without watchdog
alerts. Exact saved-state replay and the full report wait for those three.
CIFAR-100 full screens have not started.

For 5% stuck-high across three independent arrays, fault-specific HWA plus
calibration versus its own on-chip controls gave these test means (accuracy /
teacher KL): PCM four convolutions 92.69% / 0.06922, weights-only 92.83% /
0.06412, joint 92.78% / 0.06005; PCM eight convolutions 90.42% / 0.1878,
weights-only 90.71% / 0.1736, joint 90.82% / 0.1812; OM four convolutions
92.64% / 0.07634, weights-only 92.53% / 0.1091, joint 92.66% / 0.07512.
The PCM eight-convolution joint accuracy gain of 0.40 points over calibration
was positive on all three arrays, with lower KL on all three. The OM
four-convolution weight-only control degraded teacher KL on all three arrays.
These are predeclared endpoint comparisons, but interpretation remains
provisional pending the last depth/device condition and replay.

## Coordinator recovery, 2026-09-18 20:29 UTC

The local full runner, controller, and watchdog had stopped without a recorded
scientific failure while three OM eight-convolution native screens continued
on their remote hosts. The exclusive lock was free. The local controller now
adopts ledger-claimed native jobs by their existing remote handle or exit
receipt, then collects them without relaunch. The full runner restart gate
permits non-canary active claims while retaining the complete canary gate.
Both scripts passed Python compilation; all three claims matched frozen
campaign nodes. A detached host-side restart launched runner PID 2044409 and
watchdog PID 2044410; controller PID 2044448 logged ADOPT for the three
original run IDs. Remote processes remained alive with fresh metrics and no
watchdog alerts. CIFAR-100 remains sequenced after CIFAR-10 native completion,
checkpoint replay, and the 3,240-control report gate.

At 17:51 UTC both PCM four-convolution screens 1 and 2 had passed native
collection (270 controls each). On these two physical arrays at 5% stuck-high,
standard HWA plus calibration averaged 90.985% / 0.13810 KL, versus 92.06% /
0.09417 after weights-only recovery. Noise-aware HWA plus calibration averaged
92.185% / 0.09735, versus 92.37% / 0.08041 after weights-only recovery.
Fault-specific stuck-high HWA plus calibration averaged 92.70% / 0.06859,
versus 92.825% / 0.06430 after weights-only recovery and 92.805% / 0.05989
after joint recovery. The fault-specific incremental accuracy gains over its
own calibration were +0.16/+0.09 points for weights-only and +0.06/+0.15
points for joint, positive on both arrays. These remain interim two-array
comparisons; the third physical-array screen and exact checkpoint replay are
pending.


The V2 GPU check exposed one ill-conditioned OM logit: -10.88591 versus
-10.88279 among logits reaching 236079. The original elementwise-tolerance
failure is preserved. A separate FP64 calculation gave relative L2 errors
below1e-7 for both implementations, with no changed predictions. The diagnostic
now records the original exceptions and additionally checks a32*FP32-epsilon
per-example scale bound, relative L2<=1e-6, identical predictions and teacher KL
agreement (absolute2e-5 plus relative1e-6). This is an analysis-only validation
revision, covered by five regression tests; it changes no numerical run or
scientific setting. Its hashes and rationale are recorded in
`execution-v2/numerical-validation-revision.json` under the artifact directory.


Full-data launch checkpoint (2026-09-18): all eight V2 canaries are now
artifact-verified (180 native stages,720 control outcomes), and all864 exact
checkpoint evaluations passed. The capacity gate selected one worker per host
under the current shared GPU occupancy. Full CIFAR-10 is running: all four
caches reproduce93.53% digital test accuracy with identical predictions at both
suffix depths, both OM kernels passed, and the first HWA fits have produced
training metrics through multiple epochs. This digital check is not an analog
recovery result. At that checkpoint, the complete analog/HWA/recovery comparison
was still underway; CIFAR-100 was scheduled after verified CIFAR-10 collection
and reporting.
The launch checkpoint is `execution-v2/implementation-handoff.json`; the live
coordinator/watchdog files supersede it as execution advances.
