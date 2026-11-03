---
id: "exp-002"
title: "Short-production programming and endpoint model"
status: "imported"
hypotheses: ["H-002"]
---

# exp-002 — Short-production programming and endpoint model

## Question and provenance

Short-production programming and endpoint model. At the primary half-step acceptance tolerance, an adaptive fixed-amplitude pulse-count controller reduces verify reads relative to one-pulse verify without materially increasing held-out endpoint error or failure probability, and a target-dependent Gaussian is adequate only when its predeclared coverage and Wasserstein gates pass.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-002-short-production.md).

Dependencies and related controls: [Original exhaustive controller design](../results/exp-001-exhaustive-design.md).

## Frozen plans

- [Short IBM ReRAM pulse-count program-and-verify endpoint model](../../../../../studies/ibm-reram-program-verify-noise-20260821-v2.json) — `ibm-reram-program-verify-noise-20260821-v2`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.

Historical document locations (now references/forwarding pages): [ibm_reram_program_verify_noise_model.md](../../../../../docs/ibm_reram_program_verify_noise_model.md), [ibm_reram_short_production_review.md](../../../../../docs/ibm_reram_short_production_review.md).

<a id="source-ibm_reram_short_production_review-ibm-reram-short-production-review"></a>

## Historical source: IBM ReRAM short-production review

- Review date: 2026-08-21
- Study: `ibm-reram-program-verify-noise-20260821-v2`
- Evidence tier: formal, model-based AIHWKit-preset characterization
- Review disposition: approved for local launch after the test and study-prepare gates
- Runtime objective: all four arms terminal within 24 hours of the first arm launch

<a id="source-ibm_reram_short_production_review-decision"></a>

## Decision

The exhaustive v1 design is superseded for the first endpoint-model milestone.
It crosses every one of 4,096 identities and eight repeats with 41 targets,
three tolerances, two starts, and two controllers: 16,121,856 trajectories per
arm and 64,487,424 total. That replication is not required to answer the
primary half-step-tolerance controller and endpoint-model question.

The v2 study retains the scientifically matched axes:

- optimized-material and HfO2 IBM presets;
- corruption disabled and the published corruption probability enabled;
- all 41 continuous targets;
- lower-to-target/SET and upper-to-target/RESET programming;
- one-pulse and adaptive controllers with matched programming streams;
- device-identity calibration, fit, and held-out validation partitions; and
- empirical, Gaussian, failure/cost, and Wan-2022 comparisons.

It fixes the intended deployment acceptance width at `tau/step=0.5`, uses
1,024 identities and four independent cycle repeats, and blocks boundary
conditioning once per device/repeat/start across targets. The immutable total
is 671,744 trajectories per arm and 2,686,976 total, a 24x reduction.

<a id="source-ibm_reram_short_production_review-statistical-review"></a>

## Statistical review

Each condition/target contains 4,096 trajectories. Whole-device partitioning
provides 616 fit identities (2,464 trajectories) and 204 untouched validation
identities (816 trajectories). The fitted mean and log-standard-deviation
curves retain 41 target locations for a fourth-order model. The corrupt arms
are expected to contain roughly 138 OM and 100 HfO2 corrupt identities under
the published preset probabilities, with realized counts recorded rather than
assumed.

This is sufficient for the predeclared central coverage and normalized
Wasserstein gates, but it is not a high-confidence per-target extreme-tail
study. Four repeats sample cycle-to-cycle variation without pretending those
repeats are independent device identities. The resulting model is conditional
on the half-step tolerance and must not be used to claim tolerance
sensitivity. A later study must be declared if a different deployment
tolerance is selected.

<a id="source-ibm_reram_short_production_review-conditioning-review"></a>

## Conditioning review

The original eight-consecutive-quiet-pulse boundary rule dominated runtime.
In the 1,280-trajectory operational comparison it took a median 1,797.5
pulses, reached the 4,096-pulse cap at p95, and failed for 23.125% of cases.
Four quiet pulses at the unchanged `1e-6` span threshold took a median 138
pulses and p95 627.2, succeeded in every case, and left successful endpoints
at the sampled bound. Changing the threshold by 10x did not repair the
eight-pulse rule. V2 therefore changes the consecutive-quiet requirement,
not the boundary threshold.

The per-start blocked state is a simulated matched-design intervention. Each
repeat independently samples conditioning, then its immutable state is cloned
across targets and controllers. This removes target-dependent initialization
as a confound and reduces boundary work by 41x. The persisted-ledger integrity
gate proves that conditioning seed, success, pulse count, and apparent and
persistent starting states are identical across targets for every matched
device/repeat/start.

<a id="source-ibm_reram_short_production_review-cuda-and-provenance-review"></a>

## CUDA and provenance review

The local host exposes one idle NVIDIA GeForce RTX 3090 with 24 GB memory.
PyTorch 2.5.1+cu121 in the `py312` environment can execute CUDA, while the
pinned AIHWKit 1.1.0 installation is in a separate CPU environment whose
CUDA-13 PyTorch build is incompatible with the host driver. Production
therefore uses a deliberate two-environment boundary:

1. `/home/filip/miniconda3/envs/aihwkit/bin/python` constructs the exact IBM
   device population and Wan reference and writes versioned, hashed receipts.
2. `/home/filip/miniconda3/envs/py312/bin/python` loads that population and
   evaluates the explicit pulse equation and controllers on CUDA.

Every trajectory retains an identity-derived programming seed. CUDA execution
generates each complete normal stream with an independently seeded CPU
generator and stages the immutable table on the GPU. This removes scalar RNG
dispatch from the pulse loop without sharing streams or making results depend
on traversal order. The backend is exactly replayable on the recorded stack,
but its vector-draw streams are not claimed byte-identical to the legacy
scalar-CPU v1 streams.

Review gates passed:

- buffered seeded rows reproduce their individually generated reference rows;
- buffered population mean and standard deviation pass fixed standard-normal
  checks;
- noiseless CPU and CUDA pulse equations are exactly equal for the focused
  transition sequence;
- CUDA state save/restore reproduces exact pulse continuation;
- two independent end-to-end CUDA smokes produced byte-identical population
  and trajectory-database SHA-256 digests; and
- the successful CUDA smoke produced every report artifact and passed all
  persisted-ledger checks.

<a id="source-ibm_reram_short_production_review-runtime-and-storage-review"></a>

## Runtime and storage review

The exact-width OM-continuous sizing gate used 1,024 identities, four repeats,
five targets, both starts, both controllers, the 512-pulse programming cap,
the 4,096-pulse conditioning cap, blocked per-start initialization, SQLite
event persistence, full integrity validation, endpoint fits, Wan comparison,
plots, and report generation.

It completed in 61.57 seconds with:

- 81,920 trajectories;
- 2,863,651 verify events;
- zero initialization failures;
- a passing integrity report;
- 1.07 GB maximum host resident memory; and
- a 236 MB trajectory database.

Linear target scaling gives 504.9 seconds (8.4 minutes) and approximately
1.94 GB per arm, or about 7.7 GB for four arms. The host has 779 GB free under
`/home`. A deliberately conservative 10x time factor is 84.2 minutes for a
parallel campaign or 5.61 hours if the four arms were forced to run
sequentially. Both satisfy the 24-hour gate with substantial margin.

<a id="source-ibm_reram_short_production_review-launch-and-monitoring-contract"></a>

## Launch and monitoring contract

The formal launcher is
`python -m experiments.reram_program_verify.local_short_launcher`. It starts
the four independent direct `python -m ebl characterize` commands on
`CUDA_VISIBLE_DEVICES=0` and does not replace the repository study lifecycle.
It fails closed unless the worktree is clean and records the exact source
commit in its launch contract before creating any native run.
Child-side live-ledger writes are deferred until all four native manifests
have captured the same frozen source identity; the launcher then refreshes the
repository-native `current_simulations.md` view on every heartbeat and at
termination.

- Study root: `results/ibm-reram-program-verify-noise-20260821-v2/`
- Native results: `runs/<arm-id>/<native-run-id>/`
- Persistent tmux handle: `reram_pv_v2`
- Launcher attempts: `launch/<UTC-attempt-id>/`
- Per-arm logs and PIDs: `<attempt>/<arm-id>.log` and
  `<attempt>/<arm-id>.launch.json`
- Heartbeat: `<attempt>/heartbeat.json`, refreshed every 15 seconds
- Terminal launcher receipt: `<attempt>/launcher_result.json`
- Study coverage: `analysis/summary.json` and `analysis/report.md`

Immediately after launch, every arm must have exactly one reviewed process,
a native `manifest.json`/`status.json`, and growing database or metric bytes.
The active tmux handle, four child PIDs, RTX 3090 memory/process occupancy,
native status, log tails, and heartbeat deltas are checked twice before the
launch is called healthy. Thereafter, artifact progress is checked at least
every 30 minutes. A missing process, nonzero exit, absent native artifact,
stale 45-minute heartbeat, or zero GPU worker occupancy while nominally
running is an operational failure.

The numerical runner does not support in-place resume. A failed operational
attempt is retained and may be retried only as a new native run under the same
arm after the cause is understood and the unchanged config/source guards pass.
No tolerance, seed, controller, pulse budget, or population setting may change
during recovery.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
