---
id: "exp-001"
title: "Time-dependent drift degradation"
status: "complete"
hypotheses: ["H-001"]
---
# exp-001 — Time-dependent drift degradation

## Current assignment

The active assignment is the time-survey subphase at the end of this note and in the linked shared specification. It measures uncorrected degradation; the original calibration design below remains deferred.

## Question and dependencies

At fixed mean drift, age and exponents, does increasing s leave a quality
gap after accessible calibration? This tests the prerequisite of
[H-001](../../../hypotheses/H-001-dispersion-recovery.md), not the benefit of
learning. Use the [shared model](../../../../../../on-chip-learning/campaigns/drift/power-law-model.md)
and [campaign defaults](../../../../../../on-chip-learning/campaigns/drift/README.md).

Confirmed scope: OPT-125M, six analog MLPs, frozen std3 HWA source,
conductance-independent m and s, alpha=0.23 and beta=0.11 nominal, t_ref=1 h.
Pin the architecture's clean post-HWA checkpoint and fixed data/cache manifests.

## Cases and controls

Proposed first design: one development chip, no faults, separate read noise
disabled, one-hour sigma sweep. Confirm scalar m, sigma values, exact bounds and
mapping, chip count and this fixed-device setting before numerical execution.
Include s=0, low sigma and several larger sigma levels. One hour isolates the
reference-time coefficients; it does not test exponent effects.

For each aged snapshot compare deployed reference, global recalibration and
accessible-channel recalibration. Keep a separate unchanged programmed state
and the clean post-HWA reference. Reuse each chip's Gaussian draw across sigma
levels, with saved identities and no redraw per forward. Calibration uses the
declared pool and physical probes; evaluation text stays separate.

Retain the declared DRN rail mapping, physical device identities, solver and amplification settings. Conductance drift changes reciprocal loading as well as effective signed weights. Report the physical device count and avoid assuming that a crossbar gain correction is valid at an inaccessible DRN node.

## Execution contract

Follow the [native handoff](../../../configs/README.md).
Implement and check baseline identity, analytic power-law scaling, clipping,
random-state replay, and immutable faults where supported. Checks use small
CPU tensors before a bounded model readiness probe; they are not evidence of
network recovery. Record solver residuals, voltage diagnostics, effective conductance changes and unchanged conductances under calibration.

The first-read programmed state must satisfy the selected window. Keep initial
programming error separate from drift, and record the cost of programming and
calibration. No learning or refresh runs are included in this experiment.

## Metrics and decision

Primary metric: KL to the architecture's clean post-HWA teacher, evaluated
from pinned targets. Report fixed-cohort perplexity and absolute clean,
programmed, aged and calibrated quality. Measure raw/realized drift mean,
standard deviation, boundary fractions, effective weight error, calibration
reads, runtime and peak memory.

Choose representative low/high sigma and calibration methods using development
and validation chips only. Large post-calibration loss motivates exp-002 but
does not support a claim that analog learning is needed. If all tested sigma
levels are benign, report that regime instead of forcing a positive conclusion.
Select any revised grid before held-out evaluation.

## Budget, stop and expected outputs

The numerical grid, exact input hashes, readiness subset, compute/storage cap
and execution target remain open. One development chip is a proposal, not a
final replication claim. This inference/calibration pilot performs no physical
recovery writes. Stop individual cases on invalid state, nonfinite metrics or
invalid solves; preserve their records.

Expected raw root: `results/opt-drift-recovery/exp-001/`.
Expected review note: `series/001-dispersion/results/exp-001.md`, created after
evidence is collected. Expected outputs: state/cache identities, native bundles,
paired sigma/calibration tables, clipping diagnostics and throughput estimates.

## Execution and monitoring handoff

Not launched. Before running, resolve the open scientific fields and resource
caps, create exact native configs, preserve the shared specification and code
snapshot, and record commands, host, output paths and monitoring owner here.

## Assigned time-survey subphase — 28 October 2026

The user now requests degradation versus age before calibration. The [shared frozen time-survey specification](../../../../../../on-chip-learning/campaigns/drift/time-survey.md) supersedes the open one-hour grid and numeric/budget fields above for this subphase. Calibration remains deferred. Select settings with <=10% perplexity increase at 24 hours and >=2x at 30 days or one year, if such settings exist. First survey: m=-2 uS, s=0/0.5/1/2/3/6/12 uS, alpha=.23, beta=.11, 43 conditions including first read, 64 development sequences, one nominal chip. Long horizons are model extrapolations.

Native scalar-law support, bounded first-read admission, held saved epsilon, cached clean-HWA reference targets, and strict settings were added. Local numerical/native integration checks passed; the crossbar workflow suite passed 605 tests (four optional skips), and DRN focused/regression checks passed 59 tests. Live canary admission is still required.

Current Jean Zay admission: version-3 canaries crossbar `1660351`, DRN `1660353`; [execution handoff](../../../../../../on-chip-learning/campaigns/drift/execution/handoff.md). Previous version-2 crossbar attempt stopped before evaluation because Git was absent on the compute node; the explicit Git module fixes that environment dependency. All attempts remain retained.

Both version-3 canaries passed. The full surveys were released on 28 October at 13:15 UTC: crossbar Slurm 1660525, DRN Slurm 1660526. Each covers 43 states on the matched 64-sequence development subset. Monitoring and native collection details are in the linked shared handoff.

## Completed assigned subphase

The time survey completed and was reviewed on 28 October 2026: 43/43 conditions per architecture, no qualifying setting. See the [result](../results/exp-001.md). The complete status applies to the assigned time survey only; the earlier proposed calibration design remains deferred and the wider recovery campaign remains open.
