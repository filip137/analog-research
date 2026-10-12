---
id: "exp-011"
title: "Noise-referenced write thresholds with optional signed accumulation"
status: "ready"
hypotheses: []
---
# exp-011: thresholded physical writes

## Question and authorization

On October 12 the user proposed writing only updates larger than write noise,
then instructed: "all 5090s are free - we have to take advantage of this".
This assigns a bounded exploratory execution of thresholded same-array recovery.
The prior [dense LR result](../results/exp-010.md) found no improvement and
identified a large mismatch between intended updates and per-write noise.
Can thresholding reduce write-induced damage while retaining useful adaptation?

The user was asked whether to discard or accumulate subthreshold commands.
Both approaches are the disclosed default comparison while awaiting an optional
preference; any received preference takes priority before dependent dispatch.
No device-noise coefficient, HWA source, fault map or LR changes during a fit.

## Frozen comparison

Reuse exp-010's exact portable deployment P0 and all upstream lifecycle sections:
standard 3% Gaussian whole-model HWA source; final six OPT-125M MLPs (layers 6–11,
28,311,552 logical weights); permanent 3% random device faults on assignment
271001; pinned teacher, synthetic corpus and separated development/generated/
WikiText cohorts. Attention, digital base parameters and calibration stay frozen.
Each trajectory restarts from P0 with fresh Adam moments and, when used, zero
pending commands. Preserve batch size 32, objective teacher KL, seed 53 and the
existing data order. Deployment error and subsequent write noise remain enabled.

Four 8,192-sequence / 2,097,152-token / 256-update screens, all at constant LR
3e-3, the best exp-010 short-screen LR:

| Case | Nominal noise multiplier | Subthreshold commands |
| --- | ---: | --- |
| discard-s1 | 1 | Discard |
| accum-s1 | 1 | Accumulate signed commands |
| accum-s2 | 2 | Accumulate signed commands |
| accum-s3 | 3 | Accumulate signed commands |

Every native run also evaluates its matched unchanged-deployment control.
The completed dense 3e-3 screen is contextual evidence; it is not rerun.
The threshold is applied to **Adam commands**, not raw gradients. Its fixed
reference is the 2% P&V fit at nominal mid-range: sigma_q = 0.0141395421.
Thus the thresholds are 0.01414, 0.02828 and 0.04242 normalized q units.
Actual physical noise remains conductance-dependent; a nominal 2-sigma threshold
does not mean exactly two local standard deviations at every cell. The controller
uses no hidden fault identities, individual conductances or realized noise.

With accumulation, pending signed commands add across minibatches, opposite signs
cancel, and the full pending command is issued once its absolute magnitude
strictly exceeds the threshold. Subtract the requested command after every issued
attempt, including stuck attempts; do not observe success to alter the buffer.
Do not round small commands upward or flush pending residuals at the final step.
The existing unclamped physical write law, held-state forward, and fault
permanence checks remain unchanged. No verify reads or retries are introduced.

Record fixed-development KL every 64 updates in screens and 256 updates in a
full pass; retain fixed final checkpoints. Collect generated/WikiText endpoints
for reporting only. Report all physical writes, stuck attempts, touched cells,
maximum writes/cell, out-of-range states, and pending-command statistics.

## Decision and budget

After all four screens validate and collect, select minimum **final development
KL**, with exact ties resolved by fewer healthy writes then case label. Release
at most one fresh full 49,977,856-token pass only if that screen improves its
unchanged-deployment KL by at least 2%. The full run restarts from P0 using the
selected threshold/accumulation settings and LR 3e-3. No automatic LR retuning,
extra epoch, intermediate-checkpoint selection or recursive follow-up. A failed
mandatory gate or insufficient remaining budget stops downstream dispatch.

Targets: currently idle RTX5090s on loulou and trex, one worker per GPU. At the
initial inventory fifi and riri each had another user's roughly 30 GB worker;
they are not eligible. Do not alter those workloads. Use the single persistent
run-watch service and detached tmux workers. Initial reservation: 26,000 MiB plus
2,700 MiB headroom, checked against the live screen. Budget: **2 GPU-hours** and
**20 GB** additional storage across preparation, runs and collection. Execution
cutoff: **17:30 Paris on October 12**. Four screens should take roughly 10–20
minutes after staging, with a qualifying full pass adding about 30 minutes;
the first live measurement replaces this estimate.

Lifecycle IDs: `opt125m-cmo-l6-exp5-threshold-{case}-{dev,50m}` under the campaign's
`lifecycles/`. Native preparation/collection helper:
`campaigns/opt-cmo-corruption/launch/exp5_threshold.py`.
Evidence root: `results/lifecycles/opt125m-cmo-l6-exp5-threshold/`.

## Current execution and monitoring handoff

The full workflow suite passed 306 tests with one pre-existing optional skip.
After adding the conservative accumulator-storage estimate, focused physical,
schema, replay and selection-gate checks passed 70 tests with that same skip.
The four screen and four conditional full-pass lifecycles parse and round-trip.
The generic lifecycle retains the source's five assignment identities for
provenance; the recovery-only native manifest selects **only assignment 271001**
with explicit frozen inputs. The describe bound is 11.04 GB for all ten
generic trajectories, or 2.21 GB for this two-trajectory accumulated slice.
No launch is claimed until a native worker and semantic progress are observed.
The first bounded native screen also supplies the live canary; a full pass is
held until all screen artifact/replay gates, the development rule and measured
memory/time budget checks pass. The shared service owns dispatch, five-minute
semantic monitoring, read-only incident investigation and native collection.
No interactive coordinator is needed after the verified handoff. Preserve all
failed attempts and negative outcomes; scientific underperformance is not an
operational retry condition.
