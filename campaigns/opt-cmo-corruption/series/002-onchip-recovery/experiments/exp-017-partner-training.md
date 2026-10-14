---
id: "exp-017"
title: "On-chip learning through the verified partner after fault-aware rewriting"
status: "ready"
hypotheses: []
---
# exp-017: train through the verified partner

The user assigned implementation and execution on October 14, 2026, selecting
**std3 only** and the unchanged training recipe. This is a new comparison after
[exp-016](../results/exp-016.md): that experiment's learning arm updated only
original active devices, including when a healthy reference had absorbed a
stuck active. No hyperparameter search or extension to other sources is assigned.

## Frozen comparison and decision

- OPT-125M's last six MLPs, layers 6–11: 28,311,552 logical weights and
  56,623,104 devices. Same pinned teacher and frozen exp-007 std3 source.
- Exact exp-013 held P0s, assignments 271002–271004; select `nominal` and
  `random_30000ppm` without changing upstream defect declarations/provenance.
- Four arms: original hold; verify pair rewrite then hold; the same rewrite
  then original-active training; the same rewrite then verified-partner training.
  Three native jobs, six starting conditions, 24 endpoints, 12 learning fits.
- Rewrites match exp-016 exactly: 5-sigma detection, seed 241016, one apparent
  per-device scan, fitted 2% CMO endpoint programming. Targets and subtraction
  polarity stay unchanged. No calibration or adapter.
- New routing uses that original scan only: unflagged active -> active;
  flagged active/unflagged reference -> reference; both flagged -> blocked.
  The map stays fixed. Reference writes use the opposite delta-q sign. Hidden
  faults remain analysis-only; missed faults consume attempts and never heal.
- Same exp-013/016 recipe: thresholded Adam, LR .003, accumulated nominal
  2-sigma threshold, physical write noise, four issued attempts per selected
  physical device after rewriting, 256 updates, batch 32, one epoch over 8,192
  sequences / 2,097,152 input tokens. Training seed 53; data seed 20261008.
- Same separate development/generated-test/WikiText-test cohorts: 256/384/1,024
  sequences at length 256. All evaluation uses held apparent state. Final
  checkpoint fixed before running; development every 64 steps is diagnostic.
- Primary teacher KL; report perplexity. Separately for each case/domain/control,
  verified-partner learning improves consistently only if final KL is lower on
  all three assignments, harms if higher on all three, otherwise mixed.
  Compare with rewrite hold and original-active learning. No significance claim
  or general physical-viability claim follows from a sign test.

Expected possibility: opening the stuck-active weights to reference updates
improves the remaining error. Alternative: write noise or the unchanged recipe
dominates, and reference updates add cost without a held-out gain. Either outcome
is retained; scientific quality never triggers a retry.

## Implementation and validation

The optional arm field `write_roles: "verified_partner"` extends the native
thresholded CMO writer. The controller retains an int8 routing map from the
rewrite, consuming 28,311,552 bytes. No additional verify reads are introduced.
Counters distinguish active/reference learning writes, issued/stuck attempts
and touched devices; rewrite programs/reads remain separate. Map, moments,
pending commands, counters and RNG states are serialized. Legacy configurations
and active-only trajectories preserve their behavior; final-state replay is
required, while interrupted-stage resume remains unsupported.

CPU tests cover reference polarity, fixed partners, all detection combinations,
missed faults, blocked pairs, cap accounting, legacy equivalence and native
tiny-OPT execution/replay. Run the workflow suite and campaign-specific checks.
The first real array is the canary: its original hold, rewrite and original-active
endpoints must reproduce exp-016 state hashes and KL within 1e-6 relative before
the other two jobs are released. Every learning arm starts from the identical
rewrite-state hash, with matched rewrite receipts and input hashes.

## Budget and execution

Caps: **2 GPU-hours, 80 GB of new declared artifacts, 24 hours**, including
readiness and understood operational retries. Reuse previously staged P0s and
datasets without copying or modifying them. Preserve failed/partial evidence.
The unchanged network/batch has prior measurements: exp-016 jobs used about
310 seconds and 17.25 GB peak allocated GPU memory. Two learning arms replace
the former single learning arm; reserve 650 seconds and 15 GB per job, with
21,500 MiB peak plus 2,048 MiB headroom. Canary measurement checks these estimates.

Idle fifi/loulou 5090s are the targets. The shared queue is the sole monitoring
owner, with native validation and controller collection. Use explicit safe
windows within 09:00–17:20 and/or 22:00–07:20 Paris because the generic queue's
daytime policy is narrower than the repository's. Worker timeout is 1,800 seconds;
per-job total is 2,100 seconds and the aggregate two-hour cap remains controlling.
No shared policy, unrelated job, or GPU setting is changed.

Evidence: `results/lifecycles/opt125m-cmo-l6-exp17-partner-training/`.
Lifecycle: `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp17-partner-training.json`.
The launch helper plans one native on-chip stage per array using public CLIs.
Collection revalidates native exports and prior controls, releases jobs through
the canary gate, and writes complete coverage and per-array decisions after all
three collect. The result note will own interpretation.

## Execution handoff

CPU validation passed: 477 workflow tests and one optional skip, including
12 new partner-training checks; two campaign coverage/control tests also passed.
The strict lifecycle digest is `4305f2580a47c94e88ed88de3df9c0cc5ca0fb40dfb9d90dc83da5d38525507f`.
The full reusable lifecycle retains five upstream assignments for provenance
(40 endpoints, 48.7 GB estimated); the execution manifest selects only the
three approved assignments (24 endpoints, about 29.2 GB of estimated state).
Exact source revision, placements, manifest, deadline and monitoring observations
will be recorded before release. Model-based assumptions remain programmable per-weight
references, synthetic permanent faults, fitted endpoint programming, no read
noise or drift. Programming/write events are not measured pulse counts.
