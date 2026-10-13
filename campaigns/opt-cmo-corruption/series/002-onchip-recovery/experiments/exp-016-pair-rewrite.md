---
id: "exp-016"
title: "Fault-aware pair rewrite: the healthy device of each faulty pair absorbs its stuck partner"
status: "ready"
hypotheses: []
---
# exp-016: fault-aware pair rewrite after deployment

## Active first phase: healthy and random 3% only

The user selected this reduced phase and then assigned **implementation and
execution**. This supersedes the launch-pending wording and broader coverage
below for the current assignment. The full comparison remains deferred.

- **Frozen comparison:** std3, cdt3 and std9; assignments 271002–271004;
  `nominal` and `random_30000ppm`; all five arms listed below. Six analog MLPs
  (layers 6–11), 28,311,552 logical weights. No HWA retraining or new deployment.
- **Inputs:** the exact exp-013 held P0s and pinned teacher/cohorts, including
  256 development, 384 generated-test and 1,024 WikiText-test sequences of
  length 256. Arm 5 retains 2,097,152 recovery tokens, 256 updates, LR .003,
  physical write noise and the cap of four issued attempts per active device.
- **Selection:** three `opt125m-cmo-l6-exp16-pair-rewrite-<source>-random3.json`
  lifecycles use `onchip.case_labels`; upstream defect declarations and hashes
  stay unchanged. Nine single-stage jobs produce 18 starting conditions and
  90 endpoints. All starts use their named P0; the rewrite seed stays 241016.
- **Decision:** apply R0–R6 below at 3% only, independently by source and held-out
  domain, across all three assignments. Other kinds and rates are untested in
  this phase. Neither phase completion nor a favorable result completes the
  broader study or establishes viability outside this regime.
- **Physical assumptions:** the differential subtraction polarity stays fixed;
  either per-weight device can be reprogrammed. Verify-based detection uses 5σ;
  the hidden map is analysis-only. P&V is the fitted 2% CMO Gaussian endpoint
  model, not a simulation of individual pulses. Read noise and drift stay off.
- **Caps:** 6 GPU-hours, 200 GB across hosts, 24 hours. Projection: 1–2 GPU-hours
  and about 70 GB of checkpoint payload, plus staging. No source evidence is
  deleted. New inputs staged on fifi count toward the 200 GB cap.
- **Placement:** idle fifi and loulou RTX 5090s, subject to the repository's
  08:00–22:00 two-5090 limit and idle-only admission. Existing jobs are preserved.
  Fifi receives a new copy of the frozen P0s; loulou reuses its existing copy.
- **Evidence:** `results/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite/random3/`.
  Native workers retain full checkpoints; local collection retains validated
  compact evidence and exact run locations.
- **Readiness:** the first job is `std3-271002-random3`. Every job verifies native
  replay and input hashes, and reproduces P0 hashes and held KL within 1e-6
  relative. Only successful canary collection opens the handoff gate and
  releases the remaining eight jobs. Poor recovery is a scientific outcome,
  not a reason to retry or tune.
- **Monitoring:** the existing shared execution service owns admission,
  5-minute semantic progress checks, terminal validation and collection. This
  session owns operational diagnosis and scientific review; no duplicate
  watcher. Runtime/receipt identities and the deadline go in the handoff below.

## Question and authorization

On October 13 the user approved drafting this comparison. Each weight is a pair
`q = (G_a − G_r) / 39.6 µS`. When one device is stuck, the other, healthy
device can be programmed so that the pair again reads its frozen target. A stuck
active device becomes the fixed side, and its reference is reprogrammed: a role
swap. A stuck reference is absorbed by the active device. The user's framing:
the stuck device becomes the reference, and the other one is adjusted. No
gradients and no learning are used.

Confirmed by the user:
- a one-shot fault-aware rewrite of the healthy partner device;
- a known fault map (oracle) versus a fault map measured with verify reads;
- the four stuck kinds `random`, `gmin`, `gmax` and `open`;
- the three frozen HWA sources;
- starting from the exp-013 deployments.

Later the same day the user also confirmed:
- `gmin`, `gmax` and `open` at 3% only;
- the optional learning-on-top arm (arm 5);
- the R1–R3 thresholds and 5σ detection;
- caps of 6 GPU-hours, 200 GB and 24 hours.

Planning does not authorize implementation or launch.

Question: after deployment, how much of the fault-induced teacher-KL loss does a
fault-aware pair rewrite remove, compared with rewriting active devices only,
HWA-only deployment and the exp-013 noisy on-chip recovery? At what physical
cost in verify reads and programmed devices? Does the answer depend on the stuck
kind and the HWA source?

Decision informed:
- Whether fault-aware rewriting becomes the no-weight-learning baseline that
  on-chip learning must beat. It belongs to AGENTS.md's "rewrite of frozen
  targets" class.
- Whether on-chip work should then target only the residual.
- Whether corruption-aware HWA is still worth its healthy-chip cost once faults
  are compensated.

Expected outcome:
- For `random` faults, the measured-map pair rewrite closes at least 90% of the
  fault-induced KL gap at every 1–5% rate.
- It beats exp-013 noisy recovery on every assignment.
- The measured map matches the oracle.
- `open` faults leave the largest residual.

Alternatives that would change the decision:
- The rewrite falls well short of the desk estimate, for example through partner
  programming noise or an interaction with HWA. Learning then stays primary.
- A one-sided (active-only) rewrite is worse than no rewrite. The baseline must
  then always fix both roles.

## Prior evidence

- [exp-013](../results/exp-013.md): noisy recovery for std3 at 3% faults reduces
  WikiText KL from 0.188 to 0.117 (healthy deployment: 0.044). It touches about
  79% of active devices, about the same share as on healthy arrays (72%).
- [exp-009](../results/exp-009.md): digital LoRA is rank-limited. Training KL is
  flat from about update 1,000 of 6,100 at 0.073, 0.061 and 0.042 for ranks 4,
  16 and 64.
- [exp-015](exp-015-noise-free-writes.md) (planned): noise-free writes split the
  exp-013 limit into write noise, gating and a fault residual F, with active
  devices only. Its next-step list names reference writes. exp-016 tests that
  route directly, without learning.
- [exp-007](../../001-hwa-robustness/results/exp-007.md): fault-aware HWA
  (`cdt3`) pays a large healthy-chip cost (WikiText KL 0.114 vs 0.044 for std3).
- [OM local reference compensation](../../../../../docs/ibm_om_local_reference_compensation.md)
  concerns OM symmetry-point baselines on MNIST, not stuck-fault compensation.
  It is a different question.

**Desk estimate (model-based, not evidence).**
[`analysis/exp016_desk_estimates.py`](../analysis/exp016_desk_estimates.py)
setup:
- simulated CMO stuck maps on the pinned FP32 OPT-125M MLPs 6–11 (not the HWA
  masters or a deployed array);
- 6,144 real validation tokens;
- one layer at a time, with exact compensation clipped to 9–88.2 µS.

Results:
- **Pair compensation** leaves a residual of the fault output-error energy of:
  - 2.1%, 4.1% and 5.8% for `random` at 1%, 3% and 5%;
  - 2.9–3.3% for `gmin` and `gmax`;
  - 6.6–7.3% for `open`.
  It programs about twice the fault rate in weights (5.8% at 3%).
- **Active-only compensation** leaves 55–58% for `random`. For `gmin`, `gmax`
  and `open` it reaches 70–143% and exceeds no compensation at 3–5%. Opposite-sign
  stuck-active and stuck-reference errors partly cancel on inputs with a nonzero
  mean, and fixing one role removes that cancellation.
- **For comparison, the best rank-64 correction** captures 69% of the same error
  for `random` 3% on synthetic text, and 55% when transferred to WikiText.

## Frozen comparison

- **Network.** Six analog OPT-125M MLPs, **28,311,552 logical weights**, CMO
  2% P&V. The other components are digital. *Confirmed (unchanged).*
- **Sources.** `std3`, `cdt3` and `std9`, the exact exp-007 sources used by
  exp-013. *Confirmed.*
- **Assignments.** **271002, 271003, 271004**. *Confirmed via exp-013.*
- **Starting states:**
  - `random`: the 54 held P0s pinned in
    `campaigns/opt-cmo-corruption/launch/exp7_recovery_sources.json`, at
    0–5%. *Confirmed.*
  - `gmin`, `gmax` and `open`: new deployments from the same sources,
    assignments and deployment settings. Only `defects.cases[].kind` changes.
    Rate **3% only**: 27 new P0s. *Confirmed.*
    - `hierarchical_faults` draws placement from RNG streams separate from
      the stuck values. For one assignment and rate, every kind therefore has
      exactly the `random` fault positions.
    - The Codifier verifies whether healthy-device programming realizations
      also coincide. If they do, the kinds differ only in stuck values.
  - 0% is kind-free: D0 is the exp-013 healthy P0 of the same source and
    assignment.
- **Rewrite targets.** `q*` is each weight's frozen deployment target, the same
  value used to program P0. Targets are not re-optimized for the faulty array.
  - **Active stuck** at `G_s`, reference healthy: program the reference to
    `clip(G_s − 39.6·q*, 9, 88.2) µS`.
  - **Reference stuck** at `G_s`, active healthy: program the active device to
    `clip(G_s + 39.6·q*, 9, 88.2) µS`.
  - **Both stuck:** leave unchanged.
  - **Programming law.** Each partner program uses the campaign's unchanged
    deployment law: one unclamped Gaussian P&V endpoint at 2% acceptance. It
    draws from a declared rewrite seed disjoint from the deployment seeds. It
    replaces the device's held value; it is not an incremental write.
- **Arms per starting condition.** All arms start from the same P0, and no arm
  uses digital calibration (`calibration: false`).
  1. `none`: held P0 (D).
  2. `active_rewrite_oracle`: fixes stuck references only, using the hidden
     fault map and stuck values. A non-physical oracle; it isolates what the
     role swap adds.
  3. `pair_rewrite_oracle`: fixes both roles from the hidden map. A
     non-physical upper bound.
  4. `pair_rewrite_verify`: both roles, with the map and stuck values from
     verify reads. This is the physical candidate. *Detection confirmed:*
     - read each device's held conductance once at P0: 2 reads per weight,
       56,623,104 reads;
     - flag a device if `|G_read − G*| > 5·σ(G*)`, where `G*` is its
       programming target (the active target, or 48.6 µS for a reference);
     - use the read value as `G_s`;
     - the hidden mask never reaches this arm.
  5. `pair_rewrite_verify_recovery` (*confirmed*): arm 4's held state followed
     by the exp-013 recipe at cap 4. It tests whether learning adds anything on
     top.
     - Recipe: `cmo_thresholded_write_adam`, physical write noise, accumulated
       nominal 2σ threshold, LR 0.003, 256 updates, data seed 20261008,
       on-chip seed 53.
     - `random` kind only, all six rates, three sources and three assignments.
     - At 0%, with no detections, it starts from P0 itself. It then doubles as a
       replay check against exp-013's cap-4 healthy endpoint.
- **External references, not rerun.** exp-013 noisy recovery endpoints (uncapped
  and cap 4) for `random`. exp-015 arms when available. The `none` arm must
  reproduce exp-013's held-state hash and held metrics within 1e-6 relative.
- **Evaluation.** The exp-013 cohorts, unchanged: 256 development, 384
  generated-test and 1,024 WikiText-test sequences of 256 tokens. The primary
  metric is teacher KL; perplexity is reported.
- **Coverage:**
  - `random`: 54 starting conditions;
  - other kinds: 27 starting conditions;
  - each condition has the hold arm plus three rewrite arms: 81 holds and 243
    rewrite endpoints;
  - arm 5 adds 54 endpoints in nine source × assignment jobs.
  - Duplicate held arms are not replicates.

## Measurements and decision

Frozen before results. The user confirmed the numeric thresholds on October 13.
R6 is a sign rule added when arm 5 was confirmed, before any result.

```text
Unit: source × assignment × stuck kind × fault rate × held-out domain
  (generated test, WikiText); teacher KL. Development is reported, not decisive.
  Never pool sources, kinds, rates or domains.
D = held P0; D0 = healthy held P0 of the same source and assignment.
Gap closed g(X) = (D − X) / (D − D0); null if D − D0 <= 0; values above 1 retained.
"On all assignments" = all three assignments in both held-out domains, per
  source × kind × rate.

R1 faults removed:      g(pair verify) >= 0.90 on all assignments.
R2 detection adequate:  g(pair verify) >= g(pair oracle) − 0.02 on all assignments.
R3 role swap matters:   g(pair oracle) − g(active-only oracle) >= 0.20 on all
                        assignments.
R4 beats noisy recovery (random, 1–5%): pair-verify KL < exp-013 KL, uncapped
                        and cap 4, on all assignments.
R5 best source after compensation: the source with the lowest pair-verify KL on
                        all assignments, per kind × rate; otherwise "not established".
R0 healthy no-op (0%):  pair verify changes P0 only through false-positive
                        detections. Report them, and flag any relative KL change
                        above 1e-4.
R6 learning on top (random, 1–5%): arm 5 "adds" if its KL is below pair-verify KL
                        on all assignments, and "harms" if it is above on all of
                        them; otherwise "mixed". Report g(arm 5) and its writes
                        next to exp-013 cap 4.
```

Next step by outcome (not automatic; each needs new approval):
- **R1 and R4 hold for `random` 1–5%.** Pair rewrite becomes the
  no-weight-learning baseline for the viability rule. On-chip work moves to the
  residual: R6 and exp-015's write-noise question say whether that residual is
  worth learning. If R5 names `std3` at every rate, corruption-aware HWA is not
  justified in this regime.
- **R3 holds but R1 fails.** Compensation helps but leaves material loss.
  Investigate partner programming noise and both-stuck weights. Learning on top
  stays relevant.
- **R2 fails.** Detection is limiting; design the verify policy (threshold,
  repeated reads).
- **R1 fails for `open` only.** A kind-specific range limit; report it as such.

Also report, for every arm:
- perplexity and paired changes from P0;
- verify reads;
- programmed devices by role (active, reference);
- partner targets clipped to range, and both-stuck weights left as read;
- on the analysis side, detection false positives and negatives against the
  hidden map;
- out-of-range devices;
- the exp-013 write counts as a comparison;
- all three assignments with means and ranges.

Limitations:
- homogeneous synthetic faults (κ = ∞);
- no read noise, so one read suffices;
- P&V is one Gaussian endpoint, with no loop length or time cost;
- the oracle arms are non-physical;
- one HWA training seed and reused test cohorts.
- The scheme needs a per-weight programmable reference device, as in this CMO
  model (adjacent rows, NeuRRAM-style). A design with one shared reference
  column per array admits no role swap.

Source HWA training is frozen, so the `hwa_papers/` consultation does not apply
to the design. R5 informs future HWA choices.

## Implementation, budget and execution

The caps are confirmed. The Codifier refines the estimates after its cost probe,
and launch needs explicit approval.

- **Needs extension:**
  - *Technology / P&V* (`training/reram_cmo.py`, `workflow/devices.py`,
    `workflow/program_verify.py`): a controller-visible per-device verify read
    of the held conductance; P&V programming of selected reference devices to
    arbitrary targets. `CmoMatrix.program` currently sets references only to
    mid-range.
  - *Arm* (`workflow/lifecycle.py`, `workflow/onchip.py`; nearest contract row:
    "on-chip update law"): a one-shot fault-aware rewrite with
    `detection: oracle|verify`, `roles: active|pair` and the detection
    threshold. It needs exact replay. Oracle arms must be labelled, and the
    verify arm must provably not read the hidden mask.
  - New `gmin`, `gmax` and `open` deployments use supported defect kinds: a new
    lifecycle, no code.
- **Sequencing.** exp-015's write-only noise factor is committed (`b0159b7`).
  - exp-016 extensions must keep `tests/test_workflow_write_noise.py` passing.
    It pins scale-1 bitwise identity, exact scale-0 writes, the unchanged
    threshold and noise stream, and the rejection of rewrite arms when
    `write_noise_scale` is declared.
  - The one-shot rewrite arm defines its own rule explicitly. It must not change
    exp-015's semantics.
- **Compute.**
  - Estimate: about 9 s per endpoint evaluation (from exp-013 job timing), so
    about 0.8 GPU-hours for 324 hold and rewrite endpoints, plus deployment of
    27 new P0s;
  - arm 5: about 2.3 GPU-hours (nine exp-013-sized jobs);
  - cap **6 GPU-hours** including deployments, anchors and arm 5;
  - **24 hours** from first admission.
- **Storage.** About 0.23 GB per saved held state; cap **200 GB** across hosts.
  Nothing from exp-013 is deleted.
- **Hosts.** Proposed at launch, under the GPU rules: at most two RTX 5090s from
  08:00 to 22:00, and idle 3090s. At drafting time, exp-015 held riri,
  nom-cool-1 and nom-cool-2 from about 19:00 for roughly three hours, and loulou
  ran the user's CIFAR job.
- **Evidence root:** `results/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite/`.

## Current execution handoff

Ready. Session hwa-43 drafted this note as Explorer, and every Explorer field is
resolved. On October 13 the user assigned a separate Codifier/Monitor session.

**Codifier, October 13 (phase 1: implemented and probed; nothing launched).**

- **What changed** (`7008d0f`):
  - an optional per-arm `fault_rewrite` with the contract recorded in the
    [lifecycle contract](../../../../../docs/crossbar_lifecycle.md) and the
    [CMO device contract](../../../../../docs/reram_cmo_device.md): per-device
    verify reads, P&V programming of either device from a declared stream, and
    the one-shot oracle/verify, active/pair rewrite;
  - `tests/test_workflow_fault_rewrite.py`, 23 tests. They prove that verify
    decisions depend only on reads and precede any fault-map access, and that
    learning after a rewrite on a healthy array is bitwise plain learning;
  - the full suite: 2,648 passed, 13 skipped.
- **Lifecycles:** `opt125m-cmo-l6-exp16-pair-rewrite-<std3|cdt3|std9>-<random|kinds>.json`,
  derived from the exp-013 cap-4 lifecycles. Only `onchip.arms` and, for
  `kinds`, `defects.cases` change. The rewrite seed is 241016, shared by every
  rewrite arm. Helper: `launch/exp16_pair_rewrite.py`.
- **Fact 1, healthy realizations.** The code binds each matrix's endpoint
  stream to the assignment and matrix only, and `program` draws noise for every
  device. A test confirms that `gmin`, `gmax`, `open` and `random` share fault
  positions and healthy endpoints on one device type. CUDA RNG partitioning can
  differ across GPU models, so kinds deployments must run on an RTX 5090, like
  the exp-013 P0s (trex). `check-realization` verifies identity on the
  deployment host.
- **Fact 2, held reproduction.** For std3/271002 at 3% (probe on nom-cool-2,
  exp-013's own host for this group), the `none` arm reproduces exp-013's P0
  hash and prediction hashes on all three cohorts. KL differs by at most
  7.7×10⁻⁹ relative. Collection checks this for every random P0.
- **Probe** (not evidence): `std3-random-probe`, one case, every arm, RTX 3090.
  - Time: 42.6 s per held arm including rewrite, evaluation and replay; 191 s
    for the learning arm.
  - Peak reserved memory: 17.0 GiB.
  - Saved state: 0.68 GB per held arm, 1.13 GB for the learning arm.
- **Projection:**
  - random slices: about 2,300 s on a 3090 or 1,000 s on a 5090 each, and
    about 23 GB each;
  - kinds slices: about 650 s or 300 s, and about 8 GB each;
  - three deploy campaigns on a 5090;
  - in all, about 3.7 GPU-hours if the random slices run on 5090s, or 6.9 if
    they run on 3090s;
  - about 300 GB of storage.
- **Escalated before launch:**
  1. Projected storage of about 300 GB exceeds the confirmed 200 GB cap.
  2. The 6 GPU-hour cap holds only if most random slices run on RTX 5090s.

Launch needs explicit approval of hosts and of the storage decision.
