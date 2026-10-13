---
id: "exp-015"
title: "Noise-free write oracle: how much of the recovery limit is write noise"
status: "planned"
hypotheses: []
---
# exp-015: noise-free write oracle for on-chip recovery

## Question and authorization

On October 13 the user asked for recovery "without the noise, just by tuning the
arrays", and confirmed these Explorer choices:
- remove **write noise only**, starting from the exact exp-013 held deployments
  with their programming error and faults;
- two noise-free arms, thresholded (B) and dense (C), against the reused exp-013
  noisy uncapped recovery (A);
- LR 0.003 for both arms;
- the full exp-013 grid.

Only the analog arrays adapt. There is no calibration, LoRA or other digital
adapter. Planning does not authorize launch.

Question: under the frozen exp-013 recipe, how much of the KL that remains after
noisy on-chip recovery is due to unverified write noise, how much to write gating,
and how much to faults that exact on-chip learning of active devices cannot remove?

Decision informed: whether the next recovery work targets write precision
(verify-on-write, SNR-gated or decaying-LR writes), gating design, or
non-learning controls such as rewrite, reference writes and digital calibration.

Expected outcome: write noise is a material share at every nonzero rate for
standard-3% HWA, and noise-free thresholded writes no longer degrade healthy
standard-3% arrays. An alternative that changes the decision is B ≈ A: write
noise is then not the limiter, and B − C and C − C0 locate the limit instead.

## Prior evidence

- [exp-013](../results/exp-013.md): noisy recovery improves teacher KL at every
  1–5% rate on three arrays but closes only about one third to one half of the
  fault-induced WikiText KL for std3. It worsens healthy std3 arrays by 14%.
  About 72–81% of active devices are written even on healthy arrays.
- [exp-011](../results/exp-011.md): dense noisy writes worsen KL. Accumulated
  1σ/2σ/3σ thresholds all end at development KL 0.068–0.074. A 50M-token pass
  with 26.5× the writes ends at the same KL with large fluctuations.
- [exp-012](../results/exp-012.md): cap 4 retains about 99% of uncapped gain.
- [exp-009](../results/exp-009.md): rank-64 digital LoRA reaches lower KL
  without writes, so more of the damage is compensable by noiseless parameters.

No result isolates write noise. Source HWA training is frozen; this experiment
introduces no new HWA choice, so the `hwa_papers/` consultation does not apply.

## Frozen comparison

Confirmed by the user unless marked proposed.

- **Sources:** `std3`, `cdt3`, `std9`, the exact exp-007 whole-model sources used
  by exp-013. Six analog OPT-125M MLPs, **28,311,552 logical weights**.
- **Starting states:** the 54 held P0 apparent states pinned in
  `campaigns/opt-cmo-corruption/launch/exp7_recovery_sources.json`: actual
  random-stuck fault rates 0–5% on assignments **271002, 271003, 271004**. The
  deployment programming error and faults are unchanged.
- **Arm A (reused):** exp-013 `uncapped` endpoints:
  - `cmo_thresholded_write_adam`, accumulated, nominal 2σ threshold;
  - physical write noise;
  - its use requires the anchor-replay gate below.
- **Arm B:** identical to A except that every issued write is exact on healthy
  active devices: `G_a ← G_a + ΔG`.
  - The threshold stays at 2 × 0.01413954 normalized q units, the nominal
    mid-range σ.
  - Uncapped.
- **Arm C:** `cmo_idealized_write_adam` (dense: every nonzero Adam command
  writes), with exact writes. Equivalent to digital Adam fine-tuning of the
  healthy active devices from P0, with stuck active devices frozen.
- **Common to B and C:**
  - stuck devices stay stuck, reference devices are never written, and there
    are no verify reads;
  - writes are unclamped as in exp-013; out-of-range counts are reported;
  - B and C are non-physical oracles. They bound what exact writes could buy;
    they are not candidate methods.
- **Recipe, identical to exp-013:**
  - training: `teacher_kl` objective on 8,192 recovery sequences of 256 tokens
    (2,097,152 tokens), batch 32, 256 updates, one pass, constant LR 0.003;
  - seeds: data seed 20261008, on-chip seed 53;
  - checkpoint: fixed final endpoint; development evaluated every 64 updates,
    without selection;
  - evaluation: 256 development, 384 generated-test and 1,024 WikiText-test
    sequences, unchanged.
- **Coverage:** 2 arms × 3 sources × 3 assignments × 6 rates = **108 new
  endpoints** in 18 jobs (one source × assignment × arm per job, six rates
  each), plus one anchor job. Duplicate held arms are not replicates.
- **Anchor replay:**
  - **Run:** exp-013 `std3`/271002/`uncapped`, with its exact lifecycle, under
    the exp-015 code on its original host (nom-cool-2).
  - **Pass:** reproduces exp-013's recovered held-state and prediction hashes
    for all six rates.
  - **Fail:** arm A is not reusable as matched. Stop and escalate (option: rerun
    A for all nine source × assignment groups).
- **Needs extension (Codifier):** a write-only noise factor for the CMO update
  laws (contract extension point "on-chip update law"). It must leave
  programming noise, the threshold reference, the held-state contract and
  exp-013 lifecycle serialization unchanged.

## Measurements and decision

Frozen before results. The user confirmed the 0.5 and 0.2 thresholds, the
C-validity guard and no LR screen for C at launch approval on October 13.

```text
Unit: source × assignment × fault rate × held-out domain (generated test,
  WikiText), teacher KL. Development is reported, not decisive.
  Never pool rates, domains or sources.
D = held P0; A, B, C = arm endpoints; C0 = arm C at 0% faults,
  same source, assignment and domain.
Rates 1-5%:  A - C0 = N + T + F
  N = A - B    write-noise cost
  T = B - C    gating cost under exact writes
  F = C - C0   fault residual of exact dense learning at this budget
  share = component / (A - C0); null if A - C0 <= 0; negatives retained.
  Regime = source × rate. A component is "dominant" if its share >= 0.5,
  "material" if >= 0.2, on all three assignments in both held-out domains;
  otherwise "not established".
Rate 0%: write noise explains the healthy degradation for a source if A > D
  and B <= D on all three assignments in both held-out domains.
  Optimizer noise explains it instead if A > D and B > D on all three
  assignments in both held-out domains.
C validity, per source: C0 <= D on all three assignments in both held-out
  domains, and C's development KL does not rise between its last two
  64-update checkpoints. Otherwise C is "LR-confounded" for that source:
  N = A - B is reported as above, because it does not depend on C. T and F
  are reported descriptively only and cannot be classified dominant or
  material.
```

Development trajectories, recorded every 64 updates, are used only for the C
validity guard. They never select a checkpoint.

Next step by outcome (not automatic; each needs new approval):
- **N dominant** at 3–5% for std3 and std9: target physical write precision.
- **Optimizer noise explains the healthy degradation, or C is LR-confounded:**
  revisit the learning rate and schedule with a development lifecycle. This is
  not a statement about faults.
- **F dominant (C valid):** active-device learning is near its ceiling at this
  budget. Prioritize rewrite, reference writes and calibration, plus a
  longer-budget C to separate budget from structure.
- **T dominant (C valid):** revisit gating under exact writes.
- **Otherwise:** report the shares; the user decides.

The guard was added on October 13, before any exp-015 result existed, at the
Reviewer's pre-launch request. An optional development LR screen for arm C
(for example 3e-4, 1e-3 and 3e-3 on one non-test array, selected on
development KL) is a material choice for the user at launch approval; it is
not part of the confirmed design.

Also report:
- perplexity and paired changes from P0;
- for every arm: issued writes, touched healthy devices, maximum writes per
  device, stuck attempts, out-of-range devices and verify reads;
- all three assignments with means and ranges.

Limitations:
- LR 0.003 was selected for noisy thresholded writes, so C is a ceiling of this
  fixed recipe, not an optimized one.
- One HWA training seed, reused test cohorts, synthetic homogeneous faults and
  an unclamped write model.
- A and the new arms may run on different hosts. The anchor proves code
  equivalence, not cross-host trajectory identity.

## Implementation, budget and execution

Proposed; the Codifier refines these after the cost probe, and launch needs
explicit approval.

- **Lifecycles:** `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp15-noise-free-<source>-<thresholded|dense>.json`,
  derived from the exp-013 lifecycles with only `onchip` changed.
- **Compute:**
  - about 0.26 GPU-hours per job from exp-013;
  - cap **8 GPU-hours** including the anchor and any preparation;
  - **24 hours** from first admission.
- **Storage:**
  - about 10 GB per job; cap **250 GB** additional across hosts;
  - nom-cool-2 has 60 GB free and must keep at least 40 GB free, so it takes
    only the anchor and at most one job;
  - nothing from exp-013 is deleted to make room.
- **Hosts:** to be proposed at launch. GPU rules: at most two RTX 5090s from
  08:00 to 22:00; idle 3090s nom-cool-1 and nom-cool-2.
- **Evidence root:** `results/lifecycles/opt125m-cmo-l6-exp15-noise-free/`.

Roles:
- this session is Explorer, Codifier and the single Monitor;
- session hwa-43 is Reviewer and owns `results/exp-015.md` once collection
  (or a partial/failed end) is handed off with the note, lifecycle and
  evidence paths, collect outputs, incidents and retries.

## Current execution handoff

**Launch approved October 13, about 18:45 Paris time:**
- hosts: riri (RTX 5090), nom-cool-1 and nom-cool-2 (RTX 3090);
- caps: 8 GPU-hours, 250 GB, 24 hours from first admission;
- this session is the single Monitor.

The exact authorization text is in `launch/exp15_noise_free.py`.

What changed (Codifier):
- Commit `b0159b7` adds optional `onchip.write_noise_scale` (contract
  extension point "on-chip update law"). Its tests show bitwise identity at
  scale 1, exact writes at scale 0, unchanged threshold, counters and noise
  stream, and unchanged exp-013 lifecycle sections. The workflow suite passes
  (436 tests).
- Commit `b24576c` adds `launch/exp15_noise_free.py`, the six lifecycles and
  their tests. The helper refuses to queue without a recorded authorization.

Local readiness on nom-cool-2:
- The frozen worker checkout `/home/filip/analog-research/opt-cmo-exp1/hwa-exp15-noise-free-20261013`
  is at `b24576c`, cloned from `launch/exp15-source-b24576c.bundle`.
- The anchor lifecycle digest equals exp-013's (`84eaaf12…`), and its
  preparation and dry run pass.
- Readiness-only dry runs of `std3-271002-thresholded` (`6eb2ed76…`) and
  `std3-271002-dense` (`d9bb127a…`) also pass. They are not local placements.

Measured exp-013 on-chip times: about 566 s per job on an RTX 5090 and
1,390–1,500 s on an RTX 3090, with an 18.5 GiB peak.

Proposed placement:
- **nom-cool-2 (3090):** the anchor only.
- **nom-cool-1 (3090):** the `cdt3-271002-thresholded` canary, then the shared
  queue.
- **riri (5090):** the `std3-271002-dense` canary, then the shared queue. riri
  first needs the exp-013 P0 inputs (about 24 GB) staged from nom-cool-1.

Remaining jobs are released only after all three canaries validate and the
anchor is exact.
