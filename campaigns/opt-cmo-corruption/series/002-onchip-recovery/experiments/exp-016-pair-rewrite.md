---
id: "exp-016"
title: "Fault-aware pair rewrite: the healthy device of each faulty pair absorbs its stuck partner"
status: "partial"
hypotheses: []
---
# exp-016: fault-aware pair rewrite after deployment

## Fifth phase: learning through the verified partner, gmin/gmax 2/4/6%

**Assigned October 15, 2026, about 20:50.** Phase 4's learning wrote only to
the original active device. Where the active device is stuck, the rewrite
made the reference the useful device, but learning could not touch it: its
commands hit the stuck device and used up the four-attempt cap (0.79–2.55M
wasted attempts). The user asked whether learning trains "whatever device in
the cell" and, when told it does not, asked to launch the partner-routed arm
at 01:00.

- **Cases and starting state:** phase 4's, i.e. `gmin`/`gmax` at 2, 4 and 6% on
  array **271002 only**, starting from the same phase-3 deployment (`db378a16…`).
- **Arms:** phase 4's three arms unchanged (`none`, `pair_rewrite_verify`,
  `pair_rewrite_verify_recovery`), plus exp-017's
  `pair_rewrite_verified_partner_recovery`. It is the same recipe with
  `write_roles: verified_partner`:
  - the role map comes from the controller's original verify scan, before the
    rewrite, and never from the hidden fault map;
  - train the active device unless it was flagged stuck, else the reference if
    it was not flagged, else block the pair;
  - a reference write applies the opposite sign (q = a − r); the cap counts
    issued attempts per selected device;
  - undetected faults still receive and waste attempts.
- **Expectation, not a rule:** routing can retune the weights the rewrite
  compensated through the partner. It cannot repair the clipped wrong-sign
  weights, whose partner is already at its bound, and those dominate the
  leftover. exp-017 at random 3% found no gain over original-active training.
- **Rule:** sign rule per kind × rate on 271002, both held-out domains, teacher
  KL primary. Partner learning "adds" over original-active learning if its
  teacher KL is lower in both domains, "harms" if higher in both, "mixed"
  otherwise. The same reading is given against `pair_rewrite_verify`. The
  `none`, `pair_rewrite_verify` and `pair_rewrite_verify_recovery` arms must
  reproduce phase 4 exactly. Report perplexity, writes, reference writes,
  attempts on stuck devices and touched devices. Single-array and exploratory;
  no calibration arm.
- **Budget:** 0.5 GPU-hour and 20 GB on one idle 5090 (fifi), started at
  01:00 Paris (night rule), finishing by 07:30. One predeclared `--resume`
  retry after a failed first run.
  - Estimate: phase 4 took 684 s for three arms; about 15–17 minutes.

### Fifth-phase execution handoff

Owner: this session (single Monitor).

- **Lifecycle:** `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite-std3-kinds-partner-learn.json`
  (digest `7bfdd5f9…`, file sha256 `5a8a0cd3…`).
  - Generated in the `b0c1f06` worker on fifi by `exp16_partner_lifecycle.py`
    from the phase-4 lifecycle (`f582c5db…`). It changes only `lifecycle_id`,
    `question` and the appended arm; the planned stage config differs from
    phase 4's only in those fields.
- **Plan and manifest:** `workflow plan` on fifi (cuda:0, one CPU thread, as in
  phase 4); single-stage manifest `campaign-271002.json` (`c286b459…`) pins
  `device_state` to deployment `db378a16…` and the exp-013 std3 cache and
  devices. Dry run passed.
- **Run:** fifi tmux `exp16-partner-0100` (`partner-at-0100-fifi.sh`, started
  20:51) sleeps until 01:00 on October 16, then launches only when no process
  holds the GPU, checking every 5 minutes until 07:30; if the GPU never frees,
  it logs that and does not launch.
  - The user then allowed "all gpus available". At 00:21 on October 16:
    another user ran mumax3 jobs on loulou and trex, and had a CUDA MPS
    server (no compute jobs) on riri. nom-cool-1 was unreachable, and
    nom-cool-2's /home had only 7 GB free. The job is one stage of about 15
    minutes, so it is not split.
  - **Replaced launcher, 00:21:** fifi's tmux `exp16-partner-0100` now runs
    `partner-run.sh fifi wait`: the same schedule, idle check and retry,
    plus an atomic claim (`launch/exp16-partner/claim.lock` on fifi) before
    launching.
  - **Fallback on riri:** the 271002 deployment (`db378a16…`), lifecycle, plan
    and manifest (`c286b459…`) were copied to riri's `b0c1f06` worker at the
    same paths; riri's exp-013 cache/devices hashes match; dry run passed.
    A coordinator on nom-cool-2 (tmux `exp16-partner-coord`) checks from
    01:05 every 5 minutes until 07:30. If fifi has not claimed the run and
    riri's GPU holds no process of any user, it claims the lock and starts
    `partner-run.sh riri` in riri tmux `exp16-partner-riri`. riri's log is
    `partner-chain-riri.log`.
  - Worker outputs: `<worker>/results/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite-std3-partner-learn/runs-271002`.
  - Logs: `opt-cmo-exp1/launch/exp16-partner/partner-chain-fifi.log` and
    `partner-271002.log`.
- **Evidence (after collection):** `results/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite/kinds-partner-learn-std3/`;
  the launch files are already in its `launch/`.
- **Complete, 01:44 on October 16.**
  - fifi's GPU was held by another user's process from 01:00 to 01:25. The
    coordinator then claimed riri, which ran 01:25–01:44 and exited with 0.
    fifi's launcher saw the claim and exited without launching.
  - Full inspection passed, and the three phase-4 arms reproduce phase 4
    exactly.
  - Cost: 1,156 s; 20 GB of raw runs on riri.
  - Collected copies, both hosts' launch logs and `exp16_partner_report.py`
    are in the evidence folder.
  - All partner tmux sessions have ended; nothing is running.
  - Results: [fifth phase](../results/exp-016.md#fifth-phase-learning-through-the-verified-partner-array-271002-october-16).

## Fourth phase: on-chip learning on top of the rewrite, gmin/gmax 2/4/6%

**Assigned October 15, 2026.** The user asked "will we test also on-chip
learning on top?". When asked, they chose rates **2, 4 and 6%** and **learning
only**: rewrite + learning against rewrite only. There is no calibration arm, so
this phase cannot show whether digital calibration alone would match learning.

- **Cases:** `gmin` and `gmax` at 2, 4 and 6% per physical device on arrays
  271002–271004: 18 starting conditions.
- **Starting state:** the phase-3 sweep deployments themselves. The lifecycle
  keeps the sweep's `defects` (all 12 cases) and selects the six cases with
  `onchip.case_labels`.
- **Arms**, all from the same P0:
  - `none` and `pair_rewrite_verify`: reference and anchor. They must
    reproduce the phase-3 hashes.
  - `pair_rewrite_verify_recovery`: the frozen exp-016 arm 5. It is the
    verify rewrite followed by the exp-013 cap-4 recipe:
    `cmo_thresholded_write_adam`, accumulated nominal 2σ threshold,
    LR 0.003, 256 updates, 2,097,152 tokens, physical write noise, at most
    four attempts per active device, data seed 20261008 and on-chip seed 53.
    It uses the original-active update law and the fixed final endpoint.
  - The learning rate is not re-tuned. It was chosen for noisy recovery from
    unrewritten arrays, so a negative result does not rule out other recipes.
- **Rule:** the frozen R6 sign rule, applied per kind × rate. Learning "adds"
  if its teacher KL is below `pair_rewrite_verify` on all three arrays in both
  held-out domains, "harms" if above on all of them, and is "mixed" otherwise.
  Report perplexity, the paired changes and the write cost (writes, attempts on
  stuck devices, touched devices) next to the rewrite cost.
- **Budget:** 1.5 GPU-hours and 60 GB, on fifi and riri (the two-5090 daytime
  limit) after the phase-3 stages end, finishing by 21:30 Paris. One understood
  operational retry.
  - Estimate: about 95 s per learning endpoint (the exp-013 rate on a 5090)
    plus about 35 s per held endpoint, so about 17 minutes per array.

### Fourth-phase execution handoff

Owner: this session (single Monitor).

- **Lifecycle:** `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite-std3-kinds-learn.json`
  (digest `f582c5db…`).
  - Generated in the `b0c1f06` worker by `launch/exp16-kinds/exp16_learn_lifecycle.py`
    from the phase-3 lifecycle (`3b4c688c…`).
  - Only `lifecycle_id`, `question`, `onchip.arms` and `onchip.case_labels`
    change. The arms are the helper's frozen `HOLD_ARMS` `none` and
    `pair_rewrite_verify`, plus `RECOVERY_ARM`.
- **Plan:** `workflow plan` on fifi; the identical plan directory was copied
  to riri.
- **Manifests:** one single-stage manifest per array
  (`launch/exp16-kinds/exp16_single_stage.py`), each pinning `device_state`
  to that array's phase-3 deployment:
  - 271002: `57ffbff3…`, deployment `db378a16…`;
  - 271003: `09f1f043…`, deployment `7d1355ce…`;
  - 271004: `57c1dd14…`, deployment `d5544e27…`.

  All three dry runs passed.
- **Phase-3 state at launch:** on fifi, `271003` completed at 16:09. The
  watcher then ended fifi's campaign as `271004` attempt-001 started; that
  attempt is cancelled and preserved. riri is still running `271004`.
- **Run:** chained launchers start once the phase-3 work is done and the GPU
  is idle. If another process holds the GPU, they log it and do not launch.
  - fifi: tmux `exp16-learn-chain` (`learn-chain-fifi.sh`) runs `271002`
    (launched 16:25) and then `271003`. Outputs go to `<learn>/runs-27100x`,
    logs to `learn-27100x.log`.
  - riri: tmux `exp16-learn-chain` (`learn-chain-riri.sh`) was to run
    `271004` after riri's phase-3 `271004` exited with 0.
- **Scope reduced by the user, about 16:30:** "do one array only per task - no
  need to do many seeds". Phase 4 is now **array 271002 only** (6 starting
  conditions, 18 endpoints). Neither cancelled run had started:
  - `271003`: its manifest was renamed `campaign-271003.cancelled-by-user.json`,
    so fifi's chain exits without running it;
  - `271004`: riri's waiting chain was stopped.

  The R6 sign rule is read on 271002 alone, per kind × rate in both held-out
  domains: a single-array, exploratory reading with no multi-array claim.
  Phase 3's `271004` on riri was already running and is left to finish.
- **Complete, 16:36.**
  - `271002` ran 16:25–16:36 on fifi, exited 0 and passed full inspection.
    The anchor arms reproduce phase 3 exactly.
  - fifi's chain then exited with code 2 on the set-aside `271003` manifest,
    as intended; no attempt was created.
  - Cost: 684 s and 13 GB.
  - Nothing is running.
  - Results: [fourth phase](../results/exp-016.md#fourth-phase-on-chip-learning-on-top-of-the-rewrite-array-271002-october-15).

## Third phase: std3 gmin/gmax rate sweep, 1–6%

**Assigned October 15, 2026, after the second phase.** The user asked to
"increase the percentage of the corrupt devices. do a sweep from 1 to 6".

- **Cases:** `gmin` and `gmax` at 1, 2, 3, 4, 5 and 6% per physical device:
  12 cases per array. Open and random are not included.
  - The 3% cases repeat phase 2 inside the new lifecycle. Their held P0 and
    rewritten states must reproduce the phase-2 apparent hashes, which also
    shows the sweep's deployments are equivalent.
- **Unchanged from phase 2:** std3, arrays 271002–271004 and the four frozen
  held arms. Every lifecycle field except `defects.cases`, the id and the
  question text is the same.
  - The new lifecycle is `opt125m-cmo-l6-exp16-pair-rewrite-std3-kinds-sweep.json`,
    generated from the frozen `std3-kinds` lifecycle by the same helper at
    `b0c1f06`.
  - Execution reuses the phase-2 path: a trimmed plan with pinned inputs, on
    loulou.
- **Report:** WikiText and generated perplexity, healthy versus faulty
  versus rewritten, for each kind × rate × array. Teacher KL gap closed and
  R1–R3 are evaluated per kind × rate, never pooled.
- **Healthy reference:** the phase-1 nominal rows.
- **Budget:** 1.5 GPU-hours, 120 GB on loulou, ending by 21:30 Paris today;
  one understood operational retry.
  - Estimate from phase 2: about 18 s per endpoint and about 12 s per deployed
    case, so about 1,000 s per array and about 50 minutes in total.
- **Local evidence root:**
  `results/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite/kinds-sweep-std3/`.

### Third-phase execution handoff

Owner: this session (single Monitor). **Launched October 15, 15:32 Paris, on
fifi.**

- **Host change:** loulou was planned, but at 15:31 another user's job
  (kellian) occupied it. riri and trex were also busy.
  - fifi (an idle RTX 5090) received the same `b0c1f06` bundle clone, the
    frozen std3 source and the pinned exp-013 cache/devices; all hashes
    verified.
  - On fifi, `check-realization` for 271002 is identical to the exp-013
    random-3% P0.
  - fifi has a CUDA driver failure history (exp-009, exp-017). One understood
    retry is allowed.
- **Lifecycle:** `campaigns/opt-cmo-corruption/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite-std3-kinds-sweep.json`
  (digest `3b4c688c…`).
  - Generated by `launch/exp16-kinds/exp16_sweep_lifecycle.py` in the worker.
    It first asserts that the helper reproduces the frozen std3-kinds digest.
  - Only `defects`, `lifecycle_id` and `question` change.
- **Manifest:** the trimmed manifest `26bc6a67…` keeps the deploy and on-chip
  stages for 271002–271004, with pinned inputs. Its dry run passed.
- **Run:** tmux `exp16-sweep` on fifi. Log:
  `opt-cmo-exp1/launch/exp16-kinds/sweep-run.log`. Output:
  `<worker>/results/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite-std3-sweep/runs/`.
- **First attempt, failed about 15:39.**
  - The three deployments completed.
  - `onchip__…__271002` attempt-001 failed with exit −11 after about 45 s:
    `dmesg` shows a segfault inside `libcuda.so.610.57.04` at 15:39:00. That
    is the same signature as fifi's earlier driver faults.
  - The failed attempt is preserved, along with its partial run directory and
    log `sweep-run.log`.
  - loulou, riri and trex were all occupied by other users.
- **Retry 1 (the single allowed retry), 15:40 on fifi.** The same manifest
  with `--resume` runs from unchanged inputs; the deployments are reused, not
  redeployed.
  - Session `exp16-sweep-r1`, log `sweep-run-retry1.log`; the on-chip stage
    is `attempt-002`.
  - This is not a replicate.
  - If it fails again: stop and escalate. Either wait for loulou or ask the
    user.
  - **Outcome:** `onchip__…__271002` attempt-002 completed, and `271003` then
    started on fifi.
- **Placement amendment, 16:08 (the user noted riri was free).** The on-chip
  stages were wrongly estimated at about 35 minutes per array, from a
  misread partial progress count. They actually take about 15 minutes:
  `271002` attempt-002 ran 15:40–15:55 and `271003` 15:55–16:09. The move
  saved little time. `onchip__…__271004` moved to riri
  (an idle RTX 5090). With fifi and riri, the daytime limit of two 5090s holds.
  - riri received the same `b0c1f06` clone and the identical plan directory.
  - It also received fifi's `deploy__…__271004` run directory at the same
    absolute path; `deployment.pt` is `d5544e27…` on both hosts.
  - riri's exp-013 std3 cache/devices are linked as `exp16-reused/std3`; the
    hashes are verified and the stage input paths are identical.
  - Manifest `campaign-riri-271004.json` (`693fdfea…`): the single stage with
    `device_state` pinned to that deployment. Dry run passed. Output:
    `<sweep>/runs-riri`, tmux `exp16-sweep-riri`, log `sweep-riri-271004.log`.
  - On fifi, the watcher `stop-before-271004.sh` (tmux `exp16-stop-271004`)
    ends fifi's campaign when it starts `onchip__…__271004`, so the array is
    not computed twice. Any cancelled fifi attempt is preserved.
  - riri's system clock is about 85 minutes behind the other hosts. Its run
    timestamps are skewed; this is not scientific.
  - This is a placement change, not a retry or a replicate.
- **Complete, about 16:35.**
  - Collection ran per completed attempt, using `exp16_attempts_collect.py`
    on each host because fifi's aggregate file is not final after the
    deliberate stop.
  - All completed stages passed full inspection, and the 3% anchor is exact.
  - Compute was about 1.0 GPU-hour of the 1.5 cap. Completed stages took
    2,064 s on fifi and 1,457 s on riri, plus the failed and cancelled
    attempts and the readiness check.
  - Storage was 85 GB of 120: 57 GB on fifi and 28 GB on riri.
  - Results: [third phase](../results/exp-016.md#third-phase-std3-gminmax-sweep-16-october-15).

## Second phase: std3 with gmin, gmax and open at 3%

**Assigned October 15, 2026 (implementation and execution).** The user asked:
"test 3% of the weights at g_min and 3% at g_max on OPT (with the
program-and-verify scheme) and report the PPL drop". This runs the deferred
`kinds` part of the frozen comparison below, reduced to the std3 source. The
frozen `std3-kinds` lifecycle and helper are used unchanged, and nothing else
from the deferred design is launched.

- **Rate:** 3% per physical device (`rate_ppm` 30000), as in every earlier
  exp-016 case. Each kind has exactly the random-3% fault positions, so about
  5.8% of weights have one stuck device. *Proposed reading of "3% of the
  weights"; stated to the user at launch.*
- **Cases:** `gmin_30000ppm` and `gmax_30000ppm`, as requested. `open_30000ppm`
  is part of the same frozen lifecycle and deployment and is reported
  separately at little extra cost.
- **Arms:** the four frozen held arms: `none`, `active_rewrite_oracle`,
  `pair_rewrite_oracle` and `pair_rewrite_verify`. There is no learning arm.
- **Source and arrays:** frozen std3 (`86522e23…`), assignments 271002–271004.
  New deployments use the exp-013 deployment settings on an RTX 5090.
  `check-realization` must confirm that healthy devices equal the exp-013
  random-3% P0; otherwise kinds and random differ by more than stuck values.
- **Healthy reference D0:** the phase-1 `nominal`/`none` rows of the same
  arrays, which reproduce the exp-013 healthy P0.
- **Report:** the user's headline is the perplexity drop. Report healthy,
  faulty and rewritten perplexity on WikiText and generated text for each kind
  and array, with teacher KL and gap closed alongside. Apply the frozen R1–R3
  per kind (R4–R6 do not apply).
- **Expected, from the desk estimate and the weight-level partner geometry:**
  - gmin/gmax: the verify rewrite passes R1 at least as clearly as `random`;
    the residual is about half the faulty weights left at zero;
  - open: a larger residual, because 0 µS is below the 9 µS floor;
  - active-only: possibly worse than no rewrite.
- **Budget:** 1 GPU-hour, 40 GB on loulou, ending by 21:30 Paris today. One
  understood operational retry. Estimate: about 900 s to deploy, then about
  300 s for each of three slices on an RTX 5090.
- **Host:** loulou (RTX 5090; idle at 14:46). Staging copies the frozen std3
  source from trex.
- **Local evidence root:**
  `results/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite/kinds-std3/`.

### Second-phase execution handoff

Owner: this session (single Monitor). **Launched October 15, about 14:50 Paris.**

- **Worker:** loulou, a clean clone of `b0c1f06` from bundle `06287642…` at
  `/home/filip/analog-research/opt-cmo-exp1/hwa-exp16-kinds-20261015`.
- **Staged source:** frozen std3 (`86522e23…`), copied from trex to
  `opt-cmo-exp1/inputs/exp1g-frozen-sources/std3/`.
- **Pinned inputs:** the exp-013 std3 `cache.pt` (`b7438a37…`) and
  `devices.pt` (`7bc1bd14…`), reached through `opt-cmo-exp1/exp16-kinds-reused/std3`.
- **Operational deviation (no science change):**
  - The helper's untested `deploy_campaign` rejects the kinds lifecycle,
    because exp1g's `deployment_manifest` does not expect its `onchip__`
    stages.
  - Instead, the standard `workflow plan` manifest of the frozen
    `std3-kinds` lifecycle (digest verified) is trimmed by
    `launch/exp16-kinds/exp16_kinds_manifest.py`. It keeps
    `deploy__…__27100{2,3,4}`, with the frozen source as weights, and
    `onchip__…__27100{2,3,4}`.
  - Both stages take the pinned cache and devices instead of recomputing them.
  - Result: `campaign-kinds.json` (`0fb2f7bc…`); the dry run resolved.
  - There is no portable materialization step: on-chip consumes each
    in-campaign deployment directly.
- **Readiness:** `check-realization` for std3/271002 on loulou found the gmin,
  gmax and open arrays identical to the exp-013 random-3% P0 at all
  54,924,408 healthy devices.
- **Run:** tmux `exp16-kinds` on loulou. Log:
  `opt-cmo-exp1/launch/exp16-kinds/run.log`. Output:
  `<worker>/results/lifecycles/opt125m-cmo-l6-exp16-pair-rewrite-std3-deploy/runs/`.
- **Complete, about 15:03.** All six stages exited 0 with no retries.
  - Full artifact inspection passed.
  - `check-realization` is identical on all three arrays.
  - `recovery_collect export` assumes one-stage campaigns, so collection used
    `launch/exp16-kinds/exp16_kinds_collect.py` instead: `ebl runs inspect`
    plus `workflow collect` per stage.
  - The report uses the frozen helper's `checked_report` plus a lineage check.
  - Compute was under 0.3 GPU-hours; storage was 22 GB on loulou.
  - The tmux session `exp16-kinds` has ended, and nothing is running.
  - Results: [second phase](../results/exp-016.md#second-phase-std3-with-gmin-gmax-and-open-at-3-october-15).

## Completed first phase: healthy and random 3% only

**Reviewed October 13, 2026:** all nine jobs and 90 endpoints completed and
validated by 22:39:10 Paris. The assigned reduced phase is complete; `partial`
records that the broader rates/kinds design below remains deferred. The
[result](../results/exp-016.md) reports the frozen-rule decisions and physical
costs. Pair-verify rewriting removes 92.5–93.3% of std3's WikiText fault-KL gap
and beats both exp-013 recovery budgets on every assignment. No exp-016 jobs
remain running or queued. Charged compute was 0.830 GPU-hours and declared
storage was approximately 100.91 GB, within the unchanged caps.

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

### Preserved random3 launch handoff (completed)

The following is the launch record, superseded operationally by the completed
coverage and review above. The canary collected successfully, opened the handoff
gate and released all eight remaining jobs; all nine are now complete.

- Worker code is frozen at `d49d59614e952fbbad32d99a82db4fffd4e37d7c` on fifi and
  loulou, at `/home/filip/analog-research/opt-cmo-exp1/hwa-exp16-random3-20261013`.
  Both use PyTorch 2.11.0+cu128. All nine single-stage dry runs and pinned input
  hash/provenance checks passed on each host. The workflow suite passed 475
  tests with one skip; the reduced-phase and controller checks are recorded
  under `random3/readiness/`.
- Campaign ID: `opt-cmo-exp16-pair-rewrite-20261013-random3`;
  manifest SHA-256: `8f72936c57971bfd2e2ac4ac03faebbb405ac37a971c9aaa9c819d058964c384`.
  `random3/prepared/queue/` holds the exact manifest and nine job definitions.
  The preflight gate passed. Only `std3-271002-random3` is released; the other
  eight jobs wait for its successful collection and handoff gate.
- The first canary ran on fifi and produced valid native progress. Another
  campaign subsequently occupied a third 5090 during the repository's daytime
  window. Only the exp-016 canary was cancelled. Its worker and supervisor are
  confirmed terminal; its reservation was released. Attempt
  `4e4a25525a7f4218b1d4aa215ed8d5ac`, partial artifacts and **181.362764 charged
  seconds** are retained. It is not a completed scientific endpoint or a replicate.
- The generic queue uses a narrower daytime-quota window than this repository.
  An audited `recover-resource-preemption` receipt therefore narrows **all nine
  jobs** to **October 13, 22:00–October 14, 07:20 Europe/Paris**. It restarts only
  the cancelled canary from the same original P0s, with no science, command or
  budget change. Its receipt is `random3/readiness/resource-preemption-recovery.json`;
  the former charge remains inside the 6 GPU-hour cap. No other campaign or
  shared-service policy was changed.
- The shared `run-watch-queue.service` (controller PID 1384148 at handoff) has
  observed the canary as `queued`, reason `outside execution window`. The other
  eight jobs remain `held`. It owns the 5-minute semantic checks, admission,
  validation and collection. No agent needs to poll while awaiting 22:00.
- The local collection controller is `random3/controller/`. Its analysis-only
  follow-up verifies the compact collection hashes, revalidates native exports,
  applies R0–R6 and writes `random3/analysis/decision.json`, `comparison.csv` and
  `summary.md` after all nine jobs collect. Compute workers remain at d49d596.
- Exact native receipt paths are in `prepared/{fifi,loulou}.json`.
  `reviews/progress.json` tracks completed jobs/endpoints; `reviews/summary.json`
  exists only after complete phase coverage. Expected duration is roughly
  1–2 GPU-hours, plus validation/collection, within the unchanged 6 GPU-hour /
  200 GB caps. The original outer deadline remains October 14, 21:10:57 Paris;
  the narrower overnight window controls actual admissions and termination.
- Recovery is bounded to this one proven preemption. Further operational
  incidents remain visible in the shared service and require diagnosis within
  the remaining budget; scientific retuning is not authorized. Collected
  numerical outcomes, including negative results, never trigger a retry.

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

## Earlier full-comparison preparation

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
