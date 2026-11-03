---
id: "exp-001"
title: "Original exhaustive controller design"
status: "imported"
hypotheses: ["H-001"]
---

# exp-001 — Original exhaustive controller design

## Question and provenance

Original exhaustive controller design. An adaptive fixed-amplitude pulse-count controller reduces verify reads relative to one-pulse verify without materially increasing held-out endpoint error or failure probability, and a target-dependent Gaussian is adequate only when its predeclared coverage and Wasserstein gates pass.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-001-exhaustive-design.md).

## Frozen plans

- [IBM ReRAM pulse-count program-and-verify endpoint model](../../../../../studies/ibm-reram-program-verify-noise-20260821-v1.json) — `ibm-reram-program-verify-noise-20260821-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.

Historical document locations (now references/forwarding pages): [ibm_reram_program_verify_noise_model.md](../../../../../docs/ibm_reram_program_verify_noise_model.md).

<a id="source-ibm_reram_program_verify_noise_model-objective"></a>

## Objective

Derive an endpoint programming-error model by explicitly programming IBM's
normalized ReRAM array presets with fixed-amplitude SET and RESET pulses. The
controller may change the number of successive pulses between verify reads,
but it may not change pulse amplitude. The resulting empirical model and a
compact Gaussian surrogate will be compared with AIHWKit's
`ReRamWan2022NoiseModel`.

The first milestone answers these questions:

1. How accurately can the IBM array fits reach a continuous target under
   pulse-count program-and-verify?
2. How many SET pulses, RESET pulses, verify reads, and polarity reversals are
   required?
3. Does adaptive pulse batching reduce verify cost without materially
   increasing endpoint error or failure probability?
4. Is the accepted endpoint residual adequately represented by a
   target-dependent Gaussian, as in the Wan model, or is an empirical kernel
   required?
5. How does the derived IBM endpoint distribution compare with Wan-2022 after
   both are expressed in a common normalized coordinate?

HWA deployment and Tiki-Taka continuation are deliberately deferred. This
work must first establish the programming boundary that those experiments
would consume. A later continuation must begin from the exact persistent
device states produced here; it must not redraw an endpoint or silently
reassign devices.

<a id="source-ibm_reram_program_verify_noise_model-characterization-design"></a>

## Characterization design

Use a common target grid of 41 points from zero through one inclusive. Express
the acceptance tolerance relative to each preset's nominal step fraction

```text
delta_x_nominal = dw_min / 2.
```

Characterize `tau / delta_x_nominal` values of `0.25`, `0.5`, and `1.0`; use
`0.5` as the primary comparison. This sweep exposes when a requested
tolerance lies below the practical pulse resolution instead of silently
forcing convergence.

The exhaustive v1 contract samples 4,096 identities with eight repeats over
three tolerances. The short v2 contract samples 1,024 identities with four
repeats at the primary `tau/step=0.5` tolerance. Both contracts:

- retain all 41 targets and both lower/SET and upper/RESET starts;
- partition identities, not individual trajectories, into 20% controller
  calibration, 60% noise-model fitting, and 20% final validation; and
- use the same identities, repeats, partition, target grid, conditioning
  state, and programming stream for the matched one-pulse and adaptive
  controllers.

The short design supplies 2,464 fit and 816 held-out validation trajectories
per target/condition from 616 and 204 independent device identities,
respectively. Its endpoint model is conditional on the primary tolerance; it
cannot support a tolerance-sensitivity claim.

The population-corruption arm uses the same design with the published corrupt
probability enabled. Corrupt trajectories are classified by behavior and
failure, not discarded after observation.

Record at least:

- preset name, AIHWKit version, full preset parameters, and source reference;
- construction, conditioning, pulse, and repeat seeds, plus a nullable
  controller-seed field. It is explicitly null for the deterministic
  one-pulse and adaptive controllers in this version;
- device identity, data partition, start protocol, target, and tolerance;
- apparent and persistent state before and after programming;
- every verify value and the signed batch length between verifies;
- SET count, RESET count, total pulses, verifies, and reversals;
- accepted, saturated, corrupt, non-finite, and budget-exhausted
  flags; and
- endpoint residuals relative to the continuous target.

SQLite stores a non-finite apparent terminal value as `NULL`; the accompanying
`nonfinite=1` flag is mandatory, while the last persistent state, pulse
accounting, and terminal verify event remain present. A `NULL` apparent value
on any trajectory not classified non-finite fails the integrity contract.

<a id="source-ibm_reram_program_verify_noise_model-wan-2022-comparison"></a>

## Wan-2022 comparison

Instantiate

```text
ReRamWan2022NoiseModel(g_max=40.0, noise_scale=1.0)
```

and call `apply_programming_noise_to_conductance` at targets

```text
G* = 40 x* uS.
```

That method selects the model's minimum fitted time, one second. Normalize
the sampled endpoint and error by 40 uS. Use the identical target grid and
the same number of independent samples as the IBM validation partition.

Wan clamps negative conductance but does not impose a 40-uS upper clamp in
the noise method. Report both raw normalized Wan errors and errors after an
explicit `[0, 1]` deployment clamp; never apply the latter silently. Report
the corresponding IBM apparent endpoints before any analysis-only clamp.

Compare:

- target-conditioned bias and standard deviation;
- empirical quantiles, skewness, kurtosis, and tails;
- RMSE and absolute-error exceedance curves;
- the fitted `mu(x)` and `sigma(x)` functions;
- saturation or clamp mass; and
- IBM convergence and pulse/verify cost, for which Wan has no analogue.

This is a normalized operational comparison, not a claim that IBM's `[-1, 1]`
state corresponds to `0-40 uS`. It also compares an immediate, time-unspecified
IBM preset endpoint with Wan's one-second fitted endpoint. Both limitations
must appear beside every headline comparison.

<a id="source-ibm_reram_program_verify_noise_model-reproducibility-and-artifacts"></a>

## Reproducibility and artifacts

Before generating headline results, add a tracked study plan and exact
characterization configuration. Raw results belong under one
`results/<study-id>/` root and must preserve failed trajectories.

The completed characterization must produce:

- a machine-readable trajectory artifact with the recorded fields above;
- a fail-closed integrity report covering trajectory/event counts, pulse and
  reversal accounting, acceptance semantics, identity partitions, and matched
  controller/tolerance random streams;
- an empirical-kernel artifact with target bins and validation partitions;
- a bounded piecewise-uniform endpoint artifact containing the global-uniform
  baseline, held-out adequacy results, and separate terminal-outcome tables;
- a versioned fit artifact containing normalization, polynomial
  coefficients, controller settings, failure model, source versions, seeds,
  and input hashes;
- a Wan comparison summary in machine-readable form;
- plots of residual distributions, `mu(x)`, `sigma(x)`, convergence, pulse
  cost, verify cost, and reversals; and
- a Markdown report separating measured results from interpretation.

Deterministic replay must reproduce device parameters, pulse trajectories,
partitions, fitted coefficients, and summary metrics from the recorded
configuration. Construction order must not change device identities or
results.

<a id="source-ibm_reram_program_verify_noise_model-implemented-execution-surface"></a>

### Implemented execution surface

The strict experiment ID is `ibm_reram_program_verify.v1`, exposed through:

```bash
python -m ebl characterize \
  --config examples/reram_program_verify/smoke_om.json \
  --output-dir /tmp/ibm-reram-program-verify-smoke
```

Configurations declare a `smoke` profile (reported as operational evidence),
the immutable exhaustive `production` profile, the immutable 512-pulse
`production_short` profile, or the immutable 128-pulse
`production_short_cap128` successor profile. Both short profiles require
1,024 identities, four repeats, 41 targets, the half-step tolerance, both
starts and controllers, the four-quiet-pulse boundary rule, target-independent
blocked conditioning, and unchanged adaptive and Wan settings. The pulse cap
is the only difference between the two short profiles. Smoke artifacts cannot
satisfy a production plan.

AIHWKit samples each identity's fitted device-to-device parameters with an
identity-derived construction seed. The pulse transition is then evaluated by
`training/ibm_reram_program_verify.py` with explicit per-trajectory PyTorch
streams. CPU execution retains scalar per-trajectory generators. CUDA
execution generates each independently seeded stream in buffered CPU blocks
and stages those immutable draws on the GPU; exact replay uses recorded seeds,
draw positions, execution backend, and the sampled population. The local CUDA
environment does not contain AIHWKit, so population construction and Wan
sampling run through the pinned AIHWKit 1.1.0 interpreter and emit hashed
receipts. This split is required because AIHWKit 1.1.0 does not expose or
serialize the native cycle-to-cycle generator. Focused tests cover native
noiseless pulse parity, CPU/CUDA equation parity, buffered-normal moments,
CUDA continuation replay, and byte-identical end-to-end CUDA ledgers.

The runtime writes the full trajectory and verify-event stream to SQLite,
validates that persisted ledger before fitting, preserves the sampled
population in NPZ, emits integrity/calibration/empirical/bounded-uniform/
Gaussian/Wan JSON artifacts, creates dependency-free SVG plots (including target-binned
quantiles, tail exceedance, and saturation/clamp mass), and writes a Markdown
report. The exhaustive plan remains
`studies/ibm-reram-program-verify-noise-20260821-v1.json`; the reviewed local
plan is `studies/ibm-reram-program-verify-noise-20260821-v2.json`; its
128-pulse matched successor is
`studies/ibm-reram-program-verify-noise-20260822-v3.json`. Each short plan
contains 671,744 trajectories per arm and 2,686,976 total. An exact-width
five-target CUDA sizing gate completed 81,920 trajectories and 2,863,651
verify events in 61.6 seconds on the local RTX 3090, projecting 8.4 minutes
per arm before a 10x operational safety factor. Neither plan is implied by a
successful smoke run: the declared arms must be prepared and launched through
the study workflow before headline results are claimed.

The network-facing subset is separately frozen as
`studies/ibm-reram-om-hwa-device-model-20260822-v1.json`. It keeps only the two
OM corruption policies and the lower-start adaptive condition, restricts
one-pulse trajectories to calibration identities, and records sampled fixed
bounds for analysis-only reachability. Its 402,784 total trajectories are a
6.67x reduction relative to the broader cap-128 matrix. It is sufficient only
for the first OM HWA pilot and does not replace the broader device/controller
comparison.

<a id="source-ibm_reram_program_verify_noise_model-completion-criteria"></a>

## Completion criteria

The first milestone is complete when:

1. AIHWKit one-pulse transitions have a focused parity test demonstrating
   that the harness emits the requested pulse count and direction.
2. One-pulse and adaptive controllers have invariant tests for acceptance,
   reversal, pulse budgets, batching, hidden-parameter isolation, and exact
   seeded replay.
3. OM primary and HfO2 stress-control characterization is complete on the
   declared grid and partitions, including the separate corrupt-population
   arms.
4. Empirical kernels, bounded piecewise-uniform and Gaussian surrogates,
   adequacy results, failure/corrupt models, and cost models are generated from
   held-out device identities.
5. The one-second Wan comparison is generated with raw and explicitly clipped
   views and with the normalization and time limitations stated.
6. No HWA or Tiki-Taka result is claimed from this milestone.

<a id="source-ibm_reram_program_verify_noise_model-deferred-matched-training-study"></a>

## Deferred matched training study

After the endpoint model is validated, a separate predeclared study may
compare:

```text
HWA target -> IBM pulse-count P&V -> hold
HWA target -> identical IBM P&V state -> Tiki-Taka fine-tuning
```

Both arms must share the HWA checkpoint, target mapping, device identities,
sampled device parameters, controller, accepted programmed state, and all
pre-fine-tuning metrics. Tiki-Taka must continue from the preserved device
state rather than constructing a fresh preset tile or sampling the fitted
endpoint model again.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
