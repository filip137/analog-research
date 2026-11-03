# IBM ReRAM pulse-count program-and-verify noise model

- Status: short CUDA production protocol reviewed and sized; production evidence pending
- First milestone: device characterization and Wan-2022 comparison
- Reference runtime: AIHWKit 1.1.0

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

## Evidence and claim boundary

The IBM presets are hardware-derived fitted models, not raw pulse-trace
replay. Their states are normalized and do not carry a unique physical
microSiemens calibration. Results from this study therefore support claims
about the published AIHWKit fits and the declared controller, not direct
claims about an independently measured IBM array.

The source and evidence-class rules in
[`synapse_data.md`](synapse_data.md) and the physical-calibration limitations
in [`reram_device_catalog.md`](reram_device_catalog.md) are authoritative.

| Role | AIHWKit source | Native coordinate | Use here |
| --- | --- | --- | --- |
| Primary | `ReRamArrayOMPresetDevice` | Nominal state `[-1, 1]`; `dw_min=0.0949` | Optimized-material IBM 14 nm ReRAM array fit |
| Stress control | `ReRamArrayHfO2PresetDevice` | Nominal state `[-1, 1]`; `dw_min=0.4622` | Coarse baseline-HfO2 IBM array fit |
| Comparator | `ReRamWan2022NoiseModel` | Physical target conductance with a 40-uS fit reference | One-second target-dependent endpoint model |

Across the nominal IBM span of two state units, the nominal minimum steps are
`0.04745` and `0.2311` of the full span for OM and HfO2 respectively. The
corresponding full-span arithmetic is only about 21 and 4.3 nominal steps.
Those numbers are scale diagnostics, not claims about usable levels: sampled
bounds, device-to-device variation, cycle-to-cycle variation, reference
subtraction, write noise, and reversals all affect the realized trajectories.

AIHWKit disables corrupt devices in both presets by default even though the
source fits record corrupt-device probabilities of approximately `0.1348`
for OM and `0.0977` for HfO2. The primary continuous-noise characterization
sets corruption to zero so it can be compared with Wan's continuous endpoint
model. A separate population arm enables the published corrupt probability
and reports structural or stuck-device outcomes without absorbing them into
a Gaussian variance.

## State and noise semantics

Use a global normalized coordinate

```text
x = (w + 1) / 2,
```

where the nominal IBM state `w` is in `[-1, 1]`. A target `x*` is mapped with
the nominal global bounds. It must not be remapped using a cell's hidden
sampled bounds; doing so would give the controller unavailable device
knowledge and would hide unreachable targets.

For device `i`, retain a persistent state `z_i` and sample its
device-to-device parameters exactly once. A pulse of direction `d` advances
that persistent state through the preset's nonlinear stochastic transition:

```text
z_i,k+1 = F_d(z_i,k, theta_i, xi_i,k).
```

Here `theta_i` is fixed for the device and `xi_i,k` is the cycle-to-cycle
randomness for that pulse. AIHWKit presets with `write_noise_std` distinguish
the persistent programmed state from the apparent state. The verify
controller observes the apparent state because a physical verify operation
does not observe a simulator's hidden noiseless state.

For the first milestone:

- `SET/up` means a pulse intended to increase the normalized state;
- `RESET/down` means a pulse intended to decrease it;
- device-to-device parameters persist across every pulse and verify step;
- successive pulses in a batch are applied sequentially, so the nonlinear
  state changes after every pulse even though no intermediate verify occurs;
- the acceptance decision uses the apparent state;
- both apparent and persistent endpoints are recorded;
- no independent read-noise or endpoint-noise sample is added; and
- retention, relaxation, and temporal drift are out of scope.

Consequently, an accepted residual is initially called **effective endpoint
programming error**, not pure write noise. It contains finite pulse
resolution, stochastic pulse response, finite acceptance tolerance, apparent
write variation, and controller behavior. The analysis must report its bias
before describing it as zero-mean noise.

## Initialization protocols

The primary protocol is `lower_to_target`: condition each device at its lower
saturation region, then program toward `x*`, reversing only after an
overshoot. The directional control is `upper_to_target`, starting at the
upper saturation region. The final model remains conditioned on the declared
start protocol rather than averaging the two histories silently.

Boundary conditioning uses repeated same-direction pulses. The exhaustive v1
contract stops after eight successive persistent-state changes smaller than
`1e-6` of the nominal span or after 4,096 conditioning pulses. The short v2
contract uses four successive quiet pulses after an operational 1,280-case
pilot reduced median conditioning from 1,798 to 138 pulses, removed the 23.1%
cap-failure rate, and left successful endpoints at the sampled boundary. The
short study clones one independently conditioned state per
device/repeat/start across targets, controllers, and tolerances. This is a
blocked simulated comparison: it isolates target-programming dynamics while
the four repeats still sample conditioning variability. Conditioning pulses
and their seed are recorded but excluded from the target-programming pulse
budget. Failure to establish the declared boundary is an initialization
failure, not a programming residual.

Every stochastic repeat reuses the sampled device parameters but uses a new
cycle-to-cycle pulse stream and reconditions the device to its declared start
state. Device-construction and pulse-stream seeds must be distinct and
recorded. The strict configuration uses separate base seeds for construction,
repeat identity, conditioning, target programming, partitioning, analysis,
and Wan sampling; duplicate base seeds are rejected.

## Program-and-verify controllers

Let `y_k` be the apparent normalized verify value, `x*` the continuous target,
and `tau` the half-width of the acceptance window. Quantization of `x*` is
disabled in the first milestone.

At every verify step:

```text
if abs(y_k - x*) <= tau:
    accept
elif y_k < x* - tau:
    apply SET/up pulses
else:
    apply RESET/down pulses
```

Reaching the opposite side of the window changes the next pulse direction.
The controller does not RESET after every rejected SET pulse. It continues in
the required direction until a verify result calls for reversal.

### One-pulse verify reference

Apply exactly one pulse before every new verify. This is the precision
reference and establishes the controller-independent cost of the finest
available feedback.

### Adaptive pulse-count controller

Apply a predicted batch of identical pulses before the next verify. The
controller begins with a direction- and state-dependent population estimate
of the absolute one-pulse response, `s_hat(x, d)`, learned only from the
controller-calibration device partition. After a batch, update the estimate
for that device using the observed apparent displacement divided by the batch
length.

For distance `D = abs(y_k - x*)`, choose

```text
n = floor(eta * D / max(s_hat, epsilon)),
n = clamp(n, 1, n_batch_max).
```

The initial contract uses `eta=0.75`, `n_batch_max=32`, and forces `n=1` when
`D` is at most two predicted pulse steps. A commanded batch is also truncated
to the remaining total-pulse budget. These values may be changed only by a
new versioned characterization plan, not after examining the Wan comparison.

The realistic controller may use only target values, verify observations,
its recorded history, and population calibration. A diagnostic oracle may
use hidden device parameters to provide an upper bound, but oracle results
must not enter the derived model or headline comparison.

### Stopping and failure

A v1 or v2 trajectory terminates when:

- the apparent verify state enters `[x* - tau, x* + tau]`;
- 512 target-programming pulses have been applied; or
- a non-finite state is observed.

The separately versioned v3 successor changes only the target-programming
cap from 512 to 128 pulses. It does not modify the immutable v1/v2 stopping
contract or their recorded evidence.

Only accepted trajectories enter the success-conditioned continuous residual
kernel. Each other outcome contributes to a separate target-conditioned
failure probability and retains its last apparent and persistent states.

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

## Derived endpoint model

For an accepted trajectory define

```text
e = x_final_apparent - x*.
```

The authoritative output is an empirical conditional kernel

```text
p(e | x*, preset, controller, tau, start_protocol, success).
```

Keep the target-binned samples or quantiles together with a separate
`p_failure` and cost model. Do not replace failed or corrupt devices with a
large continuous error.

Accepted residuals are bounded by the verify stopping rule. The primary
compact candidate is therefore a target-conditioned eight-bin
piecewise-uniform density on `[-tau,+tau]`. Add a fixed `0.5` pseudocount to
each bin on the fitting identities. For a target between two characterized
grid locations, linearly interpolate the adjacent probability masses and then
sample uniformly inside the selected bin. Also retain one global
`Uniform(-tau,+tau)` density as a deliberately simpler baseline.

Fit the fourth-order Gaussian surrogate as a diagnostic comparator:

```text
x_programmed_raw = x* + mu(x*) + sigma(x*) * Normal(0, 1),
x_programmed_clipped = clip(x_programmed_raw, 0, 1).
```

Here `mu` is a fourth-order polynomial and `log(sigma)` is a fourth-order
polynomial with a `1e-8` standard-deviation floor, ensuring positive standard
deviation. Fit the raw endpoint on the fitting partition of accepted,
non-corrupt trajectories; report clipping only as an explicit deployment
view. Publish coefficients for each preset, controller, tolerance, and start
protocol rather than pooling incompatible conditions.

Validate every compact surrogate on held-out device identities using:

- conditional bias, standard deviation, MAE, and RMSE;
- 1%, 5%, 50%, 95%, and 99% residual quantiles;
- 90% and 95% predictive-interval coverage;
- target-binned Wasserstein distance;
- skewness and excess kurtosis;
- failure probability and failure class; and
- pulse, verify, and reversal distributions.

Call a compact surrogate adequate only if both 90% and 95% held-out
coverage errors are at most three percentage points and the median
target-binned Wasserstein distance is at most `0.1` held-out residual standard
deviations. If a compact model fails, retain it only as a documented
approximation and use the empirical kernel.

For each non-corrupt identity/repeat, pair the analysis-only persistent
boundary states

```text
L_i = persistent endpoint after lower_to_target conditioning,
U_i = persistent endpoint after upper_to_target conditioning.
```

Classify the exact target independently of controller acceptance:

```text
z_i(x*) = below  if x* < L_i,
          inside if L_i <= x* <= U_i,
          above  if x* > U_i.
```

The wider verify window is structurally reachable only when

```text
x* + tau >= L_i  and  x* - tau <= U_i.
```

This distinction matters because apparent write variation can satisfy the
verify stopping rule even when the exact target—or the entire verify
window—is outside the persistent interval. The compact outcome model is
therefore the mixture

```text
sum_z p(z | x*) p(success | z, x*, controller, start)
      p(endpoint | z, success, x*, controller, start).
```

It is not one residual distribution:

1. sample the corrupt-device identity mask once and preserve it for the life
   of the physical assignment;
2. for a non-corrupt identity, sample `below`, `inside`, or `above` from the
   fit-partition paired-boundary probabilities;
3. sample verify-window reachability and apparent programming success
   conditionally on that persistent class;
4. sample accepted apparent residuals from the bounded density while sampling
   their persistent endpoints from the corresponding class; and
5. sample failed non-corrupt and corrupt apparent/persistent terminal states
   from their separate target- and class-conditioned empirical inverse-CDF
   tables.

IBM's preset implementation collapses a corrupt device's two bounds and sets
both pulse scales to zero. No SET/RESET sequence repairs that persistent
state. Apparent write variation can nevertheless place a verify observation
inside the acceptance window; such a noisy admission must not be interpreted
as persistent reachability. Do not redraw the corrupt mask on a later write.
The compact endpoint sampler is a one-shot deployment marginal: reusing its
mask preserves defect assignment but not a cell's latent stuck value or pulse
history. Repeated fine-tuning writes must continue from the explicit
`IbmReramPlant` state rather than calling the endpoint sampler again.

The characterized target support is `[0,1]`. The deployment sampler rejects an
out-of-support target by default. A caller may request explicit target
clamping, but the returned clamp mask must be retained. Raw terminal endpoints
are likewise preserved unless the caller explicitly selects an analysis or
deployment clipping policy. The deployment path also rejects incomplete fits,
pseudocount-only target bins, and conditions that failed held-out adequacy.
`allow_inadequate=True` is reserved for explicit analysis of a complete fit;
it cannot override missing target or reachability fits.

### Exploratory compact-model check on completed v2 data

The bounded candidate was selected after v2 completed, so this check does not
rewrite the v2 protocol and remains exploratory until the v3 rerun. Re-fitting
the completed fit identities and evaluating the untouched validation
identities gave:

| Compact model | Adequate conditions |
| --- | ---: |
| Fourth-order Gaussian | 0 / 16 |
| One global `Uniform(-tau,+tau)` | 7 / 16 |
| Target-conditioned eight-bin piecewise uniform | 16 / 16 |

For the piecewise-uniform model, pooled 90% coverage was `0.8978-0.9037`, 95%
coverage was `0.9488-0.9511`, and normalized median target-binned Wasserstein
distance was `0.0323-0.0479`. The single uniform principally failed the
one-pulse conditions because it cannot represent target-dependent mass within
the verify window.

The same post-completion check validates the paired-bound reachability
probabilities. Mean absolute fit-versus-held-out class-probability error was
`0.0088-0.0153` across the four populations; the largest individual
target/class difference was `0.0531`. At the nominal endpoints, the held-out
non-corrupt populations were:

| Population | x=0 exact below lower | x=0 full window unreachable | x=1 exact above upper | x=1 full window unreachable |
| --- | ---: | ---: | ---: | ---: |
| OM continuous | 51.5% | 47.1% | 52.9% | 47.5% |
| OM corruption arm, non-corrupt subset | 52.8% | 48.3% | 51.7% | 47.8% |
| HfO2 continuous | 52.5% | 31.9% | 52.5% | 29.9% |
| HfO2 corruption arm, non-corrupt subset | 52.6% | 30.9% | 53.1% | 29.7% |

These are structural bound events, not corrupt devices. In particular, the
HfO2 one-pulse controller apparently accepted about 98-100% of the exact-
unreachable endpoint cases because of its large apparent write variation;
their persistent endpoints did not become equal to the targets.

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
`hardware/devices/om.py` with explicit per-trajectory PyTorch
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

## References

- [AIHWKit 1.1.0 ReRAM preset source](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/presets/devices.py)
- [IBM optimized-material and HfO2 14 nm ReRAM array study](https://research.ibm.com/publications/deep-learning-acceleration-in-14nm-cmos-compatible-reram-array-device-material-and-algorithm-co-optimization)
- [IBM hardware Tiki-Taka study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10811689/)
- [Wan et al. NeuRRAM paper](https://www.nature.com/articles/s41586-022-04992-8)
