# IBM ReRAM pulse-count program-and-verify noise model

- Status: short CUDA production protocol reviewed and sized; production evidence pending
- First milestone: device characterization and Wan-2022 comparison
- Reference runtime: AIHWKit 1.1.0

## Campaign records

The study narratives and current interpretations now live in the following campaigns. Original dates/status text in the retained preamble is historical; the sections below remain shared technical reference.

- [Original exhaustive controller design](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-001-exhaustive-design.md) — [results and interpretation](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/results/exp-001-exhaustive-design.md)
- [Short-production programming and endpoint model](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-002-short-production.md) — [results and interpretation](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/results/exp-002-short-production.md)

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

## References

- [AIHWKit 1.1.0 ReRAM preset source](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/simulator/presets/devices.py)
- [IBM optimized-material and HfO2 14 nm ReRAM array study](https://research.ibm.com/publications/deep-learning-acceleration-in-14nm-cmos-compatible-reram-array-device-material-and-algorithm-co-optimization)
- [IBM hardware Tiki-Taka study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10811689/)
- [Wan et al. NeuRRAM paper](https://www.nature.com/articles/s41586-022-04992-8)

## Moved study sections

<a id="objective"></a>

[objective](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-001-exhaustive-design.md#source-ibm_reram_program_verify_noise_model-objective)

<a id="characterization-design"></a>

[characterization design](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-001-exhaustive-design.md#source-ibm_reram_program_verify_noise_model-characterization-design)

<a id="exploratory-compact-model-check-on-completed-v2-data"></a>

[exploratory compact model check on completed v2 data](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/results/exp-002-short-production.md#source-ibm_reram_program_verify_noise_model-exploratory-compact-model-check-on-completed-v2-data)

<a id="wan-2022-comparison"></a>

[wan 2022 comparison](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-001-exhaustive-design.md#source-ibm_reram_program_verify_noise_model-wan-2022-comparison)

<a id="reproducibility-and-artifacts"></a>

[reproducibility and artifacts](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-001-exhaustive-design.md#source-ibm_reram_program_verify_noise_model-reproducibility-and-artifacts)

<a id="implemented-execution-surface"></a>

[implemented execution surface](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-001-exhaustive-design.md#source-ibm_reram_program_verify_noise_model-implemented-execution-surface)

<a id="completion-criteria"></a>

[completion criteria](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-001-exhaustive-design.md#source-ibm_reram_program_verify_noise_model-completion-criteria)

<a id="deferred-matched-training-study"></a>

[deferred matched training study](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-001-exhaustive-design.md#source-ibm_reram_program_verify_noise_model-deferred-matched-training-study)
