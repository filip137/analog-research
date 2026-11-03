# CMO and drift model reference recorded on 3 November 2026

Historical snapshot of the device, drift and initialization sections of
`AGENTS.md` at commit `751c2e1`, before the user moved simulation choices into
the branch responsible for each experiment. The parameters and implementation
status below describe that revision. They preserve the earlier model contract
and its limitations; they do not select a model for a new run.

For a new experiment, adopt the relevant model explicitly in its branch's
experiment or pilot note and lifecycle, and check the current implementation.
The shared scope is [corrupted devices, write noise and drift](../../AGENTS.md).
Code paths below are relative to the repository root. Existing runs retain
their original contracts.

## Corrupt devices

Defects are permanent physical faults defined by the lifecycle `defects` section:
kind (`open`, `gmax`, `random`, and for CMO `gmin`), rate and placement per physical device. All arms of
one assignment share that array's fault map, so they are matched conditions, not
independent arrays. Writes never repair a defect; verify persistence. Corruption
masks used in HWA training are temporary and are never the deployed fault map.
Report results per assignment and defect case; do not pool across defect rates.
CMO faults are placed hierarchically ([CMO device contract](../reram_cmo_device.md)):

- die and array fault rates are drawn from Beta distributions whose
  concentrations set die-to-die and array-to-array variation;
- the fault total is exact per MLP, so scenarios at the same mean differ only
  in how the faults are distributed;
- the stuck value (`kind`) is an experimental variable, with `random` as the
  default.

The rates and their spreads are synthetic, not measured CMO statistics.
Report results per scenario.

## Drift

The standard device model was set by the user on 3 November 2026. It is the
CMO deployment under [Data and initialization](#data-and-initialization),
followed by the power-law drift the user supplied on 28 October
([specification](../../../on-chip-learning/campaigns/drift/power-law-model.md)). One
knob sets the temporal correlation
([25-h evidence](../../../on-chip-learning/campaigns/drift-mitigation/series/001-device-model/results/retention-25h-correlation-20261103/README.md)):

```text
G_p   = CMO endpoint draw for the target, clamped once into 9–88.2 µS (the first read)
u     = τ / 1 h,   τ = seconds since the device's latest write (no programming-to-read delay)
G(τ)  = clip(G_p + m·u^α + s·u^β·Y(τ), 9, 88.2)
Corr(Y(τ1), Y(τ2)) = (τ1/τ2)^κ  for τ1 ≤ τ2,   κ = ln ρ* / ln(5 min / 25 h)
exact step: Y ← c·Y + √(1 − c²)·z,   c = (τ_prev/τ_next)^κ
```

- **Nominal values:** m = −2 µS, s = 3 µS, α = 0.23, β = 0.11 and
  ρ* = Corr(Y(5 min), Y(25 h)) = 0.335 (κ ≈ 0.19), as measured on the 25-h data.
  A sweep of s, the exponents or ρ* is declared in the experiment note, with
  values in µS.
- **Devices and writes:** every physical device drifts, the reference devices
  as well as the active ones. A full rewrite draws a fresh CMO endpoint,
  restarts τ and draws a fresh Y (R1). A persistent per-device share f of Y
  (`rho` in `training/writable_drift.py`) is a declared sensitivity; the
  default is f = 0.
- **What κ affects:** at every age the marginal is the power law, whatever κ
  is, so single-age results do not depend on ρ*. Anything that reads at one age
  and acts at another does. ρ* = 1 (κ = 0, ε fixed for a whole write episode)
  is the model of the runs before 3 November and is now a labelled sensitivity
  case.
- **Refinements:** read noise is folded into s. Separate read noise,
  conductance-binned coefficients, heavy tails, a programming-to-read delay and
  ages past 25 h are declared refinements or labelled extrapolations.
- **Named alternatives:** the flowers fit (`training/flowers_drift.py`,
  supported to 1.7 h), the AIHWKit retention laws in
  `training/inference_retention.py`, other observation bounds and ideal
  programming (G0 = target). Label results under these as such and never pool
  them with the standard model.
- **Information and reporting:** controllers observe only read conductances.
  ε, Y, κ and other hidden draws stay on the analysis side, except in labelled
  oracle arms. Report clipping at both bounds with every result.

**Implementation status:**
- `training/power_law_drift.py` (lifecycle `retention.power_law`) and
  `training/writable_drift.py` implement only κ = 0.
- `writable_drift.py` writes exact endpoints, without the CMO draw.
- Add both before any multi-age, rewrite or learning run under the standard
  model. Keep κ = 0 replays of earlier runs bit-exact, and declare ρ* in each
  lifecycle.
- Paired arms share drift draws only if they advance on the same time grid.
- Both modules are mirrored verbatim in `drn/`; change the two copies together.

## Data and initialization

The OPT experiments use IBM CMO/HfOx ReRAM (`devices.technology: "cmo"`), as
specified in the [CMO device contract](../reram_cmo_device.md):

- the Falcone et al. (2025) 2 % program-and-verify fit as released in AIHWKit
  1.1.0 `ReRamCMONoiseModel`, over 9–88.2 µS;
- two devices per weight, read as `q=a-r`: an active device, and a reference
  device programmed to mid-range;
- either device can be corrupted;
- one unclamped Gaussian endpoint per device at deployment, held afterwards;
- HWA injects Gaussian weight noise on every minibatch: `hwa.noise_std`, per
  matrix or per output channel (the Analog Foundation Models convention),
  declared per experiment;
- corruption-aware HWA arms add temporary i.i.d. stuck active and reference
  devices;
- the `cmo_idealized_write_adam` on-chip law: one noisy write per active
  device and step on the held state, with no verify and no retry, so write
  errors persist.

HWA training decisions — noise model, magnitude and scaling, weight clipping
and mapping, corruption-aware masks, objective and distillation, training data
and token budget, learning-rate schedule and checkpoint selection — consult the
two reference papers in [`hwa_papers/`](../../hwa_papers/README.md): Rasch et al.
2023 (arXiv:2302.08469) and Büchel et al. 2025, *Analog Foundation Models*
(arXiv:2505.09663). The experiment note states which choices follow them and
which deviate, and why.

AIHWKit's CMO drift and read-noise models stay disabled; drift follows the
[standard drift model](#drift). IBM optimized-material
(OM) ReRAM (`"om"`: literal AIHWKit cells, pulse P&V and pulse laws) and
Gaussian PCM (`"pcm"`) remain supported, separately declared regimes.
Label results exploratory and model-based, not fabricated-hardware measurements. Follow
[synapse-data provenance rules](../synapse_data.md) for measured evidence; do not
substitute datasets, preprocessing, tokenization or cohorts between matched arms.

Predeclare initialization. HWA and non-HWA arms start from the same logical state;
recovery arms start from the same named deployment unless initialization is the
intervention. Use explicit paths, revisions and hashes, never the newest checkpoint.
Strict configs and campaign lifecycles define inputs; historical
[initialization protocols](../initialization_protocols.md) provide lineage only.
