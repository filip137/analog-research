# CMO/HfOx ReRAM device contract (2 % program-and-verify)

**Status: active.** IBM CMO/HfOx ReRAM is the first device for the OPT-125M
crossbar experiments (decided 2026-10-08). A lifecycle selects it with
`devices.technology: "cmo"`. This document states the device law as
implemented. The [lifecycle contract](crossbar_lifecycle.md) states how the
device enters the stages.

## Source and evidence class

- **Characterization:** Falcone et al. (2025),
  [arXiv:2502.04524](https://arxiv.org/abs/2502.04524). They program CMO/HfOx
  devices by program-and-verify (P&V) to targets of about 10–90 µS.
- **Fitted law:** AIHWKit 1.1.0 `ReRamCMONoiseModel`
  ([source](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/reram.py))
  and the range defaults of `SingleDeviceConductanceConverter`. The
  coefficients below were checked against the installed AIHWKit 1.1.0, both
  as numbers and as sampled spreads.
- **Evidence class:** deployment uses a hardware-derived fitted model. The
  on-chip write law (below) applies the same fit to every write, which makes
  it an idealization, not a measured write model. Off-chip HWA does not use
  the fit at all: it injects a declared constant-σ Gaussian weight noise.
  Defect rates and their die/array variation are synthetic; no CMO
  wafer-level failure distribution has been measured (Falcone et al.
  characterized an 8×4 array). Label every result exploratory and
  model-based, never a fabricated-hardware measurement
  ([synapse-data rules](synapse_data.md)).

## Lifecycle declaration

```json
"devices": {"technology": "cmo", "model": "aihwkit_1.1.0_reram_cmo", "noise_scale": 1.0, ...},
"defects": {"placement": "hierarchical_die_array", "persistence": "permanent", "arrays_per_die": 48,
            "cases": [{"kind": "none", "rate_ppm": 0, "die_concentration": null, "array_concentration": null},
                      {"kind": "random", "rate_ppm": 50000, "die_concentration": 118, "array_concentration": 118}]},
"deployment": {"encoding": "active_reference_pair", "tile_size": 256, "read_noise": 0.0, ...},
"hwa": {"noise_model": "additive_gaussian_weight", "noise_std": 0.03, "noise_scaling": "output_channel_abs_max", ...},
"program_verify": {"method": "gaussian_endpoint", "acceptance_percent": 2.0,
                   "relaxation": {"model": "none"}},
"onchip": {"update_law": "cmo_idealized_write_adam", ...}
```

Start from [`examples/lifecycles/opt125m-cmo-mlp4-template.json`](../examples/lifecycles/opt125m-cmo-mlp4-template.json).
CMO is analytic, so it needs no `EBL_AIHWKIT_PYTHON` interpreter and no
`devices` populations.

## Device law

| Quantity | Value |
| --- | --- |
| Conductance range | G_min = 9.0 µS, G_max = 88.199997 µS (AIHWKit default; often rounded to 88.19 or 88.2) |
| Programming error | G = G\* + noise_scale · σ_PV(G\*) · ξ, with ξ ~ N(0, 1) independent per device |
| 2 % acceptance (baseline) | σ_PV(G) = 0.0112185391 + 0.01129027418 · G µS; σ_PV(50 µS) = 0.576 µS |
| 0.2 % acceptance (declarable) | σ_PV(G) = 0.00081107 + 0.00106879 · G µS |
| Device correlation | independent |
| Quantization | none: no 35-level codebook |

AIHWKit pitfall: `ReRamCMONoiseModel()` defaults to `acceptance_range=2e-2`.
That is not a key of its coefficient table `{0.2, 2}`, so the model silently
falls back to the **0.2 %** fit. To reproduce the 2 % law in AIHWKit, pass
`acceptance_range=2`. Here the acceptance range is always explicit
(`program_verify.acceptance_percent`), and the on-chip writes use the same σ law
as deployment.

### Encoding

Each logical weight uses two physical CMO devices (`active_reference_pair`):
an **active** device and a **reference** device. This is the same `q = a − r`
read as the OM technology, but here the reference is a real programmed device.
Each analog matrix is scaled to q in [-1, 1], either by its own absolute
maximum (`per_layer_absmax`) or by each output channel's absolute maximum
(`per_output_channel_absmax`, the CMO template default). On OPT-125M,
per-channel scaling matters. A typical channel's largest weight is only
9–25 % of its matrix's outlier maximum, so under per-matrix scaling the CMO
error alone is 0.4–1.0× each channel's weight RMS; per-channel scaling
brings it to 0.07–0.11×. Then:

- the active device is programmed to G_a\* = 48.6 + 39.6 · q µS, so
  q = -1, 0 and 1 map to 9, 48.6 and 88.2 µS;
- the reference device is programmed to mid-range, G_r\* = 48.6 µS;
- the read is q̂ = (G_a − G_r) / 39.6;
- both devices carry their own programming error, and either may be stuck;
- `mapping.json` records two physical devices per weight.

The read error of a weight is √(σ_PV(G_a\*)² + σ_PV(48.6)²) / 39.6. For a zero
weight that is 0.020 of full scale; for q = +1 it is 0.029. The reference is
programmed once at deployment and is never written on-chip. Its error is
therefore a fixed per-weight offset that on-chip learning can compensate
through the active device.

In AIHWKit terms, this is the single-device converter on a symmetric range,
with the converter's digital offset replaced by a physical reference
device.

## Deployment (P&V)

- Each physical device gets one draw at deployment, from its matrix's own
  stream: `binding_seed(endpoint_seed, "facebook/opt-125m", matrix)`.
  Identities do not depend on layer order.
- The Gaussian is unbounded. Results are **not truncated** to the ±2 %
  acceptance window and **not clamped** to [G_min, G_max]: about half of the
  devices at q = ±1 land outside the range. The P&V report records
  `out_of_range_devices`, `device_programs` (active and reference) and
  `verify_reads` (0, because P&V is an endpoint model).
- The realized conductances are held. Every forward (gradients, metrics,
  selection, final evaluation) reads the same state until a write changes it.
  Noise is never redrawn per forward or per example.
- The generator runs on the execution device. CPU and CUDA runs of one
  lifecycle therefore realize different endpoints, as with PCM.

## HWA (`additive_gaussian_weight`)

Off-chip HWA injects plain Gaussian noise into the normalized digital master,
fresh every minibatch:

```text
q̃[j, i] = q[j, i] + noise_std · noise_strength · s_j · ξ,   ξ ~ N(0, 1) per weight
```

- `hwa.noise_scaling` sets s_j:
  - `output_channel_abs_max`: s_j = maxᵢ |q[j, i]|, the current abs-max of
    output channel j. This is eq. 3 of the Analog Foundation Models paper
    (Büchel et al. 2025), and the template default at `noise_std` 0.03.
  - `matrix_full_scale`: s_j = 1, one σ for the whole matrix, as in the dense
    CIFAR-10 HWA.
- The noise does not depend on the conductance or on the CMO programming
  law. It is a literature-motivated augmentation, not the device law.
- `hwa.noise_std` and `hwa.noise_scaling` are required, and each experiment
  note declares them.
- Deviations from the Analog Foundation Models recipe:
  - the backward pass goes through the noisy weights (the AIHWKit modifier
    convention), not the noise-free weights;
  - there is no α·std weight clipping;
  - there is no DAC/ADC quantization.
- Each arm's `noise_strength` multiplies `noise_std`, and 0 trains on the
  clean weights.
- The noisy weights are not clamped. The gradient is straight-through, and
  the master is clamped to [-1, 1] after every step.
- Checkpoints are still selected on separate selection arrays programmed with
  the real CMO P&V.
- **Corruption-aware arms** (`corruption: {"kind": ..., "rates_ppm": [...]}`):
  after the noise, every active and every reference device is stuck i.i.d. at
  the drawn rate, with the `kind`'s stuck value (`random`: fresh uniform
  values). The weight is then read as a pair:
  - a stuck active device reads its stuck value;
  - a stuck reference shifts the weight.

  These masks are temporary, Bernoulli per device and redrawn every
  minibatch. They are never a deployed fault map, which is hierarchical with
  exact totals.
- Selection cases may include defect cases, which are drawn on the selection
  arrays.

## Defects (`hierarchical_die_array`)

Stuck devices are permanent and are placed hierarchically. The die-to-die
and array-to-array variation are independently controllable, while the mean
fault fraction and the exact fault count stay fixed.

**Geometry.**
- A physical array has `deployment.tile_size` × `tile_size` devices.
- A weight's active and reference devices sit on adjacent rows of the same
  array, as differential pairs do in NeuRRAM. So one array holds tile_size/2
  inputs × tile_size outputs of one deployment tile, and each tile spans two
  arrays.
- Arrays follow the mapping order (fc1 tiles, then fc2 tiles) and fill dies
  of `defects.arrays_per_die` within each **group**. A group is the module
  that owns the matrices, here one OPT decoder layer's MLP. Dies never
  straddle groups, and the last die of a group may be partial.
- With 256×256 arrays, one OPT-125M MLP is 144 arrays: 72 for fc1 and 72 for
  fc2. At 48 arrays per die that is three dies per MLP, and a die holds about
  3.1 M devices.
- A weight's two devices share their array's and die's fault rates and are
  sampled individually under them.

**Hierarchy.** For one defect case with mean μ = `rate_ppm` / 10⁶:

```text
p_d          ~ Beta(μ κ_d, (1 − μ) κ_d)               die rate        (κ_d = die_concentration)
p_da | p_d   ~ Beta(p_d κ_a, (1 − p_d) κ_a)           array rate      (κ_a = array_concentration)
```

A concentration of `null` means infinity: the rate equals its parent value.
The standard deviation of a rate around a mean μ is √(μ(1 − μ)/(κ + 1)). At
μ = 5 %, κ = 1899, 474, 118 and 52 give 0.5, 1, 2 and 3 percentage points.

**Exact totals.** Exactly round(μ · N) of a group's N physical devices fail,
active and reference alike. The
counts are allocated to the arrays multinomially, in proportion to p_da times
the array size (capped at the array size). Devices are then chosen uniformly
within each array. Independent placement (both concentrations `null`)
therefore varies only by sampling noise: about 0.085 percentage points for
256×256 arrays at 5 %.

**Stuck value** is the case's `kind`, an experimental variable:

| `kind` | Stuck conductance | Stuck active device | Stuck reference device |
| --- | --- | --- | --- |
| `random` (default) | uniform in [9, 88.2] µS, drawn once per device | q̂ uniform in [-1, 1] | weight shifted by a uniform value in [-1, 1] |
| `gmin` | 9 µS | q̂ = -1 | weight shifted by +1 |
| `gmax` | 88.2 µS | q̂ = +1 | weight shifted by -1 |
| `open` | 0 µS (open circuit) | q̂ = -1.23 | weight shifted by +1.23 |

The read values assume the other device of the pair is at its nominal
target. When both devices are stuck at the same value, the weight reads 0.

**Scenarios.** The four predeclared scenarios at a 5 % synthetic mean are
the defect cases of the template. Their labels are the kind, the rate, then
`d<κ_d>` and `a<κ_a>` for the finite levels:

| Case | κ_d | κ_a | Label |
| --- | --- | --- | --- |
| A, independent | ∞ | ∞ | `random_50000ppm` |
| B, array variation | ∞ | 118 | `random_50000ppm_a118` |
| C, die variation | 118 | ∞ | `random_50000ppm_d118` |
| D, both | 118 | 118 | `random_50000ppm_d118_a118` |

At OPT-125M scale (four MLPs, 576 arrays, 37.7 M devices), the realized
spread of array fault fractions was 0.09, 1.9, 2.1 and 3.1 percentage points
for A–D. Exactly 5 % of each MLP's devices failed, split about evenly between
active and reference devices.

**Identity and persistence.**
- Each group draws from its own NumPy stream, `binding_seed(assignment,
  model, "<group>/faults")`, split into independent die, array, count,
  device and value streams. A group's defects therefore do not depend on
  which other layers are analog, or on the execution device.
- Scenarios of one assignment reuse the same streams. They are matched
  conditions on one array identity, not independent arrays.
- Stuck devices hold their stuck conductance through every program and
  write, and are never programmed or written. Requests addressed to them are
  counted as `stuck_write_attempts`, and the runtime verifies after every
  epoch that they did not change.
- Healthy devices keep the same programming-noise stream as in the
  defect-free case.
- The array identity (the `identity` field of every deployment and
  measurement row) records, per group, every die and array rate and count,
  plus the identity hash of the fault map.
- Saved states store only that hash and are checked against the map rebuilt
  from the seeds.
- An array programmed once and never written (a deployment, or an arm that
  never writes) is saved compactly, as fault and apparent-state hashes. It is
  rebuilt by reprogramming its target on the same execution-device type, then
  verified.

## On-chip training (`cmo_idealized_write_adam`)

Each step is one idealized write per active device, applied to the held
state. Reference devices are never written.

```text
Δq   = Adam(∂L/∂q̂ at the held apparent state)          (digital, normalized units)
ΔG   = 39.6 · Δq
G_a  ← G_a + ΔG + noise_scale · σ_PV(G_a + ΔG) · ξ       (healthy active devices with ΔG ≠ 0)
```

- There is no digital master: the held conductance carries every earlier
  write error.
- There is no verify read and no retry. A write that lands outside the
  acceptance window stays there.
- Errors accumulate. With dense Adam updates, a device written n times
  random-walks by about √n · σ_PV, roughly 0.56 µS per write at mid-range.
- A zero learning rate, and a device whose command is exactly zero, write
  nothing.
- The requested G + ΔG is not clamped (see the open decisions below).
- An optional `onchip.write_noise_scale` f multiplies the write term to
  f · noise_scale · σ_PV(G_a + ΔG) · ξ. f = 0 is an exact-write oracle used to
  diagnose write noise, not a device model; programming noise is unchanged
  ([lifecycle contract](crossbar_lifecycle.md)).
- The `rewrite` arm reprograms the active devices to their frozen deployment
  targets on the same schedule. That is the same law with ΔG = G_a\* − G_a,
  which gives G_a\* + σ_PV(G_a\*) · ξ on every step. References are not
  rewritten, so the control matches the learning arms' writes.
- The `calibration` arms train digital parameters only and never write.

The cost of each epoch is reported per assignment and arm:

- `device_writes` (learning) and `device_programs` (rewrite), counting
  healthy active devices only;
- `stuck_write_attempts`;
- `touched_devices` and `max_onchip_device_writes`;
- `verify_reads` (0);
- `out_of_range_devices` (healthy devices) in the final state.

## Not modelled in this initial regime

| Effect | Status |
| --- | --- |
| Hierarchical corruption during HWA | not implemented; training masks are i.i.d. per device |
| Spatial correlation inside an array (row- or column-structured faults) | not modelled; devices fail uniformly within an array |
| Placement sensitivity (fault patterns reassigned to other layers) | not implemented |
| Drift / conductance relaxation | disabled (`relaxation: none`); AIHWKit's CMO decay fit is not applied |
| Read noise | disabled (`deployment.read_noise: 0.0` in the template); AIHWKit's CMO read term is not applied |
| Discrete levels, IR drop, ADC/DAC quantization | not modelled |

## Open decisions

- **Out-of-range conductances.** Deployment and on-chip results are
  unclamped and counted. A physical bound or saturation rule must be declared
  before the first claim that depends on it.
- **Drift.** The planned next stage applies drift to the same held
  conductances, with no P&V repeated before post-deployment retraining. It
  needs its own `relaxation` model.
- **Measured defect statistics.** The 5 % mean and the 2-percentage-point
  die and array spreads are synthetic reference points. Replace them when
  CMO wafer-level data exist.

## Code and tests

- `training/reram_cmo.py`: the law, the encoding and `CmoMatrix`.
- `workflow/devices.py`: `CmoArray`, a layer-bound array, and
  `hierarchical_faults`, the die/array fault sampler.
- `workflow/program_verify.py`: the `gaussian_endpoint` report.
- `workflow/hwa.py`: the constant-σ HWA weight noise.
- `workflow/onchip.py`: `IdealizedWriteAdam`.
- `workflow/lifecycle.py`: `TECHNOLOGIES["cmo"]`, `CMO_ACCEPTANCE_PERCENT`,
  `hwa.noise_std`, and the hierarchical defect cases with `arrays_per_die`.

`tests/test_workflow_cmo.py` checks:

- the coefficients, with a cross-check against AIHWKit that is skipped when
  AIHWKit is not installed;
- the active-minus-reference read and the error of both devices;
- unclamped, untruncated draws and held reads;
- the exact write law and persistence of write errors;
- exact state replay;
- stuck devices persisting through programs and writes;
- exact fault totals per group, with dies kept inside a group;
- the array and die spreads of the four scenarios;
- the stuck value of each kind;
- deterministic, group-bound fault maps;
- that HWA noise has one σ (`noise_std` × `noise_strength`) at every weight
  value, or scales with each output channel's abs-max, with a
  straight-through gradient;
- corruption-aware HWA on both devices of a pair;
- a deployment-only corruption-aware lifecycle with compact deployments and
  `collect` deployment rows;
- schema rejections;
- that a weight's two devices share one physical array;
- a complete tiny-OPT lifecycle with a defect-free and a scenario-D case,
  all five arms, exact replay, `collect`, and the storage estimate (at most
  35 bytes per analog weight and trajectory).
