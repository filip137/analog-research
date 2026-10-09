# ReRAM device catalog

Last verified: 2026-07-30 against AIHWKit 1.1.0.

## Short answer

There are two different ReRAM APIs in AIHWKit, and they should not be treated
as interchangeable:

1. `aihwkit.inference.noise.reram` contains two inference/deployment models:
   `ReRamWan2022NoiseModel` and `ReRamCMONoiseModel`. These use physical
   conductances in microSiemens (`uS`).
2. `aihwkit.simulator.presets.devices` contains four ReRAM pulsed-update
   device models: `ReRamESPresetDevice`, `ReRamSBPresetDevice`,
   `ReRamArrayOMPresetDevice`, and `ReRamArrayHfO2PresetDevice`. Their
   `w_min` and `w_max` values are normalized device states, not physical
   conductances in `uS`.

If "from `reram`" means the literal
`aihwkit.inference.noise.reram` module, only Wan-2022 and CMO are in scope.
The table also includes IBM's separately released PCM endpoint model because
it is now one of the repository's matched deployment/recovery devices.

## Inventory and repository support

| Model or device | What it represents | Native range | First role in this repository | Current repository access |
| --- | --- | --- | --- | --- |
| `ReRamWan2022NoiseModel` | Wan-2022 inference-time programming/retention variation | AIHWKit default target range: `0` to `25 uS`; this repository normally selects `0` to `40 uS` | HWA/deployment side | Supported as `aihwkit_reram_wan2022` device programming |
| IBM Analog Foundation Models PCM | Hardware-derived endpoint weight-programming error released with the IBM AFM work | Each output channel/tile is normalized to a unitless fit maximum of `180`; mapped values below `0.6` become zero. This is not an absolute conductance range. | HWA/deployment side | Supported as `ibm_afm2025_pcm`, including repeated noisy writes |
| `ReRamCMONoiseModel` | Measured CMO/HfOx programming, relaxation, and read variation | `9` to `88.199997 uS` | HWA/deployment side | Supported as `aihwkit_reram_cmo`, including repeated noisy writes |
| `ReRamESPresetDevice` | Nonlinear exponential-step pulsed device fitted to Gong et al. (2018) | Nominal normalized state `[-1.0, 1.0]`; no built-in `uS` calibration | Tiki-Taka | Supported through `TikiTakaReRamESPreset` |
| `ReRamSBPresetDevice` | Simpler soft-bounds/reference approximation to the same Gong et al. data | Nominal normalized state `[-0.75, 1.25]`; no built-in `uS` calibration | Tiki-Taka | Supported through `TikiTakaReRamSBPreset` |
| `ReRamArrayOMPresetDevice` | Measured 14 nm ReRAM array, optimized-material variant | Nominal normalized state `[-1.0, 1.0]`; no built-in `uS` calibration | Neither | Available in AIHWKit, but not in this repository's preset allowlist |
| `ReRamArrayHfO2PresetDevice` | Measured 14 nm ReRAM array, baseline-HfO2 variant | Nominal normalized state `[-1.0, 1.0]`; no built-in `uS` calibration | Neither | Available in AIHWKit, but not in this repository's preset allowlist |

Here, "HWA/deployment side" needs one qualification: the repository's HWA
training uses additive Gaussian parameter perturbations. Wan-2022 and CMO are
applied after ordinary or HWA training to create persistent deployed-device
realizations; they are not the distributions used by the HWA modifier itself.

## Understanding the ranges

### Inference models: physical conductance

The inference models accept target conductances in `uS`.

`ReRamWan2022NoiseModel` defaults to AIHWKit's
`SinglePairConductanceConverter`, whose device range is `0` to `25 uS`.
However, the built-in Wan polynomial coefficients were fitted with a
`40 uS` reference. This repository explicitly passes `g_max_us=40`, so the
range used in the completed Wan studies is `0` to `40 uS`, not the library
default. The configured range is a target mapping: the Wan noise model clamps
negative samples to zero but does not impose `g_max` as a hard upper bound on
each noisy sample. The repository subsequently clamps the mapped result to
the DRN parameter bounds.

The repository maps a nonnegative DRN conductance `G_drn` into Wan device
units as

```text
G_device_uS = G_drn * g_max_us / drn_conductance_at_g_max
```

and reverses that scale after applying the device realization. The common
MNIST HWA study used `g_max_us=40` and
`drn_conductance_at_g_max=1.0`.

`ReRamCMONoiseModel` defaults to a single-device converter with a physical
range of `9` to `88.199997 uS`. Unlike Wan, it therefore has a substantial
nonzero conductance floor. For the current unit DRN reference, the strict
model exposes three explicit mappings:

| Mapping | Physical target | Effective DRN value |
| --- | --- | --- |
| `literal_conductance` | `clip(88.199997 w, 9, 88.199997) uS` | nominally `max(w, 0.1020408)` |
| `affine_floor` | `9 + (88.199997-9)w uS` | nominally `0.1020408 + 0.8979592 w` |
| `normalized_offset` | same target as `affine_floor` | `w`, after subtracting and rescaling the floor |

Both literal and affine mappings retain the physical minimum connection.
Only `literal_conductance` collapses every learned target below the floor to
one value. `affine_floor` preserves ordering and uniformly scales pairwise
differences while leaving the floor in the passive DRN equations.
`normalized_offset` remains available only as an explicitly labelled
common-mode-cancellation comparison.

On the full-MNIST HWA checkpoint, literal mapping raises `98.13%` of W1 and
`88.10%` of W2 targets to `9 µS`, lowering accuracy from `96.37%` clean to
`80.61%` on the first write. Full noisy BPTT recovers to `89.73%`; rank-4
positive-only LoRA remains at `80.56%` because every factor target stays below
the same physical floor. In contrast, the matched noisy `affine_floor` probe
retains `9 µS`, clips no targets, and obtains `96.12%` on its first write.
Thus the literal collapse is dominated by many-to-one hard clipping, not the
presence of the floor alone.

The IBM Analog Foundation Models PCM helper is also normalized: in native
PyTorch layout, the absolute maximum of every output channel is mapped to
`180` in the helper's fit coordinate. With a maximum input-tile size, each
channel/tile receives its own scale. The released model then applies one of
two cubic Gaussian standard-deviation fits, switching at `52.5`, and
hard-zeros mapped magnitudes below `0.6`. The code does not assign a physical
unit to these constants. Consequently, 180 is not a global physical
calibration for the DRN, and Gaussian effective values can fall outside the
nominal interval before the repository applies the DRN parameter bounds.

### Pulsed devices: normalized state

The four pulsed devices use abstract signed state bounds. A negative state is
not a negative physical conductance; it represents an effective signed weight,
which a physical circuit can implement with a reference or differential
conductance. AIHWKit does not attach a `uS` scale to these presets.

For the two supported Tiki-Taka presets, the repository reads the realized
slow-device bounds for every crosspoint and applies the affine mapping

```text
G_drn = G_min
        + (q - q_min) / (q_max - q_min)
        * (G_max - G_min)
```

where `q` is the AIHWKit state and `G_min`/`G_max` are repository
conductance bounds. The checked-in Tiki-Taka examples use
`[1e-7, 1.0]` in DRN conductance units. That interval is not automatically a
physical range in siemens; it becomes one only if the experiment supplies a
real calibration in those units.

The table lists nominal `w_min` and `w_max`. Actual crosspoints can have
different realized bounds because all four pulsed presets include
device-to-device bound variation.

## IBM source-paper figure audit

IBM does publish non-normalized pulse-conductance traces, including data
collected specifically for Tiki-Taka. The limitation is narrower: the
AIHWKit training presets generally retain a normalized device state rather
than a traceable conversion back to the physical measurements.

| Source | Figure-level physical conductance evidence | Training context and limitation |
| --- | --- | --- |
| [Gokmen and Haensch (original Tiki-Taka, 2020)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00103/full) | No absolute conductance window is supplied. The statistically most likely simulated device has logical weight bounds of approximately `[-0.6, 0.6]` and a minimum update of `1e-3` at its symmetry point. | These are normalized/logical weight units, not siemens. Device-to-device parameters are Gaussian-distributed, and each pulse receives an additional 30% Gaussian cycle-to-cycle update variation. |
| [Gokmen (TTv2, 2021)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.699148/full) | No absolute `µS` range is supplied. The generic array model uses a nominal logical range of `[-0.6, 0.6]`, virtually remapped to `[-2, 2]` for its LSTM simulations. | The pulse-update model and its multiplicative/additive Gaussian noise are defined in logical weight units. It therefore cannot provide physical `G_min`, `G_max`, or `delta_G` without an external calibration. |
| [Gong et al. (2018)](https://www.nature.com/articles/s41467-018-04485-1) | Figures 2a and 3a plot a representative HfO2 ReRAM pulse trace on a `30` to `60 µS` axis; the measured trace spans roughly `33` to `57 µS`. | This predates Tiki-Taka and supplied device-characterization data later represented by normalized ES/SB-style presets. The plotted excursion is not declared as a universal pair of bounds. |
| [Stecconi et al. (2022)](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aelm.202200448) | Figure 9b/c explicitly reports a TaOx/HfO2 pulse-conductance window of `60` to `240 µS`. | The paper's AIHWKit simulation used a `PowStepDevice` with standard SGD; it states that Tiki-Taka simulations were future work. This is usable physical pulse evidence, but it is not an original Tiki-Taka dataset. |
| [Gong and Rasch et al. (IEDM 2022)](https://research.ibm.com/publications/deep-learning-acceleration-in-14nm-cmos-compatible-reram-array-device-material-and-algorithm-co-optimization) | The official open IBM record confirms statistics extracted from as many as 2,000 6T1R devices, but the full four-page paper is paywalled and the open record does not expose a physical conductance axis from which exact bounds can be verified. | This work demonstrated TTv2 and underlies the OM/HfO2 array presets. Those AIHWKit presets expose normalized bounds, so no physical `µS` calibration should be inferred from them. |
| [Athena et al. (ReSta, 2023)](https://research.ibm.com/publications/resta-recovery-of-accuracy-during-training-of-deep-learning-models-in-a-14-nm-technology-based-reram-array) | Figure 1c plots a representative 14 nm HfOx device in resistance, about `18` to `27 kΩ`; taking `G = 1/R` gives an inferred conductance excursion of about `37` to `56 µS`. The figure is openly reproduced as Figure 7.1 in the [first author's dissertation](https://repository.gatech.edu/entities/publication/a1245ddc-fc35-4616-8421-f87c62ebc0f5). | This is physical pulse data from a hardware TTv2 study. The companion panel converts it to normalized conductance, and the representative excursion is not an array-wide fixed range. |
| [Stecconi et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10811689/) | Figure 2b is a physical Tiki-Taka programming trace spanning about `20` to `115 µS` for one Gen2 device under the stated pulse protocol. Figure 2d separately shows retained states around `8`, `40`, and `150 µS`. Figure 4d reports a distribution of *window widths* of roughly `40` to `150 µS`, not absolute `G_min`/`G_max` values. | This is the clearest IBM Tiki-Taka-specific physical dataset. However, Figure 4a fits the device in arbitrary normalized conductance units, and the different Figure 2 panels use different tests; their extrema must not be combined into one universal operating range. |
| [Rasch et al. (2024)](https://www.nature.com/articles/s41467-024-51221-z) | The algorithm paper defines physical `g_min` and `g_max` symbolically but does not assign an absolute `µS` window to its generic device model. | Its Methods section explicitly says that physical conductance is arbitrarily normalized for the simulations. It is strong evidence for the calibration limitation, not a source of physical bounds. |
| [Büchel et al. (Analog Foundation Models, 2025)](https://arxiv.org/abs/2505.09663) | Figures 7a and 9 plot signal-to-noise ratio and programming error against **normalized** conductance or weight. They do not expose an absolute physical `G_min`/`G_max` axis. IBM's [released PCM helper](https://github.com/IBM/analog-foundation-models/blob/4f4c505b9d9af43af289788b3a29eeecaeb4e1ab/apply_noise_model.py) supplies `GMAX=180` and a `0.6` zero threshold but does not label either in siemens. | The paper establishes the error shape and its Hermes-chip provenance, while the code supplies the unitless numerical mapping used here. Per-channel/tile normalization prevents one global DRN-to-conductance calibration. |
| [Le Gallo et al. (IBM Hermes PCM chip, 2023)](https://arxiv.org/abs/2212.02872) | Extended Data Table I reports a nominal `100 kΩ–10 MΩ` resistance range per PCM device, equivalent to `0.1–10 µS`. Figure 2 and the Methods instead express reliable unit-cell maxima as 80 ADC counts for one-device programming and 160 ADC counts for two-device programming. | This supplies physical per-device evidence for the hardware underlying the AFM fit, but no published conversion equates the helper's 180 coordinate with µS. A unit cell contains four devices and its usable maximum also depends on ADC-current constraints. |
| [Falcone et al. (2025)](https://arxiv.org/abs/2502.04524) | Figure 3 and Supplementary Figure S5 program 35 target levels across approximately `10` to `90 µS`; Supplementary Figure S7 labels per-device `G_min`, `G_max`, and symmetry points for the open-loop traces. IBM's [CMO-ReRAM model documentation](https://aihwkit.readthedocs.io/en/v1.0.0/reram_inference.html) characterizes the corresponding inference range as `8` to `90 µS`. | The paper contains physical data for both training and inference, but its training-device equations again use a normalized `[-1, 1]` conductance state. |

The practical conclusion is that physical, non-normalized pulse-update curves
do exist. What is not currently available through the presets is a
machine-readable, one-to-one calibration carrying the original `G[n]` values,
pulse polarity/amplitude/duration, and per-device bounds into the simulator.
The physical windows also change with pulse conditions, cycling history, and
device-to-device variation, so a single range copied from a figure would be an
incomplete calibration.

For an absolute-conductance Tiki-Taka pulse model, Stecconi et al. (2024) is
the strongest published starting point because its Figure 2b directly records
the physical pulse sequence intended to emulate Tiki-Taka updates. Falcone et
al. (2025) provides the strongest physical CMO/HfOx inference calibration.
Neither is presently a ready-made AIHWKit pulsed preset that preserves the
source conductance scale.

## Device details

### Wan-2022 ReRAM

- AIHWKit class: `ReRamWan2022NoiseModel`
- Representation: one positive/negative conductance pair per signed weight by
  default; this repository bypasses signed conversion and applies the model
  directly to nonnegative DRN conductances.
- Physical target range: `0` to `25 uS` by AIHWKit default, or `0` to
  `40 uS` in this repository's completed studies.
- Supported retention points with the built-in fit: 1 second, 1 day, and
  2 days.
- Model behavior: conductance-dependent Gaussian variation fitted from
  Wan et al. data; no separate 1/f read-noise model.
- Repository role: deployment after ordinary or HWA training, including the
  LoRA recovery studies.

The strict configuration and mapping are implemented in
[`model/resistive/digital_low_rank_config.py`](../model/resistive/digital_low_rank_config.py)
and
[`training/device_programming.py`](../training/device_programming.py).

### CMO/HfOx ReRAM

- AIHWKit class: `ReRamCMONoiseModel`
- Representation: one bidirectional device per crosspoint.
- Physical range: `9` to `88.199997 uS`.
- Model behavior: program-and-verify programming variation plus
  time-dependent conductance relaxation and read variation.
- Repository role: deployment of ordinary and HWA checkpoints, followed by
  full BPTT or rank-4 LoRA recovery with a noisy program-and-verify write
  after every minibatch.
- Standard settings: `0.2%` acceptance fit, one-second inference time, and
  full programming/read/drift scale. Mapping is an explicit experimental
  choice: `affine_floor` is the floor-retaining calibrated transfer,
  `literal_conductance` is the absolute-scale hard-clipping stress test, and
  the first archived matched study used `normalized_offset`.
- Scope relative to Falcone et al.: their reported MVM simulation omitted
  read noise and included input/output quantization plus wire IR drop. The
  repository enables the AIHWKit class's read term but omits those two
  circuit-level effects.

In the crossbar lifecycle, CMO is the working device for the OPT experiments
(`devices.technology: "cmo"`, 2 % P&V, an active and a mid-range reference
device per weight); see the [CMO device contract](reram_cmo_device.md).
AIHWKit's default `acceptance_range=2e-2` is not a coefficient key and silently
selects the 0.2 % fit; pass `acceptance_range=2` for the 2 % law.

The MNIST floor mechanism and mitigation study is documented in
[Finite conductance floors in DRNs](conductance_floor_mitigation.md). The
earlier digits device screen remains a local ignored artifact documented in
`labs/cases/perfect_diode_hwa_lora_comparison/README.md`.

### IBM Analog Foundation Models PCM

- Public model: `add_PCM_noise` in IBM's
  `analog-foundation-models` repository.
- Representation: a signed weight-level error fit derived from Hermes unit
  cells, not an explicit simulation of the four PCM conductances in a cell.
- Nominal mapped maximum: `180` in a unitless fit coordinate.
- Exact-zero threshold: `0.6` in the same coordinate.
- Output-channel/tile split used here: at most 512 input values.
- Model behavior: endpoint Gaussian programming error with separate
  low- and high-conductance polynomial fits; no pulse sequence, retention
  law, or separate read-noise term is provided by this helper.
- Figure provenance: the AFM paper's Figures 7a and 9 use normalized axes.
  The released helper's `180`, `52.5`, and `0.6` constants have no stated
  physical unit and are not min/max values read from those figures.
- Underlying device evidence: the Hermes chip paper states a nominal
  `0.1–10 µS` range per PCM device and uses ADC counts for unit-cell mapping;
  it does not provide a conversion for the helper coordinate.
- Strict configuration names the helper coordinates `fit_max` and
  `zero_threshold`, deliberately avoiding a false microSiemens label.
- Repository role: deployment of ordinary and HWA checkpoints, followed by
  full BPTT or rank-4 LoRA recovery with a newly sampled endpoint write after
  every minibatch.

This is not AIHWKit's generic `PCMLikeNoiseModel`. The repository implements
the equations from IBM's released AFM helper directly so their tiling,
FP16 polynomial quantization, `52.5` branch point, and `0.6` zero rule
remain testable without an AIHWKit runtime dependency.

### Exponential-step ReRAM

- AIHWKit class: `ReRamESPresetDevice`
- Tiki-Taka wrapper: `TikiTakaReRamESPreset`
- Device model: `ExpStepDevice`
- Nominal state range: `[-1.0, 1.0]`
- Nominal minimum update step: `0.00135` normalized state units
- Main characteristics: nonlinear and asymmetric pulsed updates,
  cycle-to-cycle variation, device-to-device variation, and write noise.
- Repository role: the first native-device Tiki-Taka integration and all
  checked-in native Tiki-Taka MNIST configurations.

The underlying Gong et al. representative trace used a `30` to `60 µS` axis
and traversed roughly `33` to `57 µS`, but AIHWKit normalized the fitted device
state. That plotted excursion must not be substituted for an explicit
simulator calibration.

### Soft-bounds ReRAM

- AIHWKit class: `ReRamSBPresetDevice`
- Tiki-Taka wrapper: `TikiTakaReRamSBPreset`
- Device model: `SoftBoundsReferenceDevice`
- Nominal state range: `[-0.75, 1.25]`
- Nominal minimum update step: `0.002` normalized state units
- Main characteristics: a looser, simpler fit to the Gong et al. data, with a
  symmetry-point/reference treatment and both device and write variation.
- Repository role: supported and tested as a Tiki-Taka option, although the
  checked-in MNIST configurations select ES.

### Optimized-material ReRAM array

- AIHWKit class: `ReRamArrayOMPresetDevice`
- Device model: `SoftBoundsReferenceDevice`
- Nominal state range: `[-1.0, 1.0]`
- Nominal minimum update step: `0.0949` normalized state units
- Source: the optimized-material array in Gong and Rasch et al., IEDM 2022.
- Important default: the measured corrupt-device probability was 13.5%, but
  AIHWKit disables corrupt devices by default.
- Repository role: none. It would need a new explicit experiment capability
  and tests before use through repository configuration.

### Baseline-HfO2 ReRAM array

- AIHWKit class: `ReRamArrayHfO2PresetDevice`
- Device model: `SoftBoundsReferenceDevice`
- Nominal state range: `[-1.0, 1.0]`
- Nominal minimum update step: `0.4622` normalized state units
- Source: the baseline-HfO2 array in Gong and Rasch et al., IEDM 2022.
- Important default: the measured corrupt-device probability was 10%, but
  AIHWKit disables corrupt devices by default.
- Repository role: none. Like the optimized-material device, it is importable
  from AIHWKit but not selectable in this repository.

## Why the Tiki-Taka/HWA classification looks this way

The first repository integration of native AIHWKit ReRAM devices was commit
`e87e4126` (`Integrate AIHWKit Tiki-Taka MNIST training`, 2026-07-25).
It exposed the ES and SB Tiki-Taka presets; the checked-in runs selected ES.

Commit `ff79c048` (`Implement hardware-aware and ReRAM training experiments`,
2026-07-27) then reused ES/SB devices for HWA deployment and direct pulsed
training experiments. This does not change their first repository role:
ES/SB entered through Tiki-Taka.

The Wan, IBM AFM PCM, and CMO studies belong to the later
deployment-oriented HWA line. All three have strict programming
configurations; PCM and CMO additionally support repeated endpoint writes
during BPTT recovery. The two measured-array pulsed-training presets have not
been integrated into this repository.

## Selection guide

- Use **Wan-2022** for the current reproducible HWA-to-ReRAM and LoRA
  deployment path.
- Use **IBM AFM PCM** for the paper's normalized, unitless 0-to-180
  endpoint-noise model.
- Use **CMO/HfOx** for the measured 9–88.2-uS program, relaxation, and read
  model. Use `affine_floor` to retain the physical floor while preserving
  learned-value ordering. Use `literal_conductance` for an explicit
  absolute-scale/hard-clipping stress test. Select `normalized_offset` only
  to reproduce normalized weight transfer with a stated
  common-mode-cancellation assumption.
- Use **ES Tiki-Taka** for the repository's existing nonlinear pulsed-training
  baseline.
- Use **SB Tiki-Taka** for a simpler soft-bounds comparison against ES.
- Use **Array OM** or **Array HfO2** only after adding them at the narrow
  device/preset boundary, declaring the new capability combination, and
  adding numerical acceptance tests.

## Scope exclusions

AIHWKit's generic `PCMLikeNoiseModel` and `HermesNoiseModel` were included in
the earlier local measured-device screen, but they are not the IBM AFM PCM
helper integrated here. PCM is also a different memory technology from
ReRAM. ECRAM and capacitor presets are separate device families. AIHWKit
exposes many configuration
wrappers around ES and SB, including single-device, multi-device, TTv2, AGAD,
and mixed-precision variants; those wrappers are algorithms or array
compositions, not additional memristor models.

## Upstream sources

- [AIHWKit 1.1.0 ReRAM inference source](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/reram.py)
- [AIHWKit 1.1.0 conductance-converter source](https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/converter/conductance.py)
- [IBM Analog Foundation Models paper](https://arxiv.org/abs/2505.09663)
- [IBM Analog Foundation Models released PCM helper, commit 4f4c505](https://github.com/IBM/analog-foundation-models/blob/4f4c505b9d9af43af289788b3a29eeecaeb4e1ab/apply_noise_model.py)
- [IBM Hermes 64-core PCM chip](https://arxiv.org/abs/2212.02872)
- [AIHWKit ReRAM inference API](https://aihwkit.readthedocs.io/en/latest/api/aihwkit.inference.noise.reram.html)
- [AIHWKit pulsed-device preset API](https://aihwkit.readthedocs.io/en/latest/api/aihwkit.simulator.presets.devices.html)
- [Wan et al., Nature 2022](https://www.nature.com/articles/s41586-022-04992-8)
- [Gokmen and Haensch, Frontiers in Neuroscience 2020](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00103/full)
- [Gokmen, Frontiers in Artificial Intelligence 2021](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.699148/full)
- [Gong et al., Nature Communications 2018](https://www.nature.com/articles/s41467-018-04485-1)
- [Gong and Rasch et al., IEDM 2022](https://doi.org/10.1109/IEDM45625.2022.10019569)
- [Stecconi et al., Advanced Electronic Materials 2022](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aelm.202200448)
- [Athena et al., IEEE Transactions on Electron Devices 2023](https://research.ibm.com/publications/resta-recovery-of-accuracy-during-training-of-deep-learning-models-in-a-14-nm-technology-based-reram-array)
- [Stecconi et al., Nano Letters 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10811689/)
- [Rasch et al., Nature Communications 2024](https://www.nature.com/articles/s41467-024-51221-z)
- [Falcone et al., Advanced Functional Materials 2025](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adfm.202504688)
