# Synapse data and device-model sources

This file is the source registry for synaptic-device data used or exposed by
this worktree. It covers the local measured resistance trajectories, device
models fitted to published measurements, and normalized AIHWKit pulsed-device
presets. Those sources are not interchangeable.

Device-facing HWA, deployment, and on-chip-training claims must use raw
measured synaptic data. A fitted model or normalized preset may be useful as a
matched control, but the study must label it as model-based or synthetic and
must not present it as a measured-trace arm.

## Evidence classes

| Evidence class | What the simulation consumes | Permitted interpretation |
| --- | --- | --- |
| Raw measured traces | Per-device resistance or conductance samples and their recorded device identities | Primary measured-device evidence, subject to the dataset's stated provenance limits |
| Hardware-derived fitted model | Published coefficients or distributions fitted to hardware measurements; the simulator generates the realization | Model-based device control; evidence about that fit, not direct replay of the underlying devices |
| Normalized pulsed preset | Abstract device states and pulse-update parameters, sometimes derived from measured curves but without a traceable physical-conductance calibration | Pulsed-training or algorithm control |
| Idealized or analyst-selected model | Ideal states, generic noise, or manually selected parameters | Synthetic control only |

## Repository source inventory

The following sources are currently consumed by repository code or selected by
tracked configurations. The configuration selector, version, mapping, random
seed, and role of each source must be recorded per arm.

| Source | Repository selector or API | Representation | Current repository role | Evidence class |
| --- | --- | --- | --- | --- |
| Local March ReRAM traces | `--device-data data/march_slope_x3_5k.hdf5` with a `measured_cohort_*` update backend | 1,268 measured resistance trajectories, each with 5,000 samples and a physical-cell identity | Measured initialization, deployment, reassignment, direct updates, and measured LoRA controls | Raw measured traces |
| Wan-2022 ReRAM | `aihwkit_reram_wan2022` or `aihwkit_reram_wan2022_physical`; AIHWKit `ReRamWan2022NoiseModel` | Conductance-dependent programming and retention fit; AIHWKit defaults to `0–25 uS`, while completed repository studies normally use `0–40 uS` | Persistent deployment realizations and recovery controls, including LoRA | Hardware-derived fitted model |
| CMO/HfOx ReRAM | Crossbar lifecycle `devices.technology: "cmo"` ([contract](reram_cmo_device.md)); historically `aihwkit_reram_cmo`; equations from AIHWKit `ReRamCMONoiseModel` | 2 % (or 0.2 %) program-and-verify fit over `9–88.199997 uS`; the lifecycle omits relaxation and read variation | Working device for the OPT crossbar lifecycles: deployment, selection and idealized on-chip writes (HWA uses declared constant-σ Gaussian weight noise, not this fit) | Hardware-derived fitted model; the on-chip write law is an idealization |
| IBM optimized-material ReRAM array | `ReRamArrayOMPresetDevice`; `ibm_reram_program_verify.v1`; `ibm_reram_om_program_verify` modifier; `mnist_ibm_om_crossbar_relu.v1` | Normalized pulsed state nominally in `[-1,1]`, fixed sampled cell bounds and corruption identity, pulse-resolved or held-out-calibrated endpoint execution | Cap-128 pulse characterization, off-chip HWA, physical selection, persistent DRN deployment, and standard-crossbar/digital-ReLU recovery control | Hardware-derived fitted model |
| IBM baseline-HfO2 ReRAM array | `ReRamArrayHfO2PresetDevice`; `ibm_reram_program_verify.v1` | Normalized pulsed state nominally in `[-1,1]` with a much coarser nominal update | Device-characterization stress control only; not selected by the first OM network pilot | Hardware-derived fitted model |
| Exponential-step ReRAM | `TikiTakaReRamESPreset`; AIHWKit `ReRamESPresetDevice` | Normalized pulsed state nominally in `[-1, 1]`; nonlinear/asymmetric updates and device/write variation | Native AIHWKit Tiki-Taka reference selected by the tracked lab configurations | Normalized pulsed preset |
| Soft-bounds ReRAM | `TikiTakaReRamSBPreset`; AIHWKit `ReRamSBPresetDevice` | Normalized pulsed state nominally in `[-0.75, 1.25]`; simplified soft-bounds approximation | Supported and tested Tiki-Taka comparison; no tracked study configuration currently selects it | Normalized pulsed preset |
| IBM Analog Foundation Models PCM | `ibm_afm2025_pcm` | Hardware-derived endpoint programming-error fit in a unitless `0–180` coordinate; this is PCM, not ReRAM | Persistent deployment and repeated noisy-write recovery controls | Hardware-derived fitted model |

AIHWKit 1.1.0 is the version against which this inventory and the detailed
[`ReRAM device catalog`](reram_device_catalog.md) were last verified. A run
must record the version it actually loads; the version stated here is not a
substitute for run provenance.

The two Wan selectors expose different mapping contracts around the same
source model. Likewise, the CMO mapping may retain, normalize, or hard-clip its
physical conductance floor. Treat the selector and mapping as experimental
variables, not as aliases for one arm. Their strict configuration definitions
are in
[`model/resistive/device_config.py`](../model/resistive/device_config.py), and
their realizations are applied in
[`training/device_programming.py`](../training/device_programming.py).

### Other AIHWKit pulsed sources

AIHWKit also contains the following ReRAM pulsed-device sources. They are
listed here so that library availability is not mistaken for a completed
repository integration.

| AIHWKit source | Representation | Repository status |
| --- | --- | --- |
| `ReRamArrayOMPresetDevice` | Normalized optimized-material 14 nm ReRAM-array model, nominally `[-1, 1]` | Integrated for strict pulse characterization and the predeclared OM HWA/deployment pilot; evidence remains model-based |
| `ReRamArrayHfO2PresetDevice` | Normalized baseline-HfO2 14 nm ReRAM-array model, nominally `[-1, 1]` | Integrated for strict pulse characterization; network HWA remains deferred |

The Tiki-Taka backend also allowlists `TikiTakaIdealizedPreset`,
`TikiTakaCapacitorPreset`, `TikiTakaEcRamPreset`, and
`TikiTakaEcRamMOPreset`. They are idealized, capacitor, or ECRAM controls, not
additional ReRAM datasets, and no tracked experiment configuration currently
uses them. The allowlist is defined in
[`training/tiki_taka.py`](../training/tiki_taka.py).

AIHWKit's generic `PCMLikeNoiseModel` and `HermesNoiseModel` are not the IBM
AFM PCM fit integrated here. Generic additive Gaussian HWA from
[`training/add_normal.py`](../training/add_normal.py) is also not synapse data.
Any of these sources must remain explicitly labelled as a control.

## Canonical raw measured dataset

The current dataset is:

```text
march_slope_x3_5k.hdf5
SHA-256: 207b143fc1a63710c944db74bc573f16b616ce50e22c6605c12068bf6a3f230d
```

It contains 1,268 measured resistance traces. Each HDF5 dataset contains 5,000
samples and identifies its physical cell with `row` and `col` attributes.
The file has no top-level provenance metadata, so the content hash is the
repository's immutable dataset identity. Do not infer or invent fabrication
provenance that is not recorded in a study plan.

## Where to find it

On the primary development host, the canonical copy is currently:

```text
/home/filip/reram_data/march_slope_x3_5k.hdf5
```

Commands and portable campaign manifests expect a repository-local ignored
copy or symlink at:

```text
data/march_slope_x3_5k.hdf5
```

From the repository root, stage it without adding the raw file to Git:

```bash
mkdir -p data
ln -s /home/filip/reram_data/march_slope_x3_5k.hdf5 \
  data/march_slope_x3_5k.hdf5
sha256sum data/march_slope_x3_5k.hdf5
```

The Akib campaign currently stages the same content at:

```text
/home/filiposana/staged/tiki_taka_eight_threshold_20260820_inputs/march_slope_x3_5k.hdf5
```

Verify the SHA-256 after every copy or transfer. If no verified copy is
available, stop rather than replacing it with generated data or an unrelated
device preset. Campaign input expectations are also recorded in
[`campaigns/manifests/README.md`](../campaigns/manifests/README.md).

## Numerical use

Pass the file explicitly:

```bash
python -m ebl train \
  --config <declared-config.json> \
  --output-dir results/<study-id>/runs/<arm-id> \
  --device-data data/march_slope_x3_5k.hdf5
```

The measured-trace loader in [`training/measured_trace.py`](../training/measured_trace.py):

- screens traces whose first resistance sample exceeds `30 kOhm`;
- converts resistance to conductance with `G = 1/R` in SI units;
- keeps bar/non-bar traces from the same physical cell in one cohort;
- uses the declared split and assignment seeds to create deterministic
  cohort-A and cohort-B assignments; and
- records the dataset digest, active trace names, cohort counts, preprocessing,
  and assignment identity in run artifacts.

With the current seed-42 split, the formed data provide 633 cohort-A traces
from 317 cells and 632 cohort-B traces from 317 cells. Preprocessing,
interpolation, common-window construction, and pulse policy are scientific
interventions: declare them in the study plan and keep them matched across
arms unless they are the variable under test.

## Per-arm source declaration

Before launching a study, record for every base and auxiliary synaptic array:

- the source name and evidence class from this registry;
- the exact dataset path and SHA-256 for raw data, or the library/model name
  and runtime version for a fitted model or preset;
- the physical or normalized range and the DRN-to-device mapping;
- preprocessing, cohort, device-assignment, programming, construction, and
  update seeds as applicable;
- retention time, noise scales, pulse policy, and any common-window or
  interpolation policy; and
- whether the source is used for initialization, deployed base weights,
  Tiki-Taka fast or slow arrays, or LoRA factors.

Matched arms must keep these fields fixed unless a listed field is the
predeclared intervention. In particular, do not silently substitute an
AIHWKit preset when the measured HDF5 file is unavailable.

For model equations, source-paper provenance, physical-range limitations, and
the distinction between inference models and pulsed-update presets, see the
[`ReRAM device catalog`](reram_device_catalog.md).
