# AGENTS — On-chip training as post-deployment adaptation

## Scope and research question

These instructions apply to `main` (named `codex/revised-hwa-training` until 3 November
2026). This tree determines when
on-chip training is a viable post-deployment adaptation strategy for networks
deployed on crossbar arrays with corrupt devices:

> When, and at what physical update cost, does on-chip training of a deployed array
> recover performance lost to device corruption beyond hardware-aware training,
> re-programming and matched digital calibration?

On-chip training is viable for a regime only when it beats the best predeclared
no-weight-learning alternative (HWA-only deployment, rewrite of frozen targets,
digital calibration) at matched data and budget, holds across independent array
assignments and defect rates, and its write cost (pulses per cell, touched cells,
verify reads) is reported. One favorable endpoint, assignment or defect case is
not a general result. Negative and regime-limited answers are valid outcomes.

Since 30 October 2026 the priority is drift rather than corruption. The order is
a device model first, then HWA and program-and-verify, then maintenance, and
only then on-chip learning. The drift records are owned by the shared study in
the workspace, [`../on-chip-learning/`](../on-chip-learning/README.md), which is
outside this repository and lists the active campaigns. Crossbar code and
native runs stay in this tree. The standard device model is defined under
[Drift](#drift). The corruption work below remains reference evidence.

The initial target is OPT-125M. Completed MNIST, dense CIFAR-10 and ResNet-32
CIFAR-10/100 work is reference evidence and the validated reference implementation
of the lifecycle; extend it only when an assignment asks. LoRA and general EP/DRN
work belong to sibling worktrees; using a digital adapter as a comparison arm here
is a material scientific choice to escalate. Do not restore retired solvers or
adapters. Historical execution code is at `16938bf`. Keep unrelated maintenance out.

- `ebl/`: public CLI for train, validate, characterize, campaigns, studies and inspection.
- `workflow/`: the standard lifecycle: strict schema and one reusable function per step
  (devices/defects, deployment, HWA, P&V, on-chip), run as `crossbar_lifecycle.v1` stages;
  `workflow/networks/` holds the network families (`opt_mlp_suffix`, `cifar_resnet32_suffix`).
- `training/`: crossbar plants, pulse optimizers, populations, endpoint models,
  explicit parameter catalogs and checkpoint codecs.
- `experiments/reram_program_verify/`: characterization, fitting, native samplers,
  storage and analysis of the device models.
- `experiments/cifar_crossbar/`, `cifar10_crossbar/`, `mnist_analog_relu/`, `mnist_relu/`:
  reference CIFAR/MNIST packages, frozen for reproduction.
- `campaigns/`: notes, lifecycle definitions (`<campaign>/lifecycles/*.json`), results,
  generated navigation and the subprocess-only orchestrator.
- `examples/`, `studies/`, `labs/tools/`: exact configs (including the OPT lifecycle
  templates in `examples/lifecycles/`), legacy plans, CIFAR analysis.
- `results/`, `artifacts/`, `data/`: ignored raw evidence, frozen sources and datasets.
- `docs/`: active contracts and historical records; `tests/`: numerical and workflow checks.

## Transformer target

Crossbar arrays replace the MLPs: `fc1` (768→3072) and `fc2` (3072→768) of the
last N OPT-125M decoder layers (4,718,592 logical weights per layer). Attention
stays digital: `q/k/v/out_proj`, QKᵀ, softmax and attention times V. The MLP
activation, biases, LayerNorm, token and position embeddings and the tied LM
head are also digital. Moving attention projections onto arrays is a separate,
explicitly assigned regime.

The lifecycle family `opt_mlp_suffix` implements this. The number of analog
MLPs (`network.analog_decoder_layers`), the corruption kinds and fractions
(`defects.cases`), the HWA sources and the on-chip arms are lifecycle settings:
varying them never needs new code. Start from `examples/lifecycles/opt125m-cmo-mlp4-template.json`
and follow [the lifecycle contract](docs/crossbar_lifecycle.md); its extension
points name where any genuinely missing capability goes. Record the analog layers
and logical weight count with every result.

The teacher is the pinned FP32 OPT-125M checkpoint (revision and SHA-256 in
`workflow/networks/opt_mlp.py`). The corpus, sequence length, cohort sizes and
primary metric (teacher KL or perplexity) are Explorer choices, declared in the
experiment note and the lifecycle before running; the corpus is pre-tokenized
with `python -m workflow corpus` and pinned by SHA-256. Adaptation and evaluation
tokens never overlap. Measure memory, throughput and storage on one or two
analog layers before scaling: device-state cost grows with every analog weight.

The user approved task-fine-tuned teachers on October 15, 2026 (exp-018). A
lifecycle may declare `network.teacher`: an FP32 OPT-125M fine-tuned on a
labelled multiple-choice task, with LoRA merged into its weights. That teacher
is pinned by SHA-256 and has a recorded fine-tuning provenance. With a declared
`task`, the task accuracy may be the primary metric. Each such lifecycle has a
matched pretrained-teacher control that differs only in the teacher. LoRA here
only prepares teachers; a deployed digital adapter remains a separate,
escalated regime.

## Corrupt devices

Defects are permanent physical faults defined by the lifecycle `defects` section:
kind (`open`, `gmax`, `random`, and for CMO `gmin`), rate and placement per physical device. All arms of
one assignment share that array's fault map, so they are matched conditions, not
independent arrays. Writes never repair a defect; verify persistence. Corruption
masks used in HWA training are temporary and are never the deployed fault map.
Report results per assignment and defect case; do not pool across defect rates.
CMO faults are placed hierarchically ([CMO device contract](docs/reram_cmo_device.md)):

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
([specification](../on-chip-learning/campaigns/drift/power-law-model.md)). One
knob sets the temporal correlation
([25-h evidence](../on-chip-learning/campaigns/drift-mitigation/series/001-device-model/results/retention-25h-correlation-20261103/README.md)):

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

## Scientific and physical contracts

Separate architecture, deterministic mapping, stochastic programming, defects,
off-chip adaptation, same-array learning and fresh-array transfer. Distinguish OM
incremental pulses, idealized CMO incremental writes, Gaussian PCM endpoint
reprogramming and PCM drift. Gradients and optimizer state are digital.
Calibration alone is a no-weight-write control.

Crossbar MVMs use effective state `q=a-r` and digital nonlinearities. Their accuracy
is not a physically loaded passive-network result. Every device-facing forward
uses the current held post-write apparent state: gradients, metrics, validation,
selection and final evaluation. Pulses mutate persistent state; writes refresh
apparent state for touched cells. Do not substitute persistent state or redraw
write noise per example unless a declared intervention requires it. Persistent-state
forwards are named diagnostics and must not drive gradients or selection.
Off-chip HWA updates a digital master using sampled apparent hardware views;
it is not an evolving persistent-device update.

Match teacher, analog subset, data split, batch order, objective, assignment and
endpoint seeds, recovery budget and checkpoint selection across controls. Record
topology and identity-count differences; abstract AIHWKit reference subtraction does
not imply a fabricated device count. P&V controllers receive targets, apparent
verifies, their history and calibration-partition estimates. Hidden state and
parameters stay on the analysis side. Distinguish persistent reachability,
verify-window intersection and apparent acceptance.

Fresh-array tests remap the frozen FP32 master through each target array's own
bounds/codebook. Recovered persistent-state transfer is a separate named arm.
Several endpoint seeds on one assignment are repeated programming realizations,
not independent arrays. Preserve per-identity construction and explicit trajectory
seeds; do not replace the serializable pulse plant with a native tile without an
exact replay contract.

## Data and initialization

The OPT experiments use IBM CMO/HfOx ReRAM (`devices.technology: "cmo"`), as
specified in the [CMO device contract](docs/reram_cmo_device.md):

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
two reference papers in [`hwa_papers/`](hwa_papers/README.md): Rasch et al.
2023 (arXiv:2302.08469) and Büchel et al. 2025, *Analog Foundation Models*
(arXiv:2505.09663). The experiment note states which choices follow them and
which deviate, and why.

AIHWKit's CMO drift and read-noise models stay disabled; drift follows the
[standard drift model](#drift). IBM optimized-material
(OM) ReRAM (`"om"`: literal AIHWKit cells, pulse P&V and pulse laws) and
Gaussian PCM (`"pcm"`) remain supported, separately declared regimes.
Label results exploratory and model-based, not fabricated-hardware measurements. Follow
[synapse-data provenance rules](docs/synapse_data.md) for measured evidence; do not
substitute datasets, preprocessing, tokenization or cohorts between matched arms.

Predeclare initialization. HWA and non-HWA arms start from the same logical state;
recovery arms start from the same named deployment unless initialization is the
intervention. Use explicit paths, revisions and hashes, never the newest checkpoint.
Strict configs and campaign lifecycles define inputs; historical
[initialization protocols](docs/initialization_protocols.md) provide lineage only.

## Workflow and authority

Precedence: the user, then this file (science), the [execution policy](docs/experiment_workflow.md)
(process), the experiment-loop skill (procedure) and the experiment note (rules of one
experiment). Every experiment follows the [experiment-loop skill](.agents/skills/experiment-loop/SKILL.md)
(Explorer, Codifier, Monitor, Reviewer); the Explorer and Codifier procedures are its
[explorer](.agents/skills/experiment-loop/references/explorer.md) and
[codifier](.agents/skills/experiment-loop/references/codifier.md) references.
Records follow the [artifact conventions](.agents/skills/experiment-loop/references/artifacts.md):
a small pilot is one note from the [pilot template](.agents/skills/experiment-loop/references/pilot.md)
in `campaigns/pilots/`; sustained work is a campaign with series, experiment and result
notes. Each crossbar loop is a strict lifecycle in `campaigns/<campaign>/lifecycles/`
([contract](docs/crossbar_lifecycle.md)), planned with `python -m workflow plan`, run
with `python -m ebl campaign run` and collected with `python -m workflow collect`.
The note links the lifecycle and owns the decision rule, budget, handoff and
interpretation. These supersede former study/manifest lifecycles and conflicting
external closeout skills.

Assigned execution authorizes implementation, proportional checks, launch,
monitoring, collection, interpretation and understood retries within scope and
budget. Planning alone does not authorize launch. Make cases, target, budget,
duration and output paths visible before substantial execution. Escalate material
scientific choices (analog subset, corpus, metric, defect regime, comparison arms),
missing access, unrelated effects or scope/budget changes. Cleanup/migration does
not expand remote, shared-GPU or overnight permissions.

Use the public CLI, exact inputs and native bundles. Keep one monitoring owner
per case. Preserve failed/partial evidence and distinguish completion, validity,
coverage and review; partial coverage stays labelled partial.

Specifications and handoffs belong to experiment/pilot notes; measurements and
interpretation belong to result/pilot notes. Generated ledgers are navigation only.
`docs/current_simulations.md` and `docs/experimental_manifest.md` are historical:
do not refresh or append. Do not edit human `docs/current_state.md` without explicit
permission, or read/edit `docs/my_notes.md` unless asked. Keep small records versioned
and large artifacts ignored. Never mutate frozen bundles or source archives to
repair historical provenance limitations.
