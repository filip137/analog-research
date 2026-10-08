# AGENTS — On-chip training as post-deployment adaptation

## Scope and research question

These instructions apply to `codex/revised-hwa-training`. This tree determines when
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
  template in `examples/lifecycles/`), legacy plans, CIFAR analysis.
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
varying them never needs new code. Start from `examples/lifecycles/opt125m-om-mlp4-template.json`
and follow [the lifecycle contract](docs/crossbar_lifecycle.md); its extension
points name where any genuinely missing capability goes. Record the analog layers
and logical weight count with every result.

The teacher is the pinned FP32 OPT-125M checkpoint (revision and SHA-256 in
`workflow/networks/opt_mlp.py`). The corpus, sequence length, cohort sizes and
primary metric (teacher KL or perplexity) are Explorer choices, declared in the
experiment note and the lifecycle before running; the corpus is pre-tokenized
with `python -m workflow corpus` and pinned by SHA-256. Adaptation and evaluation
tokens never overlap. Measure memory, throughput and storage on one or two
analog layers before scaling: OM cost grows with every analog weight.

## Corrupt devices

Defects are permanent physical faults defined by the lifecycle `defects` section:
kind (`open`, `gmax`, `random`), rate and placement per physical device. All arms of
one assignment share that array's fault map, so they are matched conditions, not
independent arrays. Writes never repair a defect; verify persistence. Corruption
masks used in HWA training are temporary and are never the deployed fault map.
Report results per assignment and defect case; do not pool across defect rates.

## Scientific and physical contracts

Separate architecture, deterministic mapping, stochastic programming, defects,
off-chip adaptation, same-array learning and fresh-array transfer. Distinguish OM
incremental pulses, Gaussian PCM endpoint reprogramming and PCM drift. Gradients
and optimizer state are digital. Calibration alone is a no-weight-write control.

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

Initial OPT tests use IBM optimized-material (OM) ReRAM: literal AIHWKit 1.1.0
cells (`devices.technology: "om"`), one active cell per weight read as `q=a-r`
against its intrinsic reference, closed-loop or open-loop-nominal P&V and
`om_closed_loop_pulse_adam`/`om_open_loop_pulse_adam` on-chip laws. OM retention
and drift are not modelled. Gaussian PCM is a later, separately declared regime.
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
