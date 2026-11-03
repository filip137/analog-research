# AGENTS — On-chip training as post-deployment adaptation

## Scope and research question

These instructions apply to `main` (named `codex/revised-hwa-training` until
3 November 2026). This worktree studies hardware-aware training, deployment
and on-chip adaptation of crossbar arrays under three effects: **corrupted
devices, write noise and drift**, individually and in declared combinations.

> When, and at what physical update cost, does on-chip training recover
> performance lost to corrupted devices, write noise or drift beyond
> hardware-aware training, re-programming and matched digital calibration?

On-chip training is viable for a regime only when it beats the best predeclared
no-weight-learning alternative (HWA-only deployment, rewrite of frozen targets,
digital calibration) at matched data and budget, holds across independent array
assignments and the declared conditions, and its write cost (pulses per cell,
touched cells, verify reads) is reported. One favorable endpoint, assignment,
defect case, noise level or age is not a general result. Negative and
regime-limited answers are valid outcomes.

`main` owns the shared scientific and workflow contracts, reusable code and
reference evidence. The branch responsible for an experiment owns its simulation
choices. Shared crossbar/DRN study records live in
[`../on-chip-learning/`](../on-chip-learning/README.md); crossbar code and native
runs stay in this tree. Consult the assigned campaign for its current question
and direction.

The initial target is OPT-125M. Completed MNIST, dense CIFAR-10 and ResNet-32
CIFAR-10/100 work is reference evidence and the validated reference implementation
of the lifecycle; extend it only when an assignment asks. LoRA and general EP/DRN
work belong to sibling worktrees; using a digital adapter as a comparison arm here
is a material scientific choice to escalate. Do not restore retired solvers or
adapters. Historical execution code is at `16938bf`. Keep unrelated maintenance out.

- `workflow/`: agent navigation and authoritative execution policy; documentation only.
- `ebl/`: public CLI, experiment registry, lifecycle schema, runtime stages,
  subprocess runner, scheduling, recovery and collection.
- `core/`: neutral configuration, artifact identities, serialization and cache I/O.
- `networks/`: OPT/CIFAR architecture, teachers, data and physical-forward adapters.
- `hardware/`: mappings, device populations, permanent defects, held state and drift laws.
- `methods/`: HWA, on-chip learning, calibration, programming/verify, maintenance,
  digital optimizers and training checkpoint codecs.
- `experiments/reram_program_verify/`: characterization, fitting, native samplers,
  storage and analysis of the device models.
- `experiments/cifar_crossbar/`, `cifar10_crossbar/`, `mnist_analog_relu/`, `mnist_relu/`:
  reference CIFAR/MNIST packages, frozen for reproduction.
- `campaigns/`: notes, lifecycle definitions (`<campaign>/lifecycles/*.json`), results,
  and generated navigation. The subprocess-only orchestrator is in `ebl/execution/`.
- `examples/`, `studies/`, `labs/tools/`: exact configs (including the OPT lifecycle
  templates in `examples/lifecycles/`), legacy plans, CIFAR analysis.
- `results/`, `artifacts/`, `data/`: ignored raw evidence, frozen sources and datasets.
- `docs/`: active contracts and historical records; `tests/`: numerical and workflow checks.

## Simulation details belong to the experiment branch

On the experiment's `exp/<id>` branch, declare the selected settings in its
experiment or pilot note and strict lifecycle before launch. This includes:

- which of corrupted devices, write noise and drift are enabled, their models,
  parameters and interactions, and the controls that isolate their effects;
- device technology and model revision, topology and mapping, conductance bounds,
  programming/verify law, write/reset behavior and temporal correlations;
- network and analog subset, teacher and initialization, corpus and cohorts,
  objective, metric, adaptation arms, selection rule and physical cost accounting;
- assignment and trajectory seeds, repetitions, time grid, compute/storage budget,
  source and input identities, output paths and execution handoff.

Use existing contracts and templates as implementation references. A template or
a previous experiment does not select settings for a new run. Link any adopted
shared protocol and freeze its version or hash with the run. Material scientific
choices require user authorization; the Codifier does not fill unresolved choices.
The note and lifecycle are committed and integrated at the workflow's git
checkpoints. Once merged into `main`, they remain records of that experiment.

## Transformer target

The implemented OPT family is `opt_mlp_suffix`: crossbar arrays replace `fc1`
(768→3072) and `fc2` (3072→768) of the last N OPT-125M decoder layers,
4,718,592 logical weights per layer. Attention, the MLP activation, biases,
LayerNorm, embeddings and tied LM head stay digital. Moving attention
projections onto arrays is a separate, explicitly assigned regime.

Declare N, teacher, corpus, sequence length, cohorts and primary metric in the
experiment branch. Record the analog layers and logical weight count with every
result. The pinned pretrained FP32 teacher reference is in
`networks/opt.py`. A declared `network.teacher` may instead use a
task-fine-tuned FP32 teacher with merged LoRA, a pinned SHA-256 and recorded
fine-tuning provenance; retain the matched pretrained-teacher control. A deployed
digital adapter remains a separate, escalated regime.

Follow the [lifecycle contract](docs/crossbar_lifecycle.md) and select the template
in `examples/lifecycles/` that matches the declared regime. The corpus is
pre-tokenized with `python -m ebl lifecycle corpus` and pinned by SHA-256. Adaptation
and evaluation tokens never overlap. Measure memory, throughput and storage
before scaling the analog subset: device-state cost grows with every weight.

## Corrupt devices

Defects are permanent physical faults. The experiment branch declares their
kind, rate and placement in `defects`. All arms of one assignment share that
array's fault map; these are matched conditions, not independent arrays. Writes
never repair a defect; verify persistence. Corruption masks sampled during HWA
are temporary and are never the deployed fault map.

Report each assignment and defect case separately. Identify synthetic fault
statistics as such. The [CMO device contract](docs/reram_cmo_device.md) describes
the supported hierarchical placement model when that model is selected.

## Write noise

Distinguish initial programming error from noise on subsequent writes, and both
from temporary HWA perturbations or read noise. The experiment branch declares
the write law, noise distribution and strength, affected devices, correlations,
clipping and any verify/retry policy. Keep the realized apparent state after a
write until a declared physical process changes it. Noise-free writes are a
named idealized control. Report physical writes, touched cells, pulses and
verify reads at the level supported by the model; endpoint commands are not
measured pulse, energy or latency costs.

## Drift

The experiment branch declares the drift law, conductance window, coefficients,
time units and grid, temporal correlations, affected devices and behavior after
a write. State how programming error, drift, read noise and permanent faults
interact. Distinguish measured support from assumed sensitivities and time
extrapolations, and report clipping at both bounds.

Check the selected law against the implementation before running: a marginal
law at one age does not establish its temporal process or rewrite semantics.
Keep historical replay contracts intact. For shared drift modules mirrored in
`drn/`, coordinate changes to both copies. Paired stochastic trajectories must
follow the declared coupling and compatible time grids.

The [recorded CMO/drift reference](docs/archive/cmo-drift-model-20261103.md)
preserves the earlier parameters and implementation limitations. Any model
adopted from it must be selected explicitly in the experiment's note and lifecycle.

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

Declare device sources and initialization in the experiment branch. Pin exact
model revisions, data and checkpoint paths and hashes. CMO, OM and PCM are
separate supported regimes with different programming and update semantics;
the [CMO device contract](docs/reram_cmo_device.md) and
[lifecycle contract](docs/crossbar_lifecycle.md) describe their implementations.

HWA training decisions — noise model, magnitude and scaling, weight clipping
and mapping, corruption-aware masks, objective and distillation, training data
and token budget, learning-rate schedule and checkpoint selection — consult the
two reference papers in [`hwa_papers/`](hwa_papers/README.md): Rasch et al.
2023 (arXiv:2302.08469) and Büchel et al. 2025, *Analog Foundation Models*
(arXiv:2505.09663). The experiment note states which choices follow them and
which deviate, and why.

Label results exploratory and model-based, not fabricated-hardware measurements.
Follow [synapse-data provenance rules](docs/synapse_data.md) for measured evidence;
do not substitute datasets, preprocessing, tokenization or cohorts between
matched arms.

HWA and non-HWA arms start from the same logical state; recovery arms start from
the same named deployment unless initialization is the intervention. Use explicit
paths, revisions and hashes, never the newest checkpoint. Strict configs and
campaign lifecycles define inputs; historical
[initialization protocols](docs/initialization_protocols.md) provide lineage only.

## Workflow and authority

Precedence: the user, then this file (science), the [execution policy](workflow/policy.md)
(process), the experiment-loop skill (procedure) and the experiment note (rules of one
experiment). Every experiment follows the [experiment-loop skill](.agents/skills/experiment-loop/SKILL.md)
(Explorer, Codifier, Monitor, Reviewer); the Explorer and Codifier procedures are its
[explorer](.agents/skills/experiment-loop/references/explorer.md) and
[codifier](.agents/skills/experiment-loop/references/codifier.md) references.
Records follow the [artifact conventions](.agents/skills/experiment-loop/references/artifacts.md):
a small pilot is one note from the [pilot template](.agents/skills/experiment-loop/references/pilot.md)
in `campaigns/pilots/`; sustained work is a campaign with series, experiment and result
notes. Each crossbar loop is a strict lifecycle in `campaigns/<campaign>/lifecycles/`
([contract](docs/crossbar_lifecycle.md)), planned with `python -m ebl lifecycle plan`, run
with `python -m ebl campaign run` and collected with `python -m ebl lifecycle collect`.
The note links the lifecycle and owns the decision rule, budget, handoff and
interpretation. These supersede former study/manifest lifecycles and conflicting
external closeout skills. Each experiment or pilot works on its own `exp/<id>`
branch and worktree, commits at the loop's
[git checkpoints](.agents/skills/experiment-loop/SKILL.md#git-checkpoints), and is
merged into `main` and pushed at launch and at closeout. No
changed file is left uncommitted at the end of a turn.

Assigned execution authorizes implementation, proportional checks, launch,
monitoring, collection, interpretation and understood retries within scope and
budget. Planning alone does not authorize launch. Make cases, target, budget,
duration and output paths visible before substantial execution. Escalate material
scientific choices (analog subset, corpus, metric, defect/noise/drift regime, comparison arms),
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
