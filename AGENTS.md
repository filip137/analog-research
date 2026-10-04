# AGENTS — Crossbar HWA and on-chip learning

## Scope and repository

These instructions apply to `codex/revised-hwa-training`. This tree owns crossbar
HWA, deployment, on-chip learning and fresh-array transfer: MNIST controls, dense
CIFAR-10 and ResNet-32 CIFAR-10/100 suffix experiments. Its research question is:

> When does array-specific learning improve deployment beyond hardware-aware
> training and matched digital calibration, and at what physical update cost?

Use [the supported-scope guide](docs/crossbar_scope.md). LoRA and general EP/DRN
work belong to sibling worktrees. Do not restore retired solvers or adapters to
implement crossbar features. Historical execution code is at `16938bf`; historical
native evidence remains inspectable. Keep unrelated maintenance out of this tree.

- `ebl/`: public CLI for train, validate, characterize, campaigns, studies and inspection.
- `experiments/cifar_crossbar/`: ResNet suffix mapping, HWA/CDT, OM/PCM recovery and reports.
- `experiments/cifar10_crossbar/`: separate dense 3072-256-10 CIFAR-10 experiments.
- `experiments/mnist_analog_relu/`: staged MNIST v2 and shared crossbar helpers.
- `experiments/mnist_relu/`: bias-free digital teachers.
- `experiments/reram_program_verify/`: characterization, fitting, native samplers,
  storage, analysis and the existing persistent local launcher.
- `training/`: crossbar plants, pulse optimizers, populations, endpoint models,
  explicit parameter catalogs and checkpoint codecs.
- `labs/tools/`: CIFAR tables, analysis and plots; no alternate training surface.
- `examples/`: exact configs; `studies/`: optional plans and reference receipts.
- `campaigns/`: specifications, scientific results, generated navigation and the
  subprocess-only orchestrator of explicit configs and inputs.
- `results/`, `artifacts/`, `data/`: ignored raw evidence, frozen sources and datasets.
- `docs/`: active contracts and historical records; `tests/`: numerical and workflow checks.

## Scientific and physical contracts

Separate architecture, deterministic mapping, stochastic programming, off-chip
adaptation, same-array learning and fresh-array transfer. Do not infer a general
need for on-chip training from one favorable endpoint or device assignment.

CIFAR ResNet experiments retain the full network and freeze the digital prefix.
Four/eight analog convolutions describe suffix depth, plus the classifier.
Keep dense CIFAR results separate. Distinguish OM incremental pulses, Gaussian
PCM endpoint reprogramming and legacy PCM pulse/inference controls. Gradients
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

Match teacher, dimensions, data split, minibatch order, objective, assignment and
endpoint seeds, recovery budget and checkpoint selection across controls. Record
topology and identity-count differences. There are 39,700 logical weights for
784-50-10; abstract AIHWKit reference subtraction does not imply a fabricated
device count. P&V controllers receive targets, apparent verifies, their history
and calibration-partition estimates. Hidden state and parameters stay on the
analysis side. Distinguish persistent reachability, verify-window intersection
and apparent acceptance.

Frozen-model fresh-array tests remap the frozen FP32 master through each target
array's own bounds/codebook. Recovered persistent-state transfer is a separate
named arm. Several endpoint seeds on one assignment are repeated programming
realizations, not independent arrays. Report assignment-level results. Preserve
per-identity construction and explicit trajectory seeds; do not replace the
serializable pulse plant with a native CPU tile without an exact replay contract.

## Data and initialization

Imported CIFAR studies use AIHWKit 1.1.0 presets and Gaussian PCM controls. Label
them exploratory and model-based, not fabricated-hardware measurements. Follow
[synapse-data provenance rules](docs/synapse_data.md) for measured evidence; do not
substitute datasets, preprocessing or cohorts between matched arms.

Predeclare initialization. HWA and non-HWA arms start from the same logical state;
recovery arms start from the same named deployment unless initialization is the
intervention. Use explicit paths and hashes, never the newest checkpoint.
Historical [initialization protocols](docs/initialization_protocols.md) provide
lineage; current strict configs and campaign specifications define crossbar inputs.

## Workflow and authority

Use the [experiment-loop skill](.agents/skills/experiment-loop/SKILL.md) and
[execution policy](docs/experiment_workflow.md). They supersede former mandatory
study/manifest lifecycles and conflicting external closeout-skill instructions.
Start from the question and decision. Small pilots use one evolving note in
`campaigns/pilots/`; sustained work uses campaign experiment/result notes.

Assigned execution authorizes implementation, proportional checks, launch,
monitoring, collection, interpretation and understood retries within scope and
budget. Planning alone does not authorize launch. Make cases, target, budget,
duration and output paths visible before substantial execution. Escalate material
scientific choices, missing access, unrelated effects or scope/budget changes.
Cleanup/migration does not expand remote, shared-GPU or overnight permissions.

Use the public CLI, exact inputs and native bundles. Keep one monitoring owner
per case. Preserve failed/partial evidence and distinguish completion, validity,
coverage and review. Negative/inconclusive conclusions are valid; partial coverage
stays labelled partial. Claim on-chip learning is needed only when a predeclared
HWA-only control underperforms and matched recovery closes the specified gap.

Specifications and handoffs belong to experiment/pilot notes; measurements and
interpretation belong to result/pilot notes. Generated ledgers are navigation only.
`docs/current_simulations.md` and `docs/experimental_manifest.md` are historical:
do not refresh or append. Do not edit human `docs/current_state.md` without explicit
permission, or read/edit `docs/my_notes.md` unless asked. Keep small records versioned
and large artifacts ignored. Never mutate frozen bundles or source archives to
repair historical provenance limitations.
