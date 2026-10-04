# AGENTS — Crossbar HWA and on-chip learning

## Worktree scope

These instructions apply to the `codex/revised-hwa-training`
worktree. Its primary research question is:

> When does array-specific learning improve deployment beyond hardware-aware
> training and matched digital calibration, and at what physical update cost?

This tree owns standard-crossbar HWA, deployment, on-chip learning and fresh-array
transfer, including MNIST controls, dense CIFAR-10 and ResNet-32 CIFAR-10/100
suffix experiments. The CIFAR evidence and reproduction instructions are indexed
in `campaigns/cifar-crossbar-hwa-recovery/README.md`. LoRA and general EP/DRN
development belong to the sibling worktrees; retained DRN code supplies historical
comparisons and shared dependencies, not the primary direction of this tree.

Work in this tree must separate logical architecture, deterministic mapping,
stochastic programming, off-chip adaptation, same-array on-chip recovery, and
fresh-array transfer. In matched DRN studies, the standard-crossbar comparator is
an explicit architecture control. Do not infer a
general need for on-chip training from one favorable endpoint or one device
assignment.

Keep unrelated repository maintenance and unrelated experiment families out
of this worktree.

## Mandatory DRN apparent-state forwards

- Whenever a device model distinguishes a hidden persistent state from a
  post-write apparent state, **every device-facing DRN forward pass must use
  the current held apparent state**. This includes forwards used for off-chip
  HWA, on-chip recovery, gradient computation, training metrics, validation,
  checkpoint selection, and final evaluation. Never substitute the hidden
  persistent state in any of those paths.
- Pulses and physical updates mutate the persistent state. After each write,
  refresh the apparent state of touched cells according to the declared device
  and write-noise model; the next DRN forward uses that updated held apparent
  state. Do not copy apparent values into persistent state, and do not redraw
  write noise independently per example unless a separately declared
  inference-read-noise intervention requires it.
- A persistent-state DRN forward is permitted only as an explicitly named
  diagnostic or ablation. It must not drive gradients, optimizer updates,
  checkpoint selection, or headline conclusions. When both states exist,
  report apparent-state accuracy as primary and persistent-state accuracy as a
  clearly labelled secondary diagnostic on the same examples and settings.
- For off-chip HWA without an evolving physical plant, use the sampled
  noisy/apparent hardware view for the forward and update the digital master
  or desired target. Do not describe that digital-master update as a
  persistent-device update.

## Repository map

- `ebl/`: public command-line entry point. Workflow-managed training,
  validation, checkpoint handling, and study lifecycle commands enter through
  `python -m ebl`.
- `experiments/cifar_crossbar/`: ResNet-32 suffix mapping, OM pulse updates,
  PCM endpoint reprogramming, HWA/CDT, recovery and scientific reports.
- `experiments/cifar10_crossbar/`: the separate dense 3072-256-10 CIFAR-10
  experiment; do not pool its results with convolutional ResNet-32.
- `experiments/mnist_analog_relu/`: standard-crossbar MNIST HWA/recovery and
  shared population/programming helpers used by the dense CIFAR experiment.
- `model/resistive/`: perfect-diode DRN construction and numerical equations,
  including the passive and digital low-rank adapter structures used by LoRA
  controls.
- `training/`: reusable training mechanisms. The worktree-relevant modules
  include HWA modifiers, measured-trace programming and update backends,
  Tiki-Taka accumulation, parameter catalogs, probes, and checkpoint codecs.
- `training/ibm_reram_program_verify.py`: explicit-RNG transcription of the
  AIHWKit 1.1.0 `SoftBoundsReferenceDevice` pulse equation, capability-limited
  controller port, one-pulse/adaptive controllers, and identity partitions.
- `training/ibm_reram_endpoint_model.py`: deployment sampler for the fitted
  target-conditioned bounded endpoint density with explicit non-corrupt
  failure, persistent corrupt-device, target-support, and clipping outcomes.
- `experiments/reram_program_verify/`: strict `characterize` composition
  root, streaming trajectory/event store, held-out endpoint fitting, Wan
  comparison, plots, and report generation.
- `experiments/reram_program_verify/local_short_launcher.py`: persistent local
  CUDA launcher and heartbeat/receipt writer for reviewed production studies.
  It invokes the public CLI for every arm; it is not an alternative numerical
  execution surface.
- `experiments/mnist_relu_drn/`: strict teacher-to-perfect-diode-DRN mapping,
  HWA, measured-device deployment, training, validation, and provenance.
- `experiments/mnist_relu_drn_reset/`: measured pulse-zero/RESET
  initialization experiments and their learning-rate protocol.
- `experiments/small_network/`: existing composition root for DRN HWA,
  Tiki-Taka, and low-rank adapter controls when a declared study uses the
  `small_drn.v1` experiment family.
- `examples/mnist_relu_drn/`, `examples/mnist_relu_drn_reset/`, and
  `examples/small_drn/`: strict versioned experiment configs. A study plan
  references exact config files; configs do not choose input checkpoints or
  output locations.
- `examples/reram_program_verify/`: operational smoke, sizing, immutable
  production, cap-128, and focused HWA-prerequisite characterization configs.
- `studies/`: historical/optional strict JSON study plans; new pilot specifications
  live in their single Markdown note.
- `campaigns/`: question-led notes and the existing subprocess-only orchestrator
  of exact configs and explicit inputs.
- `data/`: ignored staging area for measured synapse data. Its required file,
  digest, and placement are defined in `docs/synapse_data.md`.
- `results/`: ignored raw run directories and generated study analyses. One
  prepared study owns one `results/<study-id>/` root.
- `docs/`: scientific contracts and historical provenance. New execution handoffs
  and conclusions belong to their campaign/pilot records.
- `tests/`: strict config, numerical parity, provenance, lifecycle, measured
  update, and study-workflow tests.
- `labs/`: focused reference harnesses for direct perfect-diode and AIHWKit
  Tiki-Taka comparisons. Use them for reference/parity work; workflow-managed
  evidence must use the public `python -m ebl` CLI.

## Architecture

- CIFAR ResNet experiments retain the full network, freeze the digital prefix,
  and map the specified suffix plus classifier to crossbars. Four/eight analog
  convolutions describe suffix depth, not whole-network analog mapping.
- Distinguish OM incremental pulse updates, Gaussian PCM endpoint reprogramming,
  and the separate legacy PCM pulse/inference controls. Gradients and optimizers
  in these experiments are digital. Digital calibration alone is a no-weight-write
  control, not on-chip weight learning.
- The following MNIST architecture contracts apply to historical matched DRN
  comparisons; their dimensions do not constrain the CIFAR families.
- In those comparisons the evaluated student is a dissipative resistive network (DRN)
  using `perfect_diode` nonlinearities.
- The explicit comparator has the same logical `784-50-10` dimensions and
  bias-free layer structure as the teacher-targeted DRN. It consists of a
  standard analog MVM, a digital ReLU, and a second standard analog MVM. The DRN's
  doubled `1568-100-20` tensors are rail encoding, not extra learned neurons.
- The crossbar initializes from the exact ReLU teacher tensors. The pinned DRN
  comparison initializes from its separately recovered physical checkpoint;
  they share the teacher task and dimensions, not an identical trained weight
  tensor. Compare source-to-deployment loss as well as absolute accuracy.
- The standard crossbar uses the AIHWKit OM effective state `q=a-r` for
  transfer and has no passive-DRN conductance-sum voltage denominator. Keep
  this architectural distinction visible; do not present standard MVM
  accuracy as a physically loaded DRN result.
- AIHWKit-style standard-crossbar forwards consume the post-write apparent
  effective state. The hidden persistent state controls subsequent pulse
  updates and is a required robustness diagnostic, but it must not be
  substituted for the apparent forward state. Report both states explicitly.
- Match the frozen teacher, logical dimensions, data split, minibatch order,
  objective, assignment/endpoint seeds, recovery budget, and checkpoint
  selection wherever the two architectures permit it. When topology changes
  the population shape, record rather than conceal the different identity
  and device counts.
- Report 39,700 logical weights for the `784-50-10` comparator and distinguish
  programmable active OM states from fitted reference values. Do not infer a
  fabricated device count from AIHWKit's abstract reference subtraction.
- Matched arms must keep the DRN topology, solver, data split, teacher or base
  checkpoint, and device assignment fixed unless one of them is the explicitly
  declared intervention.
- The realistic P&V controller may receive only continuous targets, apparent
  normalized verify values, its own history, and calibration-partition
  population estimates. Keep persistent states and hidden device parameters
  on the logging side of the controller port.
- Treat exact persistent target reachability, verify-window intersection, and
  apparent acceptance as three different facts. Bound classification may pair
  conditioned RESET/SET states only on the analysis side; an apparent noisy
  admission must not be relabelled as a persistently reachable target.
- A frozen-model fresh-array test must remap the frozen off-chip FP32 master
  independently through each target array's own bounds and codebook. Do not
  copy a source array's realized codebook or stochastic endpoint to a fresh
  array. A recovered-state transfer is a distinct arm and must name the
  transferred hidden persistent state explicitly.
- Several endpoint seeds on one assignment are repeated programming
  realizations on one array, not independent arrays. Array-transfer evidence
  must include several distinct target assignment seeds and report
  assignment-level as well as pooled summaries.
- Preserve per-identity AIHWKit construction seeds and explicit per-trajectory
  conditioning/programming seeds. Do not replace the explicit pulse plant with
  a native CPU tile unless its cycle-to-cycle RNG becomes serializable and an
  exact-replay test is added.

## Synapse data

- Distinguish measured-trace evidence from fitted device presets and synthetic
  endpoint models. The imported CIFAR studies use AIHWKit 1.1.0 presets and
  Gaussian PCM controls and must remain labelled exploratory and model-based;
  do not describe them as measurements on fabricated hardware.
- Claims about measured synaptic data must use that data. Synthetic noise models
  and generic device presets must remain clearly labelled controls.
- Use the dataset, placement, verification, cohort, and provenance rules in
  [`docs/synapse_data.md`](docs/synapse_data.md).
- Do not silently replace the measured dataset, alter its preprocessing, or
  mix device cohorts between arms.

## Initialization

- Every study must predeclare its initialization path and follow
  [`docs/initialization_protocols.md`](docs/initialization_protocols.md).
- HWA and non-HWA arms must start from the same logical state. Recovery arms
  must start from the same explicitly named deployed state unless
  initialization itself is the intervention.
- Never discover or substitute the newest checkpoint. Record explicit input
  paths and content hashes.

## Experimental workflow and authority

Use the [experiment-loop skill](.agents/skills/experiment-loop/SKILL.md) and
[execution policy](docs/experiment_workflow.md). They supersede the former mandatory
study/manifest lifecycle and external closeout-skill instructions that conflict.
Start from the question and decision. Small pilots use one evolving note in
`campaigns/pilots/`; sustained work uses campaign experiment/result notes. Ideas,
explorations and hypotheses are useful relationships, not obligatory paperwork stages.

Assigned execution authorizes implementation, proportional smoke/readiness checks,
launch, monitoring, collection, interpretation and understood retries within scope and
budget. Planning alone does not authorize launch. Make cases, target, budget, expected
duration and result paths visible before substantial execution. Escalate material
scientific choices, missing access, unrelated effects or scope/budget changes.
This migration does not expand remote, shared-GPU or overnight permissions.

Use the existing public ebl CLI, exact configs/inputs and native run bundles. Keep one
real monitoring owner per running case. Preserve failed/partial evidence and distinguish
process completion, artifact validity, coverage and scientific review. Review can conclude
negative or inconclusive; partial coverage must remain labelled partial. A full execution
assignment continues through interpretation. Claim on-chip training is needed only when
a predeclared HWA-only control underperforms and matched recovery closes the specified gap.

Experiment/pilot notes own specifications, current handoffs and output paths;
result/pilot notes own measurements and interpretation. Generated campaign ledgers
provide navigation only. `docs/current_simulations.md` and
`docs/experimental_manifest.md` are historical snapshots: do not refresh or append.
Keep historical records as provenance without their old update requirements.
Do not edit human `docs/current_state.md` without explicit permission, or read/edit
`docs/my_notes.md` unless asked. Keep small records versioned and large artifacts ignored.
