# Experiment runtime

## Design

The repository has three layers:

```text
experiments/small_network/  compose one versioned scientific workflow
             |
             +--> training/ generic execution, modifiers, probes, updates
             |
             +--> model/    energy functions, networks, parameter adapters
```

This keeps the useful shape of the original energy-based-learning repository:
the model and training packages describe reusable mechanics, while experiments
assemble them. Dataset choice, CLI paths, artifact layout, and paper-specific
choices do not belong in model classes or generic training loops.

`small_drn.v1` has four independent choices:

| Axis | Implementations | Responsibility |
| --- | --- | --- |
| training algorithm | `ep`, `backprop`, `digital` | estimate a complete minibatch gradient (`backprop` is BPTT through the configured minimizer iterations) |
| model adapter | `none`, `passive_low_rank`, `digital_low_rank`, `passive_layerwise_low_rank` | define the trainable parameterization |
| parameter modifier | `none`, `add_normal` | temporarily alter parameters during solver phases |
| update backend | `direct`, `tiki_taka`, `program_verify`, `measured_cohort_a`, `measured_cohort_a_sign_sgd`, `measured_cohort_a_one_pulse_down`, `measured_cohort_b`, `measured_cohort_b_lora` | apply or accumulate the completed gradient; measured backends project trainable arrays onto assigned measured device curves; the signSGD variant discards gradient magnitude before its cohort-A projection, while the one-pulse-down variant permits only a local step toward lower conductance |

LoRA is therefore not another name for Tiki-Taka. Passive low-rank recovery
changes which parameters produce the effective model weights; Tiki-Taka
changes how an already-computed gradient is accumulated and transferred. They
can share the same experiment, engine, checkpoint, and evaluation structure
without being coupled to each other. The focused circuit, configuration, and
checkpoint contract is documented in
[Passive low-rank adapter](passive_low_rank_adapter.md).
The hybrid ideal-readout experiment is documented in
[Digital low-rank recovery](digital_low_rank_recovery.md).
The two-edge physical experiment is documented in
[Passive layerwise low-rank recovery](passive_layerwise_low_rank_recovery.md).
Its frozen-base, fully reset measured-device protocol is documented in
[Measured cohort-B memristor LoRA recovery](measured_cohort_b_lora_recovery.md).

Each worktree's experiment definition lists only combinations implemented in
that worktree. The shared foundation advertises base direct and Tiki-Taka
training; the HWA branch adds `add_normal` combinations, and the LoRA branch
adds `passive_low_rank`, `digital_low_rank`, and
`passive_layerwise_low_rank` combinations. The measured-endpoint branch adds
`program_verify` for full BPTT and passive-layerwise LoRA BPTT. Unlisted
combinations, including LoRA plus the temporary `add_normal` modifier, are
rejected before numerical execution.

For the four-device MNIST protocol,
`measured_cohort_a_sign_sgd` applies
`target <- target - learning_rate * sign(gradient)` to the ideal conductance
shadow and then uses the same global-nearest measured-state projection as
`measured_cohort_a`. A zero gradient produces no shadow update. This is a
fixed-magnitude endpoint-projection experiment, not a sequential one-pulse
potentiation/depression model; the selected measured state may be anywhere on
the assigned raw trace.

`measured_cohort_a_one_pulse_down` is a separate, fail-closed protocol. It
first fits every cohort-A source trace to an isotonic non-increasing curve and
performs the declared teacher-mapped common-window initialization with one
global-nearest write. The registered four-device form uses single encoding
with `dual_rail_quad_common_window`; the eight-device form uses differential
encoding with paired `G+`/`G-` common windows and calibrates thresholds for all
four stable conductance tensors. During fine-tuning, a raw gradient strictly
greater than the stable parameter's
`positive_gradient_threshold_by_parameter` value increments that cell's pulse
index by exactly one, selecting the next non-increasing conductance. Equality,
a sub-threshold positive gradient, zero, or a negative gradient holds. Omitting
the mapping is the explicit zero-threshold control. A cell already at the final
pulse also holds. The backend does not call SGD, use learning-rate magnitude,
accumulate a digital shadow, or run global-nearest projection after
initialization. Configs therefore declare learning rates `[0.0, 0.0]`.
Checkpoints retain the stable-key threshold mapping, pulse indices, and
cumulative gating, saturation, pulse-jump, and conductance-monotonicity
diagnostics.
The trajectory is over isotonic fits to the measured endpoint sequence; it is
an explicit simulator rule, not independent evidence that one physical pulse
will reproduce the fitted next state.

## Core training interfaces

`training/core` is the per-step code that workflow-managed runs execute:
`engine`, `sgd` (EP, backprop and `DirectReadoutGradient`), `optimizers`,
`modifier`, `batch`, `probes` and `guards`. `training/lab` is the interactive
harness used only by `labs/`: `epoch`, `statistics`, `monitor` and
`diagnostics`. Core never imports lab.

Experiment setup constructs the model, cost, free-phase minimizer, gradient
estimator, update backend, and parameter modifier. `ExperimentComponents`
groups the network, cost, minimizer, parameters, estimator, and optimizer.
The modifier is passed separately to the runner or engine.
`EvaluationComponents` holds only the network, cost, and free-phase minimizer.
For unlabelled diagnostics, evaluation
accepts `cost_fn=None` with `Batch(inputs, targets=None)`.

`training.core.engine.train_epoch` owns the minibatch order: set input, settle,
set target, measure the free phase, compute and assign gradients, restore the
modifier, update once, and clamp. Gradient computation and measurements stay
inside the modifier context. `evaluate` holds its modifier context across
the whole loader. Gradient accumulation and device writes belong to the
update backend's `step()` implementation.

### One loop path

Every per-batch loop of `small_drn.v1`, `mnist_relu_drn_kd.v1` and
`mnist_relu_drn_reset*.v1` is a call to `train_epoch` or `evaluate`. The
experiment's composition decides which stages run:

| Pass | `differentiator` | `optimizer` | Used by |
|---|---|---|---|
| training step | set | set | `small_drn.v1`; KD `_train_epoch` (also gradient-based deployed recovery); RESET `train_epoch` |
| gradient pass, no update | set | `None` | RESET LR probe; direct-pulse recovery calibration |
| update without gradient | `None` | recovery, `clamp_after_update=False` | rail-refresh recovery |
| evaluation | — | — | KD `_evaluate`; RESET `evaluate`; calibration and post-hoc logit collection; decomposition replay |

Experiment-specific behaviour is plugged in, not written as a loop:

- `training.core.batch.prepared_batches` (and its MNIST wrapper
  `experiments.mnist_shared.teacher_batches`, which builds
  `TeacherTargets(logits, labels)`) applies batch and sample limits, an
  early `stop()` and device transfer without pulling extra loader items;
- the costs' `set_target` hook applies those targets;
- `ComposedModifier` joins the HWA modifier with the KD forward-logit gain
  (`ForwardGainModifier`);
- probes measure each free phase in training and each batch in evaluation;
  `PerBatchProbe` collects values whose reduction stays with the caller;
- `FiniteGradientGuard(names, label=...)` runs before any observer that reacts
  to gradient values, such as the RESET LR-safety monitor.

`tests/test_training_loop_golden.py` replays these loops bit-exactly against
goldens captured before the migration.

### One train-phase lifecycle

Above the loop, every `run_train` of `mnist_relu.v1`, `small_drn.v1`,
`mnist_relu_drn_kd.v1` and `mnist_relu_drn_reset*.v1` is one *train phase*
run by `experiments.lifecycle.train_phase`:

- `BestCheckpoint` is the strict global-best selection; it alone writes
  `weights.pt` (`restore` from a resume checkpoint, `seed` for an epoch -1
  candidate, `offer` after each validation);
- `run_epochs` fixes the order train → validate → select → epoch-boundary
  `resume.pt` → cadenced metric → observers, and leaves a `resume.pt` even
  when a continuation runs no epoch;
- `run_phase` records completion or failure (including interrupts).

Families supply hooks: build the stack and initialize or resume it, train
one epoch, validate, the selection record and payload, the resume progress
state and the epoch metric. LR selection with its exact production restart
is still family-owned, and IBM OM deployed recovery (`run_recovery_train`)
is not yet a phase. `tests/test_run_lifecycle_golden.py` replays complete
`ebl train` bundles bit-exactly against goldens captured before this move;
`python tests/run_lifecycle_golden_cases.py verify CASE...` replays chosen
cases.

`training.lab.epoch.Trainer(components, loader, reset_input=...)` and
`Evaluator(components, loader, reset_input=...)` bind loaders and statistics
to these shared loops. Training requires an explicit reset policy: existing
ordinary lab trainers reset, while custom lab trainers and `small_drn.v1`
carry states between compatible batches. Register statistics with
`add_statistic(statistic, phase="free")` or `phase="gradients"`; numeric
statistic-list indices are no longer accepted.

Observers consume `FreePhaseEvent`, `GradientsReadyEvent`,
`BeforeUpdateEvent` (after modifier restoration), and `AfterUpdateEvent`
(after clamping). `training.lab.diagnostics.GradientUpdateObserver` snapshots
parameters only when explicitly attached and reports the actual clamped
update. `LayerMeasurements` is a reusable lab observer, and
`training.core.guards.FiniteGradientGuard` rejects non-finite gradients before
the update in both `small_drn.v1` and the lab trainers.
Lab layer measurements retain raw energy-gradient infinity norms over all
layers; the solver's projected-current residual probe is a separate metric.
`BetaSize` is a separate diagnostic pass: it compares mean absolute
displacements against a nonzero nudged phase and restores voltages and
nudging, including on failure.

`training.core.optimizers` exports `SGDOptimizer` and `build_optimizer`; these
are no longer exported from `monitor` or `tiki_taka`. Core EP and backprop
remain in `training.core.sgd`. The unused recurrent-backprop and contrastive
estimators and the `detailed_gradients` trajectory dispatcher were removed;
recover them from git history if a study needs them.

Lab setup registers standard statistics once with
`training.lab.statistics.add_standard_statistics`. The monitor consumes those
registrations through `Monitor(trainer=..., evaluator=..., scheduler=...,
save_model=..., path=...)`; it does not construct statistics or optimizers.
Existing series names and units are retained. In particular, the six weight
distribution statistics retain their shared TensorBoard tag and historical
order for existing plotters, and the legacy pickle entry retains the last
(`abs_std`) series. Duplicate registrations of the same statistic are rejected.
Use `monitor.run(num_epochs)` to open and close the TensorBoard writer for
each run; repeated `run(1)` calls preserve epoch numbering and series history.

## Commands

This document describes the numerical runtime. The plan-to-review lifecycle
for multi-run studies is documented separately in
[`experiment_workflow.md`](experiment_workflow.md).

Inspect the machine-readable protocol:

```bash
python -m ebl describe --experiment small_drn.v1 --json
```

Run one mode:

```bash
python -m ebl train \
  --config examples/small_drn/base.json \
  --output-dir runs

python -m ebl linspace \
  --config examples/small_drn/base.json \
  --weights runs/<training-run>/checkpoints/weights.pt \
  --output-dir runs

python -m ebl validate \
  --config examples/small_drn/base.json \
  --weights runs/<training-run>/checkpoints/weights.pt \
  --output-dir runs
```

The JSON config is strict, nested, and immutable after parsing. It records
scientific intent. Operational paths are explicit CLI arguments:

- `--weights` initializes/evaluates the complete named parameter set;
- `--base-weights` initializes only the base group, leaving an adapter
  independently initialized;
- `--resume` restores a full epoch-boundary state;
- `--output-dir` is the parent in which a new exclusive run directory is
  created.

These initialization options are mutually exclusive.

## MNIST dataset contract

`small_drn.v1` accepts `data.dataset: "mnist"` for train and validate modes.
The loader uses the canonical 60,000/10,000 torchvision splits, applies
normalization with mean `0.1307` and standard deviation `0.3`, and flattens
each image to 784 values before the DRN creates its positive/negative input
pair. A full MNIST DRN therefore uses `model.dims[0] == 1568`.

MNIST is never downloaded during config parsing or execution. The runtime
looks under `~/datasets/mnist` by default; set `EBL_MNIST_ROOT` to select an
existing alternative root. `data.num_points` may select a deterministic
training subset for smoke tests, while the held-out test split remains
canonical. Linspace mode is rejected for MNIST.

## Run contract

Each command creates one new run directory:

```text
<output-dir>/<run-id>/
  config.resolved.json
  manifest.json
  status.json
  metrics.jsonl
  result.json
  artifacts/
  checkpoints/
  logs/
```

`result.json` is the only campaign-facing completion contract. It records the
terminal status and content-hashed artifacts. A failed run retains its
manifest and failure status; it is not silently reused.

Named weights and full resume state are intentionally different:

- `weights.pt` contains stable parameter keys and is suitable for evaluation
  or transfer.
- `resume.pt` contains model, optimizer, scheduler, modifier, tensor-bearing
  runtime continuation state (including carried equilibria), progress, random
  number generator, and named dataloader-generator state at an epoch boundary.

For `program_verify`, selected `weights.pt` stores the realized device tensors.
Only `resume.pt` additionally stores the clean digital targets and the exact
next device-noise generator state needed to continue a noisy-write trajectory.

Physical device floors are retained by default. CMO/HfOx accepts three
explicit mappings:

- `affine_floor` maps the complete logical interval across `9–88.199997 µS`
  and leaves the minimum as an effective DRN connection;
- `literal_conductance` directly scales DRN conductance by `G_max` and clips
  sub-floor targets, so it is the absolute-scale/hard-clipping stress test;
- `normalized_offset` uses affine physical targets but subtracts and rescales
  the floor on conversion back, so it is an explicit
  differential/reference-cancellation comparison.

Selected `weights.pt` always stores the realized effective DRN tensors for the
chosen interpretation. CMO programming reports distinguish total deployment
distortion relative to the clean digital target from stochastic residual
error relative to the ideal mapped target. The mapping rationale and measured
voltage effects are in
[Finite conductance floors in DRNs](conductance_floor_mitigation.md).

Restore is transactional. All payloads are validated before mutation, and any
failure rolls back already-restored components. AIHWKit-backed runs may
declare `stateful_nondeterministic`; an unsupported runtime must not emit a
resume checkpoint.
For `small_drn.v1`, every numerical setting must match on resume; only the
target `modes.train.num_epochs` horizon may be extended.

Old positional tensor lists are not auto-detected:

```bash
python -m ebl checkpoint import-legacy \
  --config examples/small_drn/base.json \
  --source legacy-model.pt \
  --kind full \
  --output named-weights.pt
```

Use `--kind base` only when the positional file contains base parameters and
an adapter must remain independently initialized.

## Cross-worktree campaigns

The campaign controller treats each Git worktree as a separate executable. It
first calls that worktree's `ebl describe ... --json`, records the exact
commit/dirty fingerprint, and then launches stages as subprocesses. This
prevents Python import-cache and module-path contamination between branches.

```bash
python -m ebl campaign run \
  --manifest examples/campaigns/base_train_validate.json \
  --output-dir campaign-runs \
  --dry-run
```

A campaign manifest declares targets, stages, dependencies, configs, and
explicit artifact references. A later validation stage can consume the
`weights` artifact produced by a training stage without guessing a filename.
The checked-in example uses the repository's current worktree layout and
Python environment; adjust its target paths when copying it to another
machine.
Use `--resume` to reuse a completed stage only when its config, inputs, source
identity, and result hash still match.

By default targets must be clean. `--allow-dirty` is intended for deliberate
local experiments and records a dirty-source fingerprint.

## Worktree development workflow

Develop cross-cutting runtime changes once on the shared foundation branch.
Then rebase the feature branches and implement only their extension:

```text
shared foundation
  +-- HWA branch: add_normal ParameterModifier
  +-- LoRA branch: passive_low_rank model adapter
  +-- hybrid recovery: digital_low_rank model adapter
  +-- layerwise passive recovery: passive_layerwise_low_rank model adapter
  +-- measured endpoint recovery: program_verify update backend
```

Do not merge HWA and LoRA branches merely to obtain the common pipeline.
Compare them with a campaign. Promote an experimentally validated combination
to the explicit capability matrix only after its focused and numerical tests
pass.

## Adding an experiment or extension

For a new experiment:

1. Add `experiments/<name>/config.py` with a versioned immutable schema.
2. Register a stable ID explicitly in `experiments/definitions.py`.
3. Build numerical objects in the experiment package.
4. Reuse `training.core.engine`, probes, named checkpoints, and `RunStore`.
5. Add a small nested example and CLI smoke test.

For a new extension, prefer one protocol over a conditional in the engine.
Test its lifecycle/order separately, then add a numerical acceptance case for
each combination exposed by `describe --json`.


## Question-led run inspection

New pilots use one [question-led record](experiment_workflow.md#records), exact
configs and native bundles; no separate JSON study is mandatory. Inspect explicit
paths with `python -m ebl runs inspect RUN_DIR --json`. Add `--verify-artifacts`
for hash validation and `--require-complete` to fail on unfinished/failed cases.
Inspection does not modify bundles or documents and does not infer scientific
coverage or conclusions. Legacy study finalization writes only a local receipt.
