# Crossbar hardware-aware training and on-chip learning

This worktree studies hardware-aware training, deployment and on-chip adaptation
of simulated crossbar arrays under **corrupted devices, write noise and drift**.
OPT-125M is the current transformer target. Completed MNIST and CIFAR studies
remain reference evidence.

Start with [AGENTS.md](AGENTS.md) for the shared scientific scope and the
[experiment workflow](docs/experiment_workflow.md). The branch responsible for
each experimental run declares its simulation models, parameters, controls,
inputs, metric and budget in its experiment or pilot note and lifecycle.
[`../on-chip-learning/`](../on-chip-learning/README.md) owns shared crossbar/DRN
study records and lists the active campaigns.

Historical CIFAR results, controls and migration provenance are in the
[CIFAR campaign](campaigns/cifar-crossbar-hwa-recovery/README.md) and
[reproduction guide](docs/cifar_reproduction.md).

The supported execution interface is the native experiment CLI:

```bash
python -m ebl describe --experiment crossbar_lifecycle.v1
python -m ebl describe --json
```

Each training invocation requires an exact config, explicit input artifacts and
an output root. It creates a new native run bundle with resolved configuration,
source/runtime identity, metrics and checkpoints. The CIFAR reproduction helper
prints these commands using verified local inputs; see the guide for setup.

New experimental loops are campaign **lifecycles**: one strict JSON definition
states the question and selected device effects, then the relevant deployment,
HWA, program-and-verify, retention and adaptation stages. Each step calls a
reusable function in `workflow/`; see the [lifecycle contract](docs/crossbar_lifecycle.md).

OM recovery models physical pulse updates with held apparent-state forwards.
The main PCM recovery studies use Gaussian endpoint reprogramming. Both compute
gradients and optimizer state digitally. Calibration, weight-only recovery and
joint recovery are reported separately, including matched no-weight-write
controls and physical update counts.

## Code and evidence

- `workflow/`: the standard lifecycle stages and their strict schema.
- `experiments/cifar_crossbar/`: ResNet mapping, HWA/CDT, deployment, recovery,
  strict configs, reports and input relocation.
- `experiments/cifar10_crossbar/`: the separate dense CIFAR-10 experiment.
- `experiments/mnist_analog_relu/` and `training/`: shared crossbar plants,
  population samplers, HWA and update controllers.
- `examples/cifar_crossbar/`, `examples/cifar10_crossbar/`: exact scientific configs.
- `campaigns/`: question-led experiment/result records and generated ledgers.
- `results/` and `artifacts/`: ignored native bundles, checkpoints, datasets,
  source snapshots, plots and migration inventories. A Git checkout alone does
  not contain these historical data.

New work follows the [experiment workflow](docs/experiment_workflow.md).
`docs/current_simulations.md` and `docs/experimental_manifest.md` are historical
snapshots; current specifications and conclusions belong to campaign or pilot notes.

See the [supported-scope guide](docs/crossbar_scope.md) and
[runtime commands](docs/experiment_runtime.md) for execution guidance;
`python -m ebl describe --json` lists the current registered experiment families.
EP/DRN, LoRA and MNIST crossbar v1 execution code is available at
pre-cleanup commit `16938bf`; historical scientific records remain in this tree.
The framework originated in the
[energy-based-learning project](https://github.com/rain-neuromorphics/energy-based-learning).
