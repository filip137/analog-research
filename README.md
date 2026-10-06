# Crossbar hardware-aware training and on-chip learning

This HWA worktree studies hardware-aware training, deployment and array-specific
recovery on simulated crossbar arrays. It includes dense MNIST/CIFAR networks and
full CIFAR-10/100 ResNet-32 networks with a frozen digital prefix and four or eight
analog suffix convolutions plus the classifier.

Start with the [CIFAR scientific campaign](campaigns/cifar-crossbar-hwa-recovery/README.md)
and [reproduction guide](docs/cifar_reproduction.md). The campaign records historical
results, their controls and limitations, and the October 2026 migration from the
original `server_code/.codex/worktrees` checkouts.

The supported execution interface is the native experiment CLI:

```bash
python -m ebl describe --experiment cifar_crossbar_fault_sweep.v1
python -m ebl describe --experiment cifar10_ibm_om_crossbar.v1
```

Each training invocation requires an exact config, explicit input artifacts and
an output root. It creates a new native run bundle with resolved configuration,
source/runtime identity, metrics and checkpoints. The CIFAR reproduction helper
prints these commands using verified local inputs; see the guide for setup.

New experimental loops are campaign **lifecycles**: one strict JSON definition
states the question, devices and defects, then the deployment of the digital
weights, HWA, program-and-verify and on-chip training. Each step calls a
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
[runtime commands](docs/experiment_runtime.md) for the 14 registered experiment
families. EP/DRN, LoRA and MNIST crossbar v1 execution code is available at
pre-cleanup commit `16938bf`; historical scientific records remain in this tree.
The framework originated in the
[energy-based-learning project](https://github.com/rain-neuromorphics/energy-based-learning).
