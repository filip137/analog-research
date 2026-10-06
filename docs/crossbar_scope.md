# Supported crossbar worktree

This branch owns crossbar hardware-aware training (HWA), deployment and on-chip
learning. Digital MNIST teachers and IBM ReRAM characterization remain as inputs
and controls. General EP/DRN and LoRA development belong to sibling worktrees.

## Supported experiments

| Family | Purpose |
| --- | --- |
| `crossbar_lifecycle.v1` | One stage of a campaign lifecycle: devices/defects, mapping, HWA, P&V, on-chip training |
| `cifar10_ibm_om_crossbar.v1` | Dense CIFAR-10 OM HWA and pulse recovery |
| `cifar_resnet_suffix_recovery.v1` | CIFAR-10/100 ResNet-32 suffix HWA and recovery |
| `cifar_pcm_fault_recovery.v1` | PCM faults and teacher-KL recovery |
| `cifar_pcm_hwa_comparison.v1` | HWA and corrupt-device training controls |
| `cifar_pcm_recovery_epochs.v1` | Fixed-array PCM recovery schedules |
| `cifar_crossbar_full_epochs.v1` | Full-epoch OM and PCM adaptation |
| `cifar_crossbar_fault_sweep.v1` | Mixed-rate fault sweeps |
| `cifar_om_open_loop.v1` | Open-loop OM programming and recovery |
| `cifar_om_closed_loop_lr.v1` | Closed-loop OM learning-rate selection |
| `mnist_relu.v1`, `mnist_relu.v2` | Digital teachers, training and validation |
| `mnist_ibm_om_crossbar_relu.v2` | Staged MNIST HWA, deployment and on-chip learning |
| `ibm_reram_program_verify.v1` | ReRAM programming characterization |

Use `python -m ebl describe --json` to inspect the registry. Scientific settings
live in strict JSON configs; checkpoints, populations and data are explicit inputs.
See [runtime commands](experiment_runtime.md) and the [CIFAR reproduction guide](cifar_reproduction.md).
The tested Python 3.12 stack is in `requirements-cifar.txt`. Population sampling
and native parity use a separate AIHWKit 1.1.0 interpreter selected by
`EBL_AIHWKIT_PYTHON`; the explicit pulse plant runs in PyTorch.

## Shared implementation

- `workflow/`: the standard lifecycle. Campaign lifecycle definitions call one
  reusable function per step (devices, deployment, HWA, P&V, on-chip); see
  [the lifecycle contract](crossbar_lifecycle.md). New crossbar loops use it;
  the historical `cifar_*` families stay frozen for reproduction.
- `training/parameters.py`: tensor parameters, named bindings and catalogs.
- `training/ibm_reram_hwa.py`: population sampling, fingerprints and NPZ storage.
- `experiments/mnist_analog_relu/crossbar_common.py`: shared programming,
  evaluation and off-chip adaptation helpers.
- `training/ibm_om_standard_crossbar.py`, `training/ibm_om_tiki_taka.py` and
  `training/closed_loop_crossbar_adam.py`: crossbar plants and updates.
- `training/checkpoint.py`: existing named-weight and resume schemas. Historical
  `drn.*` schema strings are wire-format identifiers, retained for compatibility
  with existing digital-teacher checkpoints.

## Retired interfaces and historical evidence

EP/DRN solvers, LoRA adapters, their experiment entry points and the older
`mnist_ibm_om_crossbar_relu.v1` runtime are retired. The `linspace`,
`checkpoint import-legacy` and `--base-weights` CLI surfaces are retired too.
Their implementations and exclusive configs/tests remain at pre-cleanup commit
**`16938bf`**. Reproduce an old study in a separate checkout at that revision with
its original environment and explicit input artifacts. Retired IDs fail with
that revision in the error rather than selecting a different experiment.

Historical scientific notes, reference receipts, native bundles, datasets, plots,
tables and frozen source archives remain evidence. Read old documentation links
to removed implementations against `16938bf`. Native inspection is read-only and
does not require an experiment to remain registered:

```bash
python -m ebl runs inspect /absolute/path/to/run --verify-artifacts --json
```

The [CIFAR campaign](../campaigns/cifar-crossbar-hwa-recovery/README.md) retains the
imported results and their limitations. Cleanup creates no new training evidence
and does not alter frozen manifests. See the
[cleanup record](../campaigns/pilots/20261004-crossbar-worktree-cleanup.md) for checks.
