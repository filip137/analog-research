# Reproducing the migrated CIFAR crossbar experiments

The [scientific campaign](../campaigns/cifar-crossbar-hwa-recovery/README.md)
separates the dense pilot, ResNet mapping, PCM adaptation, full-cohort sweeps and
OM controller studies. All nine native CIFAR families are registered in `ebl`.
The integrated code, strict configs and tests are versioned in this branch.

## Local evidence and environment

The October 4 migration retained the original `results/cifar*` and `artifacts/cifar*`
trees, including teachers, datasets, feature caches, physical populations, P0,
selected/final checkpoints, plots, metrics and frozen source archives. These large
files are ignored by Git. When moving to another host, copy these trees and
`artifacts/checkpoint-retention-20260925/` as well as the code. Preserve hardlinks
with an archival copy method. The current copies are independent filesystem
reflinks; they do not depend on the old worktrees being present.

The migration audit and its detailed inventory are under
[`results/cifar-migration-20261004/`](../results/cifar-migration-20261004/).
`artifact-inventory.jsonl.gz` records every copied source/destination and SHA-256;
`additional-artifacts.json` covers the materialized CIFAR-10 dataset and the
source receipts extracted from the preserved sweep archives. A compact versioned
receipt is in the [campaign](../campaigns/cifar-crossbar-hwa-recovery/migration-receipt.json).

This host has the following tested interpreters:

```bash
cd /home/filip/analog-research/hwa
export EBL_CIFAR_ROOT="$PWD/artifacts/cifar_inputs/data"
export EBL_AIHWKIT_PYTHON=/home/filip/miniconda3/envs/aihwkit/bin/python
export MPLCONFIGDIR=/tmp/hwa-matplotlib
/home/filip/miniconda3/envs/py312/bin/python -m ebl describe --experiment cifar_crossbar_fault_sweep.v1
```

The main interpreter is Python 3.12.12, PyTorch 2.5.1+cu121 and torchvision
0.20.1+cu121; the remaining tested packages are pinned in
[`requirements-cifar.txt`](../requirements-cifar.txt). The separate native sampler
has AIHWKit 1.1.0 and PyTorch 2.11.0+cu130. It passed the independent population,
PCM pulse, endpoint-noise and drift-compensation parity checks. Set these paths
to the corresponding interpreters on another machine. Training uses the GPU
specified by the scientific config; choose a compatible CUDA build for that GPU.

The historical manifests retain the actual host, Python and PyTorch versions for
each run. The local portability tests and report comparisons do not establish
bitwise training equivalence across CUDA/PyTorch versions.

## Prepare an exact run's local command

Choose an explicit historical native run. For example:

```bash
/home/filip/miniconda3/envs/py312/bin/python -m experiments.cifar_crossbar.reproduce \
  --run results/cifar-sweep-collected/v2/results/cifar10-om-conv4-fault-sweep-v2/runs/screen1/20260918T180847.578144Z-b43c5c49-65628ad1 \
  --output-dir results/cifar-reproduction/cifar10-om-conv4-screen1 \
  --format shell
```

This prints a command; it does not launch it. Run the printed command when its
training budget is intended. The default JSON output also records the original
runtime and source receipt. The helper:

- validates the historical resolved-config hash and current strict parser;
- resolves every explicit checkpoint/cache/population/selection input by SHA-256
  inside this HWA worktree, with no old-host or newest-checkpoint fallback;
- keeps scientific settings unchanged and requires a new output directory;
- sets the copied dataset location and clears `EBL_SOURCE_RECEIPT`, because this
  replay uses the integrated HWA code rather than pretending to be the old source.

All **755** archived native attempts, including two failed attempts, passed this
config/input resolution check. The audit's `reproduction-recipes.json` contains
their individual commands. Counts of native attempts include canaries, revisions
and implementation stages; they are not counts of independent arrays.

For a fresh campaign, use the versioned configs/campaign generators in
`examples/cifar_crossbar/`, `examples/cifar10_crossbar/` and
`experiments/cifar_crossbar/`. The dense pretraining config is
`examples/cifar10_crossbar/pretrain.json`. Its dataset can be relocated with
`EBL_CIFAR_ROOT` without changing preprocessing or split receipts. The native
sampler override also applies to dense OM HWA/recovery.

To inspect or reproduce the exact historical implementation, use the per-study
`source-*.tgz` archives under `artifacts/*_sources/` and their source receipts.
The migration also captured both original worktrees, including dirty ResNet
source, in `resnet-source.tar.gz` and `dense-source.tar.gz` at the audit root.
Extract into a separate directory and use the historical environment/receipt;
do not unpack over the integrated HWA tree. Old remote launch scripts preserve
historical host choices; their paths are provenance, not a new launch instruction.

## Rebuild reports without changing historical outputs

From the repository root, with the main interpreter active:

```bash
python -m experiments.cifar_crossbar.sweep_report \
  --root results/cifar-sweep-collected/v2 --revision 2 \
  --output results/cifar-reproduction/reports/sweep
python -m experiments.cifar_crossbar.full_epoch_report \
  --root results/cifar-full5-collected/v2 \
  --output results/cifar-reproduction/reports/full5
python -m experiments.cifar10_crossbar.analyze \
  --output results/cifar-reproduction/reports/dense
python labs/tools/build_cifar_five_stage_tables.py \
  --root . --output results/cifar-reproduction/reports/five-stage
```

The OM followup report APIs accept the retained execution ledgers. Open-loop
reporting uses the content index to relocate both current and baseline results:

```python
from pathlib import Path
from experiments.cifar_crossbar.reproduce import ArtifactIndex
from experiments.cifar_crossbar.om_open_loop_report import report as open_loop
from experiments.cifar_crossbar.closed_loop_lr_report import report as closed_loop

open_loop(Path("artifacts/cifar_om_open_loop_sources"),
          Path("results/cifar-reproduction/reports/open-loop"),
          artifact_index=ArtifactIndex())
closed_loop(Path("artifacts/cifar_om_closed_loop_lr_sources"),
            Path("results/cifar-reproduction/reports/closed-loop"))
```

The earlier PCM families retain their analysis modules, exact configs, execution
ledgers and original reports. Use the recorded experiment/result links to select
the appropriate arrays and source revision instead of pooling study generations.

## Retention and validation limits

Before migration, the September 25 cleanup intentionally removed 46,400 unique
epoch 1–4 checkpoint files (90,752 paths including hardlinked aliases). All missing
native artifacts encountered in the migration audit match that inventory. Final,
selected, P0, progress/resume states and numeric per-epoch curves were retained.
Intermediate deleted binary states cannot be replayed from the archive; producing
them again requires training. See the copied
[retention note](../results/cifar-migration-20261004/CHECKPOINT_RETENTION_20260925.md).

Strict bundle inspection still reports historical omissions: 112 runs reference
deleted checkpoints and 618 have an original prepared-study digest different from
the collected `study.json` (these groups overlap). The original manifests/studies
were copied unchanged. The audit verifies retained artifact hashes and distinguishes
these pre-existing receipt issues from migration damage; it does not rewrite the
history to make every bundle pass strict inspection.

Historical evidence stays `imported-summary`. The migration has its own
`validated-local` audit. This work did not rerun full scientific training, finalize
pending scientific reviews, or establish that on-chip learning is generally necessary.
