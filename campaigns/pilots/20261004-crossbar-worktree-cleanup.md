# Crossbar worktree cleanup

State: implementation and compatibility verification complete, 2026-10-04.
Authorization: implement the accepted crossbar-only cleanup plan, including retirement
of the older MNIST crossbar v1 workflow. Base revision: `16938bf`.

## Decision and resulting scope

HWA owns crossbar training, deployment and on-chip learning. Keep the nine CIFAR
families, staged MNIST crossbar v2, two digital MNIST teacher interfaces and IBM
ReRAM characterization: 13 registered families. LoRA and general EP/DRN development
belong to sibling worktrees. See [supported scope](../../docs/crossbar_scope.md).

Removed the EP/DRN solver and energy-model tree, LoRA composition roots,
EqProp Tiki-Taka, exclusive training mechanisms, experiments, scripts, configs,
campaign manifests and tests. The older MNIST crossbar v1 runtime and CLI
`linspace`, `checkpoint import-legacy` and `--base-weights` are retired.
The 17 retired experiment IDs report their historical revision explicitly.
All removed implementations remain reproducible from a checkout at `16938bf`
with their original environment and available inputs.

Shared parameter/binding/catalog primitives now live in `training/parameters.py`;
OM population sampling/storage remains in the smaller `training/ibm_reram_hwa.py`.
Twenty shared crossbar helpers moved to `crossbar_common.py`. Crossbar Tiki-Taka,
closed-loop Adam, pulse programming, endpoint models and native artifact inspection
remain. Named-weight keys, resume formats and population schemas are unchanged.
Packaging includes only supported runtime packages and CIFAR analysis tools.

The staged off-chip HWA settings bridge omitted `programming_error`, which the
shared adaptation helper accessed. It now supplies `None` for this supported
stochastic-apparent-HWA stage. A real one-minibatch adaptation test verifies the
stage bridge, held apparent-state evaluation, optimizer step and exact replay.

## Verification

- Retained suite: **1,830 passed, 12 skipped** in the Python 3.12/PyTorch 2.5.1
  environment. Five skips require CUDA; seven require native AIHWKit.
- Native program/verify suite: **22 passed, 2 CUDA skips** under AIHWKit 1.1.0 and
  PyTorch 2.11.0. This exercises the seven native checks skipped above. The
  environment lacks pytest, so its pure-Python test runner was loaded from the
  primary environment after native torch/numpy/AIHWKit imports; no packages were
  installed or changed. The native CIFAR parity check also reports `pass`.
- A fresh-process import check blocks retired modules while importing every
  supported runtime. A static import check guards the same boundary.
- All live example configs resolve through the retained registry. Historical
  native run inspection remains covered with an unregistered retired ID.
- The 20 extracted helper bodies are AST-identical apart from the retired spec
  annotation. Before/after fixtures match exactly for eight dense initialization
  modes and RNG state, sampling bindings/RNG, digital teacher checkpoint payloads,
  native population tensors/fingerprints and NPZ reload, one-batch deterministic
  QAT, and stochastic-apparent HWA outputs, reports and optimizer/RNG state.
- All **755** imported CIFAR run recipes resolve and verify explicit local inputs.
- Six rebuilt report families match the previous donor-verified migration output:
  **29 CSV tables, 172,644 rows**, with exact parsed-value equality.
- `git diff --check` passes. CIFAR configs, reference receipts, migration records,
  raw evidence and protected historical ledgers were not changed.

The first combined-suite dependency test inherited an incompatible MKL threading
setting from the test process. Its import-only subprocess now explicitly selects
GNU threading; the full suite passes. No numerical implementation changed for this
harness fix.

## Evidence and limits

The [compact verification receipt](20261004-crossbar-worktree-cleanup.json) hashes
the local audit in `results/crossbar-cleanup-20261004/`. That ignored directory
contains fixtures, detailed logs, all reproduction recipes and rebuilt reports.
The [CIFAR campaign](../cifar-crossbar-hwa-recovery/README.md) remains the scientific
record for the imported results. This cleanup is engineering verification; full
training and remote jobs were not launched, and CUDA execution was not verified.

Historical scientific records, plots, tables, datasets, native bundles and source
archives remain intact. The completed September workflow pilot remains historical
v1 evidence. Existing deleted-intermediate-checkpoint and prepared-study-digest
limitations documented by the CIFAR migration remain unchanged. The original
source checkouts and LoRA worktree were not modified.
