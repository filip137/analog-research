# Crossbar HWA and array-specific recovery on CIFAR

When does same-array weight learning improve teacher fidelity and classification
beyond hardware-aware training and equally trained digital calibration, and what
programming cost does that improvement require?

This campaign owns the dense CIFAR-10 and ResNet-32 CIFAR-10/100 evidence migrated
on 2026-10-04. Its main ResNet experiments freeze the digital prefix and map the
last four or eight convolutions plus classifier to OM or PCM arrays. OM pulse
learning, PCM endpoint reprogramming, and the earlier legacy PCM controls are
separate regimes; the dense pilot is not a ResNet baseline.

Current direction: preserve the matched HWA/calibration/rewrite/recovery
comparisons, then resolve HWA convergence and source quality before making a
necessity claim about on-chip learning. Historical fits use one training seed;
independent array draws do not measure training-seed uncertainty. The imported
notes retain their scientific-review status. This migration launched no new
training or remote jobs.

- [Ledger](ledger.md): nine historical experiments and the migration audit.
- [Dense pilot](series/001-dense/README.md), [ResNet mapping](series/002-mapping/README.md),
  [PCM adaptation](series/003-pcm/README.md), [full epochs and fault sweep](series/004-full/README.md),
  [OM controllers](series/005-om/README.md).
- [Lifecycle definitions](lifecycles/README.md): standardized loops for new work,
  starting with labelled smoke lifecycles that validate the workflow path.
- [Migration audit](series/006-migration/results/exp-010.md) and
  [reproduction guide](../../docs/cifar_reproduction.md).
- [Retained comparison tables](../../artifacts/cifar_recovery_five_stage_tables_20260920/calibrated_headline.md)
  and [full fault-sweep report](../../results/cifar-sweep-analysis/report.md).

Small campaign records and executable code are versioned. Large artifacts live
locally under `results/` and `artifacts/`; manifests and source archives preserve
original host paths and hashes. The replay helper resolves inputs by content
hash inside this worktree. Epoch 1–4 binaries removed on September 25 were
already absent at migration; numeric curves and retained later states remain.
