# HWA source robustness to stuck devices

These experiments deploy one checkpoint per HWA source on independent CMO
arrays across stuck-device rates, without retraining.

- **Shared:** OPT-125M and the CMO 2% P&V law with active/reference pairs.
  Early experiments used three or six analog MLPs and WikiText training;
  the current comparison uses six analog MLPs and the pinned synthetic
  50M-token corpus. Exact regimes belong to the individual experiment notes.
- **Not poolable:** results at different stuck-device rates or placements, and
  results from different analog subsets, training corpora or HWA noise conventions.

See the [campaign ledger](../../ledger.md).
