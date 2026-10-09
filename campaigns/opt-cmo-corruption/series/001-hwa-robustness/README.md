# HWA source robustness to stuck devices

These experiments deploy one checkpoint per HWA source on independent CMO
arrays across stuck-device rates, without retraining.

- **Shared:** OPT-125M with three analog decoder MLPs, the pinned WikiText-2
  corpus and the CMO 2 % P&V law with active/reference pairs.
- **Not poolable:** results at different stuck-device rates or placements, and
  results from different HWA noise conventions.

See the [campaign ledger](../../ledger.md).
