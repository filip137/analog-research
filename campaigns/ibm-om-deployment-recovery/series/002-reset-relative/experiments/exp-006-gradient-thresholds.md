---
id: "exp-006"
title: "Direct-pulse gradient-threshold follow-up"
status: "imported"
hypotheses: ["H-006"]
---

# exp-006 — Direct-pulse gradient-threshold follow-up

## Question and provenance

Direct-pulse gradient-threshold follow-up. Suppressing small individual-conductance updates with fixed layer-specific p90, p95, or p99 absolute raw-gradient gates will improve terminal apparent-forward validation accuracy over the matched unthresholded direct-pulse control at at least one of the 0.1, 1, or 4 slow-pulse caps, while requesting fewer slow-array writes and producing fewer logical sign flips.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-006-gradient-thresholds.md).

Dependencies and related controls: [Matched refresh, direct-pulse and Tiki-Taka recovery](../results/exp-005-pulse-recovery.md).

## Frozen plans

- [One-seed RESET-relative deployed-array direct-gradient threshold sweep](../../../../../studies/mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1.json) — `mnist-ibm-om-reset-relative-direct-gradient-threshold-20260825-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.

Historical document locations (now references/forwarding pages): [ibm_om_reset_relative_direct_gradient_thresholds.md](../../../../../docs/ibm_om_reset_relative_direct_gradient_thresholds.md).

<a id="source-ibm_om_reset_relative_direct_gradient_thresholds-reset-relative-direct-gradient-thresholding"></a>

## Historical source: RESET-relative direct-gradient thresholding

This development protocol tests whether the direct-pulse recovery rule spends
writes on gradients that are too small to justify a full physical SET or RESET
pulse. It starts every arm from the exact saved deployment used by the
unthresholded recovery pilot. It does not remap, reprogram, resample, or inspect
hidden device bounds.

<a id="source-ibm_om_reset_relative_direct_gradient_thresholds-fixed-gate"></a>

## Fixed gate

Before the first update, the runtime replays 64 deterministic training
minibatches at the source apparent state. The replay restores the training
dataloader and global RNG state afterward. For each stable conductance
parameter separately, it gathers all strictly nonzero absolute raw gradients
and computes one linear-interpolated p90, p95, or p99 threshold.

During training, cell `i` is eligible only when

```text
abs(gradient_i) > threshold_for_its_parameter
```

Equality holds the cell. The sign of `-learning_rate * gradient_i` still
chooses SET versus RESET. The gate is symmetric because this deployed OM model
permits both directions; the earlier one-pulse-down ReRAM study used only a
positive-gradient gate because its update rule could only decrease
conductance.

The thresholds are fixed after the source-state replay. A label such as p99
therefore describes the calibration distribution, not a promise that exactly
one percent of later minibatch gradients remain eligible.

<a id="source-ibm_om_reset_relative_direct_gradient_thresholds-no-pulse-redistribution"></a>

## No pulse redistribution

Each budget reuses the exact probability scale fitted by its unthresholded
control:

| Slow-pulse cap per cell | Frozen scale | Source report SHA-256 |
| ---: | ---: | --- |
| 0.1 | `0.005644706616388747` | `f3ded58c814858f3bb1a6c8da282521e04ee8f3fdaebf9cba8ffacb1cc02a1c2` |
| 1 | `0.05644706616388745` | `e9944e8ce1494e95369bc632be02a8ca60588be20d22ba59471d836290b2dbc1` |
| 4 | `0.2257882646555498` | `b36e5d09f648ad47b12c12c629587cd826587fc5460d3dac4f8aea55c3cc23b1` |

The runtime first proves that the frozen scale applied to the ungated replay
still reproduces the earlier expected full-run cap. It then zeros probabilities
below the threshold without increasing the remaining probabilities. Thus a
gate may save writes; it does not force the full budget onto fewer cells.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
