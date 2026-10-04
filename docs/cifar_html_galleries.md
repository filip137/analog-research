# Saved CIFAR HTML galleries

Catalogued on 2026-09-23 at the user's request for quick access to the recently
opened plots. This catalog links the existing exports; it does not change the
results or run new experiments.

## Quick access

Open and bookmark [CIFAR results — saved galleries](../cifar_results.html).
The repository README also links to this page.

```text
/home/filip/server_code/.codex/worktrees/cifar-resnet-suffix-recovery/cifar_results.html
```

The first two entries are the most recent selectable galleries. Both default
to CIFAR-100 / OM and allow switching between accuracy and teacher KL.

## Gallery catalog

All paths below are relative to the saved export directory:

```text
/home/filip/server_code/.codex/worktrees/cifar-resnet-suffix-recovery/artifacts/cifar_kl_fault_curves_20260922/
```

| Gallery | Saved HTML | Contents |
| --- | --- | --- |
| All training methods | [all_methods.html](../artifacts/cifar_kl_fault_curves_20260922/all_methods.html) | No HWA, Normal HWA, Noisy HWA, and Corruption-aware HWA; direct deployment, calibration, and recovery; selectable dataset/device/depth/metric. |
| Best method + recovery | [best_method/explorer.html](../artifacts/cifar_kl_fault_curves_20260922/best_method/explorer.html) | Best calibrated method by mean KL, matched recovery, and alternative recovered winner; selectable dataset/device/metric. |
| On-chip recovery gains | [onchip_gain/index.html](../artifacts/cifar_kl_fault_curves_20260922/onchip_gain/index.html) | Paired KL and accuracy gains over calibration for No HWA, Noisy HWA, and Corruption-aware HWA. |
| Separate device figures | [best_method/by_device/index.html](../artifacts/cifar_kl_fault_curves_20260922/best_method/by_device/index.html) | OM and PCM figures, each including both datasets and analog depths. |
| Per-dataset overview | [best_method/index.html](../artifacts/cifar_kl_fault_curves_20260922/best_method/index.html) | Earlier best-method figures with both devices and depths per dataset. |
| Original export index | [index.html](../artifacts/cifar_kl_fault_curves_20260922/index.html) | Original all-method curves, endpoint tables, and related exports. |
| PCM recovery progress | [pcm_recovery_tail/index.html](../artifacts/cifar_kl_fault_curves_20260922/pcm_recovery_tail/index.html) | Recovery trajectories near the five-epoch stopping budget. |
| OM recovery progress | [om_recovery_tail/index.html](../artifacts/cifar_kl_fault_curves_20260922/om_recovery_tail/index.html) | Late recovery progress and pulse-cap audit; later OM follow-ups remain separate. |

## Storage and portability

The HTML pages remain beside their original plot assets and CSVs. Open them
directly in a browser; no web server or internet connection is required.
The galleries offer their available JPG, PNG, SVG, PDF, and CSV downloads.
Keep the complete export folder together when copying it to another machine.

The [complete ZIP bundle](../artifacts/cifar_kl_fault_curves_20260922.zip)
contains all eight galleries and their assets. Extract it before opening the
HTML. This repository-level quick-access page and documentation are outside
the ZIP; inside the extracted bundle, start with `index.html` or
`all_methods.html`.

```text
/home/filip/server_code/.codex/worktrees/cifar-resnet-suffix-recovery/artifacts/cifar_kl_fault_curves_20260922.zip
```

The [export README](../artifacts/cifar_kl_fault_curves_20260922/README.md)
records the scientific definitions and reproduction instructions. Accuracy
in the best-method views retains the KL-selected methods. Calibration and
joint recovery are independent five-epoch alternatives from the same initial
deployment. The historical sweep is not replaced by later OM follow-ups.

When adding another gallery, update this catalog and `../cifar_results.html`.
Preserve the dated export directory so saved links continue to work.
