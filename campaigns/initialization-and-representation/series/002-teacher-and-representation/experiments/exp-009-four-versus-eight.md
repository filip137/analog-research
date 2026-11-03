---
id: "exp-009"
title: "Four versus eight measured devices"
status: "imported"
hypotheses: []
---

# exp-009 — Four versus eight measured devices

## Question and provenance

Four versus eight measured devices. This historical comparison is organized retrospectively around the question and protocol preserved below. No new decision threshold is introduced.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-009-four-versus-eight.md).

Dependencies and related controls: [Pairwise common window versus bounded FP32](../results/exp-008-common-window-fp32.md).

Historical document locations (now references/forwarding pages): [dual_rail_input_four_vs_eight_devices.md](../../../../../docs/dual_rail_input_four_vs_eight_devices.md).

<a id="source-dual_rail_input_four_vs_eight_devices-frozen-end-to-end-comparison"></a>

## Frozen end-to-end comparison

Both arms use the same frozen teacher, MNIST split, seed, minibatch order,
solver, model-local amplification, raw cohort-A traces, assignment seed,
teacher-KL objective, calibration subset, and ten-epoch budget. Each arm uses
its already calibrated gradient-equivalent learning rates.

- Four-device config:
  `examples/mnist_relu_drn/measured_raw_single_quad_common_window_finetune_10ep.json`
- Eight-device config:
  `examples/mnist_relu_drn/measured_raw_differential_finetune_10ep.json`
- Campaign:
  `campaigns/manifests/mnist_relu_drn_four_vs_eight_common_window_10ep.json`

The primary empirical retention criterion is that the four-device arm's
selected held-out test accuracy is no more than 1.0 percentage point below
the eight-device arm, with finite training and validation metrics. Validation
teacher KL, teacher agreement, selected epoch, mapping overlap, programming
error, and initial accuracy are recorded as diagnostics. This is an
exploratory, single-seed screen; passing it would show that the benefit is
retained in this protocol, not that the implementations are universally
equivalent.

After placing the frozen inputs described in
`campaigns/manifests/README.md`, run from the repository root:

```bash
python -m ebl campaign run \
  --manifest campaigns/manifests/mnist_relu_drn_four_vs_eight_common_window_10ep.json \
  --output-dir results \
  --fail-fast
```

<a id="source-dual_rail_input_four_vs_eight_devices-held-out-cohort-b-follow-up"></a>

## Held-out cohort-B follow-up

The matched four-device cohort-B campaign is now complete. Each selected
cohort-A checkpoint was re-encoded on the same held-out cohort using its
scheme's common-window rule, evaluated immediately, fine-tuned for ten
off-chip epochs, and evaluated again in a fresh process.

| Arm | Cohort-B stage | Test accuracy | Teacher agreement | Test KL |
| --- | --- | ---: | ---: | ---: |
| Four-device quad window | immediate deployment | 35.79% | 35.72% | 1.761729 |
| Four-device quad window | after 10 epochs | **97.31%** | 97.61% | 0.0557048 |
| Eight-device paired window | immediate deployment | 42.96% | 43.32% | 2.03767 |
| Eight-device paired window | after 10 epochs | **97.81%** | 98.12% | 0.03156 |

The four-device arm recovers `61.52` accuracy points and reduces test KL by
`96.84%`, ending `0.15` points below its own cohort-A source. It is `7.17` points
below eight devices immediately and `0.50` points below after adaptation.
The result closes the earlier experimental gap: neither scheme is portable
without adaptation, both remain expressive on cohort B, and the eight-device
common-window scheme is more robust at immediate deployment and closer to
the teacher logits after adaptation.

The cohort-B four-cell windows are not systematically worse than the
cohort-A windows. Instead, reassignment contracts layerwise effective signed-
drive RMS by factors of `6.16` and `7.32` and raises mean four-cell load by
about `9%`. Differential initialization cancels the nominal common baseline,
but independent trace projection still changes both the signed numerators and
passive denominators. Full protocol, diagnostics, and hashes are in the
[four-device cohort-B report](../../../../measured-cohort-adaptation/series/003-representation-transfer/experiments/exp-007-four-device-transfer.md#source-mnist_four_device_cohort_b_adaptation-four-device-reram-cohort-a-to-b-deployment-and-adaptation).

Raw campaign output is under
`results/mnist-dual-rail-four-vs-eight-common-window-10ep-20260819-v1/` in
the operational `tiki-taka-lora-integration` worktree. The selected four- and
eight-device weights have SHA-256 hashes
`e95eb276ed30d20c024495480c1f7d985ced853b761b8a3d5f2151aa9614fcb8` and
`898131127b636893b3cc50e895635243e1ee0e0efd4ce4d9d1d35129a50195f9`,
respectively.

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
