---
id: "exp-003"
title: "Interrupted eight-device differential-pair pilot"
status: "imported"
hypotheses: ["H-003"]
---

# exp-003 — Interrupted eight-device differential-pair pilot

## Question and provenance

Interrupted eight-device differential-pair pilot. Starting from one frozen clean ideal-differential DRN state, off-chip BPTT with compact IBM OM endpoint exposure on one exact fixed differential-pair array will reduce pulse-resolved apparent-forward deployment KL relative to matched clean FP32 continuation deployed through the same array, pair mapper, inner-window margin, endpoint stream, and apparent-forward endpoint policy.

Imported on 2026-10-02 from the existing worktree records at source revision `d56f072c106f294ffa54b2b86dd1c88b5d2ede80`. Historical dates, commands and input identities retain their original meaning. [Evidence, interpretation and next decision](../results/exp-003-eight-device.md).

Dependencies and related controls: [Corrected fixed-array common-window HWA](../results/exp-002-fixed-array.md).

## Frozen plans

- [MNIST DRN canonical eight-device IBM OM differential-pair HWA pilot](../../../../../studies/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1.json) — `mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1`.

Exact arms, configuration paths, case counts and stopping criteria remain in these plans. They are historical specifications, not authorization to launch.

Historical document locations (now references/forwarding pages): [ibm_om_hwa_program_verify_pilot.md](../../../../../docs/ibm_om_hwa_program_verify_pilot.md).

<a id="source-ibm_om_hwa_program_verify_pilot-predeclared-canonical-eight-device-differential-pair-pilot"></a>

## Predeclared canonical eight-device differential-pair pilot

- Status: interrupted partial evidence; the clean arm completed, while the
  HWA arm was stopped by user request after seven of ten epochs and is not
  eligible for workflow finalization
- Study: [`mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1.json`](../../../../../studies/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1.json)
- Claim boundary: an internally matched eight-device clean-versus-HWA test,
  not a matched comparison with the completed four-device v2 study

The next pilot uses the canonical `differential` DRN representation: every one
of the four signed dual-rail edges has a local `conductance_plus` and
`conductance_minus` cell, for eight memristors per logical weight. It does not
copy the four-device physical conductances into a different encoding. Both
arms instead load one tensor-identical current-schema canonicalization of the
existing clean ideal-differential checkpoint:

```text
source base:
  results/mnist-relu-drn-kd-exploratory-20260816/
  ideal_differential_finetune/
  20260816T135214.689196Z-9ef10aa2-642ca991/checkpoints/weights.pt
SHA-256:
  5a9dece30a938f00d2022de3e0aa57755e17f754c2df249507dd709bf9394fbc

ReLU teacher:
  results/mnist-relu-drn-kd-exploratory-20260816/
  teacher_fixed_init/
  20260816T132557.806720Z-fbff3c26-6f6867f1/checkpoints/weights.pt
SHA-256:
  9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52

IBM OM device model SHA-256:
  3030e04d6205dc90d0894ac453d2c1c522dffdc004f69b9c6ab7eaf7ef4b8ba3
```

The source selected epoch 19 under the signed ReLU teacher. Its recorded
validation accuracy is 97.16 percent, teacher agreement 99.84 percent, and KL
0.000685286. Canonicalization is metadata-only: the launcher must prove exact
equality of all four named conductance tensors, validate current metadata, and
replay the source and canonical checkpoint identically before either arm may
start. It must not invoke teacher mapping, conductance remapping, retraining,
or an optimizer update.

The two arms then perform ten continuation epochs with the same ReLU teacher,
seed `42`, data split, solver, ideal optimizer, differential learning rates,
fixed base checkpoint, and pulse-resolved checkpoint selection. The clean arm
uses no minibatch modifier. The pair-HWA arm uses the compact OM endpoint
modifier on every minibatch. Both selection paths use the same exact physical
deployment.

The physical protocol is fixed before observation:

| Item | Declared value |
| --- | --- |
| Assignment / compact endpoint / pulse endpoint seeds | `84001 / 84002 / 84003` |
| Corruption | `counterfactual_repaired` |
| Target mapping | `differential_pair_common_window` |
| Pairing | canonical adjacent plus/minus cells at the same coordinate |
| Common-window margin | `0.25` per side, the central 50 percent |
| Start/controller/budget | lower-from-RESET, adaptive, 128 pulses |
| Endpoint use | apparent forward, persistent hidden update state |
| Target support policy | fail closed with `target_out_of_support=error` |

The fixed seed-`84001` integrity fixture has population fingerprint
`d0dfae135fa7b741c46f1178e52f645da4593f7313b4e6a0a71a9b3793d1b2f0`:
317,600 cells, 158,800 two-cell pairs, zero active corrupt cells, and eight
empty pair intersections. The authoritative mapper must show zero unsupported
cells in every nonempty pair and strictly positive nonempty inner spans. The
eight empty pairs account for exactly 16 out-of-bound cells. Compact HWA uses
317,584 fitted endpoints and the exact pulse controller for those 16 cells,
and only those cells, under policy
`pulse_resolved_noncorrupt_out_of_bound_empty_pair_only`.

Before native launch, the exact pulse canary must program all cells with at
least 99 percent success and at most one percent budget exhaustion. In its one
already-programmed context, the production evaluator processes the first 100
ordered validation batches—1,600 examples—and must retain at least 25 percent
student accuracy and teacher agreement. Strict JSON/checkpoint round trips,
durable compact and pulse deployments, and population/report/hash parity are
also launch gates. After both runs, the launcher audits the two result files,
population artifacts and receipts, mapping and programming reports,
checkpoints, and deployment bundles before marking the study ready for review.

Once the implementation and focused tests are reviewed, the formal workflow
is:

```bash
python -m ebl study prepare \
  --plan studies/mnist-ibm-om-differential-pair-hwa-program-verify-pilot-20260824-v1.json \
  --results-root results

python -m experiments.mnist_relu_drn.ibm_om_differential_pair_pilot_launcher \
  --aihwkit-python /home/filip/miniconda3/envs/aihwkit/bin/python
```

## Current handoff

This is a documentation migration and local evidence reconciliation. No execution or monitoring owner is assigned by this record. Refer to the result for available bundles, recorded coverage, unresolved evidence gaps and proposed next decisions. Historical launcher handles below or in source notes are not current monitoring promises.
