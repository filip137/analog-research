---
experiment: "exp-009"
evidence: "imported-summary"
summary: "Noisy BPTT changes CMO by −0.63 points and Wan by +18.45 points in one realization."
verdicts: {}
---

# Result: exp-009 — Independently initialized CMO/Wan comparison

[Experiment and historical specification](../experiments/exp-009-wan-cmo-independent.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist_wan_cmo_head_to_head_seed17_10ep](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep) | complete: 11 | 11 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

## Observations

Noisy BPTT changes CMO by −0.63 points and Wan by +18.45 points in one realization.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [mnist-wan-cmo-headtohead-seed17-10ep-parallel-cmo/20260820T104445.512468Z-d09ec1f9-e7e0aaee](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-10ep-parallel-cmo/20260820T104445.512468Z-d09ec1f9-e7e0aaee/result.json) | complete | selected accuracy: 95.56%; terminal accuracy: 95.56% |
| [mnist-wan-cmo-headtohead-seed17-10ep-parallel-cmo-validation/20260820T105504.228847Z-563df19c-7a38436b](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-10ep-parallel-cmo-validation/20260820T105504.228847Z-563df19c-7a38436b/result.json) | complete | evaluation accuracy: 95.56% |
| [mnist-wan-cmo-headtohead-seed17-10ep-parallel-wan/20260820T104448.944422Z-88d87f7c-fb4e5f75](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-10ep-parallel-wan/20260820T104448.944422Z-88d87f7c-fb4e5f75/result.json) | complete | selected accuracy: 94.23%; terminal accuracy: 93.36% |
| [mnist-wan-cmo-headtohead-seed17-10ep-parallel-wan-validation/20260820T105504.228829Z-563df19c-fc6cfdd3](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-10ep-parallel-wan-validation/20260820T105504.228829Z-563df19c-fc6cfdd3/result.json) | complete | evaluation accuracy: 94.23% |
| [mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/cmo/akib/cmo_full_bptt/attempt-001/runs/20260820T103522.228815Z-7c297c9f-c5f3b01a](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/cmo/akib/cmo_full_bptt/attempt-001/runs/20260820T103522.228815Z-7c297c9f-c5f3b01a/result.json) | complete | selected accuracy: 95.86%; terminal accuracy: 95.86% |
| [mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/cmo/akib/fp32_cmo_deploy/attempt-001/runs/20260820T103458.293833Z-c41cc332-30621ae4](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/cmo/akib/fp32_cmo_deploy/attempt-001/runs/20260820T103458.293833Z-c41cc332-30621ae4/result.json) | complete | selected accuracy: 95.99%; terminal accuracy: 95.99% |
| [mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/cmo/akib/hwa_cmo_deploy/attempt-001/runs/20260820T103515.774083Z-c41cc332-df942b4f](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/cmo/akib/hwa_cmo_deploy/attempt-001/runs/20260820T103515.774083Z-c41cc332-df942b4f/result.json) | complete | selected accuracy: 95.85%; terminal accuracy: 95.85% |
| [mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/common/akib/fp32_test/attempt-001/runs/20260820T103405.470869Z-563df19c-efe55ebb](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/common/akib/fp32_test/attempt-001/runs/20260820T103405.470869Z-563df19c-efe55ebb/result.json) | complete | evaluation accuracy: 96.15% |
| [mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/common/akib/hwa_test/attempt-001/runs/20260820T103511.162796Z-563df19c-28678534](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/common/akib/hwa_test/attempt-001/runs/20260820T103511.162796Z-563df19c-28678534/result.json) | complete | evaluation accuracy: 95.67% |
| [mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/common/akib/hwa_train/attempt-001/runs/20260820T103410.078082Z-ae848d07-1b0f6ece](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/common/akib/hwa_train/attempt-001/runs/20260820T103410.078082Z-ae848d07-1b0f6ece/result.json) | complete | selected accuracy: 95.67%; terminal accuracy: 95.67% |
| [mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/wan/akib/fp32_wan_deploy/attempt-001/runs/20260820T103504.756178Z-66fff1fd-fda2defb](../../../../../results/mnist_wan_cmo_head_to_head_seed17_10ep/mnist-wan-cmo-headtohead-seed17-20260820-v2/mnist-wan-cmo-head-to-head-seed17-20260820-v1/stages/wan/akib/fp32_wan_deploy/attempt-001/runs/20260820T103504.756178Z-66fff1fd-fda2defb/result.json) | complete | selected accuracy: 90.16%; terminal accuracy: 90.16% |

<a id="source-mnist_wan_cmo_head_to_head-result"></a>

## Result

This is a matched, single-device-seed comparison on all 10,000 MNIST test
examples. Both device arms use the same ordinary FP32 checkpoint, the same
two-epoch HWA checkpoint, the same DRN circuit, one-day device age, retained
physical floors, and ten epochs of full BPTT with a fresh noisy endpoint write
after every minibatch.

| Stage | CMO/HfOx | Wan-2022 | CMO minus Wan |
| --- | ---: | ---: | ---: |
| Clean ordinary FP32 DRN | 96.15% | 96.15% | 0.00 pp |
| FP32 after first device write | 96.02% | 87.80% | +8.22 pp |
| Clean HWA checkpoint, before device write | 95.67% | 95.67% | 0.00 pp |
| HWA after first device write | 96.19% | 75.78% | +20.41 pp |
| HWA + device + 10-epoch noisy full BPTT | **95.56%** | **94.23%** | **+1.33 pp** |

The BPTT row is the validation-cost-selected saved checkpoint, verified in a
fresh process. CMO selected epoch 10 and reproduced 95.56%; Wan selected
epoch 9 and reproduced 94.23%. Wan's last, non-selected epoch was 93.36%.

Relative to the first HWA-device write, ten-epoch BPTT changed CMO by
`-0.63 pp` and Wan by `+18.45 pp`. Thus, in this realization, full BPTT was
not useful for CMO but was essential for Wan. Wan nevertheless remained
`1.92 pp` below the clean FP32 baseline after recovery.

<a id="source-mnist_wan_cmo_head_to_head-interpretation"></a>

## Interpretation

The finite floor is not what separates the devices here. CMO has the larger
floor ratio (`9/88.2 = 10.20%`, versus `1/40 = 2.50%` for Wan), yet it preserves
the FP32 accuracy on the first write. The affine floor adds a largely
common-mode conductance and preserves the ordering of every target before
noise; the balanced DRN representation rejects much of that common-mode term.

The stochastic endpoint error is much larger for Wan in this realization.
On the direct FP32 write, the RMS error relative to the ideal affine target is
`0.03145` DRN units for Wan and `0.00963` for CMO, a factor of 3.26. A second
independent zero-rate probe write changed CMO only from 96.02% to 95.99%, but
changed Wan from 87.80% to 90.16%. This is direct evidence that a single Wan
write has materially higher realization variance under the one-day model.

The HWA result must be interpreted cautiously. The shared HWA modifier is the
IBM-style generic additive-normal procedure, not a fit to the Wan polynomial.
It slightly lowered clean accuracy and, for Wan seed 17, moved the target to a
particularly poor first physical realization (75.78%). One realization is
insufficient to conclude that HWA generally harms Wan. The robust conclusion
from this run is narrower: the tested HWA procedure did not protect this Wan
realization, whereas noisy full BPTT recovered most of its loss.

<a id="source-mnist_wan_cmo_head_to_head-limitation"></a>

## Limitation

These are exact matched results for one device seed, not device-population
means. This is especially important for Wan, whose first two direct writes
differed by 2.36 percentage points. A population claim requires repeating the
entire physical-write and noisy-BPTT chain over matched seeds.

Historical source: [mnist-wan-cmo-head-to-head-seed17-10ep](../../../../../docs/experimental_manifest.md#mnist-wan-cmo-head-to-head-seed17-10ep).

- **Outcome:** CMO deployment-positive; Wan recovery-positive
- **Interpretation:** BPTT changed deployed HWA accuracy by `-0.63 pp` on CMO and `+18.45 pp` on Wan. The direct-write programming RMSE relative to the ideal affine target was `0.00963` CMO versus `0.03145` Wan DRN units; stochastic endpoint distortion, not finite-floor ratio, dominates the one-seed gap. This is retained as an endpoint-noise control but superseded as the primary initialization protocol because its DRN was trained independently rather than mapped from the frozen ReLU teacher.

## Interpretation and limitations

The endpoint error, not the floor ratio alone, explains the observed ordering. Initialization is superseded by the teacher-matched successor; generic HWA is not a fitted Wan noise model.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

Use the teacher-initialized successor for the primary initialization protocol, retaining this study as an endpoint-noise control.
