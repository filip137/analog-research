---
experiment: "exp-010"
evidence: "imported-summary"
summary: "Noisy BPTT reaches 96.46% CMO and 91.84% Wan test accuracy from matched teacher initialization."
verdicts: {}
---

# Result: exp-010 — Teacher-initialized CMO/Wan comparison

[Experiment and historical specification](../experiments/exp-010-wan-cmo-teacher.md)

## Evidence and coverage

Reconciled on 2026-10-02 from local JSON, Markdown and file presence. This is an imported-summary review: no training, checkpoint replay, remote collection or fresh artifact-byte validation was performed. Recorded historical validation modes and timestamps below are not new verification claims. Old running markers do not establish live processes.

| Result root / role | Native status records | Locally available terminal results |
|---|---|---|
| [mnist_wan_cmo_teacher_initialized_seed17_f60d4406](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406) | complete: 17 | 17 |

Status counts count attempts, including failed/replaced attempts and separate evaluations; they are not numbers of independent replicates.

Existing analyses and receipts: [analysis/summary.json](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/analysis/summary.json).

## Observations

Noisy BPTT reaches 96.46% CMO and 91.84% Wan test accuracy from matched teacher initialization.

### Native terminal measurements

Values below are read from the linked native result JSON. Selection and terminal metrics are distinct; the source protocol defines the evaluation split. Apparent-state recovery is primary.

| Native result | Recorded status | Measurement |
|---|---|---|
| [mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/full_bptt/attempt-001/runs/20260820T125544.212986Z-f59b7cc2-dec5b538](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/full_bptt/attempt-001/runs/20260820T125544.212986Z-f59b7cc2-dec5b538/result.json) | complete | selected student accuracy: 96.72%; initial student accuracy: 91.50%; terminal student accuracy: 96.72% |
| [mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/full_bptt_test/attempt-001/runs/20260820T130331.259859Z-a986985c-79a1b5c3](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/full_bptt_test/attempt-001/runs/20260820T130331.259859Z-a986985c-79a1b5c3/result.json) | complete | evaluation accuracy: 96.46% |
| [mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/hwa_deploy/attempt-001/runs/20260820T125539.937962Z-18307f2b-bb401644](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/hwa_deploy/attempt-001/runs/20260820T125539.937962Z-18307f2b-bb401644/result.json) | complete | selected student accuracy: 91.50%; initial student accuracy: 91.50%; terminal student accuracy: 91.50% |
| [mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/hwa_deploy_test/attempt-001/runs/20260820T130328.016017Z-a986985c-443c6879](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/hwa_deploy_test/attempt-001/runs/20260820T130328.016017Z-a986985c-443c6879/result.json) | complete | evaluation accuracy: 92.18% |
| [mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/ideal_deploy/attempt-001/runs/20260820T125536.143566Z-18307f2b-cd6fc363](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/ideal_deploy/attempt-001/runs/20260820T125536.143566Z-18307f2b-cd6fc363/result.json) | complete | selected student accuracy: 90.52%; initial student accuracy: 90.52%; terminal student accuracy: 90.52% |
| [mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/ideal_deploy_test/attempt-001/runs/20260820T130324.118951Z-a986985c-2df6b386](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_cmo_f60d4406/mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1/stages/cmo/akib/ideal_deploy_test/attempt-001/runs/20260820T130324.118951Z-a986985c-2df6b386/result.json) | complete | evaluation accuracy: 91.15% |
| [mnist_teacher_initialized_common_f60d4406/mnist-teacher-initialized-common-f60d4406-20260820-v1/stages/common/akib/hwa_test/attempt-001/runs/20260820T125050.533550Z-f1a51a5f-bdc08479](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_initialized_common_f60d4406/mnist-teacher-initialized-common-f60d4406-20260820-v1/stages/common/akib/hwa_test/attempt-001/runs/20260820T125050.533550Z-f1a51a5f-bdc08479/result.json) | complete | evaluation accuracy: 97.34% |
| [mnist_teacher_initialized_common_f60d4406/mnist-teacher-initialized-common-f60d4406-20260820-v1/stages/common/akib/hwa_train/attempt-001/runs/20260820T125017.605883Z-dc33e08e-68250a15](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_initialized_common_f60d4406/mnist-teacher-initialized-common-f60d4406-20260820-v1/stages/common/akib/hwa_train/attempt-001/runs/20260820T125017.605883Z-dc33e08e-68250a15/result.json) | complete | selected student accuracy: 97.18%; initial student accuracy: 96.74%; terminal student accuracy: 97.16% |
| [mnist_teacher_initialized_common_f60d4406/mnist-teacher-initialized-common-f60d4406-20260820-v1/stages/common/akib/ideal_map/attempt-001/runs/20260820T125011.528821Z-2fae6d08-5f284a70](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_initialized_common_f60d4406/mnist-teacher-initialized-common-f60d4406-20260820-v1/stages/common/akib/ideal_map/attempt-001/runs/20260820T125011.528821Z-2fae6d08-5f284a70/result.json) | complete | selected student accuracy: 96.74%; initial student accuracy: 96.74%; terminal student accuracy: 96.74% |
| [mnist_teacher_initialized_common_f60d4406/mnist-teacher-initialized-common-f60d4406-20260820-v1/stages/common/akib/ideal_map_test/attempt-001/runs/20260820T125014.714143Z-f1a51a5f-2507c903](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_initialized_common_f60d4406/mnist-teacher-initialized-common-f60d4406-20260820-v1/stages/common/akib/ideal_map_test/attempt-001/runs/20260820T125014.714143Z-f1a51a5f-2507c903/result.json) | complete | evaluation accuracy: 96.91% |
| [mnist_teacher_initialized_common_f60d4406/mnist-teacher-initialized-common-f60d4406-20260820-v1/stages/common/akib/teacher_test/attempt-001/runs/20260820T125009.264548Z-d3640c04-27519cda](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_initialized_common_f60d4406/mnist-teacher-initialized-common-f60d4406-20260820-v1/stages/common/akib/teacher_test/attempt-001/runs/20260820T125009.264548Z-d3640c04-27519cda/result.json) | complete | evaluation accuracy: 97.36% |
| [mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/full_bptt/attempt-001/runs/20260820T125550.038696Z-902f4e48-8366e46e](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/full_bptt/attempt-001/runs/20260820T125550.038696Z-902f4e48-8366e46e/result.json) | complete | selected student accuracy: 91.36%; initial student accuracy: 78.60%; terminal student accuracy: 90.52% |
| [mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/full_bptt_test/attempt-001/runs/20260820T130336.741976Z-a986985c-fd5170e4](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/full_bptt_test/attempt-001/runs/20260820T130336.741976Z-a986985c-fd5170e4/result.json) | complete | evaluation accuracy: 91.84% |
| [mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/hwa_deploy/attempt-001/runs/20260820T125545.164265Z-f3908ece-e0be9c4d](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/hwa_deploy/attempt-001/runs/20260820T125545.164265Z-f3908ece-e0be9c4d/result.json) | complete | selected student accuracy: 78.60%; initial student accuracy: 78.60%; terminal student accuracy: 78.60% |
| [mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/hwa_deploy_test/attempt-001/runs/20260820T130333.836016Z-a986985c-1fd8c66f](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/hwa_deploy_test/attempt-001/runs/20260820T130333.836016Z-a986985c-1fd8c66f/result.json) | complete | evaluation accuracy: 80.31% |
| [mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/ideal_deploy/attempt-001/runs/20260820T125540.943082Z-f3908ece-531e79c8](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/ideal_deploy/attempt-001/runs/20260820T125540.943082Z-f3908ece-531e79c8/result.json) | complete | selected student accuracy: 77.34%; initial student accuracy: 77.34%; terminal student accuracy: 77.34% |
| [mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/ideal_deploy_test/attempt-001/runs/20260820T130330.496612Z-a986985c-12c447d1](../../../../../results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/mnist_teacher_wan_f60d4406/mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1/stages/wan/akib/ideal_deploy_test/attempt-001/runs/20260820T130330.496612Z-a986985c-12c447d1/result.json) | complete | evaluation accuracy: 78.62% |

<a id="source-mnist_wan_cmo_teacher_initialized-outcome"></a>

## Outcome

This study corrects the initialization ambiguity in the earlier CMO/Wan
comparison. Every DRN arm now starts from the same frozen, bias-free
`784 -> 50 -> 10` ReLU teacher. No independently trained DRN checkpoint is
used as the source.

On the untouched 10,000-example MNIST test split, the frozen teacher reached
`97.36%`, and its immediate ideal DRN mapping reached `96.91%`. Two epochs of
generic additive-noise HWA raised the clean DRN to `97.34%`. HWA modestly
improved the first physical write on both devices, while post-deployment noisy
BPTT supplied most of the recovery:

| Test state | CMO accuracy | CMO teacher KL | CMO agreement | Wan accuracy | Wan teacher KL | Wan agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen FP32 ReLU teacher | `97.36%` | `0` | `100.00%` | `97.36%` | `0` | `100.00%` |
| Immediate ideal FP32 DRN mapping | `96.91%` | `0.009345` | `99.00%` | `96.91%` | `0.009345` | `99.00%` |
| Clean HWA DRN | `97.34%` | `0.002681` | `99.52%` | `97.34%` | `0.002681` | `99.52%` |
| Ideal mapping after first endpoint write | `91.15%` | `0.177332` | `92.54%` | `78.62%` | `0.550453` | `79.36%` |
| HWA after first endpoint write | `92.18%` | `0.156257` | `93.51%` | `80.31%` | `0.509778` | `81.06%` |
| HWA + endpoint + selected noisy BPTT | `96.46%` | `0.032341` | `98.11%` | `91.84%` | `0.186471` | `92.62%` |

All KL values are
`KL(teacher || DRN)` after fitting one positive output gain on the fixed
validation cohort. Gain calibration cannot change accuracy. The HWA and BPTT
checkpoints were selected by validation KL, and the test set was evaluated
only after selection.

<a id="source-mnist_wan_cmo_teacher_initialized-interpretation"></a>

## Interpretation

For this device realization, HWA is useful but not the dominant intervention.
The much larger improvement comes from post-deployment BPTT with noisy writes.
This experiment does not yet prove that HWA is necessary for BPTT recovery,
because an ideal-map-to-device-to-BPTT arm without HWA was not run.

The device gap is not ordered by the size of the finite floor. At the HWA
first write, the programming-error RMSE relative to the ideal affine physical
target is `1.152 uS` in effective DRN units for CMO and `5.130 uS` for Wan,
so the tested Wan realization is `4.45x` noisier by this measure. Conversely,
the deterministic affine mapping error relative to the zero-floor digital
target is larger for CMO (`10.681 uS` versus `2.617 uS`) because CMO has the
larger floor. The CMO floor is a highly structured common-mode offset that the
balanced dual-rail input and paired output largely reject; Wan's stochastic,
conductance-dependent error changes relative edge strengths. Total RMSE to
the zero-floor digital target therefore does not predict accuracy by itself.

The large fitted CMO output gain (`19.953`, versus `1.995` for the HWA-Wan
write) also shows that CMO strongly attenuates raw output scores. The retained
decision structure is good enough for accuracy after calibration, but this is
not evidence that analog margin, ADC resolution, or circuit noise is benign.

<a id="source-mnist_wan_cmo_teacher_initialized-evidence-boundary-and-provenance"></a>

## Evidence boundary and provenance

This is an exploratory single-teacher, single-device-seed comparison. It does
not model pulse counts, asymmetric potentiation/depression, endurance, energy,
converter quantization, IR drop, dynamic inference noise, or run-to-run
uncertainty. The next controls needed to assess HWA necessity are a matched
no-HWA BPTT arm and multiple endpoint seeds; device-matched HWA should then be
compared with the generic 3% modifier.

- Frozen source commit:
  `f60d4406c5a5cde7b75726626e4adeae5f4e4c03` (clean)
- Teacher SHA-256:
  `42b0526c4a433057b0ad0f09a7b36ede64afd7d47a3a8985724cac0e46358d54`
- CMO selected-BPTT SHA-256:
  `d1224cfb5677df0d943e873b935e090a0b93e3f58f454c4c521e7b56629816ef`
- Wan selected-BPTT SHA-256:
  `4e27f679274aa344331c52b2ca734ae92f773d174b4ffa626db27ea1af1c6ab4`
- Campaign IDs:
  `mnist-teacher-initialized-common-f60d4406-20260820-v1`,
  `mnist-teacher-initialized-cmo-seed17-f60d4406-20260820-v1`, and
  `mnist-teacher-initialized-wan-seed17-f60d4406-20260820-v1`
- Local ignored raw artifacts:
  `results/mnist_wan_cmo_teacher_initialized_seed17_f60d4406/`

Historical source: [mnist-wan-cmo-teacher-initialized-seed17-10ep](../../../../../docs/experimental_manifest.md#mnist-wan-cmo-teacher-initialized-seed17-10ep).

- **Outcome:** HWA modestly positive; noisy BPTT strongly positive; CMO ahead
- **Interpretation:** HWA modestly positive; noisy BPTT strongly positive; CMO ahead

## Interpretation and limitations

HWA modestly improves first-write accuracy; later BPTT supplies more recovery. Without a no-HWA recovery arm, this cannot establish that HWA is necessary for recovery.

Review scope: local metadata and existing scientific records. No fresh numerical replay or blanket artifact validation was performed. No formal hypothesis is declared for this descriptive import; the empty verdict map is not a numerical hypothesis verdict. Unavailable evidence and incomplete attempts remain visible above. Scientific conclusions are limited to the original devices, data splits, initialization and update semantics.

## Next decision

A matched no-HWA recovery arm and independent endpoint/base seeds are needed to identify the necessity of HWA.
