# Documentation and research records

Use [scientific campaigns](../campaigns/README.md) for questions, experiment specifications, results and interpretation. This directory retains shared contracts, implementation references, stable figure assets and historical provenance.

## Workflow and scientific contracts

- [Question-led experimental workflow](../workflow/policy.md).
- [Experiment management handoff for a small repository](experiment-management-handoff.md).
- [Architecture and ownership](architecture.md).
- [Experiment runtime](experiment_runtime.md).
- [Synapse data and device-model sources](synapse_data.md).
- [Initialization protocols](initialization_protocols.md).
- [ReRAM device catalog](reram_device_catalog.md).

## Circuit and training references

- [Differential memristor scheme](differential_scheme.md).
- [Differential pairs using signed amplifier ports](differential_pair_amplifier_implementation.md).
- [Clamped dual-rail input: four versus eight devices](dual_rail_input_four_vs_eight_devices.md).
- [Finite conductance floors in DRNs](conductance_floor_mitigation.md).
- [Hardware-aware additive weight noise](hardware_aware_training.md).
- [Digital low-rank recovery of a programmed DRN](digital_low_rank_recovery.md).
- [Passive layerwise low-rank recovery of a two-edge DRN](passive_layerwise_low_rank_recovery.md).
- [Passive low-rank adapter](passive_low_rank_adapter.md).
- [Tiki-Taka gradient accumulation](tiki_taka_gradient_accumulation.md).
- [IBM OM device layers](ibm_om_layers.md).
- [IBM ReRAM pulse-count program-and-verify noise model](ibm_reram_program_verify_noise_model.md).

## Protected synthesis and frozen provenance

The human-owned synthesis is unchanged. The simulation and experimental manifests are frozen snapshots; their activity and update instructions are historical. Campaign notes own current interpretation.

- [Current State](current_state.md).
- [Current LoRA/HWA Simulations](current_simulations.md).
- [Experimental Manifest](experimental_manifest.md).
- [Experiment workflow](archive/experiment_workflow-pre-question-led.md).

## Legacy administrative placeholder

- [Optuna Analysis State](optuna_state.md).

## Migrated study narratives

The legacy filenames remain forwarding pages, including section anchors. Shared references above keep their technical sections while linking moved study sections.

| Former document | Owning experiment records |
|---|---|
| [mnist_relu_drn_reset.md](mnist_relu_drn_reset.md) | [Learning from measured RESET](../campaigns/initialization-and-representation/series/001-reset-and-objective/experiments/exp-001-reset.md) |
| [mnist_relu_drn_reset_bias.md](mnist_relu_drn_reset_bias.md) | [Bias, loss and historical amplifier indexing](../campaigns/initialization-and-representation/series/001-reset-and-objective/experiments/exp-002-bias-and-indexing.md) |
| [mnist_relu_drn_reset_factorial.md](mnist_relu_drn_reset_factorial.md) | [RESET loss × bias × indexing factorial](../campaigns/initialization-and-representation/series/001-reset-and-objective/experiments/exp-003-reset-factorial.md) |
| [mnist_relu_drn_reset_differential_10ep.md](mnist_relu_drn_reset_differential_10ep.md) | [Ten-epoch differential RESET](../campaigns/initialization-and-representation/series/001-reset-and-objective/experiments/exp-004-differential-reset.md) |
| [mnist_relu_drn_kd.md](mnist_relu_drn_kd.md) | [Teacher-initialized measured DRN distillation](../campaigns/initialization-and-representation/series/002-teacher-and-representation/experiments/exp-005-teacher-distillation.md) |
| [mnist_relu_drn_initialized_differential_10ep.md](mnist_relu_drn_initialized_differential_10ep.md) | [Ten-epoch teacher-initialized differential control](../campaigns/initialization-and-representation/series/002-teacher-and-representation/experiments/exp-006-differential-ten-epoch.md) |
| [mnist_relu_drn_initialized_single_10ep.md](mnist_relu_drn_initialized_single_10ep.md) | [Ten-epoch teacher-initialized single-conductance control](../campaigns/initialization-and-representation/series/002-teacher-and-representation/experiments/exp-007-single-ten-epoch.md) |
| [mnist_single_rail_common_window_vs_fp32.md](mnist_single_rail_common_window_vs_fp32.md) | [Pairwise common window versus bounded FP32](../campaigns/initialization-and-representation/series/002-teacher-and-representation/experiments/exp-008-common-window-fp32.md) |
| [dual_rail_input_four_vs_eight_devices.md](dual_rail_input_four_vs_eight_devices.md) | [Four versus eight measured devices](../campaigns/initialization-and-representation/series/002-teacher-and-representation/experiments/exp-009-four-versus-eight.md) |
| [mnist_bounded_drn_teacher.md](mnist_bounded_drn_teacher.md) | [Independently trained bounded DRN reference](../campaigns/initialization-and-representation/series/003-bounded-learning/experiments/exp-010-bounded-teacher.md) |
| [measured_cohort_a_training.md](measured_cohort_a_training.md) | [Raw and isotonic cohort-A feasibility](../campaigns/measured-cohort-adaptation/series/001-transfer-and-write-policies/experiments/exp-001-cohort-a.md) |
| [measured_cohort_b_finetuning.md](measured_cohort_b_finetuning.md) | [Cohort-B transfer and learning-rate controls](../campaigns/measured-cohort-adaptation/series/001-transfer-and-write-policies/experiments/exp-002-cohort-b-transfer.md), [Fixed-rate programming deadbands](../campaigns/measured-cohort-adaptation/series/001-transfer-and-write-policies/experiments/exp-003-deadband.md), [Matched probabilistic write policies](../campaigns/measured-cohort-adaptation/series/001-transfer-and-write-policies/experiments/exp-004-probabilistic-writing.md) |
| [measured_cohort_b_lora_recovery.md](measured_cohort_b_lora_recovery.md) | [Corrected RESET passive LoRA recovery](../campaigns/measured-cohort-adaptation/series/002-low-rank-recovery/experiments/exp-005-cohort-b-lora.md) |
| [mnist_differential_cohort_ab_adaptation.md](mnist_differential_cohort_ab_adaptation.md) | [Differential A-to-B adaptation](../campaigns/measured-cohort-adaptation/series/003-representation-transfer/experiments/exp-006-differential-transfer.md) |
| [mnist_four_device_cohort_b_adaptation.md](mnist_four_device_cohort_b_adaptation.md) | [Four-device A-to-B adaptation](../campaigns/measured-cohort-adaptation/series/003-representation-transfer/experiments/exp-007-four-device-transfer.md) |
| [mnist_hwa_lora_experiment_setup.md](mnist_hwa_lora_experiment_setup.md) | [MNIST HWA and physical LoRA rank study](../campaigns/endpoint-hwa-recovery/series/002-mnist-low-rank/experiments/exp-004-rank-study.md), [Full-model rewriting control](../campaigns/endpoint-hwa-recovery/series/002-mnist-low-rank/experiments/exp-005-full-finetune.md) |
| [mnist_ibm_pcm_cmo_noisy_recovery.md](mnist_ibm_pcm_cmo_noisy_recovery.md) | [PCM/CMO repeated noisy recovery](../campaigns/endpoint-hwa-recovery/series/003-endpoint-and-floor/experiments/exp-006-pcm-cmo-noisy.md), [CMO literal-floor recovery](../campaigns/endpoint-hwa-recovery/series/003-endpoint-and-floor/experiments/exp-007-literal-floor.md), [Literal versus affine floor mechanisms](../campaigns/endpoint-hwa-recovery/series/003-endpoint-and-floor/experiments/exp-008-floor-mechanism.md) |
| [conductance_floor_mitigation.md](conductance_floor_mitigation.md) | [Literal versus affine floor mechanisms](../campaigns/endpoint-hwa-recovery/series/003-endpoint-and-floor/experiments/exp-008-floor-mechanism.md) |
| [mnist_wan_cmo_head_to_head.md](mnist_wan_cmo_head_to_head.md) | [Independently initialized CMO/Wan comparison](../campaigns/endpoint-hwa-recovery/series/003-endpoint-and-floor/experiments/exp-009-wan-cmo-independent.md) |
| [mnist_wan_cmo_teacher_initialized.md](mnist_wan_cmo_teacher_initialized.md) | [Teacher-initialized CMO/Wan comparison](../campaigns/endpoint-hwa-recovery/series/003-endpoint-and-floor/experiments/exp-010-wan-cmo-teacher.md) |
| [ibm_reram_program_verify_noise_model.md](ibm_reram_program_verify_noise_model.md) | [Original exhaustive controller design](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-001-exhaustive-design.md), [Short-production programming and endpoint model](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-002-short-production.md) |
| [ibm_reram_short_production_review.md](ibm_reram_short_production_review.md) | [Short-production programming and endpoint model](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-002-short-production.md), [128-pulse successor specification](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-003-cap128-successor.md), [Focused cap-128 HWA endpoint prerequisite](../campaigns/ibm-om-programming/series/001-controller-and-endpoint/experiments/exp-004-hwa-device-model.md) |
| [ibm_om_raw_active_common_cell_9_program_verify.md](ibm_om_raw_active_common_cell_9_program_verify.md) | [Array-wide nine-level physical-cell programming](../campaigns/ibm-om-programming/series/002-common-cell-levels/experiments/exp-005-nine-cell-levels.md) |
| [ibm_om_hwa_program_verify_pilot.md](ibm_om_hwa_program_verify_pilot.md) | [Initial population HWA/programming pilot](../campaigns/ibm-om-deployment-recovery/series/001-common-window/experiments/exp-001-population-hwa.md), [Corrected fixed-array common-window HWA](../campaigns/ibm-om-deployment-recovery/series/001-common-window/experiments/exp-002-fixed-array.md), [Interrupted eight-device differential-pair pilot](../campaigns/ibm-om-deployment-recovery/series/001-common-window/experiments/exp-003-eight-device.md) |
| [ibm_om_reset_relative_quantization.md](ibm_om_reset_relative_quantization.md) | [Shared RESET-relative nine-level QAT](../campaigns/ibm-om-deployment-recovery/series/002-reset-relative/experiments/exp-004-shared-quantization.md) |
| [ibm_om_reset_relative_on_chip_recovery.md](ibm_om_reset_relative_on_chip_recovery.md) | [Matched refresh, direct-pulse and Tiki-Taka recovery](../campaigns/ibm-om-deployment-recovery/series/002-reset-relative/experiments/exp-005-pulse-recovery.md) |
| [ibm_om_reset_relative_direct_gradient_thresholds.md](ibm_om_reset_relative_direct_gradient_thresholds.md) | [Direct-pulse gradient-threshold follow-up](../campaigns/ibm-om-deployment-recovery/series/002-reset-relative/experiments/exp-006-gradient-thresholds.md) |
| [ibm_om_raw_active_state_program_verify.md](ibm_om_raw_active_state_program_verify.md) | [Held-out raw-active seven-level QAT](../campaigns/ibm-om-deployment-recovery/series/003-raw-active/experiments/exp-009-seven-level-qat.md) |

## Existing presentation and figure assets

- [mnist_four_vs_eight_device_validation_accuracy.tex](mnist_four_vs_eight_device_validation_accuracy.tex) — retained at its existing path.
- [ibm_om_common_intervals_p50_p90_p99.png](ibm_om_common_intervals_p50_p90_p99.png) — retained at its existing path.
- [ibm_om_cell_bound_variation.png](ibm_om_cell_bound_variation.png) — retained at its existing path.
- [ibm_om_set_reset_state_distributions.png](ibm_om_set_reset_state_distributions.png) — generated local figure, still ignored by Git.

The LaTeX presentation combines four/eight-device and IBM OM range illustrations; it is linked by the initialization campaign. The amplification-equation verification artifacts are indexed by that campaign’s circuit-verification series. Personal notes are outside this migration.
