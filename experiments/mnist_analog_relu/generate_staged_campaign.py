"""Generate the immutable configs and study plans for the large v2 campaign.

This module is a development-time generator, not an alternate numerical
entrypoint.  Every generated task still executes through ``python -m ebl``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "examples" / "mnist_analog_relu" / "ibm_om_crossbar_staged_v2"
TEACHER_PLAN_ID = "mnist-ibm-om-crossbar-784-256-teacher-20260904-v1"
DEVELOPMENT_PLAN_ID = "mnist-ibm-om-crossbar-784-256-hwa-adam-tuning-20260904-v1"
PRODUCTION_PLAN_ID = "mnist-ibm-om-crossbar-784-256-production-20260904-v1"

HWA_ASSIGNMENT_SEED = 2_090_401
HWA_NOISE_SEED = 2_091_401
TUNING_ASSIGNMENT_SEED = 2_090_402
TUNING_ENDPOINT_SEED = 2_091_402
PRODUCTION_ENDPOINTS = {
    2_090_501: (2_090_601, 2_090_602, 2_090_603, 2_090_604),
    2_090_502: (2_090_611, 2_090_612, 2_090_613, 2_090_614),
    2_090_503: (2_090_621, 2_090_622, 2_090_623, 2_090_624),
    2_090_504: (2_090_631, 2_090_632, 2_090_633, 2_090_634),
}
ADAM_GRID = ((3e-4, 128), (3e-4, None), (1e-3, 128), (1e-3, None), (3e-3, 128), (3e-3, None))
ADAM_START_STATES = (
    "hwa_healthy_p0",
    "hwa_published_fault",
    "scratch_healthy_p0",
    "scratch_published_fault",
)


def _common(*, assignment_seed: int, endpoint_seed: int, profile: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "experiment_id": "mnist_ibm_om_crossbar_relu.v2",
        "runtime": {
            "seed": 42,
            "data_seed": 42,
            "device": "cuda",
            "dtype": "float32",
            "required_aihwkit_version": "1.1.0",
        },
        "data": {
            "batch_size": 16,
            "validation_points": 5000,
            "num_points": None,
            "shuffle": True,
        },
        "model": {
            "dims": [784, 256, 10],
            "bias": False,
            "digital_non_linearity": "relu",
            "maximum_input_size": 512,
            "maximum_output_size": 512,
        },
        "source": {
            "teacher_experiment_id": "mnist_relu.v2",
            "teacher_architecture": "bias_free_relu_784_256_10",
            "minimum_validation_accuracy": 0.97,
            "accuracy_comparison": ">",
        },
        "device": {
            "preset": "ReRamArrayOMPresetDevice",
            "evidence_class": "model_based_aihwkit_preset",
            "device_pair_topology": "one_active_om_plus_fixed_intrinsic_reference",
            "effective_state": "q_equals_a_minus_r",
            "assignment_seed": assignment_seed,
            "endpoint_seed": endpoint_seed,
            "healthy_programming_corruption_policy": "counterfactual_repaired",
            "fault_source_corruption_policy": "published",
            "bound_policy": "native_sampled_bounds",
            "reference_policy": "fixed_sampled_intrinsic_reference_q_equals_a_minus_r",
            "deterministic_codebook_pulses": 128,
            "maximum_programming_pulses": 128,
            "verify_tolerance_x": 0.023725,
            "controller": "one_pulse_apparent_verify_persistent_handoff",
            "fault_transition": "post_deployment_published_companion_replay",
            "fault_source_preset_default_corrupt_devices_prob": 0.0,
            "fault_source_enabled_corrupt_devices_prob": 0.1348,
            "fault_source_corrupt_devices_range": 0.01,
        },
        "mapping": {
            "trained_source_weight_scaling_omega": [1.0, 1.0],
            "trained_source_scaling_policy": "one_shared_absmax_scale_per_logical_layer",
            "scratch_weight_scaling_omega": [0.0, 0.0],
            "scratch_scaling_policy": "aihwkit_default_no_weight_scaling_direct_q",
            "scratch_initialization": "aihwkit_analog_linear_default_kaiming_uniform",
            "scratch_initialization_seed": 42,
            "out_of_bounds_policy": "retain_request_and_report_endpoint_saturation",
            "codebook_tie_rule": "lowest_pulse_index",
        },
        "evaluation": {"profile": profile},
    }


def _hwa() -> dict[str, Any]:
    value = _common(
        assignment_seed=HWA_ASSIGNMENT_SEED,
        endpoint_seed=HWA_NOISE_SEED,
        profile="tuning_validation_only",
    )
    value["stage"] = {
        "kind": "offchip_hwa",
        "policy": "stochastic_apparent_hwa",
        "training_protocol": "full_mnist_ten_epoch_fixed_final",
        "epochs": 10,
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "objective": "teacher_kl",
        "master_q_bounds": [-1.0, 1.0],
        "maximum_batches": 3438,
        "checkpoint_policy": "fixed_final_epoch_no_selection",
        "deployment_target": "fixed_final_master_fault_blind_pv",
        "evaluation_forward_policy": (
            "one_sampled_held_apparent_q_per_evaluation"
        ),
        "forward_noise": {
            "model": "aihwkit_1_1_0_softbounds_reference_additive_write_noise",
            "base_state": "support_clamped_digital_master_q_surrogate",
            "equation": "q_apparent_equals_q_digital_master_surrogate_plus_write_noise_std_times_nominal_dw_min_times_standard_normal",
            "resampling": "independent_full_array_draw_per_minibatch",
            "samples_per_minibatch": 1,
            "relative_scale": 1.0,
            "seed": HWA_NOISE_SEED,
            "gradient_estimator": "identity_straight_through",
        },
    }
    return value


def _deploy(*, assignment: int, endpoint: int, source: str, profile: str) -> dict[str, Any]:
    value = _common(assignment_seed=assignment, endpoint_seed=endpoint, profile=profile)
    value["stage"] = {"kind": "deploy", "source_kind": source}
    return value


def _corrupt(
    *, assignment: int, endpoint: int, source: str, profile: str
) -> dict[str, Any]:
    value = _common(assignment_seed=assignment, endpoint_seed=endpoint, profile=profile)
    value["stage"] = {
        "kind": "apply_corruption",
        "source_kind": source,
        "input_state_kind": "healthy_p0",
        "fault_source": "published_companion",
    }
    return value


def _adam(
    *,
    assignment: int,
    endpoint: int,
    profile: str,
    learning_rate: float | None = None,
    pulse_cap: int | None = None,
    selected: bool = False,
    start_state: str,
) -> dict[str, Any]:
    value = _common(assignment_seed=assignment, endpoint_seed=endpoint, profile=profile)
    hyperparameters: dict[str, Any]
    if selected:
        hyperparameters = {
            "source": "strict_selection_receipt",
            "allowed_learning_rates": [3e-4, 1e-3, 3e-3],
            "allowed_pulse_caps_per_cell": [128, None],
            "required_start_states": [
                "hwa_healthy_p0",
                "hwa_published_fault",
                "scratch_healthy_p0",
                "scratch_published_fault",
            ],
            "score": "mean_final_apparent_state_validation_cross_entropy_across_four_matched_starts",
            "tie_break_policy": "prefer_pulse_cap_128_then_lower_learning_rate",
        }
    else:
        if learning_rate is None:
            raise ValueError("Literal Adam configs require a learning rate.")
        hyperparameters = {
            "source": "literal",
            "learning_rate": learning_rate,
            "pulse_cap_per_cell": pulse_cap,
        }
    value["stage"] = {
        "kind": "on_chip_adam",
        "start_state": start_state,
        "epochs": 10,
        "optimizer": "pulse_adam",
        "objective": "supervised_cross_entropy",
        "repair_examples": 55000,
        "maximum_batches": 3438,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "layer_scope": "all",
        "forward_state": "held_apparent_q",
        "write_state": "persistent_q",
        "gradient_estimator": "identity_ste_apparent_q_to_persistent_pulse_update",
        "checkpoint_policy": "epoch_boundary_exact_resume_fixed_final",
        "hyperparameters": hyperparameters,
    }
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path, *, base: Path) -> str:
    return str(path.relative_to(ROOT) if base == ROOT else Path("..") / path.relative_to(ROOT))


def _arm(
    arm_id: str,
    description: str,
    configs: list[Path],
    *,
    experiment_id: str = "mnist_ibm_om_crossbar_relu.v2",
    mode: str = "train",
) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "description": description,
        "experiment_id": experiment_id,
        "mode": mode,
        "configs": [f"../{path.relative_to(ROOT)}" for path in configs],
    }


def _remove_superseded_generic_configs() -> None:
    """Remove only generator-owned pre-lineage filenames from this campaign."""

    legacy = [CONFIG_ROOT / "tuning" / "apply_corruption.json"]
    for rate, cap in ADAM_GRID:
        cap_token = "cap128" if cap == 128 else "uncapped"
        rate_token = {3e-4: "3e4", 1e-3: "1e3", 3e-3: "3e3"}[rate]
        legacy.append(CONFIG_ROOT / "tuning" / f"adam_lr{rate_token}_{cap_token}.json")
    for assignment, endpoints in PRODUCTION_ENDPOINTS.items():
        for endpoint in endpoints:
            suffix = f"a{assignment}_e{endpoint}.json"
            legacy.extend(
                (
                    CONFIG_ROOT / "production" / f"apply_corruption_{suffix}",
                    CONFIG_ROOT / "production" / f"adam_selected_{suffix}",
                )
            )
    for path in legacy:
        path.unlink(missing_ok=True)


def generate() -> dict[str, Path]:
    """Materialize all configs and the three predeclared study plans."""

    _remove_superseded_generic_configs()

    hwa_path = CONFIG_ROOT / "offchip_hwa.json"
    _write_json(hwa_path, _hwa())

    tuning_root = CONFIG_ROOT / "tuning"
    tune_hwa_deploy = tuning_root / "deploy_hwa_master.json"
    tune_scratch_deploy = tuning_root / "deploy_scratch.json"
    tune_corrupt = {
        source: tuning_root / f"apply_corruption_{source}.json"
        for source in ("hwa_master", "scratch")
    }
    _write_json(tune_hwa_deploy, _deploy(assignment=TUNING_ASSIGNMENT_SEED, endpoint=TUNING_ENDPOINT_SEED, source="hwa_master", profile="tuning_validation_only"))
    _write_json(tune_scratch_deploy, _deploy(assignment=TUNING_ASSIGNMENT_SEED, endpoint=TUNING_ENDPOINT_SEED, source="scratch", profile="tuning_validation_only"))
    for source, path in tune_corrupt.items():
        _write_json(
            path,
            _corrupt(
                assignment=TUNING_ASSIGNMENT_SEED,
                endpoint=TUNING_ENDPOINT_SEED,
                source=source,
                profile="tuning_validation_only",
            ),
        )
    tune_adam: dict[str, list[Path]] = {
        start_state: [] for start_state in ADAM_START_STATES
    }
    for start_state in ADAM_START_STATES:
        for rate, cap in ADAM_GRID:
            cap_token = "cap128" if cap == 128 else "uncapped"
            rate_token = {3e-4: "3e4", 1e-3: "1e3", 3e-3: "3e3"}[rate]
            path = tuning_root / f"adam_{start_state}_lr{rate_token}_{cap_token}.json"
            _write_json(
                path,
                _adam(
                    assignment=TUNING_ASSIGNMENT_SEED,
                    endpoint=TUNING_ENDPOINT_SEED,
                    profile="tuning_validation_only",
                    learning_rate=rate,
                    pulse_cap=cap,
                    start_state=start_state,
                ),
            )
            tune_adam[start_state].append(path)

    production_root = CONFIG_ROOT / "production"
    production: dict[str, list[Path]] = {
        "teacher": [],
        "hwa_master": [],
        "scratch": [],
        **{f"corruption_{source}": [] for source in ("teacher", "hwa_master", "scratch")},
        **{f"adam_{start_state}": [] for start_state in ADAM_START_STATES},
    }
    for assignment, endpoints in PRODUCTION_ENDPOINTS.items():
        for endpoint in endpoints:
            suffix = f"a{assignment}_e{endpoint}.json"
            for source in ("teacher", "hwa_master", "scratch"):
                path = production_root / f"deploy_{source}_{suffix}"
                _write_json(path, _deploy(assignment=assignment, endpoint=endpoint, source=source, profile="production_full_validation_and_test"))
                production[source].append(path)
            for source in ("teacher", "hwa_master", "scratch"):
                corrupt_path = production_root / f"apply_corruption_{source}_{suffix}"
                _write_json(
                    corrupt_path,
                    _corrupt(
                        assignment=assignment,
                        endpoint=endpoint,
                        source=source,
                        profile="production_full_validation_and_test",
                    ),
                )
                production[f"corruption_{source}"].append(corrupt_path)
            for start_state in ADAM_START_STATES:
                adam_path = production_root / f"adam_{start_state}_{suffix}"
                _write_json(
                    adam_path,
                    _adam(
                        assignment=assignment,
                        endpoint=endpoint,
                        profile="production_full_validation_and_test",
                        selected=True,
                        start_state=start_state,
                    ),
                )
                production[f"adam_{start_state}"].append(adam_path)

    teacher_plan = ROOT / "studies" / f"{TEACHER_PLAN_ID}.json"
    _write_json(
        teacher_plan,
        {
            "schema_version": 1,
            "study_id": TEACHER_PLAN_ID,
            "title": "Accepted 784-256-10 bias-free MNIST ReLU teacher",
            "hypothesis": "A bias-free 784-256-10 ReLU network trained for 30 epochs with the predeclared Adam protocol will produce a minimum-validation-cross-entropy checkpoint whose validation accuracy is strictly above 97%.",
            "motivation": "Every crossbar branch must start from one exact accepted teacher rather than a discovered or substituted checkpoint.",
            "evidence_class": "digital_teacher_prerequisite_cuda",
            "arms": [
                _arm("teacher", "Train and select the sole source teacher; reject accuracy at or below 97%.", [ROOT / "examples" / "mnist_relu" / "teacher_784_256_10_cuda.json"], experiment_id="mnist_relu.v2"),
                _arm("teacher-test", "Evaluate the accepted selected checkpoint once on the full official MNIST test set.", [ROOT / "examples" / "mnist_relu" / "teacher_784_256_10_cuda.json"], experiment_id="mnist_relu.v2", mode="validate"),
            ],
            "completion_criteria": [
                "The native CUDA run completes with artifact verification and a selected validation accuracy strictly greater than 0.97.",
                "The selected checkpoint records bias-free dimensions 784-256-10, seed 42, the stratified 55,000/5,000 split, and minimum validation cross-entropy selection.",
                "The accepted checkpoint is evaluated once on all 10,000 official test examples without using test performance for selection.",
                "No crossbar stage starts from a different checkpoint or a newest-checkpoint lookup.",
            ],
            "analysis_plan": [
                "Verify the strict greater-than gate, checkpoint metadata, source hash, selected epoch, validation loss/accuracy, and full official test accuracy.",
                "Treat failure of the gate as a failed prerequisite, not as evidence about crossbar deployment.",
            ],
        },
    )

    development_plan = ROOT / "studies" / f"{DEVELOPMENT_PLAN_ID}.json"
    starts = {
        "hwa_healthy_p0": ("hwa-healthy", "HWA master programmed and recovered on its exact healthy tuning P0."),
        "hwa_published_fault": ("hwa-corrupt", "The exact HWA tuning P0 faulted before recovery."),
        "scratch_healthy_p0": ("scratch-healthy", "Kaiming initialization programmed and recovered on its exact healthy tuning P0."),
        "scratch_published_fault": ("scratch-corrupt", "The exact scratch tuning P0 faulted before recovery."),
    }
    dev_arms = [
        _arm("offchip-hwa", "Ten fixed epochs of stochastic apparent-state teacher-KL HWA.", [hwa_path]),
        _arm("tune-hwa-deploy", "Program one HWA tuning realization.", [tune_hwa_deploy]),
        _arm("tune-hwa-corrupt", "Apply the published companion faults to that exact HWA P0.", [tune_corrupt["hwa_master"]]),
        _arm("tune-scratch-deploy", "Program one matched scratch tuning realization.", [tune_scratch_deploy]),
        _arm("tune-scratch-corrupt", "Apply the published companion faults to that exact scratch P0.", [tune_corrupt["scratch"]]),
    ]
    dev_arms.extend(
        _arm(f"tune-adam-{arm_suffix}", description, tune_adam[start_state])
        for start_state, (arm_suffix, description) in starts.items()
    )
    _write_json(
        development_plan,
        {
            "schema_version": 1,
            "study_id": DEVELOPMENT_PLAN_ID,
            "title": "Stochastic HWA and one-write Adam tuning for the 784-256-10 IBM-OM crossbar",
            "hypothesis": "A single Adam learning-rate and lifetime-cap pair selected by equally weighting HWA/scratch and healthy/published-fault starts will provide a fair fixed protocol for the independent production arrays.",
            "motivation": "Tuning must not consume any production assignment or test outcome, and healthy/corrupt candidates must fork identical programmed P0 artifacts.",
            "evidence_class": "model_based_aihwkit_om_protocol_development_cuda",
            "arms": dev_arms,
            "completion_criteria": [
                "The HWA master completes ten 55,000-example epochs on CUDA using sampled apparent q for every training forward and no persistent plant update.",
                "Initial, every epoch, and fixed-final HWA validation each use one explicitly seeded apparent-q sample held across all 5,000 examples; the support-clamped digital master is reported only as a nonpersistent diagnostic.",
                "The HWA stage is validation-only and never evaluates the official test set during protocol development.",
                "Exactly one tuning assignment and one programming seed create the HWA and scratch P0 sources; each corruption arm reuses its exact P0 without reprogramming.",
                "All six learning-rate/cap candidates complete ten epochs from all four exact starts, for complete 6-by-4 coverage.",
                "Selection minimizes equally weighted final validation cross-entropy; candidates within 1e-6 prefer cap 128 and then lower learning rate. Test metrics cannot select.",
                "The artifact-verified analysis writes one immutable strict selection receipt for production.",
            ],
            "analysis_plan": [
                "Verify common P0 ancestry, identical assignment/write seeds, train order, optimizer budget, and apparent-forward/persistent-write semantics across candidates.",
                "Verify every HWA validation receipt's independent derived seed, generator-state hashes, held apparent-q hash, complete 5,000-example coverage, and separately labelled nonpersistent support-clamped digital-master diagnostic.",
                "Report all 24 final validation losses and accuracies, candidate means, pulse/cap telemetry, deterministic tie handling, and the selected pair.",
                "Do not treat this one assignment/write tuning phase as array-transfer evidence.",
            ],
        },
    )

    production_plan = ROOT / "studies" / f"{PRODUCTION_PLAN_ID}.json"
    production_arms = [
        _arm("direct-deploy", "Direct teacher mapping followed only by P&V on 4x4 independent assignment/write realizations.", production["teacher"]),
        _arm("direct-corrupt", "Published-companion corruption of every exact direct P0, with no training.", production["corruption_teacher"]),
        _arm("hwa-deploy", "Frozen HWA master followed by P&V on the same 4x4 seed matrix.", production["hwa_master"]),
        _arm("hwa-corrupt", "Published-companion corruption of every exact HWA P0, with no training.", production["corruption_hwa_master"]),
        _arm("hwa-adam-healthy", "Selected Adam recovery from every exact healthy HWA P0.", production["adam_hwa_healthy_p0"]),
        _arm("hwa-adam-corrupt", "Selected Adam recovery from every exact faulted HWA P0.", production["adam_hwa_published_fault"]),
        _arm("scratch-deploy-preparation", "Matched Kaiming scratch P0 preparation on the 4x4 matrix.", production["scratch"]),
        _arm("scratch-corrupt-preparation", "Published-companion corruption of every exact scratch P0.", production["corruption_scratch"]),
        _arm("scratch-adam-healthy", "Selected Adam from scratch on every healthy programmed array.", production["adam_scratch_healthy_p0"]),
        _arm("scratch-adam-corrupt", "Selected Adam from scratch on every faulted programmed array.", production["adam_scratch_published_fault"]),
    ]
    _write_json(
        production_plan,
        {
            "schema_version": 1,
            "study_id": PRODUCTION_PLAN_ID,
            "title": "Large 4x4 IBM-OM standard-crossbar deployment and recovery comparison",
            "hypothesis": "Off-chip HWA and same-array Adam recovery will reduce apparent-state deployment loss relative to direct P&V, while the matched scratch arms will show whether recovery benefits arise from the teacher/HWA initialization or can be reached by on-array training alone.",
            "motivation": "This matched ladder separates deterministic mapping, stochastic P&V, post-write corruption, off-chip adaptation, same-array recovery, and fresh scratch training for the explicit standard-MVM digital-ReLU control.",
            "evidence_class": "model_based_aihwkit_om_standard_crossbar_control_cuda",
            "arms": production_arms,
            "completion_criteria": [
                "Every declared native run completes on CUDA with valid artifacts: 4 assignment seeds by 4 write seeds for each outcome family.",
                "Direct and HWA corrupt branches reuse their exact healthy P0; HWA and scratch Adam branches start from the exact named healthy or faulted state and never reprogram it.",
                "Every device-facing forward, gradient, checkpoint choice, and headline metric uses held apparent q; persistent q is reported only as a labelled diagnostic and controls subsequent pulses.",
                "The production Adam settings come only from the strict artifact-verified tuning receipt, with ten fixed epochs, no test selection, and exact epoch resume.",
                "All full 5,000-example validation and 10,000-example official test metrics, P&V telemetry, fault damage, recovery gain, pulse/cap telemetry, input hashes, and immutable commands are present.",
                "Assignment-level summaries average the four writes first and report mean, SD, and range across four assignments; pooled 16-realization summaries are secondary.",
            ],
            "analysis_plan": [
                "For each of eight headline outcomes report apparent accuracy/CE/KL/agreement first and persistent diagnostics second, with source-to-deployment loss, fault damage, and recovery gain.",
                "Average four programming writes within each assignment before computing the primary four-assignment mean, SD, and range; also report pooled 16-realization distributions.",
                "Compare direct versus HWA deployment, HWA healthy/corrupt recovery, and scratch healthy/corrupt training using matched assignment/write seeds and the same teacher, split, objective, data order, and update budget.",
                "Keep conclusions descriptive for the model-based standard-crossbar control; do not generalize to the DRN, fabricated devices, or a universal need for on-chip training.",
            ],
        },
    )
    return {"teacher": teacher_plan, "development": development_plan, "production": production_plan}


if __name__ == "__main__":
    for name, path in generate().items():
        print(f"{name}: {path}")
