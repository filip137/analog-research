from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_analog_relu import staged_runtime as runtime
from experiments.mnist_analog_relu.staged_config import (
    ApplyCorruptionStageSettings,
    DataSettings,
    DeployStageSettings,
    DeviceSettings,
    DiagnosticLiteralAdamHyperparameters,
    FreshApparentDiagnosticStageSettings,
    EvaluationSettings,
    LiteralAdamHyperparameters,
    MappingSettings,
    ModelSettings,
    OnChipAdamStageSettings,
    OnChipAdamDiagnosticStageSettings,
    RuntimeSettings,
    SelectionReceiptAdamHyperparameters,
    StagedCrossbarTrainSpec,
    TeacherSourceSettings,
)
from experiments.mnist_analog_relu.staged_runtime import (
    _ADAM_IMPLEMENTATION,
    _adam_objective_loss,
    _adam_selection_objective_metric,
    _adam_selection_rank,
    _authenticate_selected_teacher_metrics,
    _fresh_apparent_diagnostic_seed,
    _layout,
    _load_teacher,
    _runtime_receipt,
    _resolve_adam_settings,
    _run_fresh_apparent_diagnostic,
    _validate_diagnostic_resume,
    _validate_origin,
    _validate_request,
    _validate_resume,
    _validate_selection_receipt,
)
from training.ibm_om_standard_crossbar import tensor_sha256
from training.checkpoint import encode_named_weights, save_encoded_named_weights
from experiments.mnist_relu.model import BiasFreeReluTeacher


def _device() -> DeviceSettings:
    return DeviceSettings(
        preset="ReRamArrayOMPresetDevice",
        evidence_class="model_based_aihwkit_preset",
        device_pair_topology="one_active_om_plus_fixed_intrinsic_reference",
        effective_state="q_equals_a_minus_r",
        assignment_seed=2090402,
        endpoint_seed=2091402,
        healthy_programming_corruption_policy="counterfactual_repaired",
        fault_source_corruption_policy="published",
        bound_policy="native_sampled_bounds",
        reference_policy="fixed_sampled_intrinsic_reference_q_equals_a_minus_r",
        deterministic_codebook_pulses=128,
        maximum_programming_pulses=128,
        verify_tolerance_x=0.023725,
        controller="one_pulse_apparent_verify_persistent_handoff",
        fault_transition="post_deployment_published_companion_replay",
        fault_source_preset_default_corrupt_devices_prob=0.0,
        fault_source_enabled_corrupt_devices_prob=0.1348,
        fault_source_corrupt_devices_range=0.01,
    )


def _mapping() -> MappingSettings:
    return MappingSettings(
        trained_source_weight_scaling_omega=(1.0, 1.0),
        trained_source_scaling_policy="one_shared_absmax_scale_per_logical_layer",
        scratch_weight_scaling_omega=(0.0, 0.0),
        scratch_scaling_policy="aihwkit_default_no_weight_scaling_direct_q",
        scratch_initialization="aihwkit_analog_linear_default_kaiming_uniform",
        scratch_initialization_seed=42,
        out_of_bounds_policy="retain_request_and_report_endpoint_saturation",
        codebook_tie_rule="lowest_pulse_index",
    )


def _adam_stage(*, receipt: bool) -> OnChipAdamStageSettings:
    hyperparameters = (
        SelectionReceiptAdamHyperparameters(
            source="strict_selection_receipt",
            allowed_learning_rates=(3e-4, 1e-3, 3e-3),
            allowed_pulse_caps_per_cell=(128, None),
            required_start_states=(
                "hwa_healthy_p0",
                "hwa_published_fault",
                "scratch_healthy_p0",
                "scratch_published_fault",
            ),
            score=(
                "mean_final_apparent_state_validation_cross_entropy_"
                "across_four_matched_starts"
            ),
            tie_break_policy="prefer_pulse_cap_128_then_lower_learning_rate",
        )
        if receipt
        else LiteralAdamHyperparameters(
            source="literal",
            learning_rate=1e-3,
            pulse_cap_per_cell=128,
        )
    )
    return OnChipAdamStageSettings(
        kind="on_chip_adam",
        start_state="hwa_healthy_p0",
        epochs=10,
        optimizer="pulse_adam",
        objective="supervised_cross_entropy",
        repair_examples=55000,
        maximum_batches=3438,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        layer_scope="all",
        forward_state="held_apparent_q",
        write_state="persistent_q",
        gradient_estimator="identity_ste_apparent_q_to_persistent_pulse_update",
        checkpoint_policy="epoch_boundary_exact_resume_fixed_final",
        hyperparameters=hyperparameters,
    )


def _diagnostic_adam_stage(
    *, objective: str = "teacher_kl", learning_rate: float = 1e-5
) -> OnChipAdamDiagnosticStageSettings:
    return OnChipAdamDiagnosticStageSettings(
        kind="on_chip_adam_diagnostic",
        start_state="hwa_healthy_p0",
        epochs=3,
        optimizer="pulse_adam",
        objective=objective,
        repair_examples=55000,
        maximum_batches=3438,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        layer_scope="all",
        forward_state="held_apparent_q",
        write_state="persistent_q",
        gradient_estimator="identity_ste_apparent_q_to_persistent_pulse_update",
        checkpoint_policy=(
            "best_held_apparent_validation_accuracy_then_objective_"
            "then_earlier_epoch_including_epoch0"
        ),
        hyperparameters=DiagnosticLiteralAdamHyperparameters(
            source="literal_diagnostic_grid",
            learning_rate=learning_rate,
            pulse_cap_per_cell=640,
        ),
    )


def _spec(stage, *, production: bool = False) -> StagedCrossbarTrainSpec:
    return StagedCrossbarTrainSpec(
        experiment_id="mnist_ibm_om_crossbar_relu.v2",
        runtime=RuntimeSettings(42, 42, "cuda", "float32", "1.1.0"),
        data=DataSettings(16, 5000, None, True),
        model=ModelSettings((784, 256, 10), False, "relu", 512, 512),
        source=TeacherSourceSettings(
            "mnist_relu.v2", "bias_free_relu_784_256_10", 0.97, ">"
        ),
        device=_device(),
        mapping=_mapping(),
        evaluation=EvaluationSettings(
            profile=(
                "production_full_validation_and_test"
                if production
                else (
                    "diagnostic_validation_only"
                    if isinstance(stage, OnChipAdamDiagnosticStageSettings)
                    else "tuning_validation_only"
                )
            ),
            evaluate_validation=True,
            evaluate_test=production,
            validation_points=5000,
            test_points=10000 if production else None,
            selection_metric=(
                "stage_declared_held_apparent_validation"
                if isinstance(stage, OnChipAdamDiagnosticStageSettings)
                else "fixed_final_epoch_no_selection"
            ),
            primary_state="held_apparent_q",
            secondary_state="hidden_persistent_q_diagnostic",
        ),
        stage=stage,
    )


def _request(spec, **paths):
    values = {
        "spec": spec,
        "weights": None,
        "base_weights": None,
        "resume": None,
        "device_data": None,
        "device_model": None,
        "teacher_weights": None,
        "device_state": None,
        "selection_receipt": None,
    }
    values.update(paths)
    return SimpleNamespace(**values)


def _touch(path: Path) -> Path:
    path.write_bytes(b"artifact")
    return path


def _selection_receipt() -> dict:
    starts = (
        "hwa_healthy_p0",
        "hwa_published_fault",
        "scratch_healthy_p0",
        "scratch_published_fault",
    )
    candidates = []
    for rate in (3e-4, 1e-3, 3e-3):
        for cap in (128, None):
            scores = {name: 0.2 for name in starts}
            candidates.append(
                {
                    "learning_rate": rate,
                    "pulse_cap_per_cell": cap,
                    "start_scores": scores,
                    "result_sha256_by_start": {
                        name: f"{len(candidates) + 1:x}" * 64 for name in starts
                    },
                    "origin_device_state_sha256_by_start": {
                        name: f"{index + 10:x}" * 64
                        for index, name in enumerate(starts)
                    },
                    "mean_validation_cross_entropy": 0.2,
                }
            )
    return {
        "schema": "ebl.ibm_om_crossbar_adam_selection",
        "schema_version": 1,
        "study_id": "tuning-study",
        "source_plan_sha256": "a" * 64,
        "required_start_states": list(starts),
        "origin_device_state_sha256_by_start": {
            name: f"{index + 10:x}" * 64
            for index, name in enumerate(starts)
        },
        "grid": {
            "learning_rates": [3e-4, 1e-3, 3e-3],
            "pulse_caps_per_cell": [128, None],
        },
        "score": (
            "mean_final_apparent_state_validation_cross_entropy_"
            "across_four_matched_starts"
        ),
        "tie_tolerance": 1e-6,
        "tie_break_policy": "prefer_pulse_cap_128_then_lower_learning_rate",
        "candidates": candidates,
        "winner": {
            "learning_rate": 3e-4,
            "pulse_cap_per_cell": 128,
            "mean_validation_cross_entropy": 0.2,
        },
    }


def test_fixed_layout_has_three_tiles_and_exact_logical_cell_count() -> None:
    layout = _layout(_spec(DeployStageSettings("deploy", "teacher")))
    assert [tile.shape for tile in layout] == [(392, 256), (392, 256), (256, 10)]
    assert sum(tile.cells for tile in layout) == 203264


def test_selected_teacher_metrics_must_replay() -> None:
    metadata = {"selection_value": 0.08, "selection_accuracy": 0.975}
    validation = {"cross_entropy": 0.080000001, "accuracy": 0.975}
    _authenticate_selected_teacher_metrics(metadata, validation)

    with pytest.raises(ValueError, match="reproduce"):
        _authenticate_selected_teacher_metrics(
            metadata,
            {"cross_entropy": 0.09, "accuracy": 0.975},
        )


def test_teacher_artifact_authenticates_data_and_preprocessing_contract(
    tmp_path: Path,
) -> None:
    teacher = BiasFreeReluTeacher(
        device=torch.device("cpu"), dims=(784, 256, 10)
    )
    metadata = {
        "experiment_id": "mnist_relu.v2",
        "architecture": "bias_free_relu_784_256_10",
        "logical_dims": [784, 256, 10],
        "bias": False,
        "runtime_seed": 42,
        "data_seed": 42,
        "batch_size": 16,
        "train_shuffle": True,
        "training_examples": 55000,
        "validation_examples": 5000,
        "validation_split": (
            "torchvision_mnist_train_stratified_per_class_"
            "numpy_default_rng_sorted_indices"
        ),
        "preprocessing": "to_tensor_flatten_normalize_mean_0.1307_std_0.3",
        "selection_metric": "validation.cross_entropy",
        "selection_value": 0.08,
        "selection_accuracy": 0.975,
        "selection_epoch": 20,
        "validation_accuracy_comparison_operator": ">",
        "minimum_validation_accuracy": 0.97,
    }
    path = tmp_path / "teacher.pt"
    save_encoded_named_weights(
        path,
        encode_named_weights(teacher.catalog, metadata=metadata),
        catalog=teacher.catalog,
    )
    _loaded, loaded_metadata, _digest = _load_teacher(
        path,
        spec=_spec(DeployStageSettings("deploy", "teacher")),
        device=torch.device("cpu"),
    )
    assert loaded_metadata == metadata

    bad = dict(metadata)
    bad["data_seed"] = 43
    save_encoded_named_weights(
        path,
        encode_named_weights(teacher.catalog, metadata=bad),
        catalog=teacher.catalog,
    )
    with pytest.raises(ValueError, match="authenticated"):
        _load_teacher(
            path,
            spec=_spec(DeployStageSettings("deploy", "teacher")),
            device=torch.device("cpu"),
        )


def test_runtime_receipt_explicitly_authenticates_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "test GPU")
    assert _runtime_receipt(torch.device("cuda:3")) == {
        "configured_device": "cuda",
        "resolved_device": "cuda:3",
        "cuda_available": True,
        "cuda_device_name": "test GPU",
    }


def test_adam_is_explicitly_labeled_as_digitally_assisted() -> None:
    assert _ADAM_IMPLEMENTATION == {
        "digitally_assisted": True,
        "gradient_engine": "digital_full_apparent_q_autograd",
        "optimizer_moments": "digital",
        "fully_local_learning_primitive": False,
    }


@pytest.mark.parametrize(
    ("objective", "metric"),
    [
        (
            "supervised_cross_entropy",
            "validation.apparent_forward.cross_entropy",
        ),
        ("teacher_kl", "validation.apparent_forward.kl_teacher_student"),
    ],
)
def test_diagnostic_adam_accuracy_first_selection_and_objective_tie_break_are_matched(
    objective: str, metric: str
) -> None:
    stage = _diagnostic_adam_stage(objective=objective, learning_rate=0.0)
    rate, cap, selection = _resolve_adam_settings(
        stage=stage,
        profile="diagnostic_validation_only",
        selection_path=None,
    )

    assert (rate, cap) == (0.0, 640)
    assert selection == {
        "source": "literal_diagnostic_grid",
        "checkpoint_policy": (
            "best_held_apparent_validation_accuracy_then_objective_"
            "then_earlier_epoch_including_epoch0"
        ),
        "objective": objective,
        "learning_rate_schedule": None,
    }
    assert _adam_selection_objective_metric(objective) == (
        metric,
        metric.rsplit(".", 1)[-1],
    )

    candidates = [
        (0.95, 0.01, 0),
        (0.96, 0.20, 1),
        (0.96, 0.10, 2),
        (0.96, 0.10, 3),
    ]
    assert min(
        candidates,
        key=lambda item: _adam_selection_rank(
            student_accuracy=item[0], objective_value=item[1], epoch=item[2]
        ),
    ) == (0.96, 0.10, 2)


def test_teacher_kl_objective_matches_evaluation_direction_and_backpropagates() -> None:
    logits = torch.tensor(
        [[0.2, -0.1, 0.7], [-0.4, 0.9, 0.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [[0.6, 0.0, -0.2], [-0.1, 0.2, 0.8]], dtype=torch.float32
    )
    teacher = SimpleNamespace(logits=lambda _inputs: teacher_logits)
    inputs = torch.zeros((2, 1), dtype=torch.float32)
    labels = torch.tensor([2, 1], dtype=torch.long)

    loss = _adam_objective_loss(
        objective="teacher_kl",
        logits=logits,
        labels=labels,
        inputs=inputs,
        teacher=teacher,
    )
    teacher_log_probability = torch.log_softmax(teacher_logits, dim=1)
    expected = (
        teacher_log_probability.exp()
        * (teacher_log_probability - torch.log_softmax(logits, dim=1))
    ).sum(dim=1).mean()

    assert torch.equal(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.all(torch.isfinite(logits.grad)))
    assert float(logits.grad.abs().sum()) > 0.0


def test_selection_receipt_recomputes_full_grid_and_tie_break() -> None:
    rate, cap, report = _validate_selection_receipt(_selection_receipt())
    assert (rate, cap) == (3e-4, 128)
    assert report["candidate_count"] == 6
    assert report["winner_recomputed"] is True
    assert "result_sha256_by_start" in report["validated_receipt"]["candidates"][0]
    assert set(report["validated_receipt"]["origin_device_state_sha256_by_start"]) == {
        "hwa_healthy_p0",
        "hwa_published_fault",
        "scratch_healthy_p0",
        "scratch_published_fault",
    }


@pytest.mark.parametrize("mutation", ["missing_start", "wrong_mean", "wrong_winner"])
def test_selection_receipt_rejects_incomplete_or_false_result(mutation: str) -> None:
    receipt = _selection_receipt()
    if mutation == "missing_start":
        receipt["candidates"][0]["start_scores"].pop("scratch_published_fault")
    elif mutation == "wrong_mean":
        receipt["candidates"][0]["mean_validation_cross_entropy"] = 0.1
    else:
        receipt["winner"]["pulse_cap_per_cell"] = None
    with pytest.raises(ValueError):
        _validate_selection_receipt(receipt)


def test_on_chip_request_requires_origin_and_production_receipt(tmp_path: Path) -> None:
    teacher = _touch(tmp_path / "teacher.pt")
    origin = _touch(tmp_path / "origin.pt")
    receipt = _touch(tmp_path / "selection.json")
    spec = _spec(_adam_stage(receipt=True), production=True)

    with pytest.raises(ValueError, match="selection-receipt"):
        _validate_request(
            _request(spec, teacher_weights=teacher, device_state=origin)
        )
    records, sampler = _validate_request(
        _request(
            spec,
            teacher_weights=teacher,
            device_state=origin,
            selection_receipt=receipt,
        )
    )
    assert sampler is None
    assert [item["role"] for item in records] == [
        "teacher_weights",
        "origin_device_state",
        "adam_selection_receipt",
    ]


def test_diagnostic_adam_request_requires_only_exact_origin_and_allows_resume(
    tmp_path: Path,
) -> None:
    teacher = _touch(tmp_path / "teacher.pt")
    origin = _touch(tmp_path / "origin.pt")
    resume = _touch(tmp_path / "resume.pt")
    receipt = _touch(tmp_path / "selection.json")
    spec = _spec(_diagnostic_adam_stage())

    records, sampler = _validate_request(
        _request(
            spec,
            teacher_weights=teacher,
            device_state=origin,
            resume=resume,
        )
    )
    assert sampler is None
    assert [item["role"] for item in records] == [
        "teacher_weights",
        "origin_device_state",
        "adam_epoch_resume",
    ]
    with pytest.raises(ValueError, match="selection-receipt"):
        _validate_request(
            _request(
                spec,
                teacher_weights=teacher,
                device_state=origin,
                selection_receipt=receipt,
            )
        )


def test_fresh_apparent_request_forbids_resume_and_receipt(tmp_path: Path) -> None:
    teacher = _touch(tmp_path / "teacher.pt")
    origin = _touch(tmp_path / "origin.pt")
    resume = _touch(tmp_path / "resume.pt")
    stage = FreshApparentDiagnosticStageSettings(
        kind="fresh_apparent_diagnostic",
        start_state="hwa_healthy_p0",
        intervention=(
            "counterfactual_post_write_apparent_noise_redraw_without_device_write"
        ),
        draws=4,
        relative_scale=1.0,
        seed=2090701,
        mutation_policy="do_not_mutate_held_apparent_or_persistent_state",
        interpretation="diagnostic_only_not_physical_inference_read_noise",
    )
    spec = _spec(stage)

    records, sampler = _validate_request(
        _request(spec, teacher_weights=teacher, device_state=origin)
    )
    assert sampler is None
    assert [item["role"] for item in records] == [
        "teacher_weights",
        "origin_device_state",
    ]
    with pytest.raises(ValueError, match="--resume"):
        _validate_request(
            _request(
                spec,
                teacher_weights=teacher,
                device_state=origin,
                resume=resume,
            )
        )


def test_corruption_rejects_resume_and_deploy_hwa_requires_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher = _touch(tmp_path / "teacher.pt")
    state = _touch(tmp_path / "state.pt")
    resume = _touch(tmp_path / "resume.pt")
    corruption = _spec(
        ApplyCorruptionStageSettings(
            "apply_corruption", "teacher", "healthy_p0", "published_companion"
        )
    )
    with pytest.raises(ValueError, match="--resume"):
        _validate_request(
            _request(
                corruption,
                teacher_weights=teacher,
                device_state=state,
                resume=resume,
            )
        )

    monkeypatch.setenv("EBL_AIHWKIT_PYTHON", "/bin/true")
    deployment = _spec(DeployStageSettings("deploy", "hwa_master"))
    with pytest.raises(ValueError, match="--weights"):
        _validate_request(_request(deployment, teacher_weights=teacher))
    master = _touch(tmp_path / "hwa.pt")
    records, sampler = _validate_request(
        _request(deployment, teacher_weights=teacher, weights=master)
    )
    assert sampler == Path("/bin/true").resolve()
    assert [item["role"] for item in records] == [
        "teacher_weights",
        "hwa_master",
        "aihwkit_python",
    ]


def test_origin_validation_cross_authenticates_bundle_metadata() -> None:
    spec = _spec(_adam_stage(receipt=False))
    layout = _layout(spec)
    metadata = {
        "stage": "deploy",
        "source_kind": "scratch",
        "source_artifact_sha256": None,
        "teacher_sha256": "a" * 64,
        "assignment_seed": spec.device.assignment_seed,
        "endpoint_seed": spec.device.endpoint_seed,
    }
    bundle = SimpleNamespace(
        layout=layout,
        digital_scales=(1.0, 1.0),
        metadata=metadata,
        metadata_sha256="b" * 64,
        plant_state_sha256="c" * 64,
    )
    origin = SimpleNamespace(
        role="healthy_p0",
        source_kind="scratch",
        dims=(784, 256, 10),
        assignment_seed=spec.device.assignment_seed,
        endpoint_seed=spec.device.endpoint_seed,
        teacher_sha256="a" * 64,
        source_artifact_sha256=None,
        parent_device_state_sha256=None,
        healthy_p0=bundle,
        current=bundle,
    )
    _validate_origin(
        origin,
        spec=spec,
        teacher_sha256="a" * 64,
        layout=layout,
        allowed_roles={"healthy_p0"},
    )
    bad_bundle = SimpleNamespace(**{**vars(bundle), "metadata": {**metadata, "endpoint_seed": 7}})
    with pytest.raises(ValueError, match="exact staged configuration"):
        _validate_origin(
            SimpleNamespace(**{**vars(origin), "healthy_p0": bad_bundle, "current": bad_bundle}),
            spec=spec,
            teacher_sha256="a" * 64,
            layout=layout,
            allowed_roles={"healthy_p0"},
        )


def test_resume_validation_binds_complete_selection_summary() -> None:
    apparent = torch.tensor([0.1], dtype=torch.float32)
    persistent = torch.tensor([0.2], dtype=torch.float32)
    current = SimpleNamespace(
        layout=("layout",),
        digital_scales=(1.0, 1.0),
        state_kind="healthy",
        plant_state={"apparent": apparent, "persistent": persistent},
    )
    healthy_p0 = SimpleNamespace(plant_state_sha256="b" * 64)
    population = SimpleNamespace(fingerprint="population")
    origin = SimpleNamespace(
        source_kind="scratch",
        dims=(784, 256, 10),
        assignment_seed=1,
        endpoint_seed=2,
        teacher_sha256="c" * 64,
        source_artifact_sha256=None,
        healthy_population=population,
        published_population=population,
        healthy_p0=healthy_p0,
        current=current,
    )
    selection = {"source": "literal_predeclared_grid"}
    report = {
        "epoch": 1,
        "apparent_sha256": tensor_sha256(apparent),
        "persistent_sha256": tensor_sha256(persistent),
    }
    recovery = {
        "schema": "ebl.ibm_om_crossbar_adam_recovery",
        "schema_version": 1,
        "origin_device_state_sha256": "d" * 64,
        "start_state": "scratch_healthy_p0",
        "completed_epochs": 1,
        "total_epochs": 10,
        "learning_rate": 1e-3,
        "pulse_cap_per_cell": 128,
        "optimizer_state_dict": {},
        "train_generator_state": torch.Generator().get_state(),
        "epoch_reports": [report],
        "initial": {},
        "selection": selection,
    }
    resume = SimpleNamespace(
        role="adam_final",
        source_kind=origin.source_kind,
        dims=origin.dims,
        assignment_seed=origin.assignment_seed,
        endpoint_seed=origin.endpoint_seed,
        teacher_sha256=origin.teacher_sha256,
        source_artifact_sha256=origin.source_artifact_sha256,
        parent_device_state_sha256="d" * 64,
        healthy_population=population,
        published_population=population,
        healthy_p0=healthy_p0,
        current=current,
        recovery=recovery,
    )
    _validate_resume(
        resume,
        origin=origin,
        origin_sha="d" * 64,
        start_state="scratch_healthy_p0",
        learning_rate=1e-3,
        pulse_cap=128,
        total_epochs=10,
        selection=selection,
    )
    with pytest.raises(ValueError, match="exact epoch-boundary"):
        _validate_resume(
            resume,
            origin=origin,
            origin_sha="d" * 64,
            start_state="scratch_healthy_p0",
            learning_rate=1e-3,
            pulse_cap=128,
            total_epochs=10,
            selection={"source": "different_receipt"},
        )


def test_diagnostic_resume_accepts_selected_epoch_zero_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apparent = torch.tensor([0.1], dtype=torch.float32)
    persistent = torch.tensor([0.2], dtype=torch.float32)
    current = SimpleNamespace(
        layout=("layout",),
        digital_scales=(1.0, 1.0),
        state_kind="healthy",
        population_fingerprint="population",
        plant_state={"apparent": apparent, "persistent": persistent},
        plant_state_sha256="a" * 64,
    )
    healthy_p0 = SimpleNamespace(plant_state_sha256="b" * 64)
    population = SimpleNamespace(fingerprint="population")
    origin = SimpleNamespace(
        source_kind="hwa_master",
        dims=(784, 256, 10),
        assignment_seed=1,
        endpoint_seed=2,
        teacher_sha256="c" * 64,
        source_artifact_sha256="e" * 64,
        healthy_population=population,
        published_population=population,
        healthy_p0=healthy_p0,
        current=current,
    )
    selection = {
        "source": "literal_diagnostic_grid",
        "checkpoint_policy": (
            "best_held_apparent_validation_accuracy_then_objective_"
            "then_earlier_epoch_including_epoch0"
        ),
        "objective": "teacher_kl",
    }
    generator_state = torch.Generator().get_state()
    recovery = {
        "schema": "ebl.ibm_om_crossbar_adam_diagnostic_recovery",
        "schema_version": 2,
        "origin_device_state_sha256": "d" * 64,
        "start_state": "hwa_healthy_p0",
        "completed_epochs": 0,
        "total_epochs": 3,
        "learning_rate": 1e-5,
        "pulse_cap_per_cell": 640,
        "objective": "teacher_kl",
        "selection_metric": "validation.apparent_forward.student_accuracy",
        "selection_objective_metric": (
            "validation.apparent_forward.kl_teacher_student"
        ),
        "checkpoint_policy": (
            "best_held_apparent_validation_accuracy_then_objective_"
            "then_earlier_epoch_including_epoch0"
        ),
        "optimizer_state_dict": {},
        "train_generator_state": generator_state,
        "epoch_reports": [],
        "initial": {
            "validation": {
                "apparent_forward": {
                    "student_accuracy": 0.9,
                    "kl_teacher_student": 0.1,
                }
            }
        },
        "selection": selection,
        "best_epoch": 0,
        "best_value": {"student_accuracy": 0.9, "objective_value": 0.1},
        "best_current_state": {"token": "origin"},
        "best_optimizer_state_dict": {},
        "best_train_generator_state": generator_state,
    }
    resume = SimpleNamespace(
        role="adam_final",
        source_kind=origin.source_kind,
        dims=origin.dims,
        assignment_seed=origin.assignment_seed,
        endpoint_seed=origin.endpoint_seed,
        teacher_sha256=origin.teacher_sha256,
        source_artifact_sha256=origin.source_artifact_sha256,
        parent_device_state_sha256="d" * 64,
        healthy_population=population,
        published_population=population,
        healthy_p0=healthy_p0,
        current=current,
        recovery=recovery,
    )
    monkeypatch.setattr(
        runtime.IbmOmCrossbarStateBundle,
        "from_state_dict",
        staticmethod(lambda _value: current),
    )

    validated, best = _validate_diagnostic_resume(
        resume,
        origin=origin,
        origin_sha="d" * 64,
        start_state="hwa_healthy_p0",
        learning_rate=1e-5,
        pulse_cap=640,
        total_epochs=3,
        objective="teacher_kl",
        selection_metric="validation.apparent_forward.student_accuracy",
        selection_objective_metric=(
            "validation.apparent_forward.kl_teacher_student"
        ),
        checkpoint_policy=(
            "best_held_apparent_validation_accuracy_then_objective_"
            "then_earlier_epoch_including_epoch0"
        ),
        selection=selection,
    )

    assert validated["completed_epochs"] == 0
    assert best is current

    recovery["best_value"]["objective_value"] = 0.2
    with pytest.raises(ValueError, match="selection to replay"):
        _validate_diagnostic_resume(
            resume,
            origin=origin,
            origin_sha="d" * 64,
            start_state="hwa_healthy_p0",
            learning_rate=1e-5,
            pulse_cap=640,
            total_epochs=3,
            objective="teacher_kl",
            selection_metric="validation.apparent_forward.student_accuracy",
            selection_objective_metric=(
                "validation.apparent_forward.kl_teacher_student"
            ),
            checkpoint_policy=(
                "best_held_apparent_validation_accuracy_then_objective_"
                "then_earlier_epoch_including_epoch0"
            ),
            selection=selection,
        )

    recovery["completed_epochs"] = 1
    recovery["epoch_reports"] = [
        {
            "epoch": 1,
            "validation": {
                "apparent_forward": {
                    "student_accuracy": 0.91,
                    "kl_teacher_student": 0.09,
                }
            },
            "plant_state_sha256": current.plant_state_sha256,
            "apparent_sha256": tensor_sha256(apparent),
            "persistent_sha256": tensor_sha256(persistent),
        }
    ]
    recovery["best_epoch"] = 1
    recovery["best_value"] = {
        "student_accuracy": 0.91,
        "objective_value": 0.09,
    }
    validated, _ = _validate_diagnostic_resume(
        resume,
        origin=origin,
        origin_sha="d" * 64,
        start_state="hwa_healthy_p0",
        learning_rate=1e-5,
        pulse_cap=640,
        total_epochs=3,
        objective="teacher_kl",
        selection_metric="validation.apparent_forward.student_accuracy",
        selection_objective_metric=(
            "validation.apparent_forward.kl_teacher_student"
        ),
        checkpoint_policy=(
            "best_held_apparent_validation_accuracy_then_objective_"
            "then_earlier_epoch_including_epoch0"
        ),
        selection=selection,
    )
    assert validated["best_epoch"] == 1

    recovery["epoch_reports"][0]["plant_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="terminal and embedded best states"):
        _validate_diagnostic_resume(
            resume,
            origin=origin,
            origin_sha="d" * 64,
            start_state="hwa_healthy_p0",
            learning_rate=1e-5,
            pulse_cap=640,
            total_epochs=3,
            objective="teacher_kl",
            selection_metric="validation.apparent_forward.student_accuracy",
            selection_objective_metric=(
                "validation.apparent_forward.kl_teacher_student"
            ),
            checkpoint_policy=(
                "best_held_apparent_validation_accuracy_then_objective_"
                "then_earlier_epoch_including_epoch0"
            ),
            selection=selection,
        )


def test_fresh_apparent_diagnostic_is_replayable_and_does_not_mutate_plant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = FreshApparentDiagnosticStageSettings(
        kind="fresh_apparent_diagnostic",
        start_state="hwa_healthy_p0",
        intervention=(
            "counterfactual_post_write_apparent_noise_redraw_without_device_write"
        ),
        draws=4,
        relative_scale=1.0,
        seed=2090701,
        mutation_policy="do_not_mutate_held_apparent_or_persistent_state",
        interpretation="diagnostic_only_not_physical_inference_read_noise",
    )
    spec = SimpleNamespace(
        stage=stage,
        runtime=SimpleNamespace(seed=42),
    )
    plant = SimpleNamespace(
        apparent=torch.tensor([0.2, -0.1], dtype=torch.float32),
        persistent=torch.tensor([0.1, -0.2], dtype=torch.float32),
    )
    population = SimpleNamespace(nominal_dw_min=0.1, write_noise_std=0.2)
    origin = SimpleNamespace(
        role="healthy_p0",
        source_kind="hwa_master",
        assignment_seed=11,
        endpoint_seed=12,
        healthy_population=population,
        current=SimpleNamespace(digital_scales=(1.0, 1.0)),
    )
    before_apparent = plant.apparent.clone()
    before_persistent = plant.persistent.clone()

    class Store:
        run_dir = tmp_path

        def __init__(self) -> None:
            self.streamed: list[dict] = []

        def append_metric(self, value: dict) -> None:
            self.streamed.append(value)

        def artifact_record(self, path: Path, *, kind: str) -> dict:
            return {"path": str(path), "kind": kind}

    store = Store()
    monkeypatch.setattr(runtime, "sha256_file", lambda _path: "d" * 64)
    monkeypatch.setattr(runtime, "load_device_state", lambda _path: origin)
    monkeypatch.setattr(runtime, "_validate_origin", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_restore_current", lambda *args, **kwargs: plant)
    monkeypatch.setattr(
        runtime,
        "_evaluate_plant_splits",
        lambda **kwargs: {
            "validation": {
                "apparent_forward": {"student_accuracy": 0.97},
                "persistent_diagnostic": {"student_accuracy": 0.91},
            }
        },
    )

    def evaluate_fresh(*, state: torch.Tensor, **_kwargs) -> dict:
        mean = float(state.mean().item())
        return {
            "validation": {
                "student_accuracy": 0.9 + mean * 0.01,
                "kl_teacher_student": 0.1 - mean * 0.01,
                "cross_entropy": 0.2 - mean * 0.01,
            }
        }

    monkeypatch.setattr(runtime, "_evaluate_effective_splits", evaluate_fresh)
    monkeypatch.setattr(runtime, "_base_metrics", lambda **kwargs: {"base": True})

    metrics, artifacts = _run_fresh_apparent_diagnostic(
        request=SimpleNamespace(device_state=tmp_path / "origin.pt"),
        store=store,
        spec=spec,
        teacher=SimpleNamespace(),
        teacher_sha256="c" * 64,
        teacher_validation={},
        data=SimpleNamespace(),
        layout=(),
        device=torch.device("cpu"),
    )

    assert len(metrics["draws"]) == 4
    assert len({item["fresh_apparent_q_sha256"] for item in metrics["draws"]}) == 4
    assert metrics["mutation_check"]["passed"] is True
    assert metrics["physical_read_claim"] is False
    assert metrics["matched_across_start_states"] is True
    healthy_seed, healthy_receipt = _fresh_apparent_diagnostic_seed(
        configured_seed=stage.seed,
        runtime_seed=spec.runtime.seed,
        assignment_seed=origin.assignment_seed,
        endpoint_seed=origin.endpoint_seed,
        start_state="hwa_healthy_p0",
        intervention=stage.intervention,
    )
    faulted_seed, faulted_receipt = _fresh_apparent_diagnostic_seed(
        configured_seed=stage.seed,
        runtime_seed=spec.runtime.seed,
        assignment_seed=origin.assignment_seed,
        endpoint_seed=origin.endpoint_seed,
        start_state="hwa_published_fault",
        intervention=stage.intervention,
    )
    assert healthy_seed == faulted_seed == metrics["resolved_seed"]
    assert healthy_receipt == faulted_receipt == metrics["seed_derivation"]
    assert metrics["seed_derivation"]["excluded_pairing_dimension"] == "start_state"
    assert torch.equal(plant.apparent, before_apparent)
    assert torch.equal(plant.persistent, before_persistent)
    assert len(store.streamed) == 4
    assert artifacts == [
        {
            "path": str(
                tmp_path / "artifacts" / "fresh_apparent_diagnostic_report.json"
            ),
            "kind": "crossbar_fresh_apparent_diagnostic_report",
        }
    ]
    payload = json.loads(
        (tmp_path / "artifacts" / "fresh_apparent_diagnostic_report.json").read_text()
    )
    assert payload["mutation_check"]["before"] == payload["mutation_check"]["after"]
