from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import RunMode, resolve_experiment_config
from experiments.definitions import parse_experiment_config
from experiments.mnist_relu_drn.components import (
    BoundedDrnTeacher,
    build_student_stack,
)
from experiments.mnist_relu_drn.runtime import (
    _build_weight_modifier,
    _literal_bounded_drn_initialization,
    _load_teacher,
    _selection_evaluate,
)
from experiments.schema import ConfigError
from experiments.small_network.components import build_model_stack, seed_runtime
from experiments.small_network.provenance import selected_model_metadata
from training.checkpoint import save_named_weights


ROOT = Path(__file__).resolve().parents[1]
STUDENT_CONFIG = (
    ROOT
    / "examples"
    / "mnist_relu_drn"
    / "wan_cmo_bounded_drn_teacher"
    / "ideal_literal_copy.json"
)
TEACHER_CONFIG = (
    ROOT
    / "examples"
    / "small_drn"
    / "mnist_bounded_memristor_teacher_10ep.json"
)


def _student_spec():
    spec = resolve_experiment_config(STUDENT_CONFIG, RunMode.TRAIN)[1]
    return replace(spec, runtime=replace(spec.runtime, device="cpu"))


def _write_teacher_checkpoint(path: Path, *, mutate_metadata=None) -> None:
    spec = resolve_experiment_config(TEACHER_CONFIG, RunMode.TRAIN)[1]
    common = replace(
        spec.common,
        runtime=replace(spec.common.runtime, device="cpu"),
    )
    seed_runtime(common.runtime.seed)
    stack = build_model_stack(common)
    metadata = selected_model_metadata(
        spec,
        SimpleNamespace(stack=stack),
    )
    if mutate_metadata is not None:
        mutate_metadata(metadata)
    save_named_weights(path, stack.bundle.catalog, metadata=metadata)


def test_bounded_teacher_example_resolves_literal_protocol() -> None:
    train = resolve_experiment_config(STUDENT_CONFIG, RunMode.TRAIN)[1]
    validate = resolve_experiment_config(STUDENT_CONFIG, RunMode.VALIDATE)[1]

    assert train.teacher.type == "bounded_drn"
    assert train.teacher.initialization == "literal_named_weight_copy"
    assert train.model.encoding == "single"
    assert train.model.conductance_min == pytest.approx(9.0 / 88.199997)
    assert train.model.conductance_max == 1.0
    assert train.mapping.scale_fractions == (1.0,)
    assert train.mapping.scale_fraction_pairs is None
    assert train.settings.num_epochs == 0
    assert validate.teacher == train.teacher


def test_bounded_teacher_hwa_selects_fixed_noisy_validation_average() -> None:
    path = STUDENT_CONFIG.with_name("hwa_add_normal_10ep.json")
    train = resolve_experiment_config(path, RunMode.TRAIN)[1]
    assert train.settings.num_epochs == 10
    assert train.settings.selection_evaluation == "modifier"
    assert train.settings.selection_noise_repeats == 4
    assert train.settings.weight_modifier.type == "add_normal"
    assert train.settings.weight_modifier.parameters["noisy_evaluation"] is True


def test_legacy_relu_config_resolves_explicit_compatibility_default() -> None:
    path = (
        ROOT
        / "examples"
        / "mnist_relu_drn"
        / "wan_cmo_teacher_initialized"
        / "ideal_teacher_map.json"
    )
    spec = resolve_experiment_config(path, RunMode.TRAIN)[1]
    assert spec.teacher.type == "bias_free_relu"
    assert spec.teacher.initialization == "signed_weight_mapping"


def test_literal_named_copy_has_exact_weight_and_logit_parity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "teacher.pt"
    _write_teacher_checkpoint(checkpoint)
    spec = _student_spec()

    teacher, metadata = _load_teacher(
        checkpoint,
        device=torch.device("cpu"),
        spec=spec,
    )
    assert isinstance(teacher, BoundedDrnTeacher)
    assert metadata["checkpoint_role"] == "supervised_drn"
    student = build_student_stack(spec, enable_measured=False)
    generator = torch.Generator().manual_seed(23)
    inputs = torch.randn(8, 784, generator=generator)
    labels = torch.arange(8) % 10
    report = _literal_bounded_drn_initialization(
        student,
        teacher,
        ((inputs, labels),),
        spec,
    )

    calibration = report["selected"]["calibration"]
    assert calibration["gain"] == 1.0
    assert calibration["raw_kl"] == 0.0
    assert calibration["max_abs_logit_difference"] == 0.0
    for binding in student.bundle.catalog.trainable:
        torch.testing.assert_close(
            binding.state,
            teacher.stack.bundle.catalog.by_key[binding.key].state,
            rtol=0.0,
            atol=0.0,
        )


def test_modifier_selection_reuses_same_noise_sequence_and_restores_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "teacher.pt"
    _write_teacher_checkpoint(checkpoint)
    path = STUDENT_CONFIG.with_name("hwa_add_normal_10ep.json")
    spec = resolve_experiment_config(path, RunMode.TRAIN)[1]
    spec = replace(spec, runtime=replace(spec.runtime, device="cpu"))
    teacher, _metadata = _load_teacher(
        checkpoint,
        device=torch.device("cpu"),
        spec=spec,
    )
    assert isinstance(teacher, BoundedDrnTeacher)
    student = build_student_stack(spec, enable_measured=False)
    teacher.copy_named_weights_to(student.bundle.catalog)
    modifier = _build_weight_modifier(student, spec)
    assert modifier is not None
    generator = torch.Generator().manual_seed(29)
    inputs = torch.randn(12, 784, generator=generator)
    labels = torch.arange(12) % 10
    loader = ((inputs, labels),)

    first = _selection_evaluate(
        student,
        teacher,
        loader,
        spec=spec,
        modifier=modifier,
    )
    second = _selection_evaluate(
        student,
        teacher,
        loader,
        spec=spec,
        modifier=modifier,
    )

    assert first == second
    assert first["selection_noise_repeats"] == 4
    assert first["kl_teacher_student"] > 0.0
    for binding in student.bundle.catalog.trainable:
        torch.testing.assert_close(
            binding.state,
            teacher.stack.bundle.catalog.by_key[binding.key].state,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda metadata: metadata["model"].__setitem__(
                "weight_bounds", [0.0, 1.0]
            ),
            "model.weight_bounds",
        ),
        (
            lambda metadata: metadata["dataset"].__setitem__(
                "data_seed", 99
            ),
            "dataset.data_seed",
        ),
        (
            lambda metadata: metadata["model"].__setitem__(
                "amplification_indices", []
            ),
            "model.amplification_indices",
        ),
    ),
)
def test_bounded_teacher_metadata_mismatch_fails_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    checkpoint = tmp_path / "teacher.pt"
    _write_teacher_checkpoint(checkpoint, mutate_metadata=mutation)

    with pytest.raises(ValueError, match=message):
        _load_teacher(
            checkpoint,
            device=torch.device("cpu"),
            spec=_student_spec(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["teacher"].__setitem__(
                "initialization", "signed_weight_mapping"
            ),
            "teacher.initialization",
        ),
        (
            lambda payload: payload["model"].__setitem__(
                "encoding", "differential"
            ),
            "model.encoding",
        ),
        (
            lambda payload: payload["mapping"].__setitem__(
                "scale_fractions", [0.5, 1.0]
            ),
            "mapping.scale_fractions",
        ),
        (
            lambda payload: payload["modes"]["train"].update(
                {
                    "selection_evaluation": "modifier",
                    "selection_noise_repeats": 2,
                }
            ),
            "selection_evaluation",
        ),
    ),
)
def test_bounded_teacher_config_constraints_fail_closed(
    mutation,
    message: str,
) -> None:
    payload = json.loads(STUDENT_CONFIG.read_text())
    mutation(payload)

    with pytest.raises(ConfigError, match=message):
        parse_experiment_config(deepcopy(payload))
