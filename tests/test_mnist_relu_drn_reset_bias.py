from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from ebl.registry import get_definition, parse_experiment_config
from ebl.runtime.mnist_relu_drn.components import conductance_statistics
from ebl.runtime.mnist_relu_drn_reset.components import (
    PairedSupervision,
    build_reset_student_stack,
)
from ebl.runtime.mnist_relu_drn_reset.learning_rate_selection import (
    _derive_rates,
    _parameter_topology,
)
from ebl.runtime.mnist_relu_drn_reset.runtime import (
    _validate_checkpoint_metadata,
)
from ebl.schema.base import ConfigError, RunMode
from networks.function.cost import SquaredErrorPairedOutputs
from networks.resistive.interaction import DenseResistive
from networks.variable.layer import LinearLayer


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "paired_squared_error": ROOT
    / "examples/mnist_relu_drn_reset_bias/cohort_a_paired_squared_error.json",
    "cross_entropy": ROOT
    / "examples/mnist_relu_drn_reset_bias/cohort_a_cross_entropy.json",
    "teacher_kl": ROOT / "examples/mnist_relu_drn_reset_bias/cohort_a_kl.json",
}
LEGACY_EXAMPLES = {
    "paired_squared_error": ROOT
    / "examples/mnist_relu_drn_reset_bias_legacy/cohort_a_paired_squared_error.json",
    "cross_entropy": ROOT
    / "examples/mnist_relu_drn_reset_bias_legacy/cohort_a_cross_entropy.json",
    "teacher_kl": ROOT
    / "examples/mnist_relu_drn_reset_bias_legacy/cohort_a_kl.json",
}


def _payload(objective: str = "paired_squared_error") -> dict:
    return json.loads(EXAMPLES[objective].read_text(encoding="utf-8"))


def _legacy_payload(objective: str = "paired_squared_error") -> dict:
    return json.loads(
        LEGACY_EXAMPLES[objective].read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("objective", tuple(EXAMPLES))
def test_bias_examples_resolve_both_modes(objective: str) -> None:
    definition, document = parse_experiment_config(_payload(objective))
    train = definition.resolve(document, RunMode.TRAIN)
    validate = definition.resolve(document, RunMode.VALIDATE)

    assert definition.experiment_id == "mnist_relu_drn_reset_bias.v1"
    assert train.settings.objective == objective
    assert validate.settings.objective == objective
    assert train.model.encoding == "single"
    assert train.model.include_biases is True
    assert train.model.voltage_amp == 4.0
    assert train.model.current_amp == 0.25
    assert train.settings.learning_rates == (0.0, 0.0, 0.0)


def test_bias_definition_lists_exactly_three_closed_combinations() -> None:
    definition = get_definition("mnist_relu_drn_reset_bias.v1")
    assert {
        combination.selection.algorithm for combination in definition.combinations
    } == {"paired_squared_error", "cross_entropy", "teacher_kl"}
    assert {
        combination.selection.model_adapter
        for combination in definition.combinations
    } == {"single_bias"}
    assert {
        combination.selection.update_backend
        for combination in definition.combinations
    } == {"measured_cohort_a"}


@pytest.mark.parametrize("objective", tuple(LEGACY_EXAMPLES))
def test_legacy_bias_examples_are_explicit_historical_controls(
    objective: str,
) -> None:
    definition, document = parse_experiment_config(_legacy_payload(objective))
    train = definition.resolve(document, RunMode.TRAIN)
    validate = definition.resolve(document, RunMode.VALIDATE)

    assert definition.experiment_id == "mnist_relu_drn_reset_bias_legacy.v1"
    assert train.settings.objective == objective
    assert validate.settings.objective == objective
    assert train.model.include_biases is True
    assert train.model.amplification_indexing == "legacy_process_global"
    assert {
        combination.selection.model_adapter
        for combination in definition.combinations
    } == {"single_bias_legacy_process_index"}


def test_bias_schema_fails_closed() -> None:
    payload = _payload()
    payload["model"]["include_biases"] = False
    with pytest.raises(ConfigError, match="include_biases.*true"):
        parse_experiment_config(payload)

    payload = _payload()
    payload["modes"]["train"]["objective"] = "paired_mse"
    with pytest.raises(ConfigError, match="paired_squared_error"):
        parse_experiment_config(payload)

    payload = _payload()
    payload["modes"]["train"]["unexpected"] = 1
    with pytest.raises(ConfigError, match="contain only keys"):
        parse_experiment_config(payload)


def test_bias_stack_matches_historical_parameter_topology() -> None:
    payload = _payload()
    payload["runtime"]["device"] = "cpu"
    definition, document = parse_experiment_config(payload)
    spec = definition.resolve(document, RunMode.VALIDATE)
    stack = build_reset_student_stack(
        spec,
        device_data_path=None,
        enable_measured=False,
    )

    assert [binding.key for binding in stack.bundle.catalog.trainable] == [
        "base.dense_weight.0",
        "base.dense_weight.1",
        "base.bias.0",
    ]
    assert len(stack.optimizer.param_groups) == 3
    topology = _parameter_topology(stack)
    assert topology["weight_names"] == (
        "base.dense_weight.0",
        "base.dense_weight.1",
    )
    assert topology["bias_names"] == ("base.bias.0",)
    assert topology["bias_to_weight"] == {
        "base.bias.0": "base.dense_weight.0"
    }
    assert topology["training_reset_input"] is False
    interactions = tuple(
        interaction
        for interaction in stack.bundle.energy._interactions
        if isinstance(interaction, DenseResistive)
    )
    assert [
        (item._logical_pre_index, item._logical_post_index)
        for item in interactions
    ] == [(0, 1), (1, 2)]
    torch.testing.assert_close(
        stack.bundle.catalog.by_key["base.bias.0"].state,
        torch.zeros(100),
    )
    statistics = conductance_statistics(stack.bundle.catalog, encoding="single")
    assert statistics["parameters"]["base.bias.0"]["lower_bound_fraction"] is None
    assert statistics["parameters"]["base.bias.0"]["upper_bound_fraction"] is None


def test_legacy_bias_stack_uses_process_global_layer_indices() -> None:
    payload = _legacy_payload()
    payload["runtime"]["device"] = "cpu"
    definition, document = parse_experiment_config(payload)
    spec = definition.resolve(document, RunMode.VALIDATE)
    stack = build_reset_student_stack(
        spec,
        device_data_path=None,
        enable_measured=False,
    )
    interactions = tuple(
        interaction
        for interaction in stack.bundle.energy._interactions
        if isinstance(interaction, DenseResistive)
    )
    assert len(interactions) == 2
    for interaction in interactions:
        assert interaction._logical_pre_index == int(
            interaction._layer_pre.name.rsplit("_", 1)[-1]
        )
        assert interaction._logical_post_index == int(
            interaction._layer_post.name.rsplit("_", 1)[-1]
        )


def test_legacy_bias_rebuild_advances_amplification_indices() -> None:
    payload = _legacy_payload()
    payload["runtime"]["device"] = "cpu"
    definition, document = parse_experiment_config(payload)
    spec = definition.resolve(document, RunMode.VALIDATE)

    first = build_reset_student_stack(
        spec,
        device_data_path=None,
        enable_measured=False,
    )
    second = build_reset_student_stack(
        spec,
        device_data_path=None,
        enable_measured=False,
    )

    def resolved_pairs(stack) -> list[tuple[int, int]]:
        return [
            (
                interaction._logical_pre_index,
                interaction._logical_post_index,
            )
            for interaction in stack.bundle.energy._interactions
            if isinstance(interaction, DenseResistive)
        ]

    first_pairs = resolved_pairs(first)
    second_pairs = resolved_pairs(second)
    assert len(first_pairs) == len(second_pairs) == 2
    assert second_pairs == [
        (pre_index + 3, post_index + 3)
        for pre_index, post_index in first_pairs
    ]


def test_legacy_bias_schema_requires_explicit_indexing_marker() -> None:
    payload = _legacy_payload()
    del payload["model"]["amplification_indexing"]
    with pytest.raises(ConfigError, match="legacy_process_global"):
        parse_experiment_config(payload)


def test_paired_squared_error_matches_historical_cost_and_gradient() -> None:
    output = LinearLayer((6,), batch_size=2, device="cpu")
    output.state = torch.tensor(
        [
            [0.3, -0.2, 0.1, 0.0, -0.4, 0.2],
            [0.2, 0.1, -0.1, 0.5, 0.4, -0.3],
        ],
        requires_grad=True,
    )
    teacher = torch.tensor([[1.0, -0.5, 0.2], [-0.3, 0.7, 0.1]])
    labels = torch.tensor([0, 1])
    historical = SquaredErrorPairedOutputs(output, 3)
    historical.set_target(labels)
    corrected = PairedSupervision(output, objective="paired_squared_error")
    corrected.set_batch(teacher, labels)

    historical_value = historical.eval()
    corrected_value = corrected.eval()
    torch.testing.assert_close(corrected_value, historical_value)
    historical_gradient = torch.autograd.grad(
        historical_value.mean(), output.state, retain_graph=True
    )[0]
    corrected_gradient = torch.autograd.grad(
        corrected_value.mean(), output.state
    )[0]
    torch.testing.assert_close(corrected_gradient, historical_gradient)


def test_bias_learning_rate_derivation_matches_historical_rule() -> None:
    probe = {
        "parameter_names": [
            "base.dense_weight.0",
            "base.dense_weight.1",
            "base.bias.0",
        ],
        "weight_names": ["base.dense_weight.0", "base.dense_weight.1"],
        "bias_to_weight": {"base.bias.0": "base.dense_weight.0"},
        "normalization_unit_by_weight": {
            "base.dense_weight.0": 10.0,
            "base.dense_weight.1": 20.0,
        },
        "bias_q90_unit_by_parameter": {"base.bias.0": 100.0},
    }
    by_name, vector = _derive_rates(probe, (0.003, 0.01))
    assert vector == pytest.approx((0.0003, 0.0005, 0.00003))
    assert by_name["base.bias.0"] == pytest.approx(0.00003)


def test_bias_checkpoint_metadata_is_fail_closed() -> None:
    metadata = {
        "experiment_id": "mnist_relu_drn_reset_bias.v1",
        "encoding": "single",
        "include_biases": True,
        "objective": "teacher_kl",
        "initialization": "measured_reset",
        "fixed_logit_gain": 1.0,
        "temperature": 1.0,
        "teacher_sha256": "teacher",
    }
    _validate_checkpoint_metadata(
        metadata,
        objective="teacher_kl",
        teacher_sha256="teacher",
        experiment_id="mnist_relu_drn_reset_bias.v1",
        include_biases=True,
    )
    mutated = deepcopy(metadata)
    mutated["include_biases"] = False
    with pytest.raises(ValueError, match="include_biases"):
        _validate_checkpoint_metadata(
            mutated,
            objective="teacher_kl",
            teacher_sha256="teacher",
            experiment_id="mnist_relu_drn_reset_bias.v1",
            include_biases=True,
        )


def test_legacy_bias_checkpoint_requires_indexing_provenance() -> None:
    metadata = {
        "experiment_id": "mnist_relu_drn_reset_bias_legacy.v1",
        "encoding": "single",
        "include_biases": True,
        "amplification_indexing": "legacy_process_global",
        "objective": "paired_squared_error",
        "initialization": "measured_reset",
        "fixed_logit_gain": 1.0,
        "temperature": 1.0,
        "teacher_sha256": "teacher",
    }
    _validate_checkpoint_metadata(
        metadata,
        objective="paired_squared_error",
        teacher_sha256="teacher",
        experiment_id="mnist_relu_drn_reset_bias_legacy.v1",
        include_biases=True,
        amplification_indexing="legacy_process_global",
    )
    mutated = deepcopy(metadata)
    del mutated["amplification_indexing"]
    with pytest.raises(ValueError, match="amplification_indexing"):
        _validate_checkpoint_metadata(
            mutated,
            objective="paired_squared_error",
            teacher_sha256="teacher",
            experiment_id="mnist_relu_drn_reset_bias_legacy.v1",
            include_biases=True,
            amplification_indexing="legacy_process_global",
        )
