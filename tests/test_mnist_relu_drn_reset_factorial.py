from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from ebl.registry import get_definition, parse_experiment_config
from ebl.runtime.mnist_relu_drn_reset.components import build_reset_student_stack
from ebl.runtime.mnist_relu_drn_reset.learning_rate_selection import (
    _parameter_topology,
)
from ebl.runtime.mnist_relu_drn_reset.runtime import (
    _requires_production_restart,
    _validate_checkpoint_metadata,
)
from ebl.schema.base import ConfigError, RunMode
from labs.tools.analyze_mnist_relu_drn_reset_factorial import _factorial_effects
from networks.resistive.interaction import DenseResistive


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples/mnist_relu_drn_reset_factorial"
EXAMPLES = tuple(sorted(EXAMPLE_ROOT.glob("*/*/*.json")))


def _payload(path: Path = EXAMPLES[0]) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_pairs(stack) -> list[tuple[int, int]]:
    return [
        (item._logical_pre_index, item._logical_post_index)
        for item in stack.bundle.energy._interactions
        if isinstance(item, DenseResistive)
    ]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: str(path.relative_to(EXAMPLE_ROOT)))
def test_factorial_examples_resolve_both_modes(path: Path) -> None:
    definition, document = parse_experiment_config(_payload(path))
    train = definition.resolve(document, RunMode.TRAIN)
    validate = definition.resolve(document, RunMode.VALIDATE)

    assert definition.experiment_id == "mnist_relu_drn_reset_factorial.v1"
    assert train.settings.objective == validate.settings.objective
    assert train.settings.objective in {"paired_squared_error", "teacher_kl"}
    assert train.settings.reset_input_between_batches is True
    assert train.model.include_biases is (path.parts[-3] == "bias")
    assert train.model.amplification_indexing == path.parts[-2]
    assert train.model.voltage_amp == 4.0
    assert train.model.current_amp == 0.25
    assert train.settings.learning_rates == (
        (0.0, 0.0, 0.0)
        if train.model.include_biases
        else (0.0, 0.0)
    )
    assert _requires_production_restart(train) is True


def test_factorial_definition_lists_exactly_eight_closed_combinations() -> None:
    definition = get_definition("mnist_relu_drn_reset_factorial.v1")
    assert len(definition.combinations) == 8
    assert {
        item.selection.algorithm for item in definition.combinations
    } == {"paired_squared_error", "teacher_kl"}
    assert {
        item.selection.model_adapter for item in definition.combinations
    } == {
        "single_logical",
        "single_legacy_process_index",
        "single_bias_logical",
        "single_bias_legacy_process_index",
    }
    assert {
        item.selection.update_backend for item in definition.combinations
    } == {"measured_cohort_a"}


def test_factorial_schema_requires_all_control_factors_explicitly() -> None:
    payload = _payload()
    payload["model"]["include_biases"] = 1
    with pytest.raises(ConfigError, match="include_biases.*boolean"):
        parse_experiment_config(payload)

    payload = _payload()
    del payload["model"]["amplification_indexing"]
    with pytest.raises(ConfigError, match="explicit key 'amplification_indexing'"):
        parse_experiment_config(payload)

    payload = _payload()
    payload["modes"]["train"]["reset_input_between_batches"] = False
    with pytest.raises(ConfigError, match="reset_input_between_batches.*true"):
        parse_experiment_config(payload)

    payload = _payload()
    payload["modes"]["train"]["objective"] = "cross_entropy"
    with pytest.raises(ConfigError, match="paired_squared_error"):
        parse_experiment_config(payload)


def test_factorial_arms_share_every_non_factor_protocol_setting() -> None:
    normalized = []
    for path in EXAMPLES:
        payload = _payload(path)
        payload["model"].pop("include_biases")
        payload["model"].pop("amplification_indexing")
        payload["modes"]["train"].pop("objective")
        payload["modes"]["validate"].pop("objective")
        normalized.append(payload)
    assert normalized[1:] == normalized[:-1]


@pytest.mark.parametrize("include_biases", (False, True))
def test_logical_factorial_rebuild_is_model_local(include_biases: bool) -> None:
    path = EXAMPLE_ROOT / (
        "bias" if include_biases else "bias_free"
    ) / "logical/teacher_kl.json"
    payload = _payload(path)
    payload["runtime"]["device"] = "cpu"
    definition, document = parse_experiment_config(payload)
    spec = definition.resolve(document, RunMode.TRAIN)

    first = build_reset_student_stack(
        spec, device_data_path=None, enable_measured=False
    )
    second = build_reset_student_stack(
        spec, device_data_path=None, enable_measured=False
    )

    assert _resolved_pairs(first) == [(0, 1), (1, 2)]
    assert _resolved_pairs(second) == [(0, 1), (1, 2)]
    assert _parameter_topology(
        second,
        training_reset_input=spec.settings.reset_input_between_batches,
    )["training_reset_input"] is True


@pytest.mark.parametrize("include_biases", (False, True))
def test_legacy_factorial_rebuild_advances_exactly_one_model(
    include_biases: bool,
) -> None:
    path = EXAMPLE_ROOT / (
        "bias" if include_biases else "bias_free"
    ) / "legacy_process_global/teacher_kl.json"
    payload = _payload(path)
    payload["runtime"]["device"] = "cpu"
    definition, document = parse_experiment_config(payload)
    spec = definition.resolve(document, RunMode.TRAIN)

    first = build_reset_student_stack(
        spec, device_data_path=None, enable_measured=False
    )
    second = build_reset_student_stack(
        spec, device_data_path=None, enable_measured=False
    )

    assert _resolved_pairs(second) == [
        (pre + 3, post + 3) for pre, post in _resolved_pairs(first)
    ]
    assert _parameter_topology(
        second,
        training_reset_input=spec.settings.reset_input_between_batches,
    )["training_reset_input"] is True


def test_factorial_checkpoint_metadata_is_fail_closed_for_all_factors() -> None:
    metadata = {
        "experiment_id": "mnist_relu_drn_reset_factorial.v1",
        "encoding": "single",
        "include_biases": False,
        "amplification_indexing": "logical",
        "objective": "teacher_kl",
        "initialization": "measured_reset",
        "fixed_logit_gain": 1.0,
        "temperature": 1.0,
        "reset_input_between_batches": True,
        "teacher_sha256": "teacher",
    }
    arguments = {
        "objective": "teacher_kl",
        "teacher_sha256": "teacher",
        "experiment_id": "mnist_relu_drn_reset_factorial.v1",
        "include_biases": False,
        "amplification_indexing": "logical",
    }
    _validate_checkpoint_metadata(metadata, **arguments)

    for key, value in (
        ("include_biases", True),
        ("amplification_indexing", "legacy_process_global"),
        ("reset_input_between_batches", False),
    ):
        mutated = deepcopy(metadata)
        mutated[key] = value
        with pytest.raises(ValueError, match=key):
            _validate_checkpoint_metadata(mutated, **arguments)


def test_factorial_effects_use_standard_difference_of_marginal_means() -> None:
    accuracies = {}
    for include_biases in (False, True):
        for indexing in ("logical", "legacy_process_global"):
            for objective in ("paired_squared_error", "teacher_kl"):
                bias_code = 1 if include_biases else -1
                indexing_code = 1 if indexing == "legacy_process_global" else -1
                loss_code = 1 if objective == "teacher_kl" else -1
                accuracies[(include_biases, indexing, objective)] = (
                    0.8
                    + 0.01 * bias_code
                    + 0.02 * indexing_code
                    - 0.03 * loss_code
                    + 0.005 * bias_code * indexing_code
                )

    effects = _factorial_effects(accuracies)
    assert effects["bias_enabled_minus_disabled_pp"] == pytest.approx(2.0)
    assert effects["legacy_minus_logical_indexing_pp"] == pytest.approx(4.0)
    assert effects["teacher_kl_minus_paired_mse_pp"] == pytest.approx(-6.0)
    assert effects["bias_x_indexing_interaction_pp"] == pytest.approx(1.0)
    assert effects["bias_x_loss_interaction_pp"] == pytest.approx(0.0)
