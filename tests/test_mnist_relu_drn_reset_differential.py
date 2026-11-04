from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path

import pytest
import torch

from ebl.execution.schema import load_campaign_manifest
from ebl.cli import main
from ebl.registry import get_definition, parse_experiment_config
from ebl.runtime.mnist_relu_drn_reset.components import build_reset_student_stack
from ebl.runtime.mnist_relu_drn_reset.learning_rate_selection import (
    _derive_rates,
    _parameter_topology,
)
from ebl.runtime.mnist_relu_drn_reset.runtime import (
    _amplification_index_report,
    _requires_production_restart,
    _validate_checkpoint_metadata,
)
from ebl.schema.base import ConfigError, RunMode
from networks.resistive.interaction import SignedDenseResistive


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = (
    ROOT
    / "examples/mnist_relu_drn_reset_differential"
    / "cohort_a_paired_squared_error_10ep.json"
)
MANIFEST = (
    ROOT
    / "campaigns/manifests/mnist_relu_drn_reset_differential_10ep.json"
)
EXPERIMENT_ID = "mnist_relu_drn_reset_differential.v1"


def _payload() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _cpu_spec(mode: RunMode = RunMode.TRAIN):
    payload = _payload()
    payload["runtime"]["device"] = "cpu"
    definition, document = parse_experiment_config(payload)
    return definition.resolve(document, mode)


def _signed_interactions(stack) -> tuple[SignedDenseResistive, ...]:
    return tuple(
        item
        for item in stack.bundle.energy._interactions
        if isinstance(item, SignedDenseResistive)
    )


def test_differential_example_resolves_closed_ten_epoch_protocol() -> None:
    definition, document = parse_experiment_config(_payload())
    train = definition.resolve(document, RunMode.TRAIN)
    validate = definition.resolve(document, RunMode.VALIDATE)

    assert definition.experiment_id == EXPERIMENT_ID
    assert train.model.encoding == "differential"
    assert train.model.include_biases is False
    assert train.model.amplification_indexing == "logical"
    assert train.model.voltage_amp == 4.0
    assert train.model.current_amp == 0.25
    assert train.settings.objective == "paired_squared_error"
    assert validate.settings.objective == train.settings.objective
    assert train.settings.num_epochs == 10
    assert train.settings.reset_input_between_batches is True
    assert train.settings.learning_rates == (0.0, 0.0)
    assert _requires_production_restart(train) is True


def test_differential_definition_lists_only_requested_combination() -> None:
    definition = get_definition(EXPERIMENT_ID)
    assert len(definition.combinations) == 1
    selection = definition.combinations[0].selection
    assert selection.model_adapter == "differential_logical"
    assert selection.weight_modifier == "none"
    assert selection.update_backend == "measured_cohort_a"
    assert selection.algorithm == "paired_squared_error"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["model"].__setitem__("encoding", "single"),
            "G\\+/G- pair",
        ),
        (
            lambda payload: payload["model"].__setitem__("include_biases", True),
            "include_biases",
        ),
        (
            lambda payload: payload["model"].__setitem__("voltage_amp", 3.0),
            "voltage_amp.*equal 4.0",
        ),
        (
            lambda payload: payload["model"].__setitem__("current_amp", 0.5),
            "current_amp.*equal 0.25",
        ),
        (
            lambda payload: payload["modes"]["train"].__setitem__(
                "objective", "teacher_kl"
            ),
            "paired_squared_error",
        ),
        (
            lambda payload: payload["modes"]["train"].__setitem__(
                "reset_input_between_batches", False
            ),
            "reset_input_between_batches.*true",
        ),
    ),
)
def test_differential_schema_rejects_unlisted_semantics(
    mutation, message: str
) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_experiment_config(payload)


def test_differential_repeated_construction_is_model_local() -> None:
    spec = _cpu_spec()
    first = build_reset_student_stack(
        spec, device_data_path=None, enable_measured=False
    )
    second = build_reset_student_stack(
        spec, device_data_path=None, enable_measured=False
    )

    expected = [
        (0, 1, 1.0, 1.0),
        (1, 2, 4.0, 0.0625),
    ]

    def semantics(stack):
        return [
            (
                item._logical_pre_index,
                item._logical_post_index,
                item._forward_gain(),
                item._post_metric(),
            )
            for item in _signed_interactions(stack)
        ]

    assert semantics(first) == expected
    assert semantics(second) == expected
    assert len(_signed_interactions(first)) == 2
    assert len(_signed_interactions(second)) == 2


def test_differential_optimizer_groups_pair_branches_by_logical_edge() -> None:
    stack = build_reset_student_stack(
        _cpu_spec(), device_data_path=None, enable_measured=False
    )
    topology = _parameter_topology(stack, training_reset_input=True)

    assert topology["encoding"] == "differential"
    assert topology["rate_group_names"] == (
        "base.differential_pair.0",
        "base.differential_pair.1",
    )
    assert topology["group_members"] == {
        "base.differential_pair.0": (
            "base.conductance_plus.0",
            "base.conductance_minus.0",
        ),
        "base.differential_pair.1": (
            "base.conductance_plus.1",
            "base.conductance_minus.1",
        ),
    }
    assert topology["training_reset_input"] is True


def test_differential_learning_rate_derivation_returns_two_group_rates() -> None:
    probe = {
        "parameter_names": [
            "base.conductance_plus.0",
            "base.conductance_minus.0",
            "base.conductance_plus.1",
            "base.conductance_minus.1",
        ],
        "rate_group_names": [
            "base.differential_pair.0",
            "base.differential_pair.1",
        ],
        "weight_names": [
            "base.differential_pair.0",
            "base.differential_pair.1",
        ],
        "bias_to_weight": {},
        "normalization_unit_by_weight": {
            "base.differential_pair.0": 10.0,
            "base.differential_pair.1": 20.0,
        },
        "bias_q90_unit_by_parameter": {},
    }

    by_group, rates = _derive_rates(probe, (0.003, 0.01))

    assert rates == pytest.approx((0.0003, 0.0005))
    assert by_group == pytest.approx(
        {
            "base.differential_pair.0": 0.0003,
            "base.differential_pair.1": 0.0005,
        }
    )


def test_differential_training_path_produces_four_finite_branch_gradients() -> None:
    torch.manual_seed(7)
    stack = build_reset_student_stack(
        _cpu_spec(), device_data_path=None, enable_measured=False
    )
    inputs = torch.rand(2, 784)
    labels = torch.tensor([1, 7])
    stack.network.set_input(inputs, reset=True)
    stack.minimizer.compute_equilibrium()
    stack.cost.set_batch(torch.zeros(2, 10), labels)

    gradients = tuple(stack.differentiator.compute_gradient())

    assert len(gradients) == 4
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)


def test_differential_checkpoint_requires_resolved_stage_scales() -> None:
    stack = build_reset_student_stack(
        _cpu_spec(), device_data_path=None, enable_measured=False
    )
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "encoding": "differential",
        "include_biases": False,
        "amplification_indexing": "logical",
        "amplification_indices": _amplification_index_report(stack),
        "objective": "paired_squared_error",
        "initialization": "measured_reset",
        "fixed_logit_gain": 1.0,
        "temperature": 1.0,
        "reset_input_between_batches": True,
        "teacher_sha256": "teacher",
    }
    arguments = {
        "objective": "paired_squared_error",
        "teacher_sha256": "teacher",
        "experiment_id": EXPERIMENT_ID,
        "encoding": "differential",
        "include_biases": False,
        "amplification_indexing": "logical",
    }
    _validate_checkpoint_metadata(metadata, **arguments)

    mutated = deepcopy(metadata)
    mutated["amplification_indices"][1]["post_layer_metric"] = 1.0
    with pytest.raises(ValueError, match="amplification_indices"):
        _validate_checkpoint_metadata(mutated, **arguments)


def test_differential_describe_and_campaign_are_wired() -> None:
    output = io.StringIO()
    assert main(
        ["describe", "--experiment", EXPERIMENT_ID, "--json"],
        stdout=output,
    ) == 0
    description = json.loads(output.getvalue())
    assert description["commands"]["train"]["required_options"] == [
        "--config",
        "--output-dir",
        "--device-data",
        "--teacher-weights",
    ]

    campaign = load_campaign_manifest(MANIFEST)
    assert campaign.campaign_id == (
        "mnist-differential-reram-reset-10ep-20260817-v1"
    )
    assert [stage.stage_id for stage in campaign.stages] == [
        "differential_train",
        "differential_test",
    ]
