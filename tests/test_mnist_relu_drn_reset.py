from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ebl.registry import get_definition, parse_experiment_config
from ebl.runtime.mnist_relu_drn_reset.components import (
    PairedSupervision,
    build_reset_student_stack,
)
from ebl.runtime.mnist_relu_drn_reset.learning_rate_selection import (
    RuntimeSnapshot,
)
from ebl.runtime.mnist_relu_drn_reset.runtime import (
    _validate_checkpoint_metadata,
)
from ebl.schema.base import ConfigError, RunMode
from ebl.cli import main
from networks.variable.layer import LinearLayer
from methods.measured_trace import MeasuredTraceConfig


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    ROOT / "examples/mnist_relu_drn_reset/cohort_a_kl.json",
    ROOT / "examples/mnist_relu_drn_reset/cohort_a_cross_entropy.json",
)


def _payload(path: Path = EXAMPLES[0]) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("path", "objective"),
    ((EXAMPLES[0], "teacher_kl"), (EXAMPLES[1], "cross_entropy")),
)
def test_reset_examples_resolve_both_modes(path: Path, objective: str) -> None:
    definition, document = parse_experiment_config(_payload(path))
    train = definition.resolve(document, RunMode.TRAIN)
    validate = definition.resolve(document, RunMode.VALIDATE)

    assert definition.experiment_id == "mnist_relu_drn_reset.v1"
    assert train.settings.objective == objective
    assert validate.settings.objective == objective
    assert train.model.encoding == "single"
    assert train.model.include_biases is False
    assert train.model.voltage_amp == 4.0
    assert train.model.current_amp == 0.25
    assert train.settings.logit_gain.value == 1.0
    assert train.settings.initialization.type == "measured_reset"
    assert train.settings.learning_rates == (0.0, 0.0)
    assert train.settings.learning_rate_selection.type == (
        "bounded_relative_update_grid"
    )


def test_reset_definition_lists_exactly_two_closed_combinations() -> None:
    definition = get_definition("mnist_relu_drn_reset.v1")
    assert {
        combination.selection.algorithm for combination in definition.combinations
    } == {"teacher_kl", "cross_entropy"}
    assert {
        combination.selection.model_adapter
        for combination in definition.combinations
    } == {"single"}
    assert {
        combination.selection.update_backend
        for combination in definition.combinations
    } == {"measured_cohort_a"}


def test_reset_describe_requires_teacher_and_device_inputs() -> None:
    output = io.StringIO()
    assert main(
        [
            "describe",
            "--experiment",
            "mnist_relu_drn_reset.v1",
            "--json",
        ],
        stdout=output,
    ) == 0
    payload = json.loads(output.getvalue())
    assert payload["commands"]["train"]["required_options"] == [
        "--config",
        "--output-dir",
        "--device-data",
        "--teacher-weights",
    ]
    assert payload["commands"]["train"]["exclusive_input_options"] == [
        "--resume"
    ]
    assert "--teacher-weights" in payload["commands"]["validate"][
        "required_options"
    ]
    assert payload["capabilities"]["resume"]["weights"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload["model"].__setitem__("encoding", "differential"), "one device per physical edge"),
        (lambda payload: payload["model"].__setitem__("include_biases", True), "include_biases"),
        (lambda payload: payload["modes"]["train"]["logit_gain"].__setitem__("value", 2.0), "to equal 1.0"),
        (lambda payload: payload["modes"]["train"].__setitem__("objective", "paired_mse"), "teacher_kl.*cross_entropy"),
        (lambda payload: payload["modes"]["train"]["update_backend"].__setitem__("type", "ideal"), "measured_cohort_a"),
        (lambda payload: payload["modes"]["train"]["initialization"].__setitem__("type", "teacher_mapping"), "measured_reset"),
        (lambda payload: payload["modes"]["train"]["update_backend"]["parameters"].__setitem__("initial_target_mapping", "literal"), "contain only keys"),
    ),
)
def test_reset_schema_rejects_unsupported_semantics(mutation, message: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ConfigError, match=message):
        parse_experiment_config(payload)


def test_train_and_validate_objectives_must_match() -> None:
    payload = _payload()
    payload["modes"]["validate"]["objective"] = "cross_entropy"
    with pytest.raises(ConfigError, match="to match"):
        parse_experiment_config(payload)


def test_reset_backend_normalizes_only_nonwriting_defaults() -> None:
    definition, document = parse_experiment_config(_payload())
    spec = definition.resolve(document, RunMode.TRAIN)
    parameters = spec.settings.update_backend.parameters
    assert "initial_target_mapping" not in parameters
    normalized = MeasuredTraceConfig.from_mapping(parameters)
    assert normalized.initial_target_mapping == "literal"
    assert normalized.programming_deadband_mode == "none"
    assert normalized.probabilistic_write_mode == "none"
    assert normalized.initial_pulse_index == 0


def _cost_inputs():
    output = LinearLayer((6,), batch_size=2, device="cpu")
    output.state = torch.tensor(
        [
            [0.3, -0.2, 0.1, 0.0, -0.4, 0.2],
            [0.2, 0.1, -0.1, 0.5, 0.4, -0.3],
        ],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[1.0, -0.5, 0.2], [-0.3, 0.7, 0.1]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1])
    return output, teacher, labels


def test_reset_kl_matches_pytorch_and_ignores_labels() -> None:
    output, teacher, labels = _cost_inputs()
    cost = PairedSupervision(output, objective="teacher_kl")
    cost.set_batch(teacher, labels)
    student = output.state[:, 0::2] - output.state[:, 1::2]
    expected = F.kl_div(
        F.log_softmax(student, dim=1),
        F.softmax(teacher, dim=1),
        reduction="none",
    ).sum(dim=1)
    actual = cost.eval()
    torch.testing.assert_close(actual, expected)
    first_gradient = torch.autograd.grad(actual.mean(), output.state, retain_graph=True)[0]

    cost.set_batch(teacher, labels.flip(0))
    swapped = cost.eval()
    second_gradient = torch.autograd.grad(swapped.mean(), output.state)[0]
    torch.testing.assert_close(swapped, actual)
    torch.testing.assert_close(second_gradient, first_gradient)
    assert teacher.grad is None
    assert cost.gain == 1.0


def test_reset_cross_entropy_matches_pytorch_and_uses_labels() -> None:
    output, teacher, labels = _cost_inputs()
    cost = PairedSupervision(output, objective="cross_entropy")
    cost.set_batch(teacher, labels)
    student = output.state[:, 0::2] - output.state[:, 1::2]
    torch.testing.assert_close(
        cost.eval(), F.cross_entropy(student, labels, reduction="none")
    )
    original = cost.eval().detach()
    cost.set_batch(teacher, labels.flip(0))
    assert not torch.equal(cost.eval().detach(), original)


def test_reset_validation_stack_has_two_single_device_weights() -> None:
    payload = _payload()
    payload["runtime"]["device"] = "cpu"
    definition, document = parse_experiment_config(payload)
    spec = definition.resolve(document, RunMode.VALIDATE)
    stack = build_reset_student_stack(
        spec, device_data_path=None, enable_measured=False
    )
    assert [binding.key for binding in stack.bundle.catalog] == [
        "base.dense_weight.0",
        "base.dense_weight.1",
    ]
    assert not any(binding.role == "bias" for binding in stack.bundle.catalog)
    assert stack.cost.objective == "teacher_kl"
    assert stack.cost.gain == 1.0


class _FakeOptimizer:
    def __init__(self) -> None:
        self.rate = 0.0
        self.payload = {"counter": 0}

    def learning_rates(self):
        return (self.rate,)

    def state_dict(self):
        return {"rate": self.rate, "payload": deepcopy(self.payload)}

    def load_state_dict(self, state):
        self.rate = float(state["rate"])
        self.payload = deepcopy(state["payload"])


def test_lr_snapshot_restores_parameters_layers_optimizer_rng_and_loader() -> None:
    binding = SimpleNamespace(
        key="base.dense_weight.0",
        state=torch.tensor([1.0, 2.0]),
    )
    layer = SimpleNamespace(state=torch.tensor([[3.0]]))
    optimizer = _FakeOptimizer()
    stack = SimpleNamespace(
        bundle=SimpleNamespace(
            catalog=SimpleNamespace(trainable=(binding,)),
            energy=SimpleNamespace(layers=lambda: (layer,)),
        ),
        optimizer=optimizer,
    )
    generator = torch.Generator().manual_seed(17)
    snapshot = RuntimeSnapshot.capture(stack, generator)
    expected_random = torch.rand(3)
    expected_loader = torch.randperm(8, generator=generator)

    binding.state.add_(10.0)
    layer.state.add_(20.0)
    optimizer.rate = 0.5
    optimizer.payload["counter"] = 9
    torch.rand(4)
    torch.randperm(8, generator=generator)

    snapshot.restore(stack, generator)
    torch.testing.assert_close(binding.state, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(layer.state, torch.tensor([[3.0]]))
    assert optimizer.rate == 0.0
    assert optimizer.payload == {"counter": 0}
    torch.testing.assert_close(torch.rand(3), expected_random)
    torch.testing.assert_close(
        torch.randperm(8, generator=generator), expected_loader
    )


def test_reset_checkpoint_metadata_is_fail_closed() -> None:
    metadata = {
        "experiment_id": "mnist_relu_drn_reset.v1",
        "encoding": "single",
        "objective": "teacher_kl",
        "initialization": "measured_reset",
        "fixed_logit_gain": 1.0,
        "temperature": 1.0,
        "teacher_sha256": "teacher",
    }
    _validate_checkpoint_metadata(
        metadata, objective="teacher_kl", teacher_sha256="teacher"
    )
    metadata["fixed_logit_gain"] = 0.5
    with pytest.raises(ValueError, match="fixed_logit_gain"):
        _validate_checkpoint_metadata(
            metadata, objective="teacher_kl", teacher_sha256="teacher"
        )
