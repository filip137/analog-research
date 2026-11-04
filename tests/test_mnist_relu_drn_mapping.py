from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ebl.registry import parse_experiment_config
from ebl.runtime.mnist_relu_drn.components import (
    TeacherKLDivergence,
    build_student_stack,
    fit_positive_logit_gain,
    mapped_conductances,
    signed_differential_lift,
    signed_dual_rail_lift,
)
from ebl.schema.base import RunMode
from networks.variable.layer import LinearLayer
from methods.program_verify import ProgramVerifyOptimizer


ROOT = Path(__file__).resolve().parents[1]


def test_dual_rail_lift_preserves_signed_drive_for_both_layouts() -> None:
    weight = torch.tensor([[2.0, -3.0], [-1.5, 0.5]])
    logical_input = torch.tensor([[0.4, -0.2]])
    physical_input = torch.cat((logical_input, -logical_input), dim=1)
    expected = logical_input @ weight
    for layout in ("halves", "paired"):
        lifted = signed_dual_rail_lift(weight, target_layout=layout)
        physical = physical_input @ lifted
        if layout == "halves":
            scores = physical[:, :2] - physical[:, 2:]
        else:
            scores = physical[:, 0::2] - physical[:, 1::2]
        torch.testing.assert_close(scores, 2.0 * expected)


def test_differential_lift_has_expected_effective_signed_matrix() -> None:
    weight = torch.tensor([[1.0, -0.5], [-0.25, 0.75]])
    for layout in ("halves", "paired"):
        effective = signed_differential_lift(weight, target_layout=layout)
        positive = effective.clamp_min(0.0)
        negative = (-effective).clamp_min(0.0)
        torch.testing.assert_close(positive - negative, effective)
        assert bool((positive >= 0.0).all())
        assert bool((negative >= 0.0).all())


def test_centered_single_mapping_uses_symmetric_range_margin() -> None:
    teacher = (
        torch.tensor([[1.0, -0.5], [-0.25, 0.75]]),
        torch.tensor([[1.0, -0.5], [-0.25, 0.75]]),
    )
    targets, report = mapped_conductances(
        teacher,
        encoding="single",
        scale_fractions=(0.5, 0.25),
        conductance_min=70.0e-6,
        conductance_max=90.0e-6,
        range_placement="centered",
    )

    assert float(targets[0].min()) == pytest.approx(75.0e-6)
    assert float(targets[0].max()) == pytest.approx(85.0e-6)
    assert float(targets[1].min()) == pytest.approx(77.5e-6)
    assert float(targets[1].max()) == pytest.approx(82.5e-6)
    assert report["layers"][0]["selected_baseline_s"] == pytest.approx(
        75.0e-6
    )
    assert report["layers"][1]["selected_baseline_s"] == pytest.approx(
        77.5e-6
    )


def test_kl_matches_pytorch_teacher_to_student_direction() -> None:
    output = LinearLayer((6,), batch_size=2, device="cpu")
    output.state = torch.tensor(
        [[0.3, -0.2, 0.1, 0.0, -0.4, 0.2], [0.2, 0.1, -0.1, 0.5, 0.4, -0.3]]
    )
    teacher_logits = torch.tensor([[1.0, -0.5, 0.2], [-0.3, 0.7, 0.1]])
    labels = torch.tensor([0, 1])
    cost = TeacherKLDivergence(output, gain=2.5)
    cost.set_teacher(teacher_logits, labels)
    student_logits = (output.state[:, 0::2] - output.state[:, 1::2]) * 2.5
    expected = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="none",
    ).sum(dim=1)
    torch.testing.assert_close(cost.eval(), expected)


def test_positive_gain_fit_recovers_a_known_scale_and_reduces_kl() -> None:
    torch.manual_seed(2)
    raw = torch.randn(64, 10)
    teacher = raw * 4.0
    result = fit_positive_logit_gain(
        raw,
        teacher,
        gain_min=0.1,
        gain_max=10.0,
        steps=81,
    )
    assert abs(result["gain"] - 4.0) < 0.2
    assert result["calibrated_kl"] < result["raw_kl"]


def test_differential_student_has_four_physical_weights_and_no_biases() -> None:
    payload = json.loads(
        (ROOT / "examples/mnist_relu_drn/ideal_differential.json").read_text()
    )
    payload["runtime"]["device"] = "cpu"
    definition, document = parse_experiment_config(payload)
    spec = definition.resolve(document, RunMode.TRAIN)
    stack = build_student_stack(spec, enable_measured=False)
    assert [binding.key for binding in stack.bundle.catalog] == [
        "base.conductance_plus.0",
        "base.conductance_minus.0",
        "base.conductance_plus.1",
        "base.conductance_minus.1",
    ]
    assert not any(binding.role == "bias" for binding in stack.bundle.catalog)


def test_teacher_mapped_program_verify_build_is_model_local_and_bias_free() -> None:
    payload = json.loads(
        (
            ROOT
            / "examples/mnist_relu_drn/wan_cmo_teacher_initialized/"
            "cmo_full_bptt.json"
        ).read_text()
    )
    payload["runtime"]["device"] = "cpu"
    definition, document = parse_experiment_config(payload)
    spec = definition.resolve(document, RunMode.TRAIN)

    stacks = [
        build_student_stack(spec, enable_measured=True)
        for _ in range(2)
    ]
    signatures = []
    for stack in stacks:
        assert isinstance(stack.optimizer, ProgramVerifyOptimizer)
        assert [binding.key for binding in stack.bundle.catalog] == [
            "base.dense_weight.0",
            "base.dense_weight.1",
        ]
        assert not any(
            binding.role == "bias" for binding in stack.bundle.catalog
        )
        signatures.append(
            tuple(
                    (
                        interaction._logical_pre_index,
                        interaction._logical_post_index,
                        (
                            1.0
                            if interaction._logical_pre_index == 0
                            else float(interaction._voltage_amp)
                        ),
                        (
                            float(interaction._current_amp)
                            / float(interaction._voltage_amp)
                        )
                        ** interaction._logical_pre_index,
                    )
                for interaction in stack.bundle.energy._interactions
                if hasattr(interaction, "_logical_pre_index")
            )
        )
    assert signatures[0] == signatures[1] == (
        (0, 1, 1.0, 1.0),
        (1, 2, 4.0, 0.0625),
    )

    initial = stacks[0].optimizer.initialize_from_loaded_targets()
    assert stacks[0].optimizer.initialized is True
    assert initial["model"] == "aihwkit_reram_cmo"
    assert initial["parameter_keys"] == [
        "base.dense_weight.0",
        "base.dense_weight.1",
    ]
    floor = 0.00011 * 9.0 / 88.199997
    for binding in stacks[0].bundle.catalog.trainable:
        assert float(binding.state.min()) >= floor - 1e-10
