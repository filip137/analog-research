from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from experiments.small_network.components import build_model_stack, seed_runtime
from experiments.small_network.config import parse_small_drn_config
from model.resistive.minimizer import QuadraticMinimizer
from training.core.optimizers import SGDOptimizer, build_optimizer
from training.core.sgd import AugmentedFunction, EquilibriumProp


_ROOT = Path(__file__).parents[1]
_ORACLE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "small_network_legacy_oracle.json"
)


def _load_oracle() -> dict[str, Any]:
    return json.loads(_ORACLE_PATH.read_text(encoding="utf-8"))


def _build_stack(oracle: dict[str, Any]):
    scenario = oracle["scenario"]
    payload = json.loads(
        (_ROOT / "examples" / "small_drn" / "base.json").read_text(
            encoding="utf-8"
        )
    )
    payload["runtime"] = {
        "seed": oracle["provenance"]["seed"],
        "data_seed": oracle["provenance"]["seed"],
        "device": oracle["provenance"]["device"],
        "dtype": oracle["provenance"]["dtype"],
    }
    payload["data"] = {
        "dataset": scenario["dataset"],
        "batch_size": len(scenario["inputs"]),
        "num_points": len(scenario["inputs"]),
        "shuffle": False,
    }
    payload["model"] = {
        "dims": [
            2 * scenario["logical_input_dim"],
            *scenario["hidden_dims"],
            scenario["output_dim"],
        ],
        "input_gain": scenario["input_gain"],
        "weight_gains": scenario["weight_gains"],
        "weight_min": scenario["weight_min"],
        "weight_max": scenario["weight_max"],
        "voltage_amp": scenario["voltage_amp"],
        "current_amp": scenario["current_amp"],
        "non_linearity": {
            "type": scenario["non_linearity"],
            "quadratic_diode_param": scenario["quadratic_diode_param"],
            "exponential_diode_param": scenario["exponential_diode_param"],
            "hard_sigmoid_param": scenario["hard_sigmoid_param"],
            "iv_data_path": None,
        },
        "adapter": {"type": "none", "parameters": {}},
    }
    document = parse_small_drn_config(payload)
    seed_runtime(oracle["provenance"]["seed"])
    return build_model_stack(document.common)


def _inputs(oracle: dict[str, Any], stack) -> torch.Tensor:
    return torch.tensor(
        oracle["scenario"]["inputs"],
        dtype=stack.dtype,
        device=stack.device,
    )


def _labels(oracle: dict[str, Any], stack) -> torch.Tensor:
    return torch.tensor(
        oracle["scenario"]["labels"],
        dtype=torch.long,
        device=stack.device,
    )


def _minimizer(oracle: dict[str, Any], function, free_layers):
    scenario = oracle["scenario"]
    settings = scenario["minimizer"]
    return QuadraticMinimizer(
        fn=function,
        free_layers=list(free_layers),
        num_iterations=settings["num_iterations"],
        mode=settings["mode"],
        non_linearity=scenario["non_linearity"],
        quadratic_diode_param=scenario["quadratic_diode_param"],
        exponential_diode_param=scenario["exponential_diode_param"],
        hard_sigmoid_param=scenario["hard_sigmoid_param"],
        voltage_amp=scenario["voltage_amp"],
        current_amp=scenario["current_amp"],
    )


def _settle_free_phase(oracle: dict[str, Any], stack) -> None:
    stack.network.set_input(_inputs(oracle, stack), reset=True)
    _minimizer(
        oracle,
        stack.bundle.energy,
        stack.free_layers,
    ).compute_equilibrium()


def _assert_tensor_record(
    actual: torch.Tensor,
    expected: dict[str, Any],
    oracle: dict[str, Any],
) -> None:
    dtype = getattr(torch, expected["dtype"])
    assert actual.dtype == dtype
    assert list(actual.shape) == expected["shape"]
    expected_tensor = torch.tensor(
        expected["values"],
        dtype=dtype,
        device=actual.device,
    )
    comparison = oracle["provenance"]["comparison"]
    torch.testing.assert_close(
        actual,
        expected_tensor,
        rtol=comparison["rtol"],
        atol=comparison["atol"],
    )


def _centered_ep_gradients(oracle: dict[str, Any], stack):
    _settle_free_phase(oracle, stack)
    stack.cost_fn.set_target(_labels(oracle, stack))
    augmented = AugmentedFunction(stack.bundle.energy, stack.cost_fn)
    estimator = EquilibriumProp(
        stack.bundle.energy.params(),
        list(stack.free_layers),
        augmented,
        stack.cost_fn,
        _minimizer(oracle, augmented, stack.free_layers),
        variant=oracle["scenario"]["estimator"]["variant"],
        nudging=oracle["scenario"]["estimator"]["nudging"],
    )
    return estimator.compute_gradient()


def test_frozen_oracle_identifies_the_legacy_source() -> None:
    oracle = _load_oracle()

    assert oracle["oracle_schema_version"] == 1
    assert oracle["provenance"]["implementation"] == (
        "labs.small_network_core._build_energy_stack"
    )
    assert oracle["provenance"]["legacy_source_commit"] == (
        "d640f61a4320bdda1d5a7a52bd4601ee1bd2df78"
    )
    assert oracle["provenance"]["seed"] == 20260727


def test_composed_model_initialization_matches_frozen_legacy_oracle() -> None:
    oracle = _load_oracle()
    stack = _build_stack(oracle)
    expected = oracle["expected"]
    bindings = stack.bundle.catalog.all

    assert [list(layer.shape) for layer in stack.bundle.energy.layers()] == (
        expected["layer_shapes"]
    )
    assert [binding.key for binding in bindings] == expected["parameter_keys"]
    assert stack.bundle.catalog.all_parameters == tuple(
        stack.bundle.energy._all_params
    )
    assert stack.bundle.catalog.trainable_parameters == tuple(
        stack.bundle.energy.params()
    )
    assert len(bindings) == len(expected["initial_parameters"])
    for binding, record in zip(
        bindings,
        expected["initial_parameters"],
        strict=True,
    ):
        assert type(binding.parameter).__name__ == record["type"]
        _assert_tensor_record(binding.parameter.state, record, oracle)


def test_composed_model_free_settling_matches_frozen_legacy_oracle() -> None:
    oracle = _load_oracle()
    stack = _build_stack(oracle)
    _settle_free_phase(oracle, stack)
    expected_layers = oracle["expected"]["settled_layers"]
    layers = stack.bundle.energy.layers()

    assert len(layers) == len(expected_layers)
    for layer, record in zip(layers, expected_layers, strict=True):
        _assert_tensor_record(layer.state, record, oracle)
    assert float(layers[-1].state.abs().sum()) > 0.0


def test_repeated_model_build_keeps_logical_amplifier_depth() -> None:
    """A selector model must not renumber the production circuit equations."""

    oracle = _load_oracle()
    first = _build_stack(oracle)
    second = _build_stack(oracle)

    _settle_free_phase(oracle, first)
    _settle_free_phase(oracle, second)

    for first_layer, second_layer in zip(
        first.bundle.energy.layers(),
        second.bundle.energy.layers(),
        strict=True,
    ):
        torch.testing.assert_close(first_layer.state, second_layer.state)
    first.cost_fn.set_target(_labels(oracle, first))
    second.cost_fn.set_target(_labels(oracle, second))
    torch.testing.assert_close(first.cost_fn.eval(), second.cost_fn.eval())


def test_composed_centered_ep_gradients_match_frozen_legacy_oracle() -> None:
    oracle = _load_oracle()
    stack = _build_stack(oracle)
    gradients = _centered_ep_gradients(oracle, stack)
    expected_gradients = oracle["expected"]["centered_ep_gradients"]

    assert len(gradients) == len(expected_gradients)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in gradients)
    for gradient, record in zip(gradients, expected_gradients, strict=True):
        _assert_tensor_record(gradient, record, oracle)


def test_composed_direct_update_matches_frozen_legacy_oracle() -> None:
    oracle = _load_oracle()
    stack = _build_stack(oracle)
    gradients = _centered_ep_gradients(oracle, stack)
    parameters = tuple(stack.bundle.energy.params())
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.state.grad = gradient

    optimizer_settings = oracle["scenario"]["optimizer"]
    optimizer = build_optimizer(
        stack.bundle.energy,
        stack.cost_fn,
        [optimizer_settings["learning_rate"]] * len(parameters),
        update_pipeline={"type": optimizer_settings["type"]},
        momentum=optimizer_settings["momentum"],
        weight_decay=optimizer_settings["weight_decay"],
    )
    assert isinstance(optimizer, SGDOptimizer)
    optimizer.step()
    for parameter in parameters:
        parameter.clamp_()

    expected_parameters = oracle["expected"]["direct_update_parameters"]
    assert len(parameters) == len(expected_parameters)
    for parameter, record in zip(
        parameters,
        expected_parameters,
        strict=True,
    ):
        _assert_tensor_record(parameter.state, record, oracle)
