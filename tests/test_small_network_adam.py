from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import RunMode, resolve_experiment_config
from experiments.schema import ConfigError
from experiments.small_network.config import parse_small_drn_config
from experiments.small_network.initialization import validate_training_initialization
from model.resistive.builders import ParameterBinding, ParameterCatalog
from model.variable.parameter import DenseWeight
from training.adam import AdamOptimizer
from training.ibm_om_fp32_bounds import (
    IbmOmFp32BoundsOptimizer,
    effective_controller_bounds,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "examples"
    / "small_drn"
    / "mnist_bounded_memristor_teacher_uniform_10ep.json"
)
CONFIG_DIR = (
    ROOT
    / "examples"
    / "small_drn"
    / "mnist_om_range_uniform_optimizer_reference_20260826"
)
SGD_CONFIGS = {
    "sgd_w1_0p128_w2_0p01.json": (0.128, 0.01),
    "sgd_w1_0p128_w2_0p02.json": (0.128, 0.02),
    "sgd_w1_0p256_w2_0p01.json": (0.256, 0.01),
    "sgd_w1_0p256_w2_0p02.json": (0.256, 0.02),
    "sgd_w1_0p512_w2_0p01.json": (0.512, 0.01),
    "sgd_w1_0p512_w2_0p02.json": (0.512, 0.02),
}
ADAM_CONFIGS = {
    "adam_lr_1e4.json": 0.0001,
    "adam_lr_3e4.json": 0.0003,
    "adam_lr_1e3.json": 0.001,
    "adam_lr_3e3.json": 0.003,
}
ADAM_PARAMETERS = {
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
}
POPULATION_SHA256 = (
    "7e0bcfb76a0a613f0ba319786a39fdb540cce0a74f15d9fb007426a46ad49d2f"
)
POPULATION_FINGERPRINT = (
    "4887ad89abdd16193448c54a0cbe97be915cb4b9bc04cf1151c5e2a96ee5a3ac"
)


def _bounds_backend(*, adam: bool) -> dict:
    return {
        "type": "ibm_om_fp32_bounds",
        "parameters": {
            "preset": "reram_array_om",
            "assignment_seed": 84001,
            "corruption_policy": "counterfactual_repaired",
            "bounds_coordinate": "controller_clipped_0_1",
            "initialization_distribution": (
                "uniform_per_cell_effective_bounds"
            ),
            "initialization_seed": 17,
            "expected_population_sha256": POPULATION_SHA256,
            "expected_population_fingerprint": POPULATION_FINGERPRINT,
            "optimizer": {
                "type": "adam" if adam else "sgd",
                "parameters": (
                    dict(ADAM_PARAMETERS)
                    if adam
                    else {"momentum": 0.0, "weight_decay": 0.0}
                ),
            },
        },
    }


class _Parameter:
    def __init__(self, value: torch.Tensor) -> None:
        self.state = value


class _Function:
    def __init__(self, *parameters: _Parameter) -> None:
        self._parameters = list(parameters)

    def params(self):
        return list(self._parameters)


def _expected_payload(*, learning_rates, adam: bool) -> dict:
    payload = json.loads(BASE_CONFIG.read_text())
    payload["model"]["weight_min"] = 0.0
    payload["model"]["weight_max"] = 1.0
    payload["model"]["weight_init_mode"] = "om_cell_bounds_uniform"
    payload["modes"]["train"]["learning_rates"] = list(learning_rates)
    payload["modes"]["train"]["update_backend"] = _bounds_backend(
        adam=adam
    )
    return payload


@pytest.mark.parametrize(
    ("filename", "learning_rates"),
    tuple(SGD_CONFIGS.items()),
)
def test_om_range_sgd_configs_are_isolated(
    filename: str,
    learning_rates: tuple[float, float],
) -> None:
    path = CONFIG_DIR / filename
    assert json.loads(path.read_text()) == _expected_payload(
        learning_rates=learning_rates,
        adam=False,
    )
    spec = resolve_experiment_config(path, RunMode.TRAIN)[1]
    assert spec.extensions.update_backend == "ibm_om_fp32_bounds"
    assert spec.common.model.weight_min == 0.0
    assert spec.common.model.weight_max == 1.0


@pytest.mark.parametrize(
    ("filename", "learning_rate"),
    tuple(ADAM_CONFIGS.items()),
)
def test_om_range_adam_configs_are_isolated(
    filename: str,
    learning_rate: float,
) -> None:
    path = CONFIG_DIR / filename
    assert json.loads(path.read_text()) == _expected_payload(
        learning_rates=(learning_rate, learning_rate),
        adam=True,
    )
    spec = resolve_experiment_config(path, RunMode.TRAIN)[1]
    assert spec.extensions.update_backend == "ibm_om_fp32_bounds"
    optimizer = spec.settings.update_backend.parameters["optimizer"]
    assert optimizer["type"] == "adam"
    assert dict(optimizer["parameters"]) == ADAM_PARAMETERS


def test_om_range_configs_share_exact_external_initialization_contract() -> None:
    paths = [CONFIG_DIR / name for name in (*SGD_CONFIGS, *ADAM_CONFIGS)]
    common = resolve_experiment_config(paths[0], RunMode.TRAIN)[1].common

    for path in paths[1:]:
        candidate = resolve_experiment_config(path, RunMode.TRAIN)[1]
        candidate_common = candidate.common
        assert candidate_common == common
        backend = candidate.settings.update_backend.parameters
        assert backend["assignment_seed"] == 84001
        assert backend["initialization_seed"] == 17
        assert backend["expected_population_sha256"] == POPULATION_SHA256
        assert (
            backend["expected_population_fingerprint"]
            == POPULATION_FINGERPRINT
        )


def test_adam_backend_rejects_invalid_beta() -> None:
    payload = _expected_payload(
        learning_rates=(0.001, 0.001),
        adam=True,
    )
    payload["modes"]["train"]["update_backend"]["parameters"][
        "optimizer"
    ]["parameters"][
        "beta2"
    ] = 1.0

    with pytest.raises(ConfigError, match=r"beta1.*beta2.*\[0, 1\)"):
        parse_small_drn_config(payload)


def test_om_bounds_backend_requires_external_initialization_mode() -> None:
    payload = _expected_payload(
        learning_rates=(0.001, 0.001),
        adam=True,
    )
    payload["model"]["weight_init_mode"] = "bounded_range_uniform"
    with pytest.raises(
        ConfigError,
        match="om_cell_bounds_uniform.*ibm_om_fp32_bounds",
    ):
        parse_small_drn_config(payload)


def test_om_bounds_backend_requires_population_and_fresh_start() -> None:
    spec = resolve_experiment_config(
        CONFIG_DIR / "adam_lr_1e3.json",
        RunMode.TRAIN,
    )[1]
    request = SimpleNamespace(
        weights=None,
        base_weights=None,
        resume=None,
        device_data=None,
    )
    with pytest.raises(ValueError, match="Expected --device-data"):
        validate_training_initialization(request, spec)

    request.device_data = Path("population.npz")
    validate_training_initialization(request, spec)
    request.weights = Path("weights.pt")
    with pytest.raises(ValueError, match="per-cell uniform distribution"):
        validate_training_initialization(request, spec)


def test_adam_adapter_matches_torch_and_resumes_exactly() -> None:
    initial = torch.tensor([0.25, -0.5, 0.75], requires_grad=True)
    ours_value = initial.detach().clone().requires_grad_(True)
    reference_value = initial.detach().clone().requires_grad_(True)
    ours = AdamOptimizer(
        _Function(_Parameter(ours_value)),
        _Function(),
        (0.003,),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    reference = torch.optim.Adam(
        [{"params": reference_value, "lr": 0.003}],
        lr=1.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )

    for gradient in (
        torch.tensor([0.2, -0.4, 0.1]),
        torch.tensor([-0.3, 0.2, 0.5]),
    ):
        ours_value.grad = gradient.clone()
        reference_value.grad = gradient.clone()
        ours.step()
        reference.step()
        torch.testing.assert_close(ours_value, reference_value, rtol=0, atol=0)

    resumed_value = ours_value.detach().clone().requires_grad_(True)
    resumed = AdamOptimizer(
        _Function(_Parameter(resumed_value)),
        _Function(),
        (0.003,),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    resumed.load_state_dict(deepcopy(ours.state_dict()))
    next_gradient = torch.tensor([0.7, -0.1, -0.2])
    ours_value.grad = next_gradient.clone()
    resumed_value.grad = next_gradient.clone()
    ours.step()
    resumed.step()
    torch.testing.assert_close(ours_value, resumed_value, rtol=0, atol=0)


def test_optimizer_reference_validate_config_is_test_only() -> None:
    path = CONFIG_DIR / "selected_test.json"
    payload = json.loads(path.read_text())
    assert tuple(payload["modes"]) == ("validate",)
    spec = resolve_experiment_config(path, RunMode.VALIDATE)[1]
    assert spec.common.model.weight_min == 0.0
    assert spec.common.model.weight_max == 1.0
    assert spec.common.model.weight_init_mode == "om_cell_bounds_uniform"
    assert spec.settings.split == "test"
    assert spec.settings.sample_limit is None


def _synthetic_population() -> IbmReramArrayPopulation:
    minimum = torch.tensor([-1.0, -0.5, 0.0, 0.5, -2.0, -0.2])
    maximum = torch.tensor([1.0, 0.5, 0.8, 0.6, 0.0, 0.2])
    return IbmReramArrayPopulation(
        assignment_seed=84001,
        corruption_policy="counterfactual_repaired",
        binding_keys=("base.dense_weight.0",),
        binding_shapes=((2, 3),),
        binding_sampling_seeds=(1,),
        donor_sampling_seeds=(2,),
        nominal_dw_min=0.0949,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=maximum,
        min_bound=minimum,
        dwmin_up=torch.full((6,), 0.1),
        dwmin_down=torch.full((6,), 0.1),
        reference=torch.zeros(6),
        corrupt=torch.zeros(6, dtype=torch.bool),
        published_corrupt=torch.zeros(6, dtype=torch.bool),
        fingerprint="a" * 64,
        aihwkit_version="1.1.0",
    )


def _synthetic_bounds_parameters() -> dict:
    value = _bounds_backend(adam=False)["parameters"]
    value["expected_population_sha256"] = "b" * 64
    value["expected_population_fingerprint"] = "a" * 64
    value["initialization_seed"] = 123
    return value


def _synthetic_binding() -> ParameterBinding:
    parameter = DenseWeight(
        (2,),
        (3,),
        1.0,
        device=None,
        clamp=True,
        clamp_min=0.0,
        clamp_max=1.0,
        init_mode="om_cell_bounds_uniform",
    )
    return ParameterBinding(
        "base.dense_weight.0",
        parameter,
        role="dense_weight",
    )


def test_effective_controller_bounds_clip_per_cell() -> None:
    lower, upper, clipping = effective_controller_bounds(
        _synthetic_population()
    )
    torch.testing.assert_close(
        lower,
        torch.tensor([0.0, 0.25, 0.5, 0.75, 0.0, 0.4]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        upper,
        torch.tensor([1.0, 0.75, 0.9, 0.8, 0.5, 0.6]),
        rtol=0,
        atol=0,
    )
    assert clipping == {
        "raw_lower_below_zero": 1,
        "raw_lower_above_one": 0,
        "raw_upper_below_zero": 0,
        "raw_upper_above_one": 0,
        "zero_span_after_clipping": 0,
    }


def test_om_bounds_optimizer_initializes_projects_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import training.ibm_om_fp32_bounds as bounds_module

    population_path = tmp_path / "population.npz"
    population_path.write_bytes(b"fixture")
    monkeypatch.setattr(
        bounds_module,
        "load_om_array_population",
        lambda _path: _synthetic_population(),
    )
    monkeypatch.setattr(bounds_module, "_sha256_file", lambda _path: "b" * 64)

    binding = _synthetic_binding()
    catalog = ParameterCatalog((binding,))
    base = torch.optim.SGD([{"params": binding.state, "lr": 1.0}], lr=1.0)
    optimizer = IbmOmFp32BoundsOptimizer(
        base,
        catalog,
        _synthetic_bounds_parameters(),
        population_path,
    )
    lower, upper, _ = effective_controller_bounds(_synthetic_population())
    initialized = binding.state.detach().reshape(-1)
    assert bool(torch.all(initialized >= lower))
    assert bool(torch.all(initialized <= upper))

    binding.state.grad = torch.tensor(
        [[-10.0, 10.0, -10.0], [10.0, -10.0, 10.0]]
    )
    optimizer.step()
    projected = binding.state.detach().reshape(-1)
    torch.testing.assert_close(
        projected,
        torch.tensor([1.0, 0.25, 0.9, 0.75, 0.5, 0.4]),
        rtol=0,
        atol=0,
    )
    report = optimizer.bounds_report
    assert report["projection"]["step_count"] == 1
    assert report["projection"]["lower_events"] == 3
    assert report["projection"]["upper_events"] == 3

    saved_state = deepcopy(optimizer.state_dict())
    saved_weights = binding.state.detach().clone()
    resumed_binding = _synthetic_binding()
    resumed_catalog = ParameterCatalog((resumed_binding,))
    resumed_base = torch.optim.SGD(
        [{"params": resumed_binding.state, "lr": 1.0}],
        lr=1.0,
    )
    resumed = IbmOmFp32BoundsOptimizer(
        resumed_base,
        resumed_catalog,
        _synthetic_bounds_parameters(),
        population_path,
    )
    with torch.no_grad():
        resumed_binding.state.copy_(saved_weights)
    resumed.load_state_dict(saved_state)
    next_gradient = torch.tensor(
        [[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]]
    )
    binding.state.grad = next_gradient.clone()
    resumed_binding.state.grad = next_gradient.clone()
    optimizer.step()
    resumed.step()
    torch.testing.assert_close(
        binding.state,
        resumed_binding.state,
        rtol=0,
        atol=0,
    )
    assert optimizer.bounds_report == resumed.bounds_report
