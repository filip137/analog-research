from __future__ import annotations

import json
from pathlib import Path

import pytest

from ebl.registry import (
    ConfigError,
    EXPERIMENT_REGISTRY,
    RunMode,
    resolve_experiment_config,
    to_plain_data,
)
from ebl.schema.small_network import (
    LinspaceSpec,
    TrainSpec,
    ValidateSpec,
    parse_small_drn_config,
    resolve_small_drn_spec,
)


def _config() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "small_drn.v1",
        "runtime": {
            "seed": 7,
            "data_seed": 11,
            "device": "cpu",
            "dtype": "float32",
        },
        "data": {
            "dataset": "moons",
            "batch_size": 4,
            "num_points": 100,
            "shuffle": True,
        },
        "model": {
            "dims": [4, 8, 2],
            "input_gain": 10.0,
            "weight_gains": [1.0, 0.5],
            "weight_min": 1e-7,
            "weight_max": 1.0,
            "voltage_amp": 1.0,
            "current_amp": 1.0,
            "non_linearity": {
                "type": "hard_sigmoid",
                "quadratic_diode_param": {
                    "diode_conductance": 10.0,
                    "v_min": -1.0,
                    "v_max": 1.0,
                },
                "exponential_diode_param": {
                    "I_s": 1e-6,
                    "V_t": 0.05,
                    "V_off": 1.0,
                },
                "hard_sigmoid_param": {
                    "g_on": 10.0,
                    "g_off": 0.1,
                    "v_min": -1.2,
                    "v_max": 1.2,
                },
                "iv_data_path": None,
            },
            "adapter": {"type": "none", "parameters": {}},
        },
        "solver": {
            "inference_iterations": 4,
            "training_iterations": 4,
            "minimizer_impl": "custom",
            "minimizer_mode": "asynchronous",
            "adaptive_equilibrium": False,
            "random_initialization": False,
            "updaters": {
                "double_diode": None,
                "single_diode": None,
            },
            "tolerances": {
                "relative": 1e-5,
                "voltage": 1e-6,
                "residual_current": None,
            },
            "polish": {
                "enabled": True,
                "dynamic": False,
                "max_newton_iterations": 32,
                "z_threshold": 1e10,
                "exponential_clip": 1e5,
            },
            "anderson": {
                "memory": 8,
                "omega": 1.0,
                "tolerance_floor": 5e-3,
                "regularization": 1e-8,
            },
            "overrelaxation": {
                "factor": 1.0,
                "reject_steps": False,
                "reject_max_tries": 3,
                "reject_shrink": 0.5,
                "reject_epsilon": 0.0,
            },
            "experimental_exponential": {
                "damping": 0.5,
                "newton_max_steps": 100,
                "progressive_tolerance": True,
                "tolerance_start": 1e-5,
                "tolerance_end": 1e-5,
                "tolerance_switch_high": 1e-2,
                "tolerance_switch_low": 5e-4,
            },
        },
        "modes": {
            "train": {
                "num_epochs": 2,
                "algorithm": "ep",
                "learning_rates": [0.01, 0.02],
                "bias_learning_rates": [0.01],
                "nudging": 0.05,
                "log_every": 1,
                "max_batches": None,
                "max_validation_batches": None,
                "weight_modifier": {"type": "none", "parameters": {}},
                "update_backend": {"type": "direct", "parameters": {}},
            },
            "linspace": {
                "minimum": -1.0,
                "maximum": 1.0,
                "samples": 21,
                "record_states": True,
            },
            "validate": {
                "split": "test",
                "sample_limit": 32,
                "record_states": True,
            },
        },
    }


def _enable_passive_low_rank(payload: dict) -> dict:
    payload["model"].update(
        {
            "dims": [4, 2],
            "weight_gains": [1.0],
            "voltage_amp": 1.0,
            "current_amp": 1.0,
            "adapter": {
                "type": "passive_low_rank",
                "parameters": {
                    "rank": 2,
                    "input_factor_gain": 0.01,
                    "input_factor_min": 1e-7,
                    "conductance_max": 1,
                    "output_factor_init": "zero",
                },
            },
        }
    )
    payload["modes"]["train"].update(
        {
            "learning_rates": [0.01, 0.02],
            "bias_learning_rates": [],
        }
    )
    return payload


def _enable_digital_low_rank(payload: dict) -> dict:
    payload["model"].update(
        {
            "dims": [4, 4],
            "weight_gains": [1.0],
            "adapter": {
                "type": "digital_low_rank",
                "parameters": {
                    "rank": 2,
                    "alpha": 2.0,
                    "input_factor_gain": 0.1,
                    "device_noise": {
                        "type": "aihwkit_reram_wan2022",
                        "programming_seed": 19,
                        "g_max_us": 40.0,
                        "drn_conductance_at_g_max": 0.1,
                        "noise_scale": 1.0,
                        "t_inference_seconds": 1.0,
                    },
                },
            },
        }
    )
    payload["modes"]["train"].update(
        {
            "algorithm": "digital",
            "learning_rates": [0.01, 0.02],
            "bias_learning_rates": [],
            "nudging": 0.0,
        }
    )
    return payload


def _enable_passive_layerwise_low_rank(payload: dict) -> dict:
    def layer(seed: int) -> dict:
        return {
            "rank": 2,
            "input_factor_gain": 0.1,
            "input_factor_min": 1e-7,
            "conductance_max": 1.0,
            "output_factor_init": "zero",
            "device_noise": {
                "type": "aihwkit_reram_wan2022",
                "programming_seed": seed,
                "g_max_us": 40.0,
                "drn_conductance_at_g_max": 0.1,
                "noise_scale": 1.0,
                "t_inference_seconds": 1.0,
            },
        }

    payload["model"]["adapter"] = {
        "type": "passive_layerwise_low_rank",
        "parameters": {
            "layers": {
                "base.dense_weight.0": layer(19),
                "base.dense_weight.1": layer(23),
            }
        },
    }
    payload["modes"]["train"].update(
        {
            "algorithm": "ep",
            "learning_rates": [0.01, 0.02, 0.03, 0.04],
            "bias_learning_rates": [],
            "nudging": 0.05,
        }
    )
    return payload


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_registry_has_stable_config_selected_id() -> None:
    assert "small_drn.v1" in EXPERIMENT_REGISTRY
    document = parse_small_drn_config(_config())
    assert document.experiment_id == "small_drn.v1"


def test_digital_low_rank_config_is_strict_normalized_and_registered() -> None:
    payload = _enable_digital_low_rank(_config())
    document = parse_small_drn_config(payload)
    adapter = document.common.model.adapter
    assert adapter.type == "digital_low_rank"
    assert adapter.parameters["rank"] == 2
    assert adapter.parameters["device_noise"]["g_max_us"] == 40.0

    definition = EXPERIMENT_REGISTRY["small_drn.v1"]
    spec = definition.resolve(document, RunMode.TRAIN)
    assert spec.extensions.model_adapter == "digital_low_rank"
    assert spec.extensions.algorithm == "digital"
    assert spec.extensions.update_backend == "direct"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda payload: payload["model"]["adapter"]["parameters"].update(
                {"unknown": 1}
            ),
            "exactly the keys",
        ),
        (
            lambda payload: payload["model"]["adapter"]["parameters"][
                "device_noise"
            ].update({"t_inference_seconds": 2.0}),
            "one of",
        ),
        (
            lambda payload: payload["model"]["adapter"]["parameters"][
                "device_noise"
            ].update({"drn_conductance_at_g_max": 2.0}),
            "no greater than",
        ),
        (
            lambda payload: payload["modes"]["train"].update(
                {"nudging": 0.1}
            ),
            "equal 0.0",
        ),
    ],
)
def test_digital_low_rank_rejects_invalid_nested_settings(
    mutation,
    match,
) -> None:
    payload = _enable_digital_low_rank(_config())
    mutation(payload)
    with pytest.raises(ConfigError, match=match):
        parse_small_drn_config(payload)


def test_digital_low_rank_combination_fails_closed() -> None:
    payload = _enable_digital_low_rank(_config())
    payload["modes"]["train"]["algorithm"] = "ep"
    payload["modes"]["train"]["nudging"] = 0.05
    document = parse_small_drn_config(payload)
    definition = EXPERIMENT_REGISTRY["small_drn.v1"]
    with pytest.raises(ConfigError, match="listed explicitly"):
        definition.resolve(document, RunMode.TRAIN)


def test_passive_layerwise_low_rank_is_strict_normalized_and_registered() -> None:
    payload = _enable_passive_layerwise_low_rank(_config())
    document = parse_small_drn_config(payload)
    adapter = document.common.model.adapter

    assert adapter.type == "passive_layerwise_low_rank"
    assert tuple(adapter.parameters["layers"]) == (
        "base.dense_weight.0",
        "base.dense_weight.1",
    )
    assert (
        adapter.parameters["layers"]["base.dense_weight.1"]["device_noise"][
            "programming_seed"
        ]
        == 23
    )
    assert (
        adapter.parameters["layers"]["base.dense_weight.0"][
            "input_factor_min"
        ]
        == 1e-7
    )
    definition = EXPERIMENT_REGISTRY["small_drn.v1"]
    spec = definition.resolve(document, RunMode.TRAIN)
    assert spec.extensions.model_adapter == "passive_layerwise_low_rank"
    assert spec.extensions.algorithm == "ep"
    assert spec.settings.learning_rates == (0.01, 0.02, 0.03, 0.04)
    assert spec.settings.bias_learning_rates == ()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda payload: payload["model"]["adapter"]["parameters"][
                "layers"
            ].pop("base.dense_weight.1"),
            "exactly the keys",
        ),
        (
            lambda payload: payload["model"]["adapter"]["parameters"][
                "layers"
            ]["base.dense_weight.0"].update({"unknown": 1}),
            "keys to be drawn",
        ),
        (
            lambda payload: payload["model"].update(
                {"dims": [4, 6, 8, 2], "weight_gains": [1.0, 1.0, 1.0]}
            ),
            "exactly",
        ),
        (
            lambda payload: payload["modes"]["train"].update(
                {"learning_rates": [0.1, 0.1]}
            ),
            "exactly 4 values",
        ),
    ],
)
def test_passive_layerwise_low_rank_rejects_invalid_settings(
    mutation,
    match,
) -> None:
    payload = _enable_passive_layerwise_low_rank(_config())
    mutation(payload)
    with pytest.raises(ConfigError, match=match) as raised:
        parse_small_drn_config(payload)

    assert "Expected" in str(raised.value)
    assert "Provided value:" in str(raised.value)


def test_passive_layerwise_low_rank_combination_fails_closed() -> None:
    payload = _enable_passive_layerwise_low_rank(_config())
    payload["modes"]["train"].update(
        {"algorithm": "digital", "nudging": 0.0}
    )
    document = parse_small_drn_config(payload)
    definition = EXPERIMENT_REGISTRY["small_drn.v1"]
    with pytest.raises(ConfigError, match="listed explicitly"):
        definition.resolve(document, RunMode.TRAIN)


def test_resolves_immutable_mode_specific_specs(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _config())

    _, train = resolve_experiment_config(path, RunMode.TRAIN)
    _, linspace = resolve_experiment_config(path, RunMode.LINSPACE)
    _, validate = resolve_experiment_config(path, RunMode.VALIDATE)

    assert isinstance(train, TrainSpec)
    assert isinstance(linspace, LinspaceSpec)
    assert isinstance(validate, ValidateSpec)
    assert train.settings.learning_rates == (0.01, 0.02)
    assert linspace.settings.samples == 21
    assert validate.settings.split == "test"
    with pytest.raises(TypeError):
        train.common.model.non_linearity.hard_sigmoid_param["g_on"] = 3.0
    with pytest.raises(AttributeError):
        train.settings.num_epochs = 3

    # Frozen specs still have an explicit route into manifests/artifacts.
    json.dumps(to_plain_data(train), allow_nan=False)


def test_selected_mode_must_be_present() -> None:
    payload = _config()
    del payload["modes"]["linspace"]
    document = parse_small_drn_config(payload)
    with pytest.raises(ConfigError, match="Expected config.modes.linspace"):
        resolve_small_drn_spec(document, RunMode.LINSPACE)


def test_diode_parameter_objects_are_explicit_and_expected_is_first() -> None:
    payload = _config()
    del payload["model"]["non_linearity"]["hard_sigmoid_param"]

    with pytest.raises(ConfigError) as raised:
        parse_small_drn_config(payload)

    message = str(raised.value)
    assert message.startswith(
        "Expected config.model.non_linearity to define the required keys"
    )
    assert "Provided value:" in message
    assert message.index("Expected") < message.index("Provided value:")


def test_unknown_nested_scientific_setting_is_rejected() -> None:
    payload = _config()
    payload["runtime"]["epochs"] = 99
    with pytest.raises(ConfigError, match="contain only the keys") as raised:
        parse_small_drn_config(payload)
    assert "epochs" in str(raised.value)


def test_dataset_shape_and_training_rates_fail_during_pure_validation() -> None:
    payload = _config()
    payload["data"]["dataset"] = "digits"
    with pytest.raises(ConfigError, match=r"dims\[0\].*equal 128"):
        parse_small_drn_config(payload)

    payload = _config()
    payload["modes"]["train"]["bias_learning_rates"] = []
    with pytest.raises(ConfigError, match="one per hidden-layer bias"):
        parse_small_drn_config(payload)

    payload = _config()
    payload["modes"]["train"]["nudging"] = 0.0
    with pytest.raises(ConfigError, match="non-zero"):
        parse_small_drn_config(payload)


def test_solver_constraints_fail_before_numerical_construction() -> None:
    payload = _config()
    payload["solver"]["minimizer_mode"] = "sideways"
    with pytest.raises(ConfigError, match="'forward'.*'asynchronous'"):
        parse_small_drn_config(payload)

    payload = _config()
    payload["solver"]["overrelaxation"]["reject_shrink"] = 1.0
    with pytest.raises(ConfigError, match="smaller than 1.0"):
        parse_small_drn_config(payload)

    payload = _config()
    experimental = payload["solver"]["experimental_exponential"]
    experimental["tolerance_switch_high"] = 1e-4
    experimental["tolerance_switch_low"] = 1e-3
    with pytest.raises(ConfigError, match="tolerance_switch_high >"):
        parse_small_drn_config(payload)


def test_add_normal_parameters_are_strict_and_normalized() -> None:
    payload = _config()
    payload["modes"]["train"]["weight_modifier"] = {
        "type": "add_normal",
        "parameters": {"std_dev": 0.06},
    }

    document = parse_small_drn_config(payload)

    assert dict(document.train.weight_modifier.parameters) == {
        "std_dev": 0.06,
        "seed": None,
        "noisy_evaluation": False,
        "scale_mode": "tensor_abs_max",
    }
    with pytest.raises(TypeError):
        document.train.weight_modifier.parameters["std_dev"] = 0.2


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"std": 0.06},
        {"std_dev": 0.06, "unknown": 1},
        {"std_dev": -0.01},
        {"std_dev": float("nan")},
        {"std_dev": True},
        {"std_dev": 0.06, "seed": -1},
        {"std_dev": 0.06, "seed": 2**64},
        {"std_dev": 0.06, "seed": True},
        {"std_dev": 0.06, "noisy_evaluation": 1},
        {"std_dev": 0.06, "scale_mode": "per_tensor"},
    ],
)
def test_add_normal_rejects_invalid_parameters(parameters: dict) -> None:
    payload = _config()
    payload["modes"]["train"]["weight_modifier"] = {
        "type": "add_normal",
        "parameters": parameters,
    }

    with pytest.raises(ConfigError) as exc_info:
        parse_small_drn_config(payload)

    assert str(exc_info.value).startswith("Expected")
    assert "Provided value:" in str(exc_info.value)


def test_hardware_aware_backprop_direct_combination_is_advertised(
    tmp_path: Path,
) -> None:
    payload = _config()
    payload["modes"]["train"]["algorithm"] = "backprop"
    payload["modes"]["train"]["weight_modifier"] = {
        "type": "add_normal",
        "parameters": {
            "std_dev": 0.01,
            "seed": 13,
            "noisy_evaluation": True,
            "scale_mode": "output_channel_abs_max",
        },
    }
    payload["modes"]["train"]["update_backend"] = {
        "type": "direct",
        "parameters": {},
    }

    definition, spec = resolve_experiment_config(
        _write_config(tmp_path, payload),
        RunMode.TRAIN,
    )

    assert spec.extensions.weight_modifier == "add_normal"
    assert spec.extensions.update_backend == "direct"
    assert spec.extensions.algorithm == "backprop"
    assert any(
        combination.selection == spec.extensions
        for combination in definition.combinations
    )


@pytest.mark.parametrize(
    ("backend", "algorithm"),
    (("direct", "ep"), ("tiki_taka", "ep"), ("tiki_taka", "backprop")),
)
def test_unadvertised_hardware_aware_combinations_fail_closed(
    tmp_path: Path,
    backend: str,
    algorithm: str,
) -> None:
    payload = _config()
    payload["modes"]["train"]["algorithm"] = algorithm
    payload["modes"]["train"]["weight_modifier"] = {
        "type": "add_normal",
        "parameters": {"std_dev": 0.01},
    }
    payload["modes"]["train"]["update_backend"] = {
        "type": backend,
        "parameters": {},
    }

    with pytest.raises(ConfigError, match="listed explicitly"):
        resolve_experiment_config(
            _write_config(tmp_path, payload),
            RunMode.TRAIN,
        )


def test_unlisted_extension_combination_fails_closed(tmp_path: Path) -> None:
    payload = _enable_passive_low_rank(_config())
    payload["modes"]["train"]["weight_modifier"] = {
        "type": "add_normal",
        "parameters": {"std_dev": 0.01},
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(
        ConfigError,
        match="listed explicitly by the 'small_drn.v1' definition",
    ):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_feature_combination_unavailable_in_this_worktree_fails_closed(
    tmp_path: Path,
) -> None:
    payload = _enable_digital_low_rank(_config())
    payload["modes"]["train"]["weight_modifier"] = {
        "type": "add_normal",
        "parameters": {"std_dev": 0.01},
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(
        ConfigError,
        match="listed explicitly by the 'small_drn.v1' definition",
    ):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_lora_hardware_aware_tiki_triple_is_rejected(
    tmp_path: Path,
) -> None:
    payload = _enable_passive_low_rank(_config())
    payload["modes"]["train"]["weight_modifier"] = {
        "type": "add_normal",
        "parameters": {"std_dev": 0.01},
    }
    payload["modes"]["train"]["update_backend"] = {
        "type": "tiki_taka",
        "parameters": {"implementation": "ideal"},
    }
    path = _write_config(tmp_path, payload)

    with pytest.raises(
        ConfigError,
        match="listed explicitly by the 'small_drn.v1' definition",
    ):
        resolve_experiment_config(path, RunMode.TRAIN)


def test_passive_low_rank_config_is_strict_normalized_and_frozen() -> None:
    payload = _enable_passive_low_rank(_config())
    document = parse_small_drn_config(payload)
    adapter = document.common.model.adapter

    assert adapter.type == "passive_low_rank"
    assert dict(adapter.parameters) == {
        "rank": 2,
        "input_factor_gain": 0.01,
        "input_factor_min": 1e-7,
        "conductance_max": 1.0,
        "output_factor_init": "zero",
        "output_off_conductance": None,
    }
    assert isinstance(adapter.parameters["conductance_max"], float)
    with pytest.raises(TypeError):
        adapter.parameters["rank"] = 3


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["model"]["adapter"]["parameters"].update(
                {"mystery": 1}
            ),
            "unknown keys",
        ),
        (
            lambda payload: payload["model"]["adapter"]["parameters"].update(
                {"rank": 0}
            ),
            "positive integer",
        ),
        (
            lambda payload: payload["model"]["adapter"]["parameters"].update(
                {"input_factor_min": 1.0}
            ),
            "strictly less",
        ),
        (
            lambda payload: payload["model"]["adapter"]["parameters"].update(
                {"output_factor_init": "off_conductance"}
            ),
            "positive finite number",
        ),
    ],
)
def test_passive_low_rank_parameter_errors_report_expected_and_provided(
    mutate,
    message,
) -> None:
    payload = _enable_passive_low_rank(_config())
    mutate(payload)

    with pytest.raises(ConfigError, match=message) as raised:
        parse_small_drn_config(payload)

    assert "Expected" in str(raised.value)
    assert "Provided value:" in str(raised.value)


def test_passive_low_rank_topology_amplification_and_rates_are_strict() -> None:
    payload = _enable_passive_low_rank(_config())
    payload["model"]["dims"] = [4, 3, 2]
    payload["model"]["weight_gains"] = [1.0, 1.0]
    with pytest.raises(ConfigError, match="exactly"):
        parse_small_drn_config(payload)

    payload = _enable_passive_low_rank(_config())
    payload["model"]["current_amp"] = 2.0
    with pytest.raises(ConfigError, match="both equal 1.0"):
        parse_small_drn_config(payload)

    payload = _enable_passive_low_rank(_config())
    payload["modes"]["train"]["learning_rates"] = [0.01]
    with pytest.raises(ConfigError, match="exactly 2 values"):
        parse_small_drn_config(payload)

    payload = _enable_passive_low_rank(_config())
    payload["modes"]["train"]["bias_learning_rates"] = [0.01]
    with pytest.raises(ConfigError, match="exactly 0 values"):
        parse_small_drn_config(payload)


def test_passive_low_rank_capabilities_are_direct_or_ideal_tiki_ep_only(
    tmp_path: Path,
) -> None:
    direct = _enable_passive_low_rank(_config())
    _, direct_spec = resolve_experiment_config(
        _write_config(tmp_path, direct),
        RunMode.TRAIN,
    )
    assert direct_spec.extensions.update_backend == "direct"
    assert direct_spec.extensions.algorithm == "ep"

    tiki = _enable_passive_low_rank(_config())
    tiki["modes"]["train"]["update_backend"] = {
        "type": "tiki_taka",
        "parameters": {},
    }
    _, tiki_spec = resolve_experiment_config(
        _write_config(tmp_path, tiki),
        RunMode.TRAIN,
    )
    assert tiki_spec.extensions.update_backend == "tiki_taka"

    aihwkit = _enable_passive_low_rank(_config())
    aihwkit["modes"]["train"]["update_backend"] = {
        "type": "tiki_taka",
        "parameters": {"aihwkit_preset": "TikiTakaIdealizedPreset"},
    }
    with pytest.raises(ConfigError, match="ideal-tensor"):
        resolve_experiment_config(
            _write_config(tmp_path, aihwkit),
            RunMode.TRAIN,
        )

    backprop = _enable_passive_low_rank(_config())
    backprop["modes"]["train"]["algorithm"] = "backprop"
    with pytest.raises(ConfigError, match="listed explicitly"):
        resolve_experiment_config(
            _write_config(tmp_path, backprop),
            RunMode.TRAIN,
        )


def test_removed_algorithm_and_backend_names_are_rejected() -> None:
    payload = _config()
    payload["modes"]["train"]["algorithm"] = "bptt"
    with pytest.raises(ConfigError, match="'ep', 'backprop'"):
        parse_small_drn_config(payload)

    payload = _config()
    payload["modes"]["train"]["update_backend"]["type"] = "direct_reram"
    with pytest.raises(
        ConfigError,
        match=(
            "'direct', 'direct_adam', 'ibm_om_fp32_bounds', 'tiki_taka'"
        ),
    ):
        parse_small_drn_config(payload)


def test_accumulated_programming_deadband_is_cohort_b_only() -> None:
    example = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "small_drn"
        / "measured_cohort_b_threshold_mnist.json"
    )
    payload = json.loads(example.read_text(encoding="utf-8"))
    parse_small_drn_config(payload)

    payload["modes"]["train"]["update_backend"]["type"] = (
        "measured_cohort_a"
    )
    payload["modes"]["train"]["update_backend"]["parameters"]["cohort"] = "A"
    with pytest.raises(ConfigError, match="select measured_cohort_b"):
        parse_small_drn_config(payload)


def test_none_programming_deadband_requires_zero_relative_threshold() -> None:
    example = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "small_drn"
        / "measured_cohort_b_threshold_mnist.json"
    )
    payload = json.loads(example.read_text(encoding="utf-8"))
    parameters = payload["modes"]["train"]["update_backend"]["parameters"]
    parameters["programming_deadband_mode"] = "none"
    with pytest.raises(ConfigError, match="equal 0.0"):
        parse_small_drn_config(payload)


def test_probabilistic_write_policies_are_strict_and_mutually_exclusive() -> None:
    example = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "small_drn"
        / "measured_cohort_b_probabilistic_mnist.json"
    )
    payload = json.loads(example.read_text(encoding="utf-8"))
    parse_small_drn_config(payload)

    parameters = payload["modes"]["train"]["update_backend"]["parameters"]
    parameters["probabilistic_write_mode"] = "displacement_proportional"
    parameters["probabilistic_write_probability"] = 1.0
    parameters["probabilistic_write_scale_relative"] = 0.0005
    parsed = parse_small_drn_config(payload)
    assert (
        parsed.train.update_backend.parameters["probabilistic_write_mode"]
        == "displacement_proportional"
    )

    parameters["programming_deadband_mode"] = (
        "accumulated_shadow_relative_rms"
    )
    parameters["programming_deadband_relative"] = 0.001
    with pytest.raises(ConfigError, match="without a deterministic"):
        parse_small_drn_config(payload)
