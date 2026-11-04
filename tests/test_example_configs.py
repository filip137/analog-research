from pathlib import Path

import pytest

from ebl.registry import RunMode, resolve_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_base_example_resolves_every_documented_mode() -> None:
    path = ROOT / "examples" / "small_drn" / "base.json"

    for mode in (RunMode.TRAIN, RunMode.LINSPACE, RunMode.VALIDATE):
        definition, spec = resolve_experiment_config(path, mode)
        assert definition.experiment_id == "small_drn.v1"
        assert spec.experiment_id == "small_drn.v1"


def test_hardware_aware_example_is_nested_and_resolves_every_mode() -> None:
    path = ROOT / "examples" / "small_drn" / "hardware_aware.json"

    resolved = {}
    for mode in (RunMode.TRAIN, RunMode.LINSPACE, RunMode.VALIDATE):
        definition, spec = resolve_experiment_config(path, mode)
        assert definition.experiment_id == "small_drn.v1"
        resolved[mode] = spec

    train = resolved[RunMode.TRAIN]
    assert train.extensions.weight_modifier == "add_normal"
    assert train.extensions.update_backend == "direct"
    assert train.extensions.algorithm == "backprop"
    assert dict(train.settings.weight_modifier.parameters) == {
        "std_dev": 0.06,
        "seed": 7,
        "noisy_evaluation": True,
        "scale_mode": "tensor_abs_max",
    }


def test_passive_low_rank_example_is_nested_and_resolves_every_mode() -> None:
    path = ROOT / "examples" / "small_drn" / "lora.json"

    resolved = {}
    for mode in (RunMode.TRAIN, RunMode.LINSPACE, RunMode.VALIDATE):
        definition, spec = resolve_experiment_config(path, mode)
        assert definition.experiment_id == "small_drn.v1"
        assert spec.experiment_id == "small_drn.v1"
        resolved[mode] = spec

    common = resolved[RunMode.TRAIN].common
    train = resolved[RunMode.TRAIN].settings
    assert common.runtime.seed == 7
    assert common.runtime.data_seed == 11
    assert common.data.batch_size == 8
    assert common.data.num_points == 32
    assert common.model.dims == (4, 2)
    assert common.model.adapter.type == "passive_low_rank"
    assert dict(common.model.adapter.parameters) == {
        "rank": 2,
        "input_factor_gain": 0.01,
        "input_factor_min": 1e-7,
        "conductance_max": 1.0,
        "output_factor_init": "zero",
        "output_off_conductance": None,
    }
    assert train.learning_rates == (0.01, 0.01)
    assert train.bias_learning_rates == ()
    assert train.weight_modifier.type == "none"
    assert train.update_backend.type == "direct"


def test_digital_low_rank_reram_digits_example_resolves() -> None:
    path = (
        ROOT
        / "examples"
        / "small_drn"
        / "digital_lora_reram_wan2022_digits.json"
    )

    _, train = resolve_experiment_config(path, RunMode.TRAIN)
    _, validate = resolve_experiment_config(path, RunMode.VALIDATE)
    adapter = train.common.model.adapter
    assert adapter.type == "digital_low_rank"
    assert train.common.data.dataset == "digits"
    assert train.common.model.dims == (128, 20)
    assert train.settings.algorithm == "digital"
    assert train.settings.nudging == 0.0
    assert train.settings.learning_rates == (0.01, 0.01)
    assert validate.common.model.adapter == adapter
    assert dict(adapter.parameters["device_noise"]) == {
        "type": "aihwkit_reram_wan2022",
        "programming_seed": 17,
        "g_max_us": 40.0,
        "drn_conductance_at_g_max": 0.1,
        "noise_scale": 1.0,
        "t_inference_seconds": 1.0,
    }


def test_passive_layerwise_low_rank_reram_digits_example_resolves() -> None:
    path = (
        ROOT
        / "examples"
        / "small_drn"
        / "passive_layerwise_lora_reram_wan2022_digits.json"
    )

    _, train = resolve_experiment_config(path, RunMode.TRAIN)
    _, validate = resolve_experiment_config(path, RunMode.VALIDATE)
    adapter = train.common.model.adapter
    assert adapter.type == "passive_layerwise_low_rank"
    assert train.common.data.dataset == "digits"
    assert train.common.model.dims == (128, 64, 20)
    assert train.settings.algorithm == "ep"
    assert train.settings.nudging == 0.05
    assert train.settings.learning_rates == (0.005,) * 4
    assert validate.common.model.adapter == adapter
    assert tuple(adapter.parameters["layers"]) == (
        "base.dense_weight.0",
        "base.dense_weight.1",
    )
    assert [
        adapter.parameters["layers"][key]["conductance_max"]
        for key in adapter.parameters["layers"]
    ] == [1.0, 1.0]
    assert [
        adapter.parameters["layers"][key]["device_noise"][
            "programming_seed"
        ]
        for key in adapter.parameters["layers"]
    ] == [17, 29]


def test_mnist_hwa_reram_full_finetune_example_resolves() -> None:
    path = (
        ROOT
        / "examples"
        / "small_drn"
        / "mnist_hwa_reram_full_finetune.json"
    )

    _, train = resolve_experiment_config(path, RunMode.TRAIN)
    _, validate = resolve_experiment_config(path, RunMode.VALIDATE)

    assert train.common.data.dataset == "mnist"
    assert train.common.model.dims == (1568, 100, 20)
    assert train.common.model.weight_min == 1e-7
    assert train.common.model.weight_max == 1.1
    assert train.common.model.adapter.type == "none"
    assert train.settings.algorithm == "backprop"
    assert train.settings.num_epochs == 20
    assert train.settings.learning_rates == (0.08, 0.05)
    assert train.settings.bias_learning_rates == (0.15,)
    assert train.settings.weight_modifier.type == "none"
    assert train.settings.update_backend.type == "direct"
    assert validate.common.model == train.common.model


def test_mnist_ibm_device_study_examples_resolve() -> None:
    directory = (
        ROOT / "examples" / "small_drn" / "mnist_ibm_devices"
    )
    paths = sorted(directory.glob("*.json"))
    assert [path.name for path in paths] == [
        "cmo_affine_floor_deployment_probe.json",
        "cmo_deployment_probe.json",
        "cmo_full_bptt_recovery.json",
        "cmo_lora_bptt_recovery.json",
        "hwa_from_fp32.json",
        "pcm_deployment_probe.json",
        "pcm_full_bptt_recovery.json",
        "pcm_lora_bptt_recovery.json",
    ]

    resolved = {
        path.name: resolve_experiment_config(path, RunMode.TRAIN)[1]
        for path in paths
    }
    hwa = resolved["hwa_from_fp32.json"]
    assert hwa.settings.num_epochs == 2
    assert hwa.settings.weight_modifier.type == "add_normal"
    assert hwa.settings.weight_modifier.parameters["std_dev"] == 0.03
    assert (
        hwa.settings.weight_modifier.parameters["scale_mode"]
        == "output_channel_abs_max"
    )

    for name, spec in resolved.items():
        if name == "hwa_from_fp32.json":
            continue
        assert spec.common.data.dataset == "mnist"
        assert spec.settings.algorithm == "backprop"
        assert spec.settings.update_backend.type == "program_verify"

    for name in (
        "cmo_deployment_probe.json",
        "cmo_full_bptt_recovery.json",
        "cmo_lora_bptt_recovery.json",
    ):
        cmo = resolved[name]
        assert (
            cmo.settings.update_backend.parameters["device"]["mapping"]
            == "literal_conductance"
        )
    assert (
        resolved["cmo_affine_floor_deployment_probe.json"]
        .settings.update_backend.parameters["device"]["mapping"]
        == "affine_floor"
    )
    cmo_lora = resolved["cmo_lora_bptt_recovery.json"]
    for layer in cmo_lora.common.model.adapter.parameters["layers"].values():
        assert layer["device_noise"]["mapping"] == "literal_conductance"
    pcm_lora = resolved["pcm_lora_bptt_recovery.json"]
    assert pcm_lora.common.model.adapter.type == (
        "passive_layerwise_low_rank"
    )
    assert pcm_lora.settings.learning_rates == (0.01,) * 4


def test_mnist_wan_cmo_head_to_head_examples_resolve() -> None:
    directory = (
        ROOT
        / "examples"
        / "small_drn"
        / "mnist_wan_cmo_head_to_head"
    )
    paths = sorted(directory.glob("*.json"))
    assert [path.name for path in paths] == [
        "cmo_affine_one_day_deployment.json",
        "cmo_affine_one_day_full_bptt.json",
        "hwa_shared_from_fp32.json",
        "wan_affine_one_day_deployment.json",
        "wan_affine_one_day_full_bptt.json",
    ]

    resolved = {
        path.name: resolve_experiment_config(path, RunMode.TRAIN)[1]
        for path in paths
    }
    for spec in resolved.values():
        assert spec.common.data.dataset == "mnist"
        assert spec.common.model.voltage_amp == 4.0
        assert spec.common.model.current_amp == 0.25

    hwa = resolved["hwa_shared_from_fp32.json"]
    assert hwa.settings.num_epochs == 2
    assert hwa.settings.weight_modifier.type == "add_normal"
    assert hwa.settings.update_backend.type == "direct"

    for name, spec in resolved.items():
        if name == "hwa_shared_from_fp32.json":
            continue
        device = spec.settings.update_backend.parameters["device"]
        assert spec.settings.update_backend.type == "program_verify"
        assert device["mapping"] == "affine_floor"
        assert device["t_inference_seconds"] == 86400.0
        if name.startswith("wan_"):
            assert device["type"] == "aihwkit_reram_wan2022_physical"
            assert device["g_min_us"] == 1.0
            assert device["g_max_us"] == 40.0
        else:
            assert device["type"] == "aihwkit_reram_cmo"
            assert device["g_min_us"] == 9.0
            assert device["g_max_us"] == 88.199997

    assert resolved["wan_affine_one_day_full_bptt.json"].settings.num_epochs == 10
    assert resolved["cmo_affine_one_day_full_bptt.json"].settings.num_epochs == 10


def test_teacher_initialized_wan_cmo_examples_share_the_relu_mapping() -> None:
    directory = (
        ROOT
        / "examples"
        / "mnist_relu_drn"
        / "wan_cmo_teacher_initialized"
    )
    paths = sorted(directory.glob("*.json"))
    assert [path.name for path in paths] == [
        "cmo_deployment.json",
        "cmo_full_bptt.json",
        "hwa_add_normal.json",
        "ideal_teacher_map.json",
        "wan_deployment.json",
        "wan_full_bptt.json",
    ]
    resolved = {
        path.name: resolve_experiment_config(path, RunMode.TRAIN)[1]
        for path in paths
    }
    for spec in resolved.values():
        assert spec.model.dims == (1568, 100, 20)
        assert spec.model.input_gain == 100.0
        assert spec.model.conductance_min == 0.0
        assert spec.model.conductance_max == 0.00011
        assert spec.model.voltage_amp == 4.0
        assert spec.model.current_amp == 0.25
        assert spec.model.encoding == "single"
        assert spec.model.include_biases is False
        assert spec.data.validation_points == 5000

    ideal = resolved["ideal_teacher_map.json"]
    assert ideal.settings.num_epochs == 0
    assert ideal.settings.weight_modifier.type == "none"
    assert ideal.settings.update_backend.type == "ideal"

    hwa = resolved["hwa_add_normal.json"]
    assert hwa.settings.num_epochs == 2
    assert hwa.settings.weight_modifier.type == "add_normal"
    assert dict(hwa.settings.weight_modifier.parameters) == {
        "std_dev": 0.03,
        "seed": 101,
        "noisy_evaluation": False,
        "scale_mode": "output_channel_abs_max",
    }
    assert hwa.settings.update_backend.type == "ideal"

    for name in (
        "cmo_deployment.json",
        "cmo_full_bptt.json",
        "wan_deployment.json",
        "wan_full_bptt.json",
    ):
        spec = resolved[name]
        assert spec.settings.weight_modifier.type == "none"
        assert spec.settings.update_backend.type == "program_verify"
        device = spec.settings.update_backend.parameters["device"]
        assert device["mapping"] == "affine_floor"
        assert device["drn_conductance_at_g_max"] == 0.00011
        assert device["t_inference_seconds"] == 86400.0
        assert spec.settings.num_epochs == (
            10 if name.endswith("full_bptt.json") else 0
        )

    assert (
        resolved["cmo_deployment.json"]
        .settings.update_backend.parameters["device"]["g_min_us"]
        == 9.0
    )
    assert (
        resolved["wan_deployment.json"]
        .settings.update_backend.parameters["device"]["g_min_us"]
        == 1.0
    )


def test_measured_cohort_a_examples_resolve_matched_raw_and_isotonic_arms() -> None:
    directory = ROOT / "examples" / "small_drn"
    raw = resolve_experiment_config(
        directory / "measured_cohort_a_raw_mnist.json",
        RunMode.TRAIN,
    )[1]
    isotonic = resolve_experiment_config(
        directory / "measured_cohort_a_isotonic_mnist.json",
        RunMode.TRAIN,
    )[1]

    assert raw.extensions.update_backend == "measured_cohort_a"
    assert isotonic.extensions == raw.extensions
    assert raw.common.data.validation_points == 5000
    assert raw.common.model.input_gain == 100.0
    assert raw.common.model.weight_min == 0.0
    assert raw.common.model.weight_max == 1.1e-4
    assert raw.common.model.voltage_amp == 4.0
    assert raw.common.model.current_amp == 0.25
    assert raw.settings.learning_rates == (0.0, 0.0)
    assert raw.settings.bias_learning_rates == (0.0,)
    assert (
        raw.settings.learning_rate_selection.type
        == "bounded_relative_update_grid"
    )
    assert raw.settings.learning_rate_selection.parameters["canary_batches"] == 640
    assert raw.settings.learning_rate_selection.parameters["candidate_epochs"] == 3
    raw_backend = dict(raw.settings.update_backend.parameters)
    isotonic_backend = dict(isotonic.settings.update_backend.parameters)
    assert raw_backend.pop("curve_preprocessing") == "raw"
    assert (
        isotonic_backend.pop("curve_preprocessing")
        == "isotonic_nonincreasing"
    )
    assert raw_backend == isotonic_backend


def test_measured_cohort_b_example_resolves_matched_deployment_protocol() -> None:
    directory = ROOT / "examples" / "small_drn"
    cohort_a = resolve_experiment_config(
        directory / "measured_cohort_a_raw_mnist.json",
        RunMode.TRAIN,
    )[1]
    cohort_b = resolve_experiment_config(
        directory / "measured_cohort_b_raw_mnist.json",
        RunMode.TRAIN,
    )[1]

    assert cohort_b.extensions.update_backend == "measured_cohort_b"
    assert cohort_b.common == cohort_a.common
    assert cohort_b.settings.num_epochs == cohort_a.settings.num_epochs
    assert (
        cohort_b.settings.learning_rate_selection
        == cohort_a.settings.learning_rate_selection
    )
    cohort_a_backend = dict(cohort_a.settings.update_backend.parameters)
    cohort_b_backend = dict(cohort_b.settings.update_backend.parameters)
    assert cohort_a_backend.pop("cohort") == "A"
    assert cohort_b_backend.pop("cohort") == "B"
    assert cohort_a_backend == cohort_b_backend


def test_measured_cohort_b_lora_example_uses_reset_physical_factors() -> None:
    path = (
        ROOT
        / "examples"
        / "small_drn"
        / "measured_cohort_b_lora_mnist.json"
    )

    _, spec = resolve_experiment_config(path, RunMode.TRAIN)

    assert spec.extensions.model_adapter == "passive_layerwise_low_rank"
    assert spec.extensions.update_backend == "measured_cohort_b_lora"
    assert spec.extensions.algorithm == "backprop"
    assert spec.common.model.voltage_amp == 4.0
    assert spec.common.model.current_amp == 0.25
    assert spec.settings.learning_rates == pytest.approx(
        (1.125e-6, 6.43e-7, 2.77e-7, 6.04e-9)
    )
    assert spec.settings.bias_learning_rates == ()
    assert spec.settings.learning_rate_selection.type == "none"
    assert (
        spec.settings.update_backend.parameters["initial_pulse_index"]
        == 4999
    )
    for layer in spec.common.model.adapter.parameters["layers"].values():
        assert layer["rank"] == 4
        assert layer["device_noise"] is None


def test_measured_cohort_b_threshold_example_uses_fixed_selected_rates() -> None:
    directory = ROOT / "examples" / "small_drn"
    baseline = resolve_experiment_config(
        directory / "measured_cohort_b_raw_mnist.json",
        RunMode.TRAIN,
    )[1]
    threshold = resolve_experiment_config(
        directory / "measured_cohort_b_threshold_mnist.json",
        RunMode.TRAIN,
    )[1]

    assert threshold.common == baseline.common
    assert threshold.extensions.update_backend == "measured_cohort_b"
    assert threshold.settings.learning_rate_selection.type == "none"
    assert threshold.settings.learning_rates == pytest.approx(
        (2.5922684959447867e-7, 1.908048705521415e-8)
    )
    assert threshold.settings.bias_learning_rates == pytest.approx(
        (2.5922684959447867e-7,)
    )
    parameters = threshold.settings.update_backend.parameters
    assert (
        parameters["programming_deadband_mode"]
        == "accumulated_shadow_relative_rms"
    )
    assert parameters["programming_deadband_relative"] == pytest.approx(0.003)


def test_measured_cohort_b_probabilistic_example_is_seeded_and_fixed_rate() -> None:
    path = (
        ROOT
        / "examples"
        / "small_drn"
        / "measured_cohort_b_probabilistic_mnist.json"
    )
    spec = resolve_experiment_config(path, RunMode.TRAIN)[1]

    assert spec.extensions.update_backend == "measured_cohort_b"
    assert spec.settings.learning_rate_selection.type == "none"
    parameters = spec.settings.update_backend.parameters
    assert parameters["programming_deadband_mode"] == "none"
    assert parameters["probabilistic_write_mode"] == "uniform_bernoulli"
    assert parameters["probabilistic_write_probability"] == pytest.approx(
        0.85
    )
    assert parameters["probabilistic_write_scale_relative"] == 0.0
    assert parameters["probabilistic_write_seed"] == 314159
