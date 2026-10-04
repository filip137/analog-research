from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from experiments.artifacts import atomic_write_json, sha256_file
from training.star_targets import (
    CROSSBAR_STATE_REPRESENTATION,
    StarSourceBinding,
    StarTargetAccumulator,
    StarTargetBinding,
    StarTargetError,
    build_calibration_binding,
    load_star_targets,
    save_star_targets,
    state_components,
)


def _source() -> StarSourceBinding:
    return StarSourceBinding(
        model_checkpoint_sha256="1" * 64,
        healthy_deployment_artifact_sha256="2" * 64,
        healthy_persistent_state_sha256="3" * 64,
        healthy_forward_state_sha256="4" * 64,
        hardware_instance_id="5" * 64,
        config_sha256="6" * 64,
    )


def _cohort(examples: int = 20):
    labels = torch.arange(examples, dtype=torch.int64) % 10
    sample_ids = torch.arange(100, 100 + examples, dtype=torch.int64)
    inputs = torch.arange(examples * 4, dtype=torch.float32).reshape(examples, 4)
    inputs = inputs / 17.0
    calibration = build_calibration_binding(
        dataset_id="mnist",
        split="train",
        ordered_sample_ids=sample_ids,
        model_inputs=inputs,
        labels=labels,
    )
    return labels, sample_ids, inputs, calibration


def _states(representation: str, examples: int) -> dict[str, torch.Tensor]:
    result = {}
    for component_index, component in enumerate(state_components(representation)):
        values = torch.arange(
            examples * component.width,
            dtype=torch.float32,
        ).reshape(examples, component.width)
        result[component.key] = (
            values / float(component.width + 3) - 2.0 + component_index
        )
    return result


def _binding(
    representation: str,
    storage_dtype: str,
    calibration,
) -> StarTargetBinding:
    return StarTargetBinding(
        architecture_id="mnist.crossbar.784-50-10",
        state_representation_id=representation,
        source=_source(),
        calibration=calibration,
        component_gains=(0.25, 0.5),
        storage_dtype=storage_dtype,
    )


def _bundle(representation: str, storage_dtype: str, *, split: bool = False):
    labels, sample_ids, inputs, calibration = _cohort()
    states = _states(representation, labels.numel())
    binding = _binding(representation, storage_dtype, calibration)
    accumulator = StarTargetAccumulator(
        class_count=10,
        representation_id=representation,
        component_specs=state_components(representation),
    )
    boundaries = (slice(0, 7), slice(7, None)) if split else (slice(None),)
    for selected in boundaries:
        accumulator.observe(
            labels[selected],
            {key: value[selected] for key, value in states.items()},
            inputs=inputs[selected],
            sample_ids=sample_ids[selected],
        )
    return accumulator.finalize(binding), binding, states, labels


def test_fp64_class_means_are_batching_invariant() -> None:
    whole, _binding_value, states, labels = _bundle(
        CROSSBAR_STATE_REPRESENTATION, "fp32"
    )
    split, _binding_value, _states_value, _labels_value = _bundle(
        CROSSBAR_STATE_REPRESENTATION, "fp32", split=True
    )

    assert whole.semantic_sha256 == split.semantic_sha256
    assert whole.class_counts.tolist() == [2] * 10
    for component in state_components(CROSSBAR_STATE_REPRESENTATION):
        expected = torch.stack(
            [
                states[component.key][labels == label]
                .to(torch.float64)
                .mean(dim=0)
                .to(torch.float32)
                for label in range(10)
            ]
        )
        torch.testing.assert_close(whole.means(component.key), expected)
        assert torch.equal(
            whole.stored_means[component.key],
            split.stored_means[component.key],
        )


@pytest.mark.parametrize(
    ("representation", "storage_dtype", "value_count", "mean_bytes"),
    [
        (CROSSBAR_STATE_REPRESENTATION, "int8", 600, 600),
        (CROSSBAR_STATE_REPRESENTATION, "fp16", 600, 1_200),
    ],
)
def test_compact_storage_value_and_byte_counts(
    representation: str,
    storage_dtype: str,
    value_count: int,
    mean_bytes: int,
) -> None:
    bundle, _binding_value, _states_value, _labels_value = _bundle(
        representation, storage_dtype
    )

    assert bundle.stored_value_count == value_count
    assert bundle.stored_mean_nbytes == mean_bytes


@pytest.mark.parametrize("storage_dtype", ["fp16", "int8"])
@pytest.mark.parametrize(
    "representation", [CROSSBAR_STATE_REPRESENTATION],
)
def test_pickle_free_round_trip_checks_receipt_and_binding(
    tmp_path,
    representation: str,
    storage_dtype: str,
) -> None:
    bundle, binding, _states_value, _labels_value = _bundle(
        representation, storage_dtype, split=True
    )
    path = tmp_path / "star_targets.npz"

    record = save_star_targets(path, bundle)
    loaded = load_star_targets(path, record.receipt_path, binding)

    assert record.artifact_sha256 == sha256_file(path)
    assert record.receipt_sha256 == sha256_file(record.receipt_path)
    assert loaded.semantic_sha256 == bundle.semantic_sha256
    assert loaded.binding == binding
    assert loaded.class_counts.tolist() == [2] * 10
    for component in state_components(representation):
        assert loaded.means(component.key).dtype == torch.float32
        assert torch.equal(
            loaded.stored_means[component.key],
            bundle.stored_means[component.key],
        )


def test_int8_is_symmetric_half_away_from_zero_and_all_zero_safe() -> None:
    labels, sample_ids, inputs, calibration = _cohort(examples=10)
    binding = _binding(CROSSBAR_STATE_REPRESENTATION, "int8", calibration)
    hidden = torch.zeros((10, 50), dtype=torch.float64)
    hidden[0, 0] = 0.5
    hidden[0, 1] = -0.5
    hidden[9, 49] = 127.0
    output = torch.zeros((10, 10), dtype=torch.float64)
    accumulator = StarTargetAccumulator(CROSSBAR_STATE_REPRESENTATION)
    accumulator.observe(
        labels,
        {"hidden_post_relu": hidden, "output_logits": output},
        inputs=inputs,
        sample_ids=sample_ids,
    )

    bundle = accumulator.finalize(binding)

    assert bundle.storage["scales"]["hidden_post_relu"] == pytest.approx(1.0)
    assert bundle.storage["scales"]["output_logits"] == pytest.approx(1.0)
    assert bundle.stored_means["hidden_post_relu"][0, :2].tolist() == [1, -1]
    assert int(bundle.stored_means["hidden_post_relu"][9, 49]) == 127
    assert torch.count_nonzero(bundle.stored_means["output_logits"]) == 0


def test_constructor_class_count_and_finalize_quantization_api() -> None:
    labels = torch.arange(6, dtype=torch.int64) % 3
    sample_ids = torch.arange(6, dtype=torch.int64)
    inputs = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    calibration = build_calibration_binding(
        dataset_id="MNIST",
        split="calibration",
        ordered_sample_ids=sample_ids,
        model_inputs=inputs,
        labels=labels,
    )
    binding = replace(
        _binding(CROSSBAR_STATE_REPRESENTATION, "fp32", calibration),
        class_count=3,
    )
    states = _states(CROSSBAR_STATE_REPRESENTATION, 6)
    accumulator = StarTargetAccumulator(
        class_count=3,
        representation_id=CROSSBAR_STATE_REPRESENTATION,
        component_specs=state_components(CROSSBAR_STATE_REPRESENTATION),
    )
    accumulator.observe(
        labels,
        states,
        inputs=inputs,
        sample_ids=sample_ids,
    )

    bundle = accumulator.finalize(binding, "int8")

    assert bundle.binding.storage_dtype == "int8"
    assert bundle.class_counts.tolist() == [2, 2, 2]
    assert bundle.stored_value_count == 3 * (50 + 10)


def test_finalize_rejects_missing_class_and_wrong_observed_cohort() -> None:
    labels, sample_ids, inputs, calibration = _cohort(examples=20)
    states = _states(CROSSBAR_STATE_REPRESENTATION, 20)
    binding = _binding(CROSSBAR_STATE_REPRESENTATION, "fp16", calibration)
    missing = StarTargetAccumulator(CROSSBAR_STATE_REPRESENTATION)
    missing.observe(
        labels[:10].clone().fill_(0),
        {key: value[:10] for key, value in states.items()},
        inputs=inputs[:10],
        sample_ids=sample_ids[:10],
    )
    with pytest.raises(StarTargetError, match="observed STAR examples"):
        missing.finalize(binding)

    wrong_inputs = inputs.clone()
    wrong_inputs[0, 0] += 1.0
    wrong = StarTargetAccumulator(CROSSBAR_STATE_REPRESENTATION)
    wrong.observe(
        labels,
        states,
        inputs=wrong_inputs,
        sample_ids=sample_ids,
    )
    with pytest.raises(StarTargetError, match="observed STAR inputs"):
        wrong.finalize(binding)


def test_load_fails_closed_for_source_representation_and_receipt_mismatch(
    tmp_path,
) -> None:
    bundle, binding, _states_value, _labels_value = _bundle(
        CROSSBAR_STATE_REPRESENTATION, "fp16"
    )
    path = tmp_path / "star_targets.npz"
    record = save_star_targets(path, bundle)
    wrong_source = replace(
        binding.source,
        healthy_persistent_state_sha256="a" * 64,
    )
    wrong_binding = replace(binding, source=wrong_source)

    with pytest.raises(StarTargetError, match="binding to match exactly"):
        load_star_targets(path, record.receipt_path, wrong_binding)
    with pytest.raises(StarTargetError, match="supported STAR state representation"):
        replace(binding, state_representation_id="unsupported.state.v1")

    receipt = json.loads(record.receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_sha256"] = "0" * 64
    atomic_write_json(record.receipt_path, receipt)
    with pytest.raises(StarTargetError, match="receipt to match"):
        load_star_targets(path, record.receipt_path, binding)


def test_accumulator_rejects_nonfinite_or_structurally_wrong_states() -> None:
    accumulator = StarTargetAccumulator(CROSSBAR_STATE_REPRESENTATION)
    labels = torch.arange(10, dtype=torch.int64)
    states = _states(CROSSBAR_STATE_REPRESENTATION, 10)
    states["hidden_post_relu"][0, 0] = torch.nan

    with pytest.raises(TypeError, match="inputs"):
        accumulator.observe(labels, states)

    with pytest.raises(StarTargetError, match="finite values"):
        accumulator.observe(
            labels,
            states,
            inputs=torch.zeros((10, 2), dtype=torch.float32),
            sample_ids=torch.arange(10, dtype=torch.int64),
        )

    with pytest.raises(StarTargetError, match="component_specs"):
        StarTargetAccumulator(
            class_count=10,
            representation_id=CROSSBAR_STATE_REPRESENTATION,
            component_specs=(),
        )
