from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from ebl.registry import parse_experiment_config
from ebl.runtime.mnist_relu_drn.runtime import (
    _validate_cohort_b_source_metadata,
)
from ebl.schema.base import RunMode
from methods.measured_trace import MeasuredTraceConfig


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "examples/mnist_relu_drn/"
    "measured_raw_single_quad_common_window_finetune_10ep.json"
)
CONFIG_DIR = (
    ROOT
    / "examples/mnist_relu_drn/four_device_cohort_b_transfer"
)
SEEDS = (42, 43, 44, 45, 46)
CONFIGS = tuple(
    CONFIG_DIR / f"assignment_seed_{seed}.json" for seed in SEEDS
)


def _train(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    definition, document = parse_experiment_config(payload)
    return definition.resolve(document, RunMode.TRAIN)


def test_transfer_configs_freeze_five_zero_update_quad_assignments() -> None:
    source = _train(SOURCE)

    for expected_seed, path in zip(SEEDS, CONFIGS, strict=True):
        spec = _train(path)
        assert spec.runtime == source.runtime
        assert spec.data == source.data
        assert spec.model == source.model
        assert spec.solver == source.solver
        assert spec.mapping == source.mapping
        assert spec.settings.num_epochs == 1
        assert spec.settings.max_batches == 1
        assert spec.settings.learning_rates == (0.0, 0.0)
        assert spec.settings.update_backend.type == "measured_cohort_b"
        parameters = spec.settings.update_backend.parameters
        assert parameters["cohort"] == "B"
        assert parameters["assignment_seed"] == expected_seed
        assert (
            parameters["initial_target_mapping"]
            == "dual_rail_quad_common_window"
        )
        assert dict(parameters["dual_rail_layout_by_parameter"]) == {
            "base.dense_weight.0": "halves",
            "base.dense_weight.1": "paired",
        }
        measured = MeasuredTraceConfig.from_mapping(parameters)
        assert measured.cohort == "B"


def test_single_cohort_b_capability_remains_quad_only() -> None:
    payload = json.loads(CONFIGS[0].read_text(encoding="utf-8"))
    parameters = payload["modes"]["train"]["update_backend"]["parameters"]
    parameters["initial_target_mapping"] = "literal"
    parameters.pop("dual_rail_layout_by_parameter")

    with pytest.raises(ValueError, match="dual_rail_quad_common_window"):
        parse_experiment_config(payload)


def test_quad_cohort_b_source_provenance_requires_matching_a_checkpoint() -> None:
    spec = _train(CONFIGS[0])
    metadata = {
        "encoding": "single",
        "initialization": "teacher_mapped",
        "update_backend": "measured_cohort_a",
        "initial_target_mapping": "dual_rail_quad_common_window",
        "dual_rail_layout_by_parameter": {
            "base.dense_weight.0": "halves",
            "base.dense_weight.1": "paired",
        },
        "device_data_sha256": "device-sha",
    }
    _validate_cohort_b_source_metadata(
        metadata,
        spec=spec,
        device_data_sha256="device-sha",
    )

    for key in tuple(metadata):
        mutated = deepcopy(metadata)
        mutated[key] = "wrong"
        with pytest.raises(ValueError, match=key):
            _validate_cohort_b_source_metadata(
                mutated,
                spec=spec,
                device_data_sha256="device-sha",
            )
