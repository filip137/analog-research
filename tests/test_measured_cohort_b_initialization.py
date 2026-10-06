from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import RunMode, resolve_experiment_config
from experiments.small_network.initialization import validate_training_initialization


ROOT = Path(__file__).parents[1]


def _spec(name: str):
    return resolve_experiment_config(
        ROOT / "examples" / "small_drn" / name,
        RunMode.TRAIN,
    )[1]


def _request(**overrides):
    values = {
        "weights": None,
        "base_weights": None,
        "resume": None,
        "device_data": Path("devices.hdf5"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_new_cohort_b_run_requires_named_source_weights() -> None:
    spec = _spec("measured_cohort_b_raw_mnist.json")

    with pytest.raises(ValueError, match="explicit named cohort-A checkpoint"):
        validate_training_initialization(_request(), spec)

    validate_training_initialization(
        _request(weights=Path("cohort_a_weights.pt")),
        spec,
    )
    validate_training_initialization(
        _request(resume=Path("cohort_b_resume.pt")),
        spec,
    )


def test_cohort_a_still_rejects_named_weight_initialization() -> None:
    spec = _spec("measured_cohort_a_raw_mnist.json")

    with pytest.raises(ValueError, match="start every new run at pulse index 0"):
        validate_training_initialization(
            _request(weights=Path("weights.pt")),
            spec,
        )


def test_measured_lora_requires_base_weights_and_device_data() -> None:
    spec = _spec("measured_cohort_b_lora_mnist.json")

    with pytest.raises(ValueError, match="explicit named cohort-A base"):
        validate_training_initialization(_request(), spec)
    with pytest.raises(ValueError, match="not a full --weights"):
        validate_training_initialization(
            _request(weights=Path("full.pt")),
            spec,
        )

    validate_training_initialization(
        _request(base_weights=Path("cohort_a_base.pt")),
        spec,
    )
    validate_training_initialization(
        _request(resume=Path("lora_resume.pt")),
        spec,
    )
