from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ebl.registry import parse_experiment_config, resolve_experiment_config
from ebl.schema.raw_active_cell import (
    P90_COMMON_CELL_9_PROFILE,
    RAW_ACTIVE_CELL_CODEBOOK_BASELINE,
    RAW_ACTIVE_CELL_CODEBOOK_LEVELS,
    RAW_ACTIVE_CELL_CODEBOOK_STEP,
    RawActiveCellProgramVerifySpec,
)
import ebl.runtime.raw_active_cell as raw_active_cell_runtime
from ebl.schema.base import ConfigError, RunMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = (
    ROOT / "examples" / "reram_program_verify" / "raw_active_common_cell_9"
)


def _payload(name: str) -> dict:
    return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["smoke.json", "development.json", "heldout.json"])
def test_raw_active_common_cell_configs_resolve(name: str) -> None:
    definition, spec = resolve_experiment_config(
        CONFIG_ROOT / name, RunMode.CHARACTERIZE
    )
    assert definition.experiment_id == "ibm_om_raw_active_cell_program_verify.v1"
    assert isinstance(spec, RawActiveCellProgramVerifySpec)
    assert spec.settings.codebook_levels == RAW_ACTIVE_CELL_CODEBOOK_LEVELS
    assert spec.settings.controller == "one_pulse"
    assert spec.settings.start_protocol == "lower_to_target"


def test_production_configs_freeze_common_baseline_and_increment() -> None:
    _, development = resolve_experiment_config(
        CONFIG_ROOT / "development.json", RunMode.CHARACTERIZE
    )
    _, heldout = resolve_experiment_config(
        CONFIG_ROOT / "heldout.json", RunMode.CHARACTERIZE
    )
    assert development.settings.profile == P90_COMMON_CELL_9_PROFILE
    assert heldout.settings.profile == P90_COMMON_CELL_9_PROFILE
    assert development.runtime.assignment_seed == 84001
    assert heldout.runtime.assignment_seed == 85001
    assert development.settings.sampled_devices == 1024
    assert development.settings.repeats_per_device == 4
    assert RAW_ACTIVE_CELL_CODEBOOK_BASELINE == pytest.approx(0.6001664102077484)
    assert RAW_ACTIVE_CELL_CODEBOOK_STEP == pytest.approx(0.012380622327327728)


def test_runtime_routes_population_sampling_through_pinned_aihwkit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, spec = resolve_experiment_config(
        CONFIG_ROOT / "smoke.json", RunMode.CHARACTERIZE
    )
    population = object()
    receipt = {"backend": "external_pinned_aihwkit"}
    calls: dict[str, object] = {}

    def fake_external(
        binding_keys: tuple[str, ...],
        binding_shapes: tuple[tuple[int, ...], ...],
        **kwargs: object,
    ) -> tuple[object, dict[str, str]]:
        calls["binding_keys"] = binding_keys
        calls["binding_shapes"] = binding_shapes
        calls.update(kwargs)
        return population, receipt

    def fail_in_process(*_args: object, **_kwargs: object) -> None:
        pytest.fail("The in-process sampler must not run when the pinned path is set.")

    monkeypatch.setenv("EBL_AIHWKIT_PYTHON", "/tmp/pinned-aihwkit-python")
    monkeypatch.setattr(
        raw_active_cell_runtime,
        "sample_om_array_population_layout_external",
        fake_external,
    )
    monkeypatch.setattr(
        raw_active_cell_runtime,
        "sample_om_array_population_layout",
        fail_in_process,
    )
    population_path = tmp_path / "population.npz"
    receipt_path = tmp_path / "receipt.json"

    actual_population, actual_receipt = (
        raw_active_cell_runtime._sample_population_for_runtime(
            spec,
            population_path=population_path,
            receipt_path=receipt_path,
        )
    )

    assert actual_population is population
    assert actual_receipt is receipt
    assert calls == {
        "binding_keys": spec.device.binding_keys,
        "binding_shapes": spec.device.binding_shapes,
        "assignment_seed": spec.runtime.assignment_seed,
        "corruption_policy": spec.device.corruption_policy,
        "aihwkit_python": Path("/tmp/pinned-aihwkit-python"),
        "population_path": population_path,
        "receipt_path": receipt_path,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("modes", "characterize", "codebook", "levels"), 7),
        (("modes", "characterize", "tolerance"), 0.001),
        (("modes", "characterize", "controller"), "adaptive"),
        (("modes", "characterize", "sampled_devices"), 1000),
    ],
)
def test_raw_active_common_cell_production_contract_fails_closed(
    path: tuple[str, ...], value: object
) -> None:
    payload = copy.deepcopy(_payload("development.json"))
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ConfigError):
        definition, document = parse_experiment_config(payload)
        definition.resolve(document, RunMode.CHARACTERIZE)
