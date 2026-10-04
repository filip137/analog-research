from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ebl.cli import (
    CommandHandlers,
    TrainRequest,
    _default_train_handler,
    main,
)
from experiments.definitions import get_definition
from experiments.mnist_analog_relu.runtime import (
    _validate_request as _validate_v1_crossbar_request,
)
from experiments.schema import RunMode


ROOT = Path(__file__).resolve().parents[1]


def test_train_forwards_independent_staged_input_artifacts() -> None:
    config = ROOT / "examples" / "small_drn" / "base.json"
    seen: list[TrainRequest] = []

    result = main(
        [
            "train",
            "--config",
            str(config),
            "--output-dir",
            "runs",
            "--resume",
            "resume.pt",
            "--teacher-weights",
            "teacher.pt",
            "--device-state",
            "deployed.pt",
            "--selection-receipt",
            "selection.json",
        ],
        handlers=CommandHandlers(train=lambda request: seen.append(request)),
    )

    assert result == 0
    assert len(seen) == 1
    request = seen[0]
    assert request.resume == Path("resume.pt")
    assert request.teacher_weights == Path("teacher.pt")
    assert request.device_state == Path("deployed.pt")
    assert request.selection_receipt == Path("selection.json")
    assert request.weights is None
    assert request.base_weights is None
    for option in (
        "--resume",
        "--teacher-weights",
        "--device-state",
        "--selection-receipt",
    ):
        assert option in request.command


def test_describe_v2_exposes_staged_input_artifacts() -> None:
    stdout = io.StringIO()
    assert main(
        [
            "describe",
            "--experiment",
            "mnist_ibm_om_crossbar_relu.v2",
            "--json",
        ],
        stdout=stdout,
    ) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["experiment_id"] == "mnist_ibm_om_crossbar_relu.v2"
    assert payload["schema_version"] == 2
    assert payload["supported_modes"] == ["train"]
    assert payload["combinations"] == []
    assert payload["commands"]["train"]["optional_input_options"] == [
        "--device-data",
        "--teacher-weights",
        "--device-state",
        "--selection-receipt",
    ]


def test_v2_default_handler_lazily_dispatches_to_staged_runtime(
    monkeypatch,
) -> None:
    module_name = "experiments.mnist_analog_relu.staged_runtime"
    runtime = ModuleType(module_name)
    seen = []

    def run_train(request) -> int:
        seen.append(request)
        return 23

    runtime.run_train = run_train
    monkeypatch.setitem(sys.modules, module_name, runtime)
    definition = get_definition("mnist_ibm_om_crossbar_relu.v2")
    request = SimpleNamespace(definition=definition)

    assert definition.supported_modes == (RunMode.TRAIN,)
    assert definition.parser.__module__ == (
        "experiments.mnist_analog_relu.staged_config"
    )
    assert definition.resolver.__module__ == (
        "experiments.mnist_analog_relu.staged_config"
    )
    assert _default_train_handler(request) == 23
    assert seen == [request]


@pytest.mark.parametrize("name", ("device_state", "selection_receipt"))
def test_v1_crossbar_rejects_staged_v2_inputs(name: str) -> None:
    request = SimpleNamespace(
        teacher_weights=Path("teacher.pt"),
        weights=None,
        base_weights=None,
        resume=None,
        device_data=None,
        device_model=None,
        device_state=None,
        selection_receipt=None,
    )
    setattr(request, name, Path("wrong-artifact"))

    with pytest.raises(ValueError, match=name.replace("_", "-")):
        _validate_v1_crossbar_request(request)
