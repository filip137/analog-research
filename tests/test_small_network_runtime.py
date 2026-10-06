from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ebl.cli import (
    ImportLegacyCheckpointRequest,
    LinspaceRequest,
    TrainRequest,
    ValidateRequest,
)
from experiments.definitions import get_definition
from experiments.schema import RunMode
from experiments.small_network.components import build_model_stack
from experiments.small_network.config import (
    ComponentSettings,
    parse_small_drn_config,
)
from experiments.small_network.runtime import (
    TrainingEpochReport,
    TrainingOutcome,
    execute_train,
    import_legacy_checkpoint,
    run_linspace,
    run_train,
    run_validate,
)
from model.resistive.builders import ParameterCatalog
from training.checkpoint import NAMED_WEIGHTS_SCHEMA, save_named_weights


_REPO_ROOT = Path(__file__).parents[1]


def _document():
    payload = json.loads(
        (_REPO_ROOT / "examples" / "small_drn" / "base.json").read_text(
            encoding="utf-8"
        )
    )
    payload["data"].update(
        {"num_points": 10, "batch_size": 2, "shuffle": False}
    )
    payload["solver"].update(
        {"inference_iterations": 1, "training_iterations": 1}
    )
    payload["solver"]["polish"]["enabled"] = False
    payload["modes"]["train"].update(
        {
            "num_epochs": 1,
            "max_batches": 1,
            "max_validation_batches": 1,
        }
    )
    payload["modes"]["linspace"]["samples"] = 2
    payload["modes"]["validate"]["sample_limit"] = 4
    return parse_small_drn_config(payload)


def _hardware_aware_document():
    document = _document()
    return replace(
        document,
        train=replace(
            document.train,
            algorithm="backprop",
            weight_modifier=ComponentSettings(
                type="add_normal",
                parameters={
                    "std_dev": 0.2,
                    "seed": 17,
                    "noisy_evaluation": True,
                    "scale_mode": "output_channel_abs_max",
                },
            ),
        ),
    )


def _adapter_document(*, backend: str = "direct"):
    payload = json.loads(
        (_REPO_ROOT / "examples" / "small_drn" / "lora.json").read_text(
            encoding="utf-8"
        )
    )
    payload["data"].update(
        {"num_points": 10, "batch_size": 2, "shuffle": False}
    )
    payload["solver"].update(
        {"inference_iterations": 1, "training_iterations": 1}
    )
    payload["solver"]["polish"]["enabled"] = False
    payload["modes"]["train"].update(
        {
            "num_epochs": 1,
            "max_batches": 1,
            "max_validation_batches": 1,
            "update_backend": {"type": backend, "parameters": {}},
        }
    )
    payload["modes"]["linspace"]["samples"] = 2
    payload["modes"]["validate"]["sample_limit"] = 4
    return parse_small_drn_config(payload)


def _digital_adapter_document():
    payload = json.loads(
        (
            _REPO_ROOT
            / "examples"
            / "small_drn"
            / "digital_lora_reram_wan2022_digits.json"
        ).read_text(encoding="utf-8")
    )
    payload["data"].update(
        {"num_points": 20, "batch_size": 4, "shuffle": False}
    )
    payload["solver"].update(
        {"inference_iterations": 2, "training_iterations": 2}
    )
    payload["modes"]["train"].update(
        {
            "num_epochs": 1,
            "max_batches": 1,
            "max_validation_batches": 1,
        }
    )
    payload["modes"]["validate"]["sample_limit"] = 4
    return parse_small_drn_config(payload)


def _passive_layerwise_adapter_document(*, algorithm: str = "ep"):
    payload = json.loads(
        (
            _REPO_ROOT
            / "examples"
            / "small_drn"
            / "passive_layerwise_lora_reram_wan2022_digits.json"
        ).read_text(encoding="utf-8")
    )
    payload["data"].update(
        {"num_points": 20, "batch_size": 4, "shuffle": False}
    )
    payload["solver"].update(
        {"inference_iterations": 2, "training_iterations": 2}
    )
    payload["modes"]["train"].update(
        {
            "num_epochs": 1,
            "algorithm": algorithm,
            "nudging": 0.05 if algorithm == "ep" else 0.0,
            "max_batches": 1,
            "max_validation_batches": 1,
        }
    )
    payload["modes"]["validate"]["sample_limit"] = 4
    return parse_small_drn_config(payload)


def _write_adapter_base_weights(document, path: Path) -> dict[str, torch.Tensor]:
    stack = build_model_stack(document.common)
    base_catalog = ParameterCatalog(
        stack.bundle.catalog.for_group("base", checkpointed_only=True)
    )
    save_named_weights(path, base_catalog)
    return {
        binding.key: binding.state.detach().cpu().clone()
        for binding in base_catalog
    }


def _runs(output_root: Path) -> list[Path]:
    return sorted(
        (path for path in output_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location="cpu")


def test_public_handlers_complete_tiny_cpu_pipeline_and_exact_continuation(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _document()
    output_root = tmp_path / "runs"
    train_spec = definition.resolve(document, RunMode.TRAIN)

    assert (
        run_train(
            TrainRequest(
                definition=definition,
                spec=train_spec,
                config_path=tmp_path / "config.json",
                output_dir=output_root,
                weights=None,
                base_weights=None,
                resume=None,
                command=("ebl", "train"),
            )
        )
        == 0
    )
    first_run = _runs(output_root)[0]
    first_result = _read_json(first_run / "result.json")
    assert [(item["path"], item["kind"]) for item in first_result["artifacts"]] == [
        ("checkpoints/weights.pt", "weights"),
        ("checkpoints/resume.pt", "resume"),
    ]
    assert first_result["metrics"]["selected"]["evaluation"] == "clean"
    assert (
        first_result["metrics"]["evaluation_protocol"]["validation_effective"]
        == "held_out_test"
    )
    resume_payload = _torch_load(first_run / "checkpoints" / "resume.pt")
    assert resume_payload["selected_weights"] is not None
    assert resume_payload["runtime_state_state"] is not None

    # The target epoch is already complete.  The continuation must still
    # materialize the pre-resume selected snapshot and a fresh full resume.
    assert (
        run_train(
            TrainRequest(
                definition=definition,
                spec=train_spec,
                config_path=tmp_path / "config.json",
                output_dir=output_root,
                weights=None,
                base_weights=None,
                resume=first_run / "checkpoints" / "resume.pt",
                command=("ebl", "train", "--resume"),
            )
        )
        == 0
    )
    continued_run = _runs(output_root)[1]
    first_weights = _torch_load(first_run / "checkpoints" / "weights.pt")
    continued_weights = _torch_load(
        continued_run / "checkpoints" / "weights.pt"
    )
    assert first_weights["metadata"] == continued_weights["metadata"]
    assert first_weights["weights"].keys() == continued_weights["weights"].keys()
    for key in first_weights["weights"]:
        torch.testing.assert_close(
            first_weights["weights"][key],
            continued_weights["weights"][key],
            rtol=0.0,
            atol=0.0,
        )

    selected_path = continued_run / "checkpoints" / "weights.pt"
    linspace_spec = definition.resolve(document, RunMode.LINSPACE)
    assert (
        run_linspace(
            LinspaceRequest(
                definition=definition,
                spec=linspace_spec,
                config_path=tmp_path / "config.json",
                output_dir=output_root,
                weights=selected_path,
                command=("ebl", "linspace"),
            )
        )
        == 0
    )
    linspace_run = _runs(output_root)[2]
    linspace_result = _read_json(linspace_run / "result.json")
    assert linspace_result["metrics"]["examples"] == 4
    assert (
        linspace_result["metrics"]["input_protocol"][
            "legacy_test_assignment_effect"
        ]
        == "no_op"
    )
    with np.load(
        linspace_run / "artifacts" / "linspace_iteration_counts.npz"
    ) as values:
        assert values["iteration_grid"].shape == (2, 2)

    validate_spec = definition.resolve(document, RunMode.VALIDATE)
    assert (
        run_validate(
            ValidateRequest(
                definition=definition,
                spec=validate_spec,
                config_path=tmp_path / "config.json",
                output_dir=output_root,
                weights=selected_path,
                command=("ebl", "validate"),
            )
        )
        == 0
    )
    validate_run = _runs(output_root)[3]
    validate_result = _read_json(validate_run / "result.json")
    assert validate_result["metrics"]["evaluation"] == "clean"
    assert validate_result["metrics"]["split_protocol"] == {
        "requested": "test",
        "effective": "held_out_test",
    }
    assert validate_result["metrics"]["examples"] <= 4


def test_execute_train_reports_every_epoch_independent_of_log_every(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _document()
    document = replace(
        document,
        train=replace(document.train, num_epochs=3, log_every=10),
    )
    spec = definition.resolve(document, RunMode.TRAIN)
    reports: list[TrainingEpochReport] = []

    outcome = execute_train(
        TrainRequest(
            definition=definition,
            spec=spec,
            config_path=tmp_path / "config.json",
            output_dir=tmp_path / "runs",
            weights=None,
            base_weights=None,
            resume=None,
            command=("agent", "train"),
        ),
        observers=(reports.append,),
    )

    assert isinstance(outcome, TrainingOutcome)
    assert [report.completed_epochs for report in reports] == [1, 2, 3]
    assert [report.epoch_index for report in reports] == [0, 1, 2]
    assert outcome.last_epoch_report is reports[-1]
    assert outcome.selected_validation_accuracy is not None
    assert 0.0 <= outcome.selected_validation_accuracy <= 1.0
    assert all(
        0.0 <= report.validation_error_fraction <= 1.0
        and 0.0 <= report.validation_accuracy <= 1.0
        for report in reports
    )
    # log_every=10 still writes the terminal metric only; observers are
    # deliberately independent and saw all three epoch boundaries.
    assert (outcome.run_dir / "metrics.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 1
    with pytest.raises(FrozenInstanceError):
        reports[0].completed_epochs = 99


def test_observer_exception_marks_run_failed_and_is_reraised(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    spec = definition.resolve(_document(), RunMode.TRAIN)

    class StopTrial(RuntimeError):
        pass

    def stop(_report: TrainingEpochReport) -> None:
        raise StopTrial("pruned by observer")

    output_root = tmp_path / "runs"
    with pytest.raises(StopTrial, match="pruned by observer"):
        execute_train(
            TrainRequest(
                definition=definition,
                spec=spec,
                config_path=tmp_path / "config.json",
                output_dir=output_root,
                weights=None,
                base_weights=None,
                resume=None,
                command=("agent", "train"),
            ),
            observers=(stop,),
        )

    run_dir = _runs(output_root)[0]
    status = _read_json(run_dir / "status.json")
    assert status["status"] == "failed"
    assert status["error"] == {
        "type": "StopTrial",
        "message": "pruned by observer",
    }
    assert not (run_dir / "result.json").exists()
    # The callback runs after the completed epoch boundary is resumable.
    assert (run_dir / "checkpoints" / "resume.pt").is_file()


def test_legacy_import_is_explicit_and_produces_named_weights(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _document()
    stack = build_model_stack(document.common)
    source = tmp_path / "legacy.pt"
    torch.save(
        [
            binding.state.detach().cpu().clone()
            for binding in stack.bundle.catalog.checkpointed
        ],
        source,
    )
    output = tmp_path / "named.pt"

    assert (
        import_legacy_checkpoint(
            ImportLegacyCheckpointRequest(
                definition=definition,
                document=document,
                config_path=tmp_path / "config.json",
                source=source,
                output=output,
                kind="full",
                command=("ebl", "checkpoint", "import-legacy"),
            )
        )
        == 0
    )
    payload = _torch_load(output)
    assert payload["schema"] == NAMED_WEIGHTS_SCHEMA
    assert payload["metadata"]["legacy_profile"] == "full"
    assert payload["metadata"]["source_sha256"]


def test_passive_low_rank_training_requires_one_unambiguous_source(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    adapter_spec = definition.resolve(
        _adapter_document(),
        RunMode.TRAIN,
    )

    with pytest.raises(ValueError, match="exactly one"):
        run_train(
            TrainRequest(
                definition=definition,
                spec=adapter_spec,
                config_path=tmp_path / "adapter.json",
                output_dir=tmp_path / "missing",
                weights=None,
                base_weights=None,
                resume=None,
                command=("ebl", "train"),
            )
        )
    assert not (tmp_path / "missing").exists()

    with pytest.raises(ValueError, match="at most one"):
        run_train(
            TrainRequest(
                definition=definition,
                spec=adapter_spec,
                config_path=tmp_path / "adapter.json",
                output_dir=tmp_path / "ambiguous",
                weights=tmp_path / "full.pt",
                base_weights=tmp_path / "base.pt",
                resume=None,
                command=("ebl", "train"),
            )
        )
    assert not (tmp_path / "ambiguous").exists()

    base_spec = definition.resolve(_document(), RunMode.TRAIN)
    with pytest.raises(ValueError, match="only when"):
        run_train(
            TrainRequest(
                definition=definition,
                spec=base_spec,
                config_path=tmp_path / "base.json",
                output_dir=tmp_path / "invalid-base-source",
                weights=None,
                base_weights=tmp_path / "base.pt",
                resume=None,
                command=("ebl", "train"),
            )
        )


@pytest.mark.parametrize("backend", ["direct", "tiki_taka"])
def test_passive_low_rank_tiny_training_freezes_base_and_writes_full_catalog(
    tmp_path: Path,
    backend: str,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _adapter_document(backend=backend)
    spec = definition.resolve(document, RunMode.TRAIN)
    base_path = tmp_path / f"{backend}-base.pt"
    expected_base = _write_adapter_base_weights(document, base_path)
    output_root = tmp_path / backend

    assert (
        run_train(
            TrainRequest(
                definition=definition,
                spec=spec,
                config_path=tmp_path / "adapter.json",
                output_dir=output_root,
                weights=None,
                base_weights=base_path,
                resume=None,
                command=("ebl", "train", "--base-weights"),
            )
        )
        == 0
    )
    weights_path = _runs(output_root)[0] / "checkpoints" / "weights.pt"
    payload = _torch_load(weights_path)
    assert list(payload["weights"]) == [
        "base.dense_weight.0",
        "adapter.input_factor.0",
        "adapter.output_factor.0",
    ]
    assert torch.equal(
        payload["weights"]["base.dense_weight.0"],
        expected_base["base.dense_weight.0"],
    )

    if backend == "direct":
        assert (
            run_train(
                TrainRequest(
                    definition=definition,
                    spec=spec,
                    config_path=tmp_path / "adapter.json",
                    output_dir=tmp_path / "full-source",
                    weights=weights_path,
                    base_weights=None,
                    resume=None,
                    command=("ebl", "train", "--weights"),
                )
            )
            == 0
        )


def test_digital_low_rank_programs_base_once_and_trains_only_readout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _digital_adapter_document()
    spec = definition.resolve(document, RunMode.TRAIN)
    base_path = tmp_path / "clean-base.pt"
    expected_clean = _write_adapter_base_weights(document, base_path)
    calls = []

    def fake_program(catalog, config):
        binding = catalog.by_key["base.dense_weight.0"]
        with torch.no_grad():
            binding.state.add_(0.001)
        calls.append(config.programming_seed)
        return {
            "model": config.type,
            "programming_seed": config.programming_seed,
            "error_rmse": 0.001,
        }

    monkeypatch.setattr(
        "experiments.small_network.deployment."
        "program_wan2022_base_conductance",
        fake_program,
    )
    output_root = tmp_path / "digital"
    run_train(
        TrainRequest(
            definition=definition,
            spec=spec,
            config_path=tmp_path / "digital.json",
            output_dir=output_root,
            weights=None,
            base_weights=base_path,
            resume=None,
            command=("ebl", "train", "--base-weights"),
        )
    )

    run_dir = _runs(output_root)[0]
    payload = _torch_load(run_dir / "checkpoints" / "weights.pt")
    assert list(payload["weights"]) == [
        "base.dense_weight.0",
        "adapter.input_factor.0",
        "adapter.output_factor.0",
    ]
    torch.testing.assert_close(
        payload["weights"]["base.dense_weight.0"],
        expected_clean["base.dense_weight.0"] + 0.001,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(
        payload["weights"]["adapter.output_factor.0"]
    )
    result = _read_json(run_dir / "result.json")
    assert result["metrics"]["device_programming"] == {
        "model": "aihwkit_reram_wan2022",
        "programming_seed": 17,
        "error_rmse": 0.001,
    }
    assert calls == [17]

    run_train(
        TrainRequest(
            definition=definition,
            spec=spec,
            config_path=tmp_path / "digital.json",
            output_dir=tmp_path / "resumed",
            weights=None,
            base_weights=None,
            resume=run_dir / "checkpoints" / "resume.pt",
            command=("ebl", "train", "--resume"),
        )
    )
    assert calls == [17]


@pytest.mark.parametrize("algorithm", ("ep", "backprop"))
def test_passive_layerwise_low_rank_programs_both_edges_once_and_freezes_base(
    tmp_path: Path,
    monkeypatch,
    algorithm: str,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _passive_layerwise_adapter_document(algorithm=algorithm)
    spec = definition.resolve(document, RunMode.TRAIN)
    base_path = tmp_path / "clean-two-edge-base.pt"
    expected_clean = _write_adapter_base_weights(document, base_path)
    calls = []

    def fake_program(catalog, configs):
        keys = tuple(configs)
        for key in keys:
            with torch.no_grad():
                catalog.by_key[key].state.add_(0.001)
        calls.append(
            tuple((key, configs[key].programming_seed) for key in keys)
        )
        return {
            "model": "aihwkit_reram_wan2022",
            "parameter_keys": list(keys),
            "error_rmse": 0.001,
        }

    monkeypatch.setattr(
        "experiments.small_network.deployment."
        "program_wan2022_base_conductances",
        fake_program,
    )
    output_root = tmp_path / "passive-layerwise"
    assert (
        run_train(
            TrainRequest(
                definition=definition,
                spec=spec,
                config_path=tmp_path / "passive-layerwise.json",
                output_dir=output_root,
                weights=None,
                base_weights=base_path,
                resume=None,
                command=("ebl", "train", "--base-weights"),
            )
        )
        == 0
    )

    run_dir = _runs(output_root)[0]
    payload = _torch_load(run_dir / "checkpoints" / "weights.pt")
    assert list(payload["weights"]) == [
        "base.dense_weight.0",
        "base.dense_weight.1",
        "base.bias.0",
        "adapter.input_factor.0",
        "adapter.output_factor.0",
        "adapter.input_factor.1",
        "adapter.output_factor.1",
    ]
    for key in ("base.dense_weight.0", "base.dense_weight.1"):
        torch.testing.assert_close(
            payload["weights"][key],
            expected_clean[key] + 0.001,
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        payload["weights"]["base.bias.0"],
        expected_clean["base.bias.0"],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(
        payload["weights"]["adapter.output_factor.0"]
    )
    assert torch.count_nonzero(
        payload["weights"]["adapter.output_factor.1"]
    )
    result = _read_json(run_dir / "result.json")
    assert result["metrics"]["device_programming"] == {
        "model": "aihwkit_reram_wan2022",
        "parameter_keys": [
            "base.dense_weight.0",
            "base.dense_weight.1",
        ],
        "error_rmse": 0.001,
    }
    assert calls == [
        (
            ("base.dense_weight.0", 17),
            ("base.dense_weight.1", 29),
        )
    ]

    run_train(
        TrainRequest(
            definition=definition,
            spec=spec,
            config_path=tmp_path / "passive-layerwise.json",
            output_dir=tmp_path / "passive-layerwise-resumed",
            weights=None,
            base_weights=None,
            resume=run_dir / "checkpoints" / "resume.pt",
            command=("ebl", "train", "--resume"),
        )
    )
    assert len(calls) == 1


def test_passive_low_rank_legacy_base_import_is_explicit_and_scoped(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _adapter_document()
    stack = build_model_stack(document.common)
    base_bindings = stack.bundle.catalog.for_group(
        "base", checkpointed_only=True
    )
    source = tmp_path / "legacy-base.pt"
    torch.save(
        [binding.state.detach().cpu().clone() for binding in base_bindings],
        source,
    )
    output = tmp_path / "named-base.pt"

    assert (
        import_legacy_checkpoint(
            ImportLegacyCheckpointRequest(
                definition=definition,
                document=document,
                config_path=tmp_path / "adapter.json",
                source=source,
                output=output,
                kind="base",
                command=("ebl", "checkpoint", "import-legacy"),
            )
        )
        == 0
    )
    payload = _torch_load(output)
    assert list(payload["weights"]) == ["base.dense_weight.0"]
    assert payload["metadata"]["legacy_profile"] == "base-only"

    spec = definition.resolve(document, RunMode.TRAIN)
    assert (
        run_train(
            TrainRequest(
                definition=definition,
                spec=spec,
                config_path=tmp_path / "adapter.json",
                output_dir=tmp_path / "from-import",
                weights=None,
                base_weights=output,
                resume=None,
                command=("ebl", "train", "--base-weights"),
            )
        )
        == 0
    )


def test_split_run_restores_carried_equilibrium_state_exactly(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    one_epoch_document = _document()
    two_epoch_document = replace(
        one_epoch_document,
        train=replace(one_epoch_document.train, num_epochs=2),
    )
    one_epoch_spec = definition.resolve(
        one_epoch_document,
        RunMode.TRAIN,
    )
    two_epoch_spec = definition.resolve(
        two_epoch_document,
        RunMode.TRAIN,
    )

    uninterrupted_root = tmp_path / "uninterrupted"
    run_train(
        TrainRequest(
            definition=definition,
            spec=two_epoch_spec,
            config_path=tmp_path / "two.json",
            output_dir=uninterrupted_root,
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    )
    uninterrupted = _torch_load(
        _runs(uninterrupted_root)[0] / "checkpoints" / "resume.pt"
    )

    split_root = tmp_path / "split"
    run_train(
        TrainRequest(
            definition=definition,
            spec=one_epoch_spec,
            config_path=tmp_path / "one.json",
            output_dir=split_root,
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    )
    boundary = _runs(split_root)[0] / "checkpoints" / "resume.pt"
    run_train(
        TrainRequest(
            definition=definition,
            spec=two_epoch_spec,
            config_path=tmp_path / "two.json",
            output_dir=split_root,
            weights=None,
            base_weights=None,
            resume=boundary,
            command=("ebl", "train", "--resume"),
        )
    )
    resumed = _torch_load(
        _runs(split_root)[1] / "checkpoints" / "resume.pt"
    )

    assert uninterrupted["epoch"] == resumed["epoch"] == 2
    assert uninterrupted["global_step"] == resumed["global_step"] == 2
    assert uninterrupted["metadata"]["resume_config_sha256"] == (
        resumed["metadata"]["resume_config_sha256"]
    )
    assert uninterrupted["runtime_state_state"]["layers"].keys() == (
        resumed["runtime_state_state"]["layers"].keys()
    )
    for key, expected in uninterrupted["weights"]["weights"].items():
        torch.testing.assert_close(
            resumed["weights"]["weights"][key],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    for key, expected in uninterrupted["runtime_state_state"]["layers"].items():
        torch.testing.assert_close(
            resumed["runtime_state_state"]["layers"][key],
            expected,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("backend", ["direct", "tiki_taka"])
def test_passive_low_rank_split_run_restores_rank_layer_state_exactly(
    tmp_path: Path,
    backend: str,
) -> None:
    definition = get_definition("small_drn.v1")
    one_epoch_document = _adapter_document(backend=backend)
    two_epoch_document = replace(
        one_epoch_document,
        train=replace(one_epoch_document.train, num_epochs=2),
    )
    one_epoch_spec = definition.resolve(
        one_epoch_document,
        RunMode.TRAIN,
    )
    two_epoch_spec = definition.resolve(
        two_epoch_document,
        RunMode.TRAIN,
    )
    base_path = tmp_path / "base.pt"
    _write_adapter_base_weights(one_epoch_document, base_path)

    uninterrupted_root = tmp_path / "adapter-uninterrupted"
    run_train(
        TrainRequest(
            definition=definition,
            spec=two_epoch_spec,
            config_path=tmp_path / "two.json",
            output_dir=uninterrupted_root,
            weights=None,
            base_weights=base_path,
            resume=None,
            command=("ebl", "train", "--base-weights"),
        )
    )
    uninterrupted = _torch_load(
        _runs(uninterrupted_root)[0] / "checkpoints" / "resume.pt"
    )

    split_root = tmp_path / "adapter-split"
    run_train(
        TrainRequest(
            definition=definition,
            spec=one_epoch_spec,
            config_path=tmp_path / "one.json",
            output_dir=split_root,
            weights=None,
            base_weights=base_path,
            resume=None,
            command=("ebl", "train", "--base-weights"),
        )
    )
    boundary = _runs(split_root)[0] / "checkpoints" / "resume.pt"
    run_train(
        TrainRequest(
            definition=definition,
            spec=two_epoch_spec,
            config_path=tmp_path / "two.json",
            output_dir=split_root,
            weights=None,
            base_weights=None,
            resume=boundary,
            command=("ebl", "train", "--resume"),
        )
    )
    resumed = _torch_load(
        _runs(split_root)[1] / "checkpoints" / "resume.pt"
    )

    assert uninterrupted["epoch"] == resumed["epoch"] == 2
    assert uninterrupted["global_step"] == resumed["global_step"] == 2
    assert list(uninterrupted["weights"]["weights"]) == [
        "base.dense_weight.0",
        "adapter.input_factor.0",
        "adapter.output_factor.0",
    ]
    assert list(uninterrupted["runtime_state_state"]["layers"]) == [
        "layer.0",
        "layer.1",
        "layer.2",
    ]
    for key, expected in uninterrupted["weights"]["weights"].items():
        torch.testing.assert_close(
            resumed["weights"]["weights"][key],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    for key, expected in uninterrupted["runtime_state_state"]["layers"].items():
        torch.testing.assert_close(
            resumed["runtime_state_state"]["layers"][key],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_resume_rejects_numerical_config_changes_and_shorter_horizon(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    document = _document()
    original_spec = definition.resolve(document, RunMode.TRAIN)
    output_root = tmp_path / "original"
    run_train(
        TrainRequest(
            definition=definition,
            spec=original_spec,
            config_path=tmp_path / "original.json",
            output_dir=output_root,
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    )
    boundary = _runs(output_root)[0] / "checkpoints" / "resume.pt"

    changed_document = replace(
        document,
        train=replace(document.train, nudging=0.125, num_epochs=2),
    )
    changed_spec = definition.resolve(changed_document, RunMode.TRAIN)
    with pytest.raises(ValueError, match="numerical configuration"):
        run_train(
            TrainRequest(
                definition=definition,
                spec=changed_spec,
                config_path=tmp_path / "changed.json",
                output_dir=tmp_path / "changed",
                weights=None,
                base_weights=None,
                resume=boundary,
                command=("ebl", "train", "--resume"),
            )
        )

    payload = _torch_load(boundary)
    payload["epoch"] = 2
    future_boundary = tmp_path / "future-resume.pt"
    torch.save(payload, future_boundary)
    with pytest.raises(ValueError, match="greater than or equal"):
        run_train(
            TrainRequest(
                definition=definition,
                spec=original_spec,
                config_path=tmp_path / "shorter.json",
                output_dir=tmp_path / "shorter",
                weights=None,
                base_weights=None,
                resume=future_boundary,
                command=("ebl", "train", "--resume"),
            )
        )


def test_hardware_aware_training_runs_and_records_noise_provenance(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    spec = definition.resolve(_hardware_aware_document(), RunMode.TRAIN)

    assert run_train(
        TrainRequest(
            definition=definition,
            spec=spec,
            config_path=tmp_path / "config.json",
            output_dir=tmp_path / "runs",
            weights=None,
            base_weights=None,
            resume=None,
            command=("ebl", "train"),
        )
    ) == 0

    run_dir = _runs(tmp_path / "runs")[0]
    assert _read_json(run_dir / "status.json")["status"] == "complete"
    metrics = _read_json(run_dir / "result.json")["metrics"]
    modifier = metrics["evaluation_protocol"]["parameter_modifier"]
    assert modifier["type"] == "add_normal"
    assert modifier["resolved_seed"] == 17
    assert modifier["noisy_evaluation_executed"] is True
    assert metrics["last_noisy_validation"] is not None


def test_unimplemented_extension_fails_before_model_construction(
    tmp_path: Path,
) -> None:
    definition = get_definition("small_drn.v1")
    spec = definition.resolve(_document(), RunMode.TRAIN)
    unsupported = replace(
        spec,
        settings=replace(
            spec.settings,
            weight_modifier=ComponentSettings(
                type="future_modifier",
                parameters={},
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="hardware-aware branch"):
        run_train(
            TrainRequest(
                definition=definition,
                spec=unsupported,
                config_path=tmp_path / "config.json",
                output_dir=tmp_path / "runs",
                weights=None,
                base_weights=None,
                resume=None,
                command=("ebl", "train"),
            )
        )
    failed_run = _runs(tmp_path / "runs")[0]
    assert _read_json(failed_run / "status.json")["status"] == "failed"


def test_float64_stack_keeps_configured_dtype_across_network_reset() -> None:
    document = _document()
    common = replace(
        document.common,
        runtime=replace(document.common.runtime, dtype="float64"),
    )
    stack = build_model_stack(common)
    assert {
        binding.state.dtype for binding in stack.bundle.catalog.all
    } == {torch.float64}

    inputs = torch.zeros((3, stack.logical_input_dim), dtype=torch.float32)
    stack.network.set_input(inputs, reset=True)
    assert {layer.state.dtype for layer in stack.network.layers()} == {
        torch.float64
    }
