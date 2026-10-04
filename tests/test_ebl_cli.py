from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from types import ModuleType

from ebl.cli import (
    CampaignRunRequest,
    CommandHandlers,
    TrainRequest,
    ValidateRequest,
    build_parser,
    main,
)


def _config() -> dict:
    return json.loads((Path(__file__).resolve().parents[1] / "examples/mnist_relu/teacher.json").read_text())


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    return path


def _write_campaign_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "comparison",
                "targets": [
                    {
                        "id": "base",
                        "worktree": "base-worktree",
                        "python": ".venv/bin/python",
                    }
                ],
                "stages": [
                    {
                        "id": "base_train",
                        "case_id": "digits",
                        "target": "base",
                        "command": "train",
                        "config": "train.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_describe_lists_and_details_stable_experiment() -> None:
    stdout = io.StringIO()
    assert main(["describe"], stdout=stdout) == 0
    assert "mnist_relu.v1" in stdout.getvalue()

    stdout = io.StringIO()
    assert main(
        [
            "describe",
            "--experiment",
            "mnist_relu.v1",
            "--json",
        ],
        stdout=stdout,
    ) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["protocol_version"] == 1
    assert payload["experiment_id"] == "mnist_relu.v1"
    assert payload["supported_modes"] == ["train", "validate"]
    assert payload["capabilities"]["resume"]["full_training_state"] is True
    assert payload["capabilities"]["campaign"]["dry_run"] is True
    assert payload["commands"]["train"]["optional_input_options"] == []
    assert payload["commands"]["validate"]["optional_input_options"] == []
    assert "checkpoint import-legacy" not in payload["commands"]
    assert "campaign run" in payload["commands"]


def test_train_dispatches_an_immutable_resolved_request(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    seen = []

    def train(request: TrainRequest) -> int:
        seen.append(request)
        return 7

    result = main(
        [
            "train",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--weights",
            "base.pt",
        ],
        handlers=CommandHandlers(train=train),
    )

    assert result == 7
    assert len(seen) == 1
    assert seen[0].definition.experiment_id == "mnist_relu.v1"
    assert seen[0].spec.settings.num_epochs == 30
    assert seen[0].weights == Path("base.pt")
    assert seen[0].command[0:2] == ("ebl", "train")


def test_train_and_validate_forward_generic_external_inputs(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    train_requests: list[TrainRequest] = []
    validate_requests: list[ValidateRequest] = []
    device_data = tmp_path / "devices.hdf5"
    teacher_weights = tmp_path / "teacher.pt"

    def train(request: TrainRequest) -> int:
        train_requests.append(request)
        return 0

    def validate(request: ValidateRequest) -> int:
        validate_requests.append(request)
        return 0

    assert main(
        [
            "train",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "train-runs"),
            "--device-data",
            str(device_data),
            "--teacher-weights",
            str(teacher_weights),
        ],
        handlers=CommandHandlers(train=train),
    ) == 0
    assert main(
        [
            "validate",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "validate-runs"),
            "--weights",
            str(tmp_path / "weights.pt"),
        ],
        handlers=CommandHandlers(validate=validate),
    ) == 0

    assert train_requests[0].device_data == device_data
    assert train_requests[0].teacher_weights == teacher_weights
    assert "--device-data" in train_requests[0].command
    assert "--teacher-weights" in train_requests[0].command
    assert validate_requests[0].weights == tmp_path / "weights.pt"


def test_missing_runtime_handler_is_clear_and_post_validation(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    stderr = io.StringIO()

    result = main(
        [
            "validate",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--weights",
            "weights.pt",
        ],
        handlers=CommandHandlers(),
        stderr=stderr,
    )

    assert result == 2
    message = stderr.getvalue()
    assert "Expected a connected execution handler for 'validate'" in message
    assert "configuration validation succeeded" in message


def test_default_handlers_do_not_import_runtime_before_config_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_module = "experiments.mnist_relu.runtime"
    monkeypatch.delitem(sys.modules, runtime_module, raising=False)
    config_path = tmp_path / "invalid.json"
    config_path.write_text('{"schema_version": 1,', encoding="utf-8")
    stderr = io.StringIO()

    assert main(
        [
            "train",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
        ],
        stderr=stderr,
    ) == 2

    assert "Expected --config to contain valid JSON" in stderr.getvalue()
    assert runtime_module not in sys.modules


def test_default_handlers_lazily_dispatch_all_numerical_commands(tmp_path, monkeypatch):
    config = _write_config(tmp_path)
    runtime = ModuleType("experiments.mnist_relu.runtime")
    seen = []
    runtime.run_train = lambda request: seen.append(request) or 11
    runtime.run_validate = lambda request: seen.append(request) or 13
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    for mode, code in (("train", 11), ("validate", 13)):
        args = [mode, "--config", str(config), "--output-dir", str(tmp_path / mode)]
        if mode == "validate": args += ["--weights", "weights.pt"]
        assert main(args) == code
    assert isinstance(seen[0], TrainRequest)
    assert isinstance(seen[1], ValidateRequest)


def test_scientific_cli_override_is_not_part_of_public_surface(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    stderr = io.StringIO()
    result = main(
        [
            "train",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--num-epochs",
            "20",
        ],
        stderr=stderr,
    )
    assert result == 2
    assert "unrecognized arguments: --num-epochs 20" in stderr.getvalue()


def test_parser_exposes_all_required_command_surfaces():
    parser = build_parser()
    for args in (
        ["describe"],
        ["train", "--config", "c.json", "--output-dir", "runs"],
        ["validate", "--config", "c.json", "--output-dir", "runs", "--weights", "w.pt"],
        ["characterize", "--config", "c.json", "--output-dir", "runs"],
        ["campaign", "run", "--manifest", "campaign.json", "--output-dir", "runs"],
        ["runs", "inspect", "runs/example"],
    ): parser.parse_args(args)


def test_campaign_dispatches_strict_manifest_and_operational_flags(
    tmp_path: Path,
) -> None:
    manifest = _write_campaign_manifest(tmp_path)
    seen = []

    def campaign(request: CampaignRunRequest) -> int:
        seen.append(request)
        return 9

    result = main(
        [
            "campaign",
            "run",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--resume",
            "--dry-run",
            "--allow-dirty",
            "--fail-fast",
        ],
        handlers=CommandHandlers(campaign_run=campaign),
    )

    assert result == 9
    assert len(seen) == 1
    request = seen[0]
    assert request.spec.campaign_id == "comparison"
    assert request.manifest_path == manifest.resolve()
    assert request.output_dir == tmp_path / "outputs"
    assert request.resume is True
    assert request.dry_run is True
    assert request.allow_dirty is True
    assert request.fail_fast is True


def test_campaign_continue_policy_is_explicit_and_default(
    tmp_path: Path,
) -> None:
    manifest = _write_campaign_manifest(tmp_path)
    seen = []

    def campaign(request: CampaignRunRequest) -> None:
        seen.append(request)

    base = [
        "campaign",
        "run",
        "--manifest",
        str(manifest),
        "--output-dir",
        str(tmp_path / "outputs"),
    ]
    assert main(
        base,
        handlers=CommandHandlers(campaign_run=campaign),
    ) == 0
    assert seen[-1].fail_fast is False
    assert main(
        base + ["--continue"],
        handlers=CommandHandlers(campaign_run=campaign),
    ) == 0
    assert seen[-1].fail_fast is False


def test_campaign_failure_policies_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    manifest = _write_campaign_manifest(tmp_path)
    stderr = io.StringIO()
    result = main(
        [
            "campaign",
            "run",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--fail-fast",
            "--continue-on-error",
        ],
        handlers=CommandHandlers(campaign_run=lambda _request: 0),
        stderr=stderr,
    )
    assert result == 2
    assert "not allowed with argument --fail-fast" in stderr.getvalue()


def test_campaign_rejects_invalid_json_before_injected_handler(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "invalid.json"
    manifest.write_text('{"schema_version": 1,', encoding="utf-8")
    called = []
    stderr = io.StringIO()

    result = main(
        [
            "campaign",
            "run",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
        handlers=CommandHandlers(
            campaign_run=lambda request: called.append(request)
        ),
        stderr=stderr,
    )

    assert result == 2
    assert called == []
    assert "Expected --manifest to contain valid strict JSON" in (
        stderr.getvalue()
    )


def test_campaign_uses_lazy_default_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import campaigns.runner

    manifest = _write_campaign_manifest(tmp_path)
    seen = []

    def run_campaign(spec, **options):
        seen.append((spec, options))
        return {"base_train": {"status": "dry_run"}}

    monkeypatch.setattr(campaigns.runner, "run_campaign", run_campaign)
    result = main(
        [
            "campaign",
            "run",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--dry-run",
        ]
    )

    assert result == 0
    assert seen[0][0].campaign_id == "comparison"
    assert seen[0][1]["output_root"] == tmp_path / "outputs"
    assert seen[0][1]["manifest_path"] == manifest.resolve()
    assert seen[0][1]["dry_run"] is True
