from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from types import ModuleType

from ebl.cli import (
    CampaignRunRequest,
    CommandHandlers,
    ImportLegacyCheckpointRequest,
    TrainRequest,
    ValidateRequest,
    build_parser,
    main,
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
                "quadratic_diode_param": {"diode_conductance": 10.0},
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
    assert "small_drn.v1" in stdout.getvalue()

    stdout = io.StringIO()
    assert main(
        [
            "describe",
            "--experiment",
            "small_drn.v1",
            "--json",
        ],
        stdout=stdout,
    ) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["protocol_version"] == 1
    assert payload["experiment_id"] == "small_drn.v1"
    assert payload["supported_modes"] == ["train", "linspace", "validate"]
    assert payload["capabilities"]["resume"]["full_training_state"] is True
    assert payload["capabilities"]["campaign"]["dry_run"] is True
    assert payload["commands"]["train"]["optional_input_options"] == [
        "--device-data",
        "--teacher-weights",
        "--deployment",
    ]
    assert payload["commands"]["validate"]["optional_input_options"] == [
        "--teacher-weights"
    ]
    assert "checkpoint import-legacy" in payload["commands"]
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
            "--base-weights",
            "base.pt",
        ],
        handlers=CommandHandlers(train=train),
    )

    assert result == 7
    assert len(seen) == 1
    assert seen[0].definition.experiment_id == "small_drn.v1"
    assert seen[0].spec.settings.num_epochs == 2
    assert seen[0].base_weights == Path("base.pt")
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
            "--teacher-weights",
            str(teacher_weights),
        ],
        handlers=CommandHandlers(validate=validate),
    ) == 0

    assert train_requests[0].device_data == device_data
    assert train_requests[0].teacher_weights == teacher_weights
    assert "--device-data" in train_requests[0].command
    assert "--teacher-weights" in train_requests[0].command
    assert validate_requests[0].teacher_weights == teacher_weights
    assert "--teacher-weights" in validate_requests[0].command


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
    runtime_module = "ebl.runtime.small_network.runtime"
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


def test_default_handlers_lazily_dispatch_all_numerical_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_config(tmp_path)
    runtime = ModuleType("ebl.runtime.small_network.runtime")
    seen = []

    def handler(command_name: str, result: int):
        def invoke(request):
            seen.append((command_name, request))
            return result

        return invoke

    runtime.run_train = handler("train", 11)
    runtime.run_linspace = handler("linspace", 12)
    runtime.run_validate = handler("validate", 13)
    runtime.import_legacy_checkpoint = handler(
        "checkpoint import-legacy",
        14,
    )
    monkeypatch.setitem(
        sys.modules,
        "ebl.runtime.small_network.runtime",
        runtime,
    )

    cases = (
        (
            [
                "train",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "train-runs"),
            ],
            11,
        ),
        (
            [
                "linspace",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "linspace-runs"),
                "--weights",
                "weights.pt",
            ],
            12,
        ),
        (
            [
                "validate",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "validate-runs"),
                "--weights",
                "weights.pt",
            ],
            13,
        ),
        (
            [
                "checkpoint",
                "import-legacy",
                "--config",
                str(config_path),
                "--source",
                "legacy.pt",
                "--output",
                "weights.ebl.pt",
            ],
            14,
        ),
    )
    for argv, expected in cases:
        assert main(argv) == expected

    assert [command_name for command_name, _request in seen] == [
        "train",
        "linspace",
        "validate",
        "checkpoint import-legacy",
    ]
    assert isinstance(seen[0][1], TrainRequest)
    assert isinstance(seen[3][1], ImportLegacyCheckpointRequest)


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


def test_checkpoint_import_legacy_has_an_injectable_dispatch(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    seen = []

    def import_checkpoint(request: ImportLegacyCheckpointRequest) -> None:
        seen.append(request)

    result = main(
        [
            "checkpoint",
            "import-legacy",
            "--config",
            str(config_path),
            "--source",
            "legacy.pt",
            "--output",
            "weights.ebl.pt",
            "--kind",
            "base",
        ],
        handlers=CommandHandlers(
            checkpoint_import_legacy=import_checkpoint
        ),
    )

    assert result == 0
    assert seen[0].kind == "base"
    assert seen[0].source == Path("legacy.pt")
    assert seen[0].document.experiment_id == "small_drn.v1"


def test_parser_exposes_all_required_command_surfaces() -> None:
    parser = build_parser()
    for argv in (
        ["describe"],
        ["train", "--config", "c.json", "--output-dir", "runs"],
        [
            "linspace",
            "--config",
            "c.json",
            "--output-dir",
            "runs",
            "--weights",
            "w.pt",
        ],
        [
            "validate",
            "--config",
            "c.json",
            "--output-dir",
            "runs",
            "--weights",
            "w.pt",
        ],
        [
            "checkpoint",
            "import-legacy",
            "--config",
            "c.json",
            "--source",
            "old.pt",
            "--output",
            "new.pt",
        ],
        [
            "campaign",
            "run",
            "--manifest",
            "campaign.json",
            "--output-dir",
            "campaign-runs",
        ],
    ):
        parser.parse_args(argv)


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
    import ebl.execution.runner

    manifest = _write_campaign_manifest(tmp_path)
    seen = []

    def run_campaign(spec, **options):
        seen.append((spec, options))
        return {"base_train": {"status": "dry_run"}}

    monkeypatch.setattr(ebl.execution.runner, "run_campaign", run_campaign)
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
