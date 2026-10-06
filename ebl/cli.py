"""Public ``ebl`` command line.

This module owns argument parsing and experiment selection.  Numerical
execution is connected lazily after configuration validation, while
:class:`CommandHandlers` keeps the execution boundary explicitly injectable
for embedding and tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Sequence,
    TextIO,
    Tuple,
)

from experiments.definitions import (
    get_definition,
    list_definitions,
    load_experiment_config,
    resolve_experiment_config,
)
from experiments.schema import (
    ConfigError,
    ExperimentDefinition,
    RunMode,
    to_plain_data,
)

if TYPE_CHECKING:
    from campaigns.schema import CampaignSpec


class CliUsageError(ValueError):
    """Raised for command-line syntax errors without terminating the process."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(
            "Expected command-line arguments matching this command's help. "
            f"Provided value: {message}."
        )


@dataclass(frozen=True)
class TrainRequest:
    definition: ExperimentDefinition
    spec: Any
    config_path: Path
    output_dir: Path
    weights: Optional[Path]
    resume: Optional[Path]
    command: Tuple[str, ...]
    device_data: Optional[Path] = None
    device_model: Optional[Path] = None
    teacher_weights: Optional[Path] = None
    device_state: Optional[Path] = None
    selection_receipt: Optional[Path] = None


@dataclass(frozen=True)
class ValidateRequest:
    definition: ExperimentDefinition
    spec: Any
    config_path: Path
    output_dir: Path
    weights: Path
    command: Tuple[str, ...]


@dataclass(frozen=True)
class CharacterizeRequest:
    definition: ExperimentDefinition
    spec: Any
    config_path: Path
    output_dir: Path
    command: Tuple[str, ...]


@dataclass(frozen=True)
class CampaignRunRequest:
    spec: "CampaignSpec"
    manifest_path: Path
    output_dir: Path
    resume: bool
    dry_run: bool
    allow_dirty: bool
    fail_fast: bool
    command: Tuple[str, ...]


Handler = Callable[[Any], Optional[int]]


@dataclass(frozen=True)
class CommandHandlers:
    """Optional execution handlers connected by the application layer."""

    train: Optional[Handler] = None
    validate: Optional[Handler] = None
    characterize: Optional[Handler] = None
    campaign_run: Optional[Handler] = None


def build_parser() -> argparse.ArgumentParser:
    """Construct the stable public parser without importing numerical code."""

    parser = _ArgumentParser(
        prog="ebl",
        description=(
            "Run versioned EBL experiments from strict JSON configuration."
        ),
    )
    commands = parser.add_subparsers(
        dest="command_name",
        required=True,
        metavar="COMMAND",
    )

    describe = commands.add_parser(
        "describe",
        help="list experiment IDs or describe one registered definition",
    )
    describe.add_argument(
        "--experiment",
        dest="experiment_id",
        help="stable registered ID, for example mnist_ibm_om_crossbar_relu.v2",
    )
    describe.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable definition metadata",
    )

    train = commands.add_parser(
        "train",
        help="train the experiment selected by the config file",
    )
    _add_config_and_output(train)
    initial = train.add_mutually_exclusive_group()
    initial.add_argument(
        "--weights",
        type=Path,
        help="initialize the complete model from a weights artifact",
    )
    initial.add_argument(
        "--resume",
        type=Path,
        help="resume from a full training-state checkpoint",
    )
    train.add_argument(
        "--device-data",
        type=Path,
        help=(
            "explicit device data or cached activations required by the "
            "selected experiment"
        ),
    )
    train.add_argument(
        "--device-model",
        type=Path,
        help=(
            "explicit fitted device model or sampled population bundle used by "
            "crossbar HWA and recovery"
        ),
    )
    train.add_argument(
        "--teacher-weights",
        type=Path,
        help="explicit named ReLU teacher checkpoint for distillation",
    )
    train.add_argument(
        "--device-state",
        type=Path,
        help=(
            "explicit deployed or faulted physical-device state bundle for "
            "a staged experiment"
        ),
    )
    train.add_argument(
        "--selection-receipt",
        type=Path,
        help=(
            "explicit frozen hyperparameter-selection receipt for a staged "
            "experiment"
        ),
    )


    validate = commands.add_parser(
        "validate",
        help="validate a trained checkpoint on a configured dataset split",
    )
    _add_config_and_output(validate)
    validate.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="explicit weights artifact to validate",
    )
    characterize = commands.add_parser(
        "characterize",
        help="characterize a physical-device model selected by the config file",
    )
    _add_config_and_output(characterize)


    campaign = commands.add_parser(
        "campaign",
        help="run a strict cross-worktree experiment campaign",
    )
    campaign_commands = campaign.add_subparsers(
        dest="campaign_command",
        required=True,
        metavar="CAMPAIGN_COMMAND",
    )
    campaign_run = campaign_commands.add_parser(
        "run",
        help="execute stages from a versioned campaign manifest",
    )
    campaign_run.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="strict JSON campaign manifest",
    )
    campaign_run.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="explicit operational root for campaign results",
    )
    campaign_run.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete stages with an identical fingerprint",
    )
    campaign_run.add_argument(
        "--dry-run",
        action="store_true",
        help="preflight and write plans without launching stage processes",
    )
    campaign_run.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow targets with uncommitted worktree changes",
    )
    failure_policy = campaign_run.add_mutually_exclusive_group()
    failure_policy.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action="store_true",
        help="stop launching independent stages after the first failure",
    )
    failure_policy.add_argument(
        "--continue-on-error",
        "--continue",
        dest="fail_fast",
        action="store_false",
        help="continue launching independent stages after failures (default)",
    )
    campaign_run.set_defaults(fail_fast=False)

    study = commands.add_parser(
        "study",
        help="prepare, summarize, and finalize lightweight result studies",
    )
    study_commands = study.add_subparsers(
        dest="study_command",
        required=True,
        metavar="STUDY_COMMAND",
    )
    study_prepare = study_commands.add_parser(
        "prepare",
        help="materialize a tracked study plan below the results root",
    )
    study_prepare.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="strict tracked JSON plan containing the initial hypothesis",
    )
    study_prepare.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="parent for results/<study-id> (default: results)",
    )

    study_summarize = study_commands.add_parser(
        "summarize",
        help="validate run metadata and write a compact study handoff",
    )
    study_summarize.add_argument(
        "--study-dir",
        type=Path,
        required=True,
        help="prepared results/<study-id> directory",
    )
    study_summarize.add_argument(
        "--verify-artifacts",
        action="store_true",
        help="also hash tensor and analysis artifacts (slower, opt-in)",
    )
    study_summarize.add_argument(
        "--json",
        action="store_true",
        help="emit the complete summary JSON to stdout",
    )

    study_finalize = study_commands.add_parser(
        "finalize",
        help="write a local receipt for a reviewed complete legacy study",
    )
    study_finalize.add_argument(
        "--study-dir",
        type=Path,
        required=True,
        help="prepared results/<study-id> directory",
    )
    study_finalize.add_argument(
        "--review",
        type=Path,
        required=True,
        help="strict JSON review with outcome and final interpretation",
    )
    study_finalize.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="deprecated and ignored; historical manifests are never modified",
    )
    study_finalize.add_argument(
        "--verify-artifacts",
        action="store_true",
        help="hash every declared artifact before finalizing",
    )
    runs = commands.add_parser("runs", help="read-only native run-bundle inspection")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    inspect = run_commands.add_parser("inspect", help="check explicit bundles; no study plan required")
    inspect.add_argument("run_dirs", type=Path, nargs="+")
    inspect.add_argument("--verify-artifacts", action="store_true", help="also hash declared artifacts")
    inspect.add_argument("--require-complete", action="store_true", help="also fail for failed/running cases")
    inspect.add_argument("--json", action="store_true", help="emit evidence records including terminal metrics")
    return parser


def _add_config_and_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="strict JSON experiment document; it selects the experiment ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="operational root for the new run directory",
    )


def _definition_payload(definition: ExperimentDefinition) -> dict:
    return {
        "experiment_id": definition.experiment_id,
        "schema_version": definition.schema_version,
        "description": definition.description,
        "supported_modes": [
            mode.value for mode in definition.supported_modes
        ],
        "legacy_names": list(definition.legacy_names),
        "combinations": [
            to_plain_data(combination)
            for combination in definition.combinations
        ],
    }


def _protocol_payload(definitions: Sequence[ExperimentDefinition]) -> dict:
    combinations = [item for definition in definitions for item in definition.combinations]
    ids = {definition.experiment_id for definition in definitions}
    train_inputs = []
    if any(name.startswith("cifar_") for name in ids) or "crossbar_lifecycle.v1" in ids:
        train_inputs.extend(["--device-data", "--device-model", "--teacher-weights", "--device-state", "--selection-receipt"])
    elif "mnist_ibm_om_crossbar_relu.v2" in ids:
        train_inputs.extend(["--teacher-weights", "--device-state", "--selection-receipt"])
    return {
        "protocol_version": 1,
        "capabilities": {
            "supported_commands": [
                "describe", "train", "validate", "characterize", "campaign run",
                "runs inspect", "study prepare", "study summarize", "study finalize",
            ],
            "config_selects_experiment": True,
            "strict_json_config": True,
            "immutable_mode_specs": True,
            "extensions": {
                name: sorted({getattr(item.selection, name) for item in combinations})
                for name in ("model_adapter", "weight_modifier", "update_backend", "algorithm")
            } | {"unlisted_combinations": "rejected"},
            "resume": {
                "weights": True,
                "full_training_state": True,
                "campaign_stage_reuse": True,
            },
            "campaign": {
                "schema_version": 1, "dry_run": True,
                "stage_artifact_references": True, "allow_dirty": True,
                "failure_policies": ["continue", "fail_fast"],
            },
        },
        "commands": {
            "describe": {"available": True, "required_options": []},
            "train": {
                "available": any(RunMode.TRAIN in item.supported_modes for item in definitions),
                "required_options": ["--config", "--output-dir"],
                "exclusive_input_options": ["--weights", "--resume"],
                "optional_input_options": train_inputs,
            },
            "validate": {
                "available": any(RunMode.VALIDATE in item.supported_modes for item in definitions),
                "required_options": ["--config", "--output-dir", "--weights"],
                "optional_input_options": [],
            },
            "characterize": {
                "available": any(RunMode.CHARACTERIZE in item.supported_modes for item in definitions),
                "required_options": ["--config", "--output-dir"],
            },
            "campaign run": {
                "available": True,
                "required_options": [
                    "--manifest",
                    "--output-dir",
                ],
                "optional_flags": [
                    "--resume",
                    "--dry-run",
                    "--allow-dirty",
                    "--fail-fast",
                    "--continue-on-error",
                ],
            },
            "runs inspect": {
                "available": True,
                "positional_arguments": ["RUN_DIR", "..."],
                "optional_flags": ["--verify-artifacts", "--require-complete", "--json"],
            },
            "study prepare": {
                "available": True,
                "required_options": ["--plan"],
                "optional_options": ["--results-root"],
            },
            "study summarize": {
                "available": True,
                "required_options": ["--study-dir"],
                "optional_flags": ["--verify-artifacts", "--json"],
            },
            "study finalize": {
                "available": True,
                "required_options": ["--study-dir", "--review"],
                "optional_options": ["--manifest"],
                "optional_flags": ["--verify-artifacts"],
            },
        },
    }


def _describe(
    experiment_id: Optional[str],
    *,
    json_output: bool,
    stdout: TextIO,
) -> int:
    if experiment_id is None:
        definitions = list_definitions()
        if json_output:
            payload = _protocol_payload(definitions)
            payload["experiments"] = [
                _definition_payload(item) for item in definitions
            ]
            json.dump(payload, stdout, indent=2, sort_keys=True)
            stdout.write("\n")
        else:
            for definition in definitions:
                modes = ", ".join(
                    mode.value for mode in definition.supported_modes
                )
                stdout.write(
                    f"{definition.experiment_id}\t{modes}\t"
                    f"{definition.description}\n"
                )
        return 0

    definition = get_definition(experiment_id)
    definition_payload = _definition_payload(definition)
    if json_output:
        payload = _protocol_payload((definition,))
        payload.update(definition_payload)
        json.dump(payload, stdout, indent=2, sort_keys=True)
        stdout.write("\n")
        return 0

    stdout.write(f"{definition_payload['experiment_id']}\n")
    stdout.write(
        f"  schema version: {definition_payload['schema_version']}\n"
    )
    stdout.write(
        "  modes: "
        + ", ".join(definition_payload["supported_modes"])
        + "\n"
    )
    stdout.write(f"  {definition_payload['description']}\n")
    if definition_payload["legacy_names"]:
        stdout.write(
            "  legacy names: "
            + ", ".join(definition_payload["legacy_names"])
            + "\n"
        )
    stdout.write("  explicit train combinations:\n")
    for combination in definition.combinations:
        selection = combination.selection
        stdout.write(
            "    "
            f"{selection.model_adapter}/"
            f"{selection.weight_modifier}/"
            f"{selection.update_backend}/"
            f"{selection.algorithm}: "
            f"{combination.status}\n"
        )
    return 0


def _require_handler(
    handler: Optional[Handler],
    command_name: str,
) -> Handler:
    if handler is None:
        raise ConfigError(
            f"Expected a connected execution handler for {command_name!r}. "
            "Provided value: None. Argument parsing and configuration "
            "validation succeeded, but the numerical runtime is not "
            "connected yet."
        )
    return handler


def _handler_result(handler: Handler, request: Any) -> int:
    result = handler(request)
    if result is None:
        return 0
    if isinstance(result, bool) or not isinstance(result, int):
        raise ConfigError(
            "Expected an execution handler to return an integer exit code or "
            f"None. Provided value: {result!r}."
        )
    return result


def _default_train_handler(request: TrainRequest) -> Optional[int]:
    if request.definition.experiment_id == "cifar10_ibm_om_crossbar.v1":
        from experiments.cifar10_crossbar.runtime import run_train
        return run_train(request)
    if request.definition.experiment_id == "crossbar_lifecycle.v1":
        from workflow.runtime import run_train
    elif request.definition.experiment_id == "cifar_om_closed_loop_lr.v1":
        from experiments.cifar_crossbar.closed_loop_lr_runtime import run_train
    elif request.definition.experiment_id == "cifar_om_open_loop.v1":
        from experiments.cifar_crossbar.om_open_loop_runtime import run_train
    elif request.definition.experiment_id == "cifar_crossbar_fault_sweep.v1":
        from experiments.cifar_crossbar.sweep_runtime import run_train
    elif request.definition.experiment_id == "cifar_crossbar_full_epochs.v1":
        from experiments.cifar_crossbar.full_epoch_runtime import run_train
    elif request.definition.experiment_id == "cifar_pcm_recovery_epochs.v1":
        from experiments.cifar_crossbar.epoch_runtime import run_train
    elif request.definition.experiment_id == "cifar_pcm_hwa_comparison.v1":
        from experiments.cifar_crossbar.hwa_fault_runtime import run_train
    elif request.definition.experiment_id == "cifar_pcm_fault_recovery.v1":
        from experiments.cifar_crossbar.fault_runtime import run_train
    elif request.definition.experiment_id == "cifar_resnet_suffix_recovery.v1":
        from experiments.cifar_crossbar.runtime import run_train
    elif request.definition.experiment_id in {"mnist_relu.v1", "mnist_relu.v2"}:
        from experiments.mnist_relu.runtime import run_train
    elif request.definition.experiment_id == "mnist_ibm_om_crossbar_relu.v2":
        from experiments.mnist_analog_relu.staged_runtime import run_train
    else:  # pragma: no cover - registry and handler map change together
        raise ConfigError(
            "Expected train runtime dispatch for a registered experiment. "
            f"Provided value: {request.definition.experiment_id!r}."
        )

    return run_train(request)


def _default_validate_handler(request: ValidateRequest) -> Optional[int]:
    if request.definition.experiment_id not in {"mnist_relu.v1", "mnist_relu.v2"}:
        raise ConfigError(
            "Expected validate runtime dispatch for a digital MNIST teacher. "
            f"Provided value: {request.definition.experiment_id!r}."
        )
    from experiments.mnist_relu.runtime import run_validate

    return run_validate(request)


def _default_characterize_handler(
    request: CharacterizeRequest,
) -> Optional[int]:
    if request.definition.experiment_id != "ibm_reram_program_verify.v1":
        raise ConfigError(
            "Expected characterize runtime dispatch only for "
            "'ibm_reram_program_verify.v1'. Provided value: "
            f"{request.definition.experiment_id!r}."
        )
    from experiments.reram_program_verify.runtime import run_characterize

    return run_characterize(request)


def _default_command_handlers() -> CommandHandlers:
    """Build handlers without importing the numerical runtime."""

    return CommandHandlers(
        train=_default_train_handler,
        validate=_default_validate_handler,
        characterize=_default_characterize_handler,
    )


def _default_campaign_handler(request: CampaignRunRequest) -> int:
    """Lazily execute a campaign without connecting numerical handlers."""

    from campaigns.runner import run_campaign

    try:
        records = run_campaign(
            request.spec,
            output_root=request.output_dir,
            manifest_path=request.manifest_path,
            resume=request.resume,
            dry_run=request.dry_run,
            allow_dirty=request.allow_dirty,
            fail_fast=request.fail_fast,
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        raise ConfigError(
            "Expected campaign preflight and execution to complete "
            "successfully. "
            f"Provided value: {type(error).__name__}: {error}."
        ) from error
    return (
        1
        if any(record.get("status") == "failed" for record in records.values())
        else 0
    )


def _dispatch(
    args: argparse.Namespace,
    *,
    handlers: CommandHandlers,
    command: Tuple[str, ...],
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if args.command_name == "describe":
        return _describe(
            args.experiment_id,
            json_output=args.json,
            stdout=stdout,
        )

    if args.command_name == "train":
        definition, spec = resolve_experiment_config(
            args.config,
            RunMode.TRAIN,
        )
        request = TrainRequest(
            definition=definition,
            spec=spec,
            config_path=args.config,
            output_dir=args.output_dir,
            weights=args.weights,
            resume=args.resume,
            command=command,
            device_data=args.device_data,
            device_model=args.device_model,
            teacher_weights=args.teacher_weights,
            device_state=args.device_state,
            selection_receipt=args.selection_receipt,
        )
        return _handler_result(
            _require_handler(handlers.train, "train"),
            request,
        )


    if args.command_name == "validate":
        definition, spec = resolve_experiment_config(
            args.config,
            RunMode.VALIDATE,
        )
        request = ValidateRequest(
            definition=definition,
            spec=spec,
            config_path=args.config,
            output_dir=args.output_dir,
            weights=args.weights,
            command=command,
        )
        return _handler_result(
            _require_handler(handlers.validate, "validate"),
            request,
        )

    if args.command_name == "characterize":
        definition, spec = resolve_experiment_config(
            args.config,
            RunMode.CHARACTERIZE,
        )
        request = CharacterizeRequest(
            definition=definition,
            spec=spec,
            config_path=args.config,
            output_dir=args.output_dir,
            command=command,
        )
        return _handler_result(
            _require_handler(handlers.characterize, "characterize"),
            request,
        )


    if (
        args.command_name == "campaign"
        and args.campaign_command == "run"
    ):
        from campaigns.schema import load_campaign_manifest

        try:
            spec = load_campaign_manifest(args.manifest)
        except ValueError as error:
            raise ConfigError(str(error)) from error
        request = CampaignRunRequest(
            spec=spec,
            manifest_path=args.manifest.expanduser().resolve(),
            output_dir=args.output_dir,
            resume=args.resume,
            dry_run=args.dry_run,
            allow_dirty=args.allow_dirty,
            fail_fast=args.fail_fast,
            command=command,
        )
        return _handler_result(
            handlers.campaign_run or _default_campaign_handler,
            request,
        )

    if args.command_name == "runs":
        from experiments.study_workflow import inspect_run

        records = []
        for path in args.run_dirs:
            try:
                records.append(inspect_run(path, verify_artifacts=args.verify_artifacts))
            except (OSError, ValueError) as error:
                records.append({"path": str(path), "status": "invalid", "valid": False,
                                "process_complete": False, "errors": [str(error)]})
        if args.json:
            json.dump(records, stdout, indent=2, sort_keys=True)
            stdout.write("\n")
        else:
            for record in records:
                stdout.write(f"{record['path']}: {record['status']} valid={record['valid']}\n")
                for error in record["errors"]:
                    stdout.write(f"  {error}\n")
        return 0 if all(r["valid"] and (not args.require_complete or r["process_complete"]) for r in records) else 1

    if args.command_name == "study":
        from experiments.study_workflow import (
            finalize_study,
            prepare_study,
            summarize_study,
        )

        try:
            if args.study_command == "prepare":
                root = prepare_study(args.plan, args.results_root)
                stdout.write(f"{root}\n")
                return 0
            if args.study_command == "summarize":
                summary = summarize_study(
                    args.study_dir,
                    verify_artifacts=args.verify_artifacts,
                )
                if args.json:
                    json.dump(summary, stdout, indent=2, sort_keys=True)
                    stdout.write("\n")
                else:
                    stdout.write(
                        f"{summary['study_id']}: {summary['state']} "
                        f"(ready_for_review="
                        f"{str(summary['ready_for_review']).lower()})\n"
                    )
                return 0
            if args.study_command == "finalize":
                if args.manifest is not None:
                    stderr.write("--manifest is deprecated and ignored; writing a local receipt only.\n")
                final_path = finalize_study(
                    args.study_dir,
                    review_path=args.review,
                    manifest_path=args.manifest,
                    verify_artifacts=args.verify_artifacts,
                )
                stdout.write(f"{final_path}\n")
                return 0
        except (OSError, ValueError) as error:
            raise ConfigError(str(error)) from error

    raise CliUsageError(
        "Expected a supported ebl command. "
        f"Provided value: {args.command_name!r}."
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    handlers: Optional[CommandHandlers] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Parse and dispatch one CLI invocation.

    ``handlers=None`` connects the default numerical runtime lazily.  An
    explicitly supplied :class:`CommandHandlers` remains authoritative, so a
    missing injected handler fails clearly after the selected config has been
    fully validated.
    """

    output = stdout if stdout is not None else sys.stdout
    errors = stderr if stderr is not None else sys.stderr
    raw_argv = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    parser = build_parser()
    try:
        parsed = parser.parse_args(raw_argv)
        return _dispatch(
            parsed,
            handlers=(
                _default_command_handlers()
                if handlers is None
                else handlers
            ),
            command=("ebl",) + raw_argv,
            stdout=output,
            stderr=errors,
        )
    except (CliUsageError, ConfigError) as error:
        errors.write(f"error: {error}\n")
        return 2
