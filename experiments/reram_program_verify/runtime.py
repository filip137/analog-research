"""Runtime for IBM ReRAM pulse-count P&V characterization."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from ebl.runtime.store import RunStore
from core.artifacts import atomic_write_json, sha256_file
from experiments.reram_program_verify.analysis import (
    build_empirical_kernel,
    build_wan_comparison,
    fit_bounded_uniform_models,
    fit_gaussian_surrogates,
    write_plots,
    write_report,
)
from experiments.reram_program_verify.config import ReramProgramVerifySpec
from experiments.reram_program_verify.config import HWA_PRODUCTION_PROFILE
from experiments.reram_program_verify.integrity import validate_trajectory_database
from experiments.reram_program_verify.storage import TrajectoryStore
from core.contracts import to_plain_data
from hardware.devices.om import ConditioningResult
from methods.programming.om import ControllerSettings, PopulationStepEstimator, VerifyObservation, run_program_verify
from hardware.devices.om import IbmReramPlant, IbmReramPopulation, derive_seed, partition_device_identities, sample_ibm_reram_population

if TYPE_CHECKING:
    from ebl.cli import CharacterizeRequest


_ROOT = Path(__file__).resolve().parents[2]


def _trajectory_axes(num_devices: int, repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
    device_ids = torch.arange(num_devices, dtype=torch.int64).repeat_interleave(repeats)
    repeat_ids = torch.arange(repeats, dtype=torch.int64).repeat(num_devices)
    return device_ids, repeat_ids


def _repeat_seeds(
    base_seed: int,
    *,
    preset: str,
    device_ids: torch.Tensor,
    repeat_ids: torch.Tensor,
) -> list[int]:
    return [
        derive_seed(
            base_seed,
            preset,
            int(device_id),
            int(repeat_id),
            "repeat",
        )
        for device_id, repeat_id in zip(device_ids.tolist(), repeat_ids.tolist())
    ]


def _stream_seeds(
    base_seed: int,
    *,
    preset: str,
    start_protocol: str,
    target_index: int | None,
    repeat_seeds: list[int],
    stream: str,
) -> list[int]:
    result = []
    for repeat_seed in repeat_seeds:
        parts: tuple[object, ...]
        if target_index is None:
            parts = (preset, start_protocol, repeat_seed, stream)
        else:
            parts = (
                preset,
                start_protocol,
                target_index,
                repeat_seed,
                stream,
            )
        result.append(derive_seed(base_seed, *parts))
    return result


def _save_population(path: Path, population) -> None:
    np.savez_compressed(
        path,
        schema=np.asarray("ebl.ibm_reram.device_population"),
        schema_version=np.asarray(1, dtype=np.int64),
        preset=np.asarray(population.preset),
        aihwkit_version=np.asarray(population.aihwkit_version),
        nominal_dw_min=np.asarray(population.nominal_dw_min, dtype=np.float64),
        dw_min_std=np.asarray(population.dw_min_std, dtype=np.float64),
        write_noise_std=np.asarray(population.write_noise_std, dtype=np.float64),
        mult_noise=np.asarray(population.mult_noise, dtype=np.bool_),
        construction_seeds=population.construction_seeds.numpy(),
        max_bound=population.max_bound.numpy(),
        min_bound=population.min_bound.numpy(),
        dwmin_up=population.dwmin_up.numpy(),
        dwmin_down=population.dwmin_down.numpy(),
        reference=population.reference.numpy(),
        corrupt=population.corrupt.numpy(),
        preset_parameters_json=np.asarray(
            json.dumps(population.preset_parameters, allow_nan=False, sort_keys=True)
        ),
    )


def _load_population(path: Path) -> IbmReramPopulation:
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["schema"].item()) != "ebl.ibm_reram.device_population":
            raise RuntimeError("Expected an IBM ReRAM sampled-population artifact.")
        if int(payload["schema_version"].item()) != 1:
            raise RuntimeError("Expected sampled-population schema version 1.")
        return IbmReramPopulation(
            preset=str(payload["preset"].item()),
            aihwkit_version=str(payload["aihwkit_version"].item()),
            nominal_dw_min=float(payload["nominal_dw_min"].item()),
            dw_min_std=float(payload["dw_min_std"].item()),
            write_noise_std=float(payload["write_noise_std"].item()),
            mult_noise=bool(payload["mult_noise"].item()),
            construction_seeds=torch.from_numpy(
                payload["construction_seeds"].copy()
            ).to(torch.int64),
            max_bound=torch.from_numpy(payload["max_bound"].copy()).to(torch.float32),
            min_bound=torch.from_numpy(payload["min_bound"].copy()).to(torch.float32),
            dwmin_up=torch.from_numpy(payload["dwmin_up"].copy()).to(torch.float32),
            dwmin_down=torch.from_numpy(payload["dwmin_down"].copy()).to(torch.float32),
            reference=torch.from_numpy(payload["reference"].copy()).to(torch.float32),
            corrupt=torch.from_numpy(payload["corrupt"].copy()).to(torch.bool),
            preset_parameters=json.loads(
                str(payload["preset_parameters_json"].item())
            ),
        )


def _sample_population_for_runtime(
    *,
    spec: ReramProgramVerifySpec,
    population_path: Path,
    receipt_path: Path,
) -> tuple[IbmReramPopulation, dict[str, Any]]:
    settings = spec.settings
    if spec.runtime.device == "cpu":
        population = sample_ibm_reram_population(
            preset=spec.device.preset,
            num_devices=settings.num_devices,
            construction_seed=spec.runtime.construction_seed,
            enable_published_corruption=spec.device.enable_published_corruption,
            required_aihwkit_version=spec.runtime.required_aihwkit_version,
        )
        _save_population(population_path, population)
        receipt = {
            "schema": "ebl.ibm_reram.population_sampling_receipt",
            "schema_version": 1,
            "backend": "in_process_pinned_aihwkit",
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "aihwkit_version": population.aihwkit_version,
            "preset": population.preset,
            "num_devices": population.size,
            "construction_seed": spec.runtime.construction_seed,
            "enable_published_corruption": spec.device.enable_published_corruption,
            "population_sha256": sha256_file(population_path),
        }
        atomic_write_json(receipt_path, receipt)
        return population, receipt

    sampler = os.environ.get("EBL_AIHWKIT_PYTHON")
    if not sampler:
        raise RuntimeError(
            "Expected CUDA characterization to set EBL_AIHWKIT_PYTHON to a "
            "Python environment containing the pinned AIHWKit 1.1.0 sampler."
        )
    sampler_path = Path(sampler).expanduser().resolve()
    if not sampler_path.is_file() or not os.access(sampler_path, os.X_OK):
        raise RuntimeError(
            "Expected EBL_AIHWKIT_PYTHON to identify an executable file. "
            f"Provided value: {str(sampler_path)!r}."
        )
    command = [
        str(sampler_path),
        "-m",
        "experiments.reram_program_verify.population_sampler",
        "--preset",
        spec.device.preset,
        "--num-devices",
        str(settings.num_devices),
        "--construction-seed",
        str(spec.runtime.construction_seed),
        "--required-aihwkit-version",
        spec.runtime.required_aihwkit_version,
        "--output",
        str(population_path),
        "--receipt",
        str(receipt_path),
    ]
    if spec.device.enable_published_corruption:
        command.append("--enable-published-corruption")
    completed = subprocess.run(
        command,
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(
            "Expected the pinned external AIHWKit population sampler to "
            f"complete successfully. Exit={completed.returncode}; tail={detail!r}."
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("population_sha256") != sha256_file(population_path):
        raise RuntimeError(
            "Expected the sampled-population receipt hash to match its artifact."
        )
    population = _load_population(population_path)
    if (
        population.preset != spec.device.preset
        or population.size != settings.num_devices
        or population.aihwkit_version != spec.runtime.required_aihwkit_version
    ):
        raise RuntimeError(
            "Expected the externally sampled population to match the resolved config."
        )
    return population, receipt


def _sample_wan_for_runtime(
    *,
    spec: ReramProgramVerifySpec,
    targets: list[float],
    samples_per_target: int,
    output_path: Path,
) -> dict[str, Any] | None:
    if spec.runtime.device == "cpu":
        return None
    sampler = os.environ.get("EBL_AIHWKIT_PYTHON")
    if not sampler:
        raise RuntimeError(
            "Expected CUDA characterization to set EBL_AIHWKIT_PYTHON for "
            "the pinned Wan-2022 reference sampler."
        )
    sampler_path = Path(sampler).expanduser().resolve()
    command = [
        str(sampler_path),
        "-m",
        "experiments.reram_program_verify.wan_sampler",
        "--targets-json",
        json.dumps(targets, allow_nan=False, separators=(",", ":")),
        "--samples-per-target",
        str(samples_per_target),
        "--g-max-us",
        str(spec.settings.wan_g_max_us),
        "--noise-scale",
        str(spec.settings.wan_noise_scale),
        "--wan-seed",
        str(spec.runtime.wan_seed),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(
            "Expected the pinned external AIHWKit Wan sampler to complete "
            f"successfully. Exit={completed.returncode}; tail={detail!r}."
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class _ConditionedPopulation:
    result: ConditioningResult
    repeat_seeds: tuple[int, ...]
    conditioning_seeds: tuple[int, ...]


def _select_conditioned_population(
    conditioned: _ConditionedPopulation,
    selection: torch.Tensor,
) -> _ConditionedPopulation:
    """Select complete trajectories while preserving their recorded streams."""

    mask = torch.as_tensor(selection, dtype=torch.bool).reshape(-1)
    result = conditioned.result
    if mask.shape != result.success.detach().cpu().shape:
        raise ValueError(
            "Expected conditioned-population selection to match trajectory axes."
        )
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
    device_mask = mask.to(result.success.device)
    return _ConditionedPopulation(
        result=ConditioningResult(
            success=result.success[device_mask].clone(),
            pulse_count=result.pulse_count[device_mask].clone(),
            persistent=result.persistent[device_mask].clone(),
            apparent=result.apparent[device_mask].clone(),
        ),
        repeat_seeds=tuple(conditioned.repeat_seeds[index] for index in indices),
        conditioning_seeds=tuple(
            conditioned.conditioning_seeds[index] for index in indices
        ),
    )


def _condition_population(
    *,
    population,
    device_ids: torch.Tensor,
    repeat_ids: torch.Tensor,
    start_protocol: str,
    target_index: int,
    spec: ReramProgramVerifySpec,
) -> _ConditionedPopulation:
    repeat_seeds = tuple(
        _repeat_seeds(
            spec.runtime.repeat_seed,
            preset=population.preset,
            device_ids=device_ids,
            repeat_ids=repeat_ids,
        )
    )
    conditioning_seeds = tuple(
        _stream_seeds(
            spec.runtime.conditioning_seed,
            preset=population.preset,
            start_protocol=start_protocol,
            target_index=(
                None
                if spec.settings.conditioning_scope == "per_start"
                else target_index
            ),
            repeat_seeds=list(repeat_seeds),
            stream="conditioning",
        )
    )
    plant = IbmReramPlant(
        population.select(device_ids),
        seeds=conditioning_seeds,
        device=spec.runtime.device,
        maximum_random_draws=(
            2 * spec.settings.conditioning.maximum_pulses
            if spec.runtime.device == "cuda"
            else None
        ),
    )
    plant.reset_logical_zero()
    settings = spec.settings
    result = plant.condition_boundary(
        start_protocol=start_protocol,
        quiet_steps=settings.conditioning.quiet_steps,
        change_threshold=(
            2.0 * settings.conditioning.persistent_change_threshold_span_fraction
        ),
        maximum_pulses=settings.conditioning.maximum_pulses,
    )
    return _ConditionedPopulation(
        result=result,
        repeat_seeds=repeat_seeds,
        conditioning_seeds=conditioning_seeds,
    )


def _run_target(
    *,
    trajectory_store: TrajectoryStore,
    population,
    partitions: tuple[str, ...],
    device_ids: torch.Tensor,
    repeat_ids: torch.Tensor,
    controller_kind: str,
    start_protocol: str,
    tolerance_ratio: float,
    tolerance: float,
    target_index: int,
    target: float,
    spec: ReramProgramVerifySpec,
    estimator: PopulationStepEstimator,
    conditioned: _ConditionedPopulation,
) -> dict[str, int]:
    settings = spec.settings
    expanded_host = population.select(device_ids)
    execution_device = torch.device(spec.runtime.device)
    expanded = expanded_host.to(execution_device)
    repeat_seeds = list(conditioned.repeat_seeds)
    conditioning_seeds = list(conditioned.conditioning_seeds)
    pulse_seeds = _stream_seeds(
        spec.runtime.pulse_seed,
        preset=population.preset,
        start_protocol=start_protocol,
        target_index=target_index,
        repeat_seeds=repeat_seeds,
        stream="target_programming",
    )
    conditioning = conditioned.result
    plant = IbmReramPlant(
        expanded_host,
        seeds=pulse_seeds,
        device=execution_device,
        maximum_random_draws=(
            2 * settings.maximum_program_pulses
            if execution_device.type == "cuda"
            else None
        ),
    )
    plant.persistent.copy_(conditioning.persistent)
    plant.apparent.copy_(conditioning.apparent)

    partition_labels = tuple(partitions[int(device_id)] for device_id in device_ids.tolist())
    conditioning_success = conditioning.success.detach().cpu().tolist()
    conditioning_pulses = conditioning.pulse_count.detach().cpu().tolist()
    conditioned_apparent = (
        (conditioning.apparent + 1.0) / 2.0
    ).detach().cpu().tolist()
    conditioned_persistent = (
        (conditioning.persistent + 1.0) / 2.0
    ).detach().cpu().tolist()
    sampled_lower_persistent = (
        (expanded_host.logical_min + 1.0) / 2.0
    ).detach().cpu().tolist()
    sampled_upper_persistent = (
        (expanded_host.logical_max + 1.0) / 2.0
    ).detach().cpu().tolist()
    construction_seeds = expanded_host.construction_seeds.tolist()
    corrupt_flags = expanded_host.corrupt.tolist()
    base_rows = []
    for index in range(expanded_host.size):
        base_rows.append(
            {
                "preset": population.preset,
                "corrupt_population": int(spec.device.enable_published_corruption),
                "controller": controller_kind,
                "start_protocol": start_protocol,
                "tolerance_ratio": tolerance_ratio,
                "tolerance": tolerance,
                "target_index": target_index,
                "target": target,
                "device_id": int(device_ids[index]),
                "repeat_id": int(repeat_ids[index]),
                "partition_name": partition_labels[index],
                "construction_seed": int(construction_seeds[index]),
                "repeat_seed": int(repeat_seeds[index]),
                "conditioning_seed": int(conditioning_seeds[index]),
                "pulse_seed": int(pulse_seeds[index]),
                "controller_seed": None,
                "corrupt": int(corrupt_flags[index]),
                "conditioning_success": int(conditioning_success[index]),
                "conditioning_pulses": int(conditioning_pulses[index]),
                "conditioned_apparent": float(conditioned_apparent[index]),
                "conditioned_persistent": float(conditioned_persistent[index]),
                "sampled_lower_persistent": float(
                    sampled_lower_persistent[index]
                ),
                "sampled_upper_persistent": float(
                    sampled_upper_persistent[index]
                ),
            }
        )
    trajectory_ids = trajectory_store.begin_trajectories(base_rows)
    previous_apparent = (conditioning.apparent + 1.0) / 2.0
    calibration_mask = torch.tensor(
        [label == "calibration" for label in partition_labels],
        dtype=torch.bool,
        device=execution_device,
    )

    def observe(observation: VerifyObservation) -> None:
        nonlocal previous_apparent
        if observation.verify_index == 0:
            record_mask = conditioning.success.clone()
        else:
            record_mask = observation.pulse_count > 0
        persistent = (plant.persistent + 1.0) / 2.0
        event_rows = []
        indices = torch.nonzero(record_mask, as_tuple=False).reshape(-1)
        index_values = indices.detach().cpu().tolist()
        before_values = previous_apparent[indices].detach().cpu().tolist()
        apparent_values = observation.apparent[indices].detach().cpu().tolist()
        persistent_values = persistent[indices].detach().cpu().tolist()
        direction_values = observation.direction[indices].detach().cpu().tolist()
        count_values = observation.pulse_count[indices].detach().cpu().tolist()
        total_values = observation.total_pulses[indices].detach().cpu().tolist()
        for position, index in enumerate(index_values):
            direction = int(direction_values[position])
            count = int(count_values[position])
            event_rows.append(
                (
                    trajectory_ids[index],
                    observation.verify_index,
                    float(before_values[position]),
                    float(apparent_values[position]),
                    float(persistent_values[position]),
                    direction,
                    count,
                    direction * count,
                    int(total_values[position]),
                )
            )
        if event_rows:
            trajectory_store.insert_events(event_rows)
        if controller_kind == "one_pulse" and observation.verify_index > 0:
            estimator.observe(
                before=previous_apparent,
                after=observation.apparent,
                direction=observation.direction,
                pulse_count=observation.pulse_count,
                include=calibration_mask & record_mask,
            )
        previous_apparent = torch.where(
            record_mask, observation.apparent, previous_apparent
        )

    controller = ControllerSettings(
        kind=controller_kind,
        eta=settings.adaptive.eta,
        maximum_batch=settings.adaptive.maximum_batch,
        epsilon=settings.adaptive.epsilon,
        force_one_within_steps=settings.adaptive.force_one_within_steps,
    )
    result = run_program_verify(
        plant.controller_port(),
        targets=torch.full(
            (expanded.size,),
            target,
            dtype=torch.float32,
            device=execution_device,
        ),
        tolerance=tolerance,
        maximum_pulses=settings.maximum_program_pulses,
        settings=controller,
        estimator=estimator if controller_kind == "adaptive" else None,
        eligible=conditioning.success,
        observer=observe,
    )
    persistent_endpoint = (plant.persistent + 1.0) / 2.0
    physical = plant.persistent + expanded.reference
    bound_threshold = 2.0 * settings.conditioning.persistent_change_threshold_span_fraction
    at_lower = torch.abs(physical - expanded.min_bound) <= bound_threshold
    at_upper = torch.abs(physical - expanded.max_bound) <= bound_threshold
    saturated = at_lower | at_upper

    apparent_endpoints = result.apparent_endpoint.detach().cpu().tolist()
    persistent_endpoints = persistent_endpoint.detach().cpu().tolist()
    accepted_values = result.accepted.detach().cpu().tolist()
    nonfinite_values = result.nonfinite.detach().cpu().tolist()
    budget_values = result.budget_exhausted.detach().cpu().tolist()
    saturated_values = saturated.detach().cpu().tolist()
    set_values = result.set_count.detach().cpu().tolist()
    reset_values = result.reset_count.detach().cpu().tolist()
    total_values = result.total_pulses.detach().cpu().tolist()
    verify_values = result.verify_count.detach().cpu().tolist()
    reversal_values = result.reversals.detach().cpu().tolist()
    final_rows = []
    for index, trajectory_id in enumerate(trajectory_ids):
        apparent_value = float(apparent_endpoints[index])
        persistent_value = float(persistent_endpoints[index])
        final_rows.append(
            {
                "trajectory_id": trajectory_id,
                "accepted": int(accepted_values[index]),
                "initialization_failed": int(not conditioning_success[index]),
                "nonfinite": int(nonfinite_values[index]),
                "budget_exhausted": int(budget_values[index]),
                "saturated": int(saturated_values[index]),
                "endpoint_apparent": apparent_value,
                "endpoint_persistent": persistent_value,
                "residual_apparent": apparent_value - target,
                "residual_persistent": persistent_value - target,
                "set_count": int(set_values[index]),
                "reset_count": int(reset_values[index]),
                "total_pulses": int(total_values[index]),
                "verify_count": int(verify_values[index]),
                "reversals": int(reversal_values[index]),
            }
        )
    trajectory_store.finish_trajectories(final_rows)
    return {
        "trajectories": len(final_rows),
        "accepted": int(result.accepted.sum().item()),
        "initialization_failed": int((~conditioning.success).sum().item()),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
        "nonfinite": int(result.nonfinite.sum().item()),
    }


def _database_count(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0])
    finally:
        connection.close()


def run_characterize(request: "CharacterizeRequest") -> int:
    spec = request.spec
    if not isinstance(spec, ReramProgramVerifySpec):
        raise TypeError(
            "Expected ibm_reram_program_verify.v1 characterize to resolve "
            f"ReramProgramVerifySpec. Provided value: {type(spec).__name__}."
        )
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        resume_capability="unsupported",
    )
    try:
        settings = spec.settings
        population_path = store.run_dir / "artifacts" / "device_population.npz"
        sampling_receipt_path = (
            store.run_dir / "artifacts" / "population_sampling_receipt.json"
        )
        population, sampling_receipt = _sample_population_for_runtime(
            spec=spec,
            population_path=population_path,
            receipt_path=sampling_receipt_path,
        )
        partitions = partition_device_identities(
            settings.num_devices,
            seed=spec.runtime.partition_seed,
            fractions=(
                settings.partitions.calibration,
                settings.partitions.fit,
                settings.partitions.validation,
            ),
        )
        device_ids, repeat_ids = _trajectory_axes(
            settings.num_devices, settings.repeats_per_device
        )
        calibration_trajectory_mask = torch.tensor(
            [
                partitions[int(device_id)] == "calibration"
                for device_id in device_ids.tolist()
            ],
            dtype=torch.bool,
        )
        targets = torch.linspace(
            settings.target_minimum,
            settings.target_maximum,
            settings.target_points,
            dtype=torch.float32,
        ).tolist()
        nominal_step_fraction = population.nominal_dw_min / 2.0
        database_path = store.run_dir / "artifacts" / "trajectories.sqlite3"
        estimator_path = store.run_dir / "artifacts" / "step_estimators.json"

        metadata = {
            "execution_profile": settings.profile,
            "execution_device": spec.runtime.device,
            "pulse_rng_backend": (
                "per_trajectory_torch_cpu"
                if spec.runtime.device == "cpu"
                else "per_trajectory_buffered_torch_cpu"
            ),
            "population_sampling": sampling_receipt,
            "preset": population.preset,
            "aihwkit_version": population.aihwkit_version,
            "preset_parameters": population.preset_parameters,
            "enable_published_corruption": spec.device.enable_published_corruption,
            "observed_corrupt_devices": int(population.corrupt.sum()),
            "nominal_dw_min": population.nominal_dw_min,
            "nominal_step_fraction": nominal_step_fraction,
            "coordinate": "x=(w+1)/2",
            "evidence_class": (
                "model_based_aihwkit_preset"
                if settings.profile != "smoke"
                else "operational_smoke"
            ),
            "source": (
                "AIHWKit 1.1.0 ReRamArrayOMPresetDevice/ReRamArrayHfO2PresetDevice; "
                "Gong & Rasch et al., IEDM 2022"
            ),
            "source_references": {
                "aihwkit_preset_source": (
                    "https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/"
                    "simulator/presets/devices.py"
                ),
                "ibm_array_study": (
                    "https://research.ibm.com/publications/deep-learning-"
                    "acceleration-in-14nm-cmos-compatible-reram-array-device-"
                    "material-and-algorithm-co-optimization"
                ),
                "wan2022": "https://www.nature.com/articles/s41586-022-04992-8",
            },
            "seeds": to_plain_data(spec.runtime),
            "controller_randomness": {
                "stochastic": False,
                "seed": None,
                "trajectory_column": "controller_seed",
                "note": (
                    "not applicable: controller decisions are deterministic given "
                    "target, verify history, and the calibrated population estimator"
                ),
            },
            "controller_trajectory_scope": (
                {
                    "one_pulse": "calibration identities only",
                    "adaptive": "all calibration, fit, and validation identities",
                }
                if settings.profile == HWA_PRODUCTION_PROFILE
                else {
                    "one_pulse": "all identities",
                    "adaptive": "all identities",
                }
            ),
            "seed_derivation": {
                "construction": (
                    "derive_seed(construction_seed, preset, device_id), reduced "
                    "to AIHWKit's signed-31-bit construction-seed range"
                ),
                "repeat": (
                    "derive_seed(repeat_seed, preset, device_id, repeat_id, 'repeat')"
                ),
                "conditioning": (
                    "derive_seed(conditioning_seed, preset, start_protocol, "
                    + (
                        "repeat_seed, 'conditioning'); one blocked boundary state "
                        "is cloned across targets"
                        if settings.conditioning_scope == "per_start"
                        else "target_index, repeat_seed, 'conditioning')"
                    )
                ),
                "target_programming": (
                    "derive_seed(pulse_seed, preset, start_protocol, target_index, "
                    "repeat_seed, 'target_programming')"
                ),
            },
            "partitions_by_device_id": list(partitions),
            "conditioning_state_reuse": (
                (
                    "one immutable conditioned state per preset/start/device/repeat; "
                    "copied across matched target, controller, and tolerance conditions"
                )
                if settings.conditioning_scope == "per_start"
                else (
                    "one immutable conditioned state per preset/start/target/device/repeat; "
                    "copied across matched controller and tolerance conditions"
                )
            ),
        }
        totals = Counter()
        estimators: dict[str, Any] = {}
        conditioning_cache: dict[tuple[str, int], _ConditionedPopulation] = {}
        with TrajectoryStore(database_path) as trajectory_store:
            trajectory_store.insert_metadata(
                {
                    "characterization": json.dumps(metadata, allow_nan=False, sort_keys=True),
                    "resolved_spec": json.dumps(to_plain_data(spec), allow_nan=False, sort_keys=True),
                }
            )
            for start_protocol in settings.start_protocols:
                for tolerance_ratio in settings.tolerance_step_ratios:
                    tolerance = tolerance_ratio * nominal_step_fraction
                    estimator = PopulationStepEstimator(
                        bins=settings.calibration_bins,
                        fallback_step=nominal_step_fraction,
                    )
                    for controller_kind in settings.controllers:
                        for target_index, target in enumerate(targets):
                            conditioning_key = (
                                start_protocol,
                                -1
                                if settings.conditioning_scope == "per_start"
                                else target_index,
                            )
                            conditioned = conditioning_cache.get(conditioning_key)
                            if conditioned is None:
                                conditioned = _condition_population(
                                    population=population,
                                    device_ids=device_ids,
                                    repeat_ids=repeat_ids,
                                    start_protocol=start_protocol,
                                    target_index=target_index,
                                    spec=spec,
                                )
                                conditioning_cache[conditioning_key] = conditioned
                            run_device_ids = device_ids
                            run_repeat_ids = repeat_ids
                            run_conditioned = conditioned
                            if (
                                settings.profile == HWA_PRODUCTION_PROFILE
                                and controller_kind == "one_pulse"
                            ):
                                run_device_ids = device_ids[
                                    calibration_trajectory_mask
                                ]
                                run_repeat_ids = repeat_ids[
                                    calibration_trajectory_mask
                                ]
                                run_conditioned = _select_conditioned_population(
                                    conditioned,
                                    calibration_trajectory_mask,
                                )
                            summary = _run_target(
                                trajectory_store=trajectory_store,
                                population=population,
                                partitions=partitions,
                                device_ids=run_device_ids,
                                repeat_ids=run_repeat_ids,
                                controller_kind=controller_kind,
                                start_protocol=start_protocol,
                                tolerance_ratio=tolerance_ratio,
                                tolerance=tolerance,
                                target_index=target_index,
                                target=float(target),
                                spec=spec,
                                estimator=estimator,
                                conditioned=run_conditioned,
                            )
                            totals.update(summary)
                        store.append_metric(
                            {
                                "mode": "characterize",
                                "preset": population.preset,
                                "controller": controller_kind,
                                "start_protocol": start_protocol,
                                "tolerance_ratio": tolerance_ratio,
                                "completed_targets": len(targets),
                            }
                        )
                        if controller_kind == "one_pulse":
                            key = f"{start_protocol}__tau_step_{tolerance_ratio:g}"
                            estimators[key] = estimator.to_mapping()

        database_digest = sha256_file(database_path)
        atomic_write_json(
            estimator_path,
            {
                "schema": "ebl.ibm_reram.step_estimators",
                "schema_version": 1,
                "calibration_partition_only": True,
                "metadata": {
                    "execution_profile": settings.profile,
                    "preset": population.preset,
                    "enable_published_corruption": (
                        spec.device.enable_published_corruption
                    ),
                    "nominal_dw_min": population.nominal_dw_min,
                    "trajectory_artifact_sha256": database_digest,
                    "device_population_artifact_sha256": sha256_file(
                        population_path
                    ),
                },
                "estimators": estimators,
            },
        )
        integrity_path = store.run_dir / "artifacts" / "integrity_report.json"
        trajectories_per_condition = settings.num_devices * settings.repeats_per_device
        expected_condition_count = (
            len(settings.controllers)
            * len(settings.start_protocols)
            * len(settings.tolerance_step_ratios)
            * len(targets)
        )
        trajectories_per_controller = None
        expected_trajectory_count = expected_condition_count * trajectories_per_condition
        if settings.profile == HWA_PRODUCTION_PROFILE:
            calibration_trajectories = int(
                calibration_trajectory_mask.sum().item()
            )
            trajectories_per_controller = {
                "one_pulse": calibration_trajectories,
                "adaptive": trajectories_per_condition,
            }
            expected_trajectory_count = (
                len(settings.start_protocols)
                * len(settings.tolerance_step_ratios)
                * len(targets)
                * sum(trajectories_per_controller.values())
            )
        validate_trajectory_database(
            database_path,
            output_path=integrity_path,
            expected_trajectory_count=expected_trajectory_count,
            expected_condition_count=expected_condition_count,
            trajectories_per_condition=trajectories_per_condition,
            maximum_program_pulses=settings.maximum_program_pulses,
            controllers=settings.controllers,
            start_protocols=settings.start_protocols,
            tolerance_ratios=settings.tolerance_step_ratios,
            target_points=settings.target_points,
            repeats_per_device=settings.repeats_per_device,
            expected_partition_device_counts=dict(Counter(partitions)),
            conditioning_scope=settings.conditioning_scope,
            trajectories_per_controller=trajectories_per_controller,
        )
        trajectory_count = _database_count(database_path)
        analysis_metadata = {
            **metadata,
            "resolved_config_sha256": store.manifest["config"]["sha256"],
            "trajectory_artifact_sha256": database_digest,
            "device_population_artifact_sha256": sha256_file(population_path),
            "step_estimator_artifact_sha256": sha256_file(estimator_path),
            "trajectory_integrity_artifact_sha256": sha256_file(integrity_path),
            "controller": {
                "adaptive": asdict(settings.adaptive),
                "maximum_program_pulses": settings.maximum_program_pulses,
            },
        }
        kernel_path = store.run_dir / "artifacts" / "empirical_kernel.json"
        fit_path = store.run_dir / "artifacts" / "gaussian_surrogate.json"
        bounded_uniform_path = (
            store.run_dir / "artifacts" / "bounded_uniform_model.json"
        )
        wan_path = store.run_dir / "artifacts" / "wan2022_comparison.json"
        report_path = store.run_dir / "artifacts" / "report.md"
        build_empirical_kernel(
            database_path, output_path=kernel_path, metadata=analysis_metadata
        )
        fit = fit_gaussian_surrogates(
            database_path,
            output_path=fit_path,
            polynomial_order=settings.polynomial_order,
            standard_deviation_floor=settings.standard_deviation_floor,
            analysis_seed=spec.runtime.analysis_seed,
            metadata=analysis_metadata,
        )
        bounded_uniform = fit_bounded_uniform_models(
            database_path,
            output_path=bounded_uniform_path,
            metadata=analysis_metadata,
        )
        validation_devices = sum(label == "validation" for label in partitions)
        wan_reference_path = (
            store.run_dir / "artifacts" / "wan2022_reference_samples.json"
        )
        wan_reference = _sample_wan_for_runtime(
            spec=spec,
            targets=targets,
            samples_per_target=validation_devices * settings.repeats_per_device,
            output_path=wan_reference_path,
        )
        wan = build_wan_comparison(
            targets=targets,
            samples_per_target=validation_devices * settings.repeats_per_device,
            g_max_us=settings.wan_g_max_us,
            noise_scale=settings.wan_noise_scale,
            wan_seed=spec.runtime.wan_seed,
            fit_artifact=fit,
            output_path=wan_path,
            wan_reference=wan_reference,
        )
        plot_paths = write_plots(
            fit_artifact=fit,
            wan_artifact=wan,
            output_dir=store.run_dir / "artifacts" / "plots",
        )
        write_report(
            path=report_path,
            preset=population.preset,
            corrupt_population=spec.device.enable_published_corruption,
            execution_profile=settings.profile,
            trajectory_count=trajectory_count,
            fit_artifact=fit,
            bounded_uniform_artifact=bounded_uniform,
            wan_artifact=wan,
        )

        artifacts = [
            store.artifact_record(database_path, kind="pulse_trajectory_database"),
            store.artifact_record(population_path, kind="sampled_device_population"),
            store.artifact_record(
                sampling_receipt_path, kind="population_sampling_receipt"
            ),
            store.artifact_record(estimator_path, kind="adaptive_step_calibration"),
            store.artifact_record(integrity_path, kind="trajectory_integrity_report"),
            store.artifact_record(kernel_path, kind="empirical_endpoint_kernel"),
            store.artifact_record(fit_path, kind="gaussian_endpoint_model"),
            store.artifact_record(
                bounded_uniform_path,
                kind="bounded_piecewise_uniform_endpoint_model",
            ),
            store.artifact_record(wan_path, kind="wan2022_comparison"),
            *(
                [
                    store.artifact_record(
                        wan_reference_path,
                        kind="wan2022_reference_samples",
                    )
                ]
                if wan_reference is not None
                else []
            ),
            store.artifact_record(report_path, kind="characterization_report"),
            *(store.artifact_record(path, kind="characterization_plot") for path in plot_paths),
        ]
        adequate = {
            key: bool(value.get("adequate", False))
            for key, value in fit["conditions"].items()
        }
        bounded_uniform_adequate = {
            key: bool(value.get("adequate", False))
            for key, value in bounded_uniform["conditions"].items()
        }
        reachability_probability_validation = {
            key: value.get("validation", {}).get("reachability_probability")
            for key, value in bounded_uniform["conditions"].items()
        }
        store.complete(
            metrics={
                "evidence_class": (
                    "model_based_aihwkit_preset"
                    if settings.profile != "smoke"
                    else "operational_smoke"
                ),
                "execution_profile": settings.profile,
                "preset": population.preset,
                "corrupt_population": spec.device.enable_published_corruption,
                "sampled_devices": population.size,
                "observed_corrupt_devices": int(population.corrupt.sum()),
                "trajectories": trajectory_count,
                "accepted": totals["accepted"],
                "initialization_failed": totals["initialization_failed"],
                "budget_exhausted": totals["budget_exhausted"],
                "nonfinite": totals["nonfinite"],
                "gaussian_adequacy": adequate,
                "bounded_uniform_adequacy": bounded_uniform_adequate,
                "reachability_probability_validation": (
                    reachability_probability_validation
                ),
                "wan_programming_time_seconds": 1.0,
                "hwa_or_on_chip_training_performed": False,
            },
            artifacts=artifacts,
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_characterize"]
