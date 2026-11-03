"""ebl.runtime.sampling — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from collections.abc import Sequence
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from core.artifacts import sha256_file
from core.parameters import ParameterBinding
from hardware.devices.om import OM_PRESET, PUBLISHED_CORRUPT_PROBABILITY, PUBLISHED_CORRUPT_RANGE
from hardware.populations import IbmReramArrayPopulation, _ARRAY_SAMPLING_RECEIPT_SCHEMA, _ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION, _REQUIRED_AIHWKIT_VERSION, _ROOT, _binding_layout, load_om_array_population


def sample_om_array_population_external(
    bindings: Sequence[ParameterBinding],
    *,
    assignment_seed: int,
    corruption_policy: str,
    aihwkit_python: Path,
    population_path: Path,
    receipt_path: Path,
) -> tuple[IbmReramArrayPopulation, dict[str, Any]]:
    """Sample OM cells in pinned AIHWKit while training stays in CUDA PyTorch."""

    keys, shapes = _binding_layout(bindings)
    sampler = aihwkit_python.expanduser().resolve()
    if not sampler.is_file() or not os.access(sampler, os.X_OK):
        raise RuntimeError(
            "Expected EBL_AIHWKIT_PYTHON to identify an executable pinned "
            f"AIHWKit interpreter. Provided value: {str(sampler)!r}."
        )
    destination = population_path.expanduser().resolve()
    receipt_destination = receipt_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_destination.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "preset": OM_PRESET,
        "assignment_seed": assignment_seed,
        "corruption_policy": corruption_policy,
        "preset_default_corrupt_devices_prob": 0.0,
        "published_corrupt_devices_prob": PUBLISHED_CORRUPT_PROBABILITY[
            OM_PRESET
        ],
        "corrupt_devices_range": PUBLISHED_CORRUPT_RANGE[OM_PRESET],
        "binding_keys": list(keys),
        "binding_shapes": [list(shape) for shape in shapes],
        "required_aihwkit_version": _REQUIRED_AIHWKIT_VERSION,
    }
    command = [
        str(sampler),
        "-m",
        "experiments.reram_program_verify.hwa_population_sampler",
        "--request-json",
        json.dumps(request, allow_nan=False, sort_keys=True, separators=(",", ":")),
        "--output",
        str(destination),
        "--receipt",
        str(receipt_destination),
    ]
    environment = os.environ.copy()
    environment.pop("EBL_AIHWKIT_PYTHON", None)
    completed = subprocess.run(
        command,
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(
            "Expected the pinned external OM HWA population sampler to "
            f"complete successfully. Exit={completed.returncode}; tail={detail!r}."
        )
    try:
        receipt = json.loads(receipt_destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Expected a readable OM array sampling receipt.") from error
    expected_receipt_keys = {
        "schema",
        "schema_version",
        "backend",
        "python_executable",
        "python_version",
        "torch_version",
        "aihwkit_version",
        "request",
        "num_cells",
        "population_fingerprint",
        "population_sha256",
        "sampler_source_sha256",
        "population_implementation_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_receipt_keys
        or receipt.get("schema") != _ARRAY_SAMPLING_RECEIPT_SCHEMA
        or receipt.get("schema_version") != _ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION
        or receipt.get("backend") != "external_pinned_aihwkit_python"
        or receipt.get("aihwkit_version") != _REQUIRED_AIHWKIT_VERSION
        or receipt.get("request") != request
        or receipt.get("population_sha256") != sha256_file(destination)
        or receipt.get("sampler_source_sha256")
        != sha256_file(_ROOT / "experiments/reram_program_verify/hwa_population_sampler.py")
        or receipt.get("population_implementation_sha256")
        != sha256_file(_ROOT / "hardware/populations.py")
    ):
        raise RuntimeError("Expected the OM array sampling receipt to match its request and sources.")
    population = load_om_array_population(destination)
    if (
        population.assignment_seed != assignment_seed
        or population.corruption_policy != corruption_policy
        or population.binding_keys != keys
        or population.binding_shapes != shapes
        or population.fingerprint != receipt.get("population_fingerprint")
        or population.size != receipt.get("num_cells")
    ):
        raise RuntimeError("Expected the sampled OM population to match its declared layout.")
    return population, receipt
