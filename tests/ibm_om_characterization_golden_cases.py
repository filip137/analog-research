"""Golden cases for the characterization (per-cell RNG) device stack.

Unit cases exercise the per-cell plants, the controller, the buffered CUDA
draws, the step estimator and the identity partition on synthetic noisy
populations.  End-to-end cases run the three characterization smoke configs
through the public CLI and record every artifact except run provenance.

Run ``python tests/ibm_om_characterization_golden_cases.py capture`` on the
reference commit (with ``EBL_AIHWKIT_PYTHON`` set and a free GPU) to (re)write
``tests/golden/ibm_om_characterization``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from ibm_om_golden_cases import assert_same as _assert_same, environment  # noqa: E402
from methods.ibm_reram_program_verify import (  # noqa: E402
    ControllerSettings,
    IbmReramPlant,
    IbmReramPopulation,
    IbmReramRawActivePlant,
    PopulationStepEstimator,
    VerifyObservation,
    derive_seed,
    make_buffered_normal_draws,
    partition_device_identities,
    run_program_verify,
)

GOLDEN_DIR = ROOT / "tests" / "golden" / "ibm_om_characterization"
PY312 = Path(sys.executable)

# Keys that hash or name implementation sources or interpreters; they change
# with any code move by design and are excluded everywhere.
VOLATILE_KEYS = frozenset(
    {
        "sampler_source_sha256",
        "population_implementation_sha256",
        "python_executable",
    }
)


def requirements() -> dict[str, bool]:
    sampler = os.environ.get("EBL_AIHWKIT_PYTHON")
    return {
        "aihwkit": bool(sampler) and Path(sampler).is_file(),
        "cuda": torch.cuda.is_available(),
    }


# --------------------------------------------------------------------------
# Synthetic populations
# --------------------------------------------------------------------------


def population(size: int, *, seed: int, write_noise_std: float, corrupt_index: int | None = 3) -> IbmReramPopulation:
    generator = torch.Generator().manual_seed(seed)
    reference = 0.08 * torch.randn(size, generator=generator)
    max_bound = reference + 0.6 + 0.8 * torch.rand(size, generator=generator)
    min_bound = reference - 0.6 - 0.8 * torch.rand(size, generator=generator)
    dwmin_up = 0.0949 * (0.5 + torch.rand(size, generator=generator))
    dwmin_down = 0.0949 * (0.5 + torch.rand(size, generator=generator))
    corrupt = torch.zeros(size, dtype=torch.bool)
    if corrupt_index is not None:
        corrupt[corrupt_index] = True
        max_bound[corrupt_index] = reference[corrupt_index] + 0.1
        min_bound[corrupt_index] = reference[corrupt_index] + 0.1
        dwmin_up[corrupt_index] = 0.0
        dwmin_down[corrupt_index] = 0.0
    return IbmReramPopulation(
        preset="reram_array_om",
        aihwkit_version="1.1.0",
        nominal_dw_min=0.0949,
        dw_min_std=0.3,
        write_noise_std=write_noise_std,
        mult_noise=False,
        construction_seeds=torch.arange(1000, 1000 + size, dtype=torch.int64),
        max_bound=max_bound.float(),
        min_bound=min_bound.float(),
        dwmin_up=dwmin_up.float(),
        dwmin_down=dwmin_down.float(),
        reference=reference.float(),
        corrupt=corrupt,
        preset_parameters={"fixture": True},
    )


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def _result(result) -> dict[str, Any]:
    return _cpu({name: getattr(result, name) for name in result.__dataclass_fields__})


def _seeds(size: int, base: int) -> list[int]:
    return [derive_seed(base, "golden", index) for index in range(size)]


def _plant_kwargs(device: str, budget: int) -> dict[str, Any]:
    return {} if device == "cpu" else {"maximum_random_draws": budget}


# --------------------------------------------------------------------------
# Unit cases
# --------------------------------------------------------------------------


def run_logical_plant_case(device: str, write_noise_std: float) -> dict[str, Any]:
    size = 24
    pop = population(size, seed=11, write_noise_std=write_noise_std)
    plant = IbmReramPlant(pop, seeds=_seeds(size, 5), device=device, **_plant_kwargs(device, 4096))
    record: dict[str, Any] = {"rng_backend": plant.rng_backend}
    pattern = torch.Generator().manual_seed(3)
    pulses = []
    for _ in range(5):
        directions = torch.randint(-1, 2, (size,), generator=pattern).to(torch.int8)
        plant.pulse(directions.to(device))
        pulses.append(_cpu((plant.persistent, plant.apparent)))
    record["pulses"] = pulses
    saved = _cpu(plant.state_dict())
    record["state"] = saved
    for start in ("lower_to_target", "upper_to_target"):
        plant.reset_logical_zero()
        conditioning = plant.condition_boundary(
            start_protocol=start, quiet_steps=3, change_threshold=1e-4, maximum_pulses=200
        )
        record[f"conditioning_{start}"] = _cpu(
            {name: getattr(conditioning, name) for name in conditioning.__dataclass_fields__}
        )
    # Controller: calibrate one_pulse, then adaptive with the fitted estimator.
    targets = torch.linspace(0.15, 0.85, size, device=device)
    estimator = PopulationStepEstimator(bins=5, fallback_step=0.0949 / 2.0)
    observations: list[dict[str, Any]] = []
    last: dict[str, torch.Tensor] = {}

    def observer(observation: VerifyObservation) -> None:
        observations.append(_cpu({name: getattr(observation, name) for name in observation.__dataclass_fields__}))
        if "apparent" in last:
            estimator.observe(
                before=last["apparent"],
                after=observation.apparent,
                direction=observation.direction,
                pulse_count=observation.pulse_count,
            )
        last["apparent"] = observation.apparent

    plant.reset_logical_zero()
    plant.condition_boundary(start_protocol="lower_to_target", quiet_steps=3, change_threshold=1e-4, maximum_pulses=200)
    plant.set_seeds(_seeds(size, 6))
    one_pulse = run_program_verify(
        plant.controller_port(),
        targets=targets,
        tolerance=0.02,
        maximum_pulses=96,
        settings=ControllerSettings(kind="one_pulse"),
        observer=observer,
    )
    record["one_pulse"] = _result(one_pulse)
    record["observations"] = observations[:6] + observations[-2:]
    record["estimator"] = estimator.to_mapping()
    record["estimate"] = _cpu(estimator.estimate(targets, torch.ones(size, dtype=torch.int8, device=device)))
    plant.reset_logical_zero()
    plant.condition_boundary(start_protocol="lower_to_target", quiet_steps=3, change_threshold=1e-4, maximum_pulses=200)
    plant.set_seeds(_seeds(size, 7))
    eligible = torch.ones(size, dtype=torch.bool, device=device)
    eligible[5] = False
    adaptive = run_program_verify(
        plant.controller_port(),
        targets=targets,
        tolerance=0.02,
        maximum_pulses=96,
        settings=ControllerSettings(kind="adaptive"),
        estimator=PopulationStepEstimator.from_mapping(estimator.to_mapping()),
        eligible=eligible,
    )
    record["adaptive"] = _result(adaptive)
    record["after_adaptive"] = _cpu((plant.persistent, plant.apparent, plant.state_dict()))
    # Exact continuation from a saved state.
    replay = IbmReramPlant(pop, seeds=_seeds(size, 5), device=device, **_plant_kwargs(device, 4096))
    replay.load_state_dict(saved)
    continued = []
    for _ in range(3):
        directions = torch.randint(-1, 2, (size,), generator=pattern).to(torch.int8).to(device)
        replay.pulse(directions)
        continued.append(_cpu((replay.persistent, replay.apparent)))
    record["continued"] = continued
    return record


def run_raw_plant_case(device: str) -> dict[str, Any]:
    size = 16
    pop = population(size, seed=21, write_noise_std=1.4113, corrupt_index=None)
    plant = IbmReramRawActivePlant(
        pop,
        seeds=_seeds(size, 8),
        coordinate_min=-3.455834150314331,
        coordinate_scale=5.994294881820679,
        device=device,
        **_plant_kwargs(device, 8192),
    )
    record: dict[str, Any] = {}
    plant.reset_logical_zero()
    conditioning = plant.condition_boundary(
        start_protocol="lower_to_target", quiet_steps=4, change_threshold=2e-6, maximum_pulses=2000
    )
    record["conditioning"] = _cpu({name: getattr(conditioning, name) for name in conditioning.__dataclass_fields__})
    plant.set_seeds(_seeds(size, 9))
    targets = torch.linspace(0.45, 0.65, size, device=device)
    result = run_program_verify(
        plant.controller_port(),
        targets=targets,
        tolerance=0.00791586015294392,
        maximum_pulses=128,
        settings=ControllerSettings(kind="one_pulse"),
    )
    record["one_pulse"] = _result(result)
    record["state"] = _cpu((plant.persistent_a, plant.apparent_a, plant.state_dict()))
    port = plant.controller_port()
    record["verify"] = _cpu(port.verify())
    return record


def run_helpers_case() -> dict[str, Any]:
    record: dict[str, Any] = {
        "buffered": make_buffered_normal_draws(_seeds(5, 4), maximum_random_draws=7, device="cpu"),
        "seeds": [derive_seed(12, "a", index, "b") for index in range(4)],
    }
    for size, seed in ((3, 1), (10, 2), (37, 3), (200, 4)):
        record[f"partition_{size}_{seed}"] = partition_device_identities(size, seed=seed)
    record["partition_custom"] = partition_device_identities(50, seed=9, fractions=(0.1, 0.7, 0.2))
    estimator = PopulationStepEstimator(bins=4, fallback_step=0.05)
    generator = torch.Generator().manual_seed(2)
    before = torch.rand(40, generator=generator)
    after = (before + 0.05 * torch.randn(40, generator=generator)).clamp(0, 1)
    direction = torch.where(after > before, torch.ones(40, dtype=torch.int8), -torch.ones(40, dtype=torch.int8))
    estimator.observe(before=before, after=after, direction=direction, pulse_count=torch.randint(0, 3, (40,), generator=generator))
    record["estimator"] = estimator.to_mapping()
    record["estimator_roundtrip"] = PopulationStepEstimator.from_mapping(estimator.to_mapping()).to_mapping()
    record["estimate"] = estimator.estimate(before, direction)
    return record


# --------------------------------------------------------------------------
# End-to-end smoke runs
# --------------------------------------------------------------------------

SMOKE_RUNS = {
    "cpu": ("examples/reram_program_verify/smoke_om.json", "aihwkit"),
    "cuda": ("examples/reram_program_verify/smoke_cuda_om.json", "py312"),
    "raw_active": ("examples/reram_program_verify/raw_active_common_cell_9/smoke.json", "py312"),
}


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def _source_normalized_record(value):
    """Compare all SQLite rows while excluding embedded source identities.

    The raw-active database embeds sampler receipts, so relocation changes its
    whole-file hash. Live artifact integrity is checked before comparison.
    Existing golden files are retained unchanged.
    """
    import copy
    result = copy.deepcopy(value)
    if not isinstance(result, dict) or "trajectories.sqlite3" not in result:
        return result
    database = result["trajectories.sqlite3"]
    for row in database.get("metadata", {}).get("rows", []):
        row[1] = json.dumps(_strip(json.loads(row[1])), sort_keys=True)
    if "integrity_report.json" in result:
        result["integrity_report.json"].pop("database_sha256", None)
    return result


def assert_same(expected, actual, path="root"):
    _assert_same(_source_normalized_record(expected), _source_normalized_record(actual), path)


def _array_record(array: np.ndarray) -> tuple[str, list[int], Any]:
    """Exact array fingerprint: small arrays verbatim, large ones by digest."""

    contiguous = np.ascontiguousarray(array)
    payload = contiguous.tobytes()
    if contiguous.size > 4096:
        payload = "sha256:" + hashlib.sha256(payload).hexdigest()
    return (str(contiguous.dtype), list(contiguous.shape), payload)


def _artifacts(run: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"config.resolved.json": json.loads((run / "config.resolved.json").read_text())}
    for path in sorted((run / "artifacts").iterdir()):
        if path.suffix == ".json":
            out[path.name] = _strip(json.loads(path.read_text()))
        elif path.suffix == ".sqlite3":
            connection = sqlite3.connect(path)
            tables = {}
            for (name,) in connection.execute(
                "select name from sqlite_master where type='table' order by name"
            ):
                cursor = connection.execute(f"select * from {name} order by rowid")
                tables[name] = {
                    "columns": [item[0] for item in cursor.description],
                    "rows": [list(row) for row in cursor.fetchall()],
                }
            connection.close()
            out[path.name] = tables
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                out[path.name] = {key: _array_record(payload[key]) for key in payload.files}
        elif path.suffix == ".md":
            out[path.name] = path.read_text()
    integrity = out.get("integrity_report.json", {})
    if "database_sha256" in integrity:
        database = run / "artifacts" / "trajectories.sqlite3"
        assert hashlib.sha256(database.read_bytes()).hexdigest() == integrity["database_sha256"]
    return out


def run_smoke(kind: str) -> dict[str, Any]:
    config, interpreter = SMOKE_RUNS[kind]
    python = os.environ["EBL_AIHWKIT_PYTHON"] if interpreter == "aihwkit" else str(PY312)
    with tempfile.TemporaryDirectory() as raw_dir:
        completed = subprocess.run(
            [python, "-m", "ebl", "characterize", "--config", config, "--output-dir", raw_dir],
            cwd=ROOT,
            env={**os.environ},
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"smoke {kind} failed:\n{completed.stderr[-3000:]}")
        (run,) = list(Path(raw_dir).iterdir())
        return _artifacts(run)


# --------------------------------------------------------------------------
# Registry and capture
# --------------------------------------------------------------------------


def all_cases() -> dict[str, tuple[Callable[[], Any], tuple[str, ...]]]:
    cases: dict[str, tuple[Callable[[], Any], tuple[str, ...]]] = {
        "unit/helpers": (run_helpers_case, ()),
        "unit/logical_cpu_noisy": (lambda: run_logical_plant_case("cpu", 1.4113), ()),
        "unit/logical_cpu_noiseless_write": (lambda: run_logical_plant_case("cpu", 0.0), ()),
        "unit/raw_cpu": (lambda: run_raw_plant_case("cpu"), ()),
        "unit/logical_cuda_noisy": (lambda: run_logical_plant_case("cuda", 1.4113), ("cuda",)),
        "unit/raw_cuda": (lambda: run_raw_plant_case("cuda"), ("cuda",)),
        "smoke/cpu": (lambda: run_smoke("cpu"), ("aihwkit",)),
        "smoke/cuda": (lambda: run_smoke("cuda"), ("aihwkit", "cuda")),
        "smoke/raw_active": (lambda: run_smoke("raw_active"), ("aihwkit",)),
    }
    return cases


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.pt"


def capture() -> None:
    available = requirements()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, (fn, needs) in all_cases().items():
        missing = [need for need in needs if not available[need]]
        if missing:
            raise SystemExit(f"cannot capture {name}: missing {missing}")
        result = fn()
        path = golden_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)
        print(f"captured {name} ({path.stat().st_size} bytes)")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {**environment(), "commit": commit, "cases": sorted(all_cases())}
    (GOLDEN_DIR / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


__all__ = ["all_cases", "assert_same", "golden_path", "requirements"]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "capture":
        capture()
    else:
        print("\n".join(sorted(all_cases())))
