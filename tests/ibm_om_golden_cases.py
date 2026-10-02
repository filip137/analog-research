"""Golden cases for the IBM OM HWA / recovery layering refactor.

Every case is a deterministic CPU computation over a synthetic, noisy IBM OM
population (non-zero cycle and write noise, varied bounds, references and
corrupt identities), so a change in RNG consumption order or in any
floating-point expression changes the recorded tensors or generator states.

Run ``python tests/ibm_om_golden_cases.py capture`` on the reference commit to
(re)write ``tests/golden/ibm_om``; ``tests/test_ibm_om_golden.py`` replays the
cases and compares type-strictly and bit-exactly.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import math
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts import sha256_file  # noqa: E402
from experiments.reram_program_verify.hwa_model import CONDITION_KEY  # noqa: E402
from model.resistive.builders import ParameterBinding  # noqa: E402
from model.variable.parameter import DenseWeight  # noqa: E402
from training.ibm_reram_program_verify import PopulationStepEstimator  # noqa: E402

GOLDEN_DIR = ROOT / "tests" / "golden" / "ibm_om"

# The module attribute through which the HWA modifier obtains its population.
# Golden replays patch exactly this lookup site; it is the only line to update
# when the modifier moves.
POPULATION_LOOKUP_SITE = ("training.ibm_reram_hwa", "sample_om_array_population")

# Keys whose values hash implementation source files and therefore change with
# any edit by design.  They are excluded from comparison everywhere.
SOURCE_HASH_KEYS = frozenset(
    {"sampler_source_sha256", "population_implementation_sha256"}
)

# Explicit, stored config fields introduced by the layering refactor.  Empty
# until those fields exist; then a config-like dict on either side is compared
# without them and their values are asserted separately.
NEW_CONFIG_KEYS: tuple[str, ...] = ()

OM_NOMINAL_DW_MIN = 0.0949
OM_DW_MIN_STD = 0.3
OM_WRITE_NOISE_STD = 1.4113


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------


def _hwa():
    return importlib.import_module("training.ibm_reram_hwa")


@contextlib.contextmanager
def patched_population(population) -> Iterator[None]:
    module = importlib.import_module(POPULATION_LOOKUP_SITE[0])
    name = POPULATION_LOOKUP_SITE[1]
    original = getattr(module, name)
    setattr(module, name, lambda *args, **kwargs: population)
    try:
        yield
    finally:
        setattr(module, name, original)


def _snapshot(value: Any) -> Any:
    """Deep-copy a nested result so later in-place mutation cannot leak in."""

    return copy.deepcopy(value)


def _outcome(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "value": fn()}
    except Exception as error:  # noqa: BLE001 - behaviour includes failures
        return {"ok": False, "type": type(error).__name__, "message": str(error)}


def _generator_state(generator: torch.Generator) -> torch.Tensor:
    return generator.get_state().detach().cpu().clone()


# --------------------------------------------------------------------------
# Synthetic device model and populations
# --------------------------------------------------------------------------


def _quantiles(value: float) -> dict[str, object]:
    return {
        "count": 10,
        "probabilities": [0.0, 0.25, 0.5, 0.75, 1.0],
        "values": [value - 0.02, value - 0.01, value, value + 0.01, value + 0.02],
    }


def _endpoint_model(*, corrupt: bool) -> dict[str, object]:
    records = []
    for target in (0.0, 0.5, 1.0):
        classes = {}
        for name, probability in (
            ("target_below_lower_bound", 0.2),
            ("target_inside_bounds", 0.6),
            ("target_above_upper_bound", 0.2),
        ):
            classes[name] = {
                "probability": probability,
                "acceptance_window_reachable_probability": 0.9,
                "success_probability": 0.8,
                "accepted_terminal": {
                    "apparent_endpoint": _quantiles(target),
                    "persistent_endpoint": _quantiles(min(1.0, target + 0.01)),
                },
                "failed_terminal": {
                    "apparent_endpoint": _quantiles(min(1.0, target + 0.05)),
                    "persistent_endpoint": _quantiles(min(1.0, target + 0.04)),
                },
            }
        records.append(
            {
                "target": target,
                "tolerance": 0.04745,
                "accepted_noncorrupt_residual": {
                    "fit_count": 10,
                    "bin_edges": [-0.04745, -0.01, 0.0, 0.01, 0.04745],
                    "bin_probabilities": [0.2, 0.3, 0.3, 0.2],
                },
                "outcome_model": {
                    "corrupt_identity_fraction": 0.25 if corrupt else 0.0,
                    "noncorrupt_reachability": {"classes": classes},
                    "corrupt_terminal": {
                        "apparent_endpoint": _quantiles(0.5),
                        "persistent_endpoint": _quantiles(0.5),
                    },
                },
            }
        )
    return {
        "schema": "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model",
        "schema_version": 2,
        "metadata": {
            "preset": "reram_array_om",
            "execution_profile": "hwa_production_cap128",
            "enable_published_corruption": corrupt,
        },
        "conditions": {
            CONDITION_KEY: {
                "fit_status": "fit",
                "reachability_fit_status": "fit",
                "adequate": True,
                "validation": {"per_target": records},
            }
        },
    }


def write_device_model(path: Path) -> Path:
    estimator = PopulationStepEstimator(
        bins=4,
        fallback_step=OM_NOMINAL_DW_MIN / 2.0,
    ).to_mapping()
    payload = {
        "schema": "ebl.ibm_reram.om_hwa_device_model",
        "schema_version": 1,
        "preset": "reram_array_om",
        "programming": {
            "controller": "adaptive",
            "start_protocol": "lower_to_target",
            "condition_key": CONDITION_KEY,
            "tolerance_step_ratio": 0.5,
            "maximum_program_pulses": 128,
            "nominal_dw_min": OM_NOMINAL_DW_MIN,
            "adaptive": {
                "eta": 0.75,
                "maximum_batch": 32,
                "epsilon": 1e-8,
                "force_one_within_steps": 2.0,
            },
        },
        "endpoint_models": {
            "continuous": _endpoint_model(corrupt=False),
            "published_corruption": _endpoint_model(corrupt=True),
        },
        "step_estimators": {
            "continuous": estimator,
            "published_corruption": estimator,
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _uniform(generator: torch.Generator, size: int, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.rand(size, generator=generator)


def noisy_population(
    keys: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
    *,
    seed: int,
    corruption_policy: str,
    kind: str,
    fingerprint: str,
):
    """Return a synthetic OM population with realistic noise and defects.

    ``kind`` selects the bound regime:
    ``wide``      logical bounds always contain [-1, 1] (literal compact).
    ``logical``   varied logical bounds, a few disjoint quads/pairs.
    ``raw``       raw-active ``a`` bounds with narrow/negative-lower cells.
    """

    hwa = _hwa()
    generator = torch.Generator().manual_seed(seed)
    size = sum(math.prod(shape) for shape in shapes)
    reference = 0.08 * torch.randn(size, generator=generator)
    if kind == "wide":
        max_bound = reference + _uniform(generator, size, 1.02, 1.40)
        min_bound = reference - _uniform(generator, size, 1.02, 1.40)
    elif kind == "logical":
        max_bound = reference + _uniform(generator, size, 0.55, 1.35)
        min_bound = reference - _uniform(generator, size, 0.55, 1.35)
        # Two disjoint identities per tensor, one high-only and one low-only,
        # placed in the same dual-rail quad (rows r and r + rows/2).  Odd
        # tensors swap the roles, so the plus/minus identities at the same
        # coordinate of a differential pair are disjoint as well.
        offset = 0
        for tensor_index, shape in enumerate(shapes):
            count = math.prod(shape)
            high = offset + 1
            low = offset + (shape[0] // 2) * shape[1] + 1
            if tensor_index % 2:
                high, low = low, high
            min_bound[high] = reference[high] + 0.5
            max_bound[high] = reference[high] + 0.8
            min_bound[low] = reference[low] - 0.8
            max_bound[low] = reference[low] - 0.5
            offset += count
    elif kind == "reset":
        # RESET bounds near the bottom of the window so commissioned targets
        # stay inside [0, 1]; one narrow identity leaves exactly one quad
        # partially outside its hidden support (audit-only fallback).
        max_bound = reference + _uniform(generator, size, 0.80, 1.00)
        min_bound = reference - _uniform(generator, size, 0.80, 1.00)
        max_bound[9] = reference[9] - 0.78
    elif kind == "raw":
        a_min = hwa.IBM_OM_RAW_ACTIVE_A_MIN
        min_bound = _uniform(generator, size, -1.6, -0.4)
        max_bound = _uniform(generator, size, 0.4, 1.6)
        # One narrow cell (capacity below D90) and one cell whose lower bound
        # lies below the frozen coordinate origin (negative lower g).
        min_bound[2] = 0.30
        max_bound[2] = 0.45
        min_bound[5] = a_min - 0.05
    else:  # pragma: no cover - fixture guard
        raise ValueError(kind)
    dwmin_up = OM_NOMINAL_DW_MIN * _uniform(generator, size, 0.5, 1.5)
    dwmin_down = OM_NOMINAL_DW_MIN * _uniform(generator, size, 0.5, 1.5)
    published = torch.zeros(size, dtype=torch.bool)
    if kind == "reset":
        published[3] = True
    elif kind != "raw":
        published[3] = True
        published[size - 2] = True
    if corruption_policy == "published":
        corrupt = published.clone()
        for index in torch.where(corrupt)[0].tolist():
            collapsed = float(reference[index]) + 0.1
            max_bound[index] = collapsed
            min_bound[index] = collapsed
            dwmin_up[index] = 0.0
            dwmin_down[index] = 0.0
    else:
        corrupt = torch.zeros(size, dtype=torch.bool)
    return hwa.IbmReramArrayPopulation(
        assignment_seed=seed,
        corruption_policy=corruption_policy,
        binding_keys=tuple(keys),
        binding_shapes=tuple(tuple(shape) for shape in shapes),
        binding_sampling_seeds=tuple(1000 + index for index in range(len(keys))),
        donor_sampling_seeds=tuple(2000 + index for index in range(len(keys))),
        nominal_dw_min=OM_NOMINAL_DW_MIN,
        dw_min_std=OM_DW_MIN_STD,
        write_noise_std=OM_WRITE_NOISE_STD,
        max_bound=max_bound.to(torch.float32),
        min_bound=min_bound.to(torch.float32),
        dwmin_up=dwmin_up.to(torch.float32),
        dwmin_down=dwmin_down.to(torch.float32),
        reference=reference.to(torch.float32),
        corrupt=corrupt,
        published_corrupt=published,
        fingerprint=fingerprint,
        aihwkit_version="1.1.0",
    )


# --------------------------------------------------------------------------
# Bindings
# --------------------------------------------------------------------------

DUAL_RAIL_LAYOUT = {"base.dense_weight.0": "halves", "base.dense_weight.1": "paired"}
SINGLE_SHAPES = (("base.dense_weight.0", (8, 8)), ("base.dense_weight.1", (8, 4)))
DIFFERENTIAL_SHAPES = (
    ("base.conductance_plus.0", (8, 8), "conductance_plus"),
    ("base.conductance_minus.0", (8, 8), "conductance_minus"),
    ("base.conductance_plus.1", (8, 4), "conductance_plus"),
    ("base.conductance_minus.1", (8, 4), "conductance_minus"),
)


def _dense(shape: tuple[int, int], gmin: float, gmax: float) -> DenseWeight:
    return DenseWeight(
        (shape[0],),
        (shape[1],),
        1.0,
        "cpu",
        clamp=True,
        clamp_min=gmin,
        clamp_max=gmax,
    )


def make_bindings(
    *, differential: bool, gmin: float, gmax: float, seed: int
) -> tuple[ParameterBinding, ...]:
    generator = torch.Generator().manual_seed(seed)
    bindings = []
    if differential:
        for key, shape, role in DIFFERENTIAL_SHAPES:
            binding = ParameterBinding(key, _dense(shape, gmin, gmax), role=role)
            bindings.append(binding)
    else:
        for key, shape in SINGLE_SHAPES:
            bindings.append(ParameterBinding(key, _dense(shape, gmin, gmax)))
    span = gmax - gmin
    for binding in bindings:
        values = gmin + span * (
            0.05 + 0.9 * torch.rand(binding.state.shape, generator=generator)
        )
        binding.state.copy_(values)
    return tuple(bindings)


def _perturb(bindings, *, step: int, gmin: float, gmax: float) -> None:
    generator = torch.Generator().manual_seed(7000 + step)
    with torch.no_grad():
        for binding in bindings:
            delta = 0.04 * (gmax - gmin) * torch.randn(
                binding.state.shape, generator=generator
            )
            binding.state.add_(delta).clamp_(gmin + 1e-4, gmax - 1e-4)


# --------------------------------------------------------------------------
# HWA modifier cases
# --------------------------------------------------------------------------

HWA_CASES: dict[str, dict[str, Any]] = {}


def _hwa_case(name: str, **kwargs: Any) -> None:
    HWA_CASES[name] = kwargs


def _register_hwa_cases() -> None:
    _hwa_case("literal_compact_published", mapping="literal_global", execution="compact_endpoint", policy="published", noisy=False, kind="wide")
    _hwa_case("literal_compact_repaired_noisy", mapping="literal_global", execution="compact_endpoint", policy="counterfactual_repaired", noisy=True, kind="wide", gain=2.5)
    _hwa_case("literal_pulse_published", mapping="literal_global", execution="pulse_resolved", policy="published", noisy=True, kind="logical")
    _hwa_case("quad_compact_published_margin", mapping="dual_rail_quad_common_window", execution="compact_endpoint", policy="published", noisy=False, kind="logical", margin=0.05)
    _hwa_case("quad_compact_repaired_noisy", mapping="dual_rail_quad_common_window", execution="compact_endpoint", policy="counterfactual_repaired", noisy=True, kind="logical")
    _hwa_case("quad_pulse_published_margin", mapping="dual_rail_quad_common_window", execution="pulse_resolved", policy="published", noisy=True, kind="logical", margin=0.05)
    _hwa_case("pair_compact_repaired_margin", mapping="differential_pair_common_window", execution="compact_endpoint", policy="counterfactual_repaired", noisy=False, kind="logical", margin=0.05)
    _hwa_case("pair_pulse_published", mapping="differential_pair_common_window", execution="pulse_resolved", policy="published", noisy=True, kind="logical")
    _hwa_case("reset_compact_q9_published", mapping="shared_reset_relative_quad", execution="compact_endpoint", policy="published", noisy=True, kind="reset", reset_mode="quantized_9_level")
    _hwa_case("reset_compact_continuous_repaired", mapping="shared_reset_relative_quad", execution="compact_endpoint", policy="counterfactual_repaired", noisy=False, kind="reset", reset_mode="continuous")
    _hwa_case("reset_pulse_q9_published", mapping="shared_reset_relative_quad", execution="pulse_resolved", policy="published", noisy=True, kind="reset", reset_mode="quantized_9_level", gain=1.75)
    _hwa_case("reset_pulse_continuous_repaired", mapping="shared_reset_relative_quad", execution="pulse_resolved", policy="counterfactual_repaired", noisy=True, kind="reset", reset_mode="continuous")
    _hwa_case("reset_compact_q9_repaired", mapping="shared_reset_relative_quad", execution="compact_endpoint", policy="counterfactual_repaired", noisy=True, kind="reset", reset_mode="quantized_9_level", gain=1.25)
    _hwa_case("reset_pulse_q9_repaired", mapping="shared_reset_relative_quad", execution="pulse_resolved", policy="counterfactual_repaired", noisy=True, kind="reset", reset_mode="quantized_9_level", gain=1.75)
    _hwa_case("raw_mapped_q7", mapping="raw_active_p90_quad", execution="mapped_target", policy="counterfactual_repaired", noisy=False, kind="raw", raw_mode="quantized_7_level")
    _hwa_case("raw_mapped_continuous", mapping="raw_active_p90_quad", execution="mapped_target", policy="counterfactual_repaired", noisy=False, kind="raw", raw_mode="continuous")
    _hwa_case("raw_pulse_q7", mapping="raw_active_p90_quad", execution="pulse_resolved", policy="counterfactual_repaired", noisy=True, kind="raw", raw_mode="quantized_7_level", gain=3.0)
    _hwa_case("raw_pulse_continuous", mapping="raw_active_p90_quad", execution="pulse_resolved", policy="counterfactual_repaired", noisy=True, kind="raw", raw_mode="continuous")


_register_hwa_cases()


def hwa_config_kwargs(case: dict[str, Any]) -> dict[str, Any]:
    mapping = case["mapping"]
    raw = mapping == "raw_active_p90_quad"
    kwargs: dict[str, Any] = {
        "execution": case["execution"],
        "assignment_seed": 83001,
        "endpoint_seed": 83002,
        "corruption_policy": case["policy"],
        "noisy_evaluation": case["noisy"],
        "target_mapping": mapping,
        "common_window_margin_fraction": case.get("margin", 0.0),
        "forward_logit_gain": case.get("gain"),
    }
    if raw:
        kwargs.update(controller="one_pulse", endpoint_policy="preserve")
    if mapping in {
        "dual_rail_quad_common_window",
        "shared_reset_relative_quad",
        "raw_active_p90_quad",
    }:
        kwargs["dual_rail_layout_by_parameter"] = dict(DUAL_RAIL_LAYOUT)
    if mapping == "shared_reset_relative_quad":
        kwargs.update(
            reset_relative_mode=case["reset_mode"],
            reset_relative_contrast_step=0.125,
            reset_read_samples=4,
            reset_guard_standard_errors=1.0,
        )
    if raw:
        kwargs.update(
            raw_active_mode=case["raw_mode"],
            raw_active_unsupported_quad_policy="structural_failure",
        )
    return kwargs


def _case_bounds(case: dict[str, Any]) -> tuple[float, float]:
    return (0.0, 1.0) if case["mapping"] == "raw_active_p90_quad" else (0.1, 1.0)


def build_hwa_modifier(case: dict[str, Any], workdir: Path, *, bindings=None):
    hwa = _hwa()
    gmin, gmax = _case_bounds(case)
    differential = case["mapping"] == "differential_pair_common_window"
    if bindings is None:
        bindings = make_bindings(
            differential=differential, gmin=gmin, gmax=gmax, seed=31
        )
    population = noisy_population(
        tuple(binding.key for binding in bindings),
        tuple(tuple(binding.state.shape) for binding in bindings),
        seed=501,
        corruption_policy=case["policy"],
        kind=case["kind"],
        fingerprint=f"golden-{case['kind']}-{case['policy']}",
    )
    device_model = write_device_model(workdir / "device_model.json")
    config = hwa.IbmReramHwaConfig(**hwa_config_kwargs(case))
    with patched_population(population):
        modifier = hwa.IbmReramHwaParameterModifier(
            bindings,
            config,
            device_model_path=device_model,
            conductance_min=gmin,
            conductance_max=gmax,
        )
    return modifier, bindings, population, device_model


def _context_record(modifier, bindings, context) -> dict[str, Any]:
    with context():
        inside = [binding.state.detach().clone() for binding in bindings]
    return {
        "inside": inside,
        "restored": [binding.state.detach().clone() for binding in bindings],
        "programming_report": _snapshot(modifier.programming_report),
        "deployment": _snapshot(modifier.last_deployment_bundle),
        "state": _snapshot(modifier.state_dict()),
    }


def run_hwa_case(name: str) -> dict[str, Any]:
    case = HWA_CASES[name]
    gmin, gmax = _case_bounds(case)
    with tempfile.TemporaryDirectory() as raw_dir:
        workdir = Path(raw_dir)
        built = _outcome(lambda: build_hwa_modifier(case, workdir))
        if not built["ok"]:
            return {"construction": built}
        modifier, bindings, population, _device_model = built["value"]
        record: dict[str, Any] = {
            "config": asdict(modifier.config),
            "population_fingerprint": modifier.population_fingerprint,
            "reset_commissioning_bundle": _snapshot(modifier.reset_commissioning_bundle),
            "reset_commissioning_report": _snapshot(modifier.reset_commissioning_report),
            "initial_state": _snapshot(modifier.state_dict()),
            "preflight": _outcome(lambda: _snapshot(modifier.preflight_target_mapping())),
        }
        steps = []
        for step in range(3):
            steps.append(
                _outcome(
                    lambda: _context_record(modifier, bindings, modifier.training_context)
                )
            )
            _perturb(bindings, step=step, gmin=gmin, gmax=gmax)
        for _step in range(2):
            steps.append(
                _outcome(
                    lambda: _context_record(modifier, bindings, modifier.evaluation_context)
                )
            )
        record["steps"] = steps
        saved = _snapshot(modifier.state_dict())
        replay_bindings = make_bindings(
            differential=case["mapping"] == "differential_pair_common_window",
            gmin=gmin,
            gmax=gmax,
            seed=31,
        )
        for source, target in zip(bindings, replay_bindings):
            target.state.data.copy_(source.state)
        replay, *_rest = build_hwa_modifier(case, workdir, bindings=replay_bindings)
        replay.load_state_dict(saved)
        record["replay"] = _outcome(
            lambda: _context_record(replay, replay_bindings, replay.training_context)
        )
        record["final_clean"] = [binding.state.detach().clone() for binding in bindings]
        return record


# --------------------------------------------------------------------------
# End-to-end STE gradient cases (runtime training loop on a tiny DRN)
# --------------------------------------------------------------------------

STE_CASES = {
    "ste_literal_compact": "literal_compact_repaired_noisy",
    "ste_quad_compact": "quad_compact_repaired_noisy",
    "ste_pair_compact": "pair_compact_repaired_margin",
    "ste_reset_compact": "reset_compact_q9_repaired",
    "ste_reset_pulse": "reset_pulse_q9_repaired",
    "ste_raw_mapped": "raw_mapped_q7",
    "ste_raw_pulse": "raw_pulse_q7",
}


class _LinearTeacher:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(77)
        self.matrix = torch.randn(4, 2, generator=generator)

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.matrix


def _tiny_stack(case: dict[str, Any]):
    from experiments.mnist_relu_drn.components import build_student_stack

    gmin, gmax = _case_bounds(case)
    differential = case["mapping"] == "differential_pair_common_window"
    spec = SimpleNamespace(
        runtime=SimpleNamespace(device="cpu"),
        model=SimpleNamespace(
            dims=(8, 8, 4),
            input_gain=1.0,
            non_linearity="perfect_diode",
            exponential_diode_param={},
            quadratic_diode_param={},
            hard_sigmoid_param={},
            voltage_amp=4.0,
            current_amp=0.25,
            conductance_min=gmin,
            conductance_max=gmax,
            encoding="differential" if differential else "single",
            include_biases=False,
        ),
        solver=SimpleNamespace(
            inference_iterations=4,
            training_iterations=4,
            mode="asynchronous",
            overrelaxation_factor=1.1,
        ),
        settings=SimpleNamespace(learning_rates=(0.05, 0.05)),
    )
    return build_student_stack(spec, enable_measured=False)


def run_ste_case(name: str) -> dict[str, Any]:
    from experiments.mnist_relu_drn import runtime

    case = HWA_CASES[STE_CASES[name]]
    gmin, gmax = _case_bounds(case)
    stack = _tiny_stack(case)
    bindings = tuple(stack.bundle.catalog.trainable)
    generator = torch.Generator().manual_seed(91)
    span = gmax - gmin
    with torch.no_grad():
        for binding in bindings:
            binding.state.copy_(
                gmin + span * (0.05 + 0.9 * torch.rand(binding.state.shape, generator=generator))
            )
    inputs = torch.rand(3, 6, 4, generator=generator)
    labels = torch.randint(0, 2, (3, 6), generator=generator)
    loader = [(inputs[index], labels[index]) for index in range(3)]
    record: dict[str, Any] = {"inside": [], "gradients": [], "before_step": []}

    original_equilibrium = stack.minimizer.compute_equilibrium
    original_step = stack.optimizer.step

    def recording_equilibrium(*args, **kwargs):
        record["inside"].append([binding.state.detach().clone() for binding in bindings])
        return original_equilibrium(*args, **kwargs)

    def recording_step(*args, **kwargs):
        record["gradients"].append(
            [binding.state.grad.detach().clone() for binding in bindings]
        )
        record["before_step"].append([binding.state.detach().clone() for binding in bindings])
        return original_step(*args, **kwargs)

    stack.minimizer.compute_equilibrium = recording_equilibrium
    stack.optimizer.step = recording_step
    with tempfile.TemporaryDirectory() as raw_dir:
        built = _outcome(
            lambda: build_hwa_modifier(case, Path(raw_dir), bindings=bindings)
        )
        if not built["ok"]:
            return {"construction": built}
        modifier = built["value"][0]
        metrics = _outcome(
            lambda: runtime._train_epoch(
                stack,
                _LinearTeacher(),
                loader,
                maximum_batches=None,
                modifier=modifier,
            )
        )
    record["metrics"] = metrics
    record["after"] = [binding.state.detach().clone() for binding in bindings]
    record["modifier_state"] = _snapshot(modifier.state_dict())
    record["programming_report"] = _snapshot(modifier.programming_report)
    return record


# --------------------------------------------------------------------------
# Mapper and preflight cases (pure functions)
# --------------------------------------------------------------------------

MAPPING_CASES = (
    "literal_compact_published",
    "quad_compact_published_margin",
    "quad_compact_repaired_noisy",
    "pair_compact_repaired_margin",
    "reset_compact_q9_published",
    "reset_compact_q9_repaired",
    "reset_compact_continuous_repaired",
    "raw_mapped_q7",
    "raw_mapped_continuous",
)


def _mutations(report: dict[str, Any]) -> list[tuple[str, Any]]:
    mutations: list[tuple[str, Any]] = []
    for key, value in sorted(report.items()):
        if isinstance(value, bool):
            mutations.append((key, not value))
        elif isinstance(value, int):
            mutations.append((key, value + 1))
        elif isinstance(value, float):
            mutations.append((key, value + 0.5))
        elif isinstance(value, str):
            mutations.append((key, value + "-mutated"))
        elif value is None:
            mutations.append((key, 1))
    return mutations


def run_mapping_case(name: str) -> dict[str, Any]:
    hwa = _hwa()
    case = HWA_CASES[name]
    with tempfile.TemporaryDirectory() as raw_dir:
        modifier, bindings, population, _model = build_hwa_modifier(case, Path(raw_dir))
        kwargs = hwa_config_kwargs(case)
        commissioning = modifier._reset_commissioning
    generator = torch.Generator().manual_seed(4242)
    record: dict[str, Any] = {}
    for trial in range(3):
        fractions = torch.rand(population.size, generator=generator)
        result = _outcome(
            lambda: _snapshot(
                hwa.map_ibm_reram_array_targets(
                    fractions,
                    population,
                    target_mapping=kwargs["target_mapping"],
                    dual_rail_layout_by_parameter=kwargs.get("dual_rail_layout_by_parameter"),
                    common_window_margin_fraction=kwargs["common_window_margin_fraction"],
                    reset_commissioning=commissioning,
                    reset_relative_mode=kwargs.get("reset_relative_mode"),
                    reset_relative_contrast_step=kwargs.get("reset_relative_contrast_step"),
                    raw_active_mode=kwargs.get("raw_active_mode"),
                    raw_active_unsupported_quad_policy=kwargs.get(
                        "raw_active_unsupported_quad_policy"
                    ),
                )
            )
        )
        record[f"map_{trial}"] = result
        if result["ok"]:
            report = result["value"][1]
            record[f"preflight_{trial}"] = _outcome(
                lambda: hwa.validate_ibm_reram_target_mapping_preflight(report)
            )
            if trial == 0:
                mutated = {}
                for key, value in _mutations(report):
                    changed = dict(report)
                    changed[key] = value
                    mutated[key] = _outcome(
                        lambda: hwa.validate_ibm_reram_target_mapping_preflight(changed)
                    )
                record["preflight_mutations"] = mutated
    # Argument validation of the public mapper.
    bad = {}
    base_fractions = torch.rand(population.size, generator=generator)
    for label, override in (
        ("unknown_mapping", {"target_mapping": "unknown"}),
        ("bad_margin", {"common_window_margin_fraction": 0.5}),
        ("negative_margin", {"common_window_margin_fraction": -0.1}),
        ("layouts_none", {"dual_rail_layout_by_parameter": None}),
        ("raw_mode_on_other", {"raw_active_mode": "continuous"}),
        ("reset_mode_on_other", {"reset_relative_mode": "continuous"}),
    ):
        arguments = {
            "target_mapping": kwargs["target_mapping"],
            "dual_rail_layout_by_parameter": kwargs.get("dual_rail_layout_by_parameter"),
            "common_window_margin_fraction": kwargs["common_window_margin_fraction"],
            "reset_commissioning": commissioning,
            "reset_relative_mode": kwargs.get("reset_relative_mode"),
            "reset_relative_contrast_step": kwargs.get("reset_relative_contrast_step"),
            "raw_active_mode": kwargs.get("raw_active_mode"),
            "raw_active_unsupported_quad_policy": kwargs.get(
                "raw_active_unsupported_quad_policy"
            ),
        }
        arguments.update(override)
        bad[label] = _outcome(
            lambda: _snapshot(
                hwa.map_ibm_reram_array_targets(base_fractions, population, **arguments)
            )
        )
    record["argument_checks"] = bad
    outside = base_fractions.clone()
    outside[0] = 1.5
    record["outside_unit_interval"] = _outcome(
        lambda: _snapshot(
            hwa.map_ibm_reram_array_targets(
                outside,
                population,
                target_mapping=kwargs["target_mapping"],
                dual_rail_layout_by_parameter=kwargs.get("dual_rail_layout_by_parameter"),
                common_window_margin_fraction=kwargs["common_window_margin_fraction"],
                reset_commissioning=commissioning,
                reset_relative_mode=kwargs.get("reset_relative_mode"),
                reset_relative_contrast_step=kwargs.get("reset_relative_contrast_step"),
                raw_active_mode=kwargs.get("raw_active_mode"),
                raw_active_unsupported_quad_policy=kwargs.get(
                    "raw_active_unsupported_quad_policy"
                ),
            )
        )
    )
    return record


# --------------------------------------------------------------------------
# Config corpus and accept/reject grid
# --------------------------------------------------------------------------


def _example_configs() -> list[Path]:
    output = subprocess.run(
        ["grep", "-rl", '"ibm_reram_om_program_verify"', "examples"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    return sorted(ROOT / item for item in output)


def run_config_corpus() -> dict[str, Any]:
    from experiments.definitions import load_experiment_config, resolve_experiment_config
    from experiments.schema import to_plain_data

    record: dict[str, Any] = {}
    for path in _example_configs():
        relative = str(path.relative_to(ROOT))
        entry: dict[str, Any] = {
            "document": _outcome(lambda: to_plain_data(load_experiment_config(path)[1]))
        }
        for mode in ("train", "validate"):
            entry[mode] = _outcome(
                lambda: to_plain_data(resolve_experiment_config(path, mode)[1])
            )
        record[relative] = entry
    return record


_GRID_ALTERNATIVES: dict[str, tuple[Any, ...]] = {
    "execution": ("mapped_target", "compact_endpoint", "pulse_resolved", "bogus"),
    "corruption_policy": ("published", "counterfactual_repaired", "bogus"),
    "noisy_evaluation": (True, False, 1),
    "endpoint_policy": ("clip_0_1", "preserve"),
    "target_out_of_support": ("error", "clip"),
    "preset": ("reram_array_om", "other"),
    "controller": ("adaptive", "one_pulse"),
    "start_protocol": ("lower_to_target", "upper_to_target"),
    "tolerance_step_ratio": (0.5, 0.25),
    "maximum_program_pulses": (128, 64),
    "target_mapping": (
        "literal_global",
        "dual_rail_quad_common_window",
        "differential_pair_common_window",
        "shared_reset_relative_quad",
        "raw_active_p90_quad",
        "bogus",
    ),
    "dual_rail_layout_by_parameter": (None, {"base.dense_weight.0": "halves"}, {"x": "bad"}),
    "common_window_margin_fraction": (0.0, 0.1, 0.5, True),
    "reset_relative_mode": (None, "continuous", "quantized_9_level", "bogus"),
    "reset_relative_contrast_step": (None, 0.125, 0.0, 0.75),
    "reset_read_samples": (None, 1, 4),
    "reset_guard_standard_errors": (None, 1.0, -1.0),
    "raw_active_mode": (None, "continuous", "quantized_7_level", "bogus"),
    "raw_active_unsupported_quad_policy": (None, "structural_failure", "donor"),
    "forward_logit_gain": (None, 2.0, 0.0, True),
    "assignment_seed": (0, -1, 2**63),
    "endpoint_seed": (5, True),
}


def _normalized_config(config) -> dict[str, Any]:
    return asdict(config)


def run_config_grid() -> dict[str, Any]:
    hwa = _hwa()
    record: dict[str, Any] = {}
    for name, case in HWA_CASES.items():
        base = hwa_config_kwargs(case)
        entry = {"base": _outcome(lambda: _normalized_config(hwa.IbmReramHwaConfig(**base)))}
        for field, values in _GRID_ALTERNATIVES.items():
            for value in values:
                mutated = dict(base)
                mutated[field] = value
                entry[f"{field}={value!r}"] = _outcome(
                    lambda: _normalized_config(hwa.IbmReramHwaConfig(**mutated))
                )
        record[name] = entry
    return record


def run_parser_grid() -> dict[str, Any]:
    from experiments.mnist_relu_drn.config import _parse_weight_modifier
    from experiments.schema import to_plain_data

    record: dict[str, Any] = {}
    for name, case in HWA_CASES.items():
        base = hwa_config_kwargs(case)
        base.setdefault("endpoint_policy", "clip_0_1")
        base.setdefault("controller", "adaptive")
        parameters = {
            "target_out_of_support": "error",
            "preset": "reram_array_om",
            "start_protocol": "lower_to_target",
            "tolerance_step_ratio": 0.5,
            "maximum_program_pulses": 128,
            **base,
        }
        if parameters.get("forward_logit_gain") is None:
            parameters.pop("forward_logit_gain")
        if parameters["target_mapping"] in {
            "dual_rail_quad_common_window",
            "shared_reset_relative_quad",
            "raw_active_p90_quad",
        }:
            parameters["dual_rail_layout_by_parameter"] = dict(DUAL_RAIL_LAYOUT)
        for key in (
            "reset_relative_mode",
            "reset_relative_contrast_step",
            "reset_read_samples",
            "reset_guard_standard_errors",
            "raw_active_mode",
            "raw_active_unsupported_quad_policy",
        ):
            if parameters.get(key) is None:
                parameters.pop(key, None)
        entry = {
            "base": _outcome(
                lambda: to_plain_data(
                    _parse_weight_modifier(
                        {"type": "ibm_reram_om_program_verify", "parameters": parameters},
                        "config.modes.train.weight_modifier",
                    )
                )
            )
        }
        for field, values in _GRID_ALTERNATIVES.items():
            for value in values:
                mutated = dict(parameters)
                mutated[field] = value
                entry[f"{field}={value!r}"] = _outcome(
                    lambda: to_plain_data(
                        _parse_weight_modifier(
                            {"type": "ibm_reram_om_program_verify", "parameters": mutated},
                            "config.modes.train.weight_modifier",
                        )
                    )
                )
            dropped = dict(parameters)
            dropped.pop(field, None)
            entry[f"{field}=<absent>"] = _outcome(
                lambda: to_plain_data(
                    _parse_weight_modifier(
                        {"type": "ibm_reram_om_program_verify", "parameters": dropped},
                        "config.modes.train.weight_modifier",
                    )
                )
            )
        record[name] = entry
    return record


# --------------------------------------------------------------------------
# Recovery cases (built from an in-harness HWA deployment)
# --------------------------------------------------------------------------

RECOVERY_SOURCE_CASE = "reset_pulse_q9_repaired"
RECOVERY_LAYOUT = dict(DUAL_RAIL_LAYOUT)


def _recovery_parameters(method: str, *, fast_backend: str | None, device_model: Path, percentile: float | None = None) -> dict[str, Any]:
    keys = tuple(RECOVERY_LAYOUT)
    result: dict[str, Any] = {
        "method": method,
        "slow_pulse_budget_per_cell": 2.0,
        "pulse_step_ratio": 0.04745,
        "update_seed": 123,
        "direct_probability_scale": None,
        "fast_backend": fast_backend,
        "fast_endpoint_seed": 456 if fast_backend == "physical_om" else None,
        "fast_reset_read_samples": 3 if fast_backend == "physical_om" else None,
        "fast_reset_guard_standard_errors": 0.5 if fast_backend == "physical_om" else None,
        "expected_device_model_sha256": sha256_file(device_model),
        "source_dual_rail_layout_by_parameter": dict(RECOVERY_LAYOUT),
    }
    if percentile is not None:
        result["direct_gradient_magnitude_percentile"] = percentile
    if method == "ttv2":
        result.update(
            {
                "ttv2_transfer_every": 1,
                "ttv2_gamma0": 2.0,
                "ttv2_fast_update_model": "symmetric_soft_bounds",
                "ttv2_fast_weight_limit": 1.0,
                "ttv2_scan_mode": "dual_rail_input_pair",
                "ttv2_buffer_threshold": 1.0,
                "ttv2_buffer_residual_mode": "subtract_dispatched",
            }
        )
    elif method == "ttv2_aihwkit_1p1_minibatch_equation":
        result.update(
            {
                "ttv2_transfer_every": 1,
                "ttv2_fast_weight_limit": 1.0,
                "ttv2_fast_lr_by_parameter": {key: 0.5 for key in keys},
                "ttv2_transfer_lr": 1.0,
                "ttv2_scale_transfer_lr": True,
                "ttv2_units_in_mbatch": True,
                "ttv2_auto_scale": False,
                "ttv2_fast_granularity_by_parameter": {key: 0.2 for key in keys},
                "ttv2_buffer_granularity": 0.5,
                "ttv2_auto_granularity": 4.0,
                "ttv2_correct_gradient_magnitudes": True,
                "ttv2_desired_bl": 1,
                "ttv2_momentum": 0.0,
                "ttv2_forget_buffer": True,
                "ttv2_cap_scope": "per_parameter_proportional",
                "ttv2_cursor_policy": "zero",
                "ttv2_in_chop_probability": 0.0,
            }
        )
    return result


RECOVERY_CASES: dict[str, dict[str, Any]] = {
    "rail_refresh": {"method": "rail_refresh", "fast_backend": None},
    "direct_pulse": {"method": "direct_pulse", "fast_backend": None, "scale": 3.0},
    "direct_pulse_p90": {"method": "direct_pulse", "fast_backend": None, "scale": 3.0, "percentile": 90.0},
    "tiki_taka_ideal": {"method": "tiki_taka", "fast_backend": "ideal"},
    "tiki_taka_physical": {"method": "tiki_taka", "fast_backend": "physical_om"},
    "ttv2_ideal": {"method": "ttv2", "fast_backend": "ideal"},
    "ttv2_aihwkit": {"method": "ttv2_aihwkit_1p1_minibatch_equation", "fast_backend": "ideal"},
}

_RECOVERY_STEPS = 4


def _recovery_source(workdir: Path):
    case = HWA_CASES[RECOVERY_SOURCE_CASE]
    modifier, bindings, population, device_model = build_hwa_modifier(case, workdir)
    with modifier.evaluation_context():
        pass
    deployment = modifier.last_deployment_bundle
    return bindings, population, device_model, deployment


def _build_recovery(spec: dict[str, Any], workdir: Path):
    from training.ibm_reram_recovery import IbmOmDeployedRecovery

    bindings, population, device_model, deployment = _recovery_source(workdir)
    gmin, gmax = _case_bounds(HWA_CASES[RECOVERY_SOURCE_CASE])
    fast_population = None
    if spec["fast_backend"] == "physical_om":
        keys = tuple(f"fast_pair.{binding.key}" for binding in bindings)
        shapes = tuple(
            (2 * binding.state.shape[0], binding.state.shape[1]) for binding in bindings
        )
        fast_population = noisy_population(
            keys,
            shapes,
            seed=909,
            corruption_policy="counterfactual_repaired",
            kind="wide",
            fingerprint="golden-fast",
        )
    requested = deployment.get("mapped_target", deployment.get("requested_target"))
    recovery = IbmOmDeployedRecovery(
        bindings,
        slow_population=population,
        source_raw_apparent=deployment["raw_apparent_endpoint"],
        source_apparent=deployment["apparent_endpoint"],
        source_persistent=deployment["persistent_endpoint"],
        requested_target=requested,
        reset_baseline=deployment["reset_commissioning"]["observed"]["baseline"],
        generator_state_after_programming=deployment["generator_state_after_programming"],
        parameters=_recovery_parameters(
            spec["method"],
            fast_backend=spec["fast_backend"],
            device_model=device_model,
            percentile=spec.get("percentile"),
        ),
        learning_rates=(0.6, 0.4),
        conductance_min=gmin,
        conductance_max=gmax,
        total_steps=_RECOVERY_STEPS,
        source_deployment_sha256="d" * 64,
        device_model_path=device_model,
        fast_population=fast_population,
    )
    if spec["method"] == "direct_pulse":
        if spec.get("percentile") is not None:
            recovery.set_direct_gradient_thresholds(
                {binding.key: 0.35 for binding in bindings},
                report={"percentile": spec["percentile"], "calibration_sha256": "a" * 64},
            )
        recovery.set_direct_probability_scale(spec["scale"], report={"fixture": True})
    return recovery, bindings


def _recovery_gradients(bindings, step: int) -> None:
    generator = torch.Generator().manual_seed(6000 + step)
    for binding in bindings:
        binding.state.grad = torch.randn(binding.state.shape, generator=generator)


def _recovery_snapshot(recovery, bindings) -> dict[str, Any]:
    return {
        "bindings": [binding.state.detach().clone() for binding in bindings],
        "apparent": recovery.current_apparent_endpoint(clipped=False).clone(),
        "persistent": recovery.current_persistent_endpoint().clone(),
    }


def run_recovery_case(name: str) -> dict[str, Any]:
    spec = RECOVERY_CASES[name]
    with tempfile.TemporaryDirectory() as raw_dir:
        workdir = Path(raw_dir)
        built = _outcome(lambda: _build_recovery(spec, workdir))
        if not built["ok"]:
            return {"construction": built}
        recovery, bindings = built["value"]
        record: dict[str, Any] = {"initial": _recovery_snapshot(recovery, bindings)}
        steps = []
        midpoint = None
        for step in range(_RECOVERY_STEPS):
            if recovery.requires_gradients:
                _recovery_gradients(bindings, step)
            recovery.step()
            steps.append(_recovery_snapshot(recovery, bindings))
            if step == 1:
                midpoint = _snapshot(recovery.state_dict())
        record["steps"] = steps
        record["report"] = _snapshot(recovery.report())
        record["state"] = _snapshot(recovery.state_dict())
        record["final_bundle"] = _snapshot(recovery.final_bundle())
        resumed, resumed_bindings = _build_recovery(spec, workdir)
        resumed.load_state_dict(midpoint)
        for step in range(2, _RECOVERY_STEPS):
            if resumed.requires_gradients:
                _recovery_gradients(resumed_bindings, step)
            resumed.step()
        record["resumed_report"] = _snapshot(resumed.report())
        record["resumed_state"] = _snapshot(resumed.state_dict())
        return record


# --------------------------------------------------------------------------
# Plant RNG-order cases (direct, below the modifier)
# --------------------------------------------------------------------------


def run_plant_case() -> dict[str, Any]:
    hwa = _hwa()
    population = noisy_population(
        ("base.dense_weight.0",),
        ((4, 6),),
        seed=321,
        corruption_policy="published",
        kind="logical",
        fingerprint="golden-plant",
    )
    generator = torch.Generator().manual_seed(5)
    plant = hwa.IbmReramPulsePlant(population, generator=generator, device=torch.device("cpu"))
    record: dict[str, Any] = {
        "init_persistent": plant.persistent.clone(),
        "init_apparent": plant.apparent.clone(),
    }
    pattern = torch.Generator().manual_seed(6)
    pulses = []
    for _ in range(6):
        directions = torch.randint(-1, 2, (population.size,), generator=pattern).to(torch.int8)
        plant.pulse(directions)
        pulses.append((plant.persistent.clone(), plant.apparent.clone(), _generator_state(generator)))
    plant.pulse(torch.zeros(population.size, dtype=torch.int8))
    record["pulses"] = pulses
    record["after_zero_pulse"] = _generator_state(generator)
    raw_population = noisy_population(
        ("base.dense_weight.0",),
        ((4, 6),),
        seed=322,
        corruption_policy="counterfactual_repaired",
        kind="raw",
        fingerprint="golden-raw-plant",
    )
    raw_generator = torch.Generator().manual_seed(8)
    raw = hwa._RawActiveArrayPlant(raw_population, generator=raw_generator, device=torch.device("cpu"))
    conditioning = raw.condition_lower_boundary()
    record["raw_conditioning"] = {key: value.clone() for key, value in conditioning.items()}
    record["raw_generator"] = _generator_state(raw_generator)
    return record


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def all_cases() -> dict[str, Callable[[], Any]]:
    cases: dict[str, Callable[[], Any]] = {}
    for name in HWA_CASES:
        cases[f"hwa/{name}"] = lambda name=name: run_hwa_case(name)
    for name in STE_CASES:
        cases[f"ste/{name}"] = lambda name=name: run_ste_case(name)
    for name in MAPPING_CASES:
        cases[f"mapping/{name}"] = lambda name=name: run_mapping_case(name)
    for name in RECOVERY_CASES:
        cases[f"recovery/{name}"] = lambda name=name: run_recovery_case(name)
    cases["plant/rng_order"] = run_plant_case
    cases["config/corpus"] = run_config_corpus
    cases["config/dataclass_grid"] = run_config_grid
    cases["config/parser_grid"] = run_parser_grid
    return cases


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.pt"


def environment() -> dict[str, str]:
    return {"torch": torch.__version__, "python": platform.python_version()}


def capture(selected: list[str] | None = None) -> None:
    cases = all_cases()
    names = selected or list(cases)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    for name in names:
        result = cases[name]()
        path = golden_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, path)
        print(f"captured {name} -> {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    meta = {**environment(), "commit": commit, "cases": sorted(cases)}
    (GOLDEN_DIR / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

_BIT_VIEWS = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}


def _config_like(value: dict) -> bool:
    return "target_mapping" in value and "execution" in value


def assert_same(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected tensor, got {type(actual).__name__}"
        assert expected.dtype == actual.dtype, f"{path}: dtype {expected.dtype} != {actual.dtype}"
        assert tuple(expected.shape) == tuple(actual.shape), f"{path}: shape {tuple(expected.shape)} != {tuple(actual.shape)}"
        left = expected.detach().cpu().contiguous()
        right = actual.detach().cpu().contiguous()
        view = _BIT_VIEWS.get(left.dtype)
        if view is not None:
            left = left.view(view)
            right = right.view(view)
        assert torch.equal(left, right), f"{path}: tensor values differ"
        return
    assert type(expected) is type(actual), f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        left_keys = set(expected) - SOURCE_HASH_KEYS
        right_keys = set(actual) - SOURCE_HASH_KEYS
        if NEW_CONFIG_KEYS and _config_like(actual):
            right_keys -= set(NEW_CONFIG_KEYS)
        if NEW_CONFIG_KEYS and _config_like(expected):
            left_keys -= set(NEW_CONFIG_KEYS)
        assert left_keys == right_keys, (
            f"{path}: keys differ; missing={sorted(map(str, left_keys - right_keys))} "
            f"extra={sorted(map(str, right_keys - left_keys))}"
        )
        for key in expected:
            if key in left_keys:
                assert_same(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual), f"{path}: length {len(expected)} != {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_same(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        same = (math.isnan(expected) and math.isnan(actual)) or (
            expected == actual and math.copysign(1.0, expected) == math.copysign(1.0, actual)
        )
        assert same, f"{path}: {expected!r} != {actual!r}"
        return
    assert expected == actual, f"{path}: {expected!r} != {actual!r}"


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "capture":
        capture(sys.argv[2:] or None)
    else:
        print("usage: python tests/ibm_om_golden_cases.py capture [case ...]")
        print("\n".join(sorted(all_cases())))
