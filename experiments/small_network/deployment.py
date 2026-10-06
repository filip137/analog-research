"""One-time deployment of a new ``small_drn.v1`` run onto its device model.

After :mod:`initialization` has loaded the logical start state, a new run
(never a resume) may program it once before training:

1. adapter runs started from base weights program the frozen base
   conductances (Wan 2022 or a generic device-noise model);
2. program-verify writes the loaded trainable targets;
3. measured cohort B records the source checkpoint's validation before its
   projection;
4. measured backends project or reset onto their measured curves;
5. adapter and device-backed runs record the deployed state's validation.

Validations here restore the solver layer states afterwards: training keeps
``reset_input=False``, so they must not seed the first training minibatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.artifacts import RunStore, sha256_file
from experiments.small_network.components import TrainRuntime
from experiments.small_network.config import TrainSpec
from experiments.small_network.reporting import held_out_metrics
from model.resistive.builders import ParameterCatalog
from model.resistive.device_config import Wan2022ProgrammingConfig
from model.resistive.digital_low_rank_config import (
    parse_digital_low_rank_adapter,
)
from model.resistive.passive_layerwise_low_rank_config import (
    parse_passive_layerwise_low_rank_adapter,
)
from training.device_programming import (
    program_device_base_conductances,
    program_wan2022_base_conductance,
    program_wan2022_base_conductances,
)
from training.ibm_om_fp32_bounds import IbmOmFp32BoundsOptimizer
from training.measured_trace import (
    MeasuredCohortAOptimizer,
    MeasuredCohortBLoRAOptimizer,
    MeasuredCohortBOptimizer,
    MeasuredTraceOptimizer,
)
from training.program_verify import ProgramVerifyOptimizer


@dataclass(frozen=True)
class Deployment:
    device_programming: Mapping[str, Any] | None = None
    pre_deployment_validation: Mapping[str, Any] | None = None
    initial_validation: Mapping[str, Any] | None = None


def deploy(
    request: Any,
    spec: TrainSpec,
    runtime: TrainRuntime,
    store: RunStore,
) -> Deployment:
    """Program a new run's loaded state and record what it became."""

    if request.resume is not None:
        return Deployment()
    device_programming = None
    if request.base_weights is not None:
        device_programming = _program_adapter_base(spec, runtime)
    if isinstance(runtime.optimizer, ProgramVerifyOptimizer):
        device_programming = {
            "base": (
                None
                if device_programming is None
                else dict(device_programming)
            ),
            "trainable_initial_write": (
                runtime.optimizer.initialize_from_loaded_targets()
            ),
            "semantics": (
                "digital shadow target with a noisy program-and-verify "
                "write after every optimizer step"
            ),
            "pulse_model": False,
        }
    pre_deployment_validation = None
    if isinstance(runtime.optimizer, MeasuredCohortBOptimizer):
        pre_deployment_validation = _validation_preserving_layer_state(
            spec, runtime, split="pre_deployment_validation"
        )
        store.append_metric(
            {
                "mode": "pre_deployment",
                "source_weights": {
                    "path": str(Path(request.weights).expanduser().resolve()),
                    "sha256": sha256_file(request.weights),
                },
                "validation": dict(pre_deployment_validation),
            }
        )
    if isinstance(runtime.optimizer, MeasuredTraceOptimizer):
        device_programming = _initialize_measured(request, runtime.optimizer)
    initial_validation = None
    if spec.common.model.adapter.type in {
        "digital_low_rank",
        "passive_layerwise_low_rank",
    } or isinstance(
        runtime.optimizer,
        (
            ProgramVerifyOptimizer,
            MeasuredTraceOptimizer,
            IbmOmFp32BoundsOptimizer,
        ),
    ):
        initial_validation = _validation_preserving_layer_state(
            spec, runtime, split="initial_validation"
        )
        store.append_metric(
            {
                "mode": "initialization",
                "device_programming": (
                    None
                    if device_programming is None
                    else dict(device_programming)
                ),
                "om_fp32_bounds": (
                    runtime.optimizer.bounds_report
                    if isinstance(
                        runtime.optimizer,
                        IbmOmFp32BoundsOptimizer,
                    )
                    else None
                ),
                "validation": dict(initial_validation),
            }
        )
    return Deployment(
        device_programming=device_programming,
        pre_deployment_validation=pre_deployment_validation,
        initial_validation=initial_validation,
    )


def _program_adapter_base(
    spec: TrainSpec,
    runtime: TrainRuntime,
) -> Mapping[str, Any] | None:
    """Program the frozen base under a low-rank adapter, if it owns that."""

    catalog = runtime.stack.bundle.catalog
    adapter = spec.common.model.adapter
    if adapter.type == "digital_low_rank":
        adapter_config = parse_digital_low_rank_adapter(
            dict(adapter.parameters),
            path="config.model.adapter.parameters",
        )
        if adapter_config is None:  # pragma: no cover - schema guarantees it
            raise AssertionError("digital_low_rank config resolved to null")
        if isinstance(adapter_config.device_noise, Wan2022ProgrammingConfig):
            return program_wan2022_base_conductance(
                catalog,
                adapter_config.device_noise,
            )
        return program_device_base_conductances(
            catalog,
            {_dense_base_key(catalog): adapter_config.device_noise},
        )
    if adapter.type != "passive_layerwise_low_rank":
        return None
    if isinstance(runtime.optimizer, MeasuredCohortBLoRAOptimizer):
        # The measured LoRA optimizer owns the one-time cohort-B base
        # deployment as well as RESET-endpoint adapter initialization.
        return None
    layer_adapter = parse_passive_layerwise_low_rank_adapter(
        dict(adapter.parameters),
        path="config.model.adapter.parameters",
    )
    if layer_adapter is None:  # pragma: no cover - schema guarantees it
        raise AssertionError("passive_layerwise_low_rank config resolved to null")
    layer_configs = {
        layer.parameter_key: layer.device_noise for layer in layer_adapter.layers
    }
    if all(
        isinstance(value, Wan2022ProgrammingConfig)
        for value in layer_configs.values()
    ):
        return program_wan2022_base_conductances(
            catalog,
            layer_configs,  # type: ignore[arg-type]
        )
    return program_device_base_conductances(
        catalog,
        layer_configs,  # type: ignore[arg-type]
    )


def _dense_base_key(catalog: ParameterCatalog) -> str:
    keys = tuple(
        binding.key
        for binding in catalog.for_group("base", checkpointed_only=True)
        if binding.role == "dense_weight"
    )
    if len(keys) != 1:
        raise ValueError(
            "Expected digital_low_rank to contain exactly one "
            f"base dense weight. Provided value: {keys!r}."
        )
    return keys[0]


def _initialize_measured(
    request: Any,
    optimizer: MeasuredTraceOptimizer,
) -> dict[str, Any]:
    if isinstance(optimizer, MeasuredCohortBLoRAOptimizer):
        initialization = optimizer.initialize_from_base_and_reset_adapters()
        semantics = (
            "loaded cohort-A base frozen after one cohort-B projection; "
            "four physical low-rank factor arrays start at the fully "
            "reset trace endpoint and alone receive measured writes"
        )
    elif isinstance(optimizer, MeasuredCohortAOptimizer):
        initialization = optimizer.initialize_at_pulse_zero()
        semantics = (
            "digital shadow with global-nearest projection onto a "
            "deterministic interpolated cohort-A measured curve"
        )
    else:
        initialization = optimizer.initialize_from_loaded_targets()
        semantics = (
            "loaded cohort-A targets retained as digital shadows and "
            "globally projected onto deterministic interpolated "
            "cohort-B measured curves before fine-tuning"
        )
    source = request.weights if request.weights is not None else request.base_weights
    return {
        "semantics": semantics,
        "pulse_model": False,
        "cohort_a_used": optimizer.cohort == "A",
        "cohort_b_used": optimizer.cohort == "B",
        "source_checkpoint": (
            None
            if source is None
            else {
                "path": str(Path(source).expanduser().resolve()),
                "sha256": sha256_file(source),
            }
        ),
        "initialization": initialization,
    }


def _validation_preserving_layer_state(
    spec: TrainSpec,
    runtime: TrainRuntime,
    *,
    split: str,
) -> dict[str, Any]:
    layer_snapshot = runtime.runtime_state.state_dict()
    validation = held_out_metrics(
        runtime,
        maximum_batches=spec.settings.max_validation_batches,
        modifier=None,
        epoch=-1,
        split=split,
    )
    runtime.runtime_state.load_state_dict(layer_snapshot)
    return validation
