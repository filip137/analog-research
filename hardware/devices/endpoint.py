"""Sampling from the compact IBM ReRAM program-and-verify endpoint model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


_SCHEMA = "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model"
_SCHEMA_VERSION = 2
_REACHABILITY_CLASSES = (
    "target_below_lower_bound",
    "target_inside_bounds",
    "target_above_upper_bound",
)


@dataclass(frozen=True)
class IbmReramEndpointSample:
    """One normalized one-shot programming realization."""

    requested_target: torch.Tensor
    modeled_target: torch.Tensor
    raw_endpoint: torch.Tensor
    endpoint: torch.Tensor
    persistent_endpoint: torch.Tensor
    accepted: torch.Tensor
    success_noncorrupt: torch.Tensor
    failed_noncorrupt: torch.Tensor
    corrupt: torch.Tensor
    target_below_lower_bound: torch.Tensor
    target_inside_bounds: torch.Tensor
    target_above_upper_bound: torch.Tensor
    acceptance_window_reachable: torch.Tensor
    target_was_clamped: torch.Tensor
    endpoint_was_clipped: torch.Tensor


def _uniform_like(
    reference: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw on the generator's backend and stage onto the model device."""

    draw = torch.rand(
        reference.shape,
        device=generator.device,
        dtype=reference.dtype,
        generator=generator,
    )
    return draw.to(reference.device)


def _records(
    artifact: Mapping[str, Any],
    condition_key: str,
    *,
    allow_inadequate: bool,
) -> tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    if (
        artifact.get("schema") != _SCHEMA
        or artifact.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError(
            "Expected an IBM ReRAM bounded piecewise-uniform endpoint artifact "
            f"with schema {_SCHEMA!r} version {_SCHEMA_VERSION}. Provided value: "
            f"schema={artifact.get('schema')!r}, "
            f"version={artifact.get('schema_version')!r}."
        )
    conditions = artifact.get("conditions")
    if not isinstance(conditions, Mapping) or condition_key not in conditions:
        raise ValueError(
            "Expected condition_key to identify one artifact condition. "
            f"Provided value: {condition_key!r}."
        )
    condition = conditions[condition_key]
    if not isinstance(condition, Mapping):
        raise ValueError(
            "Expected the selected endpoint condition to be an object. "
            f"Provided value: {condition!r}."
        )
    if (
        condition.get("fit_status") != "fit"
        or condition.get("reachability_fit_status") != "fit"
    ):
        raise ValueError(
            "Expected the selected endpoint condition to contain complete "
            "accepted-residual and reachability fits. Provided value: "
            f"fit_status={condition.get('fit_status')!r}, "
            "reachability_fit_status="
            f"{condition.get('reachability_fit_status')!r}."
        )
    if condition.get("adequate") is not True and not allow_inadequate:
        raise ValueError(
            "Expected the selected endpoint condition to pass its held-out "
            "adequacy gates. Set allow_inadequate=True only for an explicit "
            "analysis of a complete but inadequate fit."
        )
    validation = condition.get("validation")
    records = validation.get("per_target") if isinstance(validation, Mapping) else None
    if not isinstance(records, Sequence) or not records:
        raise ValueError(
            "Expected the selected endpoint condition to contain target-bin "
            f"models. Provided value: {records!r}."
        )
    empty_fit_targets = []
    for record in records:
        model = (
            record.get("accepted_noncorrupt_residual")
            if isinstance(record, Mapping)
            else None
        )
        fit_count = model.get("fit_count") if isinstance(model, Mapping) else None
        if (
            isinstance(fit_count, bool)
            or not isinstance(fit_count, int)
            or fit_count < 1
        ):
            target = record.get("target") if isinstance(record, Mapping) else None
            empty_fit_targets.append(target)
    if empty_fit_targets:
        raise ValueError(
            "Expected every endpoint target bin to contain accepted "
            "non-corrupt fit observations. Empty or invalid targets: "
            f"{empty_fit_targets!r}."
        )
    return condition, records


def _target_brackets(
    targets: torch.Tensor,
    grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if grid.numel() == 1:
        zeros = torch.zeros_like(targets, dtype=torch.long)
        return zeros, zeros, torch.zeros_like(targets)
    upper = torch.searchsorted(grid, targets, right=False)
    upper = upper.clamp(1, grid.numel() - 1)
    lower = upper - 1
    low_values = grid[lower]
    high_values = grid[upper]
    fraction = (targets - low_values) / (high_values - low_values)
    return lower, upper, fraction.clamp(0.0, 1.0)


def _interpolated_scalar(
    records: Sequence[Mapping[str, Any]],
    lower_index: torch.Tensor,
    upper_index: torch.Tensor,
    fraction: torch.Tensor,
    path: tuple[str, ...],
) -> torch.Tensor:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        values.append(float("nan") if value is None else float(value))
    table = torch.tensor(values, device=fraction.device, dtype=fraction.dtype)
    lower = table[lower_index]
    upper = table[upper_index]
    both = torch.isfinite(lower) & torch.isfinite(upper)
    result = torch.where(
        both,
        lower + fraction * (upper - lower),
        torch.where(torch.isfinite(lower), lower, upper),
    )
    return result


def _sample_success_residual(
    records: Sequence[Mapping[str, Any]],
    lower_index: torch.Tensor,
    upper_index: torch.Tensor,
    fraction: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    edge_rows = []
    probability_rows = []
    for record in records:
        model = record.get("accepted_noncorrupt_residual")
        if not isinstance(model, Mapping):
            raise ValueError("Expected every target bin to contain a success model.")
        edge_rows.append(model.get("bin_edges"))
        probability_rows.append(model.get("bin_probabilities"))
    edges = torch.as_tensor(
        edge_rows, device=fraction.device, dtype=fraction.dtype
    )
    probabilities = torch.as_tensor(
        probability_rows, device=fraction.device, dtype=fraction.dtype
    )
    if (
        edges.ndim != 2
        or probabilities.ndim != 2
        or edges.shape[0] != len(records)
        or edges.shape[1] != probabilities.shape[1] + 1
        or torch.any(probabilities <= 0.0)
    ):
        raise ValueError("Expected compatible positive histogram rows.")
    lower_probabilities = probabilities[lower_index]
    upper_probabilities = probabilities[upper_index]
    mixed = lower_probabilities + fraction.unsqueeze(1) * (
        upper_probabilities - lower_probabilities
    )
    mixed = mixed / mixed.sum(dim=1, keepdim=True)
    cumulative = torch.cumsum(mixed, dim=1)
    category_draw = _uniform_like(fraction, generator=generator)
    bin_index = torch.sum(
        category_draw.unsqueeze(1) > cumulative, dim=1
    ).clamp(max=mixed.shape[1] - 1)
    lower_edges = edges[lower_index]
    upper_edges = edges[upper_index]
    mixed_edges = lower_edges + fraction.unsqueeze(1) * (
        upper_edges - lower_edges
    )
    selected_lower = mixed_edges.gather(1, bin_index.unsqueeze(1)).squeeze(1)
    selected_upper = mixed_edges.gather(
        1, (bin_index + 1).unsqueeze(1)
    ).squeeze(1)
    within_draw = _uniform_like(fraction, generator=generator)
    return selected_lower + within_draw * (selected_upper - selected_lower)


def _sample_quantile_model(
    records: Sequence[Mapping[str, Any]],
    lower_index: torch.Tensor,
    upper_index: torch.Tensor,
    fraction: torch.Tensor,
    selection: torch.Tensor,
    *,
    path: tuple[str, ...],
    generator: torch.Generator,
) -> torch.Tensor:
    result = torch.full_like(fraction, float("nan"))
    if not bool(torch.any(selection)):
        return result
    probability_reference: Sequence[float] | None = None
    value_rows = []
    for record in records:
        endpoint: Any = record
        for key in path:
            if not isinstance(endpoint, Mapping):
                endpoint = None
                break
            endpoint = endpoint.get(key)
        probabilities = (
            endpoint.get("probabilities") if isinstance(endpoint, Mapping) else None
        )
        values = endpoint.get("values") if isinstance(endpoint, Mapping) else None
        if probabilities is not None:
            if probability_reference is None:
                probability_reference = probabilities
            elif list(probabilities) != list(probability_reference):
                raise ValueError(
                    "Expected identical inverse-CDF probabilities in every "
                    f"target bin for path {path!r}."
                )
        value_rows.append(values)
    if probability_reference is None:
        raise ValueError(
            f"Expected path {path!r} to contain an endpoint inverse CDF."
        )
    width = len(probability_reference)
    table = torch.full(
        (len(records), width),
        float("nan"),
        device=fraction.device,
        dtype=fraction.dtype,
    )
    for index, values in enumerate(value_rows):
        if values is not None:
            table[index] = torch.as_tensor(
                values, device=fraction.device, dtype=fraction.dtype
            )
    lower_values = table[lower_index]
    upper_values = table[upper_index]
    lower_valid = torch.all(torch.isfinite(lower_values), dim=1)
    upper_valid = torch.all(torch.isfinite(upper_values), dim=1)
    both = lower_valid & upper_valid
    mixed = torch.where(
        both.unsqueeze(1),
        lower_values
        + fraction.unsqueeze(1) * (upper_values - lower_values),
        torch.where(lower_valid.unsqueeze(1), lower_values, upper_values),
    )
    if bool(torch.any(selection & ~torch.all(torch.isfinite(mixed), dim=1))):
        raise ValueError(
            f"Expected a fitted terminal endpoint for selected path {path!r}."
        )
    probabilities = torch.as_tensor(
        probability_reference,
        device=fraction.device,
        dtype=fraction.dtype,
    )
    draw = _uniform_like(fraction, generator=generator)
    interval = torch.sum(draw.unsqueeze(1) > probabilities[1:], dim=1)
    interval = interval.clamp(max=probabilities.numel() - 2)
    probability_lower = probabilities[interval]
    probability_upper = probabilities[interval + 1]
    value_lower = mixed.gather(1, interval.unsqueeze(1)).squeeze(1)
    value_upper = mixed.gather(1, (interval + 1).unsqueeze(1)).squeeze(1)
    within = (draw - probability_lower) / (
        probability_upper - probability_lower
    )
    sampled = value_lower + within * (value_upper - value_lower)
    result[selection] = sampled[selection]
    return result


def _sample_reachability_classes(
    records: Sequence[Mapping[str, Any]],
    lower_index: torch.Tensor,
    upper_index: torch.Tensor,
    fraction: torch.Tensor,
    noncorrupt: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.stack(
        [
            _interpolated_scalar(
                records,
                lower_index,
                upper_index,
                fraction,
                (
                    "outcome_model",
                    "noncorrupt_reachability",
                    "classes",
                    reachability_class,
                    "probability",
                ),
            )
            for reachability_class in _REACHABILITY_CLASSES
        ],
        dim=1,
    )
    if bool(
        torch.any(
            noncorrupt
            & (
                ~torch.all(torch.isfinite(probabilities), dim=1)
                | torch.any(probabilities < 0.0, dim=1)
                | (torch.sum(probabilities, dim=1) <= 0.0)
            )
        )
    ):
        raise ValueError(
            "Expected normalized fitted persistent-reachability class probabilities."
        )
    probabilities = torch.where(
        torch.isfinite(probabilities), probabilities.clamp_min(0.0), 0.0
    )
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    draw = _uniform_like(fraction, generator=generator)
    class_index = torch.sum(
        draw.unsqueeze(1) > torch.cumsum(probabilities, dim=1), dim=1
    ).clamp(max=len(_REACHABILITY_CLASSES) - 1)
    masks = tuple(
        noncorrupt & (class_index == index)
        for index in range(len(_REACHABILITY_CLASSES))
    )
    if not torch.equal(masks[0] | masks[1] | masks[2], noncorrupt):
        raise RuntimeError("Expected reachability classes to partition non-corrupt sites.")
    return masks[0], masks[1], masks[2], class_index


def _class_conditional_scalar(
    records: Sequence[Mapping[str, Any]],
    lower_index: torch.Tensor,
    upper_index: torch.Tensor,
    fraction: torch.Tensor,
    class_index: torch.Tensor,
    noncorrupt: torch.Tensor,
    *,
    field: str,
) -> torch.Tensor:
    result = torch.full_like(fraction, float("nan"))
    for index, reachability_class in enumerate(_REACHABILITY_CLASSES):
        values = _interpolated_scalar(
            records,
            lower_index,
            upper_index,
            fraction,
            (
                "outcome_model",
                "noncorrupt_reachability",
                "classes",
                reachability_class,
                field,
            ),
        )
        result = torch.where(noncorrupt & (class_index == index), values, result)
    if bool(torch.any(noncorrupt & ~torch.isfinite(result))):
        raise ValueError(
            "Expected a fitted class-conditional non-corrupt "
            f"{field.replace('_', ' ')}."
        )
    return result


def sample_ibm_reram_endpoints(
    targets: torch.Tensor,
    artifact: Mapping[str, Any],
    *,
    condition_key: str,
    generator: torch.Generator,
    corrupt_mask: torch.Tensor | None = None,
    lower_bound: torch.Tensor | None = None,
    upper_bound: torch.Tensor | None = None,
    corrupt_apparent_endpoint: torch.Tensor | None = None,
    corrupt_persistent_endpoint: torch.Tensor | None = None,
    target_out_of_support: str = "error",
    endpoint_policy: str = "preserve",
    allow_inadequate: bool = False,
) -> IbmReramEndpointSample:
    """Sample normalized one-shot endpoints without widening write noise.

    ``lower_bound`` and ``upper_bound`` let callers preserve the sampled
    reachable interval of every physical identity. When they are omitted the
    artifact's target-conditioned marginal reachability model is used.

    Passing all three corrupt-identity arguments preserves both the defect
    mask and each corrupt site's fixed stuck state. Repeated physical writes
    must still continue from an ``IbmReramPlant`` state instead.
    """

    if not isinstance(targets, torch.Tensor) or not targets.is_floating_point():
        raise ValueError(
            "Expected targets to be a floating-point torch tensor. Provided "
            f"value: {targets!r}."
        )
    if not bool(torch.all(torch.isfinite(targets))):
        raise ValueError("Expected all IBM ReRAM endpoint targets to be finite.")
    if target_out_of_support not in {"error", "clamp"}:
        raise ValueError(
            "Expected target_out_of_support to equal 'error' or 'clamp'. "
            f"Provided value: {target_out_of_support!r}."
        )
    if endpoint_policy not in {"preserve", "clip_0_1"}:
        raise ValueError(
            "Expected endpoint_policy to equal 'preserve' or 'clip_0_1'. "
            f"Provided value: {endpoint_policy!r}."
        )
    if not isinstance(allow_inadequate, bool):
        raise ValueError(
            "Expected allow_inadequate to be a boolean. "
            f"Provided value: {allow_inadequate!r}."
        )
    _, records = _records(
        artifact,
        condition_key,
        allow_inadequate=allow_inadequate,
    )
    target_grid_values = [float(record["target"]) for record in records]
    if target_grid_values != sorted(target_grid_values) or len(
        set(target_grid_values)
    ) != len(target_grid_values):
        raise ValueError("Expected strictly increasing endpoint target bins.")

    requested_shape = targets.shape
    requested = targets.detach().clone()
    work = requested.reshape(-1).to(dtype=torch.float64)
    grid = torch.tensor(
        target_grid_values, device=work.device, dtype=work.dtype
    )
    outside = (work < grid[0]) | (work > grid[-1])
    if target_out_of_support == "error" and bool(torch.any(outside)):
        offending = work[outside]
        raise ValueError(
            "Expected IBM ReRAM normalized targets inside the characterized "
            f"interval [{float(grid[0])}, {float(grid[-1])}]. Provided "
            f"extrema: minimum={float(torch.min(offending))}, "
            f"maximum={float(torch.max(offending))}."
        )
    modeled = work.clamp(float(grid[0]), float(grid[-1]))
    lower_index, upper_index, fraction = _target_brackets(modeled, grid)

    corrupt_probability = _interpolated_scalar(
        records,
        lower_index,
        upper_index,
        fraction,
        ("outcome_model", "corrupt_identity_fraction"),
    )
    if corrupt_mask is None:
        if bool(torch.any(~torch.isfinite(corrupt_probability))):
            raise ValueError("Expected fitted corrupt-identity probabilities.")
        corrupt = _uniform_like(work, generator=generator) < corrupt_probability
    else:
        if not isinstance(corrupt_mask, torch.Tensor):
            raise ValueError("Expected corrupt_mask to be a torch tensor.")
        if corrupt_mask.shape != requested_shape:
            raise ValueError(
                "Expected corrupt_mask to match target shape. Provided value: "
                f"target_shape={requested_shape!r}, "
                f"mask_shape={corrupt_mask.shape!r}."
            )
        corrupt = corrupt_mask.to(device=work.device, dtype=torch.bool).reshape(-1)

    noncorrupt = ~corrupt
    fixed_bound_arguments = (lower_bound, upper_bound)
    if any(value is not None for value in fixed_bound_arguments):
        if not all(isinstance(value, torch.Tensor) for value in fixed_bound_arguments):
            raise ValueError(
                "Expected lower_bound and upper_bound to be supplied together "
                "as torch tensors."
            )
        assert isinstance(lower_bound, torch.Tensor)
        assert isinstance(upper_bound, torch.Tensor)
        if lower_bound.shape != requested_shape or upper_bound.shape != requested_shape:
            raise ValueError(
                "Expected fixed IBM ReRAM bounds to match target shape. "
                f"Provided value: target={requested_shape!r}, "
                f"lower={lower_bound.shape!r}, upper={upper_bound.shape!r}."
            )
        fixed_lower = lower_bound.to(device=work.device, dtype=work.dtype).reshape(-1)
        fixed_upper = upper_bound.to(device=work.device, dtype=work.dtype).reshape(-1)
        if (
            not bool(torch.all(torch.isfinite(fixed_lower)))
            or not bool(torch.all(torch.isfinite(fixed_upper)))
            or bool(torch.any(noncorrupt & (fixed_lower > fixed_upper)))
        ):
            raise ValueError(
                "Expected finite fixed IBM ReRAM bounds with lower <= upper "
                "for every non-corrupt identity."
            )
        target_below_lower_bound = noncorrupt & (modeled < fixed_lower)
        target_above_upper_bound = noncorrupt & (modeled > fixed_upper)
        target_inside_bounds = noncorrupt & ~(
            target_below_lower_bound | target_above_upper_bound
        )
        reachability_class_index = torch.zeros_like(lower_index)
        reachability_class_index[target_inside_bounds] = 1
        reachability_class_index[target_above_upper_bound] = 2
    else:
        fixed_lower = None
        fixed_upper = None
        (
            target_below_lower_bound,
            target_inside_bounds,
            target_above_upper_bound,
            reachability_class_index,
        ) = _sample_reachability_classes(
            records,
            lower_index,
            upper_index,
            fraction,
            noncorrupt,
            generator=generator,
        )
    success_probability = _class_conditional_scalar(
        records,
        lower_index,
        upper_index,
        fraction,
        reachability_class_index,
        noncorrupt,
        field="success_probability",
    )
    success_noncorrupt = noncorrupt & (
        _uniform_like(work, generator=generator)
        < success_probability
    )
    failed_noncorrupt = noncorrupt & ~success_noncorrupt
    acceptance_window_probability = _class_conditional_scalar(
        records,
        lower_index,
        upper_index,
        fraction,
        reachability_class_index,
        noncorrupt,
        field="acceptance_window_reachable_probability",
    )
    if fixed_lower is None or fixed_upper is None:
        acceptance_window_reachable = noncorrupt & (
            _uniform_like(work, generator=generator)
            < acceptance_window_probability
        )
    else:
        tolerance = _interpolated_scalar(
            records,
            lower_index,
            upper_index,
            fraction,
            ("tolerance",),
        )
        acceptance_window_reachable = noncorrupt & (
            (modeled + tolerance >= fixed_lower)
            & (modeled - tolerance <= fixed_upper)
        )

    residual = _sample_success_residual(
        records,
        lower_index,
        upper_index,
        fraction,
        generator=generator,
    )
    raw_endpoint = modeled + residual
    reachability_masks = (
        target_below_lower_bound,
        target_inside_bounds,
        target_above_upper_bound,
    )

    def class_endpoint(
        selection: torch.Tensor,
        *,
        outcome: str,
        endpoint: str,
    ) -> torch.Tensor:
        sampled = torch.full_like(work, float("nan"))
        for reachability_class, reachability_mask in zip(
            _REACHABILITY_CLASSES, reachability_masks
        ):
            class_selection = selection & reachability_mask
            values = _sample_quantile_model(
                records,
                lower_index,
                upper_index,
                fraction,
                class_selection,
                path=(
                    "outcome_model",
                    "noncorrupt_reachability",
                    "classes",
                    reachability_class,
                    outcome,
                    endpoint,
                ),
                generator=generator,
            )
            sampled = torch.where(class_selection, values, sampled)
        return sampled

    failed_endpoint = class_endpoint(
        failed_noncorrupt,
        outcome="failed_terminal",
        endpoint="apparent_endpoint",
    )
    success_persistent_endpoint = class_endpoint(
        success_noncorrupt,
        outcome="accepted_terminal",
        endpoint="persistent_endpoint",
    )
    failed_persistent_endpoint = class_endpoint(
        failed_noncorrupt,
        outcome="failed_terminal",
        endpoint="persistent_endpoint",
    )
    fixed_corrupt_arguments = (
        corrupt_apparent_endpoint,
        corrupt_persistent_endpoint,
    )
    if any(value is not None for value in fixed_corrupt_arguments):
        if not all(isinstance(value, torch.Tensor) for value in fixed_corrupt_arguments):
            raise ValueError(
                "Expected corrupt_apparent_endpoint and "
                "corrupt_persistent_endpoint to be supplied together as tensors."
            )
        assert isinstance(corrupt_apparent_endpoint, torch.Tensor)
        assert isinstance(corrupt_persistent_endpoint, torch.Tensor)
        if (
            corrupt_apparent_endpoint.shape != requested_shape
            or corrupt_persistent_endpoint.shape != requested_shape
        ):
            raise ValueError(
                "Expected fixed corrupt endpoints to match target shape. "
                f"Provided value: target={requested_shape!r}, "
                f"apparent={corrupt_apparent_endpoint.shape!r}, "
                f"persistent={corrupt_persistent_endpoint.shape!r}."
            )
        corrupt_endpoint = corrupt_apparent_endpoint.to(
            device=work.device, dtype=work.dtype
        ).reshape(-1)
        sampled_corrupt_persistent = corrupt_persistent_endpoint.to(
            device=work.device, dtype=work.dtype
        ).reshape(-1)
        if bool(
            torch.any(
                corrupt
                & (
                    ~torch.isfinite(corrupt_endpoint)
                    | ~torch.isfinite(sampled_corrupt_persistent)
                )
            )
        ):
            raise ValueError("Expected fixed corrupt endpoints to be finite.")
    else:
        corrupt_endpoint = _sample_quantile_model(
            records,
            lower_index,
            upper_index,
            fraction,
            corrupt,
            path=("outcome_model", "corrupt_terminal", "apparent_endpoint"),
            generator=generator,
        )
        sampled_corrupt_persistent = _sample_quantile_model(
            records,
            lower_index,
            upper_index,
            fraction,
            corrupt,
            path=("outcome_model", "corrupt_terminal", "persistent_endpoint"),
            generator=generator,
        )
    raw_endpoint = torch.where(failed_noncorrupt, failed_endpoint, raw_endpoint)
    raw_endpoint = torch.where(corrupt, corrupt_endpoint, raw_endpoint)
    persistent_endpoint = torch.where(
        success_noncorrupt,
        success_persistent_endpoint,
        failed_persistent_endpoint,
    )
    persistent_endpoint = torch.where(
        corrupt,
        sampled_corrupt_persistent,
        persistent_endpoint,
    )
    if fixed_lower is not None and fixed_upper is not None:
        bounded_persistent = torch.maximum(persistent_endpoint, fixed_lower)
        bounded_persistent = torch.minimum(bounded_persistent, fixed_upper)
        persistent_endpoint = torch.where(
            noncorrupt,
            bounded_persistent,
            persistent_endpoint,
        )

    tolerance = _interpolated_scalar(
        records,
        lower_index,
        upper_index,
        fraction,
        ("tolerance",),
    )
    corrupt_accepted = corrupt & (torch.abs(raw_endpoint - modeled) <= tolerance)
    accepted = success_noncorrupt | corrupt_accepted
    if endpoint_policy == "clip_0_1":
        endpoint = raw_endpoint.clamp(0.0, 1.0)
        endpoint_was_clipped = endpoint != raw_endpoint
    else:
        endpoint = raw_endpoint
        endpoint_was_clipped = torch.zeros_like(corrupt)

    def restored(value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        result = value.reshape(requested_shape)
        return result.to(dtype=dtype or targets.dtype)

    return IbmReramEndpointSample(
        requested_target=requested,
        modeled_target=restored(modeled),
        raw_endpoint=restored(raw_endpoint),
        endpoint=restored(endpoint),
        persistent_endpoint=restored(persistent_endpoint),
        accepted=restored(accepted, dtype=torch.bool),
        success_noncorrupt=restored(success_noncorrupt, dtype=torch.bool),
        failed_noncorrupt=restored(failed_noncorrupt, dtype=torch.bool),
        corrupt=restored(corrupt, dtype=torch.bool),
        target_below_lower_bound=restored(
            target_below_lower_bound, dtype=torch.bool
        ),
        target_inside_bounds=restored(target_inside_bounds, dtype=torch.bool),
        target_above_upper_bound=restored(
            target_above_upper_bound, dtype=torch.bool
        ),
        acceptance_window_reachable=restored(
            acceptance_window_reachable, dtype=torch.bool
        ),
        target_was_clamped=restored(outside, dtype=torch.bool),
        endpoint_was_clipped=restored(endpoint_was_clipped, dtype=torch.bool),
    )


__all__ = ["IbmReramEndpointSample", "sample_ibm_reram_endpoints"]
