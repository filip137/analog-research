"""Measured virtual-cell assignments, held pulse states and interpolation.

The controller supplies targets or pulse indices; the bank never sees a DRN
parameter, optimizer or objective. Serialized pulse indices retain their existing
checkpoint keys in the method's composite codec.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from hardware.devices.measured_trace import _ProjectionTable, _assignment_seed, _assignment_digest
_CACHE_CHUNK_CELLS = 1024

class MeasuredTraceBank:
    def __init__(self, curve_preprocessing):
        self.curve_preprocessing = curve_preprocessing
        self.tables = {}
        self.source_curves = {}
        self.pulse_indices = {}

    def curve_ranges(
        self,
        key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the calibrated minimum and maximum of every virtual cell."""

        table = self.tables[key]
        if table.sorted_values is not None:
            return table.sorted_values[:, 0], table.sorted_values[:, -1]
        source = self.source_curves[key]
        if self.curve_preprocessing == "isotonic_nonincreasing":
            maximum = (
                table.alpha * source[table.source_left, 0]
                + (1.0 - table.alpha) * source[table.source_right, 0]
            )
            minimum = (
                table.alpha * source[table.source_left, -1]
                + (1.0 - table.alpha) * source[table.source_right, -1]
            )
            return minimum, maximum

        count = table.alpha.numel()
        minimum = torch.empty(
            count,
            dtype=table.alpha.dtype,
            device=table.alpha.device,
        )
        maximum = torch.empty_like(minimum)
        for start in range(0, count, _CACHE_CHUNK_CELLS):
            stop = min(count, start + _CACHE_CHUNK_CELLS)
            weights = table.alpha[start:stop, None]
            mixed = (
                weights * source[table.source_left[start:stop]]
                + (1.0 - weights)
                * source[table.source_right[start:stop]]
            )
            minimum[start:stop] = mixed.amin(dim=1)
            maximum[start:stop] = mixed.amax(dim=1)
        return minimum, maximum


    def assign(self, layout, conductance_s, *, assignment_seed, cohort, expected_trace_length, cache_mode):
        for key, shape, device, dtype in layout:
            source = torch.as_tensor(
                conductance_s,
                device=device,
                dtype=dtype,
            )
            self.source_curves[key] = source
            count = math.prod(shape)
            generator = np.random.default_rng(
                _assignment_seed(
                    assignment_seed,
                    key,
                    cohort,
                )
            )
            left_np = generator.integers(
                0,
                source.shape[0],
                size=count,
                dtype=np.int32,
            )
            right_np = generator.integers(
                0,
                source.shape[0] - 1,
                size=count,
                dtype=np.int32,
            )
            right_np += right_np >= left_np
            alpha_np = generator.random(count, dtype=np.float32)
            left = torch.from_numpy(left_np).to(device=device, dtype=torch.long)
            right = torch.from_numpy(right_np).to(device=device, dtype=torch.long)
            alpha = torch.from_numpy(alpha_np).to(device=device, dtype=dtype)
            sorted_values = None
            sorted_pulses = None
            if (
                self.curve_preprocessing == "raw"
                and cache_mode == "cached"
            ):
                length = expected_trace_length
                sorted_values = torch.empty(
                    (count, length), device=device, dtype=dtype
                )
                sorted_pulses = torch.empty(
                    (count, length), device=device, dtype=torch.int16
                )
                for start in range(0, count, _CACHE_CHUNK_CELLS):
                    stop = min(count, start + _CACHE_CHUNK_CELLS)
                    weights = alpha[start:stop, None]
                    mixed = (
                        weights * source[left[start:stop]]
                        + (1.0 - weights) * source[right[start:stop]]
                    )
                    values, order = torch.sort(mixed, dim=1, stable=True)
                    sorted_values[start:stop].copy_(values)
                    sorted_pulses[start:stop].copy_(order.to(torch.int16))
            self.tables[key] = _ProjectionTable(
                key=key,
                shape=tuple(shape),
                source_left=left,
                source_right=right,
                alpha=alpha,
                assignment_sha256=_assignment_digest(
                    left_np,
                    right_np,
                    alpha_np,
                ),
                sorted_values=sorted_values,
                sorted_pulses=sorted_pulses,
            )


    def project(
        self,
        key: str,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table = self.tables[key]
        if table.sorted_values is not None:
            values = table.sorted_values
            pulse_table = table.sorted_pulses
            if pulse_table is None:  # pragma: no cover - constructed together
                raise AssertionError("raw pulse table is missing")
            position = torch.searchsorted(
                values,
                targets.contiguous().unsqueeze(1),
                right=False,
            ).squeeze(1)
            low = (position - 1).clamp(0, values.shape[1] - 1)
            high = position.clamp(0, values.shape[1] - 1)
            low_value = values.gather(1, low[:, None]).squeeze(1)
            high_value = values.gather(1, high[:, None]).squeeze(1)
            low_pulse = pulse_table.gather(1, low[:, None]).squeeze(1)
            high_pulse = pulse_table.gather(1, high[:, None]).squeeze(1)
            low_distance = (targets - low_value).abs()
            high_distance = (targets - high_value).abs()
            choose_high = (high_distance < low_distance) | (
                (high_distance == low_distance) & (high_pulse < low_pulse)
            )
            return (
                torch.where(choose_high, high_value, low_value),
                torch.where(choose_high, high_pulse, low_pulse),
            )

        if self.curve_preprocessing == "raw":
            source = self.source_curves[key]
            realized = torch.empty_like(targets)
            pulses = torch.empty(
                targets.shape,
                dtype=torch.int16,
                device=targets.device,
            )
            for start in range(0, targets.numel(), _CACHE_CHUNK_CELLS):
                stop = min(targets.numel(), start + _CACHE_CHUNK_CELLS)
                weights = table.alpha[start:stop, None]
                mixed = (
                    weights * source[table.source_left[start:stop]]
                    + (1.0 - weights)
                    * source[table.source_right[start:stop]]
                )
                # torch.argmin returns the first occurrence, implementing the
                # protocol's lower-pulse tie break exactly.
                selected = (
                    mixed - targets[start:stop, None]
                ).abs().argmin(dim=1)
                realized[start:stop] = mixed.gather(
                    1,
                    selected[:, None],
                ).squeeze(1)
                pulses[start:stop] = selected.to(torch.int16)
            return realized, pulses

        source = self.source_curves[key]
        low = torch.zeros_like(table.source_left)
        high = torch.full_like(low, source.shape[1])
        for _ in range((source.shape[1] + 1).bit_length()):
            middle = torch.div(low + high, 2, rounding_mode="floor")
            active = low < high
            safe_middle = middle.clamp(max=source.shape[1] - 1)
            middle_value = (
                table.alpha * source[table.source_left, safe_middle]
                + (1.0 - table.alpha)
                * source[table.source_right, safe_middle]
            )
            move_right = active & (middle_value > targets)
            high = torch.where(active & ~move_right, middle, high)
            low = torch.where(move_right, middle + 1, low)
        later = low.clamp(max=source.shape[1] - 1)
        earlier = (low - 1).clamp(min=0, max=source.shape[1] - 1)

        def mixed_at(pulse: torch.Tensor) -> torch.Tensor:
            return (
                table.alpha * source[table.source_left, pulse]
                + (1.0 - table.alpha) * source[table.source_right, pulse]
            )

        earlier_value = mixed_at(earlier)
        later_value = mixed_at(later)
        # Equality goes to the smaller (earlier) physical pulse index.
        choose_earlier = (targets - earlier_value).abs() <= (
            targets - later_value
        ).abs()
        selected_value = torch.where(choose_earlier, earlier_value, later_value)
        # PAVA creates exact plateaus. Resolve those ties to the lowest pulse
        # index by finding the first point whose descending value is <= the
        # already selected value.
        plateau_low = torch.zeros_like(table.source_left)
        plateau_high = torch.full_like(plateau_low, source.shape[1])
        for _ in range((source.shape[1] + 1).bit_length()):
            middle = torch.div(
                plateau_low + plateau_high,
                2,
                rounding_mode="floor",
            )
            active = plateau_low < plateau_high
            middle_value = mixed_at(middle.clamp(max=source.shape[1] - 1))
            move_right = active & (middle_value > selected_value)
            plateau_high = torch.where(
                active & ~move_right,
                middle,
                plateau_high,
            )
            plateau_low = torch.where(
                move_right,
                middle + 1,
                plateau_low,
            )
        selected_pulse = plateau_low.clamp(max=source.shape[1] - 1)
        return selected_value, selected_pulse.to(torch.int16)


    def values_at_pulses(
        self,
        key: str,
        pulses: torch.Tensor,
    ) -> torch.Tensor:
        table = self.tables[key]
        source = self.source_curves[key]
        indices = pulses.to(torch.long)
        return (
            table.alpha * source[table.source_left, indices]
            + (1.0 - table.alpha) * source[table.source_right, indices]
        )
