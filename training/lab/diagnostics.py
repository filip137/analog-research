"""Optional observers and phase diagnostics for the training core.

Observers consume engine events. They never advance an optimizer or perform a
training solve. The production FiniteGradientGuard lives in training.core.guards. BetaSize is a separate diagnostic pass that restores voltages
and the nudging state after inspecting free and perturbed equilibria.
"""

from itertools import islice
import math

import torch

from training.core.batch import as_batch
from training.core.engine import (
    AfterUpdateEvent,
    BeforeUpdateEvent,
    EvaluationBatchEvent,
    FreePhaseEvent,
    GradientsReadyEvent,
)


class LayerMeasurements:
    """Record settled states and raw energy-gradient infinity norms.

    These are the historical lab residuals over all network layers, including
    the input. Solver-projected physical residuals remain separate probes.
    """

    def __init__(self, *, store_states=False, residual_currents=False):
        self.store_states = store_states
        self.residual_currents = residual_currents
        self.reset()

    def reset(self):
        self.layer_states = {}
        self.res_currents = {}

    def __call__(self, event):
        if not isinstance(event, (FreePhaseEvent, EvaluationBatchEvent)):
            return
        if self.store_states:
            for layer in event.components.network.layers():
                self.layer_states.setdefault(layer.name, []).append(
                    layer.state.detach().cpu().clone()
                )
        if self.residual_currents:
            function = event.components.network._function
            for layer in function.layers():
                gradient = function.grad_layer_fn(layer)()
                value = float(gradient.detach().abs().max().item())
                self.res_currents.setdefault(layer.name, []).append(value)


def _tensor_statistics(value):
    value = value.detach()
    return {
        "norm": value.norm().item(),
        "mean": value.mean().item(),
        "std": value.std(unbiased=False).item(),
        "min": value.min().item(),
        "max": value.max().item(),
        "shape": tuple(value.shape),
    }


class GradientUpdateObserver:
    """Measure gradients and actual post-clamp updates only when attached.

    The pre-update snapshot is taken outside the modifier context, so temporary
    forward noise is never counted as an optimizer update. Only the latest
    batch report is retained.
    """

    def __init__(self, *, verbose=False):
        self.verbose = verbose
        self.reset()

    def reset(self):
        self._before = None
        self._gradients = {}
        self.last_result = None

    def __call__(self, event):
        if isinstance(event, GradientsReadyEvent):
            self._gradients = {
                parameter.name: _tensor_statistics(gradient)
                for parameter, gradient in zip(event.components.parameters, event.gradients)
            }
        elif isinstance(event, BeforeUpdateEvent):
            self._before = tuple(
                parameter.state.detach().clone()
                for parameter in event.components.parameters
            )
        elif isinstance(event, AfterUpdateEvent):
            if self._before is None:
                raise RuntimeError("Expected a pre-update snapshot before measuring updates.")
            try:
                updates = {
                    parameter.name: _tensor_statistics(parameter.state - before)
                    for parameter, before in zip(event.components.parameters, self._before)
                }
                self.last_result = {
                    "gradient_stats": self._gradients,
                    "update_stats": updates,
                    "num_parameters": len(event.components.parameters),
                }
                if self.verbose:
                    for name, update in updates.items():
                        print(
                            f"  {name}: grad_norm={self._gradients[name]['norm']:.6f}, "
                            f"applied_update_norm={update['norm']:.6f}"
                        )
            finally:
                self._before = None


class BetaSize:
    """Measure mean-absolute free-to-nudged displacement without updating weights."""

    def __init__(self, components, dataloader, differentiator, *, max_batches=None, eps=1e-12):
        if max_batches is not None and (
            isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1
        ):
            raise ValueError("Expected max_batches to be a positive integer or None.")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Expected a finite positive displacement denominator epsilon.")
        self.components = components
        self.dataloader = dataloader
        self.differentiator = differentiator
        self.max_batches = max_batches
        self.eps = eps
        self._layer_names = [layer.name for layer in components.network.layers()]
        self._ratios_per_batch = []
        self.free_abs_mean_per_batch = []
        self.displacement_abs_mean_per_batch = []

    @torch.no_grad()
    def run(self, verbose=False):
        self._ratios_per_batch.clear()
        self.free_abs_mean_per_batch.clear()
        self.displacement_abs_mean_per_batch.clear()
        components = self.components
        estimator = self.differentiator
        layers = tuple(components.network.layers())
        snapshots = [(layer, layer.state, layer.state.clone()) for layer in layers]
        augmented = estimator._augmented_fn
        original_nudging = augmented.nudging
        nudging_term = getattr(augmented, "_nudging", None)
        original_force = getattr(nudging_term, "_force", None)
        # Centered and negative EP retain their original first (negative) phase;
        # positive EP must inspect its second phase because its first is free.
        nudging = estimator._first_nudging or estimator._second_nudging
        try:
            for index, raw_batch in enumerate(islice(self.dataloader, self.max_batches)):
                batch = as_batch(raw_batch)
                components.network.set_input(batch.inputs, reset=True)
                components.energy_minimizer.compute_equilibrium()
                components.cost_fn.set_target(batch.targets)
                free = {layer.name: layer.state.clone() for layer in layers}
                if hasattr(augmented, "prepare_nudging"):
                    augmented.prepare_nudging()
                augmented.nudging = nudging
                nudged = estimator._energy_minimizer.compute_equilibrium()
                free_mean = torch.stack([free[name].abs().mean() for name in self._layer_names])
                displacement = torch.stack([
                    (free[name] - nudged[name]).abs().mean() for name in self._layer_names
                ])
                ratios = displacement / (free_mean + self.eps)
                self.free_abs_mean_per_batch.append(free_mean.detach())
                self.displacement_abs_mean_per_batch.append(displacement.detach())
                self._ratios_per_batch.append(ratios.detach())
                if verbose:
                    print(f"\rBetaSize batch {index + 1}: median ratio = {ratios.median().item():.4f}", end="", flush=True)
        finally:
            for layer, state, value in snapshots:
                state.copy_(value)
                layer.state = state
            augmented.nudging = original_nudging
            if hasattr(nudging_term, "_force"):
                nudging_term._force = original_force
        if verbose:
            print()

    def ratios_tensor(self):
        if not self._ratios_per_batch:
            return torch.empty((0, len(self._layer_names)))
        return torch.stack(self._ratios_per_batch)

    def summary(self):
        ratios = self.ratios_tensor()
        if not ratios.numel():
            return {"layers": self._layer_names, "per_layer_median": [], "per_layer_mean": []}
        return {
            "layers": self._layer_names,
            "per_layer_median": ratios.median(dim=0).values.cpu().tolist(),
            "per_layer_mean": ratios.mean(dim=0).cpu().tolist(),
            "overall_median": ratios.median().item(),
            "overall_mean": ratios.mean().item(),
        }
