import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from model.minimizer.minimizer import LayerUpdater, Minimizer
from model.resistive.minimizer import lambertw
from model.resistive.minimizer import (
    QuadraticUpdater,
    AdaptiveQuadraticUpdater,
    QuadraticDoubleDiodeUpdaterOffset,
    ExponentialDoubleDiodeUpdater,
    ExponentialSingleDiodeUpdater,
    HardSigmoidUpdater,
    _scale_diode_strength_for_layer,
    _scale_hard_sigmoid_params_for_layer,
)
from model.resistive.layer import NonlinearResistiveLayer
from model.variable.layer import LinearLayer

DEFAULT_IV_CURVE_PATH = None


class NonFiniteDiodeError(FloatingPointError):
    """Raised when non-finite values are detected in diode updater debug checks."""


def _emit_timing_line(line: str) -> None:
    log_path = os.environ.get("LABS_TIMING_LOG_PATH")
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
    if os.environ.get("LABS_TIMING_LOG_ONLY") == "1":
        return
    print(line)


@dataclass
class MinimizerSettings:
    rel_tol: float
    vn_tol: float
    use_polish: bool
    max_newton_iters: int
    z_thresh: float
    exp_clip: float
    dynamic_polish: bool
    overrelaxation_reject_steps: bool
    overrelaxation_reject_max_tries: int
    overrelaxation_reject_shrink: float
    overrelaxation_reject_eps: float
    experimental_exponential_newton_tol_progressive: bool = True
    experimental_exponential_newton_tol_start: float = 1e-5
    experimental_exponential_newton_tol_end: float = 1e-5
    experimental_exponential_newton_tol_switch_hi: float = 1e-2
    experimental_exponential_newton_tol_switch_lo: float = 5e-4

    def __post_init__(self):
        self.rel_tol = float(self.rel_tol)
        self.vn_tol = float(self.vn_tol)
        self.use_polish = bool(self.use_polish)
        self.max_newton_iters = max(int(self.max_newton_iters), 0)
        self.z_thresh = float(self.z_thresh)
        self.exp_clip = float(self.exp_clip)
        self.dynamic_polish = bool(self.dynamic_polish)
        self.overrelaxation_reject_steps = bool(self.overrelaxation_reject_steps)
        self.overrelaxation_reject_max_tries = max(int(self.overrelaxation_reject_max_tries), 1)
        self.overrelaxation_reject_shrink = float(self.overrelaxation_reject_shrink)
        self.overrelaxation_reject_eps = float(self.overrelaxation_reject_eps)
        self.experimental_exponential_newton_tol_progressive = bool(
            self.experimental_exponential_newton_tol_progressive
        )
        self.experimental_exponential_newton_tol_start = float(self.experimental_exponential_newton_tol_start)
        self.experimental_exponential_newton_tol_end = float(self.experimental_exponential_newton_tol_end)
        self.experimental_exponential_newton_tol_switch_hi = float(self.experimental_exponential_newton_tol_switch_hi)
        self.experimental_exponential_newton_tol_switch_lo = float(self.experimental_exponential_newton_tol_switch_lo)
        if not (0.0 < self.overrelaxation_reject_shrink < 1.0):
            raise ValueError(
                "Expected overrelaxation_reject_shrink in (0, 1). "
                f"Got {self.overrelaxation_reject_shrink!r}."
            )
        if not (self.experimental_exponential_newton_tol_start > 0.0):
            raise ValueError(
                "Expected experimental_exponential_newton_tol_start > 0. "
                f"Got {self.experimental_exponential_newton_tol_start!r}."
            )
        if not (self.experimental_exponential_newton_tol_end > 0.0):
            raise ValueError(
                "Expected experimental_exponential_newton_tol_end > 0. "
                f"Got {self.experimental_exponential_newton_tol_end!r}."
            )
        if not (
            self.experimental_exponential_newton_tol_switch_hi
            > self.experimental_exponential_newton_tol_switch_lo
            > 0.0
        ):
            raise ValueError(
                "Expected experimental_exponential_newton_tol_switch_hi > "
                "experimental_exponential_newton_tol_switch_lo > 0. "
                "Got "
                f"hi={self.experimental_exponential_newton_tol_switch_hi!r}, "
                f"lo={self.experimental_exponential_newton_tol_switch_lo!r}."
            )


def _load_iv_data(path: Path) -> torch.Tensor:
    data = np.load(path)
    if "iv" in data:
        iv = data["iv"]
    elif "i" in data and "v" in data:
        iv = np.stack([data["i"], data["v"]], axis=0)
    else:
        raise ValueError(f"{path} must contain 'iv' or both 'i' and 'v' arrays.")
    return torch.as_tensor(iv)



class CustomMinimizer(Minimizer):
    def __init__(
        self,
        fn,
        updaters,
        num_iterations,
        mode,
        voltage_amp,
        current_amp,
        adaptive_equilibrium,
        *,
        settings: MinimizerSettings,
    ):
        super().__init__(fn, updaters, num_iterations, mode, voltage_amp, current_amp)
        # Keep a reference to the function for energy evaluation (unwrap AugmentedFunction when present).
        self._fn = fn
        self._energy_fn = getattr(fn, "_energy_fn", fn)
        # Disabled for performance and to keep equilibration logic simple.
        self._store_states = False
        self._stored_states = {}
        self._adaptive_equilibrium = bool(adaptive_equilibrium)
        self._equilibrium_iter_total = 0
        self._equilibrium_iter_calls = 0
        self._equilibrium_iter_last = 0
        self._equilibrium_iter_max = 0
        self._equilibrium_sample_iters_last = None
        self._force_fixed_iterations = any(
            type(updater) is ExponentialSingleDiodeUpdater for updater in updaters
        )
        self._settings = settings
        self._rel_tol = settings.rel_tol
        self._vn_tol = settings.vn_tol
        self._use_polish = settings.use_polish
        self._max_newton_iters = settings.max_newton_iters
        self._z_thresh = settings.z_thresh
        self._exp_clip = settings.exp_clip
        self._dynamic_polish = settings.dynamic_polish
        self._overrelaxation_reject_steps = settings.overrelaxation_reject_steps
        self._overrelaxation_reject_max_tries = settings.overrelaxation_reject_max_tries
        self._overrelaxation_reject_shrink = settings.overrelaxation_reject_shrink
        self._overrelaxation_reject_eps = settings.overrelaxation_reject_eps
        self._experimental_exponential_newton_tol_progressive = (
            settings.experimental_exponential_newton_tol_progressive
        )
        self._experimental_exponential_newton_tol_start = settings.experimental_exponential_newton_tol_start
        self._experimental_exponential_newton_tol_end = settings.experimental_exponential_newton_tol_end
        self._experimental_exponential_newton_tol_switch_hi = (
            settings.experimental_exponential_newton_tol_switch_hi
        )
        self._experimental_exponential_newton_tol_switch_lo = (
            settings.experimental_exponential_newton_tol_switch_lo
        )
        self._residual_currents_last = None
        self._or_step_calls = 0
        self._or_accepted_first_try = 0
        self._or_accepted_after_backtrack = 0
        self._or_fallback_plain_steps = 0
        self._or_backtracks_total = 0
        self._or_last_backtracks = 0
        self._or_last_accepted_omega = 1.0
        for updater in self._updaters:
            setattr(updater, "_use_polish", self._use_polish)
            setattr(updater, "_max_newton_iters", self._max_newton_iters)
            setattr(updater, "_z_thresh", self._z_thresh)
            setattr(updater, "_exp_clip", self._exp_clip)
            setattr(updater, "_rel_tol", self._rel_tol)
            set_dynamic_polish = getattr(updater, "set_dynamic_polish", None)
            if callable(set_dynamic_polish):
                set_dynamic_polish(
                    use_polish=self._use_polish,
                    max_newton_iters=self._max_newton_iters,
                    z_thresh=self._z_thresh,
                )
            set_newton_tol = getattr(updater, "set_newton_tol", None)
            if callable(set_newton_tol):
                set_newton_tol(self._experimental_exponential_newton_tol_start)
        if os.environ.get("DRN_DEBUG_DIODE") == "1":
            import pdb
            pdb.set_trace()

    @property
    def store_states(self):
        return self._store_states

    @store_states.setter
    def store_states(self, value):
        self._store_states = bool(value)

    @property
    def stored_states(self):
        return self._stored_states

    def clear_stored_states(self):
        self._stored_states = {}

    def eval_energy_sum(self):
        """Return the summed energy over the batch for current layer states."""
        return self.eval_energy_batch().sum()

    def eval_energy_batch(self):
        """Return per-sample energy for current layer states."""
        with torch.no_grad():
            energy = self._energy_fn.eval()
            if energy.dim() <= 1:
                return energy.reshape(-1)
            return energy.reshape(energy.shape[0], -1).sum(dim=1)

    def _layer_energy_scale(self, layer):
        """Return the positive scalar relating energy gradients to currents."""
        scale_fn = getattr(self._energy_fn, "layer_energy_scale", None)
        if not callable(scale_fn):
            return 1.0

        scale = float(scale_fn(layer))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "Expected layer energy scale to be a positive finite scalar. "
                f"Provided value for {layer.name!r}: {scale!r}."
            )
        return scale

    def _projected_current_residual(self, updater, current, scale):
        """Return h * (v - projection(v - i / h)) when supported."""
        layer = updater._layer
        if isinstance(layer, LinearLayer):
            return current
        if not isinstance(layer, NonlinearResistiveLayer):
            return current
        non_linearity = getattr(layer, "non_linearity", None)
        if non_linearity == "linear":
            return current
        if non_linearity != "perfect_diode":
            return current

        a_fn = getattr(updater, "_a", None)
        if not callable(a_fn):
            return current

        local_hessian = torch.as_tensor(
            2.0 * a_fn(),
            dtype=current.dtype,
            device=current.device,
        ).detach() / scale
        try:
            expanded_hessian = local_hessian + torch.zeros_like(current)
        except RuntimeError:
            return current
        if not torch.isfinite(expanded_hessian).all() or not (
            expanded_hessian > 0.0
        ).all():
            return current

        state = layer.state.detach()
        boundary_current = expanded_hessian * state
        dimension = layer._shape[0] // 2
        excitatory = torch.where(
            current[:, :dimension] <= boundary_current[:, :dimension],
            current[:, :dimension],
            boundary_current[:, :dimension],
        )
        inhibitory = torch.where(
            current[:, dimension:] >= boundary_current[:, dimension:],
            current[:, dimension:],
            boundary_current[:, dimension:],
        )
        return torch.cat((excitatory, inhibitory), dim=1)

    def residual_currents_inf(self):
        """Compute physical projected-current residuals for managed layers."""
        fn = self._fn
        residuals = {}
        for updater in self._updaters:
            layer = updater._layer
            scale = self._layer_energy_scale(layer)
            current = fn.grad_layer_fn(layer)().detach() / scale
            residual = self._projected_current_residual(
                updater,
                current,
                scale,
            )
            residuals[layer.name] = float(residual.abs().max().item())
        self._residual_currents_last = residuals
        return residuals

    @property
    def residual_currents_last(self):
        return self._residual_currents_last

    def reset_equilibrium_iteration_stats(self):
        self._equilibrium_iter_total = 0
        self._equilibrium_iter_calls = 0
        self._equilibrium_iter_last = 0
        self._equilibrium_iter_max = 0
        self._equilibrium_sample_iters_last = None

    def reset_overrelaxation_reject_stats(self):
        self._or_step_calls = 0
        self._or_accepted_first_try = 0
        self._or_accepted_after_backtrack = 0
        self._or_fallback_plain_steps = 0
        self._or_backtracks_total = 0
        self._or_last_backtracks = 0
        self._or_last_accepted_omega = 1.0

    def equilibrium_iteration_stats(self, reset=False):
        calls = int(self._equilibrium_iter_calls)
        stats = {
            "calls": calls,
            "avg_iterations": (
                float(self._equilibrium_iter_total) / calls if calls > 0 else 0.0
            ),
            "last_iterations": int(self._equilibrium_iter_last),
            "max_iterations": int(self._equilibrium_iter_max),
        }
        if reset:
            self.reset_equilibrium_iteration_stats()
        return stats

    def overrelaxation_reject_stats(self, reset=False):
        steps = int(self._or_step_calls)
        stats = {
            "enabled": bool(self._overrelaxation_reject_steps),
            "steps_checked": steps,
            "accepted_first_try": int(self._or_accepted_first_try),
            "accepted_after_backtrack": int(self._or_accepted_after_backtrack),
            "fallback_plain_steps": int(self._or_fallback_plain_steps),
            "backtracks_total": int(self._or_backtracks_total),
            "avg_backtracks_per_step": (float(self._or_backtracks_total) / steps) if steps > 0 else 0.0,
            "last_backtracks": int(self._or_last_backtracks),
            "last_accepted_omega": float(self._or_last_accepted_omega),
            "reject_max_tries": int(self._overrelaxation_reject_max_tries),
            "reject_shrink": float(self._overrelaxation_reject_shrink),
            "reject_eps": float(self._overrelaxation_reject_eps),
        }
        if reset:
            self.reset_overrelaxation_reject_stats()
        return stats

    def equilibrium_sample_iterations(self):
        return None

    def _set_experimental_newton_tol(self, tol: float):
        for updater in self._updaters:
            set_newton_tol = getattr(updater, "set_newton_tol", None)
            if callable(set_newton_tol):
                set_newton_tol(tol)

    def _experimental_exponential_newton_tol_policy(self, inf_delta: float) -> float:
        if not self._experimental_exponential_newton_tol_progressive:
            return self._experimental_exponential_newton_tol_end
        tol_start = self._experimental_exponential_newton_tol_start
        tol_end = self._experimental_exponential_newton_tol_end
        d_hi = self._experimental_exponential_newton_tol_switch_hi
        d_lo = self._experimental_exponential_newton_tol_switch_lo
        if tol_start == tol_end:
            return tol_end
        if inf_delta >= d_hi:
            return tol_start
        if inf_delta <= d_lo:
            return tol_end
        # Interpolate in log-space for smoother transition across orders of magnitude.
        t = (math.log10(inf_delta) - math.log10(d_lo)) / (math.log10(d_hi) - math.log10(d_lo))
        return 10.0 ** (math.log10(tol_end) + t * (math.log10(tol_start) - math.log10(tol_end)))

    def _snapshot_layer_state(self, layer):
        state = layer.state.detach()
        return state.cpu().clone()

    @staticmethod
    def _max_abs_delta(new_state, old_state):
        return float((new_state - old_state).abs().max().item())

    @staticmethod
    def _snapshot_group_state(layer_group):
        return [updater._layer.state.detach().clone() for updater in layer_group]

    @staticmethod
    def _restore_group_state(layer_group, snapshots):
        for updater, snapshot in zip(layer_group, snapshots):
            updater._layer.state = snapshot.clone()

    @staticmethod
    def _group_overrelaxation_factor(layer_group):
        for updater in layer_group:
            omega = getattr(updater, "_overrelaxation_factor", None)
            if omega is not None:
                try:
                    return float(omega)
                except (TypeError, ValueError):
                    continue
        return 1.0

    @staticmethod
    def _set_group_overrelaxation_factor(layer_group, value: float):
        applied = False
        for updater in layer_group:
            set_omega = getattr(updater, "set_overrelaxation_factor", None)
            if callable(set_omega):
                set_omega(float(value))
                applied = True
        return applied

    def _step_group_with_reject_policy(self, layer_group, *, label: str, iteration: int):
        if not layer_group:
            return

        if not self._overrelaxation_reject_steps:
            self.step(layer_group)
            self._maybe_break_on_nonfinite_state(layer_group, label=label, iteration=iteration)
            return

        base_omega = self._group_overrelaxation_factor(layer_group)
        if base_omega <= 1.0:
            self.step(layer_group)
            self._maybe_break_on_nonfinite_state(layer_group, label=label, iteration=iteration)
            return

        if not self._set_group_overrelaxation_factor(layer_group, base_omega):
            self.step(layer_group)
            self._maybe_break_on_nonfinite_state(layer_group, label=label, iteration=iteration)
            return

        energy_before = self.eval_energy_sum()
        if not torch.isfinite(energy_before):
            self.step(layer_group)
            self._maybe_break_on_nonfinite_state(layer_group, label=label, iteration=iteration)
            return

        energy_before_value = float(energy_before.item())
        snapshots = self._snapshot_group_state(layer_group)
        accepted = False
        accepted_after_backtrack = False
        backtracks = 0
        accepted_omega = base_omega
        self._or_step_calls += 1

        try:
            for attempt in range(self._overrelaxation_reject_max_tries):
                omega = 1.0 + (base_omega - 1.0) * (self._overrelaxation_reject_shrink ** attempt)
                if attempt > 0:
                    self._restore_group_state(layer_group, snapshots)
                self._set_group_overrelaxation_factor(layer_group, omega)
                self.step(layer_group)

                finite_states = all(torch.isfinite(updater._layer.state).all().item() for updater in layer_group)
                if not finite_states:
                    backtracks = attempt + 1
                    continue

                energy_after = self.eval_energy_sum()
                if torch.isfinite(energy_after):
                    if float(energy_after.item()) <= energy_before_value + self._overrelaxation_reject_eps:
                        accepted = True
                        accepted_omega = float(omega)
                        backtracks = attempt
                        accepted_after_backtrack = attempt > 0
                        break
                backtracks = attempt + 1

            if not accepted:
                self._restore_group_state(layer_group, snapshots)
                self._set_group_overrelaxation_factor(layer_group, 1.0)
                self.step(layer_group)
                self._or_fallback_plain_steps += 1
                accepted_omega = 1.0
            elif accepted_after_backtrack:
                self._or_accepted_after_backtrack += 1
            else:
                self._or_accepted_first_try += 1
        finally:
            self._set_group_overrelaxation_factor(layer_group, base_omega)

        self._or_backtracks_total += int(backtracks)
        self._or_last_backtracks = int(backtracks)
        self._or_last_accepted_omega = float(accepted_omega)
        self._maybe_break_on_nonfinite_state(layer_group, label=label, iteration=iteration)

    def _step_odd_even_with_reject_policy(self, layer_group_odd, layer_group_even, *, iteration: int):
        if not layer_group_odd and not layer_group_even:
            return
        if not layer_group_odd:
            self._step_group_with_reject_policy(layer_group_even, label="even", iteration=iteration)
            return
        if not layer_group_even:
            self._step_group_with_reject_policy(layer_group_odd, label="odd", iteration=iteration)
            return

        combined_group = [*layer_group_odd, *layer_group_even]

        if not self._overrelaxation_reject_steps:
            self.step(layer_group_odd)
            self.step(layer_group_even)
            self._maybe_break_on_nonfinite_state(combined_group, label="odd_even", iteration=iteration)
            return

        odd_base = self._group_overrelaxation_factor(layer_group_odd)
        even_base = self._group_overrelaxation_factor(layer_group_even)
        odd_applied = False
        even_applied = False
        if odd_base > 1.0:
            odd_applied = self._set_group_overrelaxation_factor(layer_group_odd, odd_base)
        if even_base > 1.0:
            even_applied = self._set_group_overrelaxation_factor(layer_group_even, even_base)

        # No overrelaxed updater in either group (or cannot set omega) -> plain sweep.
        if not odd_applied and not even_applied:
            self.step(layer_group_odd)
            self.step(layer_group_even)
            self._maybe_break_on_nonfinite_state(combined_group, label="odd_even", iteration=iteration)
            return

        energy_before = self.eval_energy_sum()
        if not torch.isfinite(energy_before):
            self.step(layer_group_odd)
            self.step(layer_group_even)
            self._maybe_break_on_nonfinite_state(combined_group, label="odd_even", iteration=iteration)
            return

        energy_before_value = float(energy_before.item())
        snapshots = self._snapshot_group_state(combined_group)
        accepted = False
        accepted_after_backtrack = False
        backtracks = 0
        accepted_omega = odd_base if odd_applied else even_base
        self._or_step_calls += 1

        try:
            for attempt in range(self._overrelaxation_reject_max_tries):
                if attempt > 0:
                    self._restore_group_state(combined_group, snapshots)
                scale = self._overrelaxation_reject_shrink ** attempt
                if odd_applied:
                    odd_trial = 1.0 + (odd_base - 1.0) * scale
                    self._set_group_overrelaxation_factor(layer_group_odd, odd_trial)
                else:
                    odd_trial = 1.0
                if even_applied:
                    even_trial = 1.0 + (even_base - 1.0) * scale
                    self._set_group_overrelaxation_factor(layer_group_even, even_trial)
                else:
                    even_trial = 1.0

                self.step(layer_group_odd)
                self.step(layer_group_even)

                finite_states = all(torch.isfinite(updater._layer.state).all().item() for updater in combined_group)
                if not finite_states:
                    backtracks = attempt + 1
                    continue

                energy_after = self.eval_energy_sum()
                if torch.isfinite(energy_after):
                    if float(energy_after.item()) <= energy_before_value + self._overrelaxation_reject_eps:
                        accepted = True
                        accepted_after_backtrack = attempt > 0
                        backtracks = attempt
                        accepted_omega = odd_trial if odd_applied else even_trial
                        break
                backtracks = attempt + 1

            if not accepted:
                self._restore_group_state(combined_group, snapshots)
                if odd_applied:
                    self._set_group_overrelaxation_factor(layer_group_odd, 1.0)
                if even_applied:
                    self._set_group_overrelaxation_factor(layer_group_even, 1.0)
                self.step(layer_group_odd)
                self.step(layer_group_even)
                self._or_fallback_plain_steps += 1
                accepted_omega = 1.0
            elif accepted_after_backtrack:
                self._or_accepted_after_backtrack += 1
            else:
                self._or_accepted_first_try += 1
        finally:
            if odd_applied:
                self._set_group_overrelaxation_factor(layer_group_odd, odd_base)
            if even_applied:
                self._set_group_overrelaxation_factor(layer_group_even, even_base)

        self._or_backtracks_total += int(backtracks)
        self._or_last_backtracks = int(backtracks)
        self._or_last_accepted_omega = float(accepted_omega)
        self._maybe_break_on_nonfinite_state(combined_group, label="odd_even", iteration=iteration)

    @staticmethod
    def _dynamic_polish_policy(inf_delta: float):
        # Policy requested by user:
        #   inf_delta > 1e-2  -> no polish
        #   1e-3 < inf_delta <= 1e-2 -> polish with 4 Newton iterations
        #   inf_delta <= 1e-3 -> polish with 8 Newton iterations
        if inf_delta > 1e-2:
            return False, 0, 1e4
        if inf_delta > 1e-3:
            return True, 4, 1e6
        return True, 8, 1e7

    @staticmethod
    def _maybe_break_on_nonfinite_state(layer_group, *, label: str, iteration: int):
        if os.environ.get("DRN_DEBUG_BREAK_ON_ERROR") != "1" and not os.environ.get("DRN_DEBUG_DIODE"):
            return
        max_abs_threshold = float(os.environ.get("DRN_DEBUG_STATE_MAX_ABS", "1e10"))
        for updater in layer_group:
            state = updater._layer.state
            name = getattr(updater._layer, "name", updater._layer.__class__.__name__)
            if not torch.isfinite(state).all():
                nan_count = torch.isnan(state).sum().item()
                inf_count = torch.isinf(state).sum().item()
                message = (
                    f"[state-check] non-finite state after {label} iter={iteration} layer={name}: "
                    f"nan={nan_count} inf={inf_count} shape={tuple(state.shape)} "
                    f"dtype={state.dtype} device={state.device}"
                )
                print(message)
                if os.environ.get("DRN_DEBUG_BREAK_ON_ERROR") == "1":
                    breakpoint()
                raise NonFiniteDiodeError(message)

            max_abs = float(state.abs().max().item())
            if max_abs > max_abs_threshold:
                message = (
                    f"[state-check] large-magnitude state after {label} iter={iteration} layer={name}: "
                    f"max_abs={max_abs:.6g} threshold={max_abs_threshold:.6g} "
                    f"shape={tuple(state.shape)} dtype={state.dtype} device={state.device}"
                )
                print(message)
                if os.environ.get("DRN_DEBUG_BREAK_ON_ERROR") == "1":
                    breakpoint()
                raise NonFiniteDiodeError(message)



    def compute_equilibrium(self):
        """Compute the minimum of the function wrt the free layers

        Performs num_iterations iterations of the minimization process.

        Returns:
            layers: dictionary of Tensors. The state of the layers at equilibrium
        """

        max_num_of_iterations = len(self._list_layers) // 2
        self._stored_states = {}
        iterations_used = max_num_of_iterations

        use_adaptive = self._adaptive_equilibrium and not self._force_fixed_iterations
        if not use_adaptive:
            # Original fixed-iteration behaviour: no tolerance-based early stop.
            if len(self._list_layers) >= 2:
                layer_group_odd, layer_group_even = self._list_layers[0], self._list_layers[1]
                self._set_experimental_newton_tol(
                    self._experimental_exponential_newton_tol_policy(float("inf"))
                )
                for i in range(max_num_of_iterations):
                    self._step_odd_even_with_reject_policy(
                        layer_group_odd,
                        layer_group_even,
                        iteration=i,
                    )
            else:
                for idx, layer_group in enumerate(self._list_layers):
                    self._step_group_with_reject_policy(layer_group, label=f"group_{idx}", iteration=0)
                max_num_of_iterations = len(self._list_layers)
                iterations_used = max_num_of_iterations
        else:
            prev_inf_delta = float("inf")
            rtol = self._rel_tol
            vntol = self._vn_tol

            for i in range(max_num_of_iterations):
                layer_group_odd, layer_group_even = self._list_layers[0], self._list_layers[1]
                self._set_experimental_newton_tol(
                    self._experimental_exponential_newton_tol_policy(prev_inf_delta)
                )

                use_polish = self._use_polish
                max_newton_iters = self._max_newton_iters
                z_thresh = self._z_thresh
                if self._dynamic_polish:
                    use_polish, max_newton_iters, z_thresh = self._dynamic_polish_policy(prev_inf_delta)
                    for updater in (*layer_group_odd, *layer_group_even):
                        set_dynamic_polish = getattr(updater, "set_dynamic_polish", None)
                        if callable(set_dynamic_polish):
                            set_dynamic_polish(
                                use_polish=use_polish,
                                max_newton_iters=max_newton_iters,
                                z_thresh=z_thresh,
                            )

                odd_old = [u._layer.state.detach().clone() for u in layer_group_odd]
                even_old = [u._layer.state.detach().clone() for u in layer_group_even]

                self._step_odd_even_with_reject_policy(
                    layer_group_odd,
                    layer_group_even,
                    iteration=i,
                )

                inf_delta = 0.0
                inf_ref = 0.0
                for updater, old_state in zip(layer_group_odd, odd_old):
                    inf_delta = max(inf_delta, self._max_abs_delta(updater._layer.state, old_state))
                    inf_ref = max(inf_ref, float(old_state.abs().max().item()))
                for updater, old_state in zip(layer_group_even, even_old):
                    inf_delta = max(inf_delta, self._max_abs_delta(updater._layer.state, old_state))
                    inf_ref = max(inf_ref, float(old_state.abs().max().item()))

                # SPICE-like global convergence test (infinity norm)
                # inf_delta = max |v_new - v_old|
                # inf_ref   = max max(|v_new|, |v_old|)
                threshold = rtol * inf_ref + vntol
                prev_inf_delta = inf_delta
                if inf_delta <= threshold:
                    iterations_used = i + 1
                    break

        self._equilibrium_iter_total += iterations_used
        self._equilibrium_iter_calls += 1
        self._equilibrium_iter_last = iterations_used
        self._equilibrium_iter_max = max_num_of_iterations
        self._equilibrium_sample_iters_last = None
        #print(
         #   f"{self.__class__.__name__}: avg equilibrium iterations={avg_iterations:.2f} "
          #  f"(last={iterations_used}, max={max_num_of_iterations})"
        #)

        layers = {layer.name: layer.state for layer in self._layers}

        return layers

class CustomQuadraticUpdater(QuadraticUpdater):
    """Override pre_activate if you want custom quadratic updates."""


class CustomAdaptiveQuadraticUpdater(AdaptiveQuadraticUpdater):
    """Override pre_activate if you want custom LPW updates."""


class CustomQuadraticDoubleDiodeUpdaterOffset(QuadraticDoubleDiodeUpdaterOffset):
    """Override pre_activate if you want custom quadratic double-diode updates."""


class CustomExponentialDoubleDiodeUpdater(ExponentialDoubleDiodeUpdater):
    """Exponential double-diode updater with optional debug checks."""

    def __init__(self, layer, fn, diode_params):
        super().__init__(layer, fn, diode_params)
        self._newton_iter_total = 0
        self._newton_iter_calls = 0

    def _debug_tensor(self, name, tensor):
        if not os.environ.get("DRN_DEBUG_DIODE"):
            return
        if not torch.is_tensor(tensor):
            print(f"[CustomExponentialDoubleDiodeUpdater] {name} is not a tensor: {type(tensor)}")
            return
        finite = torch.isfinite(tensor)
        if finite.all():
            return
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if finite.any():
            finite_vals = tensor[finite]
            finite_min = finite_vals.min().item()
            finite_max = finite_vals.max().item()
            finite_mean = finite_vals.mean().item()
        else:
            finite_min = None
            finite_max = None
            finite_mean = None
        message = (
            "[CustomExponentialDoubleDiodeUpdater] non-finite in "
            f"{name}: nan={nan_count} inf={inf_count} "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} finite_min={finite_min} "
            f"finite_max={finite_max} finite_mean={finite_mean}"
        )
        print(message)
        if os.environ.get("DRN_DEBUG_BREAK_ON_ERROR") == "1":
            breakpoint()
        raise NonFiniteDiodeError(message)

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)
        self._debug_tensor("a", a)
        self._debug_tensor("b", b)
        if isinstance(self._layer, NonlinearResistiveLayer):
            if (
                not hasattr(self, "_z_thresh")
                or not hasattr(self, "_use_polish")
                or not hasattr(self, "_exp_clip")
                or not hasattr(self, "_max_newton_iters")
            ):
                raise RuntimeError("CustomExponentialDoubleDiodeUpdater missing polish settings.")
            v = self.lambert_hidden(
                a,
                b,
                self._Is,
                self._v_off,
                self._Vt,
                z_thresh=self._z_thresh,
                polish=self._use_polish,
                exp_clip=self._exp_clip,
                max_inner_iter=self._max_newton_iters,
            )
            self._debug_tensor("v", v)
            return v
        return -b / (2.0 * a)

    def lambertw_large(self, z, terms=4):
        self._debug_tensor("lambertw_large:z", z)
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w += L2 / L1
        if terms >= 3:
            w += (L2 * (-2 + L2)) / (2 * L1**2)
        if terms >= 4:
            w += (L2 * (6 - 9 * L2 + 2 * L2**2)) / (6 * L1**3)
        self._debug_tensor("lambertw_large:w", w)
        return w

    def lambert_hidden(self, a, b, I_s, v_off, vt, z_thresh, polish, exp_clip, max_inner_iter):
        out_dtype = a.dtype
        device = a.device

        a64 = a.to(torch.float64)
        b64 = b.to(torch.float64)
        if getattr(self, "_b_clip", None) is not None:
            b64 = b64.clamp(min=-self._b_clip, max=self._b_clip)
        I_s64 = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vt64 = torch.as_tensor(vt, dtype=torch.float64, device=device)
        v_off64 = torch.as_tensor(v_off, dtype=torch.float64, device=device)
        self._debug_tensor("a64", a64)
        self._debug_tensor("b64", b64)
        self._debug_tensor("I_s64", I_s64)
        self._debug_tensor("vt64", vt64)
        self._debug_tensor("v_off64", v_off64)

        A = torch.where(b64 <= 0, (b64 - I_s64) / (2.0 * a64),
                        (b64 + I_s64) / (2.0 * a64))
        self._debug_tensor("A", A)

        exp_arg_add = (-(A + v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        exp_arg_rev = ((A - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        self._debug_tensor("exp_arg_add", exp_arg_add)
        self._debug_tensor("exp_arg_rev", exp_arg_rev)

        z_add = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_add)
        z_rev = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_rev)
        z = torch.where(b64 > 0, z_rev, z_add)
        self._debug_tensor("z_add", z_add)
        self._debug_tensor("z_rev", z_rev)
        self._debug_tensor("z", z)

        use_asym = z > z_thresh
        small_mask = ~use_asym
        W0 = torch.empty_like(z, dtype=torch.float64, device=z.device)

        if small_mask.any():
            z_small = torch.clamp(z[small_mask].to(torch.float64), max=float(z_thresh))
            self._debug_tensor("z_small", z_small)
            W0_small = lambertw(z_small).real.to(W0.dtype)
            self._debug_tensor("W0_small", W0_small)
            W0[small_mask] = W0_small

        if use_asym.any():
            z_large = torch.clamp(z[use_asym].to(torch.float64), min=1.0)
            self._debug_tensor("z_large", z_large)
            W0_large = self.lambertw_large(z_large).to(torch.float64)
            self._debug_tensor("W0_large", W0_large)
            W0[use_asym] = W0_large

        self._debug_tensor("W0", W0)
        x = torch.where(b64 > 0, vt64 * W0 - A, -vt64 * W0 - A)
        self._debug_tensor("x_pre_polish", x)

        if polish:
            iters_used = 0
            current_tol = 1e-6 * x.shape[0]
            f = torch.zeros_like(x)
            df = torch.zeros_like(x)
            for _ in range(1, max_inner_iter):
                iters_used += 1
                mask_pos = b64 <= 0
                mask_neg = ~mask_pos

                if mask_pos.any():
                    xp = x[mask_pos]
                    ap64 = a64[mask_pos]
                    bp64 = b64[mask_pos]
                    ep_arg = ((xp - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
                    ep = torch.exp(ep_arg)
                    f[mask_pos] = 2 * ap64 * xp + (bp64 - I_s) + I_s * ep
                    df[mask_pos] = 2 * ap64 + (I_s / vt) * ep

                if mask_neg.any():
                    xn = x[mask_neg]
                    an64 = a64[mask_neg]
                    bn64 = b64[mask_neg]
                    en_arg = ((-xn - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
                    en = torch.exp(en_arg)
                    f[mask_neg] = 2 * an64 * xn + (bn64 + I_s) - I_s * en
                    df[mask_neg] = 2 * an64 + (I_s / vt) * en

                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f", f)
                    self._debug_tensor("df", df)
                    break

                x = x - f / df
                self._debug_tensor("x_polish", x)
                if torch.norm(f) < current_tol:
                    break
            self._newton_iter_total += iters_used
            self._newton_iter_calls += 1

        return x.to(out_dtype)


class ExperimentalDoubleDiodeUpdater(ExponentialDoubleDiodeUpdater):
    """Double-diode updater with optional b-clamp and abs/rel Newton tolerances."""

    def __init__(self, layer, fn, diode_params, *, print_every: int = 200, synchronize_cuda: bool = True):
        super().__init__(layer, fn, diode_params)
        self._print_every = max(int(print_every), 1)
        self._synchronize_cuda = bool(synchronize_cuda)
        b_clip_env = os.environ.get("DRN_B_CLAMP")
        self._b_clip = float(b_clip_env) if b_clip_env is not None else None
        self._newton_iter_total = 0
        self._newton_iter_calls = 0

    def set_dynamic_polish(self, *, use_polish: bool, max_newton_iters: int, z_thresh: float | None = None):
        self._use_polish = bool(use_polish)
        self._max_newton_iters = max(int(max_newton_iters), 0)
        if z_thresh is not None:
            self._z_thresh = float(z_thresh)

    def _debug_tensor(self, name, tensor):
        if not os.environ.get("DRN_DEBUG_DIODE"):
            return
        if not torch.is_tensor(tensor):
            print(f"[ExperimentalDoubleDiodeUpdater] {name} is not a tensor: {type(tensor)}")
            return
        finite = torch.isfinite(tensor)
        if finite.all():
            return
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if finite.any():
            finite_vals = tensor[finite]
            finite_min = finite_vals.min().item()
            finite_max = finite_vals.max().item()
            finite_mean = finite_vals.mean().item()
        else:
            finite_min = None
            finite_max = None
            finite_mean = None
        message = (
            "[ExperimentalDoubleDiodeUpdater] non-finite in "
            f"{name}: nan={nan_count} inf={inf_count} "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} finite_min={finite_min} "
            f"finite_max={finite_max} finite_mean={finite_mean}"
        )
        print(message)
        if os.environ.get("DRN_DEBUG_BREAK_ON_ERROR") == "1":
            breakpoint()
        raise NonFiniteDiodeError(message)

    def pre_activate(self):
        b = self._b()
        a = self._a().expand_as(b)
        self._debug_tensor("a", a)
        self._debug_tensor("b", b)
        if isinstance(self._layer, NonlinearResistiveLayer):
            if (
                not hasattr(self, "_z_thresh")
                or not hasattr(self, "_use_polish")
                or not hasattr(self, "_exp_clip")
                or not hasattr(self, "_max_newton_iters")
            ):
                raise RuntimeError("ExperimentalDoubleDiodeUpdater missing polish settings.")
            v = self.lambert_hidden(
                a,
                b,
                self._Is,
                self._v_off,
                self._Vt,
                z_thresh=self._z_thresh,
                polish=self._use_polish,
                exp_clip=self._exp_clip,
                max_inner_iter=self._max_newton_iters,
            )
            self._debug_tensor("v", v)
            return v
        return -b / (2.0 * a)

    def lambertw_large(self, z, terms=4):
        self._debug_tensor("lambertw_large:z", z)
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w += L2 / L1
        if terms >= 3:
            w += (L2 * (-2 + L2)) / (2 * L1**2)
        if terms >= 4:
            w += (L2 * (6 - 9 * L2 + 2 * L2**2)) / (6 * L1**3)
        self._debug_tensor("lambertw_large:w", w)
        return w

    def lambert_hidden(self, a, b, I_s, v_off, vt, z_thresh, polish, exp_clip, max_inner_iter):
        out_dtype = a.dtype
        device = a.device

        a64 = a.to(torch.float64)
        b64 = b.to(torch.float64)
        if self._b_clip is not None:
            b64 = b64.clamp(min=-self._b_clip, max=self._b_clip)
        I_s64 = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vt64 = torch.as_tensor(vt, dtype=torch.float64, device=device)
        v_off64 = torch.as_tensor(v_off, dtype=torch.float64, device=device)
        self._debug_tensor("a64", a64)
        self._debug_tensor("b64", b64)
        self._debug_tensor("I_s64", I_s64)
        self._debug_tensor("vt64", vt64)
        self._debug_tensor("v_off64", v_off64)

        A = torch.where(b64 <= 0, (b64 - I_s64) / (2.0 * a64),
                        (b64 + I_s64) / (2.0 * a64))
        self._debug_tensor("A", A)

        exp_arg_add = (-(A + v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        exp_arg_rev = ((A - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        self._debug_tensor("exp_arg_add", exp_arg_add)
        self._debug_tensor("exp_arg_rev", exp_arg_rev)

        z_add = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_add)
        z_rev = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_rev)
        z = torch.where(b64 > 0, z_rev, z_add)
        self._debug_tensor("z_add", z_add)
        self._debug_tensor("z_rev", z_rev)
        self._debug_tensor("z", z)

        use_asym = z > z_thresh
        small_mask = ~use_asym
        W0 = torch.empty_like(z, dtype=torch.float64, device=z.device)

        if small_mask.any():
            z_small = torch.clamp(z[small_mask].to(torch.float64), max=float(z_thresh))
            self._debug_tensor("z_small", z_small)
            W0_small = lambertw(z_small).real.to(W0.dtype)
            self._debug_tensor("W0_small", W0_small)
            W0[small_mask] = W0_small

        if use_asym.any():
            z_large = torch.clamp(z[use_asym].to(torch.float64), min=1.0)
            self._debug_tensor("z_large", z_large)
            W0_large = self.lambertw_large(z_large).to(torch.float64)
            self._debug_tensor("W0_large", W0_large)
            W0[use_asym] = W0_large

        self._debug_tensor("W0", W0)
        x = torch.where(b64 > 0, vt64 * W0 - A, -vt64 * W0 - A)
        self._debug_tensor("x_pre_polish", x)

        if polish:
            iters_used = 0
            abs_tol = getattr(self, "_abs_tol", None)
            rel_tol = getattr(self, "_rel_tol", 1e-6)
            abs_tol = rel_tol if abs_tol is None else abs_tol
            f = torch.zeros_like(x)
            df = torch.zeros_like(x)
            for _ in range(1, max_inner_iter):
                iters_used += 1
                mask_pos = b64 <= 0
                mask_neg = ~mask_pos

                if mask_pos.any():
                    xp = x[mask_pos]
                    ap64 = a64[mask_pos]
                    bp64 = b64[mask_pos]
                    ep_arg = ((xp - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
                    ep = torch.exp(ep_arg)
                    f[mask_pos] = 2 * ap64 * xp + (bp64 - I_s64) + I_s64 * ep
                    df[mask_pos] = 2 * ap64 + (I_s64 / vt64) * ep

                if mask_neg.any():
                    xn = x[mask_neg]
                    an64 = a64[mask_neg]
                    bn64 = b64[mask_neg]
                    en_arg = ((-xn - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
                    en = torch.exp(en_arg)
                    f[mask_neg] = 2 * an64 * xn + (bn64 + I_s64) - I_s64 * en
                    df[mask_neg] = 2 * an64 + (I_s64 / vt64) * en

                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f", f)
                    self._debug_tensor("df", df)
                    break

                x = x - f / df
                self._debug_tensor("x_polish", x)

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(
                    2.0 * torch.abs(a64) * torch.abs(x)
                    + torch.abs(b64)
                    + torch.abs(I_s64)
                    + 1.0
                )
                if (max_abs_f < abs_tol) and (max_abs_f / (scale + 1e-30) < rel_tol):
                    break
            self._newton_iter_total += iters_used
            self._newton_iter_calls += 1

        return x.to(out_dtype)


class Float32ExponentialDoubleDiodeUpdater(ExponentialDoubleDiodeUpdater):
    """Double-diode exponential updater with a float32 Lambert solver."""

    def __init__(self, layer, fn, diode_params, *, print_every: int = 200, synchronize_cuda: bool = True):
        super().__init__(layer, fn, diode_params)
        self._print_every = max(int(print_every), 1)
        self._print_timing = os.environ.get("LABS_PRINT_DOUBLE_DIODE_TIMING") == "1"
        env_every = os.environ.get("LABS_PRINT_DOUBLE_DIODE_TIMING_EVERY")
        if env_every is not None:
            try:
                self._print_every = max(int(env_every), 1)
            except ValueError:
                pass
        self._synchronize_cuda = bool(synchronize_cuda)
        self._timing_calls = 0
        self._timing_totals = {
            "total_pre_activate": 0.0,
            "b_coef": 0.0,
            "a_coef": 0.0,
            "lambert_hidden": 0.0,
            "lambert_w": 0.0,
            "newton_polish": 0.0,
        }
        self._timing_counts = {key: 0 for key in self._timing_totals}
        self._newton_iter_total = 0
        self._newton_iter_calls = 0

    def reset_timing_stats(self):
        self._timing_calls = 0
        for key in self._timing_totals:
            self._timing_totals[key] = 0.0
            self._timing_counts[key] = 0

    def timing_stats(self, reset=False):
        stats = {
            "calls": int(self._timing_calls),
            "totals_seconds": {key: float(value) for key, value in self._timing_totals.items()},
            "call_counts": {key: int(self._timing_counts.get(key, 0)) for key in self._timing_totals},
        }
        if reset:
            self.reset_timing_stats()
        return stats

    def set_dynamic_polish(self, *, use_polish: bool, max_newton_iters: int, z_thresh: float | None = None):
        self._use_polish = bool(use_polish)
        self._max_newton_iters = max(int(max_newton_iters), 0)
        if z_thresh is not None:
            self._z_thresh = float(z_thresh)

    def _sync_if_needed(self, tensor=None):
        if not self._synchronize_cuda or not torch.cuda.is_available():
            return
        if tensor is not None and torch.is_tensor(tensor) and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        else:
            torch.cuda.synchronize()

    def _tic(self, tensor=None):
        self._sync_if_needed(tensor)
        return time.perf_counter()

    def _toc(self, key: str, t0: float, tensor=None):
        self._sync_if_needed(tensor)
        self._timing_totals[key] += time.perf_counter() - t0
        self._timing_counts[key] += 1

    def _maybe_report_timing(self):
        if not self._print_timing:
            return
        if self._timing_calls % self._print_every != 0:
            return
        calls = max(self._timing_calls, 1)
        avg_ms = {k: (v / calls) * 1e3 for k, v in self._timing_totals.items()}
        line = (
            "[Float32ExponentialDoubleDiodeUpdater][timing] "
            f"calls={self._timing_calls} "
            f"total={avg_ms['total_pre_activate']:.3f}ms "
            f"b={avg_ms['b_coef']:.3f}ms "
            f"a={avg_ms['a_coef']:.3f}ms "
            f"lambert={avg_ms['lambert_hidden']:.3f}ms "
            f"lambert_w={avg_ms['lambert_w']:.3f}ms "
            f"newton={avg_ms['newton_polish']:.3f}ms"
        )
        _emit_timing_line(line)

    def _debug_tensor(self, name, tensor):
        if not os.environ.get("DRN_DEBUG_DIODE"):
            return
        if not torch.is_tensor(tensor):
            print(f"[Float32ExponentialDoubleDiodeUpdater] {name} is not a tensor: {type(tensor)}")
            return
        finite = torch.isfinite(tensor)
        if finite.all():
            return
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if finite.any():
            finite_vals = tensor[finite]
            finite_min = finite_vals.min().item()
            finite_max = finite_vals.max().item()
            finite_mean = finite_vals.mean().item()
        else:
            finite_min = None
            finite_max = None
            finite_mean = None
        message = (
            "[Float32ExponentialDoubleDiodeUpdater] non-finite in "
            f"{name}: nan={nan_count} inf={inf_count} "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} finite_min={finite_min} "
            f"finite_max={finite_max} finite_mean={finite_mean}"
        )
        print(message)
        if os.environ.get("DRN_DEBUG_BREAK_ON_ERROR") == "1":
            breakpoint()
        raise NonFiniteDiodeError(message)

    def pre_activate(self):
        t_total = self._tic()

        t = self._tic()
        b = self._b()
        self._toc("b_coef", t, b)

        t = self._tic(b)
        a = self._a().expand_as(b)
        self._toc("a_coef", t, a)

        self._debug_tensor("a", a)
        self._debug_tensor("b", b)
        if isinstance(self._layer, NonlinearResistiveLayer):
            if (
                not hasattr(self, "_z_thresh")
                or not hasattr(self, "_use_polish")
                or not hasattr(self, "_exp_clip")
                or not hasattr(self, "_max_newton_iters")
            ):
                raise RuntimeError("Float32ExponentialDoubleDiodeUpdater missing polish settings.")
            t = self._tic(a)
            v = self.lambert_hidden(
                a,
                b,
                self._Is,
                self._v_off,
                self._Vt,
                z_thresh=self._z_thresh,
                polish=self._use_polish,
                exp_clip=self._exp_clip,
                max_inner_iter=self._max_newton_iters,
            )
            self._toc("lambert_hidden", t, v)
            self._debug_tensor("v", v)
        else:
            v = -b / (2.0 * a)

        self._toc("total_pre_activate", t_total, v)
        self._timing_calls += 1
        self._maybe_report_timing()
        return v

    def lambertw_large(self, z, terms=4):
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w += L2 / L1
        if terms >= 3:
            w += (L2 * (-2 + L2)) / (2 * L1**2)
        if terms >= 4:
            w += (L2 * (6 - 9 * L2 + 2 * L2**2)) / (6 * L1**3)
        return w

    def lambert_hidden(self, a, b, I_s, v_off, vt, z_thresh, polish, exp_clip, max_inner_iter):
        out_dtype = a.dtype
        device = a.device

        # Keep the updater in float32 and only run Lambert-W evaluation in float64.
        work_dtype = torch.float32
        a_work = a.to(work_dtype)
        b_work = b.to(work_dtype)
        if getattr(self, "_b_clip", None) is not None:
            b_work = b_work.clamp(min=-self._b_clip, max=self._b_clip)
        I_s_work = torch.as_tensor(I_s, dtype=work_dtype, device=device)
        vt_work = torch.as_tensor(vt, dtype=work_dtype, device=device)
        v_off_work = torch.as_tensor(v_off, dtype=work_dtype, device=device)

        self._debug_tensor("a_work", a_work)
        self._debug_tensor("b_work", b_work)

        A = torch.where(
            b_work <= 0,
            (b_work - I_s_work) / (2.0 * a_work),
            (b_work + I_s_work) / (2.0 * a_work),
        )

        exp_arg_add = (-(A + v_off_work) / vt_work).clamp(min=-exp_clip, max=exp_clip)
        exp_arg_rev = ((A - v_off_work) / vt_work).clamp(min=-exp_clip, max=exp_clip)
        z_add = (I_s_work / (2.0 * a_work * vt_work)) * torch.exp(exp_arg_add)
        z_rev = (I_s_work / (2.0 * a_work * vt_work)) * torch.exp(exp_arg_rev)
        z = torch.where(b_work > 0, z_rev, z_add)
        self._debug_tensor("z", z)

        use_asym = z > z_thresh
        small_mask = ~use_asym
        W0 = torch.empty_like(z, dtype=work_dtype, device=device)

        t_lw = self._tic(z)
        if small_mask.any():
            z_small = torch.clamp(z[small_mask], max=float(z_thresh)).to(torch.float64)
            W0_small = lambertw(z_small).real.to(work_dtype)
            W0[small_mask] = W0_small

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0).to(torch.float64)
            W0_large = self.lambertw_large(z_large).to(work_dtype)
            W0[use_asym] = W0_large
        self._toc("lambert_w", t_lw, W0)

        x = torch.where(b_work > 0, vt_work * W0 - A, -vt_work * W0 - A)

        if polish:
            t_newton = self._tic(x)
            iters_used = 0
            current_tol = 1e-6 * x.shape[0]
            f = torch.zeros_like(x)
            df = torch.zeros_like(x)
            for _ in range(1, max_inner_iter):
                iters_used += 1
                mask_pos = b_work <= 0
                mask_neg = ~mask_pos

                if mask_pos.any():
                    xp = x[mask_pos]
                    ap_work = a_work[mask_pos]
                    bp_work = b_work[mask_pos]
                    ep_arg = ((xp - v_off_work) / vt_work).clamp(min=-exp_clip, max=exp_clip)
                    ep = torch.exp(ep_arg)
                    f[mask_pos] = 2 * ap_work * xp + (bp_work - I_s_work) + I_s_work * ep
                    df[mask_pos] = 2 * ap_work + (I_s_work / vt_work) * ep

                if mask_neg.any():
                    xn = x[mask_neg]
                    an_work = a_work[mask_neg]
                    bn_work = b_work[mask_neg]
                    en_arg = ((-xn - v_off_work) / vt_work).clamp(min=-exp_clip, max=exp_clip)
                    en = torch.exp(en_arg)
                    f[mask_neg] = 2 * an_work * xn + (bn_work + I_s_work) - I_s_work * en
                    df[mask_neg] = 2 * an_work + (I_s_work / vt_work) * en

                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f", f)
                    self._debug_tensor("df", df)
                    break

                x = x - f / df
                if torch.norm(f) < current_tol:
                    break
            self._toc("newton_polish", t_newton, x)
            self._newton_iter_total += iters_used
            self._newton_iter_calls += 1

        return x.to(out_dtype)


class OverRelazedDoubleDiodeExponentialUpdater(Float32ExponentialDoubleDiodeUpdater):
    """Float32 double-diode updater with SOR-style overrelaxation.

    Note: class name intentionally keeps the requested spelling: "Relazed".
    """

    def __init__(
        self,
        layer,
        fn,
        diode_params,
        *,
        overrelaxation_factor: float = 1.3,
        print_every: int = 200,
        synchronize_cuda: bool = True,
    ):
        super().__init__(layer, fn, diode_params, print_every=print_every, synchronize_cuda=synchronize_cuda)
        self._overrelaxation_factor = float(overrelaxation_factor)

    def set_overrelaxation_factor(self, value: float):
        self._overrelaxation_factor = float(value)

    def pre_activate(self):
        # Standard coordinate update candidate.
        v_cd = super().pre_activate()
        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v_cd

        # SOR update: v <- v_old + omega * (v_cd - v_old)
        v_old = self._layer.state
        v_relaxed = v_old + omega * (v_cd - v_old)
        self._debug_tensor("v_relaxed", v_relaxed)
        return v_relaxed


class TimedExponentialDOubleDiodeUpdater(CustomExponentialDoubleDiodeUpdater):
    """Custom double-diode updater with lightweight per-call timing."""

    def __init__(self, layer, fn, diode_params, *, print_every: int = 200, synchronize_cuda: bool = True):
        super().__init__(layer, fn, diode_params)
        self._print_every = max(int(print_every), 1)
        self._print_timing = os.environ.get("LABS_PRINT_DOUBLE_DIODE_TIMING") == "1"
        env_every = os.environ.get("LABS_PRINT_DOUBLE_DIODE_TIMING_EVERY")
        if env_every is not None:
            try:
                self._print_every = max(int(env_every), 1)
            except ValueError:
                pass
        self._synchronize_cuda = bool(synchronize_cuda)
        self._timing_calls = 0
        self._timing_totals = {
            "total_pre_activate": 0.0,
            "b_coef": 0.0,
            "a_coef": 0.0,
            "lambert_hidden": 0.0,
            "lambert_w": 0.0,
            "newton_polish": 0.0,
        }
        self._timing_counts = {key: 0 for key in self._timing_totals}

    def reset_timing_stats(self):
        self._timing_calls = 0
        for key in self._timing_totals:
            self._timing_totals[key] = 0.0
            self._timing_counts[key] = 0

    def timing_stats(self, reset=False):
        stats = {
            "calls": int(self._timing_calls),
            "totals_seconds": {key: float(value) for key, value in self._timing_totals.items()},
            "call_counts": {key: int(self._timing_counts.get(key, 0)) for key in self._timing_totals},
        }
        if reset:
            self.reset_timing_stats()
        return stats

    def _sync_if_needed(self, tensor=None):
        if not self._synchronize_cuda or not torch.cuda.is_available():
            return
        if tensor is not None and torch.is_tensor(tensor) and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        else:
            torch.cuda.synchronize()

    def _tic(self, tensor=None):
        self._sync_if_needed(tensor)
        return time.perf_counter()

    def _toc(self, key: str, t0: float, tensor=None):
        self._sync_if_needed(tensor)
        self._timing_totals[key] += time.perf_counter() - t0
        self._timing_counts[key] += 1

    def _maybe_report_timing(self):
        if not self._print_timing:
            return
        if self._timing_calls % self._print_every != 0:
            return
        calls = max(self._timing_calls, 1)
        avg_ms = {k: (v / calls) * 1e3 for k, v in self._timing_totals.items()}
        line = (
            "[TimedExponentialDOubleDiodeUpdater] "
            f"calls={self._timing_calls} "
            f"total={avg_ms['total_pre_activate']:.3f}ms "
            f"b={avg_ms['b_coef']:.3f}ms "
            f"a={avg_ms['a_coef']:.3f}ms "
            f"lambert={avg_ms['lambert_hidden']:.3f}ms "
            f"lambert_w={avg_ms['lambert_w']:.3f}ms "
            f"newton={avg_ms['newton_polish']:.3f}ms"
        )
        _emit_timing_line(line)

    def _debug_tensor(self, name, tensor):
        if not os.environ.get("DRN_DEBUG_DIODE"):
            return
        if not torch.is_tensor(tensor):
            print(f"[CustomExponentialSingleDiodeUpdater] {name} is not a tensor: {type(tensor)}")
            return
        finite = torch.isfinite(tensor)
        if finite.all():
            return
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if finite.any():
            finite_vals = tensor[finite]
            finite_min = finite_vals.min().item()
            finite_max = finite_vals.max().item()
            finite_mean = finite_vals.mean().item()
        else:
            finite_min = None
            finite_max = None
            finite_mean = None
        message = (
            "[CustomExponentialSingleDiodeUpdater] non-finite in "
            f"{name}: nan={nan_count} inf={inf_count} "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} finite_min={finite_min} "
            f"finite_max={finite_max} finite_mean={finite_mean}"
        )
        print(message)
        if os.environ.get("DRN_DEBUG_BREAK_ON_ERROR") == "1":
            breakpoint()
        raise NonFiniteDiodeError(message)

    def pre_activate(self):
        t_total = self._tic()

        t = self._tic()
        b = self._b()
        self._toc("b_coef", t, b)

        t = self._tic(b)
        a = self._a().expand_as(b)
        self._toc("a_coef", t, a)

        self._debug_tensor("a", a)
        self._debug_tensor("b", b)

        if isinstance(self._layer, NonlinearResistiveLayer):
            t = self._tic(a)
            if (
                not hasattr(self, "_z_thresh")
                or not hasattr(self, "_use_polish")
                or not hasattr(self, "_exp_clip")
                or not hasattr(self, "_max_newton_iters")
            ):
                raise RuntimeError("TimedExponentialDOubleDiodeUpdater missing polish settings.")
            v = self.lambert_hidden(
                a,
                b,
                self._Is,
                self._v_off,
                self._Vt,
                z_thresh=self._z_thresh,
                polish=self._use_polish,
                exp_clip=self._exp_clip,
                max_inner_iter=self._max_newton_iters,
            )
            self._toc("lambert_hidden", t, v)
            self._debug_tensor("v", v)
        else:
            v = -b / (2.0 * a)

        self._toc("total_pre_activate", t_total, v)
        self._timing_calls += 1
        self._maybe_report_timing()
        return v

    def lambert_hidden(self, a, b, I_s, v_off, vt, z_thresh, polish, exp_clip, max_inner_iter=None):
        out_dtype = a.dtype
        device = a.device

        a64 = a.to(torch.float64)
        b64 = b.to(torch.float64)
        if getattr(self, "_b_clip", None) is not None:
            b64 = b64.clamp(min=-self._b_clip, max=self._b_clip)
        I_s64 = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vt64 = torch.as_tensor(vt, dtype=torch.float64, device=device)
        v_off64 = torch.as_tensor(v_off, dtype=torch.float64, device=device)
        self._debug_tensor("a64", a64)
        self._debug_tensor("b64", b64)
        self._debug_tensor("I_s64", I_s64)
        self._debug_tensor("vt64", vt64)
        self._debug_tensor("v_off64", v_off64)

        A = torch.where(b64 <= 0, (b64 - I_s64) / (2.0 * a64),
                        (b64 + I_s64) / (2.0 * a64))
        self._debug_tensor("A", A)

        exp_arg_add = (-(A + v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        exp_arg_rev = ((A - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
        self._debug_tensor("exp_arg_add", exp_arg_add)
        self._debug_tensor("exp_arg_rev", exp_arg_rev)

        z_add = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_add)
        z_rev = (I_s64 / (2.0 * a64 * vt64)) * torch.exp(exp_arg_rev)
        z = torch.where(b64 > 0, z_rev, z_add)
        self._debug_tensor("z_add", z_add)
        self._debug_tensor("z_rev", z_rev)
        self._debug_tensor("z", z)

        t_lw = self._tic(z)
        use_asym = z > z_thresh
        small_mask = ~use_asym
        W0 = torch.empty_like(z, dtype=torch.float64, device=z.device)

        if small_mask.any():
            z_small = torch.clamp(z[small_mask].to(torch.float64), max=float(z_thresh))
            self._debug_tensor("z_small", z_small)
            W0_small = lambertw(z_small).real.to(W0.dtype)
            self._debug_tensor("W0_small", W0_small)
            W0[small_mask] = W0_small

        if use_asym.any():
            z_large = torch.clamp(z[use_asym].to(torch.float64), min=1.0)
            self._debug_tensor("z_large", z_large)
            W0_large = self.lambertw_large(z_large).to(torch.float64)
            self._debug_tensor("W0_large", W0_large)
            W0[use_asym] = W0_large
        self._toc("lambert_w", t_lw, W0)

        self._debug_tensor("W0", W0)
        x = torch.where(b64 > 0, vt64 * W0 - A, -vt64 * W0 - A)
        self._debug_tensor("x_pre_polish", x)

        if polish:
            t_newton = self._tic(x)
            max_inner_iter = 16 if max_inner_iter is None else max(int(max_inner_iter), 1)
            iters_used = 0
            current_tol = 1e-6 * x.shape[0]
            f = torch.zeros_like(x)
            df = torch.zeros_like(x)
            for _ in range(1, max_inner_iter):
                iters_used += 1
                mask_pos = b64 <= 0
                mask_neg = ~mask_pos

                if mask_pos.any():
                    xp = x[mask_pos]
                    ap64 = a64[mask_pos]
                    bp64 = b64[mask_pos]
                    ep_arg = ((xp - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
                    ep = torch.exp(ep_arg)
                    f[mask_pos] = 2 * ap64 * xp + (bp64 - I_s) + I_s * ep
                    df[mask_pos] = 2 * ap64 + (I_s / vt) * ep

                if mask_neg.any():
                    xn = x[mask_neg]
                    an64 = a64[mask_neg]
                    bn64 = b64[mask_neg]
                    en_arg = ((-xn - v_off64) / vt64).clamp(min=-exp_clip, max=exp_clip)
                    en = torch.exp(en_arg)
                    f[mask_neg] = 2 * an64 * xn + (bn64 + I_s) - I_s * en
                    df[mask_neg] = 2 * an64 + (I_s / vt) * en

                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f", f)
                    self._debug_tensor("df", df)
                    break

                x = x - f / df
                self._debug_tensor("x_polish", x)
                if torch.norm(f) < current_tol:
                    break
            self._toc("newton_polish", t_newton, x)
            self._newton_iter_total += iters_used
            self._newton_iter_calls += 1

        return x.to(out_dtype)


class OverRelazedTimedExponentialDoubleDiodeUpdater(TimedExponentialDOubleDiodeUpdater):
    """Float64 timed double-diode updater with SOR-style overrelaxation.

    Note: class name intentionally keeps the requested spelling: "Relazed".
    """

    def __init__(
        self,
        layer,
        fn,
        diode_params,
        *,
        overrelaxation_factor: float = 1.3,
        print_every: int = 200,
        synchronize_cuda: bool = True,
    ):
        super().__init__(layer, fn, diode_params, print_every=print_every, synchronize_cuda=synchronize_cuda)
        self._overrelaxation_factor = float(overrelaxation_factor)

    def set_overrelaxation_factor(self, value: float):
        self._overrelaxation_factor = float(value)

    def pre_activate(self):
        v_cd = super().pre_activate()
        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v_cd

        v_old = self._layer.state
        v_relaxed = v_old + omega * (v_cd - v_old)
        self._debug_tensor("v_relaxed", v_relaxed)
        return v_relaxed


class CustomExponentialSingleDiodeUpdater(ExponentialSingleDiodeUpdater):
    """Override pre_activate if you want custom exponential single-diode updates."""

    def __init__(self, layer, fn, diode_params, *, print_every: int = 200, synchronize_cuda: bool = True):
        super().__init__(layer, fn, diode_params)
        self._print_every = max(int(print_every), 1)
        self._print_timing = os.environ.get("LABS_PRINT_SINGLE_DIODE_TIMING") == "1"
        env_every = os.environ.get("LABS_PRINT_SINGLE_DIODE_TIMING_EVERY")
        if env_every is not None:
            try:
                self._print_every = max(int(env_every), 1)
            except ValueError:
                pass
        self._synchronize_cuda = bool(synchronize_cuda)
        self._timing_calls = 0
        self.b_clip = float(os.environ.get("DRN_B_CLAMP"))
        self._timing_totals = {
            "total_pre_activate": 0.0,
            "b_coef": 0.0,
            "a_coef": 0.0,
            "lambert_forward": 0.0,
            "lambert_reverse": 0.0,
            "lambert_w": 0.0,
            "newton_polish": 0.0,
        }
        self._newton_iter_total = 0
        self._newton_iter_calls = 0

    def set_dynamic_polish(self, *, use_polish: bool, max_newton_iters: int, z_thresh: float | None = None):
        self._use_polish = bool(use_polish)
        self._max_newton_iters = max(int(max_newton_iters), 0)
        if z_thresh is not None:
            self._z_thresh = float(z_thresh)

    def _sync_if_needed(self, tensor=None):
        if not self._synchronize_cuda or not torch.cuda.is_available():
            return
        if tensor is not None and torch.is_tensor(tensor) and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        else:
            torch.cuda.synchronize()

    def _tic(self, tensor=None):
        self._sync_if_needed(tensor)
        return time.perf_counter()

    def _toc(self, key: str, t0: float, tensor=None):
        self._sync_if_needed(tensor)
        self._timing_totals[key] += time.perf_counter() - t0

    def _maybe_report_timing(self):
        if not self._print_timing:
            return
        if self._timing_calls % self._print_every != 0:
            return
        calls = max(self._timing_calls, 1)
        avg_ms = {k: (v / calls) * 1e3 for k, v in self._timing_totals.items()}
        line = (
            "[CustomExponentialSingleDiodeUpdater][timing] "
            f"calls={self._timing_calls} "
            f"total={avg_ms['total_pre_activate']:.3f}ms "
            f"b={avg_ms['b_coef']:.3f}ms "
            f"a={avg_ms['a_coef']:.3f}ms "
            f"fwd={avg_ms['lambert_forward']:.3f}ms "
            f"rev={avg_ms['lambert_reverse']:.3f}ms "
            f"lambert_w={avg_ms['lambert_w']:.3f}ms "
            f"newton={avg_ms['newton_polish']:.3f}ms"
        )
        _emit_timing_line(line)

    def pre_activate(self):
        t_total = self._tic()

        t = self._tic()
        b = self._b()                     # [N, M]
        self._toc("b_coef", t, b)

        t = self._tic(b)
        a = self._a().expand_as(b)         # [N, M]
        self._toc("a_coef", t, a)

        self._debug_tensor("a", a)
        self._debug_tensor("b", b)

        if not isinstance(self._layer, NonlinearResistiveLayer):
            v = -b / (2.0 * a)
            self._toc("total_pre_activate", t_total, v)
            self._timing_calls += 1
            self._maybe_report_timing()
            return v
        if (
            not hasattr(self, "_use_polish")
            or not hasattr(self, "_max_newton_iters")
            or not hasattr(self, "_z_thresh")
        ):
            raise RuntimeError("CustomExponentialSingleDiodeUpdater missing polish settings.")

        # node-wise mask: [M] -> broadcast to [N, M]
        M = b.shape[1]
        reverse_mask_nodes = torch.arange(M, device=b.device) >= (M // 2)  # [M]
        reverse_mask = reverse_mask_nodes.unsqueeze(0).expand_as(b)         # [N, M]

        t = self._tic(a)
        v_fwd = self.lambert_single_forward(
            a,
            b,
            self._Is,
            self._v_off,
            self._Vt,
            z_thresh=self._z_thresh,
            polish=self._use_polish,
            abs_tol=1e-6,
            rel_tol=1e-6,
            exp_clip=self._exp_clip,
            a_min=1e-30,
            max_inner_iter=self._max_newton_iters,
        )  # [N, M]
        self._toc("lambert_forward", t, v_fwd)

        t = self._tic(a)
        v_rev = self.lambert_single_reverse(
            a,
            b,
            self._Is,
            self._v_off,
            self._Vt,
            z_thresh=self._z_thresh,
            polish=self._use_polish,
            abs_tol=1e-6,
            rel_tol=1e-6,
            exp_clip=self._exp_clip,
            a_min=1e-30,
            max_inner_iter=self._max_newton_iters,
        )  # [N, M]
        self._toc("lambert_reverse", t, v_rev)

        self._debug_tensor("v_fwd", v_fwd)
        self._debug_tensor("v_rev", v_rev)
        v = torch.where(reverse_mask, v_rev, v_fwd)

        self._toc("total_pre_activate", t_total, v)
        self._timing_calls += 1
        self._maybe_report_timing()
        return v

    def lambertw_large(self, z, terms=4):
        L1 = torch.log(z)
        L2 = torch.log(L1)
        w = L1 - L2
        if terms >= 2:
            w = w + L2 / L1
        if terms >= 3:
            w = w + (L2 * (-2.0 + L2)) / (2.0 * L1**2)
        if terms >= 4:
            w = w + (L2 * (6.0 - 9.0*L2 + 2.0*L2**2)) / (6.0 * L1**3)
        return w

    def _lambertw0(self, z):
        if hasattr(torch, "special") and hasattr(torch.special, "lambertw"):
            return torch.special.lambertw(z).real
        return lambertw(z).real  # assumes you have lambertw imported elsewhere

    # ---------- Forward diode ----------
    # 2 a v + b + I_s (exp((v - v_on)/vT) - 1) = 0
    def lambert_single_forward(
        self, a, b, I_s, v_on, vT,
        z_thresh, polish,
        abs_tol, rel_tol,
        exp_clip, a_min, max_inner_iter
    ):
        out_dtype = a.dtype
        device = a.device

        a64 = torch.clamp(a.to(torch.float64), min=a_min)
        b64 = b.to(torch.float64).clamp(min = -self.b_clip, max=self.b_clip)

        Is64  = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vT64  = torch.as_tensor(vT,  dtype=torch.float64, device=device)
        von64 = torch.as_tensor(v_on, dtype=torch.float64, device=device)
        self._debug_tensor("a64", a64)
        self._debug_tensor("b64", b64)
        self._debug_tensor("Is64", Is64)
        self._debug_tensor("vT64", vT64)
        self._debug_tensor("von64", von64)

        # Ashift = (b - I_s)/(2a)
        Ashift = (b64 - Is64) / (2.0 * a64)
        self._debug_tensor("Ashift_fwd", Ashift)

        # z = (I_s/(2 a vT)) * exp(-(v_on + Ashift)/vT) with clipped exponent
        exp_arg = (-(Ashift + von64) / vT64).clamp(max=exp_clip)
        z = (Is64 / (2.0 * a64 * vT64)) * torch.exp(exp_arg)
        self._debug_tensor("z_fwd", z)
        if not torch.isfinite(z).all():
            self._debug_tensor("z_fwd_non_finite", z)
            import pdb
            pdb.set_trace()
            raise NonFiniteDiodeError("CustomExponentialSingleDiodeUpdater: non-finite z in lambert_single_forward.")

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)

        t_lw = self._tic(z)
        if (~use_asym).any():
            z_small = torch.clamp(z[~use_asym], max=float(z_thresh))
            W0[~use_asym] = self._lambertw0(z_small).to(torch.float64)

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0)
            W0[use_asym] = self.lambertw_large(z_large).to(torch.float64)
        self._toc("lambert_w", t_lw, W0)
        self._debug_tensor("W0_fwd", W0)

        # v* = -Ashift - vT * W0
        x = -Ashift - vT64 * W0
        self._debug_tensor("x_fwd", x)

        if polish:
            t_newton = self._tic(x)
            iters_used = 0
            for _ in range(max_inner_iter):
                iters_used += 1
                arg = (x - von64) / vT64
                arg = torch.clamp(arg, min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)

                f  = 2.0*a64*x + b64 + Is64*(e - 1.0)
                df = 2.0*a64   + (Is64/vT64)*e
                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f_fwd", f)
                    self._debug_tensor("df_fwd", df)
                    raise NonFiniteDiodeError(
                        "CustomExponentialSingleDiodeUpdater: non-finite f/df in lambert_single_forward."
                    )

                step = f / df
                x = x - step

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(2.0*torch.abs(a64)*torch.abs(x) + torch.abs(b64) + torch.abs(Is64) + 1.0)
                if (max_abs_f < abs_tol) and (max_abs_f/(scale + 1e-30) < rel_tol):
                    break
            self._toc("newton_polish", t_newton, x)
            self._newton_iter_total += iters_used
            self._newton_iter_calls += 1

        return x.to(out_dtype)

    # ---------- Reversed diode (your corrected equation) ----------
    # 2 a v + b - I_s (exp((-(v + v_on))/vT) - 1) = 0
    # <=> 2 a v + (b + I_s) - I_s exp(-(v + v_on)/vT) = 0
    def lambert_single_reverse(
        self, a, b, I_s, v_on, vT,
        z_thresh, polish,
        abs_tol, rel_tol,
        exp_clip, a_min, max_inner_iter
    ):
        out_dtype = a.dtype
        device = a.device

        a64 = torch.clamp(a.to(torch.float64), min=a_min)
        b64 = b.to(torch.float64).clamp(min = -self.b_clip, max=self.b_clip)

        Is64  = torch.as_tensor(I_s, dtype=torch.float64, device=device)
        vT64  = torch.as_tensor(vT,  dtype=torch.float64, device=device)
        von64 = torch.as_tensor(v_on, dtype=torch.float64, device=device)
        self._debug_tensor("a64", a64)
        self._debug_tensor("b64", b64)
        self._debug_tensor("Is64", Is64)
        self._debug_tensor("vT64", vT64)
        self._debug_tensor("von64", von64)

        # Ashift = (b + I_s)/(2a)
        Ashift = (b64 + Is64) / (2.0 * a64)
        self._debug_tensor("Ashift_rev", Ashift)

        # z = (I_s/(2 a vT)) * exp( -v_on/vT + Ashift/vT ) with clipped exponent
        exp_arg = ((-von64 + Ashift) / vT64).clamp(min=-exp_clip, max=exp_clip)
        z = (Is64 / (2.0 * a64 * vT64)) * torch.exp(exp_arg)
        self._debug_tensor("z_rev", z)
        if not torch.isfinite(z).all():
            self._debug_tensor("z_rev_non_finite", z)
            import pdb
            pdb.set_trace()
            raise NonFiniteDiodeError("CustomExponentialSingleDiodeUpdater: non-finite z in lambert_single_reverse.")

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=torch.float64, device=device)

        t_lw = self._tic(z)
        if (~use_asym).any():
            z_small = torch.clamp(z[~use_asym], max=float(z_thresh))
            W0[~use_asym] = self._lambertw0(z_small).to(torch.float64)

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0)
            W0[use_asym] = self.lambertw_large(z_large).to(torch.float64)
        self._toc("lambert_w", t_lw, W0)
        self._debug_tensor("W0_rev", W0)

        # v* = -Ashift + vT * W0
        x = -Ashift + vT64 * W0
        self._debug_tensor("x_rev", x)

        if polish:
            t_newton = self._tic(x)
            iters_used = 0
            for _ in range(max_inner_iter):
                iters_used += 1
                arg = -(x + von64) / vT64
                arg = torch.clamp(arg, min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)

                # f = 2 a v + b + I_s - I_s e^{-(v+v_on)/vT}
                f  = 2.0*a64*x + b64 + Is64 - Is64*e
                # df = 2 a + (I_s/vT) e^{-(v+v_on)/vT}
                df = 2.0*a64   + (Is64/vT64)*e
                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f_rev", f)
                    self._debug_tensor("df_rev", df)
                    raise NonFiniteDiodeError(
                        "CustomExponentialSingleDiodeUpdater: non-finite f/df in lambert_single_reverse."
                    )

                step = f / df
                x = x - step

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(2.0*torch.abs(a64)*torch.abs(x) + torch.abs(b64) + torch.abs(Is64) + 1.0)
                if (max_abs_f < abs_tol) and (max_abs_f/(scale + 1e-30) < rel_tol):
                    break
            self._toc("newton_polish", t_newton, x)
            self._newton_iter_total += iters_used
            self._newton_iter_calls += 1

        return x.to(out_dtype)


class Float32ExponentialSingleDiodeUpdater(CustomExponentialSingleDiodeUpdater):
    """Single-diode exponential updater that keeps the entire solve in float32."""

    def _debug_tensor(self, name, tensor):
        if not os.environ.get("DRN_DEBUG_DIODE"):
            return
        if not torch.is_tensor(tensor):
            print(f"[Float32ExponentialSingleDiodeUpdater] {name} is not a tensor: {type(tensor)}")
            return
        finite = torch.isfinite(tensor)
        if finite.all():
            return
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if finite.any():
            finite_vals = tensor[finite]
            finite_min = finite_vals.min().item()
            finite_max = finite_vals.max().item()
            finite_mean = finite_vals.mean().item()
        else:
            finite_min = None
            finite_max = None
            finite_mean = None
        message = (
            "[Float32ExponentialSingleDiodeUpdater] non-finite in "
            f"{name}: nan={nan_count} inf={inf_count} "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} finite_min={finite_min} "
            f"finite_max={finite_max} finite_mean={finite_mean}"
        )
        print(message)
        if os.environ.get("DRN_DEBUG_BREAK_ON_ERROR") == "1":
            breakpoint()
        raise NonFiniteDiodeError(message)

    def _maybe_report_timing(self):
        if not self._print_timing:
            return
        if self._timing_calls % self._print_every != 0:
            return
        calls = max(self._timing_calls, 1)
        avg_ms = {k: (v / calls) * 1e3 for k, v in self._timing_totals.items()}
        line = (
            "[Float32ExponentialSingleDiodeUpdater][timing] "
            f"calls={self._timing_calls} "
            f"total={avg_ms['total_pre_activate']:.3f}ms "
            f"b={avg_ms['b_coef']:.3f}ms "
            f"a={avg_ms['a_coef']:.3f}ms "
            f"fwd={avg_ms['lambert_forward']:.3f}ms "
            f"rev={avg_ms['lambert_reverse']:.3f}ms "
            f"lambert_w={avg_ms['lambert_w']:.3f}ms "
            f"newton={avg_ms['newton_polish']:.3f}ms"
        )
        _emit_timing_line(line)

    def lambert_single_forward(
        self, a, b, I_s, v_on, vT,
        z_thresh, polish,
        abs_tol, rel_tol,
        exp_clip, a_min, max_inner_iter
    ):
        out_dtype = a.dtype
        device = a.device
        work_dtype = torch.float32

        a_work = torch.clamp(a.to(work_dtype), min=a_min)
        b_work = b.to(work_dtype).clamp(min=-self.b_clip, max=self.b_clip)

        Is_work = torch.as_tensor(I_s, dtype=work_dtype, device=device)
        vT_work = torch.as_tensor(vT, dtype=work_dtype, device=device)
        von_work = torch.as_tensor(v_on, dtype=work_dtype, device=device)
        self._debug_tensor("a_work", a_work)
        self._debug_tensor("b_work", b_work)
        self._debug_tensor("Is_work", Is_work)
        self._debug_tensor("vT_work", vT_work)
        self._debug_tensor("von_work", von_work)

        Ashift = (b_work - Is_work) / (2.0 * a_work)
        self._debug_tensor("Ashift_fwd", Ashift)

        exp_arg = (-(Ashift + von_work) / vT_work).clamp(min=-exp_clip, max=exp_clip)
        z = (Is_work / (2.0 * a_work * vT_work)) * torch.exp(exp_arg)
        self._debug_tensor("z_fwd", z)
        if not torch.isfinite(z).all():
            self._debug_tensor("z_fwd_non_finite", z)
            raise NonFiniteDiodeError(
                "Float32ExponentialSingleDiodeUpdater: non-finite z in lambert_single_forward."
            )

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=work_dtype, device=device)

        t_lw = self._tic(z)
        if (~use_asym).any():
            z_small = torch.clamp(z[~use_asym], max=float(z_thresh))
            W0[~use_asym] = self._lambertw0(z_small).to(work_dtype)

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0)
            W0[use_asym] = self.lambertw_large(z_large).to(work_dtype)
        self._toc("lambert_w", t_lw, W0)
        self._debug_tensor("W0_fwd", W0)

        x = -Ashift - vT_work * W0
        self._debug_tensor("x_fwd", x)

        if polish:
            t_newton = self._tic(x)
            iters_used = 0
            for _ in range(max_inner_iter):
                iters_used += 1
                arg = ((x - von_work) / vT_work).clamp(min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)

                f = 2.0 * a_work * x + b_work + Is_work * (e - 1.0)
                df = 2.0 * a_work + (Is_work / vT_work) * e
                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f_fwd", f)
                    self._debug_tensor("df_fwd", df)
                    raise NonFiniteDiodeError(
                        "Float32ExponentialSingleDiodeUpdater: non-finite f/df in lambert_single_forward."
                    )

                step = f / df
                x = x - step

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(2.0 * torch.abs(a_work) * torch.abs(x) + torch.abs(b_work) + torch.abs(Is_work) + 1.0)
                if (max_abs_f < abs_tol) and (max_abs_f / (scale + 1e-30) < rel_tol):
                    break
            self._toc("newton_polish", t_newton, x)
            self._newton_iter_total += iters_used
            self._newton_iter_calls += 1

        return x.to(out_dtype)

    def lambert_single_reverse(
        self, a, b, I_s, v_on, vT,
        z_thresh, polish,
        abs_tol, rel_tol,
        exp_clip, a_min, max_inner_iter
    ):
        out_dtype = a.dtype
        device = a.device
        work_dtype = torch.float32

        a_work = torch.clamp(a.to(work_dtype), min=a_min)
        b_work = b.to(work_dtype).clamp(min=-self.b_clip, max=self.b_clip)

        Is_work = torch.as_tensor(I_s, dtype=work_dtype, device=device)
        vT_work = torch.as_tensor(vT, dtype=work_dtype, device=device)
        von_work = torch.as_tensor(v_on, dtype=work_dtype, device=device)
        self._debug_tensor("a_work", a_work)
        self._debug_tensor("b_work", b_work)
        self._debug_tensor("Is_work", Is_work)
        self._debug_tensor("vT_work", vT_work)
        self._debug_tensor("von_work", von_work)

        Ashift = (b_work + Is_work) / (2.0 * a_work)
        self._debug_tensor("Ashift_rev", Ashift)

        exp_arg = ((-von_work + Ashift) / vT_work).clamp(min=-exp_clip, max=exp_clip)
        z = (Is_work / (2.0 * a_work * vT_work)) * torch.exp(exp_arg)
        self._debug_tensor("z_rev", z)
        if not torch.isfinite(z).all():
            self._debug_tensor("z_rev_non_finite", z)
            raise NonFiniteDiodeError(
                "Float32ExponentialSingleDiodeUpdater: non-finite z in lambert_single_reverse."
            )

        use_asym = z > z_thresh
        W0 = torch.empty_like(z, dtype=work_dtype, device=device)

        t_lw = self._tic(z)
        if (~use_asym).any():
            z_small = torch.clamp(z[~use_asym], max=float(z_thresh))
            W0[~use_asym] = self._lambertw0(z_small).to(work_dtype)

        if use_asym.any():
            z_large = torch.clamp(z[use_asym], min=1.0)
            W0[use_asym] = self.lambertw_large(z_large).to(work_dtype)
        self._toc("lambert_w", t_lw, W0)
        self._debug_tensor("W0_rev", W0)

        x = -Ashift + vT_work * W0
        self._debug_tensor("x_rev", x)

        if polish:
            t_newton = self._tic(x)
            iters_used = 0
            for _ in range(max_inner_iter):
                iters_used += 1
                arg = (-(x + von_work) / vT_work).clamp(min=-exp_clip, max=exp_clip)
                e = torch.exp(arg)

                f = 2.0 * a_work * x + b_work + Is_work - Is_work * e
                df = 2.0 * a_work + (Is_work / vT_work) * e
                if not torch.isfinite(f).all() or not torch.isfinite(df).all():
                    self._debug_tensor("f_rev", f)
                    self._debug_tensor("df_rev", df)
                    raise NonFiniteDiodeError(
                        "Float32ExponentialSingleDiodeUpdater: non-finite f/df in lambert_single_reverse."
                    )

                step = f / df
                x = x - step

                max_abs_f = torch.max(torch.abs(f))
                scale = torch.max(2.0 * torch.abs(a_work) * torch.abs(x) + torch.abs(b_work) + torch.abs(Is_work) + 1.0)
                if (max_abs_f < abs_tol) and (max_abs_f / (scale + 1e-30) < rel_tol):
                    break
            self._toc("newton_polish", t_newton, x)
            self._newton_iter_total += iters_used
            self._newton_iter_calls += 1

        return x.to(out_dtype)


class OverRelazedSingleDiodeExponentialUpdater(CustomExponentialSingleDiodeUpdater):
    """Single-diode updater with SOR-style overrelaxation.

    Note: class name intentionally keeps the requested spelling: "Relazed".
    """

    def __init__(
        self,
        layer,
        fn,
        diode_params,
        *,
        overrelaxation_factor: float = 1.3,
        print_every: int = 200,
        synchronize_cuda: bool = True,
    ):
        super().__init__(layer, fn, diode_params, print_every=print_every, synchronize_cuda=synchronize_cuda)
        self._overrelaxation_factor = float(overrelaxation_factor)

    def set_overrelaxation_factor(self, value: float):
        self._overrelaxation_factor = float(value)

    def pre_activate(self):
        v_cd = super().pre_activate()
        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v_cd
        v_old = self._layer.state
        v_relaxed = v_old + omega * (v_cd - v_old)
        self._debug_tensor("v_relaxed", v_relaxed)
        return v_relaxed


class CustomHardSigmoidUpdater(HardSigmoidUpdater):
    """Override pre_activate if you want custom hard-sigmoid updates."""


class ExperimentalIVcurveUpdater(LayerUpdater):
    """LayerUpdater for experimental I-V curve updates."""

    def __init__(
        self,
        layer,
        fn,
        iv_data,
        *,
        damping: float = 0.5,
        max_newton_steps: int = 100,
        newton_tol: float = 1e-5,
        clamp: bool = True,
    ):
        super().__init__(layer, fn)
        self._iv_data = iv_data
        self._damping = float(damping)
        self._max_newton_steps = max(int(max_newton_steps), 1)
        self._newton_tol = max(float(newton_tol), 1e-12)
        extrapolation = os.environ.get("LABS_IV_EXTRAPOLATION", "clamp").strip().lower()
        if extrapolation not in {"clamp", "linear"}:
            extrapolation = "clamp"
        # "linear" => allow extrapolation beyond v_min/v_max using end-segment slopes.
        self._clamp = bool(clamp) and extrapolation != "linear"
        self._extrapolation = extrapolation
        self._print_newton_iters = os.environ.get("LABS_PRINT_NEWTON_ITERS") == "1"
        self._print_every = int(os.environ.get("LABS_PRINT_NEWTON_ITERS_EVERY", "1"))
        self._print_counter = 0
        self._print_newton_timing = os.environ.get("LABS_PRINT_NEWTON_TIMING") == "1"
        self._timing_every = int(os.environ.get("LABS_PRINT_NEWTON_TIMING_EVERY", "1"))
        self._timing_counter = 0
        self._timing_total = 0.0
        self._timing_calls = 0
        # Cumulative diagnostics consumed by experiment probes and reports.
        self._newton_iter_total = 0
        self._newton_iter_calls = 0
        # Standard quadratic coefficients (matching QuadraticUpdater interface)
        self._a = fn.a_coef_fn(layer)
        self._b = fn.b_coef_fn(layer)

    def pre_activate(self):
        b = self._b()
        a = self._a()
        if not isinstance(self._layer, NonlinearResistiveLayer):
            return -b / (2.0 * a)
        i_data = self._iv_data[0].to(device=a.device, dtype=a.dtype)
        v_data = self._iv_data[1].to(device=a.device, dtype=a.dtype)
        slope = (i_data[1:] - i_data[:-1]) / (v_data[1:] - v_data[:-1])
        v_min = v_data[0]
        v_max = v_data[-1]
        eps = torch.finfo(a.dtype).eps

        def newton_step(v):
            idx = (torch.bucketize(v, v_data) - 1).clamp(0, len(v_data) - 2)
            deriv = slope[idx]
            i = i_data[idx] + (v - v_data[idx]) * deriv
            fun = 2 * a * v + b + i
            fun_prime = 2 * a + deriv + eps
            v_new = v - self._damping * (fun / fun_prime)
            if self._clamp:
                v_new = torch.clamp(v_new, min=v_min, max=v_max)
            return v_new

        max_number_steps = self._max_newton_steps
        timing_start = time.perf_counter() if self._print_newton_timing else None
        v_old = -b / (2 * a)
        tol = self._newton_tol
        active = torch.ones_like(v_old, dtype=torch.bool)
        iters = torch.zeros_like(v_old, dtype=torch.int32)
        for step in range(max_number_steps):
            v = newton_step(v_old)
            delta = torch.abs(v - v_old)
            converged = delta < tol
            newly_converged = active & converged
            if newly_converged.any().item():
                iters[newly_converged] = step + 1
                active = active & ~converged
            v_old = v
            if not active.any().item():
                break
        if active.any().item():
            iters[active] = max_number_steps
        # Track Newton iterations per updater call so digits_validate metadata
        # can report comparable stats across updater implementations.
        self._newton_iter_total += int(iters.max().item())
        self._newton_iter_calls += 1
        if self._print_newton_iters:
            self._print_counter += 1
            if self._print_counter % max(self._print_every, 1) == 0:
                mean_iters = float(iters.float().mean().item())
                max_iters = int(iters.max().item())
                print(
                    f"ExperimentalIVcurveUpdater: avg newton iters={mean_iters:.2f} "
                    f"(max={max_iters})"
                )
        if self._print_newton_timing:
            elapsed = time.perf_counter() - timing_start
            self._timing_total += elapsed
            self._timing_calls += 1
            self._timing_counter += 1
            if self._timing_counter % max(self._timing_every, 1) == 0:
                avg = self._timing_total / max(self._timing_calls, 1)
                print(
                    f"ExperimentalIVcurveUpdater: avg newton time={avg:.6f}s "
                    f"(last={elapsed:.6f}s)"
                )
        return v_old

    def set_newton_tol(self, tol: float):
        self._newton_tol = max(float(tol), 1e-12)


class OverRelazedExperimentalIVcurveUpdater(ExperimentalIVcurveUpdater):
    """Experimental I-V curve updater with SOR-style overrelaxation.

    Note: class name intentionally keeps the requested spelling: "Relazed".
    """

    def __init__(
        self,
        layer,
        fn,
        iv_data,
        *,
        overrelaxation_factor: float = 1.3,
        damping: float = 0.5,
        max_newton_steps: int = 100,
        newton_tol: float = 1e-5,
        clamp: bool = True,
    ):
        super().__init__(
            layer,
            fn,
            iv_data,
            damping=damping,
            max_newton_steps=max_newton_steps,
            newton_tol=newton_tol,
            clamp=clamp,
        )
        self._overrelaxation_factor = float(overrelaxation_factor)
        v_vals = self._iv_data[1]
        self._v_min = float(v_vals.min().item())
        self._v_max = float(v_vals.max().item())

    def set_overrelaxation_factor(self, value: float):
        self._overrelaxation_factor = float(value)

    def pre_activate(self):
        v_cd = super().pre_activate()
        omega = self._overrelaxation_factor
        if omega == 1.0:
            return v_cd
        v_old = self._layer.state
        v_relaxed = v_old + omega * (v_cd - v_old)
        if self._clamp:
            v_relaxed = torch.clamp(v_relaxed, min=self._v_min, max=self._v_max)
        return v_relaxed

class CustomQuadraticMinimizer(CustomMinimizer):
    """Quadratic minimizer that uses the custom updater classes above."""

    def __init__(
        self,
        fn,
        free_layers,
        num_iterations,
        mode,
        non_linearity,
        quadratic_diode_param,
        exponential_diode_param,
        voltage_amp,
        current_amp,
        hard_sigmoid_param,
        iv_data,
        iv_data_path,
        double_diode_updater,
        adaptive_equilibrium,
        overrelaxation_factor,
        single_diode_updater,
        *,
        minimizer_settings: MinimizerSettings,
        damping: float = 0.5,
        experimental_newton_max_steps: int = 100,
    ):
        quadratic_params = dict(quadratic_diode_param)
        exponential_params = dict(exponential_diode_param)
        hard_sigmoid_params = dict(hard_sigmoid_param or {})

        def scaled_quadratic_params(layer):
            return _scale_diode_strength_for_layer(
                quadratic_params,
                layer,
                voltage_amp,
                current_amp,
                strength_key="diode_conductance",
            )

        def scaled_exponential_params(layer):
            return _scale_diode_strength_for_layer(
                exponential_params,
                layer,
                voltage_amp,
                current_amp,
                strength_key="I_s",
            )

        if not hard_sigmoid_params:
            if "diode_conductance" in quadratic_params:
                hard_sigmoid_params["g_on"] = quadratic_params["diode_conductance"]
                hard_sigmoid_params["g_off"] = quadratic_params["diode_conductance"]
        if "g_on" not in hard_sigmoid_params and "diode_conductance" in quadratic_params:
            hard_sigmoid_params["g_on"] = quadratic_params["diode_conductance"]
        if "g_off" not in hard_sigmoid_params and "g_on" in hard_sigmoid_params:
            hard_sigmoid_params["g_off"] = hard_sigmoid_params["g_on"]
        if "v_min" not in hard_sigmoid_params and "v_min" in quadratic_params:
            hard_sigmoid_params["v_min"] = quadratic_params["v_min"]
        if "v_max" not in hard_sigmoid_params and "v_max" in quadratic_params:
            hard_sigmoid_params["v_max"] = quadratic_params["v_max"]

        if non_linearity == "perfect_diode":
            updaters = [CustomQuadraticUpdater(layer, fn) for layer in free_layers]
        elif non_linearity == "lpw_diode":
            updaters = [
                CustomAdaptiveQuadraticUpdater(layer, fn, scaled_quadratic_params(layer))
                for layer in free_layers
            ]
        elif non_linearity == "double_diode_quadratic":
            updaters = [
                CustomQuadraticDoubleDiodeUpdaterOffset(
                    layer, fn, scaled_quadratic_params(layer)
                )
                for layer in free_layers
            ]
        elif non_linearity == "double_diode_exponential":
            if double_diode_updater in ("CustomExponentialDoubleDiodeUpdater", "custom"):
                updater_cls = CustomExponentialDoubleDiodeUpdater
                updaters = [
                    updater_cls(layer, fn, scaled_exponential_params(layer))
                    for layer in free_layers
                ]
            elif double_diode_updater in ("float64_experimental", "ExperimentalDoubleDiodeUpdater"):
                updater_cls = ExperimentalDoubleDiodeUpdater
                updaters = [
                    updater_cls(layer, fn, scaled_exponential_params(layer))
                    for layer in free_layers
                ]
            elif double_diode_updater in ("float32", "Float32ExponentialDoubleDiodeUpdater"):
                updater_cls = Float32ExponentialDoubleDiodeUpdater
                updaters = [
                    updater_cls(layer, fn, scaled_exponential_params(layer))
                    for layer in free_layers
                ]
            elif double_diode_updater in ("float64_timed", "TimedExponentialDOubleDiodeUpdater"):
                updater_cls = TimedExponentialDOubleDiodeUpdater
                updaters = [
                    updater_cls(layer, fn, scaled_exponential_params(layer))
                    for layer in free_layers
                ]
            elif double_diode_updater in (
                "float64_timed_overrelaxed",
                "OverRelazedTimedExponentialDoubleDiodeUpdater",
            ):
                updaters = [
                    OverRelazedTimedExponentialDoubleDiodeUpdater(
                        layer,
                        fn,
                        scaled_exponential_params(layer),
                        overrelaxation_factor=overrelaxation_factor,
                    )
                    for layer in free_layers
                ]
            elif double_diode_updater in (
                "overrelaxed",
                "overrelated",
                "OverRelazedDoubleDiodeExponentialUpdater",
            ):
                updaters = [
                    OverRelazedDoubleDiodeExponentialUpdater(
                        layer,
                        fn,
                        scaled_exponential_params(layer),
                        overrelaxation_factor=overrelaxation_factor,
                    )
                    for layer in free_layers
                ]
            else:
                raise ValueError(
                    "double_diode_updater must be one of: "
                    "'CustomExponentialDoubleDiodeUpdater'/'custom', "
                    "'float64_experimental'/'ExperimentalDoubleDiodeUpdater', "
                    "'float32'/'Float32ExponentialDoubleDiodeUpdater', "
                    "'overrelaxed'/'overrelated'/'OverRelazedDoubleDiodeExponentialUpdater', "
                    "'float64_timed'/'TimedExponentialDOubleDiodeUpdater', "
                    "'float64_timed_overrelaxed'/'OverRelazedTimedExponentialDoubleDiodeUpdater'; "
                    f"got {double_diode_updater!r}"
                )
        elif non_linearity == "single_diode_exponential":
            if single_diode_updater == "custom":
                updater_cls = CustomExponentialSingleDiodeUpdater
            elif single_diode_updater == "standard":
                updater_cls = ExponentialSingleDiodeUpdater
            elif single_diode_updater in ("float32", "Float32ExponentialSingleDiodeUpdater"):
                updater_cls = Float32ExponentialSingleDiodeUpdater
            elif single_diode_updater in (
                "overrelaxed",
                "overrelated",
                "OverRelazedSingleDiodeExponentialUpdater",
            ):
                updaters = [
                    OverRelazedSingleDiodeExponentialUpdater(
                        layer,
                        fn,
                        scaled_exponential_params(layer),
                        overrelaxation_factor=overrelaxation_factor,
                    )
                    for layer in free_layers
                ]
                updater_cls = None
            else:
                raise ValueError(
                    "single_diode_updater must be 'custom', 'standard', 'float32'/'Float32ExponentialSingleDiodeUpdater', or "
                    "'overrelaxed'/'overrelated'/'OverRelazedSingleDiodeExponentialUpdater'; "
                    f"got {single_diode_updater!r}"
                )
            if updater_cls is not None:
                updaters = [
                    updater_cls(layer, fn, scaled_exponential_params(layer))
                    for layer in free_layers
                ]
        elif non_linearity == "hard_sigmoid":
            updaters = [
                CustomHardSigmoidUpdater(
                    layer, fn,
                    _scale_hard_sigmoid_params_for_layer(
                        hard_sigmoid_params, layer, voltage_amp, current_amp,
                    ),
                )
                for layer in free_layers
            ]
        elif non_linearity == "experimental":
            if iv_data is None:
                env_path = os.environ.get("LABS_IV_CURVE_PATH")
                candidate_path = (
                    env_path
                    if env_path
                    else (iv_data_path if iv_data_path else DEFAULT_IV_CURVE_PATH)
                )
                if candidate_path is None:
                    raise FileNotFoundError(
                        "Expected experimental IV curve path via LABS_IV_CURVE_PATH or config "
                        "'iv_data_path' (existing .npz file). Provided value: None."
                    )
                path = Path(candidate_path)
                if not path.exists():
                    raise FileNotFoundError(
                        "Expected experimental IV curve path via LABS_IV_CURVE_PATH or config "
                        f"'iv_data_path' (existing .npz file). Provided value: {path}"
                    )
                iv_data = _load_iv_data(path)
            if double_diode_updater in (
                "overrelaxed",
                "overrelated",
                "OverRelazedExperimentalIVcurveUpdater",
            ):
                updaters = [
                    OverRelazedExperimentalIVcurveUpdater(
                        layer,
                        fn,
                        iv_data,
                        overrelaxation_factor=overrelaxation_factor,
                        damping=damping,
                        max_newton_steps=experimental_newton_max_steps,
                        newton_tol=minimizer_settings.experimental_exponential_newton_tol_start,
                    )
                    for layer in free_layers
                ]
            else:
                updaters = [
                    ExperimentalIVcurveUpdater(
                        layer,
                        fn,
                        iv_data,
                        damping=damping,
                        max_newton_steps=experimental_newton_max_steps,
                        newton_tol=minimizer_settings.experimental_exponential_newton_tol_start,
                    )
                    for layer in free_layers
                ]
        elif non_linearity == "linear":
            updaters = [CustomQuadraticUpdater(layer, fn) for layer in free_layers]
        else:
            raise ValueError(
                "non_linearity must be 'perfect_diode', 'double_diode_quadratic', "
                "'lpw_diode', 'double_diode_exponential', 'hard_sigmoid', or 'linear'; got {}".format(
                    non_linearity
                )
            )

        super().__init__(
            fn,
            updaters,
            num_iterations,
            mode,
            voltage_amp,
            current_amp,
            adaptive_equilibrium=adaptive_equilibrium,
            settings=minimizer_settings,
        )

        for updater in self._updaters:
            updater.voltage_amp = self.voltage_amp
            updater.current_amp = self.current_amp

        self._non_linearity = non_linearity
        self._quadratic_params = quadratic_params
        self._exponential_params = exponential_params
        self._hard_sigmoid_params = hard_sigmoid_params


class CustomAnderssonMinimizer(CustomQuadraticMinimizer):
    """Quadratic minimizer with Anderson acceleration (async sweeps, no safeguarding)."""

    def __init__(
        self,
        fn,
        free_layers,
        num_iterations,
        mode,
        non_linearity,
        quadratic_diode_param,
        exponential_diode_param,
        voltage_amp,
        current_amp,
        hard_sigmoid_param,
        iv_data,
        iv_data_path,
        double_diode_updater,
        adaptive_equilibrium,
        overrelaxation_factor,
        single_diode_updater,
        *,
        minimizer_settings: MinimizerSettings,
        anderson_m: int = 8,
        anderson_omega: float = 1.0,
        anderson_tol_floor: float = 5e-3,
        anderson_reg: float = 1e-8,
        damping: float = 0.5,
        experimental_newton_max_steps: int = 100,
    ):
        if mode != "asynchronous":
            raise ValueError(f"Expected mode='asynchronous'; got {mode!r}")
        super().__init__(
            fn=fn,
            free_layers=free_layers,
            num_iterations=num_iterations,
            mode=mode,
            non_linearity=non_linearity,
            quadratic_diode_param=quadratic_diode_param,
            exponential_diode_param=exponential_diode_param,
            voltage_amp=voltage_amp,
            current_amp=current_amp,
            hard_sigmoid_param=hard_sigmoid_param,
            iv_data=iv_data,
            iv_data_path=iv_data_path,
            double_diode_updater=double_diode_updater,
            adaptive_equilibrium=adaptive_equilibrium,
            overrelaxation_factor=overrelaxation_factor,
            single_diode_updater=single_diode_updater,
            minimizer_settings=minimizer_settings,
            damping=damping,
            experimental_newton_max_steps=experimental_newton_max_steps,
        )

        self._aa_m = max(int(anderson_m), 0)
        self._aa_omega = float(anderson_omega)
        self._aa_tol_floor = float(anderson_tol_floor)
        self._aa_reg = float(anderson_reg)
        self._aa_hist_len_total = 0.0
        self._aa_hist_len_iters_total = 0
        self._aa_hist_len_last = 0.0
        self._aa_hist_len_max = 0.0
        self._aa_restart_total = 0
        self._aa_restart_iters_total = 0
        self._aa_restart_last = 0
        self._aa_restart_rate_last = 0.0
        self._aa_iterations_last = 0
        self._aa_m_eff_last = 0
        self._aa_f_norm_inf_last = None
        self._aa_f_norm_l2_last = None

    def _pack_state(self):
        states = []
        for updater in self._updaters:
            state = updater._layer.state
            states.append(state.reshape(state.shape[0], -1))
        return torch.cat(states, dim=1)

    def _unpack_state(self, packed):
        offset = 0
        for updater in self._updaters:
            state = updater._layer.state
            batch = state.shape[0]
            width = state.numel() // batch
            chunk = packed[:, offset:offset + width].reshape(state.shape)
            updater._layer.state = chunk
            offset += width

    def _plain_sweep(self, *, iteration: int):
        layer_group_odd = self._updaters[::2]
        layer_group_even = self._updaters[1::2]

        if layer_group_odd:
            self._step_group_with_reject_policy(layer_group_odd, label="odd", iteration=iteration)
        if layer_group_even:
            self._step_group_with_reject_policy(layer_group_even, label="even", iteration=iteration)

    def anderson_history_stats(self, reset: bool = False):
        total_iters = int(self._aa_hist_len_iters_total)
        avg_len = (self._aa_hist_len_total / total_iters) if total_iters > 0 else 0.0
        stats = {
            "avg_dF_len": float(avg_len),
            "last_avg_dF_len": float(self._aa_hist_len_last),
            "max_avg_dF_len": float(self._aa_hist_len_max),
            "total_iters": total_iters,
        }
        if reset:
            self._aa_hist_len_total = 0.0
            self._aa_hist_len_iters_total = 0
            self._aa_hist_len_last = 0.0
            self._aa_hist_len_max = 0.0
        return stats

    def anderson_run_stats(self):
        return {
            "f_norm_inf": None if self._aa_f_norm_inf_last is None else float(self._aa_f_norm_inf_last),
            "f_norm_l2": None if self._aa_f_norm_l2_last is None else float(self._aa_f_norm_l2_last),
            "m_eff": float(self._aa_hist_len_last),
            "m_eff_last": int(self._aa_m_eff_last),
            "restart_rate": float(self._aa_restart_rate_last),
            "restart_count": int(self._aa_restart_last),
            "iterations": int(self._aa_iterations_last),
        }

    def _safeguard_candidate(
        self,
        *,
        x_plain,
        corr,
        E_plain,
        max_trials: int = 3,
        shrink: float = 0.5,
        eps: float = 0.0,
    ):
        """Backtrack the Anderson correction until energy does not increase."""
        if isinstance(E_plain, torch.Tensor) and E_plain.numel() == x_plain.shape[0]:
            batch = x_plain.shape[0]
            theta = torch.ones(batch, device=x_plain.device, dtype=x_plain.dtype)
            accepted = torch.zeros(batch, device=x_plain.device, dtype=torch.bool)
            nonfinite = torch.zeros(batch, device=x_plain.device, dtype=torch.bool)
            backtracks = torch.zeros(batch, device=x_plain.device, dtype=torch.int64)
            x_best = x_plain
            E_best = E_plain
            for _ in range(max_trials):
                x_trial = x_plain - theta[:, None] * corr
                trial_finite = torch.isfinite(x_trial).all(dim=1)
                nonfinite |= ~trial_finite
                if trial_finite.any():
                    self._unpack_state(x_trial)
                    E_trial = self.eval_energy_batch()
                else:
                    E_trial = E_plain
                E_trial = E_trial.reshape(-1)
                energy_finite = torch.isfinite(E_trial)
                accept_mask = trial_finite & energy_finite & (E_trial <= E_plain + eps)
                new_accept = (~accepted) & accept_mask
                if new_accept.any():
                    x_best = torch.where(new_accept[:, None], x_trial, x_best)
                    E_best = torch.where(new_accept, E_trial, E_best)
                accepted |= new_accept
                still = (~accepted) & (~nonfinite)
                if not still.any():
                    break
                backtracks[still] += 1
                theta = torch.where(still, theta * shrink, theta)
            x_out = torch.where(accepted[:, None], x_best, x_plain)
            E_out = torch.where(accepted, E_best, E_plain)
            return x_out, E_out, theta, backtracks, accepted, nonfinite

        theta = 1.0
        for trial in range(max_trials):
            x_trial = x_plain - theta * corr
            if not torch.isfinite(x_trial).all():
                self._unpack_state(x_plain)
                return x_plain, E_plain, 0.0, trial, False, True
            self._unpack_state(x_trial)
            E_trial = self.eval_energy_sum()

            if torch.isfinite(E_trial) and E_trial <= E_plain + eps:
                return x_trial, E_trial, theta, trial, True, False
            theta *= shrink

        self._unpack_state(x_plain)
        return x_plain, E_plain, 0.0, max_trials, False, False

    def compute_equilibrium(self):
        max_num_of_iterations = len(self._list_layers) // 2
        self._stored_states = {}
        iterations_used = max_num_of_iterations

        if self._force_fixed_iterations:
            return super().compute_equilibrium()

        dX = deque(maxlen=self._aa_m)
        dF = deque(maxlen=self._aa_m)
        x_prev = None
        f_prev = None
        x = self._pack_state()
        prev_inf_delta = float("inf")
        safeguard_max_trials = 5
        safeguard_shrink = 0.5
        safeguard_eps = 0.0
        theta_restart_thresh = 0.25
        dF_len_sum = 0.0
        restart_count = 0
        for i in range(max_num_of_iterations):
            restart_history = False
            self._set_experimental_newton_tol(
                self._experimental_exponential_newton_tol_policy(prev_inf_delta)
            )
            self._plain_sweep(iteration=i)
            x_plain = self._pack_state()
            E_plain = self.eval_energy_batch()
            f = x_plain - x

            if x_prev is not None:
                dX.append(x - x_prev)
                dF.append(f - f_prev)

            x_candidate = x_plain
            corr = None
            theta = 1.0
            m = len(dF)
            dF_len_sum += float(m)


            if m >= 1 and self._aa_m > 0:
                dF_stack = torch.stack(list(dF), dim=0)
                dX_stack = torch.stack(list(dX), dim=0)
                batch = dF_stack.shape[1]

                A = torch.einsum("ibn,jbn->bij", dF_stack, dF_stack)
                b = torch.einsum("ibn,bn->bi", dF_stack, f)
                eye = torch.eye(m, device=A.device, dtype=A.dtype).unsqueeze(0)
                A = A + self._aa_reg * eye
                gamma_ok = torch.ones(batch, device=A.device, dtype=torch.bool)
                try:
                    gamma = torch.linalg.solve(A, b)
                except RuntimeError:
                    gamma_list = []
                    for idx in range(batch):
                        try:
                            gamma_i = torch.linalg.solve(A[idx], b[idx])
                        except RuntimeError:
                            gamma_i = torch.zeros_like(b[idx])
                            gamma_ok[idx] = False
                        gamma_list.append(gamma_i)
                    gamma = torch.stack(gamma_list, dim=0)
                gamma_finite = gamma_ok & torch.isfinite(gamma).all(dim=1)
                corr = torch.einsum("bi,ibn->bn", gamma, dX_stack)
                if not gamma_finite.all():
                    corr = torch.where(gamma_finite[:, None], corr, torch.zeros_like(corr))
                    restart_history = True

                x_candidate, _, theta, backtracks, accepted, nonfinite = self._safeguard_candidate(
                    x_plain=x_plain,
                    corr=corr,
                    E_plain=E_plain,
                    max_trials=safeguard_max_trials,
                    shrink=safeguard_shrink,
                    eps=safeguard_eps,
                )
                if isinstance(accepted, torch.Tensor):
                    if (not torch.all(accepted)) or torch.any(nonfinite):
                        restart_history = True
                    if isinstance(theta, torch.Tensor):
                        theta_min = float(theta.min().item())
                    else:
                        theta_min = float(theta)
                    if isinstance(backtracks, torch.Tensor):
                        backtracks_max = int(backtracks.max().item())
                    else:
                        backtracks_max = int(backtracks)
                else:
                    theta_min = float(theta)
                    backtracks_max = int(backtracks)
                    if not accepted or nonfinite:
                        restart_history = True

                if theta_min < theta_restart_thresh or backtracks_max >= 2:
                    restart_history = True


            if corr is not None:
                # Accept the safeguarded candidate directly (theta already applied).
                x_new = x_candidate
            else:
                x_new = x_plain
            dx_new = x_new - x

            max_abs_dx = dx_new.abs().amax(dim=1)
            max_abs_x = x_new.abs().amax(dim=1)
            denom = torch.maximum(
                max_abs_x,
                torch.full_like(max_abs_x, self._aa_tol_floor),
            )
            rel = max_abs_dx / denom
            prev_inf_delta = float(max_abs_dx.max().item())

            self._unpack_state(x_new)
            self._maybe_break_on_nonfinite_state(self._updaters, label="aa", iteration=i)

            if self._adaptive_equilibrium and torch.max(rel).item() < self._rel_tol:
                iterations_used = i + 1
                break

            if restart_history:
                restart_count += 1
                dX.clear()
                dF.clear()
                x_prev = x
                f_prev = f
            else:
                x_prev = x
                f_prev = f
            x = x_new

        self._equilibrium_iter_total += iterations_used
        self._equilibrium_iter_calls += 1
        self._equilibrium_iter_last = iterations_used
        self._equilibrium_iter_max = max_num_of_iterations
        self._equilibrium_sample_iters_last = None
        if iterations_used > 0:
            avg_len = dF_len_sum / float(iterations_used)
        else:
            avg_len = 0.0
        self._aa_hist_len_total += dF_len_sum
        self._aa_hist_len_iters_total += int(iterations_used)
        self._aa_hist_len_last = float(avg_len)
        self._aa_hist_len_max = max(self._aa_hist_len_max, float(avg_len))
        self._aa_restart_total += int(restart_count)
        self._aa_restart_iters_total += int(iterations_used)
        self._aa_restart_last = int(restart_count)
        self._aa_restart_rate_last = (float(restart_count) / float(iterations_used)) if iterations_used > 0 else 0.0
        self._aa_iterations_last = int(iterations_used)
        self._aa_m_eff_last = int(len(dF))

        x_final = self._pack_state()
        self._plain_sweep(iteration=iterations_used)
        x_plain_final = self._pack_state()
        f_final = x_plain_final - x_final
        self._aa_f_norm_inf_last = float(f_final.abs().amax().item())
        f_l2 = torch.linalg.vector_norm(f_final, ord=2, dim=1)
        self._aa_f_norm_l2_last = float(f_l2.max().item()) if f_l2.numel() else 0.0
        self._unpack_state(x_final)


        layers = {layer.name: layer.state for layer in self._layers}
        return layers
