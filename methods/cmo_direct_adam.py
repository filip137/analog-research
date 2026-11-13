"""Direct Adam commands through a bounded apparent-state write port."""

import math

import torch


class DirectWriteAdam:
    """Adam in normalized conductance coordinates, without a command buffer."""

    state_names = ('first', 'second', 'attempt_counts')

    def contract(self):
        # Constant-LR contracts stay identical to checkpoints written before schedules existed.
        return ['cmo_drn.direct_adam.v1', self.learning_rate, *([] if self.schedule is None else [self.schedule])]

    def __init__(self, port, *, learning_rate, schedule=None):
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError('Expected a finite nonnegative direct Adam learning rate.')
        if schedule is not None:
            kind, warmup, total = schedule
            if (kind not in ('warmup_cosine', 'cosine_to_zero') or type(warmup) is not int or type(total) is not int
                    or not 0 <= warmup < total or (kind == 'cosine_to_zero' and warmup != 0)):
                raise ValueError('Expected a warmup_cosine schedule with 0 <= warmup < total updates '
                                 'or a cosine_to_zero schedule without warm-up.')
            schedule = [kind, warmup, total]
        self.port = port
        self.learning_rate = float(learning_rate)
        self.schedule = schedule
        template = port.read()
        self.first = torch.zeros_like(template)
        self.second = torch.zeros_like(template)
        self.attempt_counts = torch.zeros_like(template, dtype=torch.int64)
        self.step_index = 0

    def rate(self, step):
        """Learning rate of 1-based update ``step``: linear warm-up, then cosine to zero."""
        if self.schedule is None:
            return self.learning_rate
        kind, warmup, total = self.schedule
        if kind == 'cosine_to_zero':
            # torch CosineAnnealingLR(T_max=total, eta_min=0): update k (1-based) uses epoch k-1.
            return self.learning_rate * 0.5 * (1 + math.cos(math.pi * (step - 1) / total))
        if step <= warmup:
            return self.learning_rate * step / warmup
        return self.learning_rate * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))

    @torch.no_grad()
    def adam_command(self, gradient):
        if gradient.shape != self.first.shape or not bool(torch.isfinite(gradient).all()):
            raise ValueError('Expected a finite normalized gradient for every DRN device.')
        self.step_index += 1
        self.first.mul_(0.9).add_(gradient, alpha=0.1)
        self.second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
        command = (-self.rate(self.step_index) * (self.first / (1 - 0.9 ** self.step_index))
                   / ((self.second / (1 - 0.999 ** self.step_index)).sqrt() + 1e-8))
        if not bool(torch.isfinite(command).all()):
            raise FloatingPointError('Nonfinite direct physical write command.')
        return command

    @torch.no_grad()
    def step(self, gradient):
        command = self.adam_command(gradient)
        # The returned mask describes public target projection, never device success.
        attempted = self.port.write(command)
        if attempted.shape != command.shape or attempted.dtype != torch.bool:
            raise ValueError('Expected the bounded write port to return its attempt mask.')
        self.attempt_counts.add_(attempted)

    def diagnostics(self):
        return {'update_rule': 'direct_adam', 'issued_write_attempts': int(self.attempt_counts.sum()),
                'max_device_attempts': int(self.attempt_counts.max()), 'max_write_attempts': None,
                'capped_devices': 0, **({} if self.schedule is None else
                                        {'schedule': self.schedule, 'current_rate': self.rate(self.step_index)})}

    def state_dict(self):
        return {'contract': self.contract(),
                'step_index': self.step_index,
                **{k: getattr(self, k).detach().cpu().clone()
                   for k in self.state_names}}

    def load_state_dict(self, state):
        if state.get('contract') != self.contract():
            raise ValueError('Direct physical writer continuation contract differs.')
        step = state.get('step_index')
        if type(step) is not int or step < 0:
            raise ValueError('Invalid direct writer step.')
        for key in self.state_names:
            current, value = getattr(self, key), state[key]
            if value.shape != current.shape or value.dtype != current.dtype:
                raise ValueError(f'Writer state shape/type mismatch: {key}.')
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite direct writer continuation state.')
            current.copy_(value.to(current.device))
        if bool((self.second < 0).any()) or bool(((self.attempt_counts < 0) | (self.attempt_counts > step)).any()):
            raise ValueError('Invalid direct writer moments or attempt counts.')
        self.step_index = step


class ThresholdedBoundedWriteAdam(DirectWriteAdam):
    """Accumulate signed commands and gate against target-conditioned write rho.

    Rho is evaluated before programming at the clipped target. The gate applies
    to the accumulated command; projection may reduce a boundary write. Issued
    commands are consumed even for stuck cells or projected no-ops. Unissued
    residuals are retained and never flushed at the final step.
    """

    state_names = (*DirectWriteAdam.state_names, 'pending')

    def __init__(self, port, *, learning_rate, threshold_sigma=2.0, schedule=None):
        super().__init__(port, learning_rate=learning_rate, schedule=schedule)
        if threshold_sigma != 2.0:
            raise ValueError('Expected the declared 2-rho noisy write threshold.')
        self.threshold_sigma = float(threshold_sigma)
        self.pending = torch.zeros_like(self.first)

    def contract(self):
        # Constant-LR contracts stay identical to checkpoints written before schedules existed.
        return ['cmo_drn.bounded_thresholded_adam.v1', self.learning_rate,
                self.threshold_sigma, 'bounded_target_write_std', 'command_before_projection', 'ge',
                *([] if self.schedule is None else [self.schedule])]

    @torch.no_grad()
    def step(self, gradient):
        self.pending.add_(self.adam_command(gradient))
        if not bool(torch.isfinite(self.pending).all()):
            raise FloatingPointError('Nonfinite accumulated physical write command.')
        rho = self.port.write_noise_std_q(self.pending)
        if rho.shape != self.pending.shape or not bool(torch.isfinite(rho).all() and (rho > 0).all()):
            raise ValueError('Expected positive finite per-device pre-write noise standard deviations.')
        requested = self.pending.abs() >= self.threshold_sigma * rho
        issued = torch.where(requested, self.pending, 0.)
        attempted = self.port.write(issued)
        self.attempt_counts.add_(attempted)
        self.pending.sub_(issued)

    def diagnostics(self):
        return {**super().diagnostics(), 'update_rule':'bounded_thresholded_adam',
                'threshold_reference':'bounded_target_write_std', 'threshold_sigma':self.threshold_sigma,
                'threshold_comparison':'ge', 'threshold_applied_to':'command_before_projection',
                'pending_nonzero':int((self.pending != 0).sum()),
                'pending_rms_q':float(self.pending.square().mean().sqrt())}


class CappedNominalWriteAdam(DirectWriteAdam):
    """Accumulate signed commands; each device may be issued at most ``cap`` times.

    The gate is the crossbar's fixed nominal-midrange write standard deviation.
    It never scales with the write-noise factor, so noisy and noise-free arms
    share it exactly. An issued command consumes the device's allowance even for
    stuck cells or projected no-ops; spent devices accumulate no further
    commands. Unissued residuals are retained and never flushed.
    """

    state_names = (*DirectWriteAdam.state_names, 'pending', 'issued_counts')

    def __init__(self, port, *, learning_rate, threshold_sigma=2.0, cap=1):
        super().__init__(port, learning_rate=learning_rate)
        if threshold_sigma != 2.0 or cap not in (1, 4):
            raise ValueError('Expected the declared nominal 2-sigma gate with one or four attempts per device.')
        self.threshold_sigma = float(threshold_sigma)
        self.cap = cap
        self.threshold_q = self.threshold_sigma * float(port.nominal_write_std_q())
        if not math.isfinite(self.threshold_q) or self.threshold_q <= 0:
            raise ValueError('Expected a positive finite nominal write threshold.')
        self.pending = torch.zeros_like(self.first)
        self.issued_counts = torch.zeros_like(self.first, dtype=torch.int8)

    def contract(self):
        return ['cmo_drn.bounded_capped_nominal_adam.v1', self.learning_rate, self.threshold_sigma,
                self.cap, self.threshold_q, 'nominal_midrange_write_std', 'command_before_projection', 'gt']

    @torch.no_grad()
    def step(self, gradient):
        command = self.adam_command(gradient)
        eligible = self.issued_counts < self.cap
        self.pending.add_(torch.where(eligible, command, 0.))
        if not bool(torch.isfinite(self.pending).all()):
            raise FloatingPointError('Nonfinite accumulated physical write command.')
        requested = (self.pending.abs() > self.threshold_q) & eligible
        issued = torch.where(requested, self.pending, 0.)
        attempted = self.port.write(issued)
        self.attempt_counts.add_(attempted)
        self.issued_counts.add_(requested)
        self.pending.sub_(issued)

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if (bool(((self.issued_counts < 0) | (self.issued_counts > self.cap)).any())
                or bool((self.attempt_counts > self.issued_counts).any())
                or bool((self.pending[self.issued_counts >= self.cap] != 0).any())):
            raise ValueError('Invalid capped writer allowance state.')

    def diagnostics(self):
        return {**super().diagnostics(), 'update_rule':'bounded_capped_nominal_adam',
                'threshold_reference':'nominal_midrange_write_std', 'threshold_sigma':self.threshold_sigma,
                'threshold_q':self.threshold_q, 'threshold_comparison':'gt',
                'threshold_applied_to':'command_before_projection', 'max_write_attempts':self.cap,
                'issued_commands':int(self.issued_counts.sum()),
                'capped_devices':int((self.issued_counts >= self.cap).sum()),
                'pending_nonzero':int((self.pending != 0).sum()),
                'pending_rms_q':float(self.pending.square().mean().sqrt())}
