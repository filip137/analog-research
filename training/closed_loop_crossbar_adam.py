"""Apparent-feedback P&V recovery; hidden device state stays in the plant.

The desired state accumulates Adam commands. Without that accumulator,
sub-tolerance commands would disappear. Verification observes the held
post-write state; an ordinary verify does not resample write noise.
"""
from __future__ import annotations

import math
import torch


class ClosedLoopAdam:
    def __init__(self, read_apparent, pulse, *, learning_rate, tolerance,
                 maximum_pulses=128, total_pulse_cap=640, epsilon=1e-8):
        if not all(math.isfinite(x) and x > 0 for x in (learning_rate, tolerance, epsilon)):
            raise ValueError('Expected positive finite learning rate, tolerance and epsilon.')
        if any(type(x) is not int or x < 1 for x in (maximum_pulses, total_pulse_cap)):
            raise ValueError('Expected positive integer pulse caps.')
        self.read_apparent, self.pulse = read_apparent, pulse
        self.learning_rate, self.tolerance = learning_rate, tolerance
        self.learning_rate_scale = 1.0
        self.maximum_pulses, self.total_pulse_cap = maximum_pulses, total_pulse_cap
        self.epsilon = epsilon
        self.target = read_apparent().detach().clone()
        self.first_moment = torch.zeros_like(self.target)
        self.second_moment = torch.zeros_like(self.target)
        self.pulse_count = torch.zeros_like(self.target, dtype=torch.int64)
        self.step_index = 0
        self.verify_reads = 0
        self.burst_exhaustions = 0
        self.cap_blocks = 0

    @torch.no_grad()
    def step(self, gradient):
        value = gradient.detach().reshape(-1)
        if value.shape != self.target.shape or not bool(torch.isfinite(value).all()):
            raise ValueError('Expected one finite gradient per device.')
        self.step_index += 1
        self.first_moment.mul_(0.9).add_(value, alpha=0.1)
        self.second_moment.mul_(0.999).addcmul_(value, value, value=0.001)
        first = self.first_moment / (1 - 0.9 ** self.step_index)
        second = self.second_moment / (1 - 0.999 ** self.step_index)
        command = -self.learning_rate * first / (second.sqrt() + self.epsilon)
        if not math.isfinite(self.learning_rate_scale) or not 0 < self.learning_rate_scale <= 1:
            raise ValueError('Expected a finite learning-rate scale in (0,1].')
        command = command * self.learning_rate_scale
        self.target.add_(command)
        # No per-cell bounds or fault mask enter any controller decision.
        error = self.target - self.read_apparent()
        self.verify_reads += self.target.numel()
        pulses = 0
        rounds = 0
        for _ in range(self.maximum_pulses):
            needed = error.abs() > self.tolerance
            active = needed & (self.pulse_count < self.total_pulse_cap)
            count = int(active.sum().item())
            if not count:
                break
            direction = torch.sign(error).to(torch.int8) * active.to(torch.int8)
            self.pulse(direction)
            self.pulse_count.add_(active)
            pulses += count
            rounds += 1
            self.verify_reads += count
            error = self.target - self.read_apparent()
        unresolved = error.abs() > self.tolerance
        blocked = unresolved & (self.pulse_count >= self.total_pulse_cap)
        self.cap_blocks += int(blocked.sum().item())
        if rounds == self.maximum_pulses:
            self.burst_exhaustions += int((unresolved & ~blocked).sum().item())
        return {'pulses': pulses, 'verify_rounds': rounds,
                'unresolved_cells': int(unresolved.sum().item()),
                'blocked_at_cap': int(blocked.sum().item())}

    def state_dict(self):
        return {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for key, value in vars(self).items() if key not in {'read_apparent', 'pulse'}}
