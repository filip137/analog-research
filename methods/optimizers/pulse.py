"""methods.optimizers.pulse — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from core.tensor_io import clone_cpu


class PulseWriter:
    """Existing stochastic open-loop or accumulated-target closed-loop Adam.

    Uses no hidden population values. One Adam command per logical crosspoint;
    the backend implements the physical encoding. State includes its own RNG.
    """

    def __init__(
        self,
        port,
        *,
        method,
        learning_rate,
        tolerance,
        cap=640,
        maximum_pulses=128,
        seed=0,
    ):
        self.port, self.method = port, method
        self.learning_rate, self.tolerance, self.cap, self.maximum_pulses = (
            learning_rate,
            tolerance,
            cap,
            maximum_pulses,
        )
        self.target = port.read()
        self.first = torch.zeros_like(self.target)
        self.second = torch.zeros_like(self.target)
        self.count = torch.zeros_like(self.target, dtype=torch.int64)
        self.generator = torch.Generator(device=self.target.device).manual_seed(seed)
        self.step_index, self.verify_reads, self.probability_clipped = 0, 0, 0

    @torch.no_grad()
    def step(self, gradient):
        if gradient.shape != self.target.shape or not bool(
            torch.isfinite(gradient).all()
        ):
            raise ValueError(
                "Expected a finite normalized gradient for every crosspoint."
            )
        self.step_index += 1
        self.first.mul_(0.9).add_(gradient, alpha=0.1)
        self.second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
        command = (
            -self.learning_rate
            * (self.first / (1 - 0.9**self.step_index))
            / ((self.second / (1 - 0.999**self.step_index)).sqrt() + 1e-8)
        )
        # A zero-learning-rate control must not trigger refresh or verify writes.
        if self.learning_rate == 0:
            return
        if self.method == "open_loop":
            probability = command.abs() / self.port.nominal_step
            self.probability_clipped += int((probability > 1).sum())
            selected = (
                torch.rand(
                    probability.shape,
                    device=probability.device,
                    generator=self.generator,
                )
                < probability.clamp_max(1)
            ) & (self.count < self.cap)
            self.port.pulse(command.sign().to(torch.int8) * selected)
            self.count += selected
        elif self.method == "closed_loop":
            self.target.add_(command)
            for _ in range(self.maximum_pulses):
                error = self.target - self.port.read()
                self.verify_reads += self.port.size
                active = (error.abs() > self.tolerance) & (self.count < self.cap)
                if not bool(active.any()):
                    break
                self.port.pulse(error.sign().to(torch.int8) * active)
                self.count += active
        else:
            raise ValueError("Expected open_loop or closed_loop.")
        self.port.finish_step()

    def state_dict(self):
        return clone_cpu(
            {
                "contract": (
                    self.method,
                    self.learning_rate,
                    self.tolerance,
                    self.cap,
                    self.maximum_pulses,
                ),
                "target": self.target,
                "first": self.first,
                "second": self.second,
                "count": self.count,
                "step_index": self.step_index,
                "verify_reads": self.verify_reads,
                "probability_clipped": self.probability_clipped,
                "rng": self.generator.get_state(),
            }
        )

    def load_state_dict(self, state):
        if state["contract"] != self.state_dict()["contract"]:
            raise ValueError("Expected the same writer contract.")
        for key in ("target", "first", "second", "count"):
            setattr(self, key, state[key].to(self.target.device).clone())
        for key in ("step_index", "verify_reads", "probability_clipped"):
            setattr(self, key, state[key])
        self.generator.set_state(state["rng"].cpu())
