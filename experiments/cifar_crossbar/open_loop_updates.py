"""One-pulse open-loop updates without a cumulative pulse cap.

Retain the historical clipped Bernoulli rule; remove only the total pulse cap.
Gradients/Adam are digital. The controller never reads the device state or
inspects device identities to decide how many pulses to apply.
"""
import math
import torch
from experiments.cifar_crossbar.devices import clone_cpu


def nominal_reset_counts(target, nominal_step):
    """Invert q(n)=1-2*(1-dw_min)**n, from nominal RESET q(0)=-1.

    This uses only the public nominal OM response, not deployment identities
    or reads. The q=1 asymptote is represented by the closest interior FP32
    target; no pulse budget truncates the derived integer schedule.
    """
    if not 0 < nominal_step < 1 or target.dtype != torch.float32:
        raise ValueError("Expected FP32 targets and a nominal soft-bounds step in (0,1).")
    if not bool(torch.isfinite(target).all()) or bool((target.abs() > 1).any()):
        raise ValueError("Expected normalized targets in [-1,1].")
    upper = torch.nextafter(torch.ones_like(target), torch.zeros_like(target))
    finite = torch.minimum(target, upper).double()
    real = torch.log((1 - finite) / 2) / math.log1p(-nominal_step)
    lower, higher = real.floor(), real.ceil()
    endpoint = lambda n: 1 - 2 * torch.exp(n * math.log1p(-nominal_step))
    use_higher = (endpoint(higher) - target.double()).abs() < (endpoint(lower) - target.double()).abs()
    return torch.where(use_higher, higher, lower).to(torch.int64)


@torch.no_grad()
def apply_pulse_counts(port, direction, counts):
    """Apply every requested pulse; physical saturation remains in the plant."""
    if counts.shape != direction.shape or counts.numel() != port.size:
        raise ValueError("Pulse schedule does not match the device layout.")
    if counts.dtype != torch.int64 or bool((counts < 0).any()):
        raise ValueError("Expected nonnegative integer pulse counts.")
    if bool(((direction != -1) & (direction != 0) & (direction != 1)).any()):
        raise ValueError("Expected signed unit pulse directions.")
    if bool(((counts > 0) & (direction == 0)).any()):
        raise ValueError("Nonzero pulse count requires a direction.")
    for index in range(int(counts.max())):
        port.pulse(direction.to(torch.int8) * (counts > index))
    port.finish_step()


class UncappedOpenLoopAdam:
    def __init__(self, port, template, *, learning_rate, seed):
        if template.numel() != port.size or not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("Invalid open-loop writer shape or learning rate.")
        self.port, self.learning_rate = port, float(learning_rate)
        self.first = torch.zeros_like(template)
        self.second = torch.zeros_like(template)
        self.count = torch.zeros_like(template, dtype=torch.int64)
        self.step_index, self.verify_reads, self.probability_clipped = 0, 0, 0
        self.generator = torch.Generator(device=template.device).manual_seed(seed)

    @torch.no_grad()
    def step(self, gradient):
        if gradient.shape != self.first.shape or not bool(torch.isfinite(gradient).all()):
            raise ValueError("Expected a finite gradient for every logical weight.")
        self.step_index += 1
        self.first.mul_(0.9).add_(gradient, alpha=0.1)
        self.second.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
        command = -self.learning_rate * (self.first / (1 - 0.9**self.step_index)) / (
            (self.second / (1 - 0.999**self.step_index)).sqrt() + 1e-8)
        if self.learning_rate == 0:
            return
        probability = command.abs() / self.port.nominal_step
        self.probability_clipped += int((probability > 1).sum())
        selected = torch.rand(probability.shape, device=probability.device,
                              generator=self.generator) < probability.clamp_max(1)
        self.port.pulse(command.sign().to(torch.int8) * selected)
        self.count += selected
        self.port.finish_step()

    def state_dict(self):
        return clone_cpu({"contract": ("open_loop_one_pulse_no_total_cap.v1", self.learning_rate, self.port.nominal_step),
                          "first": self.first, "second": self.second, "count": self.count,
                          "step_index": self.step_index, "verify_reads": self.verify_reads,
                          "probability_clipped": self.probability_clipped,
                          "rng": self.generator.get_state()})

    def load_state_dict(self, state):
        if state["contract"] != self.state_dict()["contract"] or state["verify_reads"] != 0:
            raise ValueError("Open-loop update contract changed.")
        for name in ("first", "second", "count"):
            value = state[name]
            if value.shape != self.first.shape:
                raise ValueError("Open-loop checkpoint shape changed.")
            setattr(self, name, value.to(self.first.device).clone())
        self.step_index = state["step_index"]
        self.probability_clipped = state["probability_clipped"]
        self.generator.set_state(state["rng"].cpu())
