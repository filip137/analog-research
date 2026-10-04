"""Historical accumulated-target OM Adam with only the total cap removed."""
import torch
from experiments.cifar_crossbar.devices import PulseWriter


class UncappedClosedLoopAdam(PulseWriter):
    def __init__(self, port, template, *, learning_rate, seed=0):
        super().__init__(port, method='closed_loop', learning_rate=learning_rate,
                         tolerance=port.nominal_step * .5, cap=None, maximum_pulses=1, seed=seed)
        if self.target.shape != template.shape: raise ValueError('Array/target layout mismatch.')

    @torch.no_grad()
    def step(self, gradient):
        if gradient.shape != self.target.shape or not bool(torch.isfinite(gradient).all()):
            raise ValueError('Expected finite normalized gradients for every cell.')
        self.step_index += 1
        self.first.mul_(.9).add_(gradient, alpha=.1)
        self.second.mul_(.999).addcmul_(gradient, gradient, value=.001)
        command = (-self.learning_rate * (self.first / (1 - .9**self.step_index)) /
                   ((self.second / (1 - .999**self.step_index)).sqrt() + 1e-8))
        if self.learning_rate == 0: return
        self.target.add_(command)
        error = self.target - self.port.read()
        self.verify_reads += self.port.size
        active = error.abs() > self.tolerance
        if bool(active.any()):
            self.port.pulse(error.sign().to(torch.int8) * active)
            self.count += active
        self.port.finish_step()
