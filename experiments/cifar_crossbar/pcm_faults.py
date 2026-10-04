"""Permanent PCM failures and Gaussian programming endpoints, without pulses.

Failure definitions follow Li et al., APL Machine Learning 1, 016104 (2023),
doi:10.1063/5.0131797, Methods/Inference. Probability is per physical device
in the differential pair, so either bank or both banks may fail.
"""

from dataclasses import dataclass
import torch

from experiments.cifar_crossbar.devices import clone_cpu
from experiments.cifar_crossbar.model import state_hash


@dataclass
class PermanentFaults:
    mask: torch.Tensor
    conductance: torch.Tensor
    kind: str
    rate: float
    seed: int

    @classmethod
    def sample(cls, size, kind, rate, seed, device="cpu"):
        if kind not in ("open", "gmax", "random") or not 0 <= rate <= 1:
            raise ValueError(
                "Expected open/gmax/random failures and probability in [0, 1]."
            )
        # CPU draws make physical identities portable across execution devices.
        rng = torch.Generator().manual_seed(seed)
        mask = torch.rand((2, size), generator=rng) < rate
        values = torch.rand((2, size), generator=rng) * 25.0
        if kind == "open":
            values.zero_()
        elif kind == "gmax":
            values.fill_(25.0)
        return cls(mask.to(device), values.to(device), kind, rate, seed)

    def apply(self, conductance):
        return torch.where(self.mask, self.conductance, conductance)

    def state_dict(self):
        return clone_cpu(vars(self))

    def receipt(self):
        physical = self.mask.numel()
        return {
            "kind": self.kind,
            "requested_rate": self.rate,
            "seed": self.seed,
            "physical_devices": physical,
            "failed_devices": int(self.mask.sum()),
            "realized_rate": float(self.mask.sum()) / physical,
            "affected_weights": int(self.mask.any(0).sum()),
            "both_banks_failed": int(self.mask.all(0).sum()),
            "identity_sha256": state_hash(
                {"mask": self.mask, "stuck_conductance": self.conductance * self.mask}
            ),
            "placement": "independent Bernoulli per physical bank; fixed throughout recovery",
        }


class GaussianEndpointArray:
    """Reprogram requested targets once; failures cannot be repaired by writes.

    This is a complete-reprogramming abstraction, not incremental analog SGD.
    The controller uses read()/program(); the fault map is only used internally
    and for provenance. No physical pulse/verify/energy estimate is supplied.
    """

    def __init__(self, target, faults, seed, noise_scale=1.0):
        self.faults = faults
        self.generator = torch.Generator(device=target.device).manual_seed(seed)
        self.noise_scale = noise_scale
        self.program_calls = 0
        self.program(target)

    @torch.no_grad()
    def program(self, target):
        if not bool(torch.isfinite(target).all()) or bool((target.abs() > 1).any()):
            raise ValueError("Expected finite normalized target weights in [-1, 1].")
        conductance = torch.stack((target.clamp_min(0), (-target).clamp_min(0))) * 25.0
        relative = conductance / 25.0
        sigma = 0.26348 + 1.965 * relative - 1.1731 * relative.square()
        noise = torch.randn(
            conductance.shape, device=target.device, generator=self.generator
        )
        self.g = self.faults.apply(
            (conductance + self.noise_scale * sigma * noise).clamp_min(0)
        )
        self.target = target.detach().clone()
        self.program_calls += 1

    def read(self):
        return ((self.g[0] - self.g[1]) / 25.0).detach().clone()

    def state_dict(self):
        return clone_cpu(
            {
                "g": self.g,
                "target": self.target,
                "faults": self.faults.state_dict(),
                "rng": self.generator.get_state(),
                "noise_scale": self.noise_scale,
                "program_calls": self.program_calls,
            }
        )

    def load_state_dict(self, state):
        device = self.g.device
        if state_hash(
            {"mask": state["faults"]["mask"], "g": state["faults"]["conductance"]}
        ) != state_hash({"mask": self.faults.mask, "g": self.faults.conductance}):
            raise ValueError("Expected the same permanent physical fault identities.")
        self.g, self.target = (
            state["g"].to(device).clone(),
            state["target"].to(device).clone(),
        )
        self.generator.set_state(state["rng"].cpu())
        self.noise_scale, self.program_calls = (
            state["noise_scale"],
            state["program_calls"],
        )

    def verify_permanence(self):
        if not torch.equal(
            self.g[self.faults.mask], self.faults.conductance[self.faults.mask]
        ):
            raise RuntimeError("A permanent device failure changed during recovery.")

    def cost(self):
        size = self.target.numel()
        failed = int(self.faults.mask.sum())
        return {
            "array_reprogram_calls": self.program_calls,
            "logical_target_requests": self.program_calls * size,
            "healthy_device_programs": self.program_calls * (2 * size - failed),
            "failed_device_write_attempts": self.program_calls * failed,
        }
