"""Physical plants. Controllers receive only a narrow observable port.

OM reuses the validated per-cell counter RNG and pulse engine. PCM uses literal
AIHWKit-sampled cells and the ExpStep SET/reset equations. Its explicitly
declared differential refresh uses the native 0.75/0.25 thresholds followed by
bounded apparent-feedback reprogramming; it is not native PCM SGD/TT training.
"""

from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import json
import torch
from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_om_standard_crossbar import (
    IbmOmEffectiveCrossbarPlant,
    _counter_keyed_standard_normal,
)
from experiments.cifar_crossbar.model import state_hash


def clone_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_cpu(v) for v in value)
    return deepcopy(value)


def prepare_population(
    raw, *, noise_scale=1.0, fault_rate=0.0, fault_kind="low", fault_seed=0
):
    p = clone_cpu(raw)
    p["noise_scale"], p["fault_rate"], p["fault_kind"], p["fault_seed"] = (
        noise_scale,
        fault_rate,
        fault_kind,
        fault_seed,
    )
    h = p["hidden"]
    generator = torch.Generator().manual_seed(fault_seed)
    mask = torch.zeros_like(h["max_bound"], dtype=torch.bool)
    n = int(round(mask.numel() * fault_rate))
    if n:
        mask.flatten()[torch.randperm(mask.numel(), generator=generator)[:n]] = True
        fraction = {"low": 0.0, "middle": 0.5, "high": 1.0}[fault_kind]
        stuck = h["min_bound"] + (h["max_bound"] - h["min_bound"]) * fraction
        for key in ("min_bound", "max_bound"):
            h[key][mask] = stuck[mask]
        for key in ("dwmin_up", "dwmin_down"):
            h[key][mask] = 0
        h["corrupt"] |= mask
    h["injected_faults"] = mask
    p["fingerprint"] = population_fingerprint(p)
    return p


def population_fingerprint(population):
    return sha256(
        (
            state_hash(population["hidden"])
            + json.dumps(
                {
                    k: v
                    for k, v in population.items()
                    if k not in ("hidden", "fingerprint")
                },
                sort_keys=True,
            )
        ).encode()
    ).hexdigest()


class ObservablePort:
    __slots__ = ("__read", "__pulse", "__finish", "size", "nominal_step")

    def __init__(self, read, pulse, finish, size, nominal_step):
        self.__read, self.__pulse, self.__finish = read, pulse, finish
        self.size, self.nominal_step = size, nominal_step

    def read(self):
        return self.__read().detach().clone()

    def pulse(self, direction):
        self.__pulse(direction)

    def finish_step(self):
        self.__finish()


class OmPlant:
    def __init__(self, population, seed, device):
        self.source = clone_cpu(population)
        h, c = population["hidden"], population["constants"]
        flat = lambda key: h[key].flatten().to(device)
        corrupt = flat("corrupt")
        p = IbmReramArrayPopulation(
            assignment_seed=population["seed"],
            corruption_policy="published",
            binding_keys=("suffix",),
            binding_shapes=((population["size"], 1),),
            binding_sampling_seeds=(population["seed"],),
            donor_sampling_seeds=(),
            nominal_dw_min=c["dw_min"],
            dw_min_std=c["dw_min_std"],
            write_noise_std=c["write_noise_std"] * population["noise_scale"],
            max_bound=flat("max_bound"),
            min_bound=flat("min_bound"),
            dwmin_up=flat("dwmin_up"),
            dwmin_down=flat("dwmin_down"),
            reference=flat("reference"),
            corrupt=corrupt,
            published_corrupt=corrupt.clone(),
            fingerprint=population["fingerprint"],
            aihwkit_version="1.1.0",
        )
        if "trajectory_seed_blocks" in population:
            seeds = torch.cat([
                torch.arange(size, dtype=torch.int64) + int(block_seed) * 1000003 + 1
                for size, block_seed in population["trajectory_seed_blocks"]
            ])
            if seeds.numel() != p.size:
                raise ValueError("Trajectory seed blocks do not cover the population.")
        else:
            seeds = torch.arange(p.size, dtype=torch.int64) + int(seed) * 1000003 + 1
        self.engine = IbmOmEffectiveCrossbarPlant(
            p, trajectory_seeds=seeds, device=device
        )
        self.size = p.size
        self.nominal_step = c["dw_min"]

    def read(self):
        return self.engine.apparent

    def pulse(self, direction):
        self.engine.pulse(direction)

    def finish_step(self):
        pass

    def port(self):
        return ObservablePort(
            self.read, self.pulse, self.finish_step, self.size, self.nominal_step
        )

    def state_dict(self):
        return {
            "kind": "om",
            "population": clone_cpu(self.source),
            "engine": self.engine.state_dict(),
        }

    def load_state_dict(self, state):
        if state["population"]["fingerprint"] != self.source["fingerprint"]:
            raise ValueError("Expected the same OM population.")
        self.engine.load_state_dict(state["engine"])

    def cost(self):
        pulses = self.engine.upward_pulses + self.engine.downward_pulses
        return {
            "set_pulses": int(self.engine.upward_pulses.sum()),
            "reset_pulses": int(self.engine.downward_pulses.sum()),
            "physical_pulses": int(pulses.sum()),
            "max_cell_pulses": int(pulses.max()),
            "refreshes": 0,
            "refresh_reads": 0,
        }


class PcmPlant:
    def __init__(self, population, seed, device):
        self.source = clone_cpu(population)
        self.h = {k: v.to(device) for k, v in population["hidden"].items()}
        self.c = population["constants"]
        self.size = population["size"]
        self.device = self.h["max_bound"].device
        self.keys = (
            torch.arange(2 * self.size, dtype=torch.int64, device=device)
            + int(seed) * 1000003
            + 1
        ).reshape(2, -1)
        self.draws = torch.zeros_like(self.keys)
        self.g = torch.zeros((2, self.size), device=device)
        self.pulses = torch.zeros_like(self.keys)
        self.resets = torch.zeros_like(self.keys)
        self.refreshes, self.refresh_reads = 0, 0
        self.nominal_step = self.c["dw_min"] / self.c["w_max"]
        self.recovery_baseline = None
        self.recovery_cap = None
        self.reset(torch.ones_like(self.g, dtype=torch.bool))

    def _normal(self, mask):
        result = torch.zeros_like(self.g)
        result[mask] = _counter_keyed_standard_normal(self.keys[mask], self.draws[mask])
        self.draws[mask] += 1
        return result

    def reset(self, mask):
        if self.recovery_cap is not None:
            mask = mask & (
                self.pulses + self.resets - self.recovery_baseline < self.recovery_cap
            )
        value = self.h["reset_bias"] + self.c["reset_std"] * self._normal(mask)
        self.g[mask] = value.clamp(self.h["min_bound"], self.h["max_bound"])[mask]
        self.resets += mask

    def read(self):
        return (self.g[0] - self.g[1]) / self.c["w_max"]

    def pulse(self, direction):
        if direction.shape != (self.size,) or bool(
            ((direction < -1) | (direction > 1)).any()
        ):
            raise ValueError("Expected one {-1,0,1} direction per PCM pair.")
        mask = torch.stack((direction > 0, direction < 0))
        if self.recovery_cap is not None:
            mask &= (
                self.pulses + self.resets - self.recovery_baseline < self.recovery_cap
            )
        width = self.h["max_bound"] - self.h["min_bound"]
        z = 2 * self.g / width.clamp_min(1e-30) * self.c["a"] + self.c["b"]
        dw = (1 - self.c["A_up"] * torch.exp(self.c["gamma_up"] * z)).clamp_min(
            0
        ) * self.h["dwmin_up"]
        std = (
            self.c["dw_min_std"]
            * self.source["noise_scale"]
            * (
                dw.abs()
                + self.c["dw_min_std_add"]
                + self.c["dw_min_std_slope"] * self.g.abs()
            )
        )
        candidate = (self.g + dw + std * self._normal(mask)).clamp(
            self.h["min_bound"], self.h["max_bound"]
        )
        self.g[mask] = candidate[mask]
        self.pulses += mask

    def finish_step(self):
        # Each compound update checks both banks. Reset/rewrite costs are real.
        self.refresh_reads += 2 * self.size
        normalized = self.g / self.c["w_max"]
        needed = (normalized.max(0).values > 0.75) & (normalized.min(0).values > 0.25)
        if self.recovery_cap is not None:
            # Reserve at least one SET after each RESET; both count as writes.
            needed &= (
                (self.pulses + self.resets - self.recovery_baseline)
                < self.recovery_cap - 1
            ).all(0)
        if not bool(needed.any()):
            return
        target = self.read().clone()
        self.refreshes += int(needed.sum())
        self.reset(needed.unsqueeze(0).expand(2, -1))
        for _ in range(128):
            error = target - self.read()
            active = needed & (error.abs() > self.nominal_step * 0.5)
            self.refresh_reads += int(needed.sum())
            if not bool(active.any()):
                break
            self.pulse(error.sign().to(torch.int8) * active)

    def begin_recovery(self, cap):
        self.recovery_baseline = (self.pulses + self.resets).clone()
        self.recovery_cap = cap

    def port(self):
        return ObservablePort(
            self.read, self.pulse, self.finish_step, self.size, self.nominal_step
        )

    def state_dict(self):
        return clone_cpu(
            {
                "kind": "pcm",
                "population": self.source,
                "g": self.g,
                "keys": self.keys,
                "draws": self.draws,
                "pulses": self.pulses,
                "resets": self.resets,
                "refreshes": self.refreshes,
                "refresh_reads": self.refresh_reads,
                "recovery_baseline": self.recovery_baseline,
                "recovery_cap": self.recovery_cap,
            }
        )

    def load_state_dict(self, state):
        if state["population"]["fingerprint"] != self.source["fingerprint"]:
            raise ValueError("Expected the same PCM population.")
        for key in ("g", "keys", "draws", "pulses", "resets", "recovery_baseline"):
            value = state[key]
            setattr(self, key, None if value is None else value.to(self.device).clone())
        for key in ("refreshes", "refresh_reads", "recovery_cap"):
            setattr(self, key, state[key])

    def cost(self):
        return {
            "set_pulses": int(self.pulses.sum()),
            "reset_pulses": int(self.resets.sum()),
            "physical_pulses": int(self.pulses.sum() + self.resets.sum()),
            "max_cell_pulses": int((self.pulses + self.resets).max()),
            "refreshes": self.refreshes,
            "refresh_reads": self.refresh_reads,
        }


def make_plant(population, seed, device):
    if population_fingerprint(population) != population["fingerprint"]:
        raise ValueError("Physical population content does not match its fingerprint.")
    cls = {"om": OmPlant, "pcm": PcmPlant}[population["kind"]]
    return cls(population, seed, device)


@torch.no_grad()
def program(port, target, *, tolerance, maximum_pulses):
    count = torch.zeros(port.size, dtype=torch.int64, device=target.device)
    reads = 0
    for _ in range(maximum_pulses):
        error = target - port.read()
        reads += port.size
        active = error.abs() > tolerance
        if not bool(active.any()):
            break
        port.pulse(error.sign().to(torch.int8) * active)
        count += active
    error = target - port.read()
    reads += port.size
    return {
        "accepted": int((error.abs() <= tolerance).sum()),
        "exhausted": int((error.abs() > tolerance).sum()),
        "verify_reads": reads,
        "pulses": int(count.sum()),
        "residual_rms": float(error.square().mean().sqrt()),
    }


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


class EndpointKernel:
    """Target-conditioned empirical residuals, including failure terminals.

    Adjacent target bins are mixed with linear probabilities. Each logical weight
    receives an independent draw; the final deployment population is never used.
    """

    def __init__(self, table, seed, device):
        self.table = table.to(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def perturb(self, q, strength=1.0):
        position = (q.detach().clamp(-1, 1) + 1) * 0.5 * (self.table.shape[0] - 1)
        lower = position.floor().long()
        upper = (lower + 1).clamp_max(self.table.shape[0] - 1)
        choose = (
            torch.rand(q.shape, device=q.device, generator=self.generator)
            < position - lower
        )
        bins = torch.where(choose, upper, lower)
        sample = torch.randint(
            self.table.shape[1], q.shape, device=q.device, generator=self.generator
        )
        residual = self.table[bins, sample]
        return q + float(strength) * residual
