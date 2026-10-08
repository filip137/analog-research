"""(iii) Program-and-verify: write normalized targets into fresh arrays.

Declared methods:

- ``closed_loop`` (OM): at most ``max_cycles`` read-verify-pulse cycles. Each
  cycle reads every apparent state and pulses each cell outside
  +/- ``tolerance_steps`` nominal steps once toward its target.
- ``open_loop_nominal`` (OM): pulse counts from the nominal soft-bounds
  inverse, starting at RESET. No verify reads and no device identities.
- ``gaussian_endpoint`` (PCM): one Gaussian endpoint per programming request;
  permanent failures are not repaired by writes.

Relaxation (PCM only) is the AIHWKit 1.1.0 PCM drift law with 1/f read noise
after ``seconds`` and optional per-tile global drift compensation. It yields
one held apparent state per programming event, so every later forward of that
state -- gradients, selection and evaluation -- sees the same relaxed array.
IBM OM relaxation, retention and drift are not modelled (see
``docs/ibm_reram_program_verify_noise_model.md``).
"""

from __future__ import annotations

import math

import torch

from experiments.cifar_crossbar.devices import make_plant, prepare_population
from experiments.cifar_crossbar.devices import program as verify_pulses
from experiments.cifar_crossbar.full_epoch_runtime import om_population
from experiments.cifar_crossbar.model import tensor_hash
from experiments.cifar_crossbar.open_loop_updates import apply_pulse_counts, nominal_reset_counts
from experiments.cifar_crossbar.sweep_devices import binding_seed
from workflow import devices as device_stage
from workflow.lifecycle import FAILURE_KINDS

G_MAX = 25.0
DRIFT_T0 = 20.0
READ_TIME = 2.5e-7


def pcm_drift(programmed, relative, generator, seconds):
    """Conductances read ``seconds`` after programming (AIHWKit 1.1.0 PCM).

    ``relative`` is the programming target over g_max; it sets the drift
    coefficients. Equations match ``experiments.cifar_crossbar.pcm_reference``.
    """

    def normal(value):
        return torch.randn(value.shape, device=value.device, generator=generator)

    r = relative.abs().clamp_min(1e-7)
    mu = (-0.0155 * r.log() + 0.0244).clamp(0.049, 0.1)
    sd = (-0.0125 * r.log() - 0.0059).clamp(0.008, 0.045)
    nu = (mu + sd * normal(r)).abs()
    t = seconds + DRIFT_T0
    drift = programmed * (t / DRIFT_T0) ** (-nu)
    qs = (0.0088 / (programmed.abs() / G_MAX).pow(0.65).clamp_min(1e-3)).clamp_max(0.2)
    scale = qs * math.sqrt(math.log((t + READ_TIME) / (2 * READ_TIME)))
    return (drift + drift.abs() * scale * normal(drift)).clamp_min(0)


def tile_compensation(initial, current, layout, tile_size):
    """Per-tile global drift compensation from one-hot reference reads."""

    result = current.clone()
    for spec in layout:
        a = initial[spec.offset : spec.offset + spec.size].reshape(spec.shape[0], -1)
        b = result[spec.offset : spec.offset + spec.size].reshape_as(a)
        for output in range(0, a.shape[0], tile_size):
            for row in range(0, a.shape[1], tile_size):
                ref = a[output : output + tile_size, row : row + tile_size]
                new = b[output : output + tile_size, row : row + tile_size]
                factor = ref.abs().mean().clamp_min(1e-4) / new.abs().mean().clamp_min(1e-4)
                new.mul_(factor)
    return result


class PcmDrift:
    """Relax a programmed PCM array once per programming event."""

    contract = "crossbar_lifecycle.pcm_drift_aihwkit_1.1.0.v1"

    def __init__(self, seconds, compensation, layout, tile_size, seed, device):
        self.seconds, self.compensation = float(seconds), bool(compensation)
        self.layout, self.tile_size = tuple(layout), tile_size
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.realizations = 0

    def apply(self, array):
        programmed = array.programmed_read()
        pieces = []
        for child in array.children:
            target = torch.stack((child.target.clamp_min(0), (-child.target).clamp_min(0))) * G_MAX
            observed = pcm_drift(child.g, target / G_MAX, self.generator, self.seconds)
            # Permanent failures keep their stuck conductance at every time.
            observed = child.faults.apply(observed)
            pieces.append((observed[0] - observed[1]) / G_MAX)
        relaxed = torch.cat(pieces)
        if self.compensation:
            relaxed = tile_compensation(programmed, relaxed, self.layout, self.tile_size)
        self.realizations += 1
        return relaxed

    def state_dict(self):
        return {
            "contract": [self.contract, self.seconds, self.compensation, self.tile_size],
            "rng": self.generator.get_state(),
            "realizations": self.realizations,
        }

    def load_state_dict(self, state):
        if list(state["contract"]) != [self.contract, self.seconds, self.compensation, self.tile_size]:
            raise ValueError("Expected the same PCM relaxation contract.")
        self.generator.set_state(state["rng"].cpu())
        self.realizations = state["realizations"]


def configure(array, lifecycle, assignment_seed):
    """Attach the declared relaxation law to a fresh, unprogrammed array."""

    relaxation = lifecycle.program_verify.relaxation
    if relaxation.model == "none":
        return array
    if lifecycle.devices.technology != "pcm" or relaxation.model != "pcm_drift":
        raise ValueError("Expected PCM drift as the only supported relaxation law.")
    seed = binding_seed(
        lifecycle.devices.endpoint_seed(assignment_seed), lifecycle.network.binding_namespace, "pcm_drift"
    )
    array.relaxation = PcmDrift(
        relaxation.seconds,
        relaxation.compensation,
        array.layout,
        lifecycle.deployment.tile_size,
        seed,
        array.device,
    )
    return array


def fresh(lifecycle, layout, assignment_seed, case, populations, device):
    """A fresh array of one assignment and defect case, ready to program."""

    array = device_stage.fresh_array(lifecycle, layout, assignment_seed, case, populations, device)
    return configure(array, lifecycle, assignment_seed)


@torch.no_grad()
def program(array, target, lifecycle) -> dict:
    """Program ``target`` with the lifecycle's P&V method; return its report."""

    pv = lifecycle.program_verify
    if pv.method == "gaussian_endpoint":
        before = array.cost()
        array.program(target)
        after = array.cost()
        report = {"cycles": 0, "verify_reads": 0, **{k: after[k] - before[k] for k in after}}
    elif pv.method == "closed_loop":
        report = verify_pulses(
            array.port(),
            target,
            tolerance=array.nominal_step * pv.tolerance_steps,
            maximum_pulses=pv.max_cycles,
        )
        report["tolerance"] = array.nominal_step * pv.tolerance_steps
    elif pv.method == "open_loop_nominal":
        counts = nominal_reset_counts(target, array.nominal_step)
        apply_pulse_counts(array.port(), torch.ones_like(target, dtype=torch.int8), counts)
        report = {
            "verify_reads": 0,
            "pulses": int(counts.sum()),
            "max_cell_pulses": int(counts.max()),
            "pulse_counts_sha256": tensor_hash(counts),
            "nominal_step": array.nominal_step,
        }
    else:  # pragma: no cover - the schema admits only the methods above
        raise ValueError(f"Unsupported programming method: {pv.method!r}.")
    apparent = array.read()
    report.update(
        method=pv.method,
        relaxation=pv.relaxation.model,
        apparent_sha256=tensor_hash(apparent),
        apparent_residual_rms=float((target - apparent).square().mean().sqrt()),
    )
    if pv.relaxation.model != "none":
        report["programmed_residual_rms"] = float(
            (target - array.programmed_read()).square().mean().sqrt()
        )
    return report


def characterize_endpoints(populations, lifecycle, device) -> dict:
    """Residual tables of the declared P&V law on independent OM populations.

    Fit tables drive the HWA endpoint sampler; held-out tables gate adequacy
    with the existing per-bin mean test. Failed kinds program fully faulted
    copies of the same characterization cells.
    """

    c = lifecycle.devices.characterization
    grid = torch.linspace(-1, 1, c.bins, device=device).repeat_interleave(c.samples)
    tables, diagnostics = {}, []
    for role in ("fit", "heldout"):
        raw = populations["kernel_" + role]
        tables[role] = {}
        for kind in ("healthy",) + FAILURE_KINDS:
            population = (
                prepare_population(raw) if kind == "healthy" else om_population(raw, kind, 1.0, raw["seed"])
            )
            plant = make_plant(population, raw["seed"] + lifecycle.devices.endpoint_seed_offset, device)
            stats = program(plant, grid, lifecycle)
            tables[role][kind] = (plant.read() - grid).reshape(c.bins, c.samples).detach().cpu()
            diagnostics.append(
                {
                    "role": role,
                    "kind": kind,
                    "identity": population["fingerprint"],
                    "program": stats,
                    "cost": plant.cost(),
                }
            )
    gates = {}
    for kind, fit in tables["fit"].items():
        heldout = tables["heldout"][kind]
        error = (fit.mean(1) - heldout.mean(1)).abs()
        bound = torch.maximum(
            torch.full_like(error, 0.02),
            4 * (fit.var(1) / fit.shape[1] + heldout.var(1) / heldout.shape[1]).sqrt(),
        )
        gates[kind] = {
            "adequate": bool((error <= bound).all()),
            "max_mean_error": float(error.max()),
            "mean_error": error.tolist(),
            "bound": bound.tolist(),
        }
    return {
        "tables": tables,
        "gates": gates,
        "diagnostics": diagnostics,
        "adequate": all(gate["adequate"] for gate in gates.values()),
        "grid": {"bins": c.bins, "samples": c.samples},
    }
