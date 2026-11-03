"""IBM CMO/HfOx ReRAM: Gaussian program-and-verify endpoints and idealized writes.

The device law is AIHWKit 1.1.0 ``ReRamCMONoiseModel`` (Falcone et al. 2025):
programming a target conductance G* (in uS) gives G* + sigma(G*) xi, with
sigma linear in G*, coefficients chosen by the P&V acceptance range and
xi ~ N(0, 1) per physical device. Results are neither clamped nor truncated.
AIHWKit's default ``acceptance_range=2e-2`` is not a key of its coefficient
table and silently selects the 0.2 % fit; its 2 % fit needs
``acceptance_range=2``.

Each weight is a pair of physical devices read as q = (G_a - G_r) / 39.6 uS:
an active device programmed to MID_US + 39.6 q and a reference device
programmed to MID_US. Both carry their own programming error, and either may
be stuck. On-chip writes default to active devices; an explicit verified role
map can select the reference instead, with the opposite logical-update sign. Conductances are held:
reads return the same state until a program or write changes it. Stuck
devices hold their stuck conductance through every program and write. There
is no read noise, drift or quantization in this model. Off-chip HWA does not
use this law; it injects declared constant-sigma weight noise
(``methods/hwa.py``).
"""

from __future__ import annotations

from hashlib import sha256
import math

import torch

# AIHWKit 1.1.0 SingleDeviceConductanceConverter range defaults, in uS.
G_MIN_US = 9.0
G_MAX_US = 88.199997
MID_US = (G_MAX_US + G_MIN_US) / 2
HALF_RANGE_US = (G_MAX_US - G_MIN_US) / 2
# Stuck conductances (uS) of the fixed defect kinds; ``random`` draws uniformly in range.
STUCK_US = {"open": 0.0, "gmin": G_MIN_US, "gmax": G_MAX_US}
# sigma(G) = c0 + c1 * G in uS, keyed by P&V acceptance range in percent
# (AIHWKit 1.1.0 ReRamCMONoiseModel coeff_dict).
PROGRAMMING_SIGMA = {
    0.2: (0.00081107, 0.00106879),
    2.0: (0.0112185391, 0.01129027418),
}


def programming_sigma(g, acceptance):
    """Standard deviation (uS) of one programmed conductance ``g`` (uS)."""

    c0, c1 = PROGRAMMING_SIGMA[acceptance]
    return c0 + c1 * g


def to_conductance(q):
    """Active-device target conductance (uS) of normalized weights in [-1, 1]."""

    return MID_US + q * HALF_RANGE_US


def to_weight(active, reference):
    """Normalized weight of an active/reference conductance pair (uS)."""

    return (active - reference) / HALF_RANGE_US


ACTIVE, REFERENCE = 0, 1


def corrupt_weights(q, generator, kind, rate):
    """Normalized weights read from pairs whose devices are stuck i.i.d. with probability ``rate``.

    Temporary HWA corruption, never a deployed fault map. A healthy active
    device keeps ``q`` and a healthy reference reads as nominal mid-range; a
    stuck active device reads its stuck value, and a stuck reference shifts
    the weight by minus its offset from mid-range.
    """

    shape = (2,) + tuple(q.shape)
    mask = torch.rand(shape, generator=generator, device=q.device) < rate
    if kind == "random":
        stuck = G_MIN_US + (G_MAX_US - G_MIN_US) * torch.rand(shape, generator=generator, device=q.device)
    else:
        stuck = torch.full(shape, STUCK_US[kind], device=q.device)
    offset = (stuck - MID_US) / HALF_RANGE_US
    return torch.where(mask[ACTIVE], offset[ACTIVE], q) - torch.where(mask[REFERENCE], offset[REFERENCE], 0.0)


class CmoMatrix:
    """Held conductances of one analog matrix: an active and a reference device per weight.

    ``g``, ``mask`` and ``stuck`` have shape (2, size): row ``ACTIVE`` then
    row ``REFERENCE``. ``program`` draws one P&V endpoint per device: both
    rows at deployment, or the active row only when ``reference`` is false
    (the on-chip rewrite control). ``write`` is the idealized on-chip update of
    the active row: a single write per device, no verify read and no retry, so
    earlier write errors persist. Each matrix owns its noise stream. ``stuck``
    is an optional (mask, conductance in uS) pair of permanent defects: those
    devices are never programmed or written, and requests addressed to them
    are counted as ``stuck_write_attempts``. ``writes`` counts programs and
    writes of each active device.

    ``read_devices`` is the controller-visible verify read of every device's
    held conductance. ``program_devices`` reprograms selected devices of either
    row to arbitrary conductance targets with the same endpoint law, drawing
    from a caller-owned stream so that the matrix's own stream is untouched.
    """

    def __init__(self, size, seed, acceptance, noise_scale, device, stuck=None):
        if acceptance not in PROGRAMMING_SIGMA:
            raise ValueError(
                f"Expected a fitted acceptance range in {sorted(PROGRAMMING_SIGMA)}. Provided value: {acceptance!r}."
            )
        self.size = size
        self.acceptance = float(acceptance)
        self.noise_scale = float(noise_scale)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.g = None
        self.writes = torch.zeros(size, dtype=torch.int32, device=device)
        self.program_calls = 0
        self.device_programs = 0
        self.device_writes = 0
        self.stuck_write_attempts = 0
        self.verify_reads = 0
        # A seed recipe is valid only through the first full program. Even a
        # zero-update write advances the RNG, and a verify changes accounting.
        # Restored tensors have their own history and must retain full state.
        self._deployment_replayable = True
        self.mask = torch.zeros((2, size), dtype=torch.bool, device=device)
        self.stuck = torch.zeros((2, size), device=device)
        if stuck is not None:
            mask, conductance = stuck
            if mask.shape != (2, size) or conductance.shape != (2, size) or mask.dtype != torch.bool:
                raise ValueError("Expected stuck flags and conductances of shape (2, size): active, reference.")
            self.mask = mask.to(device)
            self.stuck = torch.where(self.mask, conductance.to(device, torch.float32), 0.0)

    @property
    def failed(self) -> int:
        return int(self.mask.sum())

    def can_rebuild_deployment(self) -> bool:
        """Whether the initial target and constructor seeds recover all state."""

        return self._deployment_replayable and self.program_calls == 1

    def fault_digest(self) -> str:
        """Identity of the permanent defects."""

        h = sha256(str(self.size).encode())
        h.update(self.mask.cpu().numpy().tobytes())
        h.update(self.stuck.cpu().numpy().tobytes())
        return h.hexdigest()

    def _draw(self, requested, factor=1.0):
        # The noise is drawn even when ``factor`` is zero, so the stream advances identically.
        noise = torch.randn(requested.shape, device=requested.device, generator=self.generator)
        return requested + (self.noise_scale * factor) * programming_sigma(requested, self.acceptance) * noise

    @torch.no_grad()
    def program(self, target, reference=True):
        """Program normalized targets: one unclamped Gaussian endpoint per device."""

        if target.shape != (self.size,) or not bool(torch.isfinite(target).all()) or bool((target.abs() > 1).any()):
            raise ValueError("Expected one finite normalized target in [-1, 1] per weight.")
        if not reference and self.g is None:
            raise RuntimeError("Expected programmed reference devices before reprogramming active devices only.")
        self._deployment_replayable &= self.program_calls == 0 and reference
        active = to_conductance(target)
        if reference:
            drawn = self._draw(torch.stack((active, torch.full_like(active, MID_US))))
            roles = self.mask
        else:
            drawn = torch.stack((self._draw(active), self.g[REFERENCE]))
            roles = self.mask[ACTIVE]
        self.g = torch.where(self.mask, self.stuck, drawn)
        failed = int(roles.sum())
        self.writes += ~self.mask[ACTIVE]
        self.program_calls += 1
        self.device_programs += roles.numel() - failed
        self.stuck_write_attempts += failed

    @torch.no_grad()
    def write(self, delta, noise_factor=1.0, roles=None):
        """G_a <- G_a + dG + f sigma(G_a + dG) xi on every active device with a nonzero update.

        ``noise_factor`` f scales write noise only; 1 is the physical law and 0 an
        exact-write oracle. Programming noise and stuck devices are unaffected.
        """

        if self.g is None:
            raise RuntimeError("Expected a programmed CMO matrix before writing to it.")
        if delta.shape != (self.size,) or not bool(torch.isfinite(delta).all()):
            raise ValueError("Expected one finite normalized update per weight.")
        if isinstance(noise_factor, bool) or not math.isfinite(noise_factor) or noise_factor < 0:
            raise ValueError(f"Expected a finite nonnegative write-noise factor. Provided value: {noise_factor!r}.")
        if roles is not None and (roles.shape != (self.size,) or roles.dtype != torch.int8
                                 or bool(((roles < -1) | (roles > REFERENCE)).any())):
            raise ValueError("Expected int8 write roles: -1 blocked, 0 active, 1 reference.")
        self._deployment_replayable = False
        if roles is not None:
            reference = roles == REFERENCE
            requested = (delta != 0) & (roles >= 0)
            failed = torch.where(reference, self.mask[REFERENCE], self.mask[ACTIVE])
            written = requested & ~failed
            held = torch.where(reference, self.g[REFERENCE], self.g[ACTIVE])
            change = torch.where(reference, -delta, delta) * HALF_RANGE_US
            # One draw per logical weight, just as in the legacy active-only law.
            drawn = self._draw(held + change, noise_factor)
            self.g = torch.stack((torch.where(written & ~reference, drawn, self.g[ACTIVE]),
                                  torch.where(written & reference, drawn, self.g[REFERENCE])))
            self.writes += written & ~reference
            if not hasattr(self, "reference_writes"):
                self.reference_writes = torch.zeros_like(self.writes)
            self.reference_writes += written & reference
            self.device_writes += int(written.sum())
            self.stuck_write_attempts += int(requested.sum()) - int(written.sum())
            return
        requested = delta != 0
        written = requested & ~self.mask[ACTIVE]
        active = torch.where(written, self._draw(self.g[ACTIVE] + delta * HALF_RANGE_US, noise_factor),
                             self.g[ACTIVE])
        self.g = torch.stack((active, self.g[REFERENCE]))
        self.writes += written
        self.device_writes += int(written.sum())
        self.stuck_write_attempts += int(requested.sum()) - int(written.sum())

    @torch.no_grad()
    def program_devices(self, target_us, selected, generator):
        """Reprogram ``selected`` devices of either row to ``target_us``: one P&V endpoint each.

        ``target_us`` and ``selected`` have shape (2, size). Noise is drawn for
        every device from ``generator`` (never the matrix's own stream), so a
        device receives the same draw whichever other devices are selected.
        Stuck devices keep their conductance and count as stuck attempts.
        """

        if self.g is None:
            raise RuntimeError("Expected a programmed CMO matrix before reprogramming devices.")
        if selected.shape != (2, self.size) or selected.dtype != torch.bool or target_us.shape != (2, self.size):
            raise ValueError("Expected selection flags and conductance targets of shape (2, size).")
        requested = torch.where(selected, target_us.to(self.g), self.g)
        if not bool(torch.isfinite(requested).all()) or bool((requested[selected] < 0).any()):
            raise ValueError("Expected finite nonnegative conductance targets for the selected devices.")
        self._deployment_replayable = False
        noise = torch.randn(requested.shape, device=requested.device, generator=generator)
        drawn = requested + self.noise_scale * programming_sigma(requested, self.acceptance) * noise
        programmed = selected & ~self.mask
        self.g = torch.where(programmed, drawn, self.g)
        self.writes += programmed[ACTIVE]
        self.program_calls += 1
        self.device_programs += int(programmed.sum())
        self.stuck_write_attempts += int((selected & self.mask).sum())

    def read_devices(self):
        """Verify read of every device's held conductance (uS), shape (2, size); two reads per weight."""

        if self.g is None:
            raise RuntimeError("Expected a programmed CMO matrix before reading it.")
        self._deployment_replayable = False
        self.verify_reads += self.g.numel()
        return self.g.detach().clone()

    def read(self):
        if self.g is None:
            raise RuntimeError("Expected a programmed CMO matrix before reading it.")
        return to_weight(self.g[ACTIVE], self.g[REFERENCE]).detach().clone()

    def out_of_range(self) -> int:
        """Healthy devices whose held conductance lies outside [G_MIN_US, G_MAX_US]."""

        return int((((self.g < G_MIN_US) | (self.g > G_MAX_US)) & ~self.mask).sum())

    def verify_permanence(self) -> None:
        if not torch.equal(self.g[self.mask], self.stuck[self.mask]):
            raise RuntimeError("A permanent CMO defect changed state.")

    def cost(self) -> dict:
        return {
            "array_program_calls": self.program_calls,
            "device_programs": self.device_programs,
            "device_writes": self.device_writes,
            "stuck_write_attempts": self.stuck_write_attempts,
            "verify_reads": self.verify_reads,
        }

    def state_dict(self) -> dict:
        counters = [self.program_calls, self.device_programs, self.device_writes, self.stuck_write_attempts]
        return {
            "contract": [self.acceptance, self.noise_scale],
            "faults_sha256": self.fault_digest(),
            "g": self.g.detach().cpu().clone(),
            "writes": self.writes.cpu().clone(),
            **({"reference_writes": self.reference_writes.cpu().clone()}
               if hasattr(self, "reference_writes") else {}),
            "rng": self.generator.get_state(),
            # Verify reads are recorded only once taken, so earlier states keep their bytes.
            "counters": counters + ([self.verify_reads] if self.verify_reads else []),
        }

    def load_state_dict(self, state) -> None:
        if list(state["contract"]) != [self.acceptance, self.noise_scale]:
            raise ValueError("Expected the same CMO acceptance range and noise scale.")
        if state["faults_sha256"] != self.fault_digest():
            raise ValueError("Expected the same permanent CMO defects.")
        self._deployment_replayable = False
        device = self.writes.device
        self.g = state["g"].to(device).clone()
        self.writes = state["writes"].to(device).clone()
        if "reference_writes" in state:
            self.reference_writes = state["reference_writes"].to(device).clone()
        elif hasattr(self, "reference_writes"):
            del self.reference_writes
        self.generator.set_state(state["rng"].cpu())
        counters = list(state["counters"])
        if len(counters) not in (4, 5):
            raise ValueError("Expected four CMO counters, plus verify reads when any were taken.")
        self.program_calls, self.device_programs, self.device_writes, self.stuck_write_attempts = counters[:4]
        self.verify_reads = counters[4] if len(counters) == 5 else 0


__all__ = [
    "ACTIVE",
    "REFERENCE",
    "CmoMatrix",
    "G_MAX_US",
    "G_MIN_US",
    "HALF_RANGE_US",
    "MID_US",
    "PROGRAMMING_SIGMA",
    "STUCK_US",
    "corrupt_weights",
    "programming_sigma",
    "to_conductance",
    "to_weight",
]
