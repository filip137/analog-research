"""Conductance retention laws from IBM AIHWKit 1.1.0, in microSiemens.

These are inference snapshots, not write laws. Always start from the same
programmed conductances when comparing ages. PCM's exponent is sampled once
per device; CMO and Wan supply marginal distributions, not a measured joint
stochastic process across time. The caller holds each returned snapshot for
all examples and reapplies permanent faults after sampling healthy devices.

Sources (MIT licensed, IBM):
https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/pcm.py
https://github.com/IBM/aihwkit/blob/v1.1.0/src/aihwkit/inference/noise/reram.py
"""

from __future__ import annotations

import math
from hashlib import sha256

import torch

from hardware.devices.cmo import G_MAX_US, G_MIN_US

AIHWKIT_VERSION = "1.1.0"
PCM_T0 = 20.0
PCM_READ_TIME = 250e-9
CMO_READ_TIME = 1e-6
WAN_COEFFICIENTS = {
    1.0: (0.348, 16.030, -43.853, 45.393, -16.815),
    86400.0: (0.701, 22.086, -50.773, 47.095, -16.458),
    172800.0: (0.782, 20.274, -43.507, 37.062, -11.934),
}


def _nonnegative(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Expected finite nonnegative {name}.")


def _normal(value, generator):
    return torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)


@torch.no_grad()
def pcm_coefficients(target, generator, *, g_max=25.0, drift_scale=1.0):
    """Sample a fixed PCM exponent per device from its programming target."""
    _nonnegative(drift_scale, "drift_scale")
    if not math.isfinite(g_max) or g_max <= 0:
        raise ValueError("Expected positive finite g_max.")
    relative = (target / g_max).abs().clamp_min(1e-7)
    mean = (-0.0155 * relative.log() + 0.0244).clamp(0.049, 0.1)
    std = (-0.0125 * relative.log() - 0.0059).clamp(0.008, 0.045)
    return (mean + std * _normal(relative, generator)).abs() * drift_scale


@torch.no_grad()
def pcm_snapshot(programmed, coefficients, generator, seconds, *, g_max=25.0, read_noise_scale=1.0):
    """AIHWKit PCM snapshot; time is seconds after its 20-second reference.

At seconds=0 AIHWKit still adds read noise. A programming-only baseline must
therefore use the unchanged programmed tensor, not this function with t=0.
"""
    _nonnegative(seconds, "seconds")
    _nonnegative(read_noise_scale, "read_noise_scale")
    if not math.isfinite(g_max) or g_max <= 0:
        raise ValueError("Expected positive finite g_max.")
    if coefficients.shape != programmed.shape:
        raise ValueError("Expected one fixed PCM exponent per programmed device.")
    time = seconds + PCM_T0
    drifted = programmed * (time / PCM_T0) ** (-coefficients) if seconds else programmed
    qs = (0.0088 / (programmed.abs() / g_max).pow(0.65).clamp_min(1e-3)).clamp_max(0.2)
    sigma = qs * math.sqrt(math.log((time + PCM_READ_TIME) / (2 * PCM_READ_TIME)))
    return (drifted + drifted.abs() * read_noise_scale * sigma * _normal(programmed, generator)).clamp_min(0)


@torch.no_grad()
def cmo_snapshot(programmed, generator, seconds, *, drift_scale=1.0, read_noise_scale=1.0):
    """AIHWKit CMO log-time relaxation, optional read noise, and final clamp.

The supported sweep domain is t=0 or t>=1 second. Programming endpoints stay
unclamped at t=0. For t>0 the upstream law clamps to [9, 88.199997] uS.
Open/stuck devices must be excluded by the caller from the healthy-device
calculation and restored afterwards; they do not undergo relaxation.
"""
    _nonnegative(seconds, "seconds")
    _nonnegative(drift_scale, "drift_scale")
    _nonnegative(read_noise_scale, "read_noise_scale")
    if seconds == 0:
        return programmed.clone()
    if seconds < 1:
        raise ValueError("CMO retention snapshots require t=0 or t>=1 second.")
    log_time = math.log(seconds)
    mean = programmed + (-0.08900206 * log_time * drift_scale)
    sigma = 0.04201137 * log_time + 0.41183342
    drifted = mean + _normal(programmed, generator) * sigma * drift_scale
    read_draw = _normal(programmed, generator)
    if read_noise_scale:
        if not bool((drifted > 0).all()):
            raise ValueError("CMO read noise requires positive healthy-device conductances before log10.")
        sigma_read = 0.0277316483 * drifted.log10() * math.sqrt(
            math.log((seconds + CMO_READ_TIME) / (2 * CMO_READ_TIME))
        )
        drifted = drifted + sigma_read * read_draw * read_noise_scale
    return drifted.clamp(G_MIN_US, G_MAX_US)


@torch.no_grad()
def wan_snapshot(target, generator, seconds, *, g_max=40.0, noise_scale=1.0):
    """Wan-2022 total endpoint error at exactly 1 s, 1 day or 2 days.

There is intentionally no programmed-state argument: upstream ignores it and
draws total error around the original target. Adding this to a programmed
state would double-count programming error. No interpolation is supported.

g_max=40 uS is explicit here, matching the range used to fit the coefficients;
AIHWKit's default converter instead uses 25 uS. AIHWKit 1.1.0 assigns its
effective coeff_g_max_reference to g_max, so its polynomial scale is one.
"""
    _nonnegative(seconds, "seconds")
    _nonnegative(noise_scale, "noise_scale")
    if seconds not in WAN_COEFFICIENTS:
        raise ValueError(f"Wan-2022 requires seconds in {tuple(WAN_COEFFICIENTS)}; no interpolation.")
    if not math.isfinite(g_max) or g_max <= 0:
        raise ValueError("Expected positive finite g_max.")
    coefficients = WAN_COEFFICIENTS[seconds]
    power, sigma = 1, coefficients[0]
    for coefficient in coefficients[1:]:
        power = power * (target / g_max)
        sigma = sigma + power * coefficient
    return (target + noise_scale * sigma * _normal(target, generator)).clamp_min(0)


class RetentionSeries:
    """Replayable time snapshots from one already programmed physical array.

    Inputs are physical conductances, with whatever encoding the caller has
    explicitly declared. This class does not remap weights between technologies.
    Permanent faults override every snapshot. Distinct ages use independent
    noise streams keyed by age, so visiting ages in another order cannot change
    results. PCM additionally retains one exponent per physical device.

    ``baseline()`` returns the unchanged programmed array. For Wan this is
    conventionally the 1-second programming fit, not a measured t=0 endpoint.
    CMO/Wan snapshots must not be described as an observed joint time trajectory.
    """

    contract = "aihwkit_1.1.0.retention_series.v1"
    models = ("pcm", "reram_cmo", "reram_wan2022")

    def __init__(self, model, target, programmed, *, seed, read_noise_scale,
                 fault_mask=None, stuck=None, g_max=None):
        if model not in self.models:
            raise ValueError(f"Expected retention model in {self.models}.")
        if type(seed) is not int or seed < 0:
            raise ValueError("Expected a nonnegative integer retention seed.")
        _nonnegative(read_noise_scale, "read_noise_scale")
        if model == "reram_wan2022" and read_noise_scale:
            raise ValueError("Wan-2022 supplies no separate read-noise model; use read_noise_scale=0.")
        if target.shape != programmed.shape or target.device != programmed.device or target.dtype != programmed.dtype:
            raise ValueError("Expected target and programmed conductances with matching shape, device and dtype.")
        if not target.is_floating_point() or not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(programmed).all()):
            raise ValueError("Expected finite floating-point conductances.")
        if (fault_mask is None) != (stuck is None):
            raise ValueError("Expected both a fault mask and stuck conductances, or neither.")
        if fault_mask is None:
            fault_mask = torch.zeros_like(target, dtype=torch.bool)
            stuck = torch.zeros_like(target)
        if fault_mask.dtype != torch.bool or fault_mask.shape != target.shape or stuck.shape != target.shape:
            raise ValueError("Expected one Boolean fault flag and stuck conductance per device.")
        self.model, self.seed = model, seed
        self.read_noise_scale = float(read_noise_scale)
        self.g_max = {"pcm": 25.0, "reram_cmo": G_MAX_US, "reram_wan2022": 40.0}[model] if g_max is None else g_max
        if not math.isfinite(self.g_max) or self.g_max <= 0:
            raise ValueError("Expected positive finite g_max.")
        if model == "reram_cmo" and self.g_max != G_MAX_US:
            raise ValueError("CMO retention uses the pinned 9-88.199997 uS range.")
        self.target, self.programmed = target.detach().clone(), programmed.detach().clone()
        self.mask = fault_mask.to(target.device).clone()
        self.stuck = torch.where(self.mask, stuck.to(target), 0.0)
        if not bool(torch.isfinite(self.stuck).all()) or not torch.equal(self.programmed[self.mask], self.stuck[self.mask]):
            raise ValueError("Expected programmed conductances to preserve finite stuck values exactly.")
        self.coefficients = (pcm_coefficients(self.target, self._generator("coefficients"), g_max=self.g_max)
                             if model == "pcm" else None)

    def _generator(self, label):
        key = f"{self.contract}/{self.model}/{self.seed}/{label}".encode()
        seed = int.from_bytes(sha256(key).digest()[:8], "little") % (2**63)
        return torch.Generator(device=self.target.device).manual_seed(seed)

    def baseline(self):
        return self.programmed.clone()

    @torch.no_grad()
    def snapshot(self, seconds, noise_scale=1.0):
        """``noise_scale`` multiplies Wan's endpoint sigma on the unchanged age stream.

        Every scale reuses the same standard-normal draw, so scaled snapshots are
        matched realizations and scale one reproduces the unscaled snapshot exactly.
        """
        _nonnegative(seconds, "seconds")
        if not math.isfinite(noise_scale) or noise_scale <= 0:
            raise ValueError("Expected a positive finite retention noise scale.")
        if self.model != "reram_wan2022" and noise_scale != 1:
            raise ValueError("Only Wan-2022 endpoint snapshots support a noise scale.")
        generator = self._generator(f"seconds/{float(seconds).hex()}")
        if self.model == "pcm":
            result = pcm_snapshot(self.programmed, self.coefficients, generator, seconds,
                                  g_max=self.g_max, read_noise_scale=self.read_noise_scale)
        elif self.model == "reram_cmo":
            # Stuck-open devices have G=0 and must never enter log10(G).
            healthy = torch.where(self.mask, G_MIN_US, self.programmed)
            result = cmo_snapshot(healthy, generator, seconds, read_noise_scale=self.read_noise_scale)
        else:
            result = wan_snapshot(self.target, generator, seconds, g_max=self.g_max, noise_scale=noise_scale)
        result = torch.where(self.mask, self.stuck, result)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("Nonfinite retention snapshot.")
        return result

    def state_dict(self):
        return {
            "contract": self.contract, "model": self.model, "seed": self.seed,
            "read_noise_scale": self.read_noise_scale, "g_max": self.g_max,
            "execution_device_type": self.target.device.type,
            **{name: None if value is None else value.detach().cpu().clone()
               for name, value in (("target", self.target), ("programmed", self.programmed),
                                   ("fault_mask", self.mask), ("stuck", self.stuck),
                                   ("coefficients", self.coefficients))},
        }

    @classmethod
    def from_state_dict(cls, state, device="cpu"):
        if state.get("contract") != cls.contract:
            raise ValueError("Expected the pinned retention-series contract.")
        if torch.device(device).type != state["execution_device_type"]:
            raise ValueError("Retention RNG replay requires the original execution-device type.")
        result = cls(state["model"], state["target"].to(device), state["programmed"].to(device),
                     seed=state["seed"], read_noise_scale=state["read_noise_scale"], g_max=state["g_max"],
                     fault_mask=state["fault_mask"].to(device), stuck=state["stuck"].to(device))
        coefficients = state["coefficients"]
        if result.model == "pcm":
            if coefficients is None or coefficients.shape != result.target.shape or not bool(torch.isfinite(coefficients).all()):
                raise ValueError("Expected saved finite PCM coefficients for every device.")
            result.coefficients = coefficients.to(device).clone()
        elif coefficients is not None:
            raise ValueError("Only PCM uses stored drift coefficients.")
        return result
