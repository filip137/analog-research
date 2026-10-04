"""Separate AIHWKit 1.1.0 PCM inference law with explicit random generators.

Equations transcribed from aihwkit/inference/noise/pcm.py (IBM, MIT license).
Single differential pair, g_max=25 uS, t0=20 s, read time=250 ns. This module
contains no PCM write/update law and never reports estimated pulse counts.
"""

from __future__ import annotations
import math
import torch


class PcmInferenceNoise:
    def __init__(
        self,
        student,
        seed,
        device,
        *,
        time_seconds=1.0,
        noise_scale=1.0,
        compensate=True,
    ):
        self.layout = student.layout
        self.tile_size = student.tile_size
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.time_seconds, self.noise_scale, self.compensate = (
            time_seconds,
            noise_scale,
            compensate,
        )

    def _normal(self, value):
        return torch.randn(value.shape, device=value.device, generator=self.generator)

    def realization(self, q):
        target = (
            torch.stack((q.detach().clamp_min(0), (-q.detach()).clamp_min(0))) * 25.0
        )
        relative = target / 25.0
        sigma = 0.26348 + 1.965 * relative - 1.1731 * relative.square()
        programmed = (
            target + self.noise_scale * sigma * self._normal(target)
        ).clamp_min(0)
        r = relative.abs().clamp_min(1e-7)
        mu = (-0.0155 * r.log() + 0.0244).clamp(0.049, 0.1)
        sd = (-0.0125 * r.log() - 0.0059).clamp(0.008, 0.045)
        nu = (mu + sd * self._normal(r)).abs()
        t = self.time_seconds + 20.0
        drift = programmed * (t / 20.0) ** (-nu)
        qs = (0.0088 / (programmed.abs() / 25.0).pow(0.65).clamp_min(1e-3)).clamp_max(
            0.2
        )
        scale = qs * math.sqrt(math.log((t + 2.5e-7) / (2 * 2.5e-7)))
        observed = (drift + drift.abs() * scale * self._normal(drift)).clamp_min(0)
        q0 = (programmed[0] - programmed[1]) / 25.0
        qt = (observed[0] - observed[1]) / 25.0
        if self.compensate:
            qt = self.compensation(q0, qt)
        return q0, qt

    def compensation(self, initial, current):
        result = current.clone()
        for spec in self.layout:
            a = initial[spec.offset : spec.offset + spec.size].reshape(
                spec.shape[0], -1
            )
            b = result[spec.offset : spec.offset + spec.size].reshape_as(a)
            for output in range(0, a.shape[0], self.tile_size):
                for row in range(0, a.shape[1], self.tile_size):
                    ref = a[
                        output : output + self.tile_size, row : row + self.tile_size
                    ]
                    new = b[
                        output : output + self.tile_size, row : row + self.tile_size
                    ]
                    # AIHWKit 1.1.0 uses all one-hot probes (the identity),
                    # then averages absolute outputs over the whole tile.
                    factor = ref.abs().mean().clamp_min(
                        1e-4
                    ) / new.abs().mean().clamp_min(1e-4)
                    new.mul_(factor)
        return result

    def perturb(self, q, strength=1.0):
        _, value = self.realization(q)
        return q + strength * (value - q.detach())


def evaluate_reference(spec, student, cache, device, calibration_cache=None):
    from experiments.cifar_crossbar.runtime import evaluate

    clean = evaluate(student, cache, device)
    model = PcmInferenceNoise(
        student,
        spec.endpoint_seed,
        device,
        time_seconds=spec.retention_seconds,
        noise_scale=spec.noise_scale,
        compensate=spec.drift_compensation,
    )
    q0, qt = model.realization(student.q)
    result = {
        "stage": "pcm_reference",
        "clean": clean,
        "programmed": evaluate(student, cache, device, q=q0),
        "retained": evaluate(student, cache, device, q=qt),
        "retention_seconds": spec.retention_seconds,
        "drift_compensation": spec.drift_compensation,
        "model": "AIHWKit_1.1.0_PCMLikeNoiseModel",
        "g_max_microsiemens": 25.0,
        "pulse_recovery_supported": False,
    }
    if calibration_cache is not None:
        # BN-only calibration of converted blocks, never the frozen prefix.
        for i in range(student.start, 5):
            for j in (1, 2):
                bn = student.model.get_submodule(f"layer3.{i}.bn{j}")
                bn.reset_running_stats()
                bn.momentum = None
                bn.train()
        with torch.no_grad():
            for begin in range(0, len(calibration_cache["labels"]), spec.batch_size):
                student.forward_features(
                    calibration_cache["features"][begin : begin + spec.batch_size].to(
                        device
                    ),
                    qt,
                )
        student.eval()
        result["normalization_calibrated"] = evaluate(student, cache, device, q=qt)
        result["calibration_images"] = len(calibration_cache["labels"])
    return result
