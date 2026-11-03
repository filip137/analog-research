"""Held PCM relaxation state, with its programming-event random stream."""
from __future__ import annotations
import torch
from hardware.drift.inference import PCM_READ_TIME, PCM_T0, pcm_coefficients, pcm_snapshot
from hardware.readout import tile_compensation

G_MAX = 25.0


DRIFT_T0 = PCM_T0


READ_TIME = PCM_READ_TIME


def pcm_drift(programmed, relative, generator, seconds):
    """Conductances read ``seconds`` after programming (AIHWKit 1.1.0 PCM).

    ``relative`` is the programming target over g_max; it sets the drift
    coefficients. Equations match ``experiments.cifar_crossbar.pcm_reference``.
    """

    # Keep this programming-event API and its random stream unchanged. A time
    # sweep calls pcm_coefficients once and reuses it across pcm_snapshot calls.
    nu = pcm_coefficients(relative, generator, g_max=1.0)
    return pcm_snapshot(programmed, nu, generator, seconds, g_max=G_MAX)


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
