"""Independent AIHWKit 1.1.0 checks; run with the CPU reference interpreter."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch


def checks():
    import aihwkit
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.parameters import UpdateParameters, PulseType
    from aihwkit.simulator.presets.devices import PCMPresetDevice
    from aihwkit.simulator.tiles import AnalogTile
    from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
    from aihwkit.inference.compensation.drift import GlobalDriftCompensation
    from experiments.cifar_crossbar.model import CifarResNet32, CrossbarSuffix
    from experiments.cifar_crossbar.pcm_reference import PcmInferenceNoise
    from experiments.cifar_crossbar.native import sample

    if aihwkit.__version__ != "1.1.0":
        raise RuntimeError("Expected AIHWKit 1.1.0.")
    torch.set_num_threads(1)
    population_checks = {}
    for kind in ("om", "pcm"):
        first = sample(kind, 512, 901, "repaired", 1.0)
        repeated = sample(kind, 512, 901, "repaired", 1.0)
        if first["hidden_sha256"] != repeated["hidden_sha256"]:
            raise AssertionError(
                "Native physical identities must be reproducible independently of tile initialization."
            )
        if "persistent_weights" in first["hidden"]:
            raise AssertionError(
                "A random initial tile state is not a device parameter."
            )
        population_checks[kind] = first["hidden_sha256"]
    preset = PCMPresetDevice(
        dw_min_std=0.0,
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        reset_std=0.0,
        reset_dtod=0.0,
    )
    update = UpdateParameters(
        desired_bl=1,
        pulse_type=PulseType.DETERMINISTIC_IMPLICIT,
        update_bl_management=False,
        update_management=False,
    )
    tile = AnalogTile(1, 1, SingleRPUConfig(device=preset, update=update), bias=False)
    tile.set_weights(torch.tensor([[0.5]]))
    # One full deterministic bit line. Native ExpStep granularity differs from
    # nominal dw_min; a unit learning rate ensures the single-bit train fires.
    tile.set_learning_rate(1.0)
    tile.update(torch.ones(1, 1), -torch.ones(1, 1))
    native_set = float(tile.get_weights()[0].item())
    expected = (
        0.5 + preset.dw_min * (1 - preset.A_up * torch.exp(torch.tensor(-1.25))).item()
    )
    if abs(native_set - expected) > 2e-7:
        raise AssertionError((native_set, expected))
    tile.tile.reset_columns(0, 1, 1.0)
    native_reset = float(tile.get_weights()[0].item())
    if abs(native_reset - preset.reset) > 2e-7:
        raise AssertionError((native_reset, preset.reset))
    student = CrossbarSuffix(CifarResNet32(10).eval(), "head")
    q = torch.linspace(-1, 1, student.q.numel())
    maximum = 0.0
    for seconds in (1.0, 3600.0, 86400.0, 31536000.0):
        own = PcmInferenceNoise(
            student, 927, "cpu", time_seconds=seconds, compensate=False
        )
        reference = PCMLikeNoiseModel(g_max=25.0)
        generator = torch.Generator().manual_seed(927)

        def normal(value):
            return torch.randn(value.shape, generator=generator)

        g = torch.stack((q.clamp_min(0), (-q).clamp_min(0))) * 25.0
        with patch("aihwkit.inference.noise.pcm.randn_like", normal):
            g0 = reference.apply_programming_noise_to_conductance(g)
            nu = reference.generate_drift_coefficients(g)
            gt = reference.apply_drift_noise_to_conductance(g0, nu, seconds)
        a, b = own.realization(q)
        expected0, expectedt = (g0[0] - g0[1]) / 25.0, (gt[0] - gt[1]) / 25.0
        torch.testing.assert_close(a, expected0, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(b, expectedt, atol=2e-7, rtol=2e-6)
        maximum = max(maximum, float((b - expectedt).abs().max()))
    compensation = GlobalDriftCompensation()
    initial = torch.linspace(-1, 1, student.q.numel())
    current = initial * 0.8 + 0.03
    matrix = initial.reshape(10, 64)
    aged = current.reshape_as(matrix)
    probes = compensation.get_readout_tensor(64)
    ratio = compensation.readout(probes @ matrix.T) / compensation.readout(
        probes @ aged.T
    )
    torch.testing.assert_close(own.compensation(initial, current), current * ratio)
    return {
        "repeated_native_populations": population_checks,
        "aihwkit_version": aihwkit.__version__,
        "torch_version": str(torch.__version__),
        "pcm_single_set_native": native_set,
        "pcm_single_set_expected": expected,
        "pcm_reset_native": native_reset,
        "pcm_inference_max_abs_error": maximum,
        "drift_compensation": "native_identity_probe_parity",
        "refresh": "custom_observable_PV_policy_not_native_refresh_SGD",
        "status": "pass",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = checks()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
