"""Numerical contracts of the reusable lifecycle stages.

Each workflow function is compared on CPU with the historical CIFAR code it
generalizes, using a random-initialized ResNet-32, synthetic cohorts and
synthetic AIHWKit-shaped OM populations.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.cifar_crossbar import full_epoch_runtime, sweep_runtime
from experiments.cifar_crossbar.closed_loop_lr_updates import UncappedClosedLoopAdam
from experiments.cifar_crossbar.devices import PulseWriter, clone_cpu, make_plant, prepare_population
from experiments.cifar_crossbar.model import state_hash
from experiments.cifar_crossbar.open_loop_updates import (
    UncappedOpenLoopAdam,
    apply_pulse_counts,
    nominal_reset_counts,
)
from experiments.cifar_crossbar.pcm_reference import PcmInferenceNoise
from experiments.cifar_crossbar.sweep_config import default_config as sweep_config
from experiments.cifar_crossbar.sweep_config import parse_config as parse_sweep
from experiments.cifar_crossbar.sweep_devices import LayeredPcmArray, combined_om, make_array
from workflow import devices, hwa, onchip, program_verify
from workflow.lifecycle import DefectCase, parse_lifecycle
from workflow_helpers import (
    GMAX,
    LIFECYCLES,
    NOMINAL,
    Store,
    deployed,
    lifecycle,
    network_for,
    om_populations,
    om_raw,
    synthetic_cache,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


# --- (a) devices and (iii) program-and-verify --------------------------------------


@pytest.mark.parametrize("case", [NOMINAL, GMAX, DefectCase("random", 200000)], ids=lambda c: c.label)
def test_pcm_fresh_array_matches_historical_layered_endpoints(case):
    lc = lifecycle(network__analog_convolutions=2)
    _, network = network_for(2)
    target = network.q.detach()
    array, report = deployed(lc, network, target, case, None)
    legacy = LayeredPcmArray(target, network.layout, "cifar10", case.sampler_kind, case.rate, 271001, 281001)
    assert torch.equal(array.read(), legacy.read())
    assert array.cost() == legacy.cost() and report["array_reprogram_calls"] == 1
    assert array.faults.receipt() == legacy.faults.receipt()
    array.program(target * 0.5)
    legacy.program(target * 0.5)
    assert torch.equal(array.read(), legacy.read()) and array.cost() == legacy.cost()


def test_pcm_drift_matches_transcribed_aihwkit_inference_law():
    _, network = network_for(2)
    q = network.q.detach()
    q0_ref, qt_ref = PcmInferenceNoise(network, 123, "cpu", time_seconds=3600.0).realization(q)
    generator = torch.Generator().manual_seed(123)
    target = torch.stack((q.clamp_min(0), (-q).clamp_min(0))) * 25.0
    relative = target / 25.0
    sigma = 0.26348 + 1.965 * relative - 1.1731 * relative.square()
    programmed = (target + sigma * torch.randn(target.shape, generator=generator)).clamp_min(0)
    observed = program_verify.pcm_drift(programmed, relative, generator, 3600.0)
    q0 = (programmed[0] - programmed[1]) / 25.0
    qt = program_verify.tile_compensation(q0, (observed[0] - observed[1]) / 25.0, network.layout, network.tile_size)
    assert torch.equal(q0, q0_ref) and torch.equal(qt, qt_ref)


def test_relaxed_pcm_array_holds_one_realization_per_programming_event():
    lc = lifecycle("cifar10-pcm-conv4-drift-smoke", network__analog_convolutions=0)
    _, network = network_for(0)
    target = network.q.detach()
    case = DefectCase("gmax", 200000)
    array, report = deployed(lc, network, target, case, None)
    first = array.read()
    assert torch.equal(first, array.read()), "reads must not redraw relaxation noise"
    assert not torch.equal(first, array.programmed_read())
    assert report["relaxation"] == "pcm_drift" and "programmed_residual_rms" in report
    array.verify_permanence()
    replay = program_verify.fresh(lc, network.layout, 271001, case, None, "cpu")
    devices.load_array_state(replay, devices.array_state(array, "pcm"), "pcm")
    assert torch.equal(replay.read(), first)
    array.program(target)
    replay.program(target)
    assert not torch.equal(array.read(), first) and torch.equal(array.read(), replay.read())
    # Stuck devices keep their failed conductance through relaxation.
    stuck = array.faults.mask.any(0)
    assert torch.equal(array.read()[stuck & array.faults.mask.all(0)], torch.zeros(int((stuck & array.faults.mask.all(0)).sum())))


def test_om_fresh_array_and_closed_loop_match_historical_deployment():
    lc = lifecycle("cifar10-om-conv4-smoke", network__analog_convolutions=0)
    _, network = network_for(0)
    target = network.q.detach().clone()
    populations = om_populations(network.layout, (271001,))
    array, report = deployed(lc, network, target, GMAX, populations)
    spec = SimpleNamespace(backend="om", dataset="cifar10", array_seed=271001)
    legacy = make_array(network, target, spec, "gmax", 50000, populations)
    assert torch.equal(array.read(), legacy.read()) and array.cost() == legacy.cost()
    assert {key: report[key] for key in legacy.deployment} == legacy.deployment
    identity = devices.array_identity(array, "om")
    assert identity["failed_devices"] == int(array.source["hidden"]["injected_faults"].sum()) > 0


def test_om_open_loop_programming_matches_nominal_reset_counts():
    doc = json.loads((LIFECYCLES / "cifar10-om-conv4-smoke.json").read_text())
    doc["network"]["analog_convolutions"] = 0
    doc["program_verify"] = {"method": "open_loop_nominal", "relaxation": {"model": "none"}}
    lc = parse_lifecycle(doc)
    _, network = network_for(0)
    target = network.q.detach().clone()
    populations = om_populations(network.layout, (271001,))
    array, report = deployed(lc, network, target, NOMINAL, populations)
    legacy = make_plant(
        combined_om(populations, network.layout, "cifar10", 271001, 281001, "open", 0.0), 281001, "cpu"
    )
    counts = nominal_reset_counts(target, legacy.nominal_step)
    apply_pulse_counts(legacy.port(), torch.ones_like(target, dtype=torch.int8), counts)
    assert torch.equal(array.read(), legacy.read())
    assert report["verify_reads"] == 0 and report["pulses"] == int(counts.sum())


def test_om_endpoint_characterization_matches_historical_kernel_tables():
    lc = lifecycle(
        "cifar10-om-conv4-smoke",
        devices__characterization={"bins": 5, "samples": 16, "fit_seed": 231001, "heldout_seed": 231002},
    )
    populations = om_populations((), (), kernel_size=80)
    kernels = program_verify.characterize_endpoints(populations, lc, "cpu")
    from experiments.cifar_crossbar.sweep_devices import prepare_population as legacy_prepare
    from experiments.cifar_crossbar.devices import program as legacy_program
    from experiments.cifar_crossbar.full_epoch_runtime import om_population

    grid = torch.linspace(-1, 1, 5).repeat_interleave(16)
    for role in ("fit", "heldout"):
        raw = populations["kernel_" + role]
        for kind in ("healthy", "open", "gmax", "random"):
            population = legacy_prepare(raw) if kind == "healthy" else om_population(raw, kind, 1.0, raw["seed"])
            plant = make_plant(population, raw["seed"] + 10000, "cpu")
            legacy_program(plant.port(), grid, tolerance=plant.nominal_step * 0.5, maximum_pulses=128)
            assert torch.equal(kernels["tables"][role][kind], (plant.read() - grid).reshape(5, 16))
    assert set(kernels["gates"]) == {"healthy", "open", "gmax", "random"}


# --- (iv) on-chip update laws -------------------------------------------------------


def _plant(seed=42):
    return make_plant(prepare_population(om_raw(64, 7)), seed, "cpu")


def _drive(writer, steps=6):
    generator = torch.Generator().manual_seed(3)
    for _ in range(steps):
        writer.step(torch.randn(64, generator=generator))


@pytest.mark.parametrize("pulses,cap", [(1, 640), (3, 5), (1, None)])
def test_closed_loop_pulse_adam_matches_historical_writers(pulses, cap):
    plant, legacy_plant = _plant(), _plant()
    tolerance = plant.nominal_step * 0.5
    writer = onchip.ClosedLoopPulseAdam(
        plant.port(), learning_rate=0.05, tolerance=tolerance, pulses_per_update=pulses, cap=cap
    )
    if cap is None:
        legacy = UncappedClosedLoopAdam(legacy_plant.port(), legacy_plant.read(), learning_rate=0.05)
    else:
        legacy = PulseWriter(
            legacy_plant.port(), method="closed_loop", learning_rate=0.05, tolerance=tolerance,
            cap=cap, maximum_pulses=pulses,
        )
    _drive(writer)
    _drive(legacy)
    assert torch.equal(plant.read(), legacy_plant.read()) and plant.cost() == legacy_plant.cost()
    assert torch.equal(writer.count, legacy.count) and writer.verify_reads == legacy.verify_reads


@pytest.mark.parametrize("cap", [3, None])
def test_open_loop_pulse_adam_matches_historical_writers(cap):
    plant, legacy_plant = _plant(), _plant()
    writer = onchip.OpenLoopPulseAdam(plant.port(), learning_rate=0.08, cap=cap, seed=9)
    if cap is None:
        legacy = UncappedOpenLoopAdam(legacy_plant.port(), legacy_plant.read(), learning_rate=0.08, seed=9)
    else:
        legacy = PulseWriter(
            legacy_plant.port(), method="open_loop", learning_rate=0.08, tolerance=0.0, cap=cap, seed=9
        )
    _drive(writer)
    _drive(legacy)
    assert torch.equal(plant.read(), legacy_plant.read())
    assert torch.equal(writer.count, legacy.count) and writer.probability_clipped == legacy.probability_clipped


ARMS = {
    "calibration": ("hold", True),
    "rewrite": ("rewrite", True),
    "onchip_weights": ("learn", False),
    "onchip_calibration": ("learn", True),
}


@pytest.mark.parametrize("technology", ["pcm", "om"])
@pytest.mark.parametrize("method", sorted(ARMS))
def test_onchip_arm_matches_historical_full_epoch_trajectory(tmp_path, technology, method):
    name = "cifar10-pcm-conv4-smoke" if technology == "pcm" else "cifar10-om-conv4-smoke"
    lc = lifecycle(name, network__analog_convolutions=0, onchip__epochs=2, onchip__batch_size=8)
    teacher, network = network_for(0)
    cache = synthetic_cache(teacher, network)
    populations = om_populations(network.layout, (271001,)) if technology == "om" else None
    target = network.q.detach().clone()
    pristine = clone_cpu(network.state_dict())
    arm = next(a for a in lc.onchip.arms if a.arm_id == method)

    array, _ = deployed(lc, network, target, GMAX, populations)
    result = onchip.train_arm(
        lifecycle=lc, arm=arm, network=network, array=array, target=target, cache=cache,
        store=Store(tmp_path / "workflow"), label="gmax", device="cpu",
    )
    final_state = clone_cpu(network.state_dict())

    network.load_state_dict(pristine)
    legacy_array, _ = deployed(lc, network, target, GMAX, populations)
    spec = SimpleNamespace(
        backend=technology, learning_rate=lc.onchip.learning_rate,
        calibration_lr=lc.onchip.calibration_learning_rate, epochs=2, data_seed=lc.data.data_seed,
        batch_size=8, seed=lc.onchip.seed, verify_step_fraction=0.5, recovery_pulses=640, update_pulses=1,
    )
    store = Store(tmp_path / "legacy")
    directory = store.run_dir / "gmax"
    directory.mkdir()
    (directory / "p0.pt").write_bytes(b"p0")
    expected = full_epoch_runtime.trajectory(network, legacy_array, target, cache, spec, method, store, directory)
    assert len(result["curve"]) == len(expected["curve"]) == 2
    for mine, theirs in zip(result["curve"], expected["curve"]):
        for key in ("development", "test") + (("persistent_test",) if technology == "om" else ()):
            assert mine[key] == theirs[key]
        assert mine["train_loss_mean"] == theirs["teacher_kl_train_mean"]
    assert torch.equal(array.read(), legacy_array.read())
    assert state_hash(final_state) == state_hash(clone_cpu(network.state_dict()))


@pytest.mark.parametrize("technology", ["pcm", "om"])
def test_hold_arm_without_calibration_is_a_no_write_control(tmp_path, technology):
    name = "cifar10-pcm-conv4-smoke" if technology == "pcm" else "cifar10-om-conv4-smoke"
    lc = lifecycle(name, network__analog_convolutions=0)
    teacher, network = network_for(0)
    populations = om_populations(network.layout, (271001,)) if technology == "om" else None
    target = network.q.detach().clone()
    array, _ = deployed(lc, network, target, GMAX, populations)
    before, cost = array.read(), array.cost()
    result = onchip.train_arm(
        lifecycle=lc, arm=lc.onchip.arms[0], network=network, array=array, target=target,
        cache=synthetic_cache(teacher, network), store=Store(tmp_path), label="gmax", device="cpu",
    )
    assert result["curve"] == [] and torch.equal(array.read(), before) and array.cost() == cost


# --- (ii) hardware-aware training ---------------------------------------------------


def test_hwa_arm_matches_historical_sweep_candidate_fit(tmp_path):
    teacher, network = network_for(4)
    cache = synthetic_cache(teacher, network, n=16)
    pristine = clone_cpu(network.state_dict())
    spec = parse_sweep(
        sweep_config(
            max_examples=128, hwa_epochs=1, extension_epochs=0, validation_every=1,
            fault_rates_ppm=(0, 50000), array_seed=271001, endpoint_seed=281001,
            candidate_noise=2.0, candidate_learning_rate=0.0003,
        )
    )
    store = Store(tmp_path / "legacy")
    sweep_runtime.fit_candidate(teacher, network, cache, spec, None, {}, store, {})
    legacy = torch.load(store.run_dir / "checkpoints/fit.pt", weights_only=True)

    network.load_state_dict(pristine)
    cases = [{"kind": "none", "rate_ppm": 0}] + [{"kind": k, "rate_ppm": 50000} for k in ("open", "gmax", "random")]
    lc = lifecycle(
        hwa__learning_rate=0.0003,
        hwa__arms=[{"id": "noise_hwa", "method": "hwa", "noise_strength": 2.0, "corruption": None,
                    "selection_cases": cases}],
    )
    outcome = hwa.train_source(
        lifecycle=lc, arm=lc.hwa.arms[0], network=network, teacher=teacher, cache=cache,
        device_bundle={"populations": None, "kernels": None}, store=Store(tmp_path / "workflow"), device="cpu",
    )
    best = legacy["best"]["noise_hwa"]
    assert outcome["fit"]["selected_epoch"] == best["epoch"]
    assert state_hash(outcome["model"]) == state_hash(best["model"])
    assert [h["teacher_kl"] for h in outcome["fit"]["history"]] == [
        h["scores"]["noise_hwa"] for h in legacy["fit"]["history"]
    ]
    assert outcome["fit"]["status"] == "complete"
    # Beyond selection: the trained epoch-one master itself is identical.
    progress = torch.load(store.run_dir / "checkpoints/progress.pt", weights_only=True)
    assert progress["epoch"] == 1 and state_hash(clone_cpu(network.state_dict())) == state_hash(progress["model"])


def test_corruption_aware_om_hwa_uses_characterized_kernels(tmp_path):
    lc = lifecycle("cifar10-om-conv4-smoke", network__analog_convolutions=0, hwa__batch_size=8)
    teacher, network = network_for(0)
    cache = synthetic_cache(teacher, network)
    generator = torch.Generator().manual_seed(0)
    tables = {kind: 0.01 * torch.randn(41, 8, generator=generator) for kind in ("healthy", "open", "gmax", "random")}
    bundle = {"populations": om_populations(network.layout, (211001,)),
              "kernels": {"adequate": True, "tables": {"fit": tables}}}
    arm = lc.hwa.arm("cdt_gmax")
    outcome = hwa.train_source(
        lifecycle=lc, arm=arm, network=network, teacher=teacher, cache=cache, device_bundle=bundle,
        store=Store(tmp_path), device="cpu",
    )
    fit = outcome["fit"]
    assert fit["status"] == "complete" and sum(fit["rate_histogram"].values()) == 3
    assert [row["case"] for row in fit["history"][0]["cases"]] == ["nominal", "gmax_50000ppm"]
    digital = hwa.train_source(
        lifecycle=lc, arm=lc.hwa.arm("digital"), network=network, teacher=teacher, cache=cache,
        device_bundle=bundle, store=Store(tmp_path), device="cpu",
    )
    assert digital["fit"]["method"] == "none"
    with pytest.raises(ValueError, match="adequate"):
        hwa.train_source(
            lifecycle=lc, arm=arm, network=network, teacher=teacher, cache=cache,
            device_bundle={**bundle, "kernels": {"adequate": False, "tables": {"fit": tables}}},
            store=Store(tmp_path), device="cpu",
        )


def test_restore_reproduces_saved_onchip_state(tmp_path):
    lc = lifecycle("cifar10-pcm-conv4-drift-smoke", network__analog_convolutions=0, onchip__batch_size=8)
    teacher, network = network_for(0)
    cache = synthetic_cache(teacher, network)
    target = network.q.detach().clone()
    source = clone_cpu(network.state_dict())
    array, _ = deployed(lc, network, target, GMAX, None)
    arm = lc.onchip.arms[-1]
    result = onchip.train_arm(
        lifecycle=lc, arm=arm, network=network, array=array, target=target, cache=cache,
        store=Store(tmp_path), label="gmax", device="cpu",
    )
    final = onchip.evaluations(network, array, cache, lc, "cpu")
    state = copy.deepcopy(result["state"])
    replay = program_verify.fresh(lc, network.layout, 271001, GMAX, None, "cpu")
    onchip.restore(network, replay, source, state, "pcm")
    assert onchip.evaluations(network, replay, cache, lc, "cpu") == final
