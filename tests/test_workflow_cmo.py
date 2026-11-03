"""CMO ReRAM technology: device law, hierarchical defects, HWA noise, schema and a full OPT lifecycle.

The device law is AIHWKit 1.1.0 ``ReRamCMONoiseModel`` with the 2 % P&V fit
(docs/reram_cmo_device.md); stuck devices follow a die/array hierarchy with an
exact fault total per MLP; HWA injects declared constant-sigma Gaussian weight
noise. A tiny random OPT stands in for OPT-125M; the CMO model is analytic, so
no AIHWKit interpreter is needed.
"""

import json
from pathlib import Path
import re

import pytest
import torch

from experiments.cifar_crossbar.model import tensor_hash
from experiments.cifar_crossbar.runtime import load
from experiments.schema import ConfigError
from training import reram_cmo
from training.reram_cmo import CmoMatrix
from workflow import deployment, hwa, program_verify
from workflow import devices as device_stage
from workflow.__main__ import STATE_BYTES_PER_WEIGHT
from workflow.__main__ import main as workflow_main
from workflow.__main__ import summary
from workflow.lifecycle import CMO_ACCEPTANCE_PERCENT, DefectCase, parse_lifecycle, provenance, stage_selections, verify_provenance
from workflow.networks.opt_mlp import MatrixSpec, StudentInput
from test_workflow_opt import changed, tiny_teacher, write_corpus
from workflow_helpers import lifecycle_document, run_lifecycle

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples/lifecycles/opt125m-cmo-mlp4-template.json"
NOMINAL = DefectCase("none", 0, hierarchical=True)
NAMESPACE = "facebook/opt-125m"


def scenario(kind="random", ppm=50000, die=None, array=None):
    return DefectCase(kind, ppm, die, array, hierarchical=True)


def case_doc(kind="none", ppm=0, die=None, array=None):
    return {"kind": kind, "rate_ppm": ppm, "die_concentration": die, "array_concentration": array}


def mlp_layout(layers, ffn, hidden):
    """fc1 [ffn, hidden] and fc2 [hidden, ffn] per decoder layer, as in the OPT family."""

    layout, offset = [], 0
    for layer in layers:
        for name, shape in (("fc1", (ffn, hidden)), ("fc2", (hidden, ffn))):
            layout.append(MatrixSpec(f"decoder.layers.{layer}.{name}", shape, offset, shape[0] * shape[1], 1.0))
            offset += shape[0] * shape[1]
    return layout


def template():
    return json.loads(TEMPLATE.read_text())


def same_stream(generator):
    clone = torch.Generator()
    clone.set_state(generator.get_state())
    return clone


# --- device law --------------------------------------------------------------------


def test_programming_sigma_is_the_aihwkit_two_percent_fit():
    assert reram_cmo.PROGRAMMING_SIGMA[2.0] == (0.0112185391, 0.01129027418)
    assert reram_cmo.programming_sigma(50.0, 2.0) == pytest.approx(0.0112185391 + 50 * 0.01129027418)
    assert round(reram_cmo.programming_sigma(50.0, 2.0), 3) == 0.576
    assert (reram_cmo.G_MIN_US, reram_cmo.G_MAX_US) == (9.0, 88.199997)
    assert set(reram_cmo.PROGRAMMING_SIGMA) == set(CMO_ACCEPTANCE_PERCENT)


def test_coefficients_match_installed_aihwkit():
    reram = pytest.importorskip("aihwkit.inference.noise.reram")
    for acceptance, coefficients in reram_cmo.PROGRAMMING_SIGMA.items():
        model = reram.ReRamCMONoiseModel(acceptance_range=acceptance)
        assert model.acceptance_range == acceptance
        assert tuple(model.coeff_dict[acceptance]) == coefficients
        assert (model.g_min, model.g_max) == (reram_cmo.G_MIN_US, reram_cmo.G_MAX_US)
    # The default acceptance_range=2e-2 is not a table key and silently selects the 0.2 % fit.
    assert reram.ReRamCMONoiseModel().acceptance_range == 0.2


def test_weights_read_as_active_minus_reference():
    q = torch.tensor([-1.0, 0.0, 1.0])
    active = reram_cmo.to_conductance(q)
    assert torch.allclose(active, torch.tensor([9.0, (9.0 + 88.199997) / 2, 88.199997]))
    assert torch.allclose(reram_cmo.to_weight(active, torch.full_like(active, reram_cmo.MID_US)), q, atol=1e-6)
    # A reference stuck at G_max shifts its weight by -1.
    assert torch.allclose(reram_cmo.to_weight(active, torch.full_like(active, reram_cmo.G_MAX_US)), q - 1, atol=1e-6)


def test_programming_draws_unclamped_untruncated_held_endpoints_for_both_devices():
    n = 200_000
    matrix = CmoMatrix(n, 5, 2.0, 1.0, "cpu")
    mid_sigma = reram_cmo.programming_sigma(reram_cmo.MID_US, 2.0)
    for q in (-1.0, 0.0, 1.0):
        matrix.program(torch.full((n,), q))
        target = float(reram_cmo.to_conductance(torch.tensor(q)))
        sigma = reram_cmo.programming_sigma(target, 2.0)
        active, reference = matrix.g
        assert abs(float(active.mean()) - target) < 5 * sigma / n**0.5
        assert float(active.std()) == pytest.approx(sigma, rel=0.01)
        assert abs(float(reference.mean()) - reram_cmo.MID_US) < 5 * mid_sigma / n**0.5
        assert float(reference.std()) == pytest.approx(mid_sigma, rel=0.01)
        # The read carries both devices' errors.
        read = (sigma**2 + mid_sigma**2) ** 0.5 / reram_cmo.HALF_RANGE_US
        assert float(matrix.read().std()) == pytest.approx(read, rel=0.01)
    # Neither truncated to the 2 % acceptance window nor clamped to the range.
    assert float(((matrix.g[0] - 88.199997).abs() > 0.02 * 88.199997).float().mean()) > 0.05
    assert 0.48 < matrix.out_of_range() / n < 0.52
    assert torch.equal(matrix.read(), matrix.read())
    assert matrix.cost() == {
        "array_program_calls": 3, "device_programs": 6 * n, "device_writes": 0, "stuck_write_attempts": 0, "verify_reads": 0
    }
    with pytest.raises(ValueError, match=re.escape("[-1, 1]")):
        matrix.program(torch.full((n,), 1.5))


def test_idealized_write_adds_noise_around_current_state_plus_update():
    matrix = CmoMatrix(6, 3, 2.0, 1.0, "cpu")
    matrix.program(torch.linspace(-0.5, 0.5, 6))
    before, stream = matrix.g.clone(), same_stream(matrix.generator)
    delta = torch.tensor([0.0, 0.01, -0.02, 0.0, 0.03, 0.001])
    matrix.write(delta)
    requested = before[0] + delta * reram_cmo.HALF_RANGE_US
    noise = torch.randn(6, generator=stream)
    drawn = requested + 1.0 * reram_cmo.programming_sigma(requested, 2.0) * noise
    assert torch.equal(matrix.g[0], torch.where(delta != 0, drawn, before[0]))
    # Reference devices are never written.
    assert torch.equal(matrix.g[1], before[1])
    assert matrix.writes.tolist() == [1, 2, 2, 1, 2, 2]
    assert matrix.cost() == {
        "array_program_calls": 1, "device_programs": 12, "device_writes": 4, "stuck_write_attempts": 0, "verify_reads": 0
    }
    # No verify and no retry: a zero update leaves every earlier write error in place.
    held = matrix.g.clone()
    matrix.write(torch.zeros(6))
    assert torch.equal(matrix.g, held) and matrix.cost()["device_writes"] == 4


def test_matrix_state_replays_exactly():
    source = CmoMatrix(16, 9, 2.0, 1.0, "cpu")
    source.program(torch.linspace(-1, 1, 16))
    source.write(torch.full((16,), 0.01))
    copy = CmoMatrix(16, 1, 2.0, 1.0, "cpu")
    copy.load_state_dict(source.state_dict())
    for matrix in (source, copy):
        matrix.write(torch.full((16,), -0.02))
    assert torch.equal(source.g, copy.g) and torch.equal(source.writes, copy.writes)
    with pytest.raises(ValueError, match="acceptance"):
        CmoMatrix(16, 9, 0.2, 1.0, "cpu").load_state_dict(source.state_dict())


def test_stuck_devices_persist_through_programs_and_writes():
    # Active device 0 stuck at G_min; reference device 3 stuck at G_max.
    mask = torch.zeros((2, 6), dtype=torch.bool)
    mask[0, 0] = mask[1, 3] = True
    stuck = torch.zeros((2, 6))
    stuck[0, 0], stuck[1, 3] = reram_cmo.G_MIN_US, reram_cmo.G_MAX_US
    matrix, clean = CmoMatrix(6, 3, 2.0, 1.0, "cpu", (mask, stuck)), CmoMatrix(6, 3, 2.0, 1.0, "cpu")
    for m in (matrix, clean):
        m.program(torch.zeros(6))
    # Defects do not shift the healthy devices' noise stream.
    assert torch.equal(matrix.g[~mask], clean.g[~mask])
    read = matrix.read()
    assert read[0] == (reram_cmo.G_MIN_US - matrix.g[1, 0]) / reram_cmo.HALF_RANGE_US
    assert read[3] == (matrix.g[0, 3] - reram_cmo.G_MAX_US) / reram_cmo.HALF_RANGE_US
    matrix.write(torch.full((6,), 0.01))
    reference = matrix.g[1].clone()
    matrix.program(torch.zeros(6), reference=False)
    assert torch.equal(matrix.g[mask], stuck[mask]) and torch.equal(matrix.g[1], reference)
    assert matrix.writes.tolist() == [0, 3, 3, 3, 3, 3]
    assert matrix.cost() == {
        "array_program_calls": 2, "device_programs": 15, "device_writes": 5, "stuck_write_attempts": 4, "verify_reads": 0
    }
    matrix.verify_permanence()
    state = matrix.state_dict()
    with pytest.raises(ValueError, match="permanent"):
        CmoMatrix(6, 3, 2.0, 1.0, "cpu").load_state_dict(state)
    matrix.g[1, 3] = 50.0
    with pytest.raises(RuntimeError, match="permanent"):
        matrix.verify_permanence()


# --- hierarchical defects ----------------------------------------------------------


def test_fault_totals_are_exact_per_group_and_dies_stay_within_a_group():
    # Tiny MLPs: 8 + 8 tiles of 8x8 weights per layer, two 8x8 physical arrays per tile
    # (active and reference rows of 4 inputs); dies of 12 leave a partial third die.
    layout = mlp_layout((1, 2), 32, 16)
    for case in (scenario(), scenario(array=50.0), scenario(die=50.0), scenario(die=50.0, array=50.0)):
        faults = device_stage.hierarchical_faults(layout, 8, 12, case, 271001, NAMESPACE)
        receipt = faults["receipt"]
        assert [g["group"] for g in receipt["groups"]] == ["decoder.layers.1", "decoder.layers.2"]
        for group in receipt["groups"]:
            assert group["devices"] == 2 * 1024 and group["failed_devices"] == round(0.05 * 2 * 1024)
            assert [die["arrays"] for die in group["dies"]] == [12, 12, 8]
            assert {a["devices"] for a in group["arrays"]} == {64}
            assert sum(a["failed_devices"] for a in group["arrays"]) == group["failed_devices"]
        masks = [mask for mask, _ in faults["matrices"].values()]
        assert sum(int(mask.sum()) for mask in masks) == receipt["failed_devices"] == 2 * 102
        assert receipt["failed_active"] + receipt["failed_reference"] == 2 * 102
        assert receipt["failed_active"] > 0 and receipt["failed_reference"] > 0


def test_a_weights_two_devices_share_one_physical_array():
    layout = mlp_layout((1,), 32, 16)
    for array in device_stage.physical_arrays(layout, 8):
        size = 32 * 16 if array["matrix"].endswith("fc1") else 16 * 32
        active, reference = array["indices"][:32], array["indices"][32:]
        assert len(array["indices"]) == 64 and (active < size).all() and (reference == active + size).all()


def test_scenarios_control_array_and_die_variation_at_a_fixed_total():
    # One group of 1600 physical arrays (32x32 devices), 40 dies of 40 arrays.
    layout = mlp_layout((11,), 640, 640)

    def draw(**concentrations):
        receipt = device_stage.hierarchical_faults(layout, 32, 40, scenario(**concentrations), 271001, NAMESPACE)
        [group] = receipt["receipt"]["groups"]
        rates = [[a["rate"] for a in group["arrays"] if a["die"] == d] for d in range(len(group["dies"]))]
        fractions = [a["failed_devices"] / a["devices"] for a in group["arrays"]]
        return group, rates, float(torch.tensor(fractions).std())

    independent, rates, spread = draw()
    assert all(rate == 0.05 for die in rates for rate in die) and spread < 0.01
    arrays, rates, spread = draw(array=118.0)
    assert all(die["rate"] == 0.05 for die in arrays["dies"]) and 0.015 < spread < 0.026
    dies, rates, spread = draw(die=118.0)
    assert all(len(set(die)) == 1 for die in rates)
    assert 0.01 < float(torch.tensor([die["rate"] for die in dies["dies"]]).std()) < 0.03
    both, rates, spread = draw(die=118.0, array=118.0)
    assert all(len(set(die)) == len(die) for die in rates) and spread > 0.02
    totals = {g["failed_devices"] for g in (independent, arrays, dies, both)}
    assert totals == {round(0.05 * 2 * 2 * 640 * 640)}


@pytest.mark.parametrize("kind", ["open", "gmin", "gmax", "random"])
def test_stuck_values_follow_the_declared_kind(kind):
    faults = device_stage.hierarchical_faults(mlp_layout((11,), 64, 64), 16, 4, scenario(kind), 271001, NAMESPACE)
    values = torch.cat([stuck[mask] for mask, stuck in faults["matrices"].values()])
    assert len(values) == round(0.05 * 2 * 2 * 64 * 64)
    if kind == "random":
        assert bool(((values >= reram_cmo.G_MIN_US) & (values <= reram_cmo.G_MAX_US)).all())
        assert len(set(values.tolist())) == len(values)
    else:
        assert set(values.tolist()) == {float(torch.tensor(device_stage.CMO_STUCK_US[kind]))}


def test_fault_maps_are_deterministic_and_bound_per_group():
    case = scenario(die=118.0, array=118.0)
    first = device_stage.hierarchical_faults(mlp_layout((1, 2), 32, 16), 8, 12, case, 271001, NAMESPACE)
    again = device_stage.hierarchical_faults(mlp_layout((1, 2), 32, 16), 8, 12, case, 271001, NAMESPACE)
    other = device_stage.hierarchical_faults(mlp_layout((1, 2), 32, 16), 8, 12, case, 271002, NAMESPACE)
    alone = device_stage.hierarchical_faults(mlp_layout((2,), 32, 16), 8, 12, case, 271001, NAMESPACE)
    assert first["receipt"]["identity_sha256"] == again["receipt"]["identity_sha256"]
    assert first["receipt"]["identity_sha256"] != other["receipt"]["identity_sha256"]
    # A group's defects do not depend on which other groups are analog.
    for name, (mask, stuck) in alone["matrices"].items():
        assert torch.equal(mask, first["matrices"][name][0]) and torch.equal(stuck, first["matrices"][name][1])


# --- arrays and P&V ----------------------------------------------------------------


def small(**changes):
    doc = changed(template(), "network.analog_decoder_layers", 2)
    doc = changed(doc, "deployment.tile_size", 8)
    doc = changed(doc, "defects.arrays_per_die", 12)
    for dotted, value in changes.items():
        doc = changed(doc, dotted, value)
    return parse_lifecycle(doc)


def test_arrays_bind_each_matrix_to_its_own_stream_and_report_programming():
    lc = small()
    network = deployment.build_network(tiny_teacher(), lc)
    target = network.q.detach()
    arrays = [program_verify.fresh(lc, network.layout, seed, NOMINAL, None, "cpu") for seed in (271001, 271001, 271002)]
    reports = [program_verify.program(array, target, lc) for array in arrays]
    assert torch.equal(arrays[0].read(), arrays[1].read())
    assert not torch.equal(arrays[0].read(), arrays[2].read())
    assert reports[0]["device_programs"] == 2 * target.numel() and reports[0]["verify_reads"] == 0
    assert reports[0]["acceptance_percent"] == 2.0 and "out_of_range_devices" in reports[0]
    # Each matrix's realization depends on its own name, not on layer order.
    layout = tuple(reversed(network.layout))
    endpoint = lc.devices.endpoint_seed(271001)
    reordered = device_stage.CmoArray(layout, lc.network.binding_namespace, endpoint, 2.0, 1.0, "cpu")
    reordered.program(target)
    original = {m.name: child.g for m, child in zip(network.layout, arrays[0].children)}
    assert all(torch.equal(child.g, original[m.name]) for m, child in zip(layout, reordered.children))
    identity = device_stage.array_identity(arrays[0], "cmo")
    assert identity["failed_devices"] == 0 and identity["physical_devices"] == 2 * target.numel()


@pytest.mark.parametrize("event", ["none", "zero_write", "blocked_write", "verify", "restore"])
def test_compact_array_state_preserves_future_writes_and_costs(event):
    layout = [MatrixSpec("decoder.layers.0.fc1", (4, 4), 0, 16, 1.0)]
    target = torch.linspace(-0.5, 0.5, 16)
    mask = torch.zeros((2, 16), dtype=torch.bool)
    mask[0, 0] = True
    stuck = torch.full((2, 16), reram_cmo.G_MIN_US)

    def fresh():
        return device_stage.CmoArray(
            layout, NAMESPACE, 19, 2.0, 1.0, "cpu",
            {"matrices": {layout[0].name: (mask, stuck)}, "receipt": {}},
        )

    original = fresh()
    original.program(target)
    child = original.children[0]
    if event in ("zero_write", "blocked_write"):
        delta = torch.zeros_like(target)
        if event == "blocked_write":
            delta[0] = 0.01
        # Exercise the state owner directly, not only the array wrapper.
        child.write(delta)
    elif event == "verify":
        child.read_devices()
    elif event == "restore":
        child.load_state_dict(child.state_dict())

    saved = device_stage.array_state(original, "cmo")
    assert saved["kind"] == ("layered_cmo_deployment" if event == "none" else "layered_cmo")
    restored = fresh()
    device_stage.load_array_state(restored, saved, "cmo", target)
    for step in range(3):
        assert torch.equal(original.read(), restored.read())
        assert original.cost() == restored.cost()
        assert torch.equal(child.writes, restored.children[0].writes)
        assert torch.equal(child.generator.get_state(), restored.children[0].generator.get_state())
        delta = torch.full_like(target, 0.01 * (step + 1))
        original.write(delta)
        restored.write(delta)
    assert torch.equal(original.read(), restored.read())


def test_faulted_arrays_hold_stuck_devices_and_report_their_identity():
    lc = small()
    network = deployment.build_network(tiny_teacher(), lc)
    target = network.q.detach()
    case = scenario("gmax", die=118.0, array=118.0)
    array = program_verify.fresh(lc, network.layout, 271001, case, None, "cpu")
    report = program_verify.program(array, target, lc)
    identity = device_stage.array_identity(array, "cmo")
    failed = 2 * round(0.05 * 2 * 1024)
    assert identity["failed_devices"] == failed and identity["array_shape"] == [8, 8]
    assert identity["failed_active"] + identity["failed_reference"] == failed
    assert report["device_programs"] == 2 * target.numel() - failed and report["stuck_write_attempts"] == failed
    g = torch.cat([child.g for child in array.children], 1)
    mask = torch.cat([child.mask for child in array.children], 1)
    assert bool((g[mask] == reram_cmo.G_MAX_US).all())
    # Both devices stuck at G_max read as zero; a stuck active alone reads near +1.
    both = mask.all(0)
    assert bool((array.read()[both] == 0).all())
    assert bool(((array.read() - 1).abs()[mask[0] & ~mask[1]] < 0.1).all())
    array.verify_permanence()


def test_hwa_injects_constant_sigma_gaussian_weight_noise():
    lc = small(**{"hwa.noise_std": 0.02, "hwa.noise_scaling": "matrix_full_scale", "hwa.arms.1.noise_strength": 2.0})
    sample = hwa.endpoint_sampler(lc, lc.hwa.arm("standard_hwa"), {}, "cpu")
    n = 50_000
    q = torch.cat([torch.full((n,), -0.9), torch.zeros(n), torch.full((n,), 0.9)]).requires_grad_(True)
    first, second = sample(q, "open", 0.0), sample(q, "open", 0.0)
    assert not torch.equal(first, second)
    # One sigma (noise_std x noise_strength) for every weight, unlike the CMO programming law.
    for part in range(3):
        noise = (first - q).detach()[part * n : (part + 1) * n]
        assert float(noise.std()) == pytest.approx(0.04, rel=0.02) and abs(float(noise.mean())) < 0.001
    first.sum().backward()
    assert torch.equal(q.grad, torch.ones_like(q))


def test_hwa_noise_scales_with_each_output_channel_abs_max():
    # The Analog Foundation Models convention: sigma_j = noise_std * max_i |W_ji| for output channel j.
    layout = mlp_layout((11,), 400, 200)
    lc = small(**{"hwa.noise_std": 0.03})
    sample = hwa.endpoint_sampler(lc, lc.hwa.arm("standard_hwa"), {}, "cpu", layout=layout)
    rows = torch.linspace(0.1, 1.0, 400)[:, None].expand(400, 200)
    q = torch.cat([rows.reshape(-1), torch.full((200 * 400,), 0.5)])
    noise = sample(q, "random", 0.0) - q
    fc1 = noise[: 400 * 200].reshape(400, 200)
    # Dividing by each channel's abs-max leaves one standard deviation of 0.03.
    assert float((fc1 / rows).std()) == pytest.approx(0.03, rel=0.02)
    assert torch.allclose(fc1.std(1), 0.03 * rows[:, 0], rtol=0.3)
    assert float(noise[400 * 200 :].std()) == pytest.approx(0.03 * 0.5, rel=0.02)
    assert torch.allclose(hwa.channel_abs_max(q, layout)[: 400 * 200], rows.reshape(-1))


def test_streamed_epochs_visit_every_sequence_once_per_pass():
    # 10 sequences, 4 per epoch: passes end inside epochs 3 and 5.
    epochs = [hwa.stream_order(10, 4, epoch, 7, "cpu") for epoch in range(1, 6)]
    assert all(len(order) == 4 for order in epochs)
    stream = torch.cat(epochs)
    assert sorted(stream[:10].tolist()) == sorted(stream[10:20].tolist()) == list(range(10))
    assert not torch.equal(stream[:10], stream[10:20])
    # Each epoch is a pure function of its index, so a resumed run takes the same slice.
    assert torch.equal(hwa.stream_order(10, 4, 3, 7, "cpu"), epochs[2])
    assert not torch.equal(hwa.stream_order(10, 4, 1, 8, "cpu"), epochs[0])


def test_streamed_epochs_are_optional_and_leave_digests_unchanged():
    plain = small()
    assert plain.hwa.sequences_per_epoch is None and "sequences_per_epoch" not in plain.hwa.shared_dict()
    streamed = small(**{"hwa.sequences_per_epoch": 4096})
    assert streamed.hwa.shared_dict()["sequences_per_epoch"] == 4096
    assert parse_lifecycle(streamed.to_dict()) == streamed
    assert streamed.section_digest("hwa") != plain.section_digest("hwa")
    assert streamed.section_digest("data") == plain.section_digest("data")
    with pytest.raises(ConfigError, match="hwa.sequences_per_epoch"):
        small(**{"hwa.sequences_per_epoch": plain.data.hwa_sequences + 1})
    with pytest.raises(ConfigError, match="hwa.sequences_per_epoch"):
        small(**{"hwa.sequences_per_epoch": 0})


def test_onchip_can_select_existing_training_cohorts_without_changing_source_provenance():
    original = small()
    assert original.onchip.cohort_key == "training"
    assert "training_cohort" not in original.to_dict()["onchip"]
    expanded = small(**{"onchip.training_cohort": "hwa"})
    assert expanded.onchip.cohort_key == "hwa"
    assert parse_lifecycle(expanded.to_dict()) == expanded
    for kind in ("feature_cache", "device_bundle", "hwa_source", "deployment"):
        arm = "standard_hwa" if kind in {"hwa_source", "deployment"} else None
        verify_provenance(provenance(original, kind, hwa_arm=arm), expanded, kind, hwa_arm=arm)
    for forbidden in ("test", "development", "validation", "unknown", None):
        with pytest.raises(ConfigError, match="onchip.training_cohort"):
            small(**{"onchip.training_cohort": forbidden})


def test_full_hwa_cohort_recovery_uses_all_examples_and_replays(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    lc = small(**{
        "data.corpus": {"name": "synthetic", "sha256": digest},
        "data.sequence_length": 9, "data.max_examples": 0,
        "data.hwa_sequences": 12, "data.recovery_sequences": 4,
        "data.development_sequences": 4, "data.test_sequences": 4,
        "defects.cases": [case_doc("random", 50000)],
        "hwa.arms": [{"id": "digital", "method": "none"}],
        "onchip.training_cohort": "hwa", "onchip.batch_size": 4,
        "onchip.arms": [{"id": "none", "weights": "hold", "calibration": False},
                        {"id": "learn", "weights": "learn", "calibration": False}],
    })
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    run, result = results["onchip__digital__271001"]
    rows = json.loads((run / "measurements.json").read_text())
    assert [r["sequence_presentations"] for r in rows] == [0, 12]
    assert all(r["replay_exact"] for r in rows)
    assert rows[1]["cost"]["max_onchip_device_writes"] == 3
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    cohorts = [r for r in records if r.get("stage") == "onchip_cohort"]
    assert all(r["cohort"] == "hwa" and r["examples"] == 12 for r in cohorts)


def test_corruption_aware_hwa_sticks_active_and_reference_devices():
    q = torch.full((40_000,), 0.25)
    generator = torch.Generator().manual_seed(4)
    weights = reram_cmo.corrupt_weights(q, generator, "gmax", 0.5)
    # Healthy pair: q; stuck active: +1; stuck reference: q - 1; both stuck: 0.
    for value in (0.25, 1.0, -0.75, 0.0):
        assert float(torch.isclose(weights, torch.tensor(value), atol=1e-5).float().mean()) == pytest.approx(0.25, abs=0.02)
    assert torch.equal(reram_cmo.corrupt_weights(q, generator, "random", 0.0), q)
    random = reram_cmo.corrupt_weights(q, generator, "random", 1.0)
    assert bool((random.abs() <= 2.0).all()) and float(random.std()) == pytest.approx((2 / 3) ** 0.5, rel=0.05)
    lc = small(**{"hwa.arms.1.corruption": {"kind": "random", "rates_ppm": [30000]}})
    sample = hwa.endpoint_sampler(lc, lc.hwa.arm("standard_hwa"), {}, "cpu", layout=mlp_layout((10, 11), 32, 16))
    master = torch.zeros(2 * 2 * 32 * 16, requires_grad=True)
    corrupted = sample(master, "random", 0.03)
    changed = float((corrupted.detach().abs() > 0.2).float().mean())
    assert 0.03 < changed < 0.09
    corrupted.sum().backward()
    assert torch.equal(master.grad, torch.ones_like(master))


# --- schema ------------------------------------------------------------------------


def test_template_is_valid_and_round_trips():
    lc = parse_lifecycle(template())
    assert lc.devices.technology == "cmo" and lc.program_verify.acceptance_percent == 2.0
    assert lc.onchip.update_law == "cmo_idealized_write_adam"
    assert (lc.hwa.noise_model, lc.hwa.noise_std, lc.hwa.noise_scaling) == (
        "additive_gaussian_weight", 0.03, "output_channel_abs_max"
    )
    assert lc.deployment.encoding == "active_reference_pair"
    assert lc.defects.arrays_per_die == 48 and lc.deployment.tile_size == 256
    assert [case.label for case in lc.defects.cases] == [
        "nominal",
        "random_50000ppm",
        "random_50000ppm_a118",
        "random_50000ppm_d118",
        "random_50000ppm_d118_a118",
    ]
    assert parse_lifecycle(lc.to_dict()) == lc


def test_per_output_channel_scaling_maps_every_channel_to_full_scale():
    teacher = tiny_teacher()
    lc = small()
    assert lc.deployment.scaling == "per_output_channel_absmax"
    network = deployment.build_network(teacher, lc)
    for spec in network.layout:
        rows = network.q.detach()[spec.offset : spec.offset + spec.size].reshape(spec.shape).abs().amax(1)
        assert torch.allclose(rows, torch.ones_like(rows))
    tokens = torch.randint(0, 64, (3, 9), generator=torch.Generator().manual_seed(3))
    student, reference = network(tokens), teacher(tokens)[:, :-1]
    assert torch.allclose(student, reference, atol=1e-4)
    receipt = deployment.mapping_receipt(network, lc)
    assert receipt["scaling"] == "per_output_channel_absmax" and "channel_scales_sha256" in receipt
    assert "channel_scales" in network.state_dict()
    matrix = deployment.build_network(teacher, small(**{"deployment.scaling": "per_layer_absmax"}))
    assert "channel_scales" not in matrix.state_dict()
    with pytest.raises(ConfigError, match="deployment.scaling"):
        parse_lifecycle(
            changed(lifecycle_document("cifar10-pcm-conv4-smoke"), "deployment.scaling", "per_output_channel_absmax")
        )


def test_hierarchical_keys_belong_to_cmo_only():
    pcm = lifecycle_document("cifar10-pcm-conv4-smoke")
    with pytest.raises(ConfigError, match="defects fields"):
        parse_lifecycle(changed(pcm, "defects.arrays_per_die", 24))
    with pytest.raises(ConfigError, match=re.escape("defects.cases[0] fields")):
        parse_lifecycle(changed(pcm, "defects.cases.0.die_concentration", None))


@pytest.mark.parametrize("key", ["noise_std", "noise_scaling"])
def test_hwa_noise_keys_are_required_for_cmo_and_rejected_elsewhere(key):
    doc = template()
    del doc["hwa"][key]
    with pytest.raises(ConfigError, match="hwa fields"):
        parse_lifecycle(doc)
    with pytest.raises(ConfigError, match="hwa fields"):
        parse_lifecycle(changed(lifecycle_document("cifar10-pcm-conv4-smoke"), f"hwa.{key}", template()["hwa"][key]))


def test_a_deployment_only_lifecycle_has_no_onchip_stage(capsys):
    lc = parse_lifecycle(changed(template(), "onchip", None))
    assert lc.onchip is None and lc.to_dict()["onchip"] is None and parse_lifecycle(lc.to_dict()) == lc
    assert {s.kind for s in stage_selections(lc)} == {"prepare", "devices", "hwa", "deploy"}
    value = summary(lc)
    assert value["onchip"] is None and value["trajectories"] == 0 and value["deployed_cases"] == 2 * 5


@pytest.mark.parametrize(
    "dotted,value,message",
    [
        ("defects.cases", [case_doc(), {"kind": "gmax", "rate_ppm": 20000}], "defects.cases[1] fields"),
        ("defects.cases.1.kind", "stuck", "defects.cases[1].kind"),
        ("defects.cases.1.die_concentration", 0.0, "defects.cases[1].die_concentration"),
        ("defects.cases.0.array_concentration", 118, "null concentrations"),
        ("defects.arrays_per_die", 0, "defects.arrays_per_die"),
        ("defects.placement", "bernoulli_per_physical_device", "defects.placement"),
        ("hwa.arms.1.corruption", {"kind": "stuck", "rates_ppm": [20000]}, "hwa.arms[1].corruption.kind"),
        ("hwa.noise_scaling", "per_weight", "hwa.noise_scaling"),
        ("hwa.arms.1.selection_cases", [{"kind": "open", "rate_ppm": 1000}], "selection_cases[0] fields"),
        ("hwa.noise_model", "pcm_gaussian_endpoint", "hwa.noise_model"),
        ("hwa.noise_std", 0.0, "hwa.noise_std"),
        ("program_verify.relaxation", {"model": "pcm_drift", "seconds": 1.0, "compensation": False}, "relaxation.model"),
        ("program_verify.acceptance_percent", 1.0, "program_verify.acceptance_percent"),
        ("program_verify", {"method": "gaussian_endpoint", "relaxation": {"model": "none"}}, "program_verify fields"),
        ("program_verify.method", "closed_loop", "program_verify.method"),
        ("deployment.encoding", "single_device_midrange_reference", "deployment.encoding"),
        ("devices.policy", "repaired", "devices fields"),
        ("onchip.update_law", "pcm_endpoint_reprogramming", "onchip.update_law"),
    ],
)
def test_cmo_lifecycle_rejects_other_regimes(dotted, value, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        parse_lifecycle(changed(template(), dotted, value))


def test_describe_uses_the_cmo_state_estimate(capsys):
    assert workflow_main(["describe", str(TEMPLATE), "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["devices"]["technology"] == "cmo" and value["trajectories"] == 50
    assert value["onchip_state_gb_estimate"] == round(50 * 18_874_368 * STATE_BYTES_PER_WEIGHT["cmo"] / 1e9, 2)


# --- the full lifecycle ------------------------------------------------------------


@pytest.fixture
def cmo_inputs(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    teacher = tiny_teacher()
    monkeypatch.setattr(deployment, "load_teacher", lambda path, network, device: (teacher, "f" * 64))
    corpora = tmp_path / "corpora"
    monkeypatch.setenv("EBL_CORPUS_ROOT", str(corpora))
    weights = tmp_path / "teacher.pt"
    weights.write_bytes(b"synthetic teacher")
    return weights, write_corpus(corpora)


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        # Generator states are a fixed per-matrix overhead, not a per-weight cost.
        return sum(tensor_bytes(v) for k, v in value.items() if k != "rng")
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def test_opt_cmo_lifecycle_runs_end_to_end_with_matched_controls(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    lc = small(
        **{
            "data.corpus": {"name": "synthetic", "sha256": digest},
            "data.sequence_length": 9,
            "data.max_examples": 8,
            "defects.cases": [case_doc(), case_doc("random", 50000, 118.0, 118.0)],
            "hwa.batch_size": 4,
            "onchip.batch_size": 4,
        }
    )
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    assert results["devices"][1]["metrics"]["populations"] == {"kind": "analytic_cmo_endpoint", "sampled_cells": 0}
    # The tiny OPT's own count; lc.network.analog_weights describes OPT-125M.
    analog = json.loads((results["prepare"][0] / "mapping.json").read_text())["logical_weights"]
    assert analog == 2 * 2 * 16 * 32
    steps = lc.data.max_examples // lc.onchip.batch_size * lc.onchip.epochs
    arms = [arm.arm_id for arm in lc.onchip.arms]
    labels = [case.label for case in lc.defects.cases]
    assert labels == ["nominal", "random_50000ppm_d118_a118"]
    failed = {"nominal": 0, labels[1]: 2 * round(0.05 * 2 * 1024)}
    state_bytes = []
    for hwa_arm in lc.hwa.arms:
        _, deploy = results[f"deploy__{hwa_arm.arm_id}__271001"]
        for label in labels:
            programming = deploy["metrics"]["cases"][label]["programming"]
            assert programming["device_programs"] == 2 * analog - failed[label] and programming["verify_reads"] == 0
            assert programming["stuck_write_attempts"] == failed[label]
            assert programming["acceptance_percent"] == 2.0 and programming["out_of_range_devices"] >= 0
        run_dir, _ = results[f"onchip__{hwa_arm.arm_id}__271001"]
        rows = json.loads((run_dir / "measurements.json").read_text())
        assert [(r["case"], r["onchip_arm"]) for r in rows] == [(label, arm) for label in labels for arm in arms]
        for row in rows:
            assert row["replay_exact"] and row["identity"]["failed_devices"] == failed[row["case"]]
            stuck_active = row["identity"].get("failed_active", 0)
            healthy = analog - stuck_active
            assert "persistent_test" not in row["final"] and row["final"]["test"]["scored_tokens"] == 8 * 8
            cost = row["cost"]
            if row["onchip_arm"] == "none":
                assert row["final"] == row["initial"] and row["curve"] == [] and cost == {}
            elif row["weights"] == "hold":
                assert cost["device_writes"] == cost["device_programs"] == cost["touched_devices"] == 0
            elif row["weights"] == "rewrite":
                assert cost["device_programs"] == steps * healthy and cost["device_writes"] == 0
                assert cost["stuck_write_attempts"] == steps * stuck_active
                assert cost["max_onchip_device_writes"] == steps
            else:
                assert cost["device_programs"] == 0 and 0 < cost["device_writes"] <= steps * healthy
                assert 0 < cost["touched_devices"] <= healthy and cost["max_onchip_device_writes"] <= steps
                assert cost["stuck_write_attempts"] <= steps * stuck_active
                assert row["final_apparent_sha256"] != row["initial_apparent_sha256"]
            if row["onchip_arm"] != "none":
                assert cost["verify_reads"] == 0 and "out_of_range_devices" in cost
            state = load(run_dir / "checkpoints/onchip" / row["case"] / f"{row['onchip_arm']}.pt")
            state_bytes.append(tensor_bytes(state) / analog)
    # The describe estimate bounds the per-weight tensor storage of every trajectory.
    assert max(state_bytes) <= STATE_BYTES_PER_WEIGHT["cmo"]
    collected = tmp_path / "collected"
    assert workflow_main(["collect", str(tmp_path / "runs"), "--output", str(collected)]) == 0
    rows = json.loads((collected / "rows.json").read_text())
    assert len(rows) == len(lc.hwa.arms) * len(labels) * len(arms)
    assert {row["technology"] for row in rows} == {"cmo"}


def test_deployment_only_cdt_lifecycle_runs_and_collects_deployments(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    lc = small(
        **{
            "data.corpus": {"name": "synthetic", "sha256": digest},
            "data.sequence_length": 9,
            "data.max_examples": 8,
            "defects.cases": [case_doc(), case_doc("random", 10000), case_doc("random", 50000)],
            "hwa.batch_size": 4,
            "hwa.arms.1.corruption": {"kind": "random", "rates_ppm": [30000]},
            "hwa.arms.1.selection_cases": [case_doc(), case_doc("random", 30000)],
            "onchip": None,
        }
    )
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    assert not any(stage.startswith("onchip") for stage in results)
    run_dir, deploy = results["deploy__standard_hwa__271001"]
    assert deploy["metrics"]["cases"]["random_50000ppm"]["case"]["rate_ppm"] == 50000
    # Programmed-once CMO deployments are stored compactly and rebuilt from their target.
    bundle = load(run_dir / "checkpoints/deployment.pt")
    assert {case["array"]["kind"] for case in bundle["cases"].values()} == {"layered_cmo_deployment"}
    network = deployment.build_network(tiny_teacher(), lc)
    for case in lc.defects.cases:
        array = program_verify.fresh(lc, network.layout, 271001, case, None, "cpu")
        device_stage.load_array_state(array, bundle["cases"][case.label]["array"], "cmo", bundle["target"])
        assert tensor_hash(array.read()) == bundle["cases"][case.label]["programming"]["apparent_sha256"]
    _, fit = results["hwa__standard_hwa"]
    assert fit["metrics"]["fit"]["corruption"] == {"kind": "random", "rates_ppm": [30000]}
    collected = tmp_path / "collected"
    assert workflow_main(["collect", str(tmp_path / "runs"), "--output", str(collected)]) == 0
    assert json.loads((collected / "rows.json").read_text()) == []
    rows = json.loads((collected / "deployments.json").read_text())
    assert len(rows) == len(lc.hwa.arms) * len(lc.defects.cases)
    assert {(r["hwa_arm"], r["case"]) for r in rows} == {
        (arm.arm_id, case.label) for arm in lc.hwa.arms for case in lc.defects.cases
    }
    failed = {r["case"]: r["failed_devices"] for r in rows}
    assert failed == {"nominal": 0, "random_10000ppm": 2 * round(0.01 * 2048), "random_50000ppm": 2 * round(0.05 * 2048)}
    assert all("deployed_teacher_kl" in r and "clean_perplexity" in r and not r["duplicate_coverage"] for r in rows)


def test_streamed_hwa_takes_the_next_sequences_each_epoch(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    lc = small(
        **{
            "data.corpus": {"name": "synthetic", "sha256": digest},
            "data.sequence_length": 9,
            "data.max_examples": 8,
            "defects.cases": [case_doc(), case_doc("random", 50000)],
            "hwa.batch_size": 4,
            "hwa.sequences_per_epoch": 3,
            "onchip": None,
        }
    )
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    fit = results["hwa__standard_hwa"][1]["metrics"]["fit"]
    assert fit["status"] == "complete"
    assert fit["stream"] == {
        "sequences_per_epoch": 3,
        "cohort_sequences": 8,
        "passes": fit["epochs_run"] * 3 / 8,
    }


# --- whole-model HWA and transfer tests -------------------------------------------------


def model_arm(**changes):
    arm = {
        "id": "standard_hwa_model",
        "method": "hwa",
        "noise_strength": 1.0,
        "corruption": None,
        "selection_cases": [case_doc()],
        "scope": "model",
        "digital_learning_rate": 1e-4,
    }
    arm.update(changes)
    return arm


def transfer_doc(digest, label="other", split="test", sequences=3):
    return [{"label": label, "corpus": {"name": "other", "sha256": digest}, "split": split, "sequences": sequences}]


def test_whole_model_scope_is_an_optional_arm_setting_that_keeps_analog_digests():
    plain = small()
    explicit = template()
    explicit["hwa"]["arms"][1]["scope"] = "analog"
    explicit["hwa"]["arms"][1]["digital_learning_rate"] = None
    explicit = small(**{"hwa.arms": explicit["hwa"]["arms"]})
    assert explicit.hwa_arm_digest("standard_hwa") == plain.hwa_arm_digest("standard_hwa")
    assert explicit.digest() == plain.digest()
    arms = template()["hwa"]["arms"] + [model_arm()]
    lc = small(**{"hwa.arms": arms})
    arm = lc.hwa.arm("standard_hwa_model")
    assert (arm.scope, arm.digital_learning_rate) == ("model", 1e-4)
    assert arm.to_dict()["scope"] == "model" and parse_lifecycle(lc.to_dict()) == lc
    # Adding a whole-model arm does not change the other arms' source identity.
    assert lc.hwa_arm_digest("standard_hwa") == plain.hwa_arm_digest("standard_hwa")
    with pytest.raises(ConfigError, match="digital_learning_rate"):
        small(**{"hwa.arms": template()["hwa"]["arms"] + [model_arm(digital_learning_rate=None)]})
    with pytest.raises(ConfigError, match="digital_learning_rate"):
        small(**{"hwa.arms": template()["hwa"]["arms"] + [model_arm(scope="analog")]})
    with pytest.raises(ConfigError, match="scope"):
        small(**{"hwa.arms": template()["hwa"]["arms"] + [model_arm(scope="all")]})


def test_transfer_tests_are_optional_and_never_enter_source_provenance():
    digest = "a" * 64
    plain = small()
    doc = small().to_dict()
    doc["transfer_tests"] = transfer_doc(digest)
    lc = parse_lifecycle(doc)
    assert lc.transfer_tests.tests[0].cache_key == "transfer_other"
    assert parse_lifecycle(lc.to_dict()) == lc and lc.digest() != plain.digest()
    assert "transfer_tests" not in plain.to_dict() and "transfer_tests" in lc.to_dict()
    assert "transfer_tests" in provenance(lc, "feature_cache")["sections"]
    assert "transfer_tests" in provenance(lc, "deployment", hwa_arm="standard_hwa")["sections"]
    # A source trained without transfer tests is reusable by a lifecycle that adds them.
    source = provenance(plain, "hwa_source", hwa_arm="standard_hwa")
    verify_provenance(source, lc, "hwa_source", hwa_arm="standard_hwa")
    with pytest.raises(ValueError, match="transfer_tests"):
        verify_provenance(provenance(plain, "feature_cache"), lc, "feature_cache")
    for bad, message in (
        (transfer_doc(digest, split="train"), "split"),
        (transfer_doc(digest) + transfer_doc(digest), "unique"),
        (transfer_doc(plain.data.corpus_sha256, split="validation"), "validation"),
    ):
        broken = small().to_dict()
        broken["transfer_tests"] = bad
        with pytest.raises(ConfigError, match=message):
            parse_lifecycle(broken)


def test_whole_model_student_starts_as_the_analog_student_with_its_own_digital_copy():
    teacher = tiny_teacher()
    arms = template()["hwa"]["arms"] + [model_arm()]
    lc = small(**{"hwa.arms": arms})
    analog = deployment.build_network(teacher, lc, "standard_hwa")
    whole = deployment.build_network(teacher, lc, "standard_hwa_model")
    assert analog.digital is None and whole.digital is not None
    tokens = torch.randint(0, teacher.config.vocab_size, (2, 7), generator=torch.Generator().manual_seed(5))
    inputs = whole.features(tokens)
    assert isinstance(inputs, StudentInput) and torch.equal(inputs.teacher_hidden, analog.features(tokens))
    assert torch.equal(whole.forward_features(inputs), analog.forward_features(analog.features(tokens)))
    assert torch.equal(whole.teacher_logits(inputs), analog.teacher_logits(analog.features(tokens)))
    keys = set(whole.state_dict())
    assert keys - set(analog.state_dict()) == {k for k in keys if k.startswith("digital.")}
    start = len(teacher.layers) - lc.network.analog_decoder_layers
    assert not any(f"digital.layers.{i}.fc1" in k or f"digital.layers.{i}.final_layer_norm" in k
                   for k in keys for i in range(start, len(teacher.layers)))
    assert "digital.layers.0.fc1.weight" in keys and "digital.embed_tokens.weight" in keys
    whole.enable_calibration(True)
    whole.enable_digital(True)
    digital = {id(p) for p in whole.digital_parameters()}
    assert digital and not digital & {id(p) for p in whole.calibration_parameters()}
    whole.enable_digital(False)
    assert not any(p.requires_grad for p in whole.digital_parameters())


def test_whole_model_hwa_trains_digital_weights_and_deploys_them_with_transfer_scores(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    other = write_corpus(tmp_path / "corpora", name="other", seed=12)
    arms = template()["hwa"]["arms"][:2] + [model_arm(selection_cases=[case_doc(), case_doc("random", 30000)])]
    doc = small(
        **{
            "data.corpus": {"name": "synthetic", "sha256": digest},
            "data.sequence_length": 9,
            "data.max_examples": 8,
            "defects.cases": [case_doc(), case_doc("random", 50000)],
            "hwa.batch_size": 4,
            "hwa.arms": arms,
            "onchip": None,
        }
    ).to_dict()
    doc["transfer_tests"] = transfer_doc(other, sequences=3)
    lc = parse_lifecycle(doc)
    teacher = tiny_teacher()
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    fit = results["hwa__standard_hwa_model"][1]["metrics"]["fit"]
    assert fit["status"] == "complete" and fit["scope"]["scope"] == "model"
    assert fit["scope"]["digital_learning_rate"] == 1e-4 and fit["scope"]["digital_parameters"] > 0
    assert "scope" not in results["hwa__standard_hwa"][1]["metrics"]["fit"]
    source = load(next((tmp_path / "runs/hwa__standard_hwa_model").glob("*/checkpoints/source.pt")))
    trained = {k[len("digital."):]: v for k, v in source["model"].items() if k.startswith("digital.")}
    reference = teacher.state_dict()
    assert any(not torch.equal(v.cpu(), reference[k]) for k, v in trained.items())
    run_dir, deploy = results["deploy__standard_hwa_model__271001"]
    assert set(deploy["metrics"]["clean"]) == {"development", "test", "transfer_other"}
    assert deploy["metrics"]["clean"]["transfer_other"]["sequences"] == 3
    collected = tmp_path / "collected"
    assert workflow_main(["collect", str(tmp_path / "runs"), "--output", str(collected)]) == 0
    rows = json.loads((collected / "deployments.json").read_text())
    assert len(rows) == len(lc.hwa.arms) * len(lc.defects.cases)
    assert all("deployed_transfer_other_teacher_kl" in r and "clean_transfer_other_perplexity" in r for r in rows)



def test_onchip_recovery_holds_whole_model_digital_weights_and_reports_transfer_scores(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    other = write_corpus(tmp_path / "corpora", name="other", seed=12)
    arms = template()["hwa"]["arms"][:1] + [model_arm(selection_cases=[case_doc()])]
    doc = small(
        **{
            "data.corpus": {"name": "synthetic", "sha256": digest},
            "data.sequence_length": 9,
            "data.max_examples": 8,
            "defects.cases": [case_doc("random", 50000)],
            "hwa.batch_size": 4,
            "hwa.arms": arms,
            "onchip.batch_size": 4,
        }
    ).to_dict()
    doc["transfer_tests"] = transfer_doc(other, sequences=2)
    lc = parse_lifecycle(doc)
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    run_dir, _ = results["onchip__standard_hwa_model__271001"]
    rows = json.loads((run_dir / "measurements.json").read_text())
    assert {r["onchip_arm"] for r in rows} == {arm.arm_id for arm in lc.onchip.arms}
    assert all("transfer_other" in r["final"] and r["replay_exact"] for r in rows)
    # On-chip arms never train the source's digital weights: none is saved as a
    # trained parameter, and the exact replay restores them from the source.
    for row in rows:
        state = load(run_dir / "checkpoints/onchip" / row["case"] / f"{row['onchip_arm']}.pt")
        assert not any(name.startswith("digital.") for name in state["calibration"])
    collected = tmp_path / "collected"
    assert workflow_main(["collect", str(tmp_path / "runs"), "--output", str(collected)]) == 0
    collected_rows = json.loads((collected / "rows.json").read_text())
    assert collected_rows and all("final_transfer_other_teacher_kl" in r for r in collected_rows)
