"""CMO drift recovery (cmo-drift-recovery exp-001): reset_absorb writes, two-sided mapping, gate, arms."""
import json
import math
from pathlib import Path

import pytest
import torch

import hardware.arrays as devices
import methods.drift_tuning as dt
import methods.fault_tuning as ft
import methods.programming.lifecycle as program_verify
from ebl.schema.lifecycle import parse_lifecycle, stage_selections
from hardware.config import _defects
from hardware.devices.cmo import programming_sigma
from hardware.mapping import MatrixSpec
from networks.mapping import fault_domains

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "campaigns/pilots/20261118-cmo-drift-recovery/lifecycles"
OLD = ROOT / "campaigns/pilots/20261116-drift-token-budget/lifecycles"
FULL = PILOT / "full/cmo-drift-recovery-20261118-xb6-m4.json"
GRID = [0.0, 60.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0, 57600.0, 86400.0]
TWO_SIDED = {"kind": "two_sided", "gain": {"rule": "unit"}, "margin_us": 0.0}


def law(**changes):
    raw = json.loads(FULL.read_text())["fault_tuning"]["drift"]
    raw.update(changes)
    return dt.DriftTuning.parse(raw)


def array(noise_scale=1.0, mapping=None, target=None):
    layout = [MatrixSpec("decoder.layers.6.fc1", (64, 32), 0, 64 * 32, 1.0),
              MatrixSpec("decoder.layers.6.fc2", (32, 64), 64 * 32, 32 * 64, 1.0)]
    case = _defects({"placement": "hierarchical_die_array", "persistence": "permanent", "arrays_per_die": 48,
                     "cases": [{"kind": "none", "rate_ppm": 0, "die_concentration": None,
                                "array_concentration": None}]}, "cmo").cases[0]
    a = devices.fresh_array(technology="cmo", layout=layout, assignment_seed=271001, case=case, populations=None,
                            device="cpu", namespace="facebook/opt-125m", endpoint_seed=281001,
                            noise_scale=noise_scale, tile_size=256, arrays_per_die=48, acceptance_percent=2.0,
                            domains=fault_domains(layout))
    target = torch.linspace(-0.8, 0.8, 2 * 64 * 32) if target is None else target
    if mapping is None:
        a.program(target)
    else:
        a.program_conductances(mapping.targets(target))
    return a


def conductances(plant):
    return [child.g.clone() for child in plant.array.children]


# --- write clock reset_absorb --------------------------------------------------------------------

def test_forced_write_absorbs_the_drifted_read_and_restarts_only_written_clocks():
    d, m = law(), dt.Mapping(TWO_SIDED)
    learner, hold = dt.DriftPlant(array(mapping=m), d, m), dt.DriftPlant(array(mapping=m), d, m)
    t = 900.0
    learner.read(t, keep=True)
    drifted_at_t = [g.clone() for g in learner._last[1]]
    issued = torch.zeros(learner.array.size)
    issued[:50] = 0.05        # weights near -0.8: sign side G1, no crossing
    issued[-50:] = 0.05       # weights near +0.8: sign side G0
    states = [child.generator.get_state() for child in learner.array.children]
    learner.write(issued, t)
    for k, (child, state) in enumerate(zip(learner.array.children, states)):
        noise = torch.randn((2, child.size), generator=torch.Generator().set_state(state))
        part = issued[learner.array.layout[k].offset:][:child.size]
        gd = drifted_at_t[k]
        positive = (gd[0] - gd[1]) > 0
        step = torch.stack((torch.where(positive, 79.2 * part, 0.0), torch.where(positive, 0.0, -79.2 * part)))
        written = (part != 0) & torch.stack((positive, ~positive))
        requested = gd + step
        expected = requested + programming_sigma(requested, 2.0) * noise
        assert torch.equal(learner.written[k] if learner.written[k] is not None else torch.zeros_like(written),
                           written)
        if bool(written.any()):
            assert torch.allclose(child.g[written], expected[written], atol=1e-5, rtol=0)
            assert torch.equal(learner.write_seconds[k][written], torch.full_like(child.g[written], t))
        held = hold.array.children[k].g
        assert torch.equal(child.g[~written], held[~written])
    later = 7200.0
    a, b = learner.conductances(later), hold.conductances(later)
    for k in range(2):
        w = learner.written[k] if learner.written[k] is not None else torch.zeros_like(a[k], dtype=torch.bool)
        assert torch.equal(a[k][~w], b[k][~w])     # unwritten devices: bit for bit as A0
        if bool(w.any()):
            g = torch.Generator().manual_seed(dt.draw_key(d.draw_seed, k, later))
            y = torch.randn(a[k].shape, generator=g)
            log_tau = math.log(later - t)
            m4 = d.multiplier
            manual = (learner.array.children[k].g + m4 * d.mean_us * log_tau
                      + m4 * (d.spread_slope_us * log_tau + d.spread_offset_us) * y).clamp(5, 150)
            assert torch.allclose(a[k][w], manual[w], atol=1e-4, rtol=0)   # tau = t - t_w restarts
    assert learner.cost()["device_writes"] == 100 and learner.cost()["crossing_updates"] == 0


def test_a_write_requires_the_learner_read_at_the_same_age():
    d, m = law(), dt.Mapping(TWO_SIDED)
    plant = dt.DriftPlant(array(mapping=m), d, m)
    issued = torch.zeros(plant.array.size)
    issued[0] = 0.1
    with pytest.raises(RuntimeError):
        plant.write(issued, 60.0)
    plant.read(60.0, keep=True)
    with pytest.raises(RuntimeError):
        plant.write(issued, 61.5)


def test_a_just_written_device_reads_its_held_value_and_later_relaxes_from_its_write():
    d, m = law(), dt.Mapping(TWO_SIDED)
    plant = dt.DriftPlant(array(mapping=m), d, m)
    plant.read(30.0, keep=True)
    issued = torch.zeros(plant.array.size)
    issued[-1] = 0.1
    plant.write(issued, 30.0)
    k = len(plant.array.children) - 1
    w = plant.written[k]
    assert torch.equal(plant.conductances(30.0)[k][w], plant.array.children[k].g[w])   # tau = 0: no drift
    assert not torch.equal(plant.conductances(31.5)[k][w], plant.array.children[k].g[w])


def test_continues_is_unchanged():
    raw = json.loads((OLD / "full/drift-token-budget-20261116-xb6-m4.json").read_text())["fault_tuning"]["drift"]
    d = dt.DriftTuning.parse(raw)
    assert d.to_dict() == raw and d.mapping is None
    a = array()
    plant = dt.DriftPlant(a, d, dt.Mapping())
    for t in GRID:
        assert torch.equal(plant.read(t), dt.drifted_weights(a, d, t))
    assert torch.equal(plant.read(None), a.read())
    with pytest.raises(RuntimeError):
        plant.read(60.0, keep=True)
        plant.write(torch.ones(a.size), 60.0)
    with pytest.raises(ValueError):
        dt.DriftTuning.parse({**d.to_dict(), "mapping": TWO_SIDED})


# --- two-sided mapping ---------------------------------------------------------------------------

def test_two_sided_targets_and_clean_readout():
    m = dt.Mapping(TWO_SIDED)
    q = torch.linspace(-1, 1, 1001)
    g = m.targets(q)
    assert torch.allclose(g[0], 9 + 79.2 * q.clamp_min(0)) and torch.allclose(g[1], 9 + 79.2 * (-q).clamp_min(0))
    assert float(g.min()) == 9.0 and float(g.max()) == pytest.approx(88.2)
    assert float((m.clean(q) - q).abs().max()) < 1e-6
    # Noise-free programming of exact targets reads back q.
    target = torch.linspace(-0.8, 0.8, 2 * 64 * 32)
    a = array(noise_scale=0.0, mapping=m, target=target)
    assert float((m.read_array(a) - target).abs().max()) < 1e-6
    # The network multiplies the readout by each channel's digital scale c: readout * c = q * c.
    c = torch.rand(64) + 0.5
    rows = (m.read_array(a)[:64 * 32].view(64, 32) * c[:, None])
    assert torch.allclose(rows, target[:64 * 32].view(64, 32) * c[:, None], atol=1e-6)
    assert dt.Mapping().span == pytest.approx(39.6) and m.span == 79.2


def test_two_sided_sign_side_and_zero_crossing_writes():
    d, m = law(), dt.Mapping(TWO_SIDED)
    target = torch.tensor([0.5, -0.5, 0.5, -0.5, 0.0, 0.0] + [0.1] * (2 * 64 * 32 - 6))
    plant = dt.DriftPlant(array(noise_scale=0.0, mapping=m, target=target), d, m, drift=False)
    issued = torch.zeros(plant.array.size)
    issued[:6] = torch.tensor([0.2, -0.2, -0.7, 0.7, 0.3, -0.3])
    before = plant.read(None)
    plant.read(60.0, keep=True)
    plant.write(issued, 60.0)
    g = plant.array.children[0].g
    after = plant.read(None)
    assert torch.allclose(after[:6], before[:6] + issued[:6], atol=1e-5)          # noise free: q + dq
    w = plant.written[0]
    assert w[:, 0].tolist() == [True, False] and w[:, 1].tolist() == [False, True]   # sign side only
    assert w[:, 2].tolist() == [True, True] and w[:, 3].tolist() == [True, True]     # crossing: both
    assert float(g[0, 2]) == 9.0 and float(g[1, 3]) == 9.0                            # sign side to 9 uS
    assert float(g[1, 2]) == pytest.approx(9 + 79.2 * 0.2) and float(g[0, 3]) == pytest.approx(9 + 79.2 * 0.2)
    assert w[:, 4].tolist() == [True, False] and w[:, 5].tolist() == [False, True]   # q = 0: update's sign
    assert plant.cost()["crossing_updates"] == 2 and plant.cost()["device_writes"] == 8
    assert torch.equal(plant.weight_writes()[:6], torch.tensor([1, 1, 2, 2, 1, 1], dtype=torch.int32))


def test_two_sided_crossing_write_noise_is_sigma_at_nine_microsiemens():
    d, m = law(), dt.Mapping(TWO_SIDED)
    n = 2 * 64 * 32
    plant = dt.DriftPlant(array(noise_scale=0.0, mapping=m, target=torch.full((n,), 0.5)), d, m, drift=False)
    plant.array.children[0].noise_scale = 1.0
    issued = torch.zeros(n)
    issued[:64 * 32] = -0.7
    plant.read(60.0, keep=True)
    plant.write(issued, 60.0)
    g0 = plant.array.children[0].g[0]
    assert float(g0.std()) == pytest.approx(float(programming_sigma(torch.tensor(9.0), 2.0)), rel=0.1)
    assert float(g0.mean()) == pytest.approx(9.0, abs=0.01)


def test_two_sided_verify_rewrite_uses_the_mapped_targets():
    m = dt.Mapping(TWO_SIDED)
    target = torch.linspace(-0.8, 0.8, 16)
    expected = m.targets(target)
    reads = expected.clone()
    reads[0, 3] += 10.0
    flags = program_verify.detect_stuck_devices(reads, target, acceptance=2.0, noise_scale=1.0, threshold_sigma=5.0,
                                                expected=expected)
    assert int(flags.sum()) == 1 and bool(flags[0, 3])
    selected, targets, _ = program_verify.pair_rewrite_plan(flags, reads, target, "pair", m.span)
    assert bool(selected[1, 3]) and float(targets[1, 3]) == pytest.approx(
        max(9.0, float(reads[0, 3] - 79.2 * target[3])))
    # Default arguments keep the standard pair rewrite.
    standard = torch.stack((48.6 + 39.6 * target, torch.full_like(target, 48.6)))
    assert not bool(program_verify.detect_stuck_devices(standard, target, acceptance=2.0, noise_scale=1.0,
                                                        threshold_sigma=5.0).any())


# --- gate ----------------------------------------------------------------------------------------

def test_gate_is_two_sigma_at_mid_range_in_conductance():
    gate = dt.gate_us(2.0, 1.0, 2.0)
    assert gate == pytest.approx(2 * (0.0112185391 + 0.01129027418 * 48.6), rel=1e-6)
    assert gate == pytest.approx(1.12, abs=0.001) and gate / 79.2 == pytest.approx(0.01414, abs=1e-5)


def test_gate_fires_exactly_at_the_threshold():
    d, m = law(), dt.Mapping(TWO_SIDED)
    plant = dt.DriftPlant(array(mapping=m), d, m)
    gate = dt.gate_us(2.0, 1.0, 2.0)
    writer = dt.GatedDriftAdam(plant, learning_rate=1e-3, threshold_us=gate)
    at = torch.tensor(gate / 79.2)
    while float((at * 79.2).abs()) < gate:
        at = torch.nextafter(at, torch.tensor(1.0))
    below = torch.nextafter(at, torch.tensor(0.0))
    pending = torch.stack((at, -at, below, -below, torch.tensor(0.0)))
    assert writer.fires(pending).tolist() == [True, True, False, False, False]
    assert float((at * 79.2).abs()) >= gate > float((below * 79.2).abs())


def test_gated_writer_issues_accumulated_commands_and_respects_the_mask():
    d, m = law(), dt.Mapping(TWO_SIDED)
    plant = dt.DriftPlant(array(mapping=m), d, m)
    mask = torch.zeros(plant.array.size, dtype=torch.bool)
    mask[::10] = True
    writer = dt.GatedDriftAdam(plant, learning_rate=0.05, threshold_us=dt.gate_us(2.0, 1.0, 2.0), mask=mask)
    gradient = torch.randn(plant.array.size, generator=torch.Generator().manual_seed(1))
    for step in range(3):
        age = d.age(step + 1)
        plant.read(age, keep=True)
        writer.step(gradient, age)
    writes = plant.weight_writes()
    assert int(writes.sum()) > 0 and int(writes[~mask].sum()) == 0
    assert bool((writer.pending[~mask] == 0).all())
    assert bool((writer.pending.abs() * 79.2 < writer.threshold_us).all())


# --- replay and controls -------------------------------------------------------------------------

def test_zero_learning_rate_replays_the_hold_at_every_age():
    d, m = law(), dt.Mapping(TWO_SIDED)
    learner, hold = dt.DriftPlant(array(mapping=m), d, m), dt.DriftPlant(array(mapping=m), d, m)
    writer = dt.GatedDriftAdam(learner, learning_rate=0.0, threshold_us=dt.gate_us(2.0, 1.0, 2.0))
    for step in range(5):
        age = d.age(step + 1)
        q = learner.read(age, keep=True)
        writer.step(torch.randn_like(q), age)
    assert learner.cost()["device_writes"] == 0
    for t in GRID:
        assert torch.equal(learner.read(t), hold.read(t))


def test_drift_off_plant_reads_the_release_state_at_every_age():
    d, m = law(), dt.Mapping(TWO_SIDED)
    twin, a0 = dt.DriftPlant(array(mapping=m), d, m, drift=False), dt.DriftPlant(array(mapping=m), d, m)
    for t in GRID:
        assert torch.equal(twin.read(t), a0.read(0.0))
    assert not torch.equal(a0.read(60.0), a0.read(0.0))


# --- lifecycles ----------------------------------------------------------------------------------

def test_cmo_drift_recovery_lifecycles():
    seen = set()
    for path in sorted(PILOT.glob("*/*.json")):
        raw = json.loads(path.read_text())
        model, multiplier = path.stem.split("-")[4], path.stem.split("-")[5]
        seen.add((model, multiplier, path.parent.name))
        old = json.loads((OLD / path.parent.name / (f"drift-token-budget-20261116-{model}-m4"
                                                    + ("-canary" if path.parent.name == "canary" else "")
                                                    + ".json")).read_text())
        lc = parse_lifecycle(raw)
        assert [s.kind for s in stage_selections(lc)] == ["prepare", "devices", "hwa", "fault_tuning"]
        assert parse_lifecycle(lc.to_dict()).digest() == lc.digest()
        f, g = dict(raw["fault_tuning"]), dict(old["fault_tuning"])
        drift, old_drift = f.pop("drift"), g.pop("drift")
        assert drift.pop("mapping") == TWO_SIDED and drift.pop("write_clock") == "reset_absorb"
        assert drift.pop("multiplier") == float(multiplier[1:])
        assert {k: v for k, v in old_drift.items() if k not in ("multiplier", "write_clock")} == drift
        f.pop("arms"), g.pop("arms")
        assert f == g                      # pool, targets, dev set, source, schedule, gate, rewrite: unchanged
        assert {k: v for k, v in raw.items() if k not in ("fault_tuning", "lifecycle_id", "question")} == \
            {k: v for k, v in old.items() if k not in ("fault_tuning", "lifecycle_id", "question")}
        labels = [a.label for a in lc.fault_tuning.arms]
        assert labels[:7] == ["A0", "A0-twin", "CAL", "LN", "CT", "L1", "LoRA"]
        assert ("CT-nodrift" in labels) == (multiplier == "m4")
        lr = {a.label: a.learning_rate for a in lc.fault_tuning.arms}
        assert lr["CT"] == (1e-3 if model == "xb6" else 3e-3) and lr["LoRA"] == (3e-4 if model == "xb6" else 1e-3)
        assert lr["CAL"] == 1e-3 and lr["LN"] == 3e-3 and lr["L1"] == 1e-3
    assert seen == {(m, x, "full") for m in ("xb6", "xb12") for x in ("m1", "m4", "m10")} | \
        {(m, "m4", "canary") for m in ("xb6", "xb12")}


def test_invalid_drift_arm_sets_are_rejected():
    raw = json.loads(FULL.read_text())

    def broken(edit):
        doc = json.loads(json.dumps(raw))
        edit(doc)
        with pytest.raises(Exception):
            parse_lifecycle(doc)

    arms = raw["fault_tuning"]["arms"]
    broken(lambda d: d["fault_tuning"].update(arms=[a for a in arms if a["label"] != "A0"]))
    broken(lambda d: d["fault_tuning"].update(arms=arms + [{"label": "T2", "method": "hold_fault_aware",
                                                            "no_drift": True}]))
    broken(lambda d: d["fault_tuning"].update(arms=arms + [{"label": "C2", "method": "calibration",
                                                            "learning_rate": 1e-3, "no_drift": True}]))
    broken(lambda d: d["fault_tuning"].update(arms=arms + [{"label": "X", "method": "matched_noiseless",
                                                            "learning_rate": 1e-3}]))
    broken(lambda d: d["fault_tuning"]["arms"][1].update(no_drift=False))
    broken(lambda d: d["fault_tuning"]["drift"].update(write_clock="continues"))     # continues keeps the pair
    broken(lambda d: d["fault_tuning"]["drift"].update(mapping={**TWO_SIDED, "margin_us": 1.0}))
    broken(lambda d: d["fault_tuning"]["drift"].update(mapping={**TWO_SIDED, "gain": {"rule": "saturation",
                                                                                       "fraction": 0.01}}))
    # Without the drift block, layernorm and no_drift arms do not exist.
    fault = json.loads((ROOT / "campaigns/fault-recovery-paper/lifecycles/fault-token-budget-20261116/full/"
                               "fault-token-budget-20261116-xb6-n8192.json").read_text())
    for arm in ({"label": "LN", "method": "layernorm", "learning_rate": 3e-3},
                {"label": "T", "method": "hold_fault_aware", "no_drift": True}):
        doc = json.loads(json.dumps(fault))
        doc["fault_tuning"]["arms"].append(arm)
        with pytest.raises(Exception):
            parse_lifecycle(doc)


def test_global_child_mask_pools_scores_and_breaks_ties_in_layout_order():
    fisher = torch.tensor([1., 0., 0., 0., 5., 5., 5., 5., 6., 9.])
    selected = ft.global_child_mask(fisher, 0.3)
    smaller = ft.global_child_mask(fisher, 0.1)
    assert selected.nonzero().flatten().tolist() == [4, 8, 9]
    assert smaller.nonzero().flatten().tolist() == [9]
    assert not (smaller & ~selected).any()
    with pytest.raises(ValueError, match="finite Fisher"):
        ft.global_child_mask(torch.tensor([float("nan")]), 0.1)


def test_global_child_lifecycle_opt_in_preserves_legacy_serialization():
    raw = json.loads(FULL.read_text())
    original = parse_lifecycle(raw).to_dict()["fault_tuning"]["arms"]
    assert "child_mask_scope" not in next(a for a in original if a["label"] == "CT")
    assert "child_mask_scope" not in next(a for a in original if a["label"] == "CT-nodrift")
    for arm in raw["fault_tuning"]["arms"]:
        if arm["method"] == "child_tuning_d":
            arm["child_mask_scope"] = "global"
    parsed = parse_lifecycle(raw)
    assert [a.child_mask_scope for a in parsed.fault_tuning.arms if a.method == "child_tuning_d"] == [
        "global", "global"
    ]
    assert parse_lifecycle(parsed.to_dict()).digest() == parsed.digest()
    raw["fault_tuning"]["arms"][0]["child_mask_scope"] = "global"
    with pytest.raises(Exception, match="Expected exactly"):
        parse_lifecycle(raw)
