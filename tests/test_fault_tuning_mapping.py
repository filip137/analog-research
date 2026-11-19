"""Deployment mapping without drift (fault-recovery-paper series 003 exp-001): schema, readout, statistics."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from core.tensor_identity import tensor_hash
from ebl.runtime import fault_tuning as runtime
from ebl.schema.lifecycle import parse_lifecycle, stage_selections
from hardware.config import _defects
import methods.drift_tuning as dt
from methods import fault_tuning as ft
import networks.opt as opt
from tests.test_workflow_opt import tiny_teacher

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "campaigns/fault-recovery-paper/lifecycles/paper-grid-20261116/paper"
GRID = PAPER / "fault-paper-grid-20261116-xb6-r03-271002.json"
DRIFT = ROOT / "campaigns/pilots/20261118-cmo-drift-recovery/lifecycles/full/cmo-drift-recovery-20261118-xb6-m4.json"
REFERENCE = {"kind": "reference", "gain": {"rule": "unit"}, "margin_us": 0.0}
TWO_SIDED = {"kind": "two_sided", "gain": {"rule": "unit"}, "margin_us": 0.0}
HOLDS = [{"label": "naive_hold", "method": "hold_naive"}, {"label": "fault_aware_hold", "method": "hold_fault_aware"}]


def hold_only(mapping=None, rate_ppm=30000):
    doc = json.loads(GRID.read_text())
    doc["fault_tuning"].update(arms=copy.deepcopy(HOLDS), analog_lora=None)
    doc["defects"]["cases"][0].update(kind="random" if rate_ppm else "none", rate_ppm=rate_ppm)
    if mapping is not None:
        doc["fault_tuning"]["mapping"] = dict(mapping)
    return doc


# --- (c) schema ----------------------------------------------------------------------------------

def test_schema_accepts_a_hold_only_mapping_at_any_fault_rate_and_keeps_legacy_digests():
    for path in sorted(PAPER.glob("*.json")):
        lc = parse_lifecycle(json.loads(path.read_text()))
        assert lc.fault_tuning.mapping is None and "mapping" not in lc.fault_tuning.to_dict()
    base = parse_lifecycle(hold_only())
    for rate in (0, 10000, 30000, 50000):
        for mapping in (REFERENCE, TWO_SIDED):
            lc = parse_lifecycle(hold_only(mapping, rate))
            assert lc.fault_tuning.mapping == mapping and lc.fault_tuning.drift is None
            assert lc.fault_tuning.deployment_mapping() == mapping
            assert lc.fault_tuning.to_dict()["mapping"] == mapping
            assert parse_lifecycle(lc.to_dict()).digest() == lc.digest() != base.digest()
            assert [s.kind for s in stage_selections(lc)] == ["prepare", "devices", "hwa", "fault_tuning"]
    # The no-HWA source: the unchanged pretrained teacher as the single frozen arm.
    doc = hold_only(TWO_SIDED)
    doc["hwa"]["arms"] = [{"id": "pretrained", "method": "none"}]
    assert parse_lifecycle(doc).hwa.arms[0].trains is False


def test_schema_rejects_writers_adapters_and_drift_with_a_non_reference_mapping():
    def rejected(doc):
        with pytest.raises(Exception):
            parse_lifecycle(doc)

    doc = hold_only(TWO_SIDED)
    doc["fault_tuning"]["arms"].append({"label": "noisy", "learning_rate": 0.001, "method": "onchip_noisy"})
    rejected(doc)
    doc = hold_only(TWO_SIDED)
    doc["fault_tuning"]["arms"].append({"label": "dlora", "learning_rate": 0.001, "method": "digital_lora", "rank": 4})
    rejected(doc)
    doc = json.loads(GRID.read_text())
    doc["fault_tuning"]["mapping"] = TWO_SIDED
    rejected(doc)  # analog LoRA tiles and writing arms
    rejected(hold_only({**TWO_SIDED, "margin_us": 1.0}))
    rejected(hold_only({"kind": "ratio3", "gain": {"rule": "unit"}, "margin_us": 0.0}))
    drift = json.loads(DRIFT.read_text())
    drift["fault_tuning"]["mapping"] = TWO_SIDED
    rejected(drift)  # under drift the mapping stays in drift.mapping
    # A reference mapping keeps every arm.
    doc = json.loads(GRID.read_text())
    doc["fault_tuning"]["mapping"] = REFERENCE
    assert len(parse_lifecycle(doc).fault_tuning.arms) == 9


# --- runtime -------------------------------------------------------------------------------------

def context(mapping, rate_ppm=30000, noise_scale=1.0):
    """A real ``Context`` on the paper-grid lifecycle with a tiny OPT suffix (CPU)."""
    torch.set_num_threads(1)
    lc = parse_lifecycle(hold_only(mapping, rate_ppm))
    lc = replace(lc, devices=replace(lc.devices, noise_scale=noise_scale))
    net = opt.OptMlpSuffix(tiny_teacher(), 2, 8).eval().requires_grad_(False)
    ctx = runtime.Context.__new__(runtime.Context)
    ctx.lc, ctx.cfg, ctx.network, ctx.device = lc, lc.fault_tuning, net, torch.device("cpu")
    ctx.seed, ctx.case, ctx.layout, ctx.fisher = 271002, lc.defects.cases[0], list(net.layout), None
    spec = ctx.cfg.deployment_mapping()
    ctx.mapping = None if spec is None else dt.Mapping(spec)
    ctx.readout = (lambda array: array.read()) if spec is None else ctx.mapping.read_array
    generator = torch.Generator().manual_seed(3)
    dev = torch.randint(0, 64, (2, 6), generator=generator)
    held = torch.randint(0, 64, (2, 6), generator=generator)
    events, scored = [], []

    def evaluate(network, split, device, *, q):
        scored.append(q.clone())
        return opt.evaluate(network, split, device, q=q)

    ctx.stage = NS(store=NS(append_metric=events.append), family=NS(evaluate=evaluate))
    ctx.evaluation_cohorts = [({"name": "held", "sequences": 2}, held)]
    ctx.evaluate = lambda q: opt.evaluate(net, {"tokens": dev}, "cpu", q=q)
    target = net.q.detach().clone()
    bundle = runtime.deploy(ctx, target)
    ctx.deployed = bundle["states"]
    ctx.deployed_cost = ctx.array("fault_aware").cost()
    return ctx, target, bundle, scored


def rows(ctx):
    return [runtime.run_arm(ctx, ft.Arm(a["label"], a["method"]))[0] for a in HOLDS]


def strip_seconds(row):
    row = copy.deepcopy(row)
    row.pop("seconds")
    for point in row["held_out"]:
        point.pop("seconds")
    return row


def test_base_mapping_reproduces_the_standard_pair_bit_for_bit():
    legacy, target, old, _ = context(None)
    declared, _, new, _ = context(REFERENCE)
    assert "mapping" not in old and "mapping" not in new   # same deployment schema and identity
    for key in ("naive", "fault_aware"):
        assert old["states"][key]["apparent_sha256"] == new["states"][key]["apparent_sha256"]
        a, b = legacy.array(key), declared.array(key)
        for x, y in zip(a.children, b.children):
            assert torch.equal(x.g, y.g) and torch.equal(x.mask, y.mask)
        # The declared reference readout is the array's own readout, bit for bit.
        assert torch.equal(declared.mapping.read_array(b), a.read())
    assert old["states"]["fault_aware"]["report"]["controller"] == new["states"]["fault_aware"]["report"]["controller"]
    for x, y in zip(rows(legacy), rows(declared)):
        assert strip_seconds(x) == strip_seconds(y)


def test_two_sided_readout_at_zero_faults_matches_the_targets_without_doubling():
    exact, target, bundle, _ = context(TWO_SIDED, rate_ppm=0, noise_scale=0.0)
    assert bundle["mapping"] == TWO_SIDED
    array = exact.array("fault_aware")
    assert float((runtime.read(exact, array) - target).abs().max()) < 1e-5
    # The base readout (G0 - G1) / 39.6 of a two-sided deployment doubles every weight.
    assert torch.allclose(array.read(), 2 * target, atol=1e-4)

    ctx, target, bundle, scored = context(TWO_SIDED, rate_ppm=0)
    controller = bundle["states"]["fault_aware"]["report"]["controller"]
    assert controller["flagged_active"] + controller["flagged_reference"] == 0
    expected = dt.Mapping(TWO_SIDED).targets(target)
    sigma = dt.programming_sigma(expected, 2.0) / 79.2   # per-device programming error in weight units
    bound = 6 * float((sigma[0].square() + sigma[1].square()).sqrt().max())
    for key, label in (("naive", "naive_hold"), ("fault_aware", "fault_aware_hold")):
        q = runtime.read(ctx, ctx.array(key))
        assert float((q - target).abs().max()) < bound
        assert float((q - target).square().mean().sqrt()) < bound / 3
    measured = rows(ctx)
    # Every scored forward (development and held-out) used the mapped readout, never 2 q.
    for row in measured:
        assert row["initial_apparent_sha256"] == row["final_apparent_sha256"]
        assert row["held_out"][0]["scores"]["teacher_kl"] < 1.0
    assert len(scored) == 2
    for q in scored:
        assert float((q - target).abs().max()) < bound
    assert measured[1]["initial_apparent_sha256"] == tensor_hash(ctx.mapping.read_array(ctx.array("fault_aware")))


def test_device_statistics_report_floors_faults_and_clipping():
    ctx, target, bundle, _ = context(TWO_SIDED, rate_ppm=0)
    stats = runtime.device_statistics(ctx, target)
    # Two-sided: one device of every nonzero weight sits at the 9 uS floor; half of them read below it.
    assert stats["target_at_floor_fraction"] >= 0.5 and stats["naive"]["stuck_fraction"] == 0
    assert 0.15 < stats["naive"]["below_floor_fraction"] < 0.35
    ctx, target, bundle, _ = context(TWO_SIDED, rate_ppm=50000)
    stats = runtime.device_statistics(ctx, target)
    assert stats["naive"]["stuck_fraction"] == pytest.approx(0.05, abs=0.002)
    assert stats["fault_aware"]["stuck_fraction"] == stats["naive"]["stuck_fraction"]
    rewrite = stats["rewrite"]
    assert rewrite["clipped_targets_low"] + rewrite["clipped_targets_high"] == rewrite["clipped_targets"] > 0
    assert rewrite["programmed_active"] + rewrite["programmed_reference"] > 0
    assert stats["fault_aware"]["apparent_residual_rms"] < stats["naive"]["apparent_residual_rms"]
    records = runtime.deployment_records(ctx, target, ctx.evaluate(target))
    assert records["mapping"] == TWO_SIDED and records["span_us_per_unit_q"] == 79.2
    assert records["clean_mapped_readout_max_abs_error"] < 1e-5
    base, target, _, _ = context(REFERENCE, rate_ppm=50000)
    stats = runtime.device_statistics(base, target)
    assert stats["target_at_floor_fraction"] < 0.01   # only q = -1 puts an active device at 9 uS
    assert stats["rewrite"]["clipped_targets"] == (stats["rewrite"]["clipped_targets_low"]
                                                   + stats["rewrite"]["clipped_targets_high"])
