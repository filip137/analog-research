"""Verified-partner learning: polarity, controller information, caps and native replay."""
import copy
import json

import pytest
import torch

from core.tensor_io_native import load
from core.contracts import ConfigError
from hardware.devices.cmo import CmoMatrix, HALF_RANGE_US
import methods.programming.lifecycle as program_verify
from ebl.schema.lifecycle import parse_lifecycle
from methods.onchip import ThresholdedWriteAdam
from test_workflow_cmo import cmo_inputs, case_doc, small, template
from test_workflow_fault_rewrite import arms, VERIFY_PAIR
from workflow_helpers import run_lifecycle


def matrix():
    mask = torch.tensor([[False, True, False, True, True], [False, False, True, True, False]])
    m = CmoMatrix(5, 7, 2., 1., 'cpu', (mask, torch.full((2, 5), 30.)))
    m.program(torch.zeros(5))
    return m


def test_routes_use_detection_not_the_true_map_and_freeze():
    flags = torch.tensor([[False, True, False, True, False], [False, False, True, True, False]])
    roles = program_verify.verified_write_roles(flags)
    assert roles.tolist() == [0, 1, 0, -1, 0]
    flags.zero_()
    assert roles.tolist() == [0, 1, 0, -1, 0]


def test_reference_sign_fixed_partners_missed_fault_and_no_extra_reads():
    m = matrix(); before = m.g.clone(); cost = m.cost()
    roles = torch.tensor([0, 1, 0, -1, 0], dtype=torch.int8)
    m.write(torch.tensor([.1, .1, -.1, .1, .1]), noise_factor=0., roles=roles)
    assert torch.allclose(m.read()[:3] - (before[0] - before[1])[:3] / HALF_RANGE_US,
                          torch.tensor([.1, .1, -.1]), atol=1e-7)
    assert torch.equal(m.g[0, [1, 3, 4]], before[0, [1, 3, 4]])
    assert torch.equal(m.g[1, [0, 2, 3, 4]], before[1, [0, 2, 3, 4]])
    assert m.cost()['device_writes'] - cost['device_writes'] == 3
    assert m.cost()['stuck_write_attempts'] - cost['stuck_write_attempts'] == 1
    assert m.cost()['verify_reads'] == 0
    m.verify_permanence()


def test_all_active_routing_is_bitwise_legacy_including_noise_stream():
    old, new = matrix(), matrix()
    for delta in (torch.tensor([.1, 0., -.05, .03, .2]), torch.zeros(5), torch.full((5,), -.07)):
        old.write(delta); new.write(delta, roles=torch.zeros(5, dtype=torch.int8))
        assert torch.equal(old.g, new.g)
        assert torch.equal(old.writes, new.writes)
        assert torch.equal(old.generator.get_state(), new.generator.get_state())
        assert old.cost() == new.cost()


def test_role_device_state_restores_rng_reference_counts_and_legacy_state():
    m = matrix(); original = m.state_dict()
    roles = torch.tensor([0, 1, 0, -1, 0], dtype=torch.int8)
    m.write(torch.full((5,), .1), roles=roles)
    replay = matrix(); replay.load_state_dict(m.state_dict())
    for x in (m, replay): x.write(torch.full((5,), -.2), roles=roles)
    assert torch.equal(m.g, replay.g) and m.cost() == replay.cost()
    assert torch.equal(m.reference_writes, replay.reference_writes)
    replay.load_state_dict(original)
    assert 'reference_writes' not in replay.state_dict()


def test_cap_counts_missed_faults_and_excludes_blocked_pairs():
    m = matrix(); roles = torch.tensor([0, 1, 0, -1, 0], dtype=torch.int8)
    w = ThresholdedWriteAdam(m, learning_rate=.1, threshold_sigma=.01, accumulate_updates=True,
                            acceptance_percent=2., noise_scale=1., max_write_attempts_per_device=4,
                            write_roles=roles)
    cost = m.cost()
    for _ in range(10): w.step(torch.ones(5))
    assert w.attempt_counts.tolist() == [4, 4, 4, 0, 4]
    assert w.pending[3] == 0
    assert m.cost()['device_writes'] - cost['device_writes'] == 12
    assert m.cost()['stuck_write_attempts'] - cost['stuck_write_attempts'] == 4
    assert torch.equal(w.state_dict()['write_role_map'], roles)
    assert w.diagnostics()['reference_training_devices'] == 1


def document():
    d = template()
    d['onchip'].update(update_law='cmo_thresholded_write_adam', threshold_sigma=2.,
                       accumulate_updates=True, max_write_attempts_per_device=4)
    d['onchip']['arms'] = arms(
        {'id': 'active', 'weights': 'learn', 'calibration': False, 'fault_rewrite': VERIFY_PAIR},
        {'id': 'partner', 'weights': 'learn', 'calibration': False, 'fault_rewrite': VERIFY_PAIR,
         'write_roles': 'verified_partner'})
    return copy.deepcopy(d)


def test_schema_distinguishes_roles_and_preserves_omitted_legacy_field():
    lc = parse_lifecycle(document())
    assert parse_lifecycle(lc.to_dict()) == lc
    assert 'write_roles' not in lc.onchip.arms[1].to_dict()
    assert lc.onchip.arms[2].write_roles == 'verified_partner'


@pytest.mark.parametrize('change', ['hold', 'oracle', 'active', 'missing', 'unknown'])
def test_schema_rejects_ineligible_role_policy(change):
    d = document(); arm = d['onchip']['arms'][-1]
    if change == 'hold': arm['weights'] = 'hold'
    if change == 'oracle': arm['fault_rewrite'] = {'detection': 'oracle', 'roles': 'pair', 'seed': 241016}
    if change == 'active': arm['fault_rewrite']['roles'] = 'active'
    if change == 'missing': del arm['fault_rewrite']
    if change == 'unknown': arm['write_roles'] = 'both'
    with pytest.raises(ConfigError): parse_lifecycle(d)


def test_native_partner_training_replays_and_matches_controls(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    doc = document()
    lc = small(**{'data.corpus': {'name': 'synthetic', 'sha256': digest},
        'data.sequence_length': 9, 'data.max_examples': 8,
        'defects.cases': [case_doc(), case_doc('random', 50000)],
        'hwa.arms': [{'id': 'digital', 'method': 'none'}],
        'onchip.batch_size': 4, 'onchip.epochs': 1, 'onchip.learning_rate': .1,
        'onchip.update_law': 'cmo_thresholded_write_adam', 'onchip.threshold_sigma': .01,
        'onchip.accumulate_updates': True, 'onchip.max_write_attempts_per_device': 4,
        'onchip.arms': doc['onchip']['arms'] + [
            {'id': 'rewritten', 'weights': 'hold', 'calibration': False, 'fault_rewrite': VERIFY_PAIR}]})
    run, result = run_lifecycle(lc, weights, tmp_path/'runs')['onchip__digital__271001']
    assert result['metrics']['exact_replays'] == 8
    rows = {(r['case'], r['onchip_arm']): r for r in json.loads((run/'measurements.json').read_text())}
    assert rows['nominal', 'active']['final'] == rows['nominal', 'partner']['final']
    for case in lc.onchip_cases:
        row = rows[case.label, 'partner']; cost = row['cost']
        assert row['fault_rewrite'] == rows[case.label, 'active']['fault_rewrite']
        assert row['fault_rewrite']['rewritten_apparent_sha256'] == rows[case.label, 'rewritten']['final_apparent_sha256']
        assert cost['verify_reads'] == 0 and cost['max_onchip_device_write_attempts'] <= 4
        assert cost['active_write_attempts'] + cost['reference_write_attempts'] == cost['issued_write_attempts']
        assert cost['active_device_writes'] + cost['reference_device_writes'] == cost['device_writes']
        saved = load(run/'checkpoints/onchip'/case.label/'partner.pt')
        roles = saved['writer']['write_role_map']
        assert roles.dtype == torch.int8 and row['write_roles'] == 'verified_partner'
        assert row['write_routing']['write_role_map_bytes'] == roles.numel()
    assert rows['random_50000ppm', 'partner']['cost']['reference_device_writes'] > 0
