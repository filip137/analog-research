import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1]/'campaigns/opt-cmo-corruption/launch/exp17_partner_training.py'
spec = importlib.util.spec_from_file_location('exp17', path)
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)


def test_plan_keeps_frozen_inputs_and_limits_jobs_and_endpoints():
    old, new = exp.previous_lifecycle('std3', 'random3').to_dict(), exp.lifecycle().to_dict()
    for key in old.keys() - {'onchip', 'question', 'lifecycle_id'}:
        assert new[key] == old[key]
    for key in old['onchip'].keys() - {'arms'}:
        assert new['onchip'][key] == old['onchip'][key]
    assert exp.ARRAYS == (271002, 271003, 271004)
    assert len(exp.lifecycle().onchip_cases)*len(exp.lifecycle().onchip.arms)*len(exp.ARRAYS) == 24


def evidence():
    rows, previous = [], []
    for case in ('nominal', 'random_30000ppm'):
        rewrite = {'controller': {'flagged_both': 1, 'programmed_reference': 2}}
        for arm in ('none', exp.HOLD, exp.ACTIVE, exp.PARTNER):
            row = {'array_seed': 271002, 'case': case, 'onchip_arm': arm, 'replay_exact': True,
                   'initial_apparent_sha256': 'p0', 'final_apparent_sha256': arm,
                   'final': {s:{'teacher_kl': .1} for s in exp.SPLITS}, 'cost': {}}
            if arm != 'none': row['fault_rewrite'] = copy.deepcopy(rewrite)
            if arm == exp.PARTNER:
                row.update(write_roles='verified_partner', write_routing={
                    'blocked_training_pairs': 1, 'reference_training_devices': 2})
                row['cost'] = dict(verify_reads=0, device_programs=0, max_onchip_device_write_attempts=4,
                    active_device_writes=10, reference_device_writes=6, device_writes=16,
                    active_write_attempts=11, reference_write_attempts=6, issued_write_attempts=17,
                    stuck_write_attempts=1)
            else:
                previous.append({'tag': 'std3', 'arm': arm, **copy.deepcopy(row)})
            rows.append(row)
    return rows, {'rows': previous}


def test_collection_rejects_changed_controls_and_bad_role_accounting():
    rows, old = evidence()
    assert exp.checked_report(271002, rows, old)['controls_reproduced']
    bad = copy.deepcopy(rows); bad[2]['final_apparent_sha256'] = 'changed'
    with pytest.raises(ValueError, match='control'): exp.checked_report(271002, bad, old)
    bad = copy.deepcopy(rows); bad[3]['cost']['reference_device_writes'] = 8
    with pytest.raises(ValueError, match='accounting'): exp.checked_report(271002, bad, old)
    with pytest.raises(ValueError, match='endpoints'): exp.checked_report(271002, rows[:-1], old)
