"""Reject incomplete/incorrect final evaluation before transport completion."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
from ebl.schema.opt_cmo_drn import parse_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('paper_worker', ROOT / 'campaigns/pilots/20261116-fault-paper-grid/run_local.py')
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def bundle(tmp_path):
    config = ROOT / 'examples/opt_cmo_drn/fault_paper_grid/canary/drn6-all-arms.json'
    c = json.loads(config.read_text())
    rows = []
    for arm in c['fault_tuning']['arms']:
        rows.append({'label': arm['label'], 'updates': 0 if arm['method'] in ('hold', 'naive_hold') else 4,
                     'development': {'teacher_kl': .1}, 'held_out': {
                         s['name']: {'cohort': s, 'metrics': {'teacher_kl': .2, 'scored_tokens': s['sequences']*255}}
                         for s in c['fault_tuning']['evaluation_cohorts']}})
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'result.json').write_text(json.dumps({'status':'complete'}))
    return config, rows


@pytest.mark.parametrize('defect', ['missing_domain','wrong_hash','short_test','nonfinite','short_training'])
def test_rejects_incomplete_or_wrong_live_endpoint(tmp_path, defect):
    config, rows = bundle(tmp_path)
    if defect == 'missing_domain': rows[0]['held_out'].pop('generated')
    elif defect == 'wrong_hash':
        rows[0]['held_out']['generated']['cohort'] = copy.deepcopy(rows[0]['held_out']['generated']['cohort'])
        rows[0]['held_out']['generated']['cohort']['cohort_sha256'] = '0'*64
    elif defect == 'short_test': rows[0]['held_out']['generated']['metrics']['scored_tokens'] -= 1
    elif defect == 'nonfinite': rows[0]['held_out']['generated']['metrics']['teacher_kl'] = float('nan')
    else: rows[-1]['updates'] = 3
    (tmp_path / 'artifacts/summary.json').write_text(json.dumps({'arms': rows}))
    with pytest.raises(ValueError): worker.verify(config, tmp_path)


def test_exact_endpoint_coverage_and_native_canaries(tmp_path):
    config, rows = bundle(tmp_path)
    (tmp_path / 'artifacts/summary.json').write_text(json.dumps({'arms': rows}))
    assert worker.verify(config, tmp_path)['status'] == 'complete'
    for file in (ROOT/'examples/opt_cmo_drn/fault_paper_grid/canary').glob('*.json'):
        parsed = parse_config(json.loads(file.read_text()))
        assert len(parsed.fault_tuning.arms) == 8
        assert [s.sequences for s in parsed.fault_tuning.evaluation_cohorts] == [1024,384]
