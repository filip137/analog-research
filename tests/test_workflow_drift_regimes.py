import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.artifacts import sha256_file
from training.drift_family import ReramFamily
from training.drift_recovery import tensor_hash
from training.drift_regimes_config import DriftRegimes
from workflow.drift_regimes import endpoint_weights
from workflow.lifecycle import parse_lifecycle, stage_selections, StageSelection, provenance, verify_provenance
from test_workflow_cmo import cmo_inputs, small, case_doc
from workflow_helpers import run_stage


def config(**changes):
    base = DriftRegimes('a' * 64, 'synthetic', 'b' * 64, 'c' * 64, 'R0', asdict(ReramFamily()))
    return replace(base, **changes)


@pytest.mark.parametrize('key,value', [('arm', 'A8'), ('deployment_attempts', 1), ('deployment_threshold', -1.),
                                      ('evaluation_seconds', [60, 0]), ('canary', 1), ('extra', 1),
                                      ('law', {'rho_star': .3}), ('evaluation_split', 'test')])
def test_settings_fail_closed(key, value):
    raw = config().to_dict()
    raw[key] = value
    with pytest.raises(ValueError):
        DriftRegimes.parse(raw)


def test_settings_round_trip_and_endpoint_weights():
    raw = config().to_dict()
    assert DriftRegimes.parse(json.loads(json.dumps(raw))) == DriftRegimes.parse(raw)
    unverified = config(deployment_attempts=1, deployment_threshold=None).to_dict()
    assert DriftRegimes.parse(unverified).deployment_threshold is None
    w = endpoint_weights([0., 60., 3600., 86400.])
    assert w.sum() == pytest.approx(1)
    assert float(w @ np.array([1., 1., 1., 1.])) == pytest.approx(1)
    assert float(w @ np.array([0., 60., 3600., 86400.])) == pytest.approx(86400 / 2)


@pytest.mark.parametrize('law', [dict(), dict(rho_star=1., persistent_share=.7)])
def test_native_stage_deploys_by_pv_releases_after_it_and_replays(cmo_inputs, tmp_path, law):
    teacher, digest = cmo_inputs
    lc = small(**{'onchip': None, 'defects.cases': [case_doc()], 'data.corpus': {'name': 'synthetic', 'sha256': digest},
                  'data.sequence_length': 9, 'data.max_examples': 4, 'hwa.arms': [{'id': 'digital', 'method': 'none'}]})
    results = {}
    for selection in stage_selections(lc):
        if selection.kind != 'deploy':
            results[selection.stage_id] = run_stage(lc, selection, results, teacher, tmp_path / 'parents')
    directory, result = results['hwa__digital']
    source = directory / next(a['path'] for a in result['artifacts'] if a['kind'] == 'hwa_source')
    corpus = torch.load(Path(os.environ['EBL_CORPUS_ROOT']) / 'synthetic.pt', weights_only=False)
    tokens = corpus['splits']['validation'][:18].reshape(2, 9).contiguous()
    cfg = config(source_sha256=sha256_file(source), evaluation_corpus_sha256=digest,
                 cohort_sha256=tensor_hash(tokens), law=asdict(ReramFamily(**law)),
                 assignment_seeds=(271001,), evaluation_sequences=2,
                 evaluation_seconds=(0., 60., 3600.), canary=True)
    doc = lc.to_dict()
    doc['drift_regimes'] = cfg.to_dict()
    custom = parse_lifecycle(doc)
    assert [s.kind for s in stage_selections(custom)][-1] == 'drift_regimes'
    verify_provenance(provenance(lc, 'hwa_source', hwa_arm='digital'), custom, 'hwa_source', hwa_arm='digital')
    directory, result = run_stage(custom, StageSelection('drift_regimes', 'digital', 271001), results, teacher,
                                  tmp_path / 'run')
    metrics = result['metrics']
    assert metrics['source_unchanged'] and metrics['replay_exact'] and metrics['snapshots'] == 3
    cost = metrics['cost']
    n = cost['physical_devices']
    assert n > 0 and n % 2 == 0          # active + reference device per weight (tiny test network)
    assert n <= cost['deployment_reset_attempts'] <= 4 * n
    assert sum(cost['accepted_at_attempt']) == n
    assert cost['deployment_verify_reads'] == cost['deployment_reset_attempts'] - cost['deployment_capped_devices']
    # Release is the end of deployment P&V: at most three 60-s verify rounds.
    assert 0 < cost['release_seconds'] <= 180
    rows = json.loads((directory / 'drift_regimes_measurements.json').read_text())
    assert [r['seconds'] for r in rows] == [0., 60., 3600.]
    assert all(r['clock_seconds'] == r['release_seconds'] + r['seconds'] for r in rows)
    assert all(r['canary'] and r['metrics']['scored_tokens'] == 16 for r in rows)
    assert all(r['bounds']['active']['devices'] == n // 2 for r in rows)
    assert rows[-1]['metrics']['teacher_kl'] > 0
    assert metrics['mean_teacher_kl'] == pytest.approx(
        float(endpoint_weights([0., 60., 3600.]) @ np.array([r['metrics']['teacher_kl'] for r in rows])))
    again_dir, again = run_stage(custom, StageSelection('drift_regimes', 'digital', 271001), results, teacher,
                                 tmp_path / 'again')
    assert again['metrics']['final_physical_sha256'] == metrics['final_physical_sha256']


def test_lifecycle_rejects_other_branches_and_windows(cmo_inputs):
    teacher, digest = cmo_inputs
    lc = small(**{'onchip': None, 'defects.cases': [case_doc()], 'hwa.arms': [{'id': 'digital', 'method': 'none'}]})
    doc = lc.to_dict()
    doc['drift_regimes'] = config(assignment_seeds=(999,)).to_dict()
    with pytest.raises(ValueError, match='assignment'):
        parse_lifecycle(doc)
    doc['drift_regimes'] = config(law=asdict(ReramFamily(program_min=10.)),
                                  assignment_seeds=lc.devices.assignment_seeds[:1]).to_dict()
    with pytest.raises(ValueError, match='window'):
        parse_lifecycle(doc)
