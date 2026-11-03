import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.artifacts import sha256_file
from training.drift_recovery import run
from training.drift_recovery_config import DriftRecovery
from training.drift_recovery_shards import screens, trajectories, select_learning_rate
from test_workflow_drift_recovery import settings
from test_workflow_opt import tiny_teacher, tokens
from workflow.drift_recovery import CrossbarAdapter
from workflow.networks.opt_mlp import OptMlpSuffix
from workflow_helpers import Store


def publish(root, result, artifacts):
    (root/'status.json').write_text(json.dumps({'status': 'complete'}))
    (root/'result.json').write_text(json.dumps({'metrics': result, 'artifacts': [
        {'path': str(p.relative_to(root)), 'sha256': sha256_file(p)} for _, p in artifacts]}))


def scientific(value):
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items() if k != 'elapsed_seconds'}
    if isinstance(value, list): return [scientific(v) for v in value]
    return value


def test_split_matches_uninterrupted_with_full_selection_and_sparse_evaluation(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    adapter = CrossbarAdapter(OptMlpSuffix(tiny_teacher(), 2, 8), 'cpu')
    cfg = replace(settings(), canary=False, learning_rate=None, learning_rates=(.0001, .001),
                  curve_validation_sequences=1, heldout_evaluation='final')
    train, valid = tokens(4), tokens(2)
    def execute(name, execution=None):
        root = tmp_path/name
        c = replace(cfg, execution=execution)
        assert DriftRecovery.parse(c.to_dict()) == c
        result, artifacts = run(adapter, c, train, valid, {'transfer_test': tokens(3)}, {}, Store(root), 271001)
        publish(root, result, artifacts)
        return result, root
    full, full_root = execute('full')
    prep, prep_root = execute('prep', {'phase': 'prepare', 'key': '', 'prepared_run': None, 'screen_runs': []})
    screen_paths = []
    for screen in screens(cfg):
        _, path = execute(screen['key'], {'phase': 'screen', 'key': screen['key'],
                         'prepared_run': str(prep_root), 'screen_runs': []})
        screen_paths.append(str(path))
        assert json.loads((path/'screen.json').read_text())['row']['sequences'] == 2
    merged = []
    for trajectory in trajectories(cfg):
        result, path = execute(trajectory['key'], {'phase': 'trajectory', 'key': trajectory['key'],
                              'prepared_run': str(prep_root), 'screen_runs': screen_paths})
        assert result['learning_rate'] == full['learning_rate']
        merged.extend(json.loads((path/'drift_recovery_measurements.json').read_text()))
        actual = torch.load(path/'checkpoints'/f"{trajectory['key']}.pt", weights_only=True)
        expected = torch.load(full_root/'checkpoints'/f"{trajectory['key']}.pt", weights_only=True)
        for a, b in zip(actual['cells'], expected['cells']):
            for k in a:
                if isinstance(a[k], torch.Tensor): assert torch.equal(a[k], b[k])
                else: assert a[k] == b[k]
    expected_rows = json.loads((full_root/'drift_recovery_measurements.json').read_text())
    assert scientific(merged) == scientific(expected_rows)
    for row in merged:
        assert row['metrics']['validation']['sequences'] == 1
        assert row['baseline']['validation']['sequences'] == 1
        assert ('transfer_test' in row['metrics']) == (row['seconds'] == cfg.horizon_seconds)
    digest = sha256_file(prep_root/'preparation.json')
    with pytest.raises(ValueError, match='Incomplete'):
        select_learning_rate(cfg, digest, screen_paths[:-1])
    with pytest.raises(ValueError, match='Duplicate'):
        select_learning_rate(cfg, digest, screen_paths + screen_paths[:1])
    with pytest.raises(ValueError, match='different'):
        select_learning_rate(cfg, '0'*64, screen_paths)
    (prep_root/'status.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='Incomplete native'):
        execute('invalid', {'phase': 'trajectory', 'key': trajectories(cfg)[0]['key'],
                           'prepared_run': str(prep_root), 'screen_runs': screen_paths})


def test_full_matrix_is_unchanged_and_unknown_fields_rejected():
    cfg = DriftRecovery('a'*64)
    assert len(trajectories(cfg)) == 71 and len(screens(cfg)) == 16
    assert sum(len(cfg.times(t['protocol'])) + (len(cfg.intervention_times(t['protocol']))
        if t['arm'] != 'untreated' else 0) for t in trajectories(cfg)) == 1670
    for updates in ({'curve_validation_sequences': 257}, {'heldout_evaluation': 'random'}, {'execution': {}}):
        with pytest.raises(ValueError): DriftRecovery.parse(cfg.to_dict() | updates)
