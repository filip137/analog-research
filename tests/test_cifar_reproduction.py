import json
from pathlib import Path

import pytest

from experiments.artifacts import content_hash, sha256_file
from experiments.cifar_crossbar.config import default_config
from experiments.cifar_crossbar.reproduce import ArtifactIndex, recipe


def migrated_run(tmp_path):
    run = tmp_path / 'results' / 'archived' / 'one'
    run.mkdir(parents=True)
    config = default_config()
    (run / 'config.resolved.json').write_text(json.dumps(config))
    weights = tmp_path / 'artifacts' / 'teacher.pt'
    weights.parent.mkdir()
    weights.write_bytes(b'frozen teacher fixture')
    digest = sha256_file(weights)
    manifest = {
        'schema': 'ebl.run', 'run_id': 'one', 'experiment_id': config['experiment_id'],
        'config': {'path': 'config.resolved.json', 'sha256': content_hash(config)},
        'inputs': [{'role': 'teacher_weights', 'path': '/unavailable/teacher.pt', 'sha256': digest}],
        'runtime': {'python': '3.12'},
    }
    (run / 'manifest.json').write_text(json.dumps(manifest))
    inventory = tmp_path / 'inventory.jsonl'
    inventory.write_text(json.dumps({'destination': 'artifacts/teacher.pt', 'sha256': digest}) + '\n')
    return run, weights, ArtifactIndex(tmp_path, inventory)


def test_command_uses_verified_local_inputs_without_changing_the_archive(tmp_path):
    run, weights, index = migrated_run(tmp_path)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    result = recipe(run, tmp_path / 'replay', index=index, python='python')
    assert result['command'][-2:] == ['--teacher-weights', str(weights)]
    assert result['command'][1:4] == ['-m', 'ebl', 'train']
    assert result['training_launched'] is False
    assert result['unset_environment'] == ['EBL_SOURCE_RECEIPT']
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    assert not (tmp_path / 'replay').exists()


def test_changed_input_cannot_be_replayed(tmp_path):
    run, weights, index = migrated_run(tmp_path)
    weights.write_bytes(b'different teacher')
    with pytest.raises(ValueError, match='digest changed'):
        recipe(run, tmp_path / 'replay', index=index)


def test_missing_input_does_not_fall_back_to_an_old_path(tmp_path):
    run, weights, index = migrated_run(tmp_path)
    weights.unlink()
    with pytest.raises(FileNotFoundError, match='No retained local artifact'):
        recipe(run, tmp_path / 'replay', index=index)


def test_changed_scientific_config_is_rejected(tmp_path):
    run, _, index = migrated_run(tmp_path)
    (run / 'config.resolved.json').write_text(json.dumps(default_config(seed=999)))
    with pytest.raises(ValueError, match='configuration changed'):
        recipe(run, tmp_path / 'replay', index=index)


def test_inventory_cannot_resolve_outside_migrated_root(tmp_path):
    run, _, index = migrated_run(tmp_path)
    index.paths['bad'] = {'../outside.pt'}
    with pytest.raises(ValueError, match='escapes migrated root'):
        index.resolve('bad')


def test_output_must_be_new_and_outside_historical_run(tmp_path):
    run, _, index = migrated_run(tmp_path)
    for output in (run, run / 'replay'):
        with pytest.raises(ValueError, match='new output directory'):
            recipe(run, output, index=index)
