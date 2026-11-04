"""Constant-LR horizon extension: explicit admission, resume hashes and declared continuation."""

from dataclasses import asdict, replace
import json

import pytest

from ebl.runtime.store import content_hash
from ebl.schema.opt_cmo_drn import compatible_config_hashes, parse_config, resume_config_hashes
import ebl.lifecycle.slurm_recovery as s

DENSE = {'update_rule': 'direct_adam', 'target_bounds': 'gmin_gmax', 'max_write_attempts': None,
         'write_noise_factor': 0., 'learning_rate': 1e-4, 'arm': 'learning'}


def legacy_hash(config):
    value = asdict(config)
    for key in ('lr_schedule', 'warmup_updates', 'extends_updates'):value['recovery'].pop(key)
    return content_hash(value)


def test_extension_admits_only_longer_constant_learning():
    c = parse_config({'recovery': {**DENSE, 'updates': 4096, 'extends_updates': 2048}})
    assert c.recovery.extends_updates == 2048
    for change in ({'extends_updates': 4096}, {'extends_updates': -1}, {'arm': 'hold'},
                   {'lr_schedule': 'warmup_cosine', 'warmup_updates': 10}):
        with pytest.raises(ValueError):
            parse_config({'recovery': {**DENSE, 'updates': 4096, 'extends_updates': 2048, **change}})


def test_extension_resumes_original_horizon_states_but_validation_does_not_accept_them():
    extended = parse_config({'recovery': {**DENSE, 'updates': 4096, 'extends_updates': 2048}})
    original = parse_config({'recovery': {**DENSE, 'updates': 2048}})
    other = parse_config({'recovery': {**DENSE, 'updates': 2048, 'learning_rate': 3e-4}})
    resumable = resume_config_hashes(extended)
    assert legacy_hash(original) in resumable and content_hash(asdict(original)) in resumable
    assert legacy_hash(other) not in resumable
    assert legacy_hash(original) not in compatible_config_hashes(extended)
    assert resume_config_hashes(original) == compatible_config_hashes(original)
    assert legacy_hash(original) in compatible_config_hashes(original)


def test_declared_continuation_resumes_the_earlier_state_once(tmp_path, monkeypatch):
    origin_run = tmp_path / 'old/run';(origin_run / 'checkpoints').mkdir(parents=True)
    latest = origin_run / 'checkpoints/latest.pt';latest.write_bytes(b'full state at 1958')
    (origin_run / 'metrics.jsonl').write_text(json.dumps({'event': 'recovery_start', 'initial_device_state_sha256': 'device'}) + '\n')
    origin = {'run': str(origin_run), 'path': str(latest), 'sha256': s.digest(latest), 'step': 1958}
    case = {'id': 'random01-noiseless-4096', 'rate': '01', 'config': 'frozen.json', 'continue_from': origin}
    calls = []
    def native(manifest, config, destination, inputs, *, action='train', progress=None):
        calls.append((action, inputs))
        result = destination / 'run';progress(str(result))
        if action == 'train':
            (result / 'checkpoints').mkdir(parents=True);(result / 'checkpoints/final.pt').write_bytes(b'final')
        return result, 0
    monkeypatch.setenv('SLURM_JOB_ID', '100')
    monkeypatch.setattr(s, 'baseline', lambda *a: {'device_state_sha256': 'device', 'device_state': 'deployed.pt'})
    monkeypatch.setattr(s, 'run_native', native)
    monkeypatch.setattr(s, 'inspect', lambda run, complete=True: None)
    monkeypatch.setattr(s, 'artifact', lambda run, kind: str(run / 'checkpoints/final.pt'))
    state = s.case_worker({'output': str(tmp_path / 'out')}, case)
    assert state['state'] == 'complete' and state['continued_from'] == origin
    assert calls[0] == ('train', {'resume': str(latest)}) and calls[1][0] == 'validate'
    other = {**case, 'id': 'other'}
    monkeypatch.setattr(s, 'baseline', lambda *a: {'device_state_sha256': 'another', 'device_state': 'x.pt'})
    with pytest.raises(ValueError, match='another device state'):s.case_worker({'output': str(tmp_path / 'out')}, other)
