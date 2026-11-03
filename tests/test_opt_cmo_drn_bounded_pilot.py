"""Readiness, fixed-final selection and finite resource coverage for this pilot."""

from dataclasses import asdict
import json

import pytest

from experiments.artifacts import atomic_write_json, content_hash, sha256_file
from experiments.opt_cmo_drn.config import Config, parse_config, compatible_config_hashes
from workflow.bounded_recovery import BoundedRecoveryExecution, CASES, RATES, pilot_config, winners, readiness_projection


def test_case_grid_and_noise_interventions_are_frozen():
    assert len(CASES)==12 and len(set(CASES))==12
    for arm in ('noiseless','noisy'):
        for _,lr in RATES:
            cfg=pilot_config(Config(),arm,lr);parse_config(asdict(cfg))
            assert cfg.recovery.updates==2048 and cfg.training.effective_batch==32
            assert cfg.recovery.max_write_attempts is None and cfg.recovery.target_bounds=='gmin_gmax'
            assert cfg.recovery.update_rule==('direct_adam' if arm=='noiseless' else 'bounded_thresholded_adam')
            assert cfg.recovery.write_noise_factor==(0. if arm=='noiseless' else 1.)


def test_selection_requires_complete_finite_fixed_final_coverage_and_breaks_ties_by_lr():
    rows={f'{arm}-lr{label}':{'teacher_kl':.1,'updates':2048,'learning_rate':lr}
          for arm in ('noiseless','noisy') for label,lr in RATES}
    assert winners(rows)=={'noiseless':'noiseless-lr1e-5','noisy':'noisy-lr1e-5'}
    rows['noisy-lr3e-4']['teacher_kl']=.09
    assert winners(rows)['noisy']=='noisy-lr3e-4'
    with pytest.raises(ValueError):winners({k:v for k,v in rows.items() if k!='noiseless-lr1e-5'})
    rows['noisy-lr3e-4']['updates']=512
    with pytest.raises(ValueError):winners(rows)
    rows['noisy-lr3e-4'].update(updates=2048,teacher_kl=float('nan'))
    with pytest.raises(ValueError):winners(rows)


def test_readiness_cost_uses_measured_training_evaluation_and_both_save_types():
    metrics=[{'event':'recovery_update','training_step_seconds':x,'peak_memory_bytes':1000} for x in (2.,3.)]
    metrics+=[{'event':'recovery_evaluation_timing','seconds':5.}]
    metrics+=[{'event':'recovery_checkpoint','checkpoint_seconds':x} for x in (7.,4.)]
    assert readiness_projection(metrics)['expected_seconds']==3*2048+5*20+7*17+60
    with pytest.raises(ValueError):readiness_projection(metrics[:-1])


def execution(tmp_path):
    base=tmp_path/'base.json';atomic_write_json(base,asdict(Config()))
    weights=tmp_path/'weights.pt';weights.write_bytes(b'weights')
    state=tmp_path/'state.pt';state.write_bytes(b'physical')
    note=tmp_path/'pilot.md';note.write_text('# Pilot\n')
    settings=tmp_path/'settings.json';atomic_write_json(settings,{
        'output':str(tmp_path/'execution'),'base_config':str(base),'weights':str(weights),
        'weights_sha256':sha256_file(weights),'device_state':str(state),'device_state_sha256':sha256_file(state),
        'hosts':['loulou','riri'],'peak_memory_mib':20480,'gpu_budget_seconds':86400,
        'storage_budget_bytes':100*1024**3,'project_dir':str(tmp_path),'handoff':'pilot.md','job_prefix':'test-'})
    return BoundedRecoveryExecution(settings)


def test_readiness_blocks_sweep_admission_and_preparation_is_idempotent(tmp_path,monkeypatch):
    e=execution(tmp_path);e.initialize();initial=json.dumps(e.state,sort_keys=True)
    e.initialize();assert json.dumps(e.state,sort_keys=True)==initial
    assert len(e.state['jobs'])==15
    submitted=[]
    monkeypatch.setattr(e,'reconcile_failures',lambda:None)
    monkeypatch.setattr(e,'storage_usage',lambda:0)
    monkeypatch.setattr(e,'free_storage',lambda:200*1024**3)
    monkeypatch.setattr(e,'input_copy_bytes',lambda job:0)
    monkeypatch.setattr(e,'submit',lambda name:submitted.append(name))
    e.advance()
    assert submitted==['readiness-noiseless','readiness-noisy']
    assert e.state['budget']['charged_plus_reserved_seconds']<=86400
    assert not e.state.get('selected')


def test_budget_exhaustion_blocks_all_new_admissions(tmp_path,monkeypatch):
    e=execution(tmp_path);e.s['gpu_budget_seconds']=1
    monkeypatch.setattr(e,'reconcile_failures',lambda:None)
    monkeypatch.setattr(e,'storage_usage',lambda:0)
    monkeypatch.setattr(e,'free_storage',lambda:200*1024**3)
    monkeypatch.setattr(e,'submit',lambda name:pytest.fail('Over-budget admission'))
    e.advance();assert 'budget' in e.state['blocked']


def test_old_default_configs_resume_but_cannot_be_reinterpreted_as_bounded_recovery():
    old=asdict(Config())
    for key in ('update_rule','target_bounds'):old['recovery'].pop(key)
    digest=content_hash(old)
    assert digest in compatible_config_hashes(Config())
    for arm in ('noisy','noiseless'):
        assert digest not in compatible_config_hashes(pilot_config(Config(),arm,.0003))


def test_resume_collection_preserves_case_identity_and_charges_interrupted_work_once(tmp_path,monkeypatch):
    from workflow.queue_execution import Execution
    e=execution(tmp_path);e.initialize()
    original='noiseless-lr1e-4';replacement='resume-'+original
    e.state['resume_replacements']={replacement:original}
    e.state['jobs'][replacement]=dict(e.state['jobs'][original])
    e.state['failed']={original:{'attempt':'old-token','charged_seconds':601.}}
    def collect(self,job_id):
        self.state['completed'][replacement]={'attempt':'new-token','charged_seconds':2500.}
    monkeypatch.setattr(Execution,'collect',collect)
    e.collect('test-'+replacement)
    assert e.state['completed'][original]['charged_seconds']==3101.
    assert e.state['completed'][original]['resumed_from_attempt']=='old-token'
    assert replacement not in e.state['jobs'] and replacement not in e.state['completed']
    e.collect('test-'+replacement)
    assert e.state['completed'][original]['charged_seconds']==3101.
