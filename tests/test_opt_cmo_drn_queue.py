"""Transport invariants: complete coverage, exact dependencies, bounded admission."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from ebl.runtime.store import atomic_write_json, sha256_file
from ebl.schema.opt_cmo_drn import Config
from ebl.lifecycle.queue_execution import Execution, artifact, artifact_removable, removable, runnable, storage_reservation, terminal_failure, terminal_receipt


def test_collection_requires_one_success_and_charges_failures():
    attempts=[{'state':'failed','receipt':json.dumps({'receipt':{'elapsed_seconds':3}})},
              {'state':'complete','target':'fifi','receipt':json.dumps({'receipt':{'elapsed_seconds':7,'run_exit_code':0,'validation_exit_code':0}})}]
    attempt,charge=terminal_receipt({'attempts':attempts})
    assert attempt['target']=='fifi' and charge==10
    with pytest.raises(ValueError):terminal_receipt({'attempts':attempts[:1]})
    with pytest.raises(ValueError):terminal_receipt({'attempts':attempts+attempts[1:]})


def test_physical_branching_and_replica_lifetime():
    jobs={
        'deploy':{'inputs':{},'submitted':True},
        'learning':{'inputs':{'device_state':{'parent':'deploy','kind':'device_state'}}},
        'hold':{'inputs':{'device_state':{'parent':'deploy','kind':'device_state'}}},
        'learning_calibration':{'inputs':{'device_state':{'parent':'learning','kind':'recovery_state'}}},
    }
    assert runnable(jobs,{})==[]
    assert runnable(jobs,{'deploy':{}})==['learning','hold']
    assert not removable('deploy',jobs,{'deploy':{},'learning':{}})
    assert removable('deploy',jobs,{'deploy':{},'learning':{},'hold':{}})
    assert not removable('learning',jobs,{'deploy':{},'learning':{}})
    assert 'learning_calibration' in runnable(jobs,{'deploy':{},'learning':{}})


def test_artifact_refuses_missing_kind_tamper_and_traversal(tmp_path):
    p=tmp_path/'final.pt';p.write_bytes(b'checkpoint')
    a={'kind':'final_weights','path':'final.pt','sha256':sha256_file(p)}
    atomic_write_json(tmp_path/'result.json',{'artifacts':[a]})
    assert artifact(tmp_path,'final_weights')[0]==p
    with pytest.raises(ValueError):artifact(tmp_path,'best_weights')
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError):artifact(tmp_path,'final_weights')
    a['path']='../final.pt'
    atomic_write_json(tmp_path/'result.json',{'artifacts':[a]})
    with pytest.raises(ValueError):artifact(tmp_path,'final_weights')


def execution(tmp_path):
    base=tmp_path/'base.json';atomic_write_json(base,asdict(Config()))
    settings=tmp_path/'settings.json';atomic_write_json(settings,{
        'output':str(tmp_path/'out'),'base_config':str(base),'tier':'full','mapped':str(tmp_path/'mapped.pt'),
        'mapped_sha256':'abc','hosts':['fifi'],'timings':{'update_seconds':2,'selection_seconds':20,'checkpoint_seconds':3,'readiness_seconds':100}})
    return Execution(settings)


def test_phase_never_uses_partial_screen_coverage(tmp_path):
    e=execution(tmp_path);e.tuning_phase('screen')
    assert len(e.state['jobs'])==8
    assert {read_config(j).training.max_updates for j in e.state['jobs'].values()}=={256}
    e.tuning_phase('refine','screen')
    assert e.state['phases']==['screen']
    # Reopening is idempotent and preserves the exact declared transport jobs.
    reopened=Execution(e.settings_path)
    assert reopened.state==e.state


def read_config(j):
    from ebl.schema.opt_cmo_drn import parse_config
    return parse_config(json.loads(Path(j['config']).read_text()))


def test_collection_retry_does_not_recollect_or_relaunch(tmp_path,monkeypatch):
    e=execution(tmp_path);e.s['job_prefix']='study-';e.state['jobs']['x']={};e.state['completed']['x']={'run':'saved'}
    monkeypatch.setattr(e,'queue',lambda *a:pytest.fail('already-collected job must not query or launch'))
    e.collect('study-x')
    with pytest.raises(ValueError):e.collect('other-x')


def test_storage_bounds_include_recovery_save_overlap():
    c=Config()
    assert storage_reservation(replace(c,stage='recover'),'train')==9*1024**3
    assert storage_reservation(c,'validate')<1024**3
    assert storage_reservation(c,'train')==11*1024**3
    assert storage_reservation(replace(c,training=replace(c.training,max_updates=2048)),'train')==8*1024**3


def test_retire_unused_training_state_while_selected_weights_have_consumers():
    jobs={'hwa':{'inputs':{}},'deploy':{'inputs':{'weights':{'parent':'hwa','kind':'best_weights'}}},
          'held-out':{'inputs':{'weights':{'parent':'hwa','kind':'final_weights'}}}}
    completed={'hwa':{},'held-out':{}}
    assert artifact_removable('hwa','training_state',jobs,completed)
    assert artifact_removable('hwa','final_weights',jobs,completed)
    assert not artifact_removable('hwa','best_weights',jobs,completed)
    jobs['deploy']['paused']='await scientific review'
    assert artifact_removable('hwa','best_weights',jobs,completed)
    jobs['deploy']['submitted']=True
    assert not artifact_removable('hwa','best_weights',jobs,completed)


def test_recreate_plan_after_interrupted_registration(tmp_path):
    e=execution(tmp_path)
    e.tuning_phase('screen')
    e.state={'jobs':{},'completed':{},'phases':[]}
    e.save()
    e.tuning_phase('screen')
    assert len(e.state['jobs'])==8


def test_failed_budget_gate_never_submits(tmp_path,monkeypatch):
    e=execution(tmp_path);e.tuning_phase('screen')
    e.s['timings']['readiness_seconds']=24*3600
    monkeypatch.setattr(e,'retire_replicas',lambda:None)
    monkeypatch.setattr(e,'retire_inputs',lambda:None)
    monkeypatch.setattr(e,'submit',lambda *a:pytest.fail('over-budget job submitted'))
    with pytest.raises(ValueError,match='24GPUh'):e.advance()


def test_storage_defers_without_submitting_or_changing_science(tmp_path,monkeypatch):
    e=execution(tmp_path);e.tuning_phase('screen')
    Path(e.s['mapped']).write_bytes(b'weights')
    monkeypatch.setattr(e,'retire_replicas',lambda:None)
    monkeypatch.setattr(e,'retire_inputs',lambda:None)
    monkeypatch.setattr(e,'storage_usage',lambda:99*1024**3)
    monkeypatch.setattr(e,'submit',lambda *a:pytest.fail('storage cap exceeded'))
    e.advance()
    assert len(e.state['storage']['deferred'])==8
    assert not e.state.get('execution_complete')


def test_complete_ladder_expands_exact_matched_recovery_dag(tmp_path,monkeypatch):
    import torch
    import ebl.lifecycle.tuning as tuning
    e=execution(tmp_path)
    teacher=tmp_path/'teacher.pt';teacher.write_bytes(b'frozen teacher')
    base=replace(Config(),teacher_sha256=sha256_file(teacher))
    atomic_write_json(Path(e.s['base_config']),asdict(base))
    e.s.update(local_teacher=str(teacher),project_dir=str(tmp_path),physical_timings={
        'deploy_seconds':17,'calibration_update_seconds':16,'learning_update_seconds':2,'rewrite_update_seconds':.03})
    def complete_rows(expected,runs):
        assert set(expected)==set(runs)
        return [{'id':n,'config':c,'score':0.1,'initial_score':0.2,'mapped_sha256':'abc'} for n,c in expected.items()]
    monkeypatch.setattr(tuning,'verify_coverage',complete_rows)
    e.tuning_phase('screen')
    for phase,previous in [('refine','screen'),('confirm','refine'),('full','confirm')]:
        for n,j in e.state['jobs'].items():
            if j['phase']==previous:e.state['completed'][n]={'run':str(tmp_path/n)}
        e.tuning_phase(phase,previous)
    assert len(e.state['jobs'])==20
    for n in ('full-clean','full-std3'):
        root=tmp_path/n;root.mkdir();path=root/'best.pt'
        payload={'schema':'opt_cmo_drn.weights.v1','contract':{
            'teacher_sha256':base.teacher_sha256,'source_sha256':base.source_sha256,
            'initialization':base.initialization,'solver':asdict(base.solver)},
            'selection':{'step':512,'teacher_kl':0.1}}
        torch.save(payload,path)
        atomic_write_json(root/'result.json',{'artifacts':[{'kind':'best_weights','path':'best.pt','sha256':sha256_file(path)}]})
        e.state['completed'][n]={'run':str(root),'host':'fifi'}
    e.expand_recovery()
    assert len(e.state['jobs'])==68
    # A callback may die after writing native plans but before recording the
    # phase boundary. Repeating that work must not fail on tuple/list JSON encoding.
    e.state['phases'].remove('recovery')
    e.expand_recovery()
    assert len(e.state['jobs'])==68
    for arm in ('clean','std3'):
        for kind in ('none','random'):
            learning=arm+'-'+kind+'-learning'
            calibration=arm+'-'+kind+'-learning_calibration'
            deploy=arm+'-deploy-'+kind
            assert e.state['jobs'][learning]['inputs']['device_state']=={'parent':deploy,'kind':'device_state'}
            assert e.state['jobs'][calibration]['inputs']['device_state']=={'parent':learning,'kind':'recovery_state'}
            for control in ('hold','calibration','rewrite'):
                job=e.state['jobs'][arm+'-'+kind+'-'+control]
                assert job['inputs']['device_state']=={'parent':deploy,'kind':'device_state'}
            c=read_config(e.state['jobs'][learning])
            assert c.recovery.updates==2048 and c.recovery.max_write_attempts==4
            assert c.deployment.assignment_seed==271002
        assert read_config(e.state['jobs']['full-'+arm]).training.max_updates==0


def test_full_filesystem_defers_even_below_study_storage_cap(tmp_path,monkeypatch):
    e=execution(tmp_path);e.tuning_phase('screen')
    Path(e.s['mapped']).write_bytes(b'weights')
    monkeypatch.setattr(e,'retire_replicas',lambda:None)
    monkeypatch.setattr(e,'retire_inputs',lambda:None)
    monkeypatch.setattr(e,'storage_usage',lambda:10*1024**3)
    monkeypatch.setattr(e,'free_storage',lambda:50*1024**2)
    monkeypatch.setattr(e,'submit',lambda *a:pytest.fail('job admitted to full collection filesystem'))
    e.advance()
    assert len(e.state['storage']['deferred'])==8
    assert e.state['storage']['local_free_bytes']==50*1024**2


def test_uncapped_revision_preserves_started_arms_and_exact_deployments(tmp_path):
    e=execution(tmp_path)
    e.state['phases']=['screen','refine','confirm','full','recovery']
    for arm in ('clean','std3'):
        for kind in ('none','random'):
            prefix=arm+'-'+kind
            base=replace(Config(),stage='recover')
            for control in ('learning','learning_calibration'):
                name=prefix+'-'+control
                parent=arm+'-deploy-'+kind if control=='learning' else prefix+'-learning'
                ref={'parent':parent,'kind':'device_state' if control=='learning' else 'recovery_state'}
                cfg=replace(base,recovery=replace(base.recovery,arm=control))
                e.register(name,cfg,{'weights':{'parent':'full-'+arm,'kind':'best_weights'},'device_state':ref},
                           ['fifi'],'recovery',100,200)
                e.register(name+'-held-out',cfg,{'weights':{'parent':name,'kind':'recovery_state'}},
                           ['fifi'],'recovery',10,20,'validate')
    e.state['jobs']['clean-none-learning']['submitted']=True
    e.state['completed']['clean-none-learning']={'run':'saved-capped-reference'}
    original=json.loads(json.dumps(e.state['jobs']['clean-none-learning']))
    e.enable_uncapped_recovery()
    assert len(e.state['jobs'])==20 and len(e.state['superseded_jobs'])==12
    assert e.state['jobs']['clean-none-learning']==original
    for arm in ('clean','std3'):
        for kind in ('none','random'):
            name=arm+'-'+kind+'-uncapped-learning'
            j=e.state['jobs'][name]
            c=read_config(j)
            assert c.recovery.max_write_attempts is None and c.recovery.updates==2048
            assert c.recovery.learning_rate==.003 and c.recovery.threshold_sigma==2
            assert j['inputs']['device_state']=={'parent':arm+'-deploy-'+kind,'kind':'device_state'}
            cal=e.state['jobs'][name+'_calibration']
            assert cal['inputs']['device_state']=={'parent':name,'kind':'recovery_state'}
    saved=json.loads(json.dumps(e.state))
    e.enable_uncapped_recovery()
    assert e.state==saved


def failed_status():
    return {'job':{'state':'attention','reason':'invalid conductance',
                   'spec':{'placements':{'fifi':{'run_dir':'/remote/failed'}}}},
            'attempts':[{'token':'failed-token','target':'fifi','state':'failed',
                'receipt':json.dumps({'alive':False,'reservation_held':False,
                    'receipt':{'finished_at':10.,'run_exit_code':1,'elapsed_seconds':2.}})}]}


def test_failure_release_requires_stopped_worker_and_released_lease():
    status=failed_status()
    assert terminal_failure(status)[0]['token']=='failed-token'
    for key in ('alive','reservation_held'):
        changed=failed_status();r=json.loads(changed['attempts'][0]['receipt']);r[key]=True
        changed['attempts'][0]['receipt']=json.dumps(r)
        assert terminal_failure(changed) is None
    status['job']['state']='queued'
    assert terminal_failure(status) is None


def test_failed_reservation_does_not_block_independent_case_or_complete_dependents(tmp_path,monkeypatch):
    import ebl.lifecycle.queue_execution as module
    e=execution(tmp_path)
    e.s.update(job_prefix='study-',project_dir=str(tmp_path))
    e.state['phases']=['screen','refine','confirm','full','recovery']
    common={'phase':'recovery','storage_reservation_bytes':9*1024**3,'inputs':{}}
    e.state['jobs']={'failed':{**common,'submitted':True},'independent':dict(common),
                     'dependent':{**common,'inputs':{'device_state':{'parent':'failed','kind':'recovery_state'}}}}
    monkeypatch.setattr(e,'queue',lambda *args:failed_status())
    monkeypatch.setattr(module,'run',lambda *args,**kwargs:'')
    monkeypatch.setattr(module,'inspect_run',lambda *args,**kwargs:{'valid':True,'process_complete':False})
    monkeypatch.setattr(e,'retire_replicas',lambda:None)
    monkeypatch.setattr(e,'retire_inputs',lambda:None)
    monkeypatch.setattr(e,'storage_usage',lambda:90*1024**3)
    monkeypatch.setattr(e,'free_storage',lambda:100*1024**3)
    submitted=[]
    def submit(name):
        submitted.append(name);e.state['jobs'][name]['submitted']=True
    monkeypatch.setattr(e,'submit',submit)
    e.advance()
    assert submitted==['independent']
    assert 'failed' in e.state['failed'] and 'failed' not in e.state['completed']
    assert e.state['failed']['failed']['retry_authorized'] is False
    assert (Path(e.state['failed']['failed']['run']).parent/'queue-failure.json').exists()
    assert not e.state['jobs']['dependent'].get('submitted')
    assert not e.state.get('execution_complete')
    assert e.state['storage']['reserved_bytes']==9*1024**3


def test_noiseless_lr_revision_keeps_failed_baseline_and_pauses_old_unsubmitted_cases(tmp_path):
    e=execution(tmp_path)
    e.s['physical_timings']={'calibration_update_seconds':15.}
    e.state['phases']=['screen','refine','confirm','full','recovery','recovery_uncapped']
    cfg=replace(Config(),stage='recover',deployment=replace(Config().deployment,fault_kind='none',fault_rate=0.),
                recovery=replace(Config().recovery,max_write_attempts=None))
    refs={'weights':{'parent':'full-clean','kind':'best_weights'},
          'device_state':{'parent':'clean-deploy-none','kind':'device_state'}}
    for name in ('clean-none-uncapped-learning','std3-none-uncapped-learning'):
        e.register(name,cfg,refs,['fifi'],'recovery_uncapped',4000,7400)
    baseline=e.state['jobs']['clean-none-uncapped-learning']
    baseline['submitted']=True
    digest=sha256_file(Path(baseline['config']))
    e.state['failed']={'clean-none-uncapped-learning':{'attempt':'preserved'}}
    e.enable_noiseless_lr()
    assert e.state['failed']=={'clean-none-uncapped-learning':{'attempt':'preserved'}}
    assert sha256_file(Path(baseline['config']))==digest
    assert not baseline.get('paused')
    assert e.state['jobs']['std3-none-uncapped-learning']['paused']
    cases=e.state['noiseless_lr_revision']['cases']
    assert len(cases)==4
    assert runnable(e.state['jobs'],{'full-clean':{},'clean-deploy-none':{}})==cases
    for name in cases:
        j=e.state['jobs'][name];c=read_config(j)
        assert j['inputs']==refs
        assert c.recovery.updates==2048 and c.recovery.max_write_attempts is None
        assert c.recovery.threshold_sigma==2 and c.recovery.write_noise_factor==0.
        assert c.recovery.development_every==128 and not c.recovery.retain_final_optimizer
    snapshot=json.loads(json.dumps(e.state))
    e.enable_noiseless_lr()
    assert e.state==snapshot


def test_input_retirement_tracks_future_consumers_per_host_and_restages_retired_copy(tmp_path,monkeypatch):
    e=execution(tmp_path)
    e.s.update(remote_python='python',remote_root='/remote')
    e.state['phases']=['full']
    source=tmp_path/'source';source.mkdir();weights=source/'best.pt';weights.write_bytes(b'weights')
    digest=sha256_file(weights)
    atomic_write_json(source/'result.json',{'artifacts':[{'kind':'best_weights','path':'best.pt','sha256':digest}]})
    e.state['completed']={'source':{'run':str(source),'host':'fifi'},'deploy':{'host':'fifi'}}
    e.state['jobs']={
        'source':{'inputs':{},'hosts':['fifi','trex']},
        'deploy':{'inputs':{},'hosts':['fifi','trex']},
        'learning':{'inputs':{'device_state':{'parent':'deploy','kind':'device_state'}},'hosts':['fifi','trex']},
        'calibration':{'inputs':{'device_state':{'parent':'learning','kind':'recovery_state'},
                               'weights':{'parent':'source','kind':'best_weights'}},'hosts':['fifi','trex']}}
    # A completed deployment manifest is needed for the already known bank input.
    deploy=tmp_path/'deploy';deploy.mkdir();atomic_write_json(deploy/'result.json',{'artifacts':[]})
    e.state['completed']['deploy']['run']=str(deploy)
    e.state['input_copies']={host+':'+digest:{'host':host,'sha256':digest,'path':'/remote/'+digest+'.pt'}
                             for host in ('fifi','trex')}
    retired=[]
    monkeypatch.setattr(e,'ssh',lambda host,args:retired.append(host))
    e.retire_inputs()
    assert retired==['trex']
    assert not e.state['input_copies']['fifi:'+digest].get('retired')
    assert e.state['input_copies']['trex:'+digest]['retired']
    transferred=[]
    monkeypatch.setattr(e,'transfer',lambda host,local,remote:transferred.append((host,local)))
    e.resolve_input({'path':str(weights),'sha256':digest},'trex')
    assert transferred==[('trex',weights)]
    assert not e.state['input_copies']['trex:'+digest].get('retired')


def test_failed_replica_retirement_preserves_canonical_continuation(tmp_path,monkeypatch):
    e=execution(tmp_path);e.s['remote_python']='python'
    root=tmp_path/'failed';(root/'checkpoints').mkdir(parents=True)
    checkpoint=root/'checkpoints/latest.pt';checkpoint.write_bytes(b'full optimizer and device state')
    e.state['failed']={'case':{'bundle_valid':True,'run':str(root),'remote_run':'/remote/case','host':'fifi'}}
    calls=[]
    monkeypatch.setattr(e,'ssh',lambda host,args:calls.append((host,args)))
    e.retire_failed_replicas()
    assert checkpoint.read_bytes()==b'full optimizer and device state'
    assert len(calls)==1 and calls[0][0]=='fifi'
    record=e.state['failed']['case']['retired_checkpoints'][0]
    assert record['sha256']==sha256_file(checkpoint)
    assert record['path']=='checkpoints/latest.pt'
    e.retire_failed_replicas()
    assert len(calls)==1
