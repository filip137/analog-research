"""Fail-closed Slurm accounting, immutable submissions and continuation transport."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from workflow import slurm_recovery as s


def manifest(tmp_path):
    root = tmp_path / 'output'
    source = tmp_path / 'source'
    source.mkdir()
    item = source / 'worker.py'
    item.write_text('frozen\n')
    return {'source': str(source), 'output': str(root), 'files': {'worker.py': s.digest(item)},
            'account': 'umg@v100', 'partition': 'gpu_p13', 'qos': 'qos_gpu-dev',
            'constraint': 'v100-32g', 'cpus': 10, 'walltime': '02:00:00',
            'concurrency': 5, 'job_prefix': 'drn-rates-', 'cases': [{'id': f'case{i}'} for i in range(10)],
            'allocation_seconds': 7200, 'gpu_budget_seconds': 216000, 'max_waves': 4,
            'storage_budget_bytes': 200 * 1024**3, 'storage_reserve_per_case': 8 * 1024**3}


def row(index, *, parent='100', raw=None, state='COMPLETED', code='0:0'):
    return f"{raw or str(101+index)}|{parent}_{index}|{state}|{code}|umg@v100|gpu_p13|qos_gpu-dev|v100-32g|600"


def test_optional_environment_and_authoritative_index_fallback():
    assert s.task_index({'SLURM_ARRAY_TASK_ID': '4'}) == 4
    assert s.task_index({}, explicit=7) == 7
    assert s.task_index({'SLURM_JOB_ID': '100'}, query=lambda args: 'JobId=100 ArrayTaskId=0 ArrayJobId=100') == 0
    with pytest.raises(ValueError):
        s.task_index({'SLURM_JOB_ID': '100'}, query=lambda args: 'JobId=100 ArrayTaskId=0-4')
    with pytest.raises(ValueError):
        s.task_index({})


def test_singleton_parent_raw_id_and_complete_multi_task_accounting():
    singleton = s.accounting(row(0, raw='100'), '100', [0])
    assert singleton[0]['raw_id'] == '100'
    rows = s.accounting('\n'.join(row(i) for i in range(10)), '100', list(range(10)))
    assert len(rows) == 10 and sum(v['elapsed_seconds'] for v in rows.values()) == 6000


@pytest.mark.parametrize('text,indices', [('', [0]), (row(0), [0, 1]),
    (row(0)+'\n'+row(0), [0]), (row(0)+'\n'+row(1), [0]), ('bad', [0]),
    (row(0, parent='999'), [0])])
def test_missing_duplicate_unexpected_rows_fail_closed(text, indices):
    with pytest.raises(ValueError):
        s.accounting(text, '100', indices)


def test_resource_contract_and_full_command(tmp_path):
    m = manifest(tmp_path)
    good = s.accounting(row(0), '100', [0])
    s.contract_rows(good, m)
    good[0]['qos'] = 'qos_gpu-t3'
    with pytest.raises(ValueError):
        s.contract_rows(good, m)
    cmd = s.command(tmp_path/'manifest.json', m, list(range(10)), wave=1)
    assert '--array=0,1,2,3,4,5,6,7,8,9%5' in cmd
    for value in ['--time=02:00:00', '--constraint=v100-32g', '--qos=qos_gpu-dev', '--gres=gpu:1']:
        assert value in cmd
    assert '--array=0-0' in s.command(tmp_path/'manifest.json', m, [0], canary=True)
    with pytest.raises(ValueError):
        s.command(tmp_path/'manifest.json', m, [0, 1], canary=True)


def test_source_mutation_blocks_submission(tmp_path):
    m = manifest(tmp_path)
    s.verify_source(m)
    (Path(m['source'])/'worker.py').write_text('changed\n')
    with pytest.raises(ValueError):
        s.verify_source(m)


def test_uncertain_submission_persists_and_never_submits_twice(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    ledger = {'jobs': []}
    calls = []
    def uncertain(argv):
        calls.append(argv)
        raise RuntimeError('connection lost after sbatch')
    monkeypatch.setattr(s, 'output', uncertain)
    with pytest.raises(RuntimeError):
        s.submit_once(tmp_path/'manifest.json', m, ledger, [0], canary=True)
    persisted = s.read(Path(m['output'])/'scheduler.json')
    with pytest.raises(ValueError):
        s.submit_once(tmp_path/'manifest.json', m, persisted, [0], canary=True)
    assert len(calls) == 1


def test_advance_requires_live_canary_gate(tmp_path, monkeypatch):
    m = manifest(tmp_path)
    path = tmp_path/'manifest.json'
    s.write(path, m)
    monkeypatch.setattr(s, 'submit_once', lambda *a, **k: pytest.fail('Ungated submission'))
    assert s.advance(path, release=True)['state'] == 'awaiting_gate'
    s.write(Path(m['output'])/'gate.json', {'gate_passed': True, 'manifest_sha256': 'wrong'})
    with pytest.raises(ValueError):
        s.advance(path, release=True)


@pytest.mark.parametrize('state,code', [('FAILED','1:0'), ('TIMEOUT','0:0'), ('COMPLETED','1:0')])
def test_failed_or_mixed_arrays_do_not_advance(tmp_path, monkeypatch, state, code):
    m = manifest(tmp_path)
    path = tmp_path/'manifest.json'
    s.write(path,m)
    s.write(Path(m['output'])/'scheduler.json', {'jobs': [{'parent':'100','indices':[0,1]}]})
    monkeypatch.setattr(s, 'output', lambda argv: '' if argv[0]=='squeue' else row(0)+'\n'+row(1,state=state,code=code))
    with pytest.raises(ValueError):
        s.advance(path)


def test_pending_jobs_never_trigger_duplicate_submission(tmp_path, monkeypatch):
    m=manifest(tmp_path);path=tmp_path/'manifest.json';s.write(path,m)
    s.write(Path(m['output'])/'scheduler.json', {'jobs':[{'parent':'100','indices':[0]}]})
    monkeypatch.setattr(s,'output',lambda argv:'PENDING\n')
    monkeypatch.setattr(s,'submit_once',lambda *a,**k:pytest.fail('Duplicate'))
    assert s.advance(path)['state']=='queued'


def test_retired_parent_falls_back_to_accounting(tmp_path, monkeypatch):
    m=manifest(tmp_path);path=tmp_path/'manifest.json';s.write(path,m)
    s.write(Path(m['output'])/'scheduler.json', {'jobs':[{'parent':'100','indices':[0]}]})
    def query(argv):
        if argv[0]=='squeue':
            raise subprocess.CalledProcessError(1,argv,stderr='slurm_load_jobs error: Invalid job id specified')
        return row(0)
    monkeypatch.setattr(s,'output',query)
    result=s.advance(path)
    assert result['state']=='awaiting_gate'
    assert result['jobs'][0]['accounting'][0]['state']=='COMPLETED'
    assert s.probe(path)['jobs']['100']['0']['state']=='COMPLETED'
    def unavailable(argv):
        raise subprocess.CalledProcessError(1,argv,stderr='Unable to contact slurm controller')
    monkeypatch.setattr(s,'output',unavailable)
    with pytest.raises(subprocess.CalledProcessError):s.live_job('100')


def test_endpoint_worker_only_validates_and_preserves_accepted_checkpoint(tmp_path,monkeypatch):
    m=manifest(tmp_path);m['operation']='evaluate_accepted_endpoints'
    endpoint=tmp_path/'latest.pt';endpoint.write_bytes(b'held state and optimizer')
    hold=tmp_path/'hold';s.write(hold/'result.json',{'metrics':{'start_held_hashes':['initial']}})
    case={'id':'case0','config':'frozen.json',
          'endpoint':{'path':str(endpoint),'sha256':s.digest(endpoint),'step':1954},
          'baseline':{'device_state_sha256':'device','hold_run':str(hold)}}
    calls=[]
    def native(manifest,config,destination,inputs,*,action,progress):
        assert action=='validate' and inputs=={'weights':str(endpoint)}
        assert config=='frozen.json'
        calls.append(destination)
        run=destination/'run';progress(str(run))
        s.write(run/'result.json',{'metrics':{'checkpoint':{
            'updates':1954,'schema':'opt_cmo_drn.recovery_resume.v1','training_complete':False,
            'primary_state':'held_apparent','initial_device_state_sha256':'device','start_held_hashes':['initial']}}})
        return run,0
    monkeypatch.setenv('SLURM_JOB_ID','100')
    monkeypatch.setattr(s,'run_native',native)
    monkeypatch.setattr(s,'inspect',lambda run:None)
    monkeypatch.setattr(s,'case_worker',lambda *a,**k:pytest.fail('Training invoked'))
    assert s.endpoint_worker(m,case)['state']=='complete'
    assert s.endpoint_worker(m,case)['state']=='complete'
    assert len(calls)==1 and endpoint.read_bytes()==b'held state and optimizer'
    assert s.pending_indices(m)==list(range(1,10))
    bad={**case,'id':'bad','endpoint':{**case['endpoint'],'sha256':'wrong'}}
    with pytest.raises(ValueError,match='digest differs'):s.endpoint_worker(m,bad)


def test_resume_excludes_completed_cases_and_requires_exact_checkpoint_error(tmp_path):
    m=manifest(tmp_path)
    s.write(Path(m['output'])/'cases/case0/state.json', {'state':'complete'})
    s.write(Path(m['output'])/'cases/case1/state.json', {'state':'checkpointed'})
    assert s.pending_indices(m)==list(range(1,10))
    run=tmp_path/'run';(run/'checkpoints').mkdir(parents=True)
    (run/'checkpoints/latest.pt').write_bytes(b'full checkpoint')
    (run/'metrics.jsonl').write_text(json.dumps({'event':'recovery_checkpoint','step':128})+'\n')
    s.write(run/'status.json',{'error':{'type':'TimeoutError','message':s.TIMEOUT_MESSAGE}})
    assert s.checkpointed(run)['step']==128
    s.write(run/'status.json',{'error':{'type':'FloatingPointError','message':'invalid state'}})
    with pytest.raises(ValueError):s.checkpointed(run)


def test_batch_wrapper_resolves_source_from_manifest_even_in_slurm_spool(tmp_path):
    source=tmp_path/'source';(source/'launch').mkdir(parents=True)
    manifest_path=source/'launch/manifest.json';manifest_path.write_text('{}')
    spool=tmp_path/'spool';spool.mkdir()
    wrapper=spool/'slurm_script'
    wrapper.write_text((Path(__file__).parents[1]/'workflow/slurm_recovery.slurm').read_text())
    binary=tmp_path/'bin';binary.mkdir()
    for name,body in [('nvidia-smi','exit 0'),('srun','pwd\nprintf "%s\\n" "$@"')]:
        path=binary/name;path.write_text('#!/bin/sh\n'+body+'\n');path.chmod(0o755)
    env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH']}
    prelude='module() { :; }; export -f module; bash "$@"'
    result=subprocess.run(['bash','-c',prelude,'wrapper',str(wrapper),str(manifest_path),'canary'],
                           env=env,text=True,capture_output=True,check=True)
    assert result.stdout.splitlines()[0]==str(source)
    assert str(manifest_path) in result.stdout and 'canary' in result.stdout
