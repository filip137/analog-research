"""First-pilot gate, duplicate prevention and scheduled Slurm observations."""
import json
from pathlib import Path
import subprocess
import pytest
from experiments.sparse_equilibrium import launch as l,watch as w


def manifest(tmp_path):
    source=tmp_path/'source';source.mkdir()
    return {'source':str(source),'output':str(tmp_path/'out'),'files':{},'source_commit':'frozen',
        'account':'umg@v100','partition':'gpu_p13','qos':'qos_gpu-dev','constraint':'v100-32g',
        'cpus':10,'concurrency':2,'cases':[{'id':'mlp'},{'id':'drn'}],'job_prefix':'sea-',
        'begin_slurm':'2026-11-09T23:00:00','deadline_slurm':'2026-11-10T06:00:00',
        'deadline':'2026-11-10T06:00:00+01:00'}


def test_commands_have_same_wrapper_and_environment_with_bounded_smoke(tmp_path):
    m=manifest(tmp_path);p=tmp_path/'manifest.json'
    a=l.command(p,m,'canary');b=l.command(p,m,'production')
    assert a[-3:-1]==b[-3:-1]
    for x in ('--begin=2026-11-09T23:00:00','--export=ALL','--gres=gpu:1','--cpus-per-task=10'):
        assert x in a and x in b
    assert not any(x.startswith('--mem') for x in a+b)
    assert '--array=0-0' in a and '--time=00:20:00' in a
    assert '--array=0,1%2' in b and '--time=02:00:00' in b


def test_failed_or_missing_canary_blocks_release(tmp_path,monkeypatch):
    m=manifest(tmp_path);p=tmp_path/'m.json';l.s.write(p,m)
    monkeypatch.setattr(l,'candidate',lambda *a:None)
    monkeypatch.setattr(l.s,'output',lambda argv:pytest.fail('Ungated submit'))
    with pytest.raises(ValueError,match='one canary'):l.submit(p,'production')


def test_resume_does_not_duplicate_submitted_or_uncertain_job(tmp_path,monkeypatch):
    m=manifest(tmp_path);p=tmp_path/'m.json';l.s.write(p,m);ledger=Path(m['output'])/'scheduler.json'
    l.s.write(ledger,{'jobs':[{'stage':'canary','parent':'100'}]})
    monkeypatch.setattr(l.s,'output',lambda argv:pytest.fail('Duplicate submit'))
    assert l.submit(p,'canary')=='100'
    l.s.write(ledger,{'jobs':[],'submitting':{'stage':'canary'}})
    with pytest.raises(ValueError,match='Uncertain'):l.submit(p,'canary')


def test_uncertain_sbatch_is_persisted(tmp_path,monkeypatch):
    m=manifest(tmp_path);p=tmp_path/'m.json';l.s.write(p,m)
    monkeypatch.setattr(l,'candidate',lambda *a:None)
    def broken(argv):raise RuntimeError('SSH lost after submission')
    monkeypatch.setattr(l.s,'output',broken)
    with pytest.raises(RuntimeError):l.submit(p,'canary')
    assert l.s.read(Path(m['output'])/'scheduler.json')['submitting']['stage']=='canary'
    with pytest.raises(ValueError,match='Uncertain'):l.submit(p,'canary')


def test_pending_and_terminal_probe_states():
    run={'id':'test','case':'canary'}
    data={'jobs':[{'stage':'canary'}],'items':{'canary':{'allocation':{'state':'PENDING','reason':'BeginTime'},
        'parent':'100','index':0,'has_artifacts':False,'metric_bytes':0,'terminal':{},'valid':False,
        'log_error':None,'log_tail':'','heartbeat_at':None}}}
    assert w.classify(run,data)['state']=='queued'
    item=data['items']['canary'];item['allocation']={'state':'COMPLETED','exit_code':'0:0'}
    assert w.classify(run,data)['state']=='failed'
    item['valid']=True
    assert w.classify(run,data)['complete_evidence']
    assert w.classify({'id':'later','case':'drn'},data)['state']=='queued'


def test_missing_optional_slurm_variable_uses_authoritative_query():
    assert l.s.task_index({'SLURM_JOB_ID':'123'},query=lambda argv:'JobId=123 ArrayJobId=123 ArrayTaskId=0')==0
    with pytest.raises(ValueError):l.s.task_index({})


def test_wrapper_parses_and_resolves_from_manifest_not_spool():
    wrapper=Path(__file__).parents[1]/'experiments/sparse_equilibrium/recovery.slurm'
    subprocess.run(['bash','-n',str(wrapper)],check=True)
    text=wrapper.read_text()
    assert 'dirname "$1"' in text and '${SLURM_ARRAY_TASK_ID}' not in text


def test_watch_uses_declared_python_instead_of_old_login_default(monkeypatch,capsys):
    import io
    request={'runs':[{'id':'a','case':'canary','remote_manifest':'/tmp/source/launch/manifest.json',
                      'remote_python':'/module/python311/bin/python'}]}
    monkeypatch.setattr(w.sys,'stdin',io.StringIO(json.dumps(request)))
    def query(argv,**kwargs):
        assert argv[-1].startswith('/module/python311/bin/python -c ')
        return json.dumps({'jobs':[{'stage':'canary'}],'items':{}})
    monkeypatch.setattr(w.subprocess,'check_output',query)
    w.main();assert json.loads(capsys.readouterr().out)['runs']['a']['probe_error']=='Missing submitted task'
