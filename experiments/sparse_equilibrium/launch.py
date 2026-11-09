"""Bounded Slurm transport using the repository's accounting and lock helpers.

The run-watch owner reviews the completed canary, calls release once, then owns
collection. No background process is allowed to submit an ungated array.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
from ebl.lifecycle import slurm_recovery as s


def command(path,m,stage):
    if stage not in ('canary','production'):raise ValueError('Unknown stage')
    local={**m,'walltime':'00:20:00' if stage=='canary' else '02:00:00'}
    cmd=s.command(path,local,[0] if stage=='canary' else [0,1],canary=stage=='canary',wave=1)
    # IDRIS assigns host memory from the allocation; explicit --mem is forbidden.
    cmd[1:1]=['--begin='+m['begin_slurm'],'--deadline='+m['deadline_slurm']]
    return cmd


def candidate(path,m):
    s.verify_source(m)
    if m['cpus']!=10 or m['concurrency']!=2 or len(m['cases'])!=2:
        raise ValueError('Changed first-pilot resource envelope')
    if (m['account'],m['partition'],m['qos'],m['constraint'])!=('umg@v100','gpu_p13','qos_gpu-dev','v100-32g'):
        raise ValueError('Changed reviewed allocation')
    static=s.read(Path(m['source'])/'launch/static.json')
    if static.get('passed') is not True or static.get('source_commit')!=m['source_commit']:
        raise ValueError('Missing passing static gate for this source')
    if m['commands']!={stage:command(path,m,stage) for stage in ('canary','production')}:
        raise ValueError('Effective commands differ from reviewed commands')


def completion(m,stage,index=0):
    name='canary' if stage=='canary' else m['cases'][index]['id']
    root=Path(m['output'])/name
    value=s.read(root/'completion.json')
    if value.get('schema')!='sparse-equilibrium.completion.v1' or value.get('status')!='complete' or value.get('task')!=name:
        raise ValueError('Missing semantic completion: '+name)
    run=Path(value['run_dir']); summary=run/'artifacts/summary.json'
    if not run.resolve().is_relative_to(root.resolve()) or s.digest(summary)!=value['summary_sha256']:
        raise ValueError('Completion artifact identity differs')
    result=s.read(summary); records=result['records']
    expected=30 if name=='canary' or name=='mlp' else 12
    # Graph has 18 records, plus 12 sentiment cases for canary/MLP.
    if result.get('status')!='complete' or result.get('task')!=name or len(records)!=expected:
        raise ValueError('Incomplete pilot coverage: '+name)
    native=s.read(run/'result.json')
    if native.get('status')!='complete':raise ValueError('Native run not complete')
    for item in native['artifacts']:
        p=(run/item['path']).resolve()
        if not p.is_relative_to(run.resolve()) or s.digest(p)!=item['sha256']:
            raise ValueError('Native artifact digest differs')
    return value


def verify_gate(path,m,ledger):
    candidate(path,m)
    jobs=ledger.get('jobs',[])
    canaries=[j for j in jobs if j['stage']=='canary']
    if len(canaries)!=1:raise ValueError('Expected exactly one canary parent')
    job=canaries[0];parent=job['parent']
    if s.live_job(parent):raise ValueError('Canary is still pending/running')
    rows=s.scheduler_rows(parent,[0]);s.contract_rows(rows,m)
    if any(r['state']!='COMPLETED' or r['exit_code']!='0:0' for r in rows.values()):
        raise ValueError('Canary scheduler failure')
    artifact=completion(m,'canary')
    logs=[]
    for ext in ('out','err'):
        p=Path(m['output'])/'slurm'/f"{m['job_prefix']}canary-{parent}_0.{ext}"
        text=p.read_text()
        if re.search(r'Traceback|CUDA out of memory|ModuleNotFoundError|FloatingPointError|slurmstepd: error',text,re.I):
            raise ValueError('Unexplained canary log error: '+str(p))
        logs.append({'path':str(p),'sha256':s.digest(p)})
    receipt=Path(m['output'])/'canary-gate.json'
    subprocess.run([sys.executable,str(Path(m['source'])/'launch/verify_canary.py'),
        '--job-id',parent,'--expected-account',m['account'],'--expected-partition',m['partition'],
        '--expected-qos',m['qos'],'--expected-constraint',m['constraint'],'--expected-tasks','1',
        '--require-file',str(Path(m['output'])/'canary/completion.json'),'--receipt',str(receipt)],check=True)
    gate={'gate_passed':True,'manifest_sha256':s.digest(path),'source_commit':m['source_commit'],
          'canary':parent,'artifact':artifact,'logs':logs,'reviewed_at':s.now()}
    s.write(Path(m['output'])/'gate.json',gate)
    return gate


def submit(path,stage):
    m=s.read(path);root=Path(m['output']);(root/'slurm').mkdir(parents=True,exist_ok=True)
    with s.lock(root/'scheduler.lock'):
        ledger=s.read(root/'scheduler.json') if (root/'scheduler.json').exists() else {'jobs':[]}
        if ledger.get('submitting'):raise ValueError('Uncertain previous submission; reconcile, never duplicate')
        matches=[j for j in ledger['jobs'] if j['stage']==stage]
        if matches:
            if len(matches)!=1:raise ValueError('Duplicate parent ledger')
            return matches[0]['parent']
        candidate(path,m)
        if datetime.now().astimezone()>=datetime.fromisoformat(m['deadline']):raise ValueError('Pilot window closed')
        if stage=='production':verify_gate(path,m,ledger)
        elif ledger['jobs']:raise ValueError('Canary must be first')
        argv=command(path,m,stage)
        ledger['submitting']={'command':argv,'stage':stage,'created_at':s.now()}
        s.write(root/'scheduler.json',ledger)
        parent=s.output(argv).strip().split(';')[0]
        if not parent.isdigit():raise ValueError('Uncertain sbatch output')
        job=ledger.pop('submitting');job['parent']=parent;ledger['jobs'].append(job)
        s.write(root/'scheduler.json',ledger)
        # Submission is persisted before querying; a transient query cannot duplicate it.
        control=s.output(['scontrol','show','job','-o',parent])
        for value in ('Account='+m['account'],'QOS='+m['qos'],'Partition='+m['partition'],'Features='+m['constraint']):
            if value not in control:raise ValueError('Submitted resource contract differs: '+value)
        s.write(root/f'{stage}-submission.json',{'parent':parent,'command':argv,'scontrol':control,'observed_at':s.now()})
        return parent


def worker(path,stage,index):
    m=s.read(path);candidate(path,m)
    if (stage=='canary' and index!=0) or index not in (0,1):raise ValueError('Unexpected array index')
    name='canary' if stage=='canary' else m['cases'][index]['id']
    config=Path(m['source'])/'launch'/f'{name}.json'
    destination=Path(m['output'])/name
    with s.lock(destination/'worker.lock',nonblocking=True):
        if (destination/'run').exists():raise ValueError('Refusing to repeat an existing native run')
        s.write(destination/'environment.json',{'stage':stage,'index':index,'job_id':os.environ.get('SLURM_JOB_ID'),
                    'source_commit':m['source_commit'],'python':sys.executable,'started_at':s.now()})
        subprocess.run(['sha256sum','--check','inputs.sha256'],cwd=m['input_root'],check=True)
        env={**os.environ,'SPARSE_RUN_ID':'run'}
        subprocess.run([sys.executable,'-m','ebl','train','--config',str(config),'--output-dir',str(destination)],
                       cwd=m['source'],env=env,check=True)
        completion(m,stage,index)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('commands','canary-submit','release','worker','check'))
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--stage',choices=('canary','production'),default='production')
    a=p.parse_args();m=s.read(a.manifest)
    if a.action=='commands':print(__import__('json').dumps({v:command(a.manifest,m,v) for v in ('canary','production')},indent=2))
    elif a.action=='worker':worker(a.manifest,a.stage,s.task_index(os.environ))
    elif a.action=='check':candidate(a.manifest,m);print('Candidate verified')
    else:print(submit(a.manifest,'canary' if a.action=='canary-submit' else 'production'))


if __name__=='__main__':main()
