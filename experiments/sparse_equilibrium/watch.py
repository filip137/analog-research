"""Read-only Slurm/run-bundle adapter for the existing run-watch service."""
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from ebl.lifecycle import slurm_recovery as s


def remote_observation(path):
    m=s.read(path);root=Path(m['output'])
    ledger=s.read(root/'scheduler.json');out={'jobs':ledger['jobs'],'items':{}}
    for stage,indices in (('canary',[0]),('production',[0,1])):
        jobs=[j for j in ledger['jobs'] if j['stage']==stage]
        if len(jobs)>1:raise ValueError('Duplicate parent jobs')
        if not jobs:continue
        job=jobs[0];parent=job['parent']
        live=s.live_job(parent,'%i|%T|%R|%a|%P|%q|%b')
        allocations={}
        for line in live.splitlines():
            ident,state,reason,account,partition,qos,gres=line.split('|')
            if (account,partition,qos)!=(m['account'],m['partition'],m['qos']) or 'gpu' not in gres:
                raise ValueError('Unexpected live resource contract')
            # A pending aggregate may use 123_[0-1%2]; expand only declared indices.
            if ident.startswith(parent+'_['):
                for i in indices:allocations[i]={'state':state,'reason':reason}
            elif ident.startswith(parent+'_') and ident.split('_')[-1].isdigit():
                allocations[int(ident.split('_')[-1])]={'state':state,'reason':reason}
            elif ident==parent and indices==[0]:allocations[0]={'state':state,'reason':reason}
            else:raise ValueError('Unexpected live array task identity: '+ident)
        if not live:
            allocations=s.scheduler_rows(parent,indices);s.contract_rows(allocations,m)
        elif set(allocations)!=set(indices):
            rows=s.scheduler_rows(parent,indices);s.contract_rows(rows,m)
            for i,row in rows.items():allocations.setdefault(i,row)
        if set(allocations)!=set(indices):raise ValueError('Missing or unexpected array task')
        for i in indices:
            name='canary' if stage=='canary' else m['cases'][i]['id']
            d=root/name;native=d/'run';metric=native/'metrics.jsonl'
            logs=[]
            for ext in ('out','err'):
                p=root/'slurm'/f"{m['job_prefix']}{'canary' if stage=='canary' else 'w1'}-{parent}_{i}.{ext}"
                if p.exists():
                    with p.open('rb') as f:
                        f.seek(max(0,p.stat().st_size-65536));logs.append(f.read().decode(errors='replace'))
            terminal=s.read(d/'completion.json') if (d/'completion.json').exists() else {}
            valid=False
            if terminal.get('status')=='complete':
                summary=native/'artifacts/summary.json'
                valid=summary.is_file() and s.digest(summary)==terminal.get('summary_sha256')
            native_status=s.read(native/'status.json') if (native/'status.json').exists() else {}
            out['items'][name]={'allocation':allocations[i],'parent':parent,'index':i,
                'has_artifacts':(d/'environment.json').exists() or (native/'manifest.json').exists(),
                'metric_bytes':metric.stat().st_size if metric.exists() else 0,
                'heartbeat_at':metric.stat().st_mtime if metric.exists() else None,
                'terminal':terminal,'valid':valid,'native_error':native_status.get('error'),
                'log_error':next((t[-2000:] for t in logs if re.search(r'Traceback|CUDA out of memory|FloatingPointError|slurmstepd: error',t,re.I)),None),
                'log_tail':'\n'.join(logs)[-2000:]}
    return out


def classify(run,data):
    name=run['case'];r={'id':run['id'],'state':'queued','alive':False,'has_artifacts':False,
        'complete_evidence':False,'log_error':None,'log_tail':'Waiting for the canary gate and batch release.',
        'heartbeat_at':None,'progress_token':'prepared'}
    item=data['items'].get(name)
    if item is None:
        if name=='canary' or any(j['stage']=='production' for j in data['jobs']):r['probe_error']='Missing submitted task'
        return r
    a=item['allocation'];state=a['state']
    r.update(process={'job_id':f"{item['parent']}_{item['index']}"},has_artifacts=item['has_artifacts'],
             log_tail=item['log_tail'],log_error=item['log_error'],heartbeat_at=item['heartbeat_at'],
             progress_token=json.dumps([item['parent'],item['metric_bytes'],item['terminal'].get('status')]))
    if state=='PENDING':r.update(state='queued',log_tail='Pending: '+a.get('reason','unknown'))
    elif state in s.ACTIVE:r.update(state='running',alive=True,log_error=r['log_error'] or (str(item['native_error']) if item['native_error'] else None))
    elif state=='COMPLETED' and a.get('exit_code')=='0:0' and item['valid']:
        r.update(state='complete',complete_evidence=True)
    else:r.update(state='failed',log_error=r['log_error'] or json.dumps({'scheduler':a,'terminal':item['terminal']}))
    return r


def main():
    request=json.load(sys.stdin);runs=request['runs'];paths={r['remote_manifest'] for r in runs}
    try:
        if len(paths)!=1:raise ValueError('Expected one immutable launch manifest')
        path=paths.pop();source=str(Path(path).parent.parent)
        interpreters={r['remote_python'] for r in runs}
        if len(interpreters)!=1:raise ValueError('Expected one declared remote Python interpreter')
        interpreter=interpreters.pop()
        if not Path(interpreter).is_absolute():raise ValueError('Remote Python must be an absolute module interpreter path')
        code='import json,sys;sys.path.insert(0,sys.argv[1]);from experiments.sparse_equilibrium.watch import remote_observation;print(json.dumps(remote_observation(sys.argv[2])))'
        cmd=shlex.quote(interpreter)+' -c '+shlex.quote(code)+' '+shlex.quote(source)+' '+shlex.quote(path)
        data=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','jean-zay',cmd],text=True,stderr=subprocess.PIPE,timeout=90))
        result={'gpus':[],'gpu_error':'Capacity is allocated by Slurm.','runs':{r['id']:classify(r,data) for r in runs}}
    except Exception as error:
        result={'gpus':[],'probe_error':str(error),'runs':{r['id']:{'id':r['id'],'probe_error':str(error)} for r in runs}}
    print(json.dumps(result))


if __name__=='__main__':main()
