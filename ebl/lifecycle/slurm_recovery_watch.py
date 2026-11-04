"""Read-only run-watch adapter for the declared Jean Zay DRN recovery cases."""

from datetime import datetime, timezone
import json
import shlex
import subprocess
import sys


REMOTE = r'''
import json,re,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1]);sys.path.insert(0,str(p.parent.parent))
from ebl.lifecycle.slurm_recovery import read,probe,digest
m=read(p);d=probe(p);allocations={}
canary=Path(m['output'])/'canary/complete.json'
if canary.exists():
    receipt=read(canary)
    d['canary_complete']=receipt.get('state')=='complete' and receipt.get('manifest_sha256')==digest(p) and len(receipt.get('cases',[]))==2 and all(c.get('state')=='complete' and len(c.get('attempts',[]))>1 for c in receipt['cases'])
for job in d['scheduler'].get('jobs',[]):
    parent=job['parent']
    if 'accounting' in job:
        for row in job['accounting'].values():
            if (row['account'],row['partition'],row['qos'],row['constraint'])!=(m['account'],m['partition'],m['qos'],m['constraint']):
                raise ValueError('Recorded terminal accounting contract differs')
            allocations[row['job_id']]=row
        continue
    queue=subprocess.check_output(['squeue','-h','-r','-j',parent,'-o','%i|%T|%R|%a|%P|%q|%b'],text=True)
    for line in queue.splitlines():
        ident,state,reason,account,partition,qos,gres=line.split('|')
        if account!=m['account'] or partition!=m['partition'] or qos!=m['qos'] or 'gpu' not in gres:
            raise ValueError('Live Slurm resources differ from declared GPU allocation')
        allocations[ident]={'state':state,'reason':reason,'gres':gres}
    fields='JobIDRaw,JobID%40,State,ExitCode,Account,Partition,QOS,Constraints,ElapsedRaw'
    records=subprocess.check_output(['sacct','-n','-P','-j',parent,'--format='+fields],text=True)
    for line in records.splitlines():
        values=[v.strip() for v in line.split('|')]
        if len(values)!=9:raise ValueError('Invalid accounting response')
        raw,ident,state,code,account,partition,qos,constraint,elapsed=values
        if not re.fullmatch(parent+r'_\d+',ident):continue
        if (account,partition,qos,constraint)!=(m['account'],m['partition'],m['qos'],m['constraint']):
            raise ValueError('Accounting contract differs')
        allocations.setdefault(ident,{'state':state.split()[0].rstrip('+'),'exit_code':code})
for key,item in d['cases'].items():
    path=Path(item['native_run']);metric=path/'metrics.jsonl'
    item['has_artifacts']=(path/'manifest.json').is_file()
    item['metric_bytes']=metric.stat().st_size if metric.exists() else 0
    item['heartbeat_at']=metric.stat().st_mtime if metric.exists() else None
    status=path/'status.json'
    if status.exists():
        st=read(status);item['native_error']=st.get('error')
    log=path.parent/'launcher.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-2000));item['log_tail']=f.read().decode(errors='replace')
d['allocations']=allocations;d['observed_at']=time.time()
print(json.dumps(d))
'''


def classify(run, data):
    ident = run['id']
    controller = run.get('controller', False)
    scheduler = data.get('scheduler', {})
    owner = data.get('supervisor', {})
    result = {'id': ident, 'alive': False, 'state': 'missing', 'has_artifacts': False,
              'complete_evidence': False, 'log_error': None, 'log_tail': '', 'heartbeat_at': None}
    if controller:
        state = owner.get('state', 'missing')
        stamp = owner.get('updated_at')
        fresh = bool(stamp and (data['observed_at'] - datetime.fromisoformat(stamp).timestamp()) < 180)
        result.update(state='complete' if state == 'complete' else 'failed' if state == 'failed' else
                      'queued' if fresh and state in ('queued', 'awaiting_gate', 'awaiting_release') else 'running',
                      alive=fresh and state not in ('complete', 'failed'), has_artifacts=bool(owner),
                      heartbeat_at=stamp, process={'pid': owner.get('pid')},
                      complete_evidence=state == 'complete' and data.get('complete_evidence', False),
                      log_error=owner.get('error'), progress_token=json.dumps([scheduler.get('state'), data.get('cases', {})], sort_keys=True))
        if fresh and state in ('awaiting_gate', 'awaiting_release') and data.get('canary_complete') and not scheduler.get('released'):
            result.update(state='running', log_error='Completed canary requires the authorized controller gate review and production release.')
        return result
    canary = run.get('canary', False)
    index = 0 if canary else run['case_index']
    parents = [j for j in scheduler.get('jobs', []) if j['canary'] == canary and index in j['indices']]
    if not parents:
        if not canary and not scheduler.get('released') and any(j['canary'] for j in scheduler.get('jobs', [])):
            result.update(state='queued', log_tail='Prepared; awaiting canary review and production release.')
            return result
        result['probe_error'] = 'Production case has no submitted allocation.'
        return result
    parent = parents[-1]['parent']
    job = f'{parent}_{index}'
    allocation = data.get('allocations', {}).get(job)
    if allocation is None:
        result['probe_error'] = 'Missing live and accounting evidence for ' + job
        return result
    if canary:
        states = [v for k, v in data.get('cases', {}).items() if k.startswith('canary/')]
        case = states[-1] if states else {}
        case = {**case, 'state': 'complete' if data.get('canary_complete') else 'running'}
    else:
        case = data.get('cases', {}).get('cases/' + run['case_id'], {})
    state = allocation['state']
    result.update(process={'job_id': job}, has_artifacts=case.get('has_artifacts', False),
                  heartbeat_at=case.get('heartbeat_at'), log_tail=case.get('log_tail', ''),
                  progress_token=json.dumps([job, case.get('native_run'), case.get('metric_bytes'), case.get('progress')]))
    if state == 'PENDING':
        result.update(state='queued', log_tail='Pending: ' + allocation.get('reason', 'unknown'))
    elif state in ('RUNNING', 'CONFIGURING', 'COMPLETING'):
        result.update(state='running', alive=True)
        error = case.get('native_error')
        if error and error.get('type') != 'TimeoutError':
            result['log_error'] = json.dumps(error)
    elif state == 'COMPLETED' and allocation.get('exit_code') == '0:0':
        if case.get('state') == 'complete':
            result.update(state='complete', complete_evidence=True)
        elif case.get('state') == 'checkpointed':
            result.update(state='queued', log_tail='Saved full checkpoint; waiting for the reviewed continuation wave.')
        else:
            result.update(state='failed', log_error='Scheduler completed without terminal semantic case evidence.')
    else:
        result.update(state='failed', log_error=json.dumps({'allocation': allocation, 'case_error': case.get('error')}))
    return result


def main():
    request = json.load(sys.stdin)
    runs = request['runs']
    manifests = {r['remote_manifest'] for r in runs}
    if len(manifests) != 1:
        raise ValueError('Expected one declared remote manifest.')
    remote = 'python3 -c ' + shlex.quote(REMOTE) + ' ' + shlex.quote(manifests.pop())
    try:
        data = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                                  'jean-zay', remote], text=True, stderr=subprocess.PIPE, timeout=90))
        result = {'gpus': [], 'gpu_error': 'GPU allocation is checked through Slurm.',
                  'runs': {r['id']: classify(r, data) for r in runs}}
    except Exception as error:
        result = {'gpus': [], 'probe_error': str(error), 'runs': {
            r['id']: {'id': r['id'], 'probe_error': str(error)} for r in runs}}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
