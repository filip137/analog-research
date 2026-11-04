"""CPU-only completion hooks for this study's existing shared GPU queue.

Scientific configurations and decisions remain in tuning/recovery and numerical
work always enters through ebl. This module only stages, submits, collects and
advances completed native plans. It does not allocate GPUs or poll workers.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from ebl.runtime.store import atomic_write_json, content_hash, sha256_file
from ebl.schema.opt_cmo_drn import parse_config
from ebl.lifecycle.study import inspect_run
import ebl.lifecycle.tuning as tuning


def read(path):
    return json.loads(Path(path).read_text())


def run(argv, *, cwd=None):
    return subprocess.check_output([str(x) for x in argv], cwd=cwd, text=True, stderr=subprocess.PIPE)


def artifact(root, kind):
    matches=[a for a in read(Path(root)/'result.json')['artifacts'] if a['kind']==kind]
    if len(matches)!=1:
        raise ValueError(f'Expected one {kind} in {root}.')
    a=matches[0]
    path=(Path(root)/a['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or sha256_file(path)!=a['sha256']:
        raise ValueError('Artifact path or digest differs.')
    return path,a


def terminal_receipt(status):
    successful=[]
    charge=0.
    for attempt in status['attempts']:
        receipt=attempt.get('receipt') or {}
        if isinstance(receipt,str): receipt=json.loads(receipt)
        receipt=receipt.get('receipt',receipt)
        charge+=receipt.get('elapsed_seconds',0.)
        if attempt['state']=='complete' and receipt.get('run_exit_code')==0 and receipt.get('validation_exit_code')==0:
            successful.append((attempt,receipt))
    if len(successful)!=1:
        raise ValueError('Collection requires exactly one successful validated attempt.')
    return successful[0][0],charge


def terminal_failure(status):
    """Only a stopped, unreserved, terminal worker can release output reserve."""
    if status['job'].get('state') not in ('failed','attention'):return None
    attempts=status.get('attempts',[])
    if not attempts:return None
    attempt=attempts[-1]
    outer=attempt.get('receipt') or {}
    if isinstance(outer,str):outer=json.loads(outer)
    receipt=outer.get('receipt',{})
    if (attempt.get('state')!='failed' or outer.get('alive') is not False
            or outer.get('reservation_held') is not False or receipt.get('finished_at') is None):
        return None
    if not any(type(receipt.get(k)) is int and receipt[k]!=0 for k in ('run_exit_code','validation_exit_code')):
        return None
    return attempt,receipt


def runnable(jobs, completed):
    ready=[name for name,j in jobs.items() if not j.get('submitted') and not j.get('paused') and
           all(ref['parent'] in completed for ref in j['inputs'].values() if 'parent' in ref)]
    return sorted(ready,key=lambda name:-jobs[name].get('priority',0))


def removable(name,jobs,completed):
    return name in completed and all(consumer in completed for consumer,j in jobs.items()
        if any(ref.get('parent')==name for ref in j['inputs'].values()))


def storage_reservation(config, command):
    """Bound native temporary saves plus controller collection overlap."""
    if command=='validate':return 32*1024**2
    if config.stage=='hwa':
        gib=11 if config.training.max_updates==0 else (8 if config.training.max_updates>config.training.checkpoint_every else 6)
        return gib*1024**3
    if config.stage=='deploy':return 3*1024**3
    return (9 if config.recovery.arm=='learning' else 6)*1024**3


def artifact_removable(name,kind,jobs,completed):
    return name in completed and all(consumer in completed or (j.get('paused') and not j.get('submitted'))
        for consumer,j in jobs.items()
        if any(ref.get('parent')==name and ref.get('kind')==kind for ref in j['inputs'].values()))


class Execution:
    def __init__(self, settings):
        self.settings_path=Path(settings).resolve()
        self.s=read(self.settings_path)
        self.root=Path(self.s['output']).resolve()
        self.path=self.root/'execution-state.json'
        self.root.mkdir(parents=True,exist_ok=True)
        self.state=read(self.path) if self.path.exists() else {'jobs':{},'completed':{},'phases':[]}

    def save(self): atomic_write_json(self.path,self.state)

    def queue(self,*args):
        return json.loads(run([sys.executable,self.s['queue_script'],*args]))

    def ssh(self,host,argv):
        return run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','filip@'+host,shlex.join([str(x) for x in argv])])

    def transfer(self,host,local,remote):
        self.ssh(host,['mkdir','-p',str(Path(remote).parent)])
        run(['rsync','-a','--partial','--checksum','-e','ssh -o BatchMode=yes -o ConnectTimeout=8',local,'filip@'+host+':'+remote])

    def register(self,name,config,inputs,hosts,phase,expected,budget,command='train'):
        config_path=self.root/'configs'/f'{name}.json'
        config_path.parent.mkdir(parents=True,exist_ok=True)
        value=asdict(config)
        if config_path.exists() and content_hash(read(config_path))!=content_hash(value): raise ValueError('Existing config differs.')
        atomic_write_json(config_path,value)
        job={'config':str(config_path),'inputs':inputs,'hosts':hosts,'phase':phase,
             'expected_seconds':math.ceil(expected),'budget_seconds':math.ceil(budget),'command':command,
             'storage_reservation_bytes':storage_reservation(config,command)}
        old=self.state['jobs'].get(name)
        if old and any(old[k]!=v for k,v in job.items()): raise ValueError('Existing job differs.')
        if not old:self.state['jobs'][name]=job

    def mapped(self): return {'path':self.s['mapped'],'sha256':self.s['mapped_sha256']}

    def tuning_phase(self,phase,previous=None):
        base=parse_config(read(self.s['base_config']))
        tier=self.s['tier']
        rows=None
        if previous:
            names=[n for n,j in self.state['jobs'].items() if j['phase']==previous]
            if not names or not all(n in self.state['completed'] for n in names):return
            configs={n:parse_config(read(self.state['jobs'][n]['config'])) for n in names}
            rows=tuning.verify_coverage(configs,{n:self.state['completed'][n]['run'] for n in names})
            if any(r['mapped_sha256']!=self.s['mapped_sha256'] for r in rows):raise ValueError('Mapped identity differs.')
        if phase=='screen':configs=tuning.screen_configs(base,tier)
        elif phase=='refine':configs=tuning.refinement_configs(base,tier,rows)
        elif phase=='confirm':configs=tuning.confirmation_configs(base,tier,rows)
        else:
            configs,verdict=tuning.final_configs(base,rows)
            self.state['tuning_verdict']=verdict
        for name,c in configs.items():
            if phase=='full':name='full-'+name
            updates=c.training.max_updates or 30505
            selections=1+math.ceil(updates/512)+(5 if phase=='full' else 0)
            t=self.s['timings']
            expected=updates*t['update_seconds']+selections*t['selection_seconds']+(selections+1)*t['checkpoint_seconds']+20
            # Bounds cover operational variation, never extend tuning's aggregate allowance.
            budget=expected*1.7+60
            self.register(name,c,{'weights':self.mapped()},self.s['hosts'],phase,expected,budget)
        self.state['phases'].append(phase)
        self.save()

    def expand_recovery(self):
        from ebl.lifecycle.recovery import build_plan
        if not all(n in self.state['completed'] for n in ('full-clean','full-std3')):return
        for arm in ('clean','std3'):
            parent='full-'+arm
            c=parse_config(read(self.state['jobs'][parent]['config']))
            source,_=artifact(self.state['completed'][parent]['run'],'best_weights')
            plan_dir=self.root/'recovery-plans'/arm
            plan=build_plan(asdict(c),self.s['local_teacher'],source,plan_dir,sys.executable,self.s['project_dir'])
            # A physical case stays on its deployment host, with exact saved-state dependencies.
            for st in plan['stages']:
                name=arm+'-'+st['id']
                cfg=parse_config(read(plan_dir/st['config']))
                inputs={}
                for role,ref in st['inputs'].items():
                    if role=='teacher_weights':continue
                    if 'stage' in ref:inputs[role]={'parent':arm+'-'+ref['stage'],'kind':ref['artifact_kind']}
                    else:inputs[role]={'parent':parent,'kind':'best_weights'}
                d=self.s['physical_timings']
                if st['command']=='validate':expected=90
                elif cfg.stage=='deploy':expected=d['deploy_seconds']
                elif cfg.recovery.arm in ('calibration','learning_calibration'):
                    expected=200*d['calibration_update_seconds']+20
                elif cfg.recovery.arm=='learning':expected=2048*d['learning_update_seconds']+40
                elif cfg.recovery.arm=='rewrite':expected=2048*d['rewrite_update_seconds']+40
                else:expected=20
                hosts=self.s['hosts']
                self.register(name,cfg,inputs,hosts,'recovery',expected,expected*1.8+120,st['command'])
            for kind in ('best_weights','final_weights'):
                self.register(arm+'-'+kind+'-held-out',c,{'weights':{'parent':parent,'kind':kind}},
                              self.s['hosts'],'held-out',150,600,'validate')
        self.state['phases'].append('recovery')
        self.save()

    def enable_uncapped_recovery(self):
        """User-authorized revision; retain started capped arms as references."""
        if 'recovery_uncapped' in self.state['phases']:return
        if 'recovery' not in self.state['phases']:
            raise ValueError('Uncapped recovery requires the declared recovery plan.')
        replacements=[]
        for arm in ('clean','std3'):
            for kind in ('none','random'):
                prefix=arm+'-'+kind
                old_names=[prefix+'-'+suffix for suffix in
                           ('learning','learning-held-out','learning_calibration','learning_calibration-held-out')]
                old={name:self.state['jobs'][name] for name in old_names}
                learning=old[prefix+'-learning']
                config=parse_config(read(learning['config']))
                if config.recovery.max_write_attempts!=4:
                    raise ValueError('Expected the original capped learning comparison.')
                # Never mutate a submitted job, its config, or its input history.
                if all(not j.get('submitted') and n not in self.state['completed'] for n,j in old.items()):
                    for name,j in old.items():
                        self.state.setdefault('superseded_jobs',{})[name]={
                            'job':j,'reason':'user removed per-device four-write cap before submission'}
                        del self.state['jobs'][name]
                    replacements.extend(old_names)
                new_learning=prefix+'-uncapped-learning'
                for control in ('learning','learning_calibration'):
                    name=prefix+'-uncapped-'+control
                    cfg=replace(config,recovery=replace(config.recovery,arm=control,max_write_attempts=None))
                    inputs=json.loads(json.dumps(learning['inputs']))
                    if control=='learning_calibration':
                        inputs['device_state']={'parent':new_learning,'kind':'recovery_state'}
                    template=old[prefix+'-'+control]
                    self.register(name,cfg,inputs,template['hosts'],'recovery_uncapped',
                                  template['expected_seconds'],template['budget_seconds'])
                    validation=old[prefix+'-'+control+'-held-out']
                    self.register(name+'-held-out',cfg,{'weights':{'parent':name,'kind':'recovery_state'}},
                                  validation['hosts'],'recovery_uncapped',validation['expected_seconds'],
                                  validation['budget_seconds'],'validate')
        self.state['phases'].append('recovery_uncapped')
        self.state['uncapped_revision']={'replaced_unsubmitted_jobs':replacements,
            'preserved_started_arms':True,'updates':2048,'max_write_attempts':None,
            'intervention':'remove per-device write cap; retain LR, threshold, write noise and initial deployment'}
        self.state.pop('execution_complete',None)
        self.save()

    def resolve_input(self,ref,host):
        if 'parent' in ref:
            done=self.state['completed'][ref['parent']]
            local,a=artifact(done['run'],ref['kind'])
            if host==done['host'] and a['path'] not in done.get('retired_artifacts',[]) and not done.get('retired'):
                return str(Path(done['remote_run'])/a['path']),a['sha256']
            digest=a['sha256']
        else:
            local=Path(ref['path']);digest=ref['sha256']
            if sha256_file(local)!=digest:raise ValueError('Explicit input digest differs.')
        remote=self.s['remote_root']+'/inputs/'+digest+'.pt'
        key=host+':'+digest
        if key not in self.state.setdefault('input_copies',{}) or self.state['input_copies'][key].get('retired'):
            self.transfer(host,local,remote)
            self.state['input_copies'][key]={'host':host,'path':remote,'sha256':digest}
            self.save()
        return remote,digest

    def enable_noiseless_lr(self):
        """Matched user-assigned LR check; never retune an existing attempt."""
        phase='recovery_noiseless_lr'
        if phase in self.state['phases']:return
        template=self.state['jobs']['clean-none-uncapped-learning']
        base=parse_config(read(template['config']))
        if base.recovery.max_write_attempts is not None or base.deployment.fault_kind!='none':
            raise ValueError('Expected the canonical uncapped clean/nominal recovery control.')
        paused=[]
        for name,j in self.state['jobs'].items():
            if j['phase']=='recovery_uncapped' and not j.get('submitted'):
                if (self.root/'queue-jobs'/f'{name}.json').exists():
                    raise ValueError('Reconcile an already prepared queue job before pausing its inputs.')
                j['paused']='Await noiseless/LR review before further old-LR recovery.'
                paused.append(name)
        cases=[]
        for index,(label,lr) in enumerate((('3e-3',.003),('1e-3',.001),('3e-4',.0003),('1e-4',.0001))):
            name='clean-none-noiseless-lr'+label
            # Full development scoring adds 17 evaluations plus the first write.
            expected=template['expected_seconds']+18*self.s['physical_timings']['calibration_update_seconds']
            cfg=replace(base,training=replace(base.training,max_wall_seconds=math.ceil(expected*1.8)),
                recovery=replace(base.recovery,learning_rate=lr,write_noise_factor=0.,
                    development_every=128,retain_final_optimizer=False))
            self.register(name,cfg,json.loads(json.dumps(template['inputs'])),template['hosts'],phase,
                          expected,expected*1.8+120)
            self.state['jobs'][name]['priority']=100-index
            cases.append(name)
        self.state['phases'].append(phase)
        self.state['noiseless_lr_revision']={'cases':cases,'paused_unsubmitted':paused,
            'updates':base.recovery.updates,'development_every':128,'write_noise_factor':0.,
            'selection':'fixed-final full-development KL; no-write ties are not recovery',
            'intervention':'recovery write noise only; preserve deployment, threshold, batches and unlimited attempts',
            'retention':'full partial continuation; final weights/physical bank and controller diagnostics, omit final optimizer tensors'}
        self.state.pop('execution_complete',None)
        self.save()

    def placement_hosts(self,j):
        hosts=j['hosts']
        # Device-state children stay with the exact physical parent; held-out
        # recovery reads the same saved bank on that host.
        physical=[r['parent'] for r in j['inputs'].values() if r.get('kind') in ('device_state','recovery_state')]
        if physical:hosts=[self.state['completed'][physical[0]]['host']]
        return hosts

    def input_copy_bytes(self,j):
        total=0
        for ref in j['inputs'].values():
            done=None
            if 'parent' in ref:
                done=self.state['completed'][ref['parent']]
                path,a=artifact(done['run'],ref['kind']);digest=a['sha256']
            else:path=Path(ref['path']);digest=ref['sha256']
            for host in self.placement_hosts(j):
                if done and host==done['host'] and a['path'] not in done.get('retired_artifacts',[]) and not done.get('retired'):continue
                copy=self.state.get('input_copies',{}).get(host+':'+digest)
                if copy and not copy.get('retired'):continue
                total+=path.stat().st_size
        return total

    def submit(self,name):
        j=self.state['jobs'][name]
        hosts=self.placement_hosts(j)
        spec_path=self.root/'queue-jobs'/f'{name}.json'
        if spec_path.exists():spec=read(spec_path)
        else:
            template=read(self.s['job_template'])
            spec={k:v for k,v in template.items() if k!='placements'}
            spec.update(id=self.s['job_prefix']+name,expected_seconds=j['expected_seconds'],budget_seconds=j['budget_seconds'],
                        stale_seconds=2700,startup_grace_seconds=300,recovery=True,retry_safe=False,
                        priority=j.get('priority',0))
            completion = (['-m', self.s['completion_module']] if self.s.get('completion_module')
                          else ['-m', 'ebl', 'lifecycle', 'queue'])
            spec['completion']={'argv':[sys.executable,*completion,
                                       'complete','--settings',str(self.settings_path)],
                                'workdir':self.s['controller_dir'],'timeout_seconds':1800,'retry_safe':True}
            spec['placements']={}
            for host in hosts:
                p=json.loads(json.dumps(next(iter(template['placements'].values()))))
                output=self.s['remote_root']+'/results/'+name
                config=self.s['remote_root']+'/plans/execution/'+name+'.json'
                self.transfer(host,j['config'],config)
                p.update(run_dir=output)
                p['argv']=[self.s['remote_python'],'-m','ebl',j['command'],'--config',config,
                           '--teacher-weights',self.s['remote_teacher'],'--output-dir',output]
                p['prepared_files']=[p['prepared_files'][0],{'path':config,'sha256':sha256_file(Path(j['config']))},
                                      {'path':self.s['remote_teacher'],'sha256':self.s['teacher_sha256']}]
                for role,ref in j['inputs'].items():
                    path,digest=self.resolve_input(ref,host)
                    p['argv']+=['--'+role.replace('_','-'),path]
                    p['prepared_files'].append({'path':path,'sha256':digest})
                p['validate_argv']=[self.s['remote_python'],'-m','ebl','runs','inspect',output+'/run','--verify-artifacts','--require-complete']
                spec['placements'][host]=p
            atomic_write_json(spec_path,spec)
        self.queue('submit',spec_path)
        j['submitted']=True
        self.save()

    def collect(self,job_id):
        prefix=self.s['job_prefix']
        if not job_id.startswith(prefix):raise ValueError('Completion belongs to a different study.')
        name=job_id[len(prefix):]
        if name not in self.state['jobs']:raise ValueError('Unknown job.')
        if name in self.state['completed']:return
        status=self.queue('status',job_id)
        attempt,charged=terminal_receipt(status)
        host=attempt['target']
        spec=status['job']['spec']
        if isinstance(spec,str):spec=json.loads(spec)
        remote=spec['placements'][host]['run_dir']+'/run'
        local=self.root/'runs'/name/'run'
        local.mkdir(parents=True,exist_ok=True)
        run(['rsync','-a','--partial','-e','ssh -o BatchMode=yes -o ConnectTimeout=8','filip@'+host+':'+remote+'/',str(local)+'/'])
        checked=inspect_run(local,verify_artifacts=True)
        if not checked['valid'] or not checked['process_complete']:raise ValueError('Collected bundle is not complete and valid.')
        # Use the public inspector as the transport validation surface too.
        run([sys.executable,'-m','ebl','runs','inspect',local,'--verify-artifacts','--require-complete'],cwd=self.s['project_dir'])
        self.state['completed'][name]={'run':str(local),'remote_run':remote,'host':host,'attempt':attempt['token'],
                                       'charged_seconds':charged,'retired':False}
        self.save()

    def reconcile_failures(self):
        """Preserve terminal failures without satisfying scientific dependencies."""
        for name,j in self.state['jobs'].items():
            if not j.get('submitted') or name in self.state['completed']:continue
            status=self.queue('status',self.s['job_prefix']+name)
            failed=terminal_failure(status)
            recorded=self.state.get('failed',{}).get(name)
            if failed is None:
                # An explicitly rearmed attempt must reserve storage again.
                if recorded:
                    self.state.setdefault('failure_history',[]).append({'job':name,**recorded})
                    del self.state['failed'][name]
                    self.save()
                continue
            attempt,receipt=failed
            if recorded and recorded['attempt']==attempt['token']:continue
            spec=status['job']['spec']
            if isinstance(spec,str):spec=json.loads(spec)
            host=attempt['target']
            remote=spec['placements'][host]['run_dir']+'/run'
            local=self.root/'failed-runs'/name/attempt['token']/'run'
            local.mkdir(parents=True,exist_ok=True)
            run(['rsync','-a','--partial','-e','ssh -o BatchMode=yes -o ConnectTimeout=8',
                 'filip@'+host+':'+remote+'/',str(local)+'/'])
            checked=inspect_run(local,verify_artifacts=True)
            if checked['valid']:
                run([sys.executable,'-m','ebl','runs','inspect',local,'--verify-artifacts'],cwd=self.s['project_dir'])
            atomic_write_json(local.parent/'queue-failure.json',{'status':status,'inspection':checked})
            self.state.setdefault('failed',{})[name]={
                'run':str(local),'remote_run':remote,'host':host,'attempt':attempt['token'],
                'charged_seconds':receipt.get('elapsed_seconds'),
                'bundle_valid':checked['valid'],'process_complete':False,
                'reason':status['job'].get('reason'),'retry_authorized':False}
            self.save()

    def retire_replicas(self):
        for name,done in self.state['completed'].items():
            if done.get('retired'):continue
            phase=self.state['jobs'][name]['phase']
            if phase=='full' and 'recovery' not in self.state['phases']:continue
            root=Path(done['run'])
            checkpoints=[a for a in read(root/'result.json')['artifacts'] if a['path'].startswith('checkpoints/')]
            artifacts=[a for a in checkpoints if a['path'] not in done.get('retired_artifacts',[]) and
                       artifact_removable(name,a['kind'],self.state['jobs'],self.state['completed'])]
            if not artifacts:continue
            for a in artifacts:
                if sha256_file(root/a['path'])!=a['sha256']:raise ValueError('Canonical artifact lost before replica retirement.')
            # Only exact manifest-listed checkpoint replicas, after all declared
            # consumers finish. Original metadata and canonical bundles are retained.
            code="""import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); items=json.loads(sys.argv[2])
for a in items:
 p=(root/a['path']).resolve()
 if not p.is_relative_to(root.resolve()):raise ValueError('Unsafe artifact path')
 if p.exists():
  h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
  if h!=a['sha256']:raise ValueError('Replica digest differs')
  p.unlink()
path=root/'replica-migration.json'
receipt=json.loads(path.read_text()) if path.exists() else {'policy':'verified_canonical_migration','canonical_run':sys.argv[3],'removed':[]}
known={a['path'] for a in receipt['removed']}
receipt['removed'] += [a for a in items if a['path'] not in known]
path.write_text(json.dumps(receipt,indent=2)+'\\n')
"""
            self.ssh(done['host'],[self.s['remote_python'],'-c',code,done['remote_run'],json.dumps(artifacts),str(root)])
            done.setdefault('retired_artifacts',[]).extend(a['path'] for a in artifacts)
            done['retired']=all(a['path'] in done['retired_artifacts'] for a in checkpoints)
            self.save()

    def retire_inputs(self):
        needed=set()
        canonical={self.s['mapped_sha256']:Path(self.s['mapped'])}
        for done in self.state['completed'].values():
            root=Path(done['run'])
            for a in read(root/'result.json')['artifacts']:
                candidate=(root/a['path']).resolve()
                if not candidate.is_relative_to(root.resolve()):raise ValueError('Unsafe canonical input path.')
                canonical[a['sha256']]=candidate
        verified=set()
        def planned_hosts(name):
            if name in self.state['completed']:return [self.state['completed'][name]['host']]
            job=self.state['jobs'][name]
            physical=[r['parent'] for r in job['inputs'].values()
                      if r.get('kind') in ('device_state','recovery_state')]
            return planned_hosts(physical[0]) if physical else job['hosts']
        if 'full' not in self.state['phases']:
            needed.update((host,self.s['mapped_sha256']) for host in self.s['hosts'])
        for name,j in self.state['jobs'].items():
            if name in self.state['completed'] or (j.get('paused') and not j.get('submitted')):continue
            for ref in j['inputs'].values():
                digests=[]
                if 'parent' not in ref:digests=[ref['sha256']]
                elif ref['parent'] in self.state['completed']:
                    root=self.state['completed'][ref['parent']]['run']
                    digests=[a['sha256'] for a in read(Path(root)/'result.json')['artifacts'] if a['kind']==ref['kind']]
                needed.update((host,digest) for host in planned_hosts(name) for digest in digests)
        for key,copy in self.state.get('input_copies',{}).items():
            if copy.get('retired') or (copy['host'],copy['sha256']) in needed:continue
            digest=copy['sha256']
            if digest not in verified:
                if digest not in canonical or sha256_file(canonical[digest])!=digest:
                    raise ValueError('Canonical input missing or changed before cache retirement.')
                verified.add(digest)
            code="""import hashlib,pathlib,sys,json
p=pathlib.Path(sys.argv[1])
if p.exists():
 if hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=sys.argv[2]:raise ValueError('Cache digest differs')
 p.unlink()
p.with_suffix('.retired.json').write_text(json.dumps({'sha256':sys.argv[2],'policy':'no_remaining_consumers_on_host'})+'\\n')
"""
            self.ssh(copy['host'],[self.s['remote_python'],'-c',code,copy['path'],copy['sha256']])
            copy['retired']=True
            self.save()

    def retire_failed_replicas(self):
        """A stopped failure keeps its full checkpoint in the canonical bundle."""
        for name,failed in self.state.get('failed',{}).items():
            if not failed.get('bundle_valid') or failed.get('retired_checkpoints'):continue
            root=Path(failed['run'])
            items=[]
            for label in ('latest.pt','final.pt'):
                path=root/'checkpoints'/label
                if path.is_file():items.append({'path':'checkpoints/'+label,'sha256':sha256_file(path),
                                               'size_bytes':path.stat().st_size})
            if not items:continue
            code="""import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);items=json.loads(sys.argv[2])
for a in items:
 p=root/a['path']
 if p.exists():
  if hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=a['sha256']:raise ValueError('Failed replica digest differs')
  p.unlink()
(root/'replica-migration.json').write_text(json.dumps({'policy':'verified_failed_canonical_migration',
 'canonical_run':sys.argv[3],'removed':items},indent=2)+'\\n')
"""
            self.ssh(failed['host'],[self.s['remote_python'],'-c',code,failed['remote_run'],json.dumps(items),str(root)])
            failed['retired_checkpoints']=items
            self.save()

    def storage_usage(self):
        used=int(run(['du','-sb',self.s['study_root']]).split()[0])
        for host in self.s['hosts']:
            used+=int(self.ssh(host,['du','-sb',self.s['remote_root']]).split()[0])
        return used

    def free_storage(self):
        return shutil.disk_usage(self.root).free

    def advance(self):
        for phase,previous in [('screen',None),('refine','screen'),('confirm','refine'),('full','confirm')]:
            if phase not in self.state['phases']:
                if previous and previous not in self.state['phases']:break
                self.tuning_phase(phase,previous)
                if phase not in self.state['phases']:break
        if 'full' in self.state['phases'] and 'recovery' not in self.state['phases'] and self.state.get('tuning_verdict',{}).get('verdict')=='confirmed':
            self.expand_recovery()
        self.reconcile_failures()
        self.retire_failed_replicas()
        self.retire_replicas()
        self.retire_inputs()
        # Worst-case queue allowances, including all attempts/readiness, cannot
        # exceed the existing aggregate tuning cap. Successful jobs release slack.
        charge=self.s['timings']['readiness_seconds']
        for name,j in self.state['jobs'].items():
            if j['phase'] not in ('screen','refine','confirm'):continue
            charge+=self.state['completed'][name]['charged_seconds'] if name in self.state['completed'] else j['budget_seconds']
        if charge>24*3600:raise ValueError('Tuning allowances would exceed the original24GPUh cap.')
        used=self.storage_usage()
        free=self.free_storage()
        reserved=sum(j['storage_reservation_bytes'] for n,j in self.state['jobs'].items()
                     if j.get('submitted') and n not in self.state['completed'] and n not in self.state.get('failed',{}))
        deferred=[]
        for name in runnable(self.state['jobs'],self.state['completed']):
            j=self.state['jobs'][name]
            # Include worst-case explicit input replication to all eligible hosts.
            needed=j['storage_reservation_bytes']+self.input_copy_bytes(j)
            if used+reserved+needed>100*1024**3 or reserved+needed+2*1024**3>free:
                deferred.append(name)
                continue
            self.submit(name)
            reserved+=needed
        self.state['storage']={'observed_bytes':used,'reserved_bytes':reserved,'deferred':deferred,
                               'limit_bytes':100*1024**3,'local_free_bytes':free,'local_headroom_bytes':2*1024**3}
        self.save()
        if self.state['jobs'] and len(self.state['completed'])==len(self.state['jobs']):
            self.state['execution_complete']=True
            self.state['scientific_review']='pending'
            self.save()


@contextmanager
def locked(settings):
    s=read(settings);root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield Execution(settings)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['advance','complete','uncap','noiseless-lr'])
    p.add_argument('--settings',required=True,type=Path)
    args=p.parse_args(argv)
    with locked(args.settings) as execution:
        if args.action=='complete':execution.collect(os.environ['RUN_WATCH_JOB_ID'])
        if args.action=='uncap':execution.enable_uncapped_recovery()
        if args.action=='noiseless-lr':execution.enable_noiseless_lr()
        execution.advance()
    return 0


if __name__=='__main__':raise SystemExit(main())
