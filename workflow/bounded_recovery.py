"""Bounded recovery pilot using the existing ebl and shared-queue transport."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path

from experiments.artifacts import atomic_write_json, sha256_file
from experiments.opt_cmo_drn.config import parse_config
from workflow.queue_execution import Execution, artifact, read, runnable


RATES=(('1e-5',1e-5),('3e-5',3e-5),('1e-4',1e-4),('3e-4',3e-4),('1e-3',1e-3),('3e-3',3e-3))
NOISE_ARMS=('noiseless','noisy')
CASES=tuple(f'{arm}-lr{label}' for arm in NOISE_ARMS for label,_ in RATES)
MARKER='<!-- generated-evidence -->'


def pilot_config(base,arm,lr):
    return replace(base,stage='recover',training=replace(base.training,max_wall_seconds=6600),
        recovery=replace(base.recovery,arm='learning',updates=2048,learning_rate=lr,
            write_noise_factor=0. if arm=='noiseless' else 1.,
            update_rule='direct_adam' if arm=='noiseless' else 'bounded_thresholded_adam',
            target_bounds='gmin_gmax',conductance_floor='none',max_write_attempts=None,
            checkpoint_every=128,development_every=128,retain_final_optimizer=False))


def winners(rows):
    if set(rows)!=set(CASES):raise ValueError('Selection requires all twelve completed LR candidates.')
    result={}
    for arm in NOISE_ARMS:
        names=[f'{arm}-lr{label}' for label,_ in RATES]
        for name in names:
            row=rows[name]
            if row['updates']!=2048 or not math.isfinite(row['teacher_kl']):
                raise ValueError('Selection requires finite fixed-final 2048-update metrics.')
        result[arm]=min(names,key=lambda name:(rows[name]['teacher_kl'],rows[name]['learning_rate']))
    return result


def readiness_projection(metrics):
    steps=[r['training_step_seconds'] for r in metrics if r['event']=='recovery_update']
    evaluations=[r['seconds'] for r in metrics if r['event']=='recovery_evaluation_timing']
    saves=[r['checkpoint_seconds'] for r in metrics if r['event']=='recovery_checkpoint']
    if len(steps)!=2 or not evaluations or len(saves)!=2:
        raise ValueError('Readiness requires two real updates, development evaluation and resumable/final saves.')
    return {'training_step_seconds':max(steps),'evaluation_seconds':max(evaluations),
        'checkpoint_seconds':max(saves),
        'expected_seconds':max(steps)*2048+max(evaluations)*20+max(saves)*17+60,
        'peak_memory_bytes':max(r['peak_memory_bytes'] for r in metrics if r['event']=='recovery_update')}


class BoundedRecoveryExecution(Execution):
    def collect(self,job_id):
        name=job_id.removeprefix(self.s['job_prefix'])
        original=self.state.get('resume_replacements',{}).get(name)
        if original and original in self.state['completed']:
            return  # A repeated collection hook must not double-charge the attempt.
        super().collect(job_id)
        if original:
            previous=self.state['failed'].pop(original)
            resumed=self.state['completed'].pop(name)
            self.state.setdefault('interrupted_attempts',[]).append({'case':original,**previous})
            resumed['charged_seconds']+=previous['charged_seconds']
            resumed['resumed_from_attempt']=previous['attempt']
            resumed['queue_case']=name
            self.state['completed'][original]=resumed
            self.state['jobs'][original]=self.state['jobs'].pop(name)
            self.save()

    def initialize(self):
        if 'prepared' in self.state['phases']:return
        base=parse_config(read(self.s['base_config']))
        inputs={role:{'path':self.s[role],'sha256':self.s[role+'_sha256']}
                for role in ('weights','device_state')}
        for role,ref in inputs.items():
            if sha256_file(Path(ref['path']))!=ref['sha256']:raise ValueError(f'Pilot {role} digest differs.')
        for arm in NOISE_ARMS:
            cfg=pilot_config(base,arm,.003)
            smoke=replace(cfg,training=replace(cfg.training,max_wall_seconds=480),
                          recovery=replace(cfg.recovery,updates=2,checkpoint_every=1))
            self.register('readiness-'+arm,smoke,inputs,self.s['hosts'],'readiness',120,600)
            for label,lr in RATES:
                cfg=pilot_config(base,arm,lr)
                self.register(f'{arm}-lr{label}',cfg,inputs,self.s['hosts'],'sweep',4800,6750)
        hold=replace(pilot_config(base,'noiseless',.0003),
            training=replace(base.training,max_wall_seconds=240),
            recovery=replace(pilot_config(base,'noiseless',.0003).recovery,arm='hold'))
        self.register('hold',hold,inputs,self.s['hosts'],'baseline',60,300)
        self.state['phases'].append('prepared')
        self.state['readiness_admitted']=False
        self.save()

    def check_readiness(self):
        names=['readiness-'+arm for arm in NOISE_ARMS]
        if self.state.get('readiness_admitted') or not all(n in self.state['completed'] for n in names):return
        rows={}
        for name in names:
            run=Path(self.state['completed'][name]['run'])
            metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            rows[name]=readiness_projection(metrics)
        estimated=sum(rows['readiness-'+arm]['expected_seconds']*6 for arm in NOISE_ARMS)+2100
        # Reserve 20% runtime headroom within each already declared allowance.
        accepted=all(r['expected_seconds']*1.2<=6750 and r['peak_memory_bytes']<=self.s['peak_memory_mib']*1024**2
                     for r in rows.values()) and estimated*1.2<=self.s['gpu_budget_seconds']
        self.state['readiness']={'arms':rows,'projected_gpu_seconds':estimated,'accepted':accepted}
        self.state['readiness_admitted']=accepted
        if not accepted:self.state['blocked']='Measured dense-write costs exceed the declared resource envelope; review required.'
        self.save()

    def candidate_rows(self):
        rows={}
        for name in CASES:
            if name not in self.state['completed']:continue
            run=Path(self.state['completed'][name]['run'])
            result=read(run/'result.json')['metrics']
            config=parse_config(read(self.state['jobs'][name]['config']))
            if result['initial_device_state_sha256']!=self.s['device_state_sha256']:
                raise ValueError('A candidate used a different physical start.')
            rows[name]={'learning_rate':config.recovery.learning_rate,
                'teacher_kl':result['development']['teacher_kl'],'updates':result['updates'],
                'costs':result['costs'],'run':str(run)}
        return rows

    def select(self):
        if 'selected' in self.state['phases'] or not all(n in self.state['completed'] for n in CASES):return
        selected=winners(self.candidate_rows())
        for arm,parent in {**selected,'hold':'hold'}.items():
            cfg=parse_config(read(self.state['jobs'][parent]['config']))
            cfg=replace(cfg,training=replace(cfg.training,max_wall_seconds=480))
            self.register(arm+'-held-out',cfg,{'weights':{'parent':parent,'kind':'recovery_state'}},
                          self.s['hosts'],'held_out',90,600,command='validate')
        self.state['selected']=selected
        self.state['phases'].append('selected')
        self.save()

    def report(self):
        rows=self.candidate_rows()
        baseline=None
        if 'hold' in self.state['completed']:
            baseline=read(Path(self.state['completed']['hold']['run'])/'result.json')['metrics']['development']['teacher_kl']
        summary={'coverage':{'completed_candidates':len(rows),'expected_candidates':12,
                            'failed':self.state.get('failed',{})},
                 'baseline_development_kl':baseline,'candidates':rows,'selected':self.state.get('selected'),
                 'readiness':self.state.get('readiness'),'resources':self.state.get('budget'),
                 'held_out':{},'interpretation':'Pending complete candidate and held-out coverage.'}
        for name in ('hold-held-out','noiseless-held-out','noisy-held-out'):
            if name in self.state['completed']:
                summary['held_out'][name]=read(Path(self.state['completed'][name]['run'])/'result.json')['metrics']
        if len(rows)==12 and len(summary['held_out'])==3:
            baseline_held=summary['held_out']['hold-held-out']
            verdict={}
            for arm in NOISE_ARMS:
                row=rows[self.state['selected'][arm]]
                held=summary['held_out'][arm+'-held-out']
                improved={cohort:held[cohort]['teacher_kl']<baseline_held[cohort]['teacher_kl']
                          for cohort in ('generated','wikitext')}
                verdict[arm]={'development_improved':row['teacher_kl']<baseline,
                    'held_out_improved':improved,'boundary_lr':row['learning_rate'] in (RATES[0][1],RATES[-1][1]),
                    'beneficial_recovery':row['teacher_kl']<baseline and all(improved.values())
                                           and row['costs']['state_changing_writes']>0}
            summary['interpretation']={'predeclared_rule':verdict,
                'scope':'One fixed 3% deployment; CMO fitted-model evidence. Noise and write-gating differ between arms.',
                'next_decision':'Review before replication or the 1–6% corruption series; no automatic expansion.'}
            self.state['execution_complete']=True
            self.state['scientific_review']='predeclared comparison recorded; user review before expansion'
        atomic_write_json(self.root/'summary.json',summary)
        note=Path(self.s['project_dir'])/self.s['handoff']
        prefix=note.read_text().split(MARKER)[0]
        lines=[MARKER,'','## Current evidence and handoff','',
            f"Updated {datetime.now(timezone.utc).isoformat()}. Completed LR cases: {len(rows)}/12.",
            f"Shared queue owns submitted cases; job prefix `{self.s['job_prefix']}`. Hosts: {', '.join(self.s['hosts'])}.",
            f"Native bundles: `{self.root}/runs/`; execution state: `{self.path}`; summary: `{self.root}/summary.json`.",
            'Queue checks every 60 seconds; semantic-stall limit 45 minutes. Collection and selection use the existing completion hook.',
            f"Readiness admitted: {self.state.get('readiness_admitted',False)}. Blocker: {self.state.get('blocked','none')}.",'',
            '| Case | Final development KL | Writes |','| --- | ---: | ---: |']
        for name,row in rows.items():lines.append(f"| {name} | {row['teacher_kl']:.10g} | {row['costs']['write_attempts']} |")
        lines+=['',f'Zero-write development KL: {baseline}.',
                '```json',json.dumps(summary['interpretation'],indent=2),'```','',
                'Pending/failed coverage is not successful recovery. No result is claimed from an unfinished trajectory.','']
        note.write_text(prefix+'\n'.join(lines))
        self.save()

    def advance(self):
        self.initialize()
        self.reconcile_failures()
        self.check_readiness()
        if self.state.get('readiness_admitted'):self.select()
        # Reserve every candidate and all three final validators before admission.
        charge=1800 if 'selected' not in self.state['phases'] else 0
        for name,job in self.state['jobs'].items():
            done=self.state['completed'].get(name) or self.state.get('failed',{}).get(name)
            charge+=done['charged_seconds'] if done else job['budget_seconds']
        self.state['budget']={'charged_plus_reserved_seconds':charge,'limit_seconds':self.s['gpu_budget_seconds']}
        if charge>self.s['gpu_budget_seconds']:
            self.state['blocked']='Aggregate GPU budget exhausted.'
        used=self.storage_usage();free=self.free_storage()
        reserved=sum(j['storage_reservation_bytes'] for n,j in self.state['jobs'].items()
                     if j.get('submitted') and n not in self.state['completed'] and n not in self.state.get('failed',{}))
        deferred=[]
        for name in runnable(self.state['jobs'],self.state['completed']):
            job=self.state['jobs'][name]
            if self.state.get('blocked') or (job['phase']!='readiness' and not self.state.get('readiness_admitted')):continue
            needed=job['storage_reservation_bytes']+self.input_copy_bytes(job)
            if used+reserved+needed>self.s['storage_budget_bytes'] or reserved+needed+2*1024**3>free:
                deferred.append(name);continue
            self.submit(name);reserved+=needed
        self.state['storage']={'observed_bytes':used,'reserved_bytes':reserved,'deferred':deferred,
            'limit_bytes':self.s['storage_budget_bytes'],'local_free_bytes':free}
        self.report()


@contextmanager
def locked(settings):
    root=Path(read(settings)['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield BoundedRecoveryExecution(settings)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','advance','complete','report'))
    parser.add_argument('--settings',required=True,type=Path)
    args=parser.parse_args()
    with locked(args.settings) as execution:
        if args.action=='prepare':execution.initialize();execution.report()
        elif args.action=='report':execution.report()
        else:
            if args.action=='complete':execution.collect(os.environ['RUN_WATCH_JOB_ID'])
            execution.advance()


if __name__=='__main__':main()
