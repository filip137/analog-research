"""Plan and collect preparation/HWA; numerical execution belongs to ebl."""

import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys

from experiments.artifacts import sha256_file, atomic_write_json
from experiments.opt_cmo_drn.config import parse_config
from campaigns.schema import CampaignSpec


def build_plan(config,teacher,source,output,python,worktree,reuse=None):
    from experiments.study_workflow import inspect_run
    from experiments.opt_cmo_drn.data import pinned_load
    config=parse_config(config)
    teacher=Path(teacher).resolve();source=Path(source).resolve();output=Path(output).resolve()
    if sha256_file(teacher)!=config.teacher_sha256 or sha256_file(source)!=config.source_sha256:
        raise ValueError('Expected the exact pinned teacher and std3 initialization inputs.')
    files={}
    stages=[]
    upstream={'stage':'prepare','artifact_kind':'mapped_weights'}
    if reuse:
        run=Path(reuse).resolve()
        record=inspect_run(run,verify_artifacts=True)
        result=json.loads((run/'result.json').read_text())
        if not record['valid'] or result['status']!='complete':
            raise ValueError('Expected a complete hash-verified reuse bundle.')
        matches=[a for a in result['artifacts'] if a['kind']=='mapped_weights']
        if len(matches)!=1: raise ValueError('Expected exactly one mapped artifact in reuse bundle.')
        path=run/matches[0]['path'];payload=pinned_load(path,matches[0]['sha256'])
        expected={'teacher_sha256':config.teacher_sha256,'source_sha256':config.source_sha256,
                  'initialization':config.initialization,'solver':asdict(config.solver)}
        old=parse_config(json.loads((run/'config.resolved.json').read_text()))
        mapping_data=('corpus','sha256','sequences','sequence_length','seed','mapping_sequences')
        if payload['contract']!=expected or any(getattr(old.data,k)!=getattr(config.data,k) for k in mapping_data):
            raise ValueError('Mapped artifact dependencies differ from the requested lifecycle.')
        upstream={'path':str(path)}
    else:
        files['prepare.json']=asdict(replace(config,stage='prepare'))
        stages.append({'id':'prepare','case_id':'prepare','target':'local','command':'train','config':'prepare.json',
                       'inputs':{'teacher_weights':{'path':str(teacher)},'weights':{'path':str(source)}}})
    files['hwa.json']=asdict(replace(config,stage='hwa'))
    stages.append({'id':'hwa','case_id':config.training.arm,'target':'local','command':'train','config':'hwa.json',
                   'depends_on':[] if reuse else ['prepare'],
                   'inputs':{'teacher_weights':{'path':str(teacher)},'weights':upstream}})
    manifest={'schema_version':1,'campaign_id':f'opt-cmo-drn-{config.training.arm}',
              'targets':[{'id':'local','worktree':str(Path(worktree).resolve()),'python':str(Path(python).resolve())}],
              'stages':stages}
    CampaignSpec.parse(manifest,base_dir=output)
    files['campaign.json']=manifest
    for name,value in files.items():
        target=output/name
        if target.exists() and json.loads(target.read_text())!=value:
            raise ValueError(f'Existing plan differs: {target}.')
    output.mkdir(parents=True,exist_ok=True)
    for name,value in files.items(): atomic_write_json(output/name,value)
    return manifest


def collect(paths):
    from experiments.study_workflow import inspect_run
    rows=[]
    for path in paths:
        path=Path(path).resolve()
        if not path.exists():
            rows.append({'run_dir':str(path),'coverage':'missing'});continue
        record=inspect_run(path,verify_artifacts=True)
        try:
            terminal=json.loads((path/'result.json').read_text())
        except (OSError,ValueError):
            terminal={}
        try:
            config=json.loads((path/'config.resolved.json').read_text())
        except (OSError,ValueError):
            config=None
        rows.append({'run_dir':str(path),'coverage':('complete' if record['valid'] else 'invalid')
                     if terminal.get('status')=='complete' else terminal.get('status','partial'),
                     'config':config,'inspection':record,'metrics':terminal.get('metrics')})
    complete=[r for r in rows if r['coverage']=='complete']
    from experiments.artifacts import content_hash
    seen={}
    for row in complete:
        key=content_hash(row['config'])
        if key in seen:
            row['coverage']='duplicate';seen[key]['coverage']='duplicate'
        seen[key]=row
    return {'schema':'opt_cmo_drn.collection.v1','runs':rows,'complete':bool(rows) and all(r['coverage']=='complete' for r in rows)}


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('config',type=Path)
    for name in ('teacher-weights','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reuse',type=Path)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--worktree',type=Path,default=Path(__file__).resolve().parents[1])
    p=sub.add_parser('collect');p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='plan':
        build_plan(json.loads(args.config.read_text()),args.teacher_weights,args.source,args.output,args.python,args.worktree,args.reuse)
        print(args.output/'campaign.json')
    else:
        report=collect(args.runs);atomic_write_json(args.output,report)
        return 0 if report['complete'] else 1
    return 0


if __name__=='__main__': raise SystemExit(main())
