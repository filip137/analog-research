"""Emit native deployment/recovery campaigns from one explicitly selected HWA source."""

import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys

import torch

from ebl.execution.schema import CampaignSpec
from ebl.runtime.store import atomic_write_json,content_hash,sha256_file
from ebl.schema.opt_cmo_drn import parse_config


def build_plan(config,teacher,source,output,python,worktree):
    c=parse_config(config)
    teacher=Path(teacher).resolve();source=Path(source).resolve();output=Path(output).resolve()
    if sha256_file(teacher)!=c.teacher_sha256:raise ValueError('Frozen teacher digest differs.')
    payload=torch.load(source,map_location='cpu',weights_only=True)
    expected={'teacher_sha256':c.teacher_sha256,'source_sha256':c.source_sha256,
              'initialization':c.initialization,'solver':asdict(c.solver)}
    if payload.get('schema')!='opt_cmo_drn.weights.v1' or payload.get('contract')!=expected or 'selection' not in payload:
        raise ValueError('Expected an explicit selected/final HWA weights artifact with matching initialization.')
    files={};stages=[]
    for kind in ('none','random'):
        deployment=replace(c.deployment,fault_kind=kind,fault_rate=0. if kind=='none' else .03)
        base=replace(c,deployment=deployment)
        deploy_id='deploy-'+kind
        files[deploy_id+'.json']=asdict(replace(base,stage='deploy'))
        inputs={'teacher_weights':{'path':str(teacher)},'weights':{'path':str(source)}}
        stages.append({'id':deploy_id,'case_id':kind,'target':'local','command':'train',
                       'config':deploy_id+'.json','inputs':inputs})
        for arm in ('hold','calibration','learning','learning_calibration','rewrite'):
            identifier=kind+'-'+arm
            files[identifier+'.json']=asdict(replace(base,stage='recover',recovery=replace(c.recovery,arm=arm)))
            parent=kind+'-learning' if arm=='learning_calibration' else deploy_id
            artifact='recovery_state' if arm=='learning_calibration' else 'device_state'
            stages.append({'id':identifier,'case_id':identifier,'target':'local','command':'train',
                           'config':identifier+'.json','depends_on':[parent],
                           'inputs':{**inputs,'device_state':{'stage':parent,'artifact_kind':artifact}}})
            stages.append({'id':identifier+'-held-out','case_id':identifier,'target':'local','command':'validate',
                           'config':identifier+'.json','depends_on':[identifier],
                           'inputs':{'teacher_weights':{'path':str(teacher)},
                                     'weights':{'stage':identifier,'artifact_kind':'recovery_state'}}})
    manifest={'schema_version':1,'campaign_id':'opt-cmo-drn-recovery-'+c.training.arm,
              'targets':[{'id':'local','worktree':str(Path(worktree).resolve()),'python':str(Path(python).resolve())}],
              'stages':stages}
    CampaignSpec.parse(manifest,base_dir=output)
    files['campaign.json']=manifest
    files['inputs.json']={'source':{'path':str(source),'sha256':sha256_file(source)},
                          'teacher':{'path':str(teacher),'sha256':c.teacher_sha256},
                          'source_selection':payload['selection'],'expected_stages':len(stages),
                          'expected_physical_cases':2,'expected_recovery_controls':10}
    for name,value in files.items():
        path=output/name
        if path.exists() and content_hash(json.loads(path.read_text()))!=content_hash(value):raise ValueError(f'Existing recovery plan differs: {path}.')
    output.mkdir(parents=True,exist_ok=True)
    for name,value in files.items():atomic_write_json(output/name,value)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    for name in ('teacher-weights','source','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--worktree',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    manifest=build_plan(json.loads(args.config.read_text()),args.teacher_weights,args.source,args.output,args.python,args.worktree)
    print(json.dumps({'manifest':str(args.output/'campaign.json'),'stages':len(manifest['stages'])}))


if __name__=='__main__':main()
