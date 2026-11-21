"""Generate one native EBL campaign per architecture, fault rate and task; every stage is an EBL train."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ebl.execution.schema import CampaignSpec
from ebl.schema.opt_cmo_drn import parse_config
from ebl.runtime.store import sha256_file

BASE = Path('/home/filip/analog-research')
PYTHON = Path('/home/filip/miniconda3/envs/py312/bin/python')
METHODS = [('lora', .01), ('child', .01), ('child', .1), ('full', .01)]
ROUTES = [f'{name}{int(100*f) if name == "child" else ""}-{kind}'
          for name,f in METHODS for kind in ('offline','online') if (name,kind) != ('full','online')]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def prepare(a):
    receipts = json.loads(a.receipt.read_text())
    template = json.loads((ROOT/'examples/opt_cmo_drn/mapping_faults'/f'{a.model}-ratio3-r{a.rate}-271002-deploy.json').read_text())
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = []
    scope = {'model':a.model,'rate':a.rate,'task':a.task}
    selection = None
    if a.phase == 'confirm':
        if not a.selection:
            raise ValueError('Confirmation requires frozen development winners.')
        selection = json.loads(a.selection.read_text())
        if selection['scope'] != scope or not set(a.route or ROUTES).issubset(selection['winners']):
            raise ValueError('Selection was not frozen for the complete matching development screen.')
    weights = 'dfb9f22f' if a.model == 'drn6' else '6805b629'
    def add(label, action, method='lora', information='recipient', fraction=.01, lr=.001, **artifacts):
        assignment = 281102 if information == 'proxy' else (281201 if a.phase == 'confirm' else 281101)
        raw = dict(template)
        raw['deployment'] = {**raw['deployment'],'assignment_seed':assignment}
        raw['stage'] = 'task_adaptation'
        settings = {'task':a.task, 'task_sha256':receipts[a.task]['sha256'], 'wiki_sha256':receipts['wikitext']['sha256'],
                    'action':action,'method':method,'information':information,'fraction':fraction,'lr':lr,
                    'microbatch':a.microbatch,'eval_microbatch':a.eval_microbatch,
                    'split':'test' if a.phase == 'confirm' and information == 'recipient' else 'development'}
        if a.phase == 'canary':
            settings.update(steps=2, fisher_items=32, calibration_items=4, limit=8)
        raw['task_adaptation'] = settings
        parse_config(raw)
        name = label + '.json'
        write(out/name, raw)
        inputs = {
            'weights':{'path':str(BASE/'cmo-drift-recovery-20261118/inputs'/f'source-{weights}.pt')},
            'teacher_weights':{'path':str(BASE/'opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin')}}
        if information != 'clean':
            bank = a.maps/f'{a.model}-ratio3-r{a.rate}-{assignment}-deploy/run/checkpoints/deployed.pt'
            inputs['device_state'] = {'path':str(bank)}
        dependencies = []
        for key,value in artifacts.items():
            if isinstance(value, Path):
                inputs[key] = {'path':str(value.resolve())}
            else:
                inputs[key] = {'stage':value,'artifact_kind':key}
                dependencies.append(value)
        stages.append({'id':label, 'case_id':label,'target':'drn','command':'train','config':name,
                       'depends_on':dependencies,'inputs':inputs})
        return label

    add('controls','controls')
    ranks = {}
    def rank(info):
        if info not in ranks:
            ranks[info] = add('rank-'+info,'rank','child',info)
        return ranks[info]

    for method,fraction in METHODS:
        tag = method + (str(int(100*fraction)) if method=='child' else '')
        center = .001 if method=='child' or (method=='lora' and a.task=='pubmed') else (
            .0003 if method=='lora' or a.task=='pubmed' else .00003)
        for kind in ('online','offline'):
            route = tag+'-'+kind
            if route not in ROUTES or (a.route and route not in a.route):
                continue
            if a.phase == 'confirm':
                winner = selection['winners'][route]
                if kind == 'offline':
                    fit = Path(winner['fit'])
                    if sha256_file(fit) != winner['sha256']:
                        raise ValueError(f'Frozen fit changed: {route}.')
                    add(route,'receive',method,fraction=fraction,lr=winner['lr'],task_fit=fit)
                else:
                    kwargs = {'task_masks':rank('recipient')} if method=='child' else {}
                    add(route,'learn',method,fraction=fraction,lr=winner['lr'],**kwargs)
                continue
            infos = ('recipient',) if kind == 'online' else ('clean','proxy')
            for info in infos:
                kwargs = {'task_masks':rank(info)} if method=='child' else {}
                rates = [center/3,center,center*3] if a.phase=='development' else [.05 if method=='child' else center]
                for i,lr in enumerate(rates):
                    prefix = f'{route}-{info}-lr{i}'
                    if kind == 'offline':
                        trained = add(prefix+'-fit','fit',method,info,fraction,lr,**kwargs)
                        add(prefix,'receive',method,fraction=fraction,lr=lr,task_fit=trained)
                    else:
                        add(prefix,'learn',method,info,fraction,lr,**kwargs)
    manifest = {'schema_version':1,'campaign_id':'-'.join([a.model,a.rate,a.task,a.phase]),
                'targets':[{'id':'drn','worktree':str(a.worktree.resolve()),'python':str(PYTHON)}],
                'stages':stages}
    CampaignSpec.parse(manifest,base_dir=out)
    write(out/'campaign.json',manifest)
    print(out/'campaign.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['drn6','drn12'],required=True)
    p.add_argument('--rate',choices=['00','03'],required=True)
    p.add_argument('--task',choices=['pubmed','hellaswag','boolq'],required=True)
    p.add_argument('--phase',choices=['canary','development','confirm'],required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worktree',type=Path,default=ROOT)
    p.add_argument('--maps',type=Path,default=BASE/'s7-task-faults/maps')
    p.add_argument('--microbatch',type=int,default=1)
    p.add_argument('--eval-microbatch',type=int,default=1)
    p.add_argument('--selection',type=Path)
    p.add_argument('--route',choices=ROUTES,action='append',help='Budgeted subset to materialize for this launch.')
    prepare(p.parse_args())
