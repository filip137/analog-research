"""Verify operational canaries or freeze the best rates and offline fits from a complete development screen."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from ebl.execution.runner import _validate_run_result
from ebl.runtime.store import sha256_file
from prepare import ROUTES, write


def review(run, output, require_all):
    aggregate = json.loads((run/'aggregate/results.json').read_text())
    model,rate,task,phase = aggregate['campaign_id'].split('-')
    if phase not in ('canary','development','confirm'):
        raise ValueError('Unrecognized phase.')
    summaries, fits, reusable = {}, {}, {}
    for stage in aggregate['stages']:
        name = stage['stage_id']
        if stage['status'] != 'complete':
            raise ValueError(f'Incomplete stage {name}.')
        path = Path(stage['run_result'])
        if sha256_file(path) != stage['run_result_sha256']:
            raise ValueError(f'Run receipt changed for {name}.')
        result = _validate_run_result(path)
        artifacts = {x['kind']:path.parent/x['path'] for x in result['artifacts']}
        row = json.loads(artifacts['task_summary'].read_text())
        summaries[name] = row
        if name=='controls' or name.startswith('rank-'):
            key = 'task_summary' if name=='controls' else 'task_masks'
            artifact = artifacts[key]
            reusable[name] = {'path':str(artifact),'sha256':sha256_file(artifact)}
        if 'task_fit' in artifacts:
            fits[name] = str(artifacts['task_fit'])
        settings = row['settings']
        if settings['action']=='rank':
            if any(row['counts'][str(f)] != round(row['eligible']*f) for f in (.01,.10)):
                raise ValueError(f'Mask counts differ from eligible population: {name}.')
        if 'endpoint' not in row:
            continue
        costs = row['bank_costs']
        if (settings['method']=='lora' or settings['action']=='controls') and costs['write_attempts'] != 0:
            raise ValueError(f'Unexpected analog writes: {name}.')
        if settings['method']=='child':
            if row['attempts_outside_mask'] or (phase=='canary' and costs['device_writes'] < 1):
                raise ValueError(f'Child mask or live write validation failed: {name}.')
    if len({r['bank_sha256'] for r in summaries.values() if 'endpoint' in r}) != 1:
        raise ValueError('Endpoints used different recipients.')
    result = {'scope':{'model':model,'rate':rate,'task':task},'phase':phase,'completed':len(summaries),'reusable':reusable}
    if phase=='development':
        winners = {}
        for route in ROUTES:
            rows = [(name,row) for name,row in summaries.items() if name.startswith(route+'-') and 'endpoint' in row]
            if not rows and not require_all:
                continue
            expected = {'clean','proxy'} if route.endswith('offline') else {'recipient'}
            by_info = {info:set() for info in expected}
            candidates = []
            for name,row in rows:
                info = row.get('training_information','recipient')
                lr = row['settings']['lr']
                by_info[info].add(lr)
                score = row['endpoint']['task']
                loss = score['nll'] if task=='pubmed' else -score['accuracy_percent']
                candidates.append((loss,lr,info,name))
            if any(len(rates)!=3 for rates in by_info.values()) or len(candidates) != 3*len(expected):
                raise ValueError(f'Incomplete matched development trials for {route}.')
            loss,lr,info,name = min(candidates)
            winners[route] = {'lr':lr,'information':info,'stage':name,'score':summaries[name]['endpoint']['task']}
            if info!='recipient':
                artifact = Path(fits[name+'-fit'])
                winners[route].update(fit=str(artifact),sha256=sha256_file(artifact))
        result['winners'] = winners
    write(output,result)
    print(json.dumps({'phase':phase,'stages':len(summaries),'output':str(output)}))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign_root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-all',action='store_true')
    args = p.parse_args()
    review(args.campaign_root,args.output,args.require_all)
