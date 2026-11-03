"""Predeclared optimizer ladder and budget decisions for the first DRN study.

This module never guesses missing scores, ranks partial coverage or tunes on
held-out data. Queue transport is separate from these deterministic decisions.
"""

from dataclasses import asdict,replace
import json
import math
from pathlib import Path

from experiments.opt_cmo_drn.config import Config,Training,parse_config
from experiments.study_workflow import inspect_run
from experiments.artifacts import content_hash

TIERS={'full':(256,512,2048),'medium':(256,512,1024),'minimum':(256,256,512)}
GPU_BUDGET_SECONDS=24*3600
PLANNED_SECONDS=.8*GPU_BUDGET_SECONDS


def _config(base,recipe,conductance,digital,calibration,updates):
    return replace(base,stage='hwa',training=replace(base.training,arm='std3',recipe=recipe,
        conductance_lr=conductance,digital_lr=digital,calibration_lr=calibration,
        max_updates=updates,warmup_updates=32,schedule='constant',retention='final'))


def screen_configs(base,tier):
    return {f'screen-{recipe}-{i}':_config(base,recipe,lr,1e-5,lr,TIERS[tier][0])
            for recipe in ('A','B') for i,lr in enumerate((1e-4,3e-4,1e-3,3e-3))}


def choose(rows):
    if not rows or any(not math.isfinite(r['score']) for r in rows):
        raise ValueError('Expected complete finite candidate scores.')
    minimum=min(r['score'] for r in rows)
    tied=[r for r in rows if r['score']<=minimum+.01*abs(minimum)]
    def order(row):
        c=row['config'].training
        return c.conductance_lr,c.digital_lr,c.calibration_lr,c.recipe,row['id']
    return min(tied,key=order)


def verify_coverage(expected,runs):
    """One explicit complete native bundle per declared candidate; no recency lookup."""
    if set(expected)!=set(runs):
        raise ValueError(f'Candidate coverage differs: missing={set(expected)-set(runs)}, extra={set(runs)-set(expected)}.')
    rows=[]
    input_identity=None
    for name,c in expected.items():
        root=Path(runs[name])
        inspected=inspect_run(root,verify_artifacts=True)
        if not inspected['valid'] or not inspected['process_complete']:
            raise ValueError(f'Incomplete or invalid candidate: {name}.')
        resolved=json.loads((root/'config.resolved.json').read_text())
        if content_hash(resolved)!=content_hash(asdict(c)):
            raise ValueError(f'Candidate config mismatch: {name}.')
        result=json.loads((root/'result.json').read_text())
        manifest=json.loads((root/'manifest.json').read_text())
        inputs={item['role']:item['sha256'] for item in manifest['inputs']}
        identity={'weights':result['metrics']['initialization_sha256'],'teacher_weights':inputs.get('teacher_weights')}
        if inputs.get('weights',identity['weights'])!=identity['weights'] or not ('weights' in inputs or 'resume' in inputs) or identity['teacher_weights']!=c.teacher_sha256:
            raise ValueError(f'Candidate inputs are not explicit and matched: {name}.')
        if input_identity is None: input_identity=identity
        elif input_identity!=identity: raise ValueError('Candidates do not share identical initialization and teacher.')
        if result['metrics']['updates']!=c.training.max_updates:
            raise ValueError(f'Candidate did not reach its declared update budget: {name}.')
        initial=result['metrics'].get('initial_selection')
        if initial is None: raise ValueError(f'Expected initialization score for {name}.')
        score=result['metrics']['final_selection']['teacher_kl']
        if not math.isfinite(score): raise ValueError(f'Nonfinite score: {name}.')
        rows.append({'id':name,'config':c,'score':score,'initial_score':initial['teacher_kl'],'run_dir':str(root),
                     'mapped_sha256':identity['weights']})
    return rows


def refinement_configs(base,tier,screen_rows):
    if len(screen_rows)!=8: raise ValueError('All eight screens must finish before refinement.')
    output={}
    for recipe in ('A','B'):
        rows=[r for r in screen_rows if r['config'].training.recipe==recipe]
        if len(rows)!=4: raise ValueError('Expected four screen rates per recipe.')
        lr=choose(rows)['config'].training.conductance_lr
        for i,digital in enumerate((3e-6,1e-5)):
            for calibration_name,calibration in (('conductance',lr),('digital',digital)):
                output[f'refine-{recipe}-{i}-{calibration_name}']=_config(base,recipe,lr,digital,calibration,TIERS[tier][1])
    return output


def confirmation_configs(base,tier,refinement_rows):
    if len(refinement_rows)!=8: raise ValueError('All eight refinements must finish before confirmation.')
    output={}
    for recipe in ('A','B'):
        rows=[r for r in refinement_rows if r['config'].training.recipe==recipe]
        if len(rows)!=4: raise ValueError('Expected four refinements per recipe.')
        t=choose(rows)['config'].training
        output[f'confirm-{recipe}']=_config(base,recipe,t.conductance_lr,t.digital_lr,t.calibration_lr,TIERS[tier][2])
    return output


def final_configs(base,confirmation_rows):
    if len(confirmation_rows)!=2 or {r['config'].training.recipe for r in confirmation_rows}!={'A','B'}:
        raise ValueError('Both complete recipe confirmations are required.')
    improved=[r for r in confirmation_rows if r['score']<r['initial_score']]
    if not improved:
        return {},{'verdict':'inconclusive','reason':'No confirmed recipe improves on mapped initialization.'}
    winner=choose(improved)
    chosen=winner['config'].training
    output={arm:replace(base,stage='hwa',training=replace(chosen,arm=arm,epochs=5,max_updates=0,warmup_updates=245,
                        schedule='cosine',minimum_lr_factor=.2,max_wall_seconds=base.training.max_wall_seconds))
            for arm in ('clean','std3')}
    output={arm:replace(c,training=replace(c.training,retention='best_latest_final')) for arm,c in output.items()}
    return output,{'verdict':'confirmed','winner':winner['id'],'score':winner['score'],'initial_score':winner['initial_score']}


def choose_tier(*,update_seconds,selection_seconds,checkpoint_seconds,readiness_seconds):
    values=(update_seconds,selection_seconds,checkpoint_seconds,readiness_seconds)
    if any(not math.isfinite(v) or v<0 for v in values) or update_seconds==0:
        raise ValueError('Expected measured finite timings and positive update time.')
    estimates={}
    for name,(screen,refine,confirm) in TIERS.items():
        updates=8*screen+8*refine+2*confirm
        # Epoch zero, every 512 updates, and the final budget. Final coincides
        # with the interval when divisible by 512.
        selections=8*(1+math.ceil(screen/512))+8*(1+math.ceil(refine/512))+2*(1+math.ceil(confirm/512))
        checkpoints=18+8*math.ceil(screen/512)+8*math.ceil(refine/512)+2*math.ceil(confirm/512)
        estimate=readiness_seconds+updates*update_seconds+selections*selection_seconds+checkpoints*checkpoint_seconds
        estimates[name]=estimate
        if estimate<=PLANNED_SECONDS:
            return name,estimates
    return None,estimates


def write_configs(output,configs,metadata):
    from experiments.artifacts import atomic_write_json
    output=Path(output)
    files={f'{name}.json':asdict(c) for name,c in configs.items()}
    files['ladder.json']={**metadata,'configs':{name:f'{name}.json' for name in configs}}
    for name,value in files.items():
        path=output/name
        if path.exists() and content_hash(json.loads(path.read_text()))!=content_hash(value):
            raise ValueError(f'Refusing to replace a different tuning plan: {path}.')
    output.mkdir(parents=True,exist_ok=True)
    for name,value in files.items():atomic_write_json(output/name,value)


def main(argv=None):
    import argparse
    from experiments.artifacts import atomic_write_json
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='phase',required=True)
    p=sub.add_parser('size');p.add_argument('--timings',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for phase in ('screen','refine','confirm','final'):
        p=sub.add_parser(phase)
        p.add_argument('--base-config',type=Path,required=True)
        p.add_argument('--budget-decision',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        if phase!='screen':
            p.add_argument('--previous',type=Path,required=True,help='Previous phase ladder.json')
            p.add_argument('--runs',type=Path,required=True,help='JSON mapping candidate IDs to explicit native bundle paths')
    args=parser.parse_args(argv)
    if args.phase=='size':
        timings=json.loads(args.timings.read_text())
        tier,estimates=choose_tier(**timings)
        decision={'schema':'opt_cmo_drn.tuning_budget.v1','tier':tier,'estimates_seconds':estimates,
                  'measurements':timings,'gpu_cap_seconds':GPU_BUDGET_SECONDS,'planning_cap_seconds':PLANNED_SECONDS}
        atomic_write_json(args.output,decision)
        print(json.dumps(decision,indent=2))
        return 0 if tier else 2
    base=parse_config(json.loads(args.base_config.read_text()))
    decision=json.loads(args.budget_decision.read_text())
    tier=decision['tier']
    if tier not in TIERS or decision['estimates_seconds'][tier]>PLANNED_SECONDS:
        raise ValueError('No measured complete tuning tier fits the declared allowance.')
    metadata={'schema':'opt_cmo_drn.tuning_ladder.v1','phase':args.phase,'tier':tier,
              'base_config_sha256':content_hash(asdict(base)),'budget_decision_sha256':content_hash(decision)}
    if args.phase=='screen':
        configs=screen_configs(base,tier)
    else:
        previous=json.loads(args.previous.read_text())
        expected_phase={'refine':'screen','confirm':'refine','final':'confirm'}[args.phase]
        if previous['phase']!=expected_phase or previous['tier']!=tier or previous['base_config_sha256']!=metadata['base_config_sha256']:
            raise ValueError('Previous phase or base configuration differs from the ladder.')
        expected={name:parse_config(json.loads((args.previous.parent/path).read_text())) for name,path in previous['configs'].items()}
        rows=verify_coverage(expected,json.loads(args.runs.read_text()))
        mapped=rows[0]['mapped_sha256']
        if previous.get('mapped_sha256',mapped)!=mapped:
            raise ValueError('Tuning tiers use different mapped initializations.')
        metadata['mapped_sha256']=mapped
        metadata['previous_results']=[{k:v for k,v in r.items() if k!='config'} for r in rows]
        if args.phase=='refine':configs=refinement_configs(base,tier,rows)
        elif args.phase=='confirm':configs=confirmation_configs(base,tier,rows)
        else:
            configs,verdict=final_configs(base,rows)
            metadata['decision']=verdict
    write_configs(args.output,configs,metadata)
    print(args.output/'ladder.json')
    return 0


if __name__=='__main__':raise SystemExit(main())
