"""Frozen candidates for recovery-data-preparation exp-003 (issue #25).

All evaluation LR candidates are prepared before the canary. The launcher
selects one per method using only realization 271004 and validation 64:128.
"""
import copy
import json
from pathlib import Path

import make_measured_g_lifecycles as measured
import make_screen_lifecycles as screen
from make_sample_count_lifecycles import GATE

RATES=(1e-3,3e-3,1e-2)
MODES=('random:32','importance:32','loss_only:32','context:32')

def name(mode): return mode.replace(':','-').replace('_','-')
def arm(mode,lr):
    return {**screen.lora('TB-'+name(mode)+'-lr'+format(lr,'g'),screen.EVENTS,lr),
            'target':'samples:4','token_backward':mode,'microbatch':1}

def unit(seed,part,arms,evaluation=None):
    doc=copy.deepcopy(measured.unit(seed,'tokens-'+part,arms,
        question='Does full-context token-selective LoRA backward recover drift better than short context?',
        evaluation=evaluation))
    doc['question']['decision']='recovery-data-preparation exp-003: retained recovery versus working memory and work'
    return doc

def documents():
    out=[]
    full=arm('full',3e-3)
    native={**screen.lora('T-samp4-native-lr0.003',screen.EVENTS,3e-3),'target':'samples:4'}
    controls=[{'label':'A0','method':'none'},screen.lora('A5-lr0.01',screen.EVENTS,1e-2),GATE,native,full]
    canary=unit(screen.TUNE_REALIZATION,'canary',controls+[arm(m,3e-3) for m in MODES],screen.TUNE_COHORT)
    canary['drift_regimes']['canary']=True
    canary['drift_regimes']['adaptation']['steps']=3
    for a in canary['drift_regimes']['arms']:
        if 'events' in a: a['events']=[900.]
    out.append(canary)
    for mode in MODES:
        out.append(unit(screen.TUNE_REALIZATION,'tune-'+name(mode),
                        [GATE]+[arm(mode,lr) for lr in RATES],screen.TUNE_COHORT))
    for seed in screen.REALIZATIONS:
        out.append(unit(seed,'eval-reference',copy.deepcopy(controls)))
        for mode in MODES:
            for lr in RATES:
                out.append(unit(seed,'eval-'+name(mode)+'-lr'+format(lr,'g'),[GATE,arm(mode,lr)]))
    return out

def main():
    root=Path(__file__).resolve().parent/'lifecycles'
    docs=documents()
    for doc in docs:
        (root/(doc['lifecycle_id']+'.json')).write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps({'written':len(docs),'ids':[d['lifecycle_id'] for d in docs]}))

if __name__=='__main__': main()
