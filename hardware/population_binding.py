"""hardware.population_binding — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from core.tensor_io import clone_cpu
from hardware.plants import population_fingerprint
from hardware.faults import om_population
from core.seeds import binding_seed


def combined_om(populations, layout, dataset, seed, endpoint, kind, rate):
    pieces=[]
    for m in layout:
        raw=populations['layers'][str(seed)][m.name]
        if raw['size']!=m.size or raw['seed']!=binding_seed(seed,dataset,m.name): raise ValueError('OM layer binding mismatch.')
        pieces.append(om_population(raw,kind,rate,binding_seed(seed,dataset,m.name)))
    p={k:clone_cpu(v) for k,v in pieces[0].items() if k not in ('hidden','hidden_sha256','fingerprint')}
    if any(x['constants']!=p['constants'] for x in pieces): raise ValueError('Different OM presets.')
    p.update(size=sum(m.size for m in layout),seed=seed,
             hidden={k:torch.cat([x['hidden'][k] for x in pieces],1) for k in pieces[0]['hidden']},
             layer_names=[m.name for m in layout],
             trajectory_seed_blocks=[(m.size,binding_seed(endpoint,dataset,m.name)) for m in layout])
    p['fingerprint']=population_fingerprint(p)
    return p
