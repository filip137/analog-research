"""Read-only trained-checkpoint iteration/gradient diagnostic, GPU FP32."""
from dataclasses import dataclass,fields
from ebl.schema.base import ConfigError,RunMode

EXPERIMENT_ID='energy_transformer_iterations.v1'


@dataclass(frozen=True)
class Config:
    experiment_id:str=EXPERIMENT_ID
    schema_version:int=1
    case:str='small'
    iterations:tuple=(8,16,32,64,96,128)
    reference_steps:tuple=(512,1024)
    batches:int=3
    batch_size:int=8
    beta:float=.005
    threshold:float=.9
    wall_seconds:int=600
    dataset_root:str='/home/filip/datasets/mnist'


def parse_config(raw):
    if not isinstance(raw,dict) or set(raw)-{f.name for f in fields(Config)}:
        raise ConfigError('Expected declared gradient diagnostic fields')
    data=dict(raw)
    for name in ('iterations','reference_steps'):
        if name in data:
            if not isinstance(data[name],(tuple,list)):raise ConfigError('Expected iteration sequence')
            data[name]=tuple(data[name])
    c=Config(**data)
    if c.experiment_id!=EXPERIMENT_ID or type(c.schema_version) is not int or c.schema_version!=1:
        raise ConfigError('Unknown iteration diagnostic schema')
    if c.case not in ('small','reference','more_tokens'):raise ConfigError('Expected a declared trained model size')
    if not c.iterations or any(type(t) is not int or not 1<=t<=128 for t in c.iterations) or tuple(sorted(set(c.iterations)))!=c.iterations:
        raise ConfigError('Candidate phase caps must increase and stay within 128')
    if c.reference_steps!=(512,1024):raise ConfigError('Reference qualification is frozen at 512 versus 1024')
    for name,upper in (('batches',8),('batch_size',8),('wall_seconds',600)):
        v=getattr(c,name)
        if type(v) is not int or not 1<=v<=upper:raise ConfigError(f'Invalid {name}')
    if c.beta!=.005 or c.threshold!=.9:raise ConfigError('Frozen beta .005 and target cosine .9')
    return c


def resolve_config(config,mode):
    if mode!=RunMode.VALIDATE:raise ConfigError('This trained-checkpoint diagnostic supports validate only')
    return config
