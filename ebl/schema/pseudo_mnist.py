from dataclasses import dataclass,fields
from ebl.schema.base import ConfigError,RunMode

@dataclass(frozen=True)
class Config:
    experiment_id: str='pseudo_mnist.v1'
    schema_version: int=1
    task: str='completion'
    method: str='ep'
    steps: int=32
    updates: int=2000
    batch_size: int=64
    seed: int=20261114
    wall_seconds: int=900
    data_root: str='/home/filip/datasets/mnist'
    device: str='cuda'

def parse_config(raw):
    if not isinstance(raw,dict) or set(raw)-{f.name for f in fields(Config)}: raise ConfigError('Unknown fields')
    c=Config(**raw)
    if c.experiment_id!='pseudo_mnist.v1' or c.schema_version!=1: raise ConfigError('Wrong schema')
    if c.task not in ('completion','permuted_classification') or c.method not in ('ep','bptt'): raise ConfigError('Unknown task/method')
    for name,limit in [('steps',128),('updates',2000),('batch_size',64),('wall_seconds',900)]:
        if type(getattr(c,name)) is not int or not 1<=getattr(c,name)<=limit: raise ConfigError(name)
    if type(c.seed) is not int or not 0<=c.seed<2**32: raise ConfigError('seed')
    if c.device not in ('cpu','cuda'): raise ConfigError('device')
    return c

def resolve_config(c,mode):
    if mode!=RunMode.TRAIN: raise ConfigError('train only')
    return c
