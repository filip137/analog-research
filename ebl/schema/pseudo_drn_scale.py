from dataclasses import dataclass,fields
from ebl.schema.base import ConfigError,RunMode

@dataclass(frozen=True)
class Config:
    experiment_id: str='pseudo_drn_scale.v1'
    schema_version: int=1
    method: str='ep'
    score: str='dot'
    mlp: str='drn'
    updates: int=2000
    seed: int=20261115
    wall_seconds: int=900
    data_root: str='/home/filip/datasets/cifar10/cifar-10-batches-py'

def parse_config(raw):
    if not isinstance(raw,dict) or set(raw)-{f.name for f in fields(Config)}:raise ConfigError('Unknown configuration fields')
    c=Config(**raw)
    if c.experiment_id!='pseudo_drn_scale.v1' or c.schema_version!=1:raise ConfigError('Wrong experiment/version')
    if c.method not in ('ep','bptt') or c.score not in ('dot','l1','l1_raw') or c.mlp not in ('drn','digital'):raise ConfigError('Invalid variant')
    for name,limit in [('updates',2000),('wall_seconds',900)]:
        if type(getattr(c,name)) is not int or not 1<=getattr(c,name)<=limit:raise ConfigError(name)
    return c

def resolve_config(c,mode):
    if mode!=RunMode.TRAIN:raise ConfigError('Train only')
    return c
