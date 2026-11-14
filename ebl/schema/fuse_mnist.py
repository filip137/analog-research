"""Bounded exploratory fuse-network comparison on the existing MNIST task."""
from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode
from ebl.schema.energy_transformer_mnist import CASES

EXPERIMENT_ID='fuse_mnist.v1'

@dataclass(frozen=True)
class Config:
    experiment_id: str=EXPERIMENT_ID
    schema_version: int=1
    case: str='small'
    kernel: str='gaussian'
    seed: int=20261114
    updates: int=2000
    batch_size: int=64
    learning_rate: float=.003
    beta: float=.005
    floor: float=.02
    fuse_sigma: float=.25
    fuse_conductance: float=1.
    readout_gain: float=4.
    heads: int=4
    max_steps: int=512
    residual_tolerance: float=1e-6
    train_seconds: int=600
    wall_seconds: int=850
    evaluation_count: int=10000
    validation_count: int=5000
    probe_count: int=256
    permutation_seed: int | None=None
    dataset_root: str='/home/filip/datasets/mnist'

def parse_config(raw):
    if not isinstance(raw,dict) or set(raw)-{f.name for f in fields(Config)}:
        raise ConfigError('Unknown fuse MNIST fields')
    c=Config(**raw)
    if c.experiment_id!=EXPERIMENT_ID or type(c.schema_version) is not int or c.schema_version!=1:
        raise ConfigError('Expected fuse_mnist.v1 schema 1')
    if c.case not in ('small','reference','more_tokens') or c.kernel not in ('gaussian','linear'):
        raise ConfigError('Unknown case/kernel')
    for name,upper in (('updates',2000),('batch_size',64),('max_steps',512),('train_seconds',600),
                       ('wall_seconds',850),('evaluation_count',10000),('validation_count',5000),('probe_count',256)):
        if type(getattr(c,name)) is not int or not 1<=getattr(c,name)<=upper:
            raise ConfigError('Invalid '+name)
    if c.wall_seconds<=c.train_seconds:
        raise ConfigError('Reserve time for evaluation')
    for name,value in (('learning_rate',.003),('beta',.005),('floor',.02),('fuse_sigma',.25),
                       ('fuse_conductance',1.),('readout_gain',4.),('heads',4),('residual_tolerance',1e-6)):
        if type(getattr(c,name)) not in (int,float) or getattr(c,name)!=value:
            raise ConfigError('Fixed pilot setting '+name)
    if type(c.seed) is not int or not 0<=c.seed<2**32:
        raise ConfigError('Invalid seed')
    if c.permutation_seed is not None and (type(c.permutation_seed) is not int or not 0<=c.permutation_seed<2**32):
        raise ConfigError('Invalid permutation seed')
    return c

def resolve_config(config,mode):
    if mode!=RunMode.TRAIN:
        raise ConfigError('Fuse MNIST supports train only')
    return config
