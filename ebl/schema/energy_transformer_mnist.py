"""Strict bounded FP32 MNIST EP/BP comparison."""
from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID='energy_transformer_mnist.v1'


@dataclass(frozen=True)
class Config:
    experiment_id: str=EXPERIMENT_ID
    schema_version: int=1
    arm: str='energy_ep'
    case: str='small'
    precision: str='fp32'
    phase_mode: str='settled'
    seed: int=20261114
    updates: int=2000
    batch_size: int=64
    learning_rate: float=.003
    beta: float=.005
    epsilon: float=.5
    residual_tolerance: float=1e-6
    max_steps: int=4096
    train_seconds: int=1500
    wall_seconds: int=2100
    evaluation_count: int=10000
    validation_count: int=5000
    probe_count: int=256
    dataset_root: str='/home/filip/datasets/mnist'


CASES={'small':(16,32,7),'reference':(16,64,7),'wider':(16,128,7),'more_tokens':(49,64,4)}


def parse_config(raw):
    if not isinstance(raw,dict) or set(raw)-{f.name for f in fields(Config)}:
        raise ConfigError('Expected declared energy Transformer MNIST fields only')
    c=Config(**raw)
    if c.experiment_id!=EXPERIMENT_ID or type(c.schema_version) is not int or c.schema_version!=1:
        raise ConfigError('Expected energy_transformer_mnist.v1 schema 1')
    if c.arm not in ('energy_ep','digital_bp') or c.case not in CASES or c.precision!='fp32':
        raise ConfigError('Unknown arm/case or non-FP32 precision')
    for name,upper in (('updates',2000),('batch_size',64),('max_steps',4096),
                       ('train_seconds',1500),('wall_seconds',2100),('evaluation_count',10000),
                       ('validation_count',5000),('probe_count',256)):
        if type(getattr(c,name)) is not int or not 1<=getattr(c,name)<=upper:
            raise ConfigError(f'Out-of-budget {name}')
    if c.phase_mode not in ('settled','fixed_steps'):
        raise ConfigError('Expected settled or fixed_steps phase mode')
    if c.phase_mode=='fixed_steps' and c.max_steps>128:
        raise ConfigError('Finite-time phase cap cannot exceed 128')
    if c.wall_seconds<=c.train_seconds or type(c.seed) is not int or not 0<=c.seed<2**32:
        raise ConfigError('Invalid seed or training/evaluation budget')
    for name,value in (('learning_rate',.003),('beta',.005),('epsilon',.5),('residual_tolerance',1e-6)):
        if type(getattr(c,name)) not in (int,float) or getattr(c,name)!=value:
            raise ConfigError(f'This comparison fixes {name}={value}')
    if not isinstance(c.dataset_root,str) or not c.dataset_root:
        raise ConfigError('Expected dataset root')
    return c


def resolve_config(config,mode):
    if mode!=RunMode.TRAIN:
        raise ConfigError('The MNIST comparison supports train only')
    return config
