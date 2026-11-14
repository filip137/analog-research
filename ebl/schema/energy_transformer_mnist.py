"""Strict bounded FP32 MNIST EP/BP comparison."""
from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID='energy_transformer_mnist.v1'


@dataclass(frozen=True)
class Config:
    experiment_id: str=EXPERIMENT_ID
    schema_version: int=1
    arm: str='energy_ep'
    energy_architecture: str='recurrent'
    case: str='small'
    precision: str='fp32'
    phase_mode: str='settled'
    device_domain: str='declared4'
    solver_dt: float=.5
    solver_rule: str="energy_backtracking"
    initialization: str="standard"
    recurrent_constraint: str="none"
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
    if c.arm not in ('energy_ep','energy_bp','digital_bp') or c.case not in CASES or c.precision!='fp32':
        raise ConfigError('Unknown arm/case or non-FP32 precision')
    if c.energy_architecture not in ('recurrent','input_conditioned'):
        raise ConfigError('Unknown energy architecture')
    if c.arm=='digital_bp' and c.energy_architecture!='recurrent':
        raise ConfigError('Digital baseline does not use an energy architecture')
    if c.arm=='energy_bp' and c.phase_mode!='fixed_steps':
        raise ConfigError('Matched energy BP requires a fixed trajectory')
    for name,upper in (('updates',2000),('batch_size',64),('max_steps',4096),
                       ('train_seconds',1500),('wall_seconds',2100),('evaluation_count',10000),
                       ('validation_count',5000),('probe_count',256)):
        if type(getattr(c,name)) is not int or not 1<=getattr(c,name)<=upper:
            raise ConfigError(f'Out-of-budget {name}')
    if c.phase_mode not in ('settled','fixed_steps'):
        raise ConfigError('Expected settled or fixed_steps phase mode')
    if c.device_domain not in ('declared4','physical'):
        raise ConfigError('Expected declared4 or physical MOS domain policy')
    if type(c.solver_dt) not in (int,float) or c.solver_dt not in (.5,1.):
        raise ConfigError('Qualified solver steps are .5 and 1')
    if c.solver_rule not in ('energy_backtracking','cycle_backtracking'):
        raise ConfigError('Unknown finite-time step selection rule')
    if (c.device_domain!='declared4' or c.solver_dt!=.5 or c.solver_rule!='energy_backtracking') and (c.arm not in ('energy_ep','energy_bp') or c.phase_mode!='fixed_steps'):
        raise ConfigError('Physical-domain/full-step repair requires finite-time energy EP')
    if c.initialization not in ('standard','contractive_blocks_v1','input_attention_v1') or c.recurrent_constraint not in ('none','contractive_blocks_v1','input_attention_v1'):
        raise ConfigError('Unknown initialization or recurrent constraint')
    if c.recurrent_constraint == 'contractive_blocks_v1':
        if (c.arm not in ('energy_ep','energy_bp') or c.energy_architecture!='recurrent' or c.initialization != 'contractive_blocks_v1'
                or c.phase_mode != 'fixed_steps' or c.device_domain != 'physical' or c.solver_dt != 1.):
            raise ConfigError('Contraction requires paired initialization, physical finite-time EP and dt1')
    if c.initialization == 'contractive_blocks_v1' and c.arm in ('energy_ep','energy_bp') and c.recurrent_constraint != 'contractive_blocks_v1':
        raise ConfigError('Paired energy initialization requires its declared recurrent constraint')
    if c.energy_architecture=='input_conditioned':
        if (c.arm not in ('energy_ep','energy_bp') or c.case!='more_tokens'
                or c.initialization!='input_attention_v1' or c.recurrent_constraint!='input_attention_v1'
                or c.phase_mode!='fixed_steps' or c.device_domain!='physical' or c.solver_dt!=1.):
            raise ConfigError('Input attention requires its explicit projected 49-token finite-time protocol')
    elif 'input_attention_v1' in (c.initialization,c.recurrent_constraint):
        raise ConfigError('Input-attention policies require that architecture')
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
