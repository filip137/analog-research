from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode


@dataclass(frozen=True)
class Config:
    experiment_id: str = 'pseudo_drn_large.v1'
    schema_version: int = 1
    mlp: str = 'drn'
    method: str = 'ep'
    width: int = 256
    depth: int = 6
    heads: int = 8
    hidden: int = 1024
    steps: int = 32
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = .0005
    weight_decay: float = .05
    warmup_epochs: int = 5
    mixup: float = .2
    label_smoothing: float = .1
    seed: int = 20261115
    wall_seconds: int = 36000
    probe_every: int = 20
    max_updates: int = 0
    data_root: str = '/home/filip/datasets/cifar10/cifar-10-batches-py'
    budget_file: str = '/tmp/pseudo-drn-large-20261115/budget.json'


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw)-{f.name for f in fields(Config)}:
        raise ConfigError('Unknown configuration fields')
    c = Config(**raw)
    if c.experiment_id != 'pseudo_drn_large.v1' or c.schema_version != 1:
        raise ConfigError('Wrong experiment/version')
    if c.mlp not in ('drn', 'digital') or c.method not in ('ep', 'bptt'):
        raise ConfigError('Invalid model/training method')
    for name, limit in [('width',256),('depth',6),('heads',8),('hidden',1024),('steps',64),
                        ('epochs',200),('batch_size',128),('wall_seconds',36000),('probe_every',200)]:
        if type(getattr(c,name)) is not int or not 1 <= getattr(c,name) <= limit:
            raise ConfigError(name)
    if c.width % c.heads or not 0 <= c.warmup_epochs <= c.epochs or type(c.max_updates) is not int or c.max_updates < 0:
        raise ConfigError('Shape/schedule')
    if not 0 < c.learning_rate <= .01 or not 0 <= c.weight_decay <= 1 or not 0 <= c.mixup <= 1 or not 0 <= c.label_smoothing < 1:
        raise ConfigError('Optimizer/augmentation')
    return c


def resolve_config(c, mode):
    if mode != RunMode.TRAIN:
        raise ConfigError('Train only')
    return c
