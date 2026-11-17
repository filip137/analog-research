"""Chain A (bank attention) on MNIST, offline GPU training (#63)."""
from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID = 'analog_attention_mnist.v1'


@dataclass(frozen=True)
class Config:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    heads: int = 4
    d_k: int = 4
    d_v: int = 16
    epochs: int = 20
    batch_size: int = 256
    lr: float = .01
    cosine: bool = False
    seed: int = 0
    scale: float = .5
    data_root: str = '/home/filip/datasets/mnist'
    device: str = 'cuda'
    wall_seconds: int = 1800


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Unknown analog_attention_mnist field')
    c = Config(**raw)
    if c.experiment_id != EXPERIMENT_ID or c.schema_version != 1:
        raise ConfigError('Expected analog_attention_mnist.v1 schema 1')
    for k, lo, hi in (('heads', 1, 16), ('d_k', 1, 16), ('d_v', 1, 64), ('epochs', 1, 200), ('batch_size', 8, 4096),
                      ('seed', 0, 2**31), ('wall_seconds', 1, 3600)):
        if type(getattr(c, k)) is not int or not lo <= getattr(c, k) <= hi:
            raise ConfigError(f'{k} out of range')
    if not (0 < c.lr < 1 and 0 < c.scale <= .75 and type(c.cosine) is bool and c.device in ('cuda', 'cpu')):
        raise ConfigError('lr, scale, cosine or device invalid')
    return c


def resolve_config(c, mode):
    if mode != RunMode.TRAIN:
        raise ConfigError('train only')
    return c
