"""Strict, bounded exploratory sparse-equilibrium pilot configuration."""
from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID = 'sparse_equilibrium_pilot.v1'

@dataclass(frozen=True)
class Config:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    task: str = 'canary'
    input_root: str = ''
    device: str = 'cuda'
    cpu_threads: int = 10
    seed: int = 271109
    source_updates: int = 128
    updates: int = 128
    batch: int = 8
    mask_size: int = 46080
    fisher_examples: int = 16
    learning_rates: tuple = (0.0003, 0.001)
    wall_seconds: int = 6600

def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Expected only declared sparse-equilibrium fields.')
    d = dict(raw)
    if 'learning_rates' in d:
        d['learning_rates'] = tuple(d['learning_rates'])
    c = Config(**d)
    if c.experiment_id != EXPERIMENT_ID or c.schema_version != 1:
        raise ConfigError('Expected sparse_equilibrium_pilot.v1 schema 1.')
    if c.task not in ('canary', 'mlp', 'drn', 'graph') or not c.input_root:
        raise ConfigError('Expected explicit input_root and supported task.')
    if c.device not in ('cpu', 'cuda'):
        raise ConfigError('Expected cpu or cuda.')
    for k in ('cpu_threads', 'source_updates', 'updates', 'batch', 'mask_size', 'fisher_examples', 'wall_seconds'):
        if type(getattr(c,k)) is not int or getattr(c,k) <= 0:
            raise ConfigError(f'Expected positive integer {k}.')
    if c.mask_size % 12 or c.wall_seconds > 6900 or c.updates > 128 or c.source_updates > 128:
        raise ConfigError('Expected twelve equal mask quotas and bounded first-pilot budget.')
    if not c.learning_rates or any(not 0 < x <= 0.01 for x in c.learning_rates):
        raise ConfigError('Expected bounded positive learning rates.')
    return c

def resolve_config(config, mode):
    if mode != RunMode.TRAIN:
        raise ConfigError('Sparse-equilibrium pilot supports train only.')
    return config
