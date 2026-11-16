"""Bump-chain attention scorer feeding the explicit MOS pseudoconductance bank."""
from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID = 'bump_scorer_attention.v1'


@dataclass(frozen=True)
class Config:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    model: str = 'bsim3'
    seed: int = 20261116
    wall_seconds: int = 600
    transient_seconds: float = .5
    tail_current: float = 8e-10
    spice_executable: str = '/tmp/pseudoconductance-spice-env/bin/ngspice'


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Unknown bump scorer configuration field')
    c = Config(**raw)
    if c.experiment_id != EXPERIMENT_ID or type(c.schema_version) is not int or c.schema_version != 1:
        raise ConfigError('Expected bump_scorer_attention.v1 schema 1')
    if c.model not in ('ideal', 'bsim3'):
        raise ConfigError('Expected ideal or bsim3 model')
    if type(c.transient_seconds) not in (int, float) or not .02 <= c.transient_seconds <= .5:
        raise ConfigError('Transient duration must be in [0.02,0.5] seconds')
    if type(c.tail_current) not in (int, float) or not 1e-11 <= c.tail_current <= 2e-9:
        raise ConfigError('Tail current must be in [10 pA, 2 nA]')
    if type(c.seed) is not int or not 0 <= c.seed < 2**32:
        raise ConfigError('Invalid seed')
    if type(c.wall_seconds) is not int or not 1 <= c.wall_seconds <= 600:
        raise ConfigError('Out-of-budget wall_seconds')
    if not isinstance(c.spice_executable, str) or not c.spice_executable:
        raise ConfigError('Expected explicit simulator path')
    return c


def resolve_config(config, mode):
    if mode != RunMode.CHARACTERIZE:
        raise ConfigError('This pilot supports characterize only')
    return config
