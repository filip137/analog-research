"""Summed anti-bump attention scorer (star or Elfadel-Wyatt softmax) driving the bank."""
from dataclasses import dataclass, fields
import math
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID = 'antibump_scorer_attention.v1'


@dataclass(frozen=True)
class Config:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    model: str = 'bsim3'
    variant: str = 'softmax'
    seed: int = 20261116
    wall_seconds: int = 600
    transient_seconds: float = .5
    tail_current: float = 1e-9
    s_ratio: float = 16.
    amplitude: float = .57
    v_low: float = .6
    v_cm: float = 1.05
    scale: float = .75
    kappa_design: float = .7911
    temperature_g: float = 3.5
    projection: tuple = ()
    value_weights: tuple = ()
    weights_source: str = ''
    weights_source_sha256: str = ''
    spice_executable: str = '/tmp/pseudoconductance-spice-env/bin/ngspice'


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Unknown anti-bump configuration field')
    raw = dict(raw)
    pj = raw.get('projection')
    if not (isinstance(pj, list) and len(pj) == 2 and all(isinstance(r, list) and len(r) == 3 for r in pj)
            and all(type(x) in (int, float) and math.isfinite(x) for r in pj for x in r)):
        raise ConfigError('projection must be a finite 2x3 list')
    raw['projection'] = tuple(tuple(float(x) for x in r) for r in pj)
    vw = raw.get('value_weights')
    if not (isinstance(vw, list) and len(vw) == 3 and all(type(x) in (int, float) and math.isfinite(x) for x in vw)):
        raise ConfigError('value_weights must be three finite numbers')
    raw['value_weights'] = tuple(float(x) for x in vw)
    c = Config(**raw)
    if c.experiment_id != EXPERIMENT_ID or type(c.schema_version) is not int or c.schema_version != 1:
        raise ConfigError('Expected antibump_scorer_attention.v1 schema 1')
    if c.model not in ('ideal', 'bsim3') or c.variant not in ('star', 'softmax'):
        raise ConfigError('Expected model ideal|bsim3 and variant star|softmax')
    if not (1e-11 <= c.tail_current <= 2e-9 and 1 <= c.s_ratio <= 100 and 0 < c.amplitude <= 2):
        raise ConfigError('Tail current, S ratio or amplitude out of range')
    if not (0 < c.kappa_design < 2 and 0 < c.temperature_g < 50 and .3 <= c.v_low <= 1.2):
        raise ConfigError('Invalid design constants')
    if not (0 < c.scale and c.v_cm - c.scale >= 0 and c.v_cm + c.scale <= 1.8):
        raise ConfigError('Query/key inputs must stay within the rails')
    if type(c.transient_seconds) not in (int, float) or not .02 <= c.transient_seconds <= .5:
        raise ConfigError('Transient duration must be in [0.02,0.5] seconds')
    if not c.weights_source or len(c.weights_source_sha256) != 64:
        raise ConfigError('Trained-weight provenance (source and sha256) is required')
    if type(c.seed) is not int or not 0 <= c.seed < 2**32:
        raise ConfigError('Invalid seed')
    if type(c.wall_seconds) is not int or not 1 <= c.wall_seconds <= 600:
        raise ConfigError('Out-of-budget wall_seconds')
    return c


def resolve_config(config, mode):
    if mode != RunMode.CHARACTERIZE:
        raise ConfigError('This pilot supports characterize only')
    return config
