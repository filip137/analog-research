"""Trained resistive-crossbar query/key projection driving the bump scorer and bank."""
from dataclasses import dataclass, fields
import math
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID = 'crossbar_qk_attention.v1'


@dataclass(frozen=True)
class Config:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    model: str = 'bsim3'
    seed: int = 20261116
    wall_seconds: int = 600
    transient_seconds: float = .5
    tail_current: float = 8e-10
    v_cm: float = 1.05
    scale: float = .75
    conductance_pos: tuple = ()
    conductance_neg: tuple = ()
    value_weights: tuple = ()
    reference_kappa: float = .7911
    reference_amplitude_ratio: float = 1.1495
    weights_source: str = ''
    weights_source_sha256: str = ''
    spice_executable: str = '/tmp/pseudoconductance-spice-env/bin/ngspice'


def _matrix(value, name):
    if not (isinstance(value, list) and len(value) == 2 and all(isinstance(r, list) and len(r) == 3 for r in value)):
        raise ConfigError(f'{name} must be a 2x3 list')
    flat = [x for r in value for x in r]
    if not all(type(x) in (int, float) and math.isfinite(x) and 1e-7 <= x <= 1e-3 for x in flat):
        raise ConfigError(f'{name} entries must be conductances in [0.1 uS, 1 mS]')
    return tuple(tuple(float(x) for x in r) for r in value)


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Unknown crossbar configuration field')
    raw = dict(raw)
    raw['conductance_pos'] = _matrix(raw.get('conductance_pos'), 'conductance_pos')
    raw['conductance_neg'] = _matrix(raw.get('conductance_neg'), 'conductance_neg')
    vw = raw.get('value_weights')
    if not (isinstance(vw, list) and len(vw) == 3 and all(type(x) in (int, float) and math.isfinite(x) for x in vw)):
        raise ConfigError('value_weights must be three finite numbers')
    raw['value_weights'] = tuple(float(x) for x in vw)
    c = Config(**raw)
    if c.experiment_id != EXPERIMENT_ID or type(c.schema_version) is not int or c.schema_version != 1:
        raise ConfigError('Expected crossbar_qk_attention.v1 schema 1')
    if c.model not in ('ideal', 'bsim3'):
        raise ConfigError('Expected ideal or bsim3 model')
    if type(c.transient_seconds) not in (int, float) or not .02 <= c.transient_seconds <= .5:
        raise ConfigError('Transient duration must be in [0.02,0.5] seconds')
    if type(c.tail_current) not in (int, float) or not 1e-11 <= c.tail_current <= 2e-9:
        raise ConfigError('Tail current must be in [10 pA, 2 nA]')
    if not (0 < c.v_cm < 1.8 and 0 < c.scale and c.v_cm - c.scale >= 0 and c.v_cm + c.scale <= 1.8):
        raise ConfigError('Inputs V_cm +/- scale must stay within the 0-1.8 V rails')
    if not (0 < c.reference_kappa < 2 and 0 < c.reference_amplitude_ratio < 10):
        raise ConfigError('Invalid reference law constants')
    if not c.weights_source or len(c.weights_source_sha256) != 64:
        raise ConfigError('Trained-weight provenance (source and sha256) is required')
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
