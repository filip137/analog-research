"""End-to-end analog attention step: token crossbar, KV sample-and-hold, anti-bump scorer
with capacitor integration, Elfadel-Wyatt softmax, then bank (chain A) or translinear
value weighting (chain B)."""
from dataclasses import dataclass, fields
import math
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID = 'end_to_end_attention.v1'
GMIN, GMAX = 9e-6, 88.2e-6


@dataclass(frozen=True)
class Config:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    model: str = 'bsim3'
    chain: str = 'A'
    seed: int = 20261117
    wall_seconds: int = 900
    tail_current: float = 1e-9
    s_ratio: float = 16.
    amplitude: float = .57
    v_low: float = .6
    v_cm: float = 1.05
    scale: float = .75
    kappa_design: float = .7911
    temperature_g: float = 5.
    c_score: float = 1e-12
    c_hold: float = 5e-12
    t_write: float = 20e-6
    t_reset: float = 10e-6
    t_settle: float = .3
    hold_sweep: tuple = ()
    key_gp: tuple = ()
    key_gm: tuple = ()
    value_gp: tuple = ()
    value_gm: tuple = ()
    readout_alpha: float = 1.
    readout_gamma: float = 0.
    weights_source: str = ''
    weights_source_sha256: str = ''
    spice_executable: str = '/tmp/pseudoconductance-spice-env/bin/ngspice'


def _matrix(raw, name, rows):
    m = raw.get(name)
    if not (isinstance(m, list) and len(m) == rows and all(isinstance(r, list) and len(r) == 3 for r in m)
            and all(type(x) in (int, float) and GMIN * (1 - 1e-9) <= x <= GMAX * (1 + 1e-9) for r in m for x in r)):
        raise ConfigError(f'{name} must be a {rows}x3 list of conductances in [9, 88.2] uS')
    return tuple(tuple(float(x) for x in r) for r in m)


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Unknown end-to-end attention configuration field')
    raw = dict(raw)
    for name, rows in (('key_gp', 2), ('key_gm', 2), ('value_gp', 1), ('value_gm', 1)):
        raw[name] = _matrix(raw, name, rows)
    hs = raw.get('hold_sweep', [])
    if not (isinstance(hs, list) and len(hs) <= 4 and all(type(x) in (int, float) and .01 <= x <= 5 for x in hs)):
        raise ConfigError('hold_sweep must list at most four hold times in [0.01, 5] s')
    raw['hold_sweep'] = tuple(float(x) for x in hs)
    c = Config(**raw)
    if c.experiment_id != EXPERIMENT_ID or type(c.schema_version) is not int or c.schema_version != 1:
        raise ConfigError('Expected end_to_end_attention.v1 schema 1')
    if c.model not in ('ideal', 'bsim3') or c.chain not in ('A', 'B'):
        raise ConfigError('Expected model ideal|bsim3 and chain A|B')
    if not (1e-11 <= c.tail_current <= 2e-9 and 1 <= c.s_ratio <= 100 and 0 < c.amplitude <= 2):
        raise ConfigError('Tail current, S ratio or amplitude out of range')
    if not (0 < c.kappa_design < 2 and 0 < c.temperature_g < 50 and .3 <= c.v_low <= 1.2):
        raise ConfigError('Invalid design constants')
    if not (0 < c.scale and c.v_cm - c.scale >= 0 and c.v_cm + c.scale <= 1.8):
        raise ConfigError('Token inputs must stay within the rails')
    if not (1e-14 <= c.c_score <= 1e-10 and 1e-14 <= c.c_hold <= 1e-10):
        raise ConfigError('Capacitors out of range')
    if not (1e-6 <= c.t_write <= 1e-3 and 1e-6 <= c.t_reset <= 1e-3 and .005 <= c.t_settle <= 1):
        raise ConfigError('Timing out of range')
    if not all(math.isfinite(x) for x in (c.readout_alpha, c.readout_gamma)):
        raise ConfigError('Readout must be finite')
    if not c.weights_source or len(c.weights_source_sha256) != 64:
        raise ConfigError('Trained-weight provenance (source and sha256) is required')
    if type(c.seed) is not int or not 0 <= c.seed < 2**32:
        raise ConfigError('Invalid seed')
    if type(c.wall_seconds) is not int or not 1 <= c.wall_seconds <= 1800:
        raise ConfigError('Out-of-budget wall_seconds')
    return c


def resolve_config(config, mode):
    if mode != RunMode.CHARACTERIZE:
        raise ConfigError('This pilot supports characterize only')
    return config
