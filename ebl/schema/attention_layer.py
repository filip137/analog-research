"""End-to-end analog attention layer (#61): chain B attention, L1 layer norm, DRN MLP."""
from dataclasses import dataclass, fields
import math
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID = 'attention_layer.v1'


@dataclass(frozen=True)
class Config:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    model: str = 'bsim3'
    seed: int = 20261117
    n_examples: int = 96
    wall_seconds: int = 1800
    v_cm: float = 1.05
    scale: float = .5
    s_ratio: float = 4.
    tail_current: float = 1e-9
    amplitude: float = .57
    v_low: float = .6
    c_score: float = 3e-13
    c_hold: float = 3e-13
    t_write: float = 2e-6
    t_reset: float = 2e-4
    t_warm: float = 2e-4
    t_settle: float = .02
    temperature_g: float = 5.
    kappa_design: float = .7911
    geometry: tuple = ()
    norm_tail: float = 1e-11
    eps: float = .05
    norm_width: float = 2e-6
    norm_length: float = 2e-6
    norm_pedestal: float = .25
    v_rect: float = 1.2
    rect_bulk: str = 'vdd'
    v_u: float = .3
    drn_ref: float = .9
    g_s: float = 9e-6
    key_gp: tuple = ()
    key_gm: tuple = ()
    value_gp: tuple = ()
    value_gm: tuple = ()
    drn_g1: tuple = ()
    drn_g2: tuple = ()
    law_temperature_g: float = 5.
    law_cell_s: float = 54.6
    law_cell_kappa: float = .78
    law_kappa: float = .7911
    abstol: float = 3e-15
    weights_source: str = ''
    weights_source_sha256: str = ''
    spice_executable: str = '/tmp/pseudoconductance-spice-env/bin/ngspice'


def _mat(v, shape, name, lo=9e-6 * (1 - 1e-9), hi=88.2e-6 * (1 + 1e-9)):
    if not (isinstance(v, list) and len(v) == shape[0] and all(isinstance(r, list) and len(r) == shape[1] for r in v)
            and all(type(x) in (int, float) and lo <= x <= hi for r in v for x in r)):
        raise ConfigError(f'{name} must be a {shape} list of conductances in [9, 88.2] uS')
    return tuple(tuple(float(x) for x in r) for r in v)


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Unknown attention-layer configuration field')
    raw = dict(raw)
    for k in ('key_gp', 'key_gm', 'value_gp', 'value_gm'):
        raw[k] = _mat(raw.get(k), (2, 4), k)
    g1 = raw.get('drn_g1')
    if not (isinstance(g1, list) and len(g1) == 6 and g1 and isinstance(g1[0], list)):
        raise ConfigError('drn_g1 must be 6 x H')
    raw['drn_g1'] = _mat(g1, (6, len(g1[0])), 'drn_g1')
    g2 = raw.get('drn_g2')
    if not (isinstance(g2, list) and len(g2) == len(g1[0]) and g2 and isinstance(g2[0], list)):
        raise ConfigError('drn_g2 must be H x C')
    raw['drn_g2'] = _mat(g2, (len(g2), len(g2[0])), 'drn_g2')
    geo = raw.get('geometry', {})
    if not isinstance(geo, dict) or not all(isinstance(v, list) and len(v) == 2 and all(type(x) in (int, float) and .02 <= x <= 10 for x in v)
                                            for v in geo.values()):
        raise ConfigError('geometry maps device families to [width_factor, length_factor]')
    raw['geometry'] = tuple(sorted((k, tuple(float(x) for x in v)) for k, v in geo.items()))
    c = Config(**raw)
    if c.experiment_id != EXPERIMENT_ID or c.schema_version != 1:
        raise ConfigError('Expected attention_layer.v1 schema 1')
    if c.model not in ('ideal', 'bsim3') or c.rect_bulk not in ('vdd', 'vrect'):
        raise ConfigError('model ideal|bsim3, rect_bulk vdd|vrect')
    for k in ('scale', 'c_score', 'c_hold', 't_write', 't_reset', 't_settle', 'temperature_g', 'norm_tail', 'eps', 'v_u', 'g_s',
              'law_temperature_g', 'law_cell_s', 'law_cell_kappa', 'law_kappa', 'kappa_design'):
        if not (type(getattr(c, k)) in (int, float) and math.isfinite(getattr(c, k)) and getattr(c, k) > 0):
            raise ConfigError(f'{k} must be positive')
    if not (1 <= c.n_examples <= 400 and 1 <= c.wall_seconds <= 3600 and 0 <= c.seed < 2**32):
        raise ConfigError('n_examples, wall_seconds or seed out of range')
    if not c.weights_source or len(c.weights_source_sha256) != 64:
        raise ConfigError('Trained-weight provenance (source and sha256) is required')
    return c


def resolve_config(config, mode):
    if mode != RunMode.CHARACTERIZE:
        raise ConfigError('This pilot supports characterize only')
    return config
