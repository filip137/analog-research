"""Size ladder of the end-to-end analog attention step (#60): attention_energy circuit plus
per-family geometry, Pelgrom mismatch and kT/C Monte Carlo."""
from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode
from ebl.schema import attention_energy as energy

EXPERIMENT_ID = 'attention_size.v1'
FAMILIES = ('cell', 'cell_mid', 'mirror', 'softmax', 'value', 'bank_read', 'bank_input', 'bank', 'translinear', 'switch')
EXTRA = {'geometry': {}, 'avt': 7e-3, 'abeta': 0.01, 'mc_instances': 20, 'mc_seed': 1000, 'avt_sensitivity': 14e-3,
         'sensitivity_instances': 10, 'cal_examples': 12, 'yield_target': 0.9}


@dataclass(frozen=True)
class Config(energy.Config):
    experiment_id: str = EXPERIMENT_ID
    geometry: tuple = ()
    avt: float = 7e-3
    abeta: float = 0.01
    mc_instances: int = 20
    mc_seed: int = 1000
    avt_sensitivity: float = 14e-3
    sensitivity_instances: int = 10
    cal_examples: int = 12
    yield_target: float = 0.9


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Unknown attention-size configuration field')
    if raw.get('experiment_id') != EXPERIMENT_ID or raw.get('schema_version') != 1:
        raise ConfigError('Expected attention_size.v1 schema 1')
    extra = {k: raw[k] for k in EXTRA if k in raw}
    c0 = energy.parse_config({k: v for k, v in raw.items() if k not in EXTRA} | {'experiment_id': energy.EXPERIMENT_ID})
    geo = extra.get('geometry', {})
    if not isinstance(geo, dict) or set(geo) - set(FAMILIES) or not all(
            isinstance(v, list) and len(v) == 2 and all(type(x) in (int, float) and .02 <= x <= 10 for x in v) for v in geo.values()):
        raise ConfigError('geometry maps device families to [width_factor, length_factor] in [0.02, 10]')
    for k, lo, hi in (('avt', 0, 50e-3), ('abeta', 0, .1), ('avt_sensitivity', 0, 50e-3), ('yield_target', 0, 1)):
        if type(extra.get(k, EXTRA[k])) not in (int, float) or not lo <= extra.get(k, EXTRA[k]) <= hi:
            raise ConfigError(f'{k} out of range')
    for k, lo, hi in (('mc_instances', 0, 100), ('mc_seed', 0, 2**31), ('sensitivity_instances', 0, 100), ('cal_examples', 4, 36)):
        if type(extra.get(k, EXTRA[k])) is not int or not lo <= extra.get(k, EXTRA[k]) <= hi:
            raise ConfigError(f'{k} out of range')
    vals = {f.name: getattr(c0, f.name) for f in fields(energy.Config)} | {'experiment_id': EXPERIMENT_ID}
    extra['geometry'] = tuple(sorted((k, tuple(float(x) for x in v)) for k, v in geo.items()))
    return Config(**vals, **extra)


def resolve_config(config, mode):
    if mode != RunMode.CHARACTERIZE:
        raise ConfigError('This pilot supports characterize only')
    return config
