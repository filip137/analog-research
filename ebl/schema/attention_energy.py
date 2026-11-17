"""Energy ladder of the end-to-end analog attention step (#59): the end_to_end_attention
circuit with power accounting and declared power-saving options."""
from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode
from ebl.schema import end_to_end_attention as base

EXPERIMENT_ID = 'attention_energy.v1'
EXTRA = {'rung': '', 'duty_cycle': False, 'query_hold': False, 'ref_hold': False, 'gate_scorer': False,
         't_warm': 0., 'c_state': 1e-12, 'abstol': 1e-15, 'settle_reference_s': 0.}


@dataclass(frozen=True)
class Config(base.Config):
    experiment_id: str = EXPERIMENT_ID
    rung: str = ''
    duty_cycle: bool = False
    query_hold: bool = False
    ref_hold: bool = False
    gate_scorer: bool = False
    t_warm: float = 0.
    c_state: float = 1e-12
    abstol: float = 1e-15
    settle_reference_s: float = 0.


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Unknown attention-energy configuration field')
    if raw.get('experiment_id') != EXPERIMENT_ID or raw.get('schema_version') != 1:
        raise ConfigError('Expected attention_energy.v1 schema 1')
    extra = {k: raw[k] for k in EXTRA if k in raw}
    c0 = base.parse_config({k: v for k, v in raw.items() if k not in EXTRA} | {'experiment_id': base.EXPERIMENT_ID})
    for k in ('duty_cycle', 'query_hold', 'ref_hold', 'gate_scorer'):
        if type(extra.get(k, False)) is not bool:
            raise ConfigError(f'{k} must be boolean')
    if not isinstance(extra.get('rung', ''), str) or not extra.get('rung'):
        raise ConfigError('rung label required')
    for k, lo, hi in (('t_warm', 0, 1e-3), ('c_state', 0, 1e-11), ('abstol', 1e-18, 1e-13), ('settle_reference_s', 0, 1)):
        v = extra.get(k, EXTRA[k])
        if type(v) not in (int, float) or not lo <= v <= hi:
            raise ConfigError(f'{k} out of range')
    if extra.get('duty_cycle') and not extra.get('query_hold'):
        raise ConfigError('duty_cycle requires query_hold (the live query would follow V_cm)')
    if extra.get('ref_hold') and not extra.get('query_hold'):
        raise ConfigError('ref_hold requires query_hold')
    if extra.get('gate_scorer') and not extra.get('t_warm'):
        raise ConfigError('gate_scorer requires a warm-up t_warm')
    vals = {f.name: getattr(c0, f.name) for f in fields(base.Config)} | {'experiment_id': EXPERIMENT_ID}
    return Config(**vals, **{k: (float(v) if k not in ('rung',) and type(v) is not bool else v) for k, v in extra.items()})


def resolve_config(config, mode):
    if mode != RunMode.CHARACTERIZE:
        raise ConfigError('This pilot supports characterize only')
    return config
