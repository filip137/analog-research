"""On-chip learning of chain A (#62): attention_energy circuit plus a SPICE-in-the-loop training
schedule driven by EP in the bank and a local backward rule."""
from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode
from ebl.schema import attention_energy as energy

EXPERIMENT_ID = 'onchip_attention.v1'
EXTRA = {'init': 'random', 'init_seed': 0, 'steps': 60, 'batch': 12, 'lr': .01, 'beta': .1, 't_settle_train': .15,
         't_phase': .05, 'eval_every': 20, 'n_test': 96, 'rule_cell_s': 75., 'rule_cell_kappa': .79, 'kappa_value': .7911,
         'train_seed': 777, 'mirror_bias': 1e-9}


@dataclass(frozen=True)
class Config(energy.Config):
    experiment_id: str = EXPERIMENT_ID
    init: str = 'random'
    init_seed: int = 0
    steps: int = 60
    batch: int = 12
    lr: float = .01
    beta: float = .1
    t_settle_train: float = .15
    t_phase: float = .05
    eval_every: int = 20
    n_test: int = 96
    rule_cell_s: float = 75.
    rule_cell_kappa: float = .79
    kappa_value: float = .7911
    train_seed: int = 777
    mirror_bias: float = 1e-9


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Unknown on-chip attention configuration field')
    if raw.get('experiment_id') != EXPERIMENT_ID or raw.get('schema_version') != 1:
        raise ConfigError('Expected onchip_attention.v1 schema 1')
    extra = {k: raw[k] for k in EXTRA if k in raw}
    c0 = energy.parse_config({k: v for k, v in raw.items() if k not in EXTRA} | {'experiment_id': energy.EXPERIMENT_ID})
    if c0.chain != 'A':
        raise ConfigError('On-chip learning uses chain A (the EP bank)')
    for k, lo, hi in (('steps', 1, 2000), ('batch', 1, 64), ('eval_every', 1, 2000), ('n_test', 4, 400), ('init_seed', 0, 2**31),
                      ('train_seed', 0, 2**31)):
        if type(extra.get(k, EXTRA[k])) is not int or not lo <= extra.get(k, EXTRA[k]) <= hi:
            raise ConfigError(f'{k} out of range')
    for k, lo, hi in (('lr', 1e-5, 1), ('beta', 1e-4, 1), ('t_settle_train', 1e-3, 1), ('t_phase', 1e-3, 1), ('rule_cell_s', 1, 1e3),
                      ('rule_cell_kappa', .1, 1.5), ('kappa_value', .1, 1.5), ('mirror_bias', 0, 1e-8)):
        if type(extra.get(k, EXTRA[k])) not in (int, float) or not lo <= extra.get(k, EXTRA[k]) <= hi:
            raise ConfigError(f'{k} out of range')
    if not isinstance(extra.get('init', 'random'), str):
        raise ConfigError('init must be "random" or a weights-file description')
    vals = {f.name: getattr(c0, f.name) for f in fields(energy.Config)} | {'experiment_id': EXPERIMENT_ID}
    return Config(**vals, **extra)


def resolve_config(config, mode):
    if mode != RunMode.CHARACTERIZE:
        raise ConfigError('This pilot supports characterize only')
    return config
