"""Strict exp-003 drift-regimes configuration for the frozen OPT CMO DRN.

A separate experiment id keeps the ``opt_cmo_drn_hwa.v1`` digests and checkpoint
contracts unchanged. The solver section must equal the source's saved contract
(T=4, K=4, A=5, B=0.1); the screen's forward policy is versioned separately.
"""
from dataclasses import asdict, dataclass
from typing import Mapping

from ebl.schema.base import ConfigError, RunMode
from ebl.schema.opt_cmo_drn import SOURCE_SHA256, TEACHER_SHA256, _parse
from methods.config.drift_regimes import DriftRegimes
from networks.config.opt import Solver

EXPERIMENT_ID = 'opt_cmo_drn_drift_regimes.v1'


@dataclass(frozen=True)
class Config:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    stage: str = 'drift_regimes'
    model: str = 'facebook/opt-125m'
    analog_layers: int = 6
    initialization: str = 'full_span_quad.v1'
    device: str = 'cpu'
    cpu_threads: int = 2
    teacher_sha256: str = TEACHER_SHA256
    source_sha256: str = SOURCE_SHA256
    solver: Solver = Solver()
    drift_regimes: object = None


def parse_config(raw):
    if not isinstance(raw, Mapping):
        raise ConfigError('Expected a configuration object.')
    raw = dict(raw)
    settings = raw.pop('drift_regimes', None)
    if settings is None:
        raise ConfigError('Expected explicit drift_regimes settings.')
    try:
        settings = DriftRegimes.parse(settings)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    known = {k for k in Config.__dataclass_fields__ if k != 'drift_regimes'}
    if set(raw) - known:
        raise ConfigError(f'Expected known config fields {sorted(known)}. Provided value: {sorted(set(raw) - known)}.')
    solver = _parse(Solver, raw.pop('solver', {}), 'config.solver')
    c = Config(**raw, solver=solver, drift_regimes=settings)
    if c.experiment_id != EXPERIMENT_ID or c.schema_version != 1 or c.stage not in ('drift_regimes', 'readiness'):
        raise ConfigError('Expected opt_cmo_drn_drift_regimes.v1 schema 1, stage drift_regimes or readiness.')
    if (c.model, c.analog_layers, c.initialization) != ('facebook/opt-125m', 6, 'full_span_quad.v1'):
        raise ConfigError('Expected OPT-125M with six full-span four-device DRN MLPs.')
    if (c.teacher_sha256, c.source_sha256) != (TEACHER_SHA256, SOURCE_SHA256):
        raise ConfigError('Expected the pinned FP32 teacher and crossbar std3 mapping origin.')
    if c.solver != Solver():
        raise ConfigError('Expected the saved T=4, K=4, A=5, B=0.1, gain-100 solver contract.')
    policy = settings.forward_policy
    if (policy.settle_sweeps, policy.gradient_sweeps) != (c.solver.settling_iterations, c.solver.gradient_iterations):
        raise ConfigError('The forward policy must train with the saved settle/gradient sweeps.')
    if type(c.cpu_threads) is not int or c.cpu_threads <= 0:
        raise ConfigError('Expected positive cpu_threads.')
    return c


def resolve_config(config, mode):
    if mode != RunMode.TRAIN:
        raise ConfigError('The drift-regimes screen supports train mode only.')
    return config


def plain(config):
    return asdict(config)
