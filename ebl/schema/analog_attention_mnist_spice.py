"""SPICE inference samples of the MNIST chain A network (#63)."""
from dataclasses import dataclass, fields
from ebl.schema.base import ConfigError, RunMode

EXPERIMENT_ID = 'analog_attention_mnist_spice.v1'


@dataclass(frozen=True)
class Config:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    variant: str = ''
    model_run: str = ''
    model_export_sha256: str = ''
    first_image: int = 10
    n_images: int = 20
    amp: float = .66
    score_offset_extra: float = 0.
    offset_quantile: float = .1
    calibration_images: int = 1000
    v_cm: float = 1.05
    scale: float = .5
    s_ratio: float = 16.
    tail_current: float = 1e-9
    amplitude: float = .57
    v_low: float = .6
    c_score: float = 1e-12
    c_hold: float = 1e-12
    t_write: float = 2e-6
    t_reset: float = 2e-4
    t_warm: float = 2e-4
    t_settle: float = .04
    mirror_bias: float = 1e-9
    abstol: float = 3e-15
    reltol: float = 1e-4
    kappa_design: float = .7911
    workers: int = 10
    data_root: str = '/home/filip/datasets/mnist'
    wall_seconds: int = 3600
    spice_executable: str = '/tmp/pseudoconductance-spice-env/bin/ngspice'


def parse_config(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise ConfigError('Unknown MNIST SPICE field')
    c = Config(**raw)
    if c.experiment_id != EXPERIMENT_ID or c.schema_version != 1:
        raise ConfigError('Expected analog_attention_mnist_spice.v1 schema 1')
    if not c.model_run or len(c.model_export_sha256) != 64 or not c.variant:
        raise ConfigError('model_run, model_export_sha256 and variant are required')
    if not (0 <= c.first_image < 10000 and 1 <= c.n_images <= 100 and 0 < c.amp <= 2 and 0 <= c.offset_quantile <= 1
            and 1 <= c.workers <= 16 and 1 <= c.wall_seconds <= 7200 and 1e-7 <= c.reltol <= 1e-2):
        raise ConfigError('Out-of-range field')
    return c


def resolve_config(c, mode):
    if mode != RunMode.CHARACTERIZE:
        raise ConfigError('characterize only')
    return c
