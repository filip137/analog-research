"""Frozen mixed-rate corruption-aware CIFAR experiment contract."""
from dataclasses import dataclass, asdict, fields
import math
from experiments.schema import RunMode
from experiments.cifar_crossbar.fault_config import METHODS

EXPERIMENT_ID = 'cifar_crossbar_fault_sweep.v1'
KINDS = ('open', 'gmax', 'random')
RATES = (0, 10000, 20000, 30000, 50000)
LRS = (0.00003, 0.0001, 0.0003)
STRENGTHS = (1., 2., 5.)
SOURCES = ('digital', 'standard_hwa', 'noise_hwa', 'cdt_open', 'cdt_gmax', 'cdt_random')

@dataclass(frozen=True)
class SweepSpec:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    stage: str = 'screen'
    dataset: str = 'cifar10'
    suffix: str = 'last_two_blocks'
    backend: str = 'pcm'
    tile_size: int = 512
    device: str = 'cpu'
    disable_cudnn: bool = False
    cpu_threads: int = 1
    seed: int = 41
    data_seed: int = 20260917
    array_seed: int = 251001
    endpoint_seed: int = 261001
    fault_kinds: tuple = KINDS
    fault_rates_ppm: tuple = RATES
    methods: tuple = METHODS
    epochs: int = 5
    recovery_images: int = 45000
    validation_images: int = 1000
    batch_size: int = 64
    learning_rate: float = .0001
    calibration_lr: float = .0003
    program_pulses: int = 128
    recovery_pulses: int = 640
    update_pulses: int = 1
    verify_step_fraction: float = .5
    kernel_bins: int = 41
    kernel_samples: int = 512
    hwa_epochs: int = 30
    extension_epochs: int = 30
    hwa_batch_size: int = 256
    validation_every: int = 5
    candidate_family: str = 'generic'
    candidate_learning_rate: float = .0001
    candidate_noise: float = 1.
    max_examples: int = 0


def parse_config(payload):
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(SweepSpec)}:
        raise ValueError('Expected exactly the fault-sweep v1 fields.')
    v = dict(payload)
    for k in ('fault_kinds','fault_rates_ppm','methods'):
        if not isinstance(v[k], (list, tuple)): raise ValueError(k)
        v[k] = tuple(v[k])
    s = SweepSpec(**v)
    for f in fields(s):
        x, d = getattr(s, f.name), getattr(SweepSpec(), f.name)
        if type(d) is int and (type(x) is not int or x < 0): raise ValueError(f.name)
        if type(d) is bool and type(x) is not bool: raise ValueError(f.name)
        if type(d) is float and (type(x) not in (int,float) or not math.isfinite(x)): raise ValueError(f.name)
    if s.experiment_id != EXPERIMENT_ID or s.schema_version != 1: raise ValueError('Wrong schema.')
    if s.stage not in ('cache','kernel','fit','select_generic','sources','screen'): raise ValueError('stage')
    if s.dataset not in ('cifar10','cifar100') or s.backend not in ('pcm','om'): raise ValueError('dataset/backend')
    if s.suffix not in ('last_two_blocks','last_four_blocks'): raise ValueError('suffix')
    if s.candidate_family not in ('generic', *KINDS): raise ValueError('candidate family')
    if s.candidate_learning_rate not in LRS or s.candidate_noise not in STRENGTHS: raise ValueError('Candidate grid.')
    frozen = dict(tile_size=512, epochs=5, recovery_images=45000, validation_images=1000,
                  batch_size=64, learning_rate=.0001, program_pulses=128, recovery_pulses=640,
                  update_pulses=1, verify_step_fraction=.5, kernel_bins=41, kernel_samples=512,
                  hwa_batch_size=256, data_seed=20260917, seed=41)
    if any(getattr(s,k)!=v for k,v in frozen.items()): raise ValueError('Frozen scientific budget changed.')
    if s.calibration_lr != (.0003 if s.dataset=='cifar10' else .001): raise ValueError('Calibration rate.')
    if s.fault_kinds != KINDS or s.methods != METHODS: raise ValueError('Incomplete controls.')
    if s.max_examples not in (0,128): raise ValueError('Only the labelled 128-image canary is supported.')
    if (s.hwa_epochs,s.extension_epochs,s.validation_every) != ((1,0,1) if s.max_examples else (30,30,5)):
        raise ValueError('HWA budget.')
    if s.fault_rates_ppm != ((0,50000) if s.max_examples else RATES): raise ValueError('Fault rate coverage.')
    if s.array_seed not in ((271001,) if s.max_examples else (251001,251002,251003)) or s.endpoint_seed != s.array_seed+10000:
        raise ValueError('Final array namespace.')
    if s.cpu_threads < 1 or not (s.device=='cpu' or s.device=='cuda' or s.device.startswith('cuda:')): raise ValueError('Execution settings.')
    return s


def default_config(**changes):
    return asdict(parse_config({**asdict(SweepSpec()), **changes}))


def resolve_spec(document, mode):
    if mode != RunMode.TRAIN: raise ValueError('Expected native train mode.')
    return document


def cases(spec, source=None, *, selection=False):
    kinds = (source[4:],) if source and source.startswith('cdt_') else KINDS
    rates = (20000,30000,50000) if selection else tuple(p for p in spec.fault_rates_ppm if p)
    return [('open',0)]+[(k,p) for k in kinds for p in rates]
