"""Development-selected OM closed-loop learning rates, with no total pulse cap."""
from dataclasses import asdict, dataclass, fields
import math
from experiments.schema import RunMode

EXPERIMENT_ID = 'cifar_om_closed_loop_lr.v1'
LRS = (.0001, .0003, .001)
SOURCES = ('digital', 'standard_hwa')
CASES = (('open', 0), ('gmax', 30000), ('gmax', 50000))
DEV_SEEDS = (211001, 211002)
FINAL_SEEDS = (251001, 251002, 251003)


@dataclass(frozen=True)
class ClosedLoopLRSpec:
    experiment_id: str = EXPERIMENT_ID
    schema_version: int = 1
    stage: str = 'screen'
    phase: str = 'development'
    dataset: str = 'cifar10'
    suffix: str = 'last_two_blocks'
    backend: str = 'om'
    tile_size: int = 512
    device: str = 'cpu'
    disable_cudnn: bool = False
    cpu_threads: int = 1
    seed: int = 41
    data_seed: int = 20260917
    array_seed: int = 211001
    endpoint_seed: int = 221001
    sources: tuple = SOURCES
    fault_cases: tuple = CASES
    learning_rates: tuple = LRS
    epochs: int = 5
    recovery_images: int = 45000
    validation_images: int = 1000
    batch_size: int = 64
    calibration_lr: float = .0003
    program_pulses: int = 128
    recovery_pulses: None = None
    update_pulses: int = 1
    verify_step_fraction: float = .5
    max_examples: int = 0


def parse_config(payload):
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(ClosedLoopLRSpec)}:
        raise ValueError('Expected exactly the closed-loop LR v1 fields.')
    values = dict(payload)
    for name in ('sources', 'fault_cases', 'learning_rates'):
        if not isinstance(values[name], (tuple, list)): raise ValueError(name)
        values[name] = tuple(tuple(v) for v in values[name]) if name == 'fault_cases' else tuple(values[name])
    spec, default = ClosedLoopLRSpec(**values), ClosedLoopLRSpec()
    variable = {'phase', 'dataset', 'suffix', 'device', 'disable_cudnn', 'cpu_threads',
                'array_seed', 'endpoint_seed', 'calibration_lr', 'max_examples'}
    for field in fields(spec):
        value, expected = getattr(spec, field.name), getattr(default, field.name)
        if type(expected) is int and (type(value) is not int or value < 0): raise ValueError(field.name)
        if type(expected) is bool and type(value) is not bool: raise ValueError(field.name)
        if type(expected) is float and (type(value) not in (int, float) or not math.isfinite(value)): raise ValueError(field.name)
        if field.name not in variable and value != expected: raise ValueError('Frozen field: ' + field.name)
    if spec.phase not in ('development', 'confirmation'): raise ValueError('Phase.')
    if spec.dataset not in ('cifar10', 'cifar100') or spec.suffix not in ('last_two_blocks', 'last_four_blocks'):
        raise ValueError('Dataset/suffix.')
    if spec.calibration_lr != (.0003 if spec.dataset == 'cifar10' else .001): raise ValueError('Calibration LR.')
    if spec.max_examples not in (0, 128): raise ValueError('Canary/full cohort.')
    seeds = (271001,) if spec.max_examples else DEV_SEEDS if spec.phase == 'development' else FINAL_SEEDS
    if spec.array_seed not in seeds or spec.endpoint_seed != spec.array_seed + 10000: raise ValueError('Array partition.')
    if spec.max_examples and spec.phase != 'development': raise ValueError('Canaries cannot confirm a selected LR.')
    if spec.cpu_threads < 1 or not (spec.device in ('cpu', 'cuda') or spec.device.startswith('cuda:')):
        raise ValueError('Execution settings.')
    return spec


def default_config(**changes):
    return asdict(parse_config({**asdict(ClosedLoopLRSpec()), **changes}))


def resolve_spec(document, mode):
    if mode != RunMode.TRAIN: raise ValueError('Expected native train mode.')
    return document
