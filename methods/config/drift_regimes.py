"""Strict settings of the exp-003 DRN drift-regimes screen (no numerical imports).

Arms reuse the crossbar exp-002 settings objects verbatim (``label``, ``method`` and
their keys), so each DRN arm can be checked against its crossbar counterpart row.
Only the retained exp-003 methods are accepted; LoRA, oracle and ideal arms are not.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

RERAM_FAMILY_FIELDS = {
    'rho_star': float, 'persistent_share': float, 'cmo_scale': float, 'cmo_acceptance': float,
    'amplitude_sigma': float, 'mean_us': float, 'spread_us': float, 'mean_exponent': float,
    'spread_exponent': float, 'reference_seconds': float, 'program_min': float, 'program_max': float,
    'drift_min': float, 'drift_max': float, 'time_dtype': str,
}
EVALUATION_SECONDS = (0., 60., 900., 1800., 3600., 7200., 14400., 28800., 57600., 86400.)
EVENT_SECONDS = (900., 3600., 14400., 57600.)
COMMON = {'label', 'method'}
ARM_KEYS = {
    'none': (COMMON, {'twin'}),
    'calibration': (COMMON | {'events'}, set()),
    'layernorm': (COMMON | {'events', 'learning_rate'}, set()),
    'rewrite': (COMMON | {'events', 'fraction', 'attempts', 'threshold'}, set()),
    'precompensated': (COMMON | {'events', 'fraction', 'attempts', 'threshold'}, set()),
    'learning_newton': (COMMON | {'events', 'fraction', 'attempts', 'threshold', 'rounds',
                                  'batch_sequences', 'score'}, {'ideal', 'twin'}),
}
SHA = set('0123456789abcdef')


def _fail(message):
    raise ValueError(message)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or set(value) - SHA:
        _fail(f'Expected lowercase SHA-256 {name}.')
    return value


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        _fail(f'Expected integer {name} >= {minimum}.')
    return value


def _float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f'Expected finite numeric {name}.')
    return float(value)


def _exact(raw, keys, name):
    if not isinstance(raw, Mapping) or set(raw) != set(keys):
        _fail(f'Expected {name} fields {sorted(keys)}. Provided value: {raw!r}.')


def _events(value, horizon, name):
    if not isinstance(value, (list, tuple)) or not value:
        _fail(f'Expected nonempty {name}.events.')
    events = tuple(_float(v, f'{name}.events') for v in value)
    if list(events) != sorted(set(events)) or events[0] <= 0 or events[-1] >= horizon:
        _fail(f'Expected increasing positive {name}.events before the horizon.')
    return events


def parse_arm(raw, horizon):
    if not isinstance(raw, Mapping) or raw.get('method') not in ARM_KEYS:
        _fail(f'Expected an arm with a method in {sorted(ARM_KEYS)}. Provided value: {raw!r}.')
    required, optional = ARM_KEYS[raw['method']]
    if not required <= set(raw) <= required | optional:
        _fail(f'Arm {raw.get("label")!r} expects fields {sorted(required)} plus optional {sorted(optional)}.')
    arm = dict(raw)
    name = f'arm {arm["label"]!r}'
    if not isinstance(arm['label'], str) or not arm['label']:
        _fail('Expected a nonempty arm label.')
    if 'twin' in arm and type(arm['twin']) is not bool:
        _fail(f'Expected boolean {name}.twin.')
    if 'events' in arm:
        arm['events'] = _events(arm['events'], horizon, name)
    if 'learning_rate' in arm and not _float(arm['learning_rate'], f'{name}.learning_rate') > 0:
        _fail(f'Expected a positive {name}.learning_rate.')
    if 'fraction' in arm:
        fraction = _float(arm['fraction'], f'{name}.fraction')
        if not 0 < fraction <= 1 or (arm['method'] != 'rewrite' and fraction == 1):
            _fail(f'Expected {name}.fraction in (0, 1]; full refresh belongs only to rewrite.')
        _int(arm['attempts'], f'{name}.attempts', 1)
        if (arm['attempts'] == 1) != (arm['threshold'] is None):
            _fail(f'{name} needs a threshold exactly when attempts > 1.')
        if arm['threshold'] is not None and not _float(arm['threshold'], f'{name}.threshold') > 0:
            _fail(f'Expected a positive {name}.threshold.')
    if arm['method'] == 'learning_newton':
        _int(arm['rounds'], f'{name}.rounds', 1)
        _int(arm['batch_sequences'], f'{name}.batch_sequences', 1)
        if arm['score'] != 'fisher' or arm.get('ideal', False) is not False:
            _fail('exp-003 retains only the non-ideal Fisher-guided rewrite (score "fisher").')
        if arm.get('twin') and arm['attempts'] != 1:
            _fail('A Fisher-rewrite twin writes once without verify (attempts 1).')
    return arm


@dataclass(frozen=True)
class Cohort:
    split: str
    start: int
    sequences: int
    sha256: str


@dataclass(frozen=True)
class Adaptation:
    split: str
    seed: int
    sequences: int
    probe_sequences: int
    steps: int
    batch: int
    microbatch: int
    sha256: str


@dataclass(frozen=True)
class ForwardPolicy:
    """Solver sweeps per forward. Training: settle without a graph, then differentiate through
    ``gradient_sweeps``; evaluation and other no-graph forwards: ``evaluation_sweeps``."""
    name: str
    settle_sweeps: int
    gradient_sweeps: int
    evaluation_sweeps: int


@dataclass(frozen=True)
class DriftRegimes:
    regime: str
    seed: int
    realization: int
    source_sha256: str
    corpus: str
    corpus_sha256: str
    sequence_length: int
    law: Mapping[str, Any]
    deployment_attempts: int
    deployment_threshold: float
    verify_seconds: float
    evaluation_seconds: tuple
    evaluation: Cohort
    evaluation_microbatch: int
    adaptation: Adaptation
    forward_policy: ForwardPolicy
    arms: tuple
    canary: bool = False

    @classmethod
    def parse(cls, raw):
        keys = {f for f in cls.__dataclass_fields__}
        _exact(raw, keys, 'drift_regimes')
        if not isinstance(raw['regime'], str) or not raw['regime']:
            _fail('Expected a regime label.')
        _int(raw['seed'], 'seed')
        _int(raw['realization'], 'realization')
        _sha(raw['source_sha256'], 'source_sha256')
        _sha(raw['corpus_sha256'], 'corpus_sha256')
        if not isinstance(raw['corpus'], str) or not raw['corpus']:
            _fail('Expected a corpus name.')
        _int(raw['sequence_length'], 'sequence_length', 2)
        law = raw['law']
        _exact(law, RERAM_FAMILY_FIELDS, 'law')
        for k, kind in RERAM_FAMILY_FIELDS.items():
            if kind is float:
                _float(law[k], f'law.{k}')
            elif law[k] not in ('float32', 'float64'):
                _fail('Expected law.time_dtype float32/float64.')
        if not 0 < law['rho_star'] <= 1 or not 0 <= law['persistent_share'] <= 1:
            _fail('Expected rho_star in (0, 1] and persistent_share in [0, 1].')
        if not law['drift_min'] <= law['program_min'] < law['program_max'] <= law['drift_max']:
            _fail('The drift window must contain the programming window.')
        _int(raw['deployment_attempts'], 'deployment_attempts', 1)
        threshold = raw['deployment_threshold']
        if raw['deployment_attempts'] > 1 and not _float(threshold, 'deployment_threshold') > 0:
            _fail('Expected a positive deployment threshold.')
        if not _float(raw['verify_seconds'], 'verify_seconds') > 0:
            _fail('Expected a positive verify delay.')
        ages = raw['evaluation_seconds']
        if not isinstance(ages, (list, tuple)) or len(ages) < 2:
            _fail('Expected at least two evaluation ages.')
        ages = tuple(_float(a, 'evaluation_seconds') for a in ages)
        if ages[0] != 0 or list(ages) != sorted(set(ages)):
            _fail('Expected increasing evaluation ages from release (0 s).')
        _exact(raw['evaluation'], Cohort.__dataclass_fields__, 'evaluation')
        evaluation = Cohort(**raw['evaluation'])
        _int(evaluation.start, 'evaluation.start')
        _int(evaluation.sequences, 'evaluation.sequences', 1)
        _sha(evaluation.sha256, 'evaluation.sha256')
        _int(raw['evaluation_microbatch'], 'evaluation_microbatch', 1)
        _exact(raw['adaptation'], Adaptation.__dataclass_fields__, 'adaptation')
        adaptation = Adaptation(**raw['adaptation'])
        for k in ('seed',):
            _int(getattr(adaptation, k), f'adaptation.{k}')
        for k in ('sequences', 'probe_sequences', 'steps', 'batch', 'microbatch'):
            _int(getattr(adaptation, k), f'adaptation.{k}', 1)
        _sha(adaptation.sha256, 'adaptation.sha256')
        if adaptation.probe_sequences > adaptation.sequences or adaptation.batch % adaptation.microbatch:
            _fail('Expected probe <= cohort and a microbatch dividing the effective batch.')
        if evaluation.split == adaptation.split:
            _fail('Adaptation and evaluation cohorts must come from different splits.')
        _exact(raw['forward_policy'], ForwardPolicy.__dataclass_fields__, 'forward_policy')
        policy = ForwardPolicy(**raw['forward_policy'])
        for k in ('settle_sweeps', 'gradient_sweeps', 'evaluation_sweeps'):
            _int(getattr(policy, k), f'forward_policy.{k}', 1)
        if policy.evaluation_sweeps != policy.settle_sweeps + policy.gradient_sweeps:
            _fail('Training and evaluation forwards must run the same number of sweeps.')
        if not isinstance(raw['arms'], (list, tuple)) or not raw['arms']:
            _fail('Expected a nonempty arm list.')
        arms = tuple(parse_arm(a, ages[-1]) for a in raw['arms'])
        if len({a['label'] for a in arms}) != len(arms):
            _fail('Expected distinct arm labels.')
        for a in arms:
            if a['method'] == 'layernorm' or a['method'] == 'learning_newton':
                if a.get('batch_sequences', 1) > adaptation.sequences:
                    _fail('Fisher batch exceeds the adaptation cohort.')
        if type(raw['canary']) is not bool:
            _fail('Expected boolean canary.')
        return cls(**{**raw, 'law': dict(law), 'evaluation_seconds': ages, 'evaluation': evaluation,
                      'adaptation': adaptation, 'forward_policy': policy, 'arms': arms,
                      'verify_seconds': float(raw['verify_seconds']),
                      'deployment_threshold': None if threshold is None else float(threshold)})
