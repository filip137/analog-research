"""Strict settings for one drift-regimes regime point and its realizations.

Campaign: on-chip-learning/campaigns/drift-regimes (exp-001 instrument
qualification). One lifecycle covers one regime point; each realization is one
trajectory: deployment by optimal-threshold P&V resets, release at the end of
deployment P&V, then teacher KL at the declared ages after release.
"""
from dataclasses import asdict, dataclass, fields
import math

from training.drift_family import ReramFamily

ARMS = ('A0',)          # untreated; maintenance arms are added by exp-002
SPLITS = ('validation',)


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'Invalid {name}.')


def _number(value, name, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid {name}.')
    if positive and value <= 0:
        raise ValueError(f'Expected positive {name}.')
    return float(value)


@dataclass(frozen=True)
class DriftRegimes:
    source_sha256: str
    evaluation_corpus: str
    evaluation_corpus_sha256: str
    cohort_sha256: str
    regime: str
    law: dict
    arm: str = 'A0'
    deployment_attempts: int = 4
    deployment_threshold: float | None = .7051477
    verify_seconds: float = 60.
    assignment_seeds: tuple = (271001, 271002, 271003)
    seed: int = 391103
    evaluation_split: str = 'validation'
    evaluation_sequences: int = 64
    evaluation_seconds: tuple = (0., 60., 900., 1800., 3600., 7200., 14400., 28800., 57600., 86400.)
    canary: bool = False

    def to_dict(self):
        value = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
        value['law'] = dict(self.law)
        return value

    @property
    def family(self):
        return ReramFamily(**self.law)

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or set(raw) != {f.name for f in fields(cls)}:
            raise ValueError('Expected exactly the drift-regimes fields.')
        value = dict(raw)
        for key in ('source_sha256', 'evaluation_corpus_sha256', 'cohort_sha256'):
            _hash(value[key], key)
        for key in ('evaluation_corpus', 'regime'):
            if (not isinstance(value[key], str) or not value[key]
                    or not value[key].replace('-', '').replace('_', '').replace('.', '').isalnum()):
                raise ValueError(f'Invalid {key}.')
        law = value['law']
        if not isinstance(law, dict) or set(law) != {f.name for f in fields(ReramFamily)}:
            raise ValueError('Expected every ReramFamily field in law.')
        ReramFamily(**law)
        value['law'] = dict(law)
        if value['arm'] not in ARMS or value['evaluation_split'] not in SPLITS:
            raise ValueError('Unsupported arm or evaluation split.')
        if type(value['canary']) is not bool:
            raise ValueError('Expected boolean canary.')
        for key in ('deployment_attempts', 'seed', 'evaluation_sequences'):
            if type(value[key]) is not int or not 0 < value[key] < 2**31:
                raise ValueError(f'Invalid positive integer {key}.')
        if (value['deployment_attempts'] == 1) != (value['deployment_threshold'] is None):
            raise ValueError('A deployment threshold is required exactly when attempts > 1.')
        if value['deployment_threshold'] is not None:
            value['deployment_threshold'] = _number(value['deployment_threshold'], 'deployment_threshold')
        value['verify_seconds'] = _number(value['verify_seconds'], 'verify_seconds')
        for key in ('assignment_seeds', 'evaluation_seconds'):
            if not isinstance(value[key], (list, tuple)) or not value[key]:
                raise ValueError(f'Expected nonempty {key}.')
            value[key] = tuple(value[key])
        seeds = value['assignment_seeds']
        if len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 < s < 2**31 for s in seeds):
            raise ValueError('Invalid realization seeds.')
        times = tuple(_number(t, 'evaluation_seconds', positive=False) for t in value['evaluation_seconds'])
        if tuple(sorted(set(times))) != times or times[0] != 0 or times[-1] <= 0:
            raise ValueError('Evaluation ages must increase from zero.')
        value['evaluation_seconds'] = times
        return cls(**value)
