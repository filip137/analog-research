"""Strict settings for one drift-regimes regime point and its realizations.

Campaign: on-chip-learning/campaigns/drift-regimes (exp-001 qualification,
exp-002 regime screen). One lifecycle covers one regime point. Each
realization is deployed once by optimal-threshold P&V resets; release is the
end of deployment P&V. Every arm then starts from a copy of that released
state, applies its maintenance events and is evaluated (teacher KL) at the
declared ages after release. An age that coincides with an event is evaluated
just before the event.

Legacy exp-001 files carry ``arm: "A0"`` instead of ``arms``/``adaptation``;
they parse to the single untreated arm.
"""
from dataclasses import asdict, dataclass, fields
import math

from training.drift_family import ReramFamily

SPLITS = ('validation',)
ADAPTATION_SPLITS = ('train',)
UNTREATED = {'label': 'A0', 'method': 'none'}
# Keys of each arm method besides label and method.
ARM_KEYS = {
    'none': set(),
    'calibration': {'events'},                                   # per-column affine readout fit
    'layernorm': {'events', 'learning_rate'},                    # analog-MLP LayerNorm affine
    'lora': {'events', 'rank', 'learning_rate'},                 # digital LoRA on each analog matrix
    'rewrite': {'events', 'fraction', 'attempts', 'threshold'},  # P&V reset of the largest |e| devices
}
DIGITAL = ('layernorm', 'lora')
ADAPTATION_KEYS = {'corpus', 'corpus_sha256', 'split', 'sequences', 'cohort_sha256', 'seed',
                   'probe_sequences', 'steps', 'batch'}


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'Invalid {name}.')


def _label(value, name):
    if not isinstance(value, str) or not value or not value.replace('-', '').replace('_', '').replace('.', '').isalnum():
        raise ValueError(f'Invalid {name}.')


def _integer(value, name):
    if type(value) is not int or not 0 < value < 2**31:
        raise ValueError(f'Invalid positive integer {name}.')


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
    arms: tuple = (UNTREATED,)
    adaptation: dict | None = None
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
        value['arms'] = [{k: list(v) if isinstance(v, tuple) else v for k, v in arm.items()} for arm in self.arms]
        value['adaptation'] = None if self.adaptation is None else dict(self.adaptation)
        return value

    @property
    def event_ages(self):
        return tuple(sorted({t for arm in self.arms for t in arm.get('events', ())}))

    @property
    def family(self):
        return ReramFamily(**self.law)

    @classmethod
    def parse(cls, raw):
        names = {f.name for f in fields(cls)}
        legacy = (names - {'arms', 'adaptation'}) | {'arm'}
        if not isinstance(raw, dict) or set(raw) not in (names, legacy):
            raise ValueError('Expected exactly the drift-regimes fields.')
        value = dict(raw)
        if 'arm' in value:
            if value.pop('arm') != 'A0':
                raise ValueError('A legacy drift-regimes file has only the untreated arm A0.')
            value['arms'], value['adaptation'] = [dict(UNTREATED)], None
        for key in ('source_sha256', 'evaluation_corpus_sha256', 'cohort_sha256'):
            _hash(value[key], key)
        for key in ('evaluation_corpus', 'regime'):
            _label(value[key], key)
        law = value['law']
        if not isinstance(law, dict) or set(law) != {f.name for f in fields(ReramFamily)}:
            raise ValueError('Expected every ReramFamily field in law.')
        ReramFamily(**law)
        value['law'] = dict(law)
        if value['evaluation_split'] not in SPLITS:
            raise ValueError('Unsupported evaluation split.')
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
        value['arms'] = cls._arms(value['arms'], times[-1])
        value['adaptation'] = cls._adaptation(value['adaptation'], value['arms'])
        return cls(**value)

    @staticmethod
    def _arms(raw, horizon):
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError('Expected a nonempty arm list.')
        arms = []
        for item in raw:
            if not isinstance(item, dict) or item.get('method') not in ARM_KEYS:
                raise ValueError('Unknown drift-regimes arm method.')
            if set(item) != {'label', 'method'} | ARM_KEYS[item['method']]:
                raise ValueError(f"Expected exactly the {item['method']} arm fields.")
            arm = dict(item)
            _label(arm['label'], 'arm label')
            if 'events' in arm:
                if not isinstance(arm['events'], (list, tuple)) or not arm['events']:
                    raise ValueError('Expected nonempty arm events.')
                events = tuple(_number(t, 'event age') for t in arm['events'])
                if tuple(sorted(set(events))) != events or events[-1] >= horizon:
                    raise ValueError('Arm events must increase strictly inside the horizon.')
                arm['events'] = events
            if 'learning_rate' in arm:
                arm['learning_rate'] = _number(arm['learning_rate'], 'learning_rate')
            if 'rank' in arm:
                _integer(arm['rank'], 'rank')
            if arm['method'] == 'rewrite':
                fraction = _number(arm['fraction'], 'fraction')
                if fraction > 1:
                    raise ValueError('Expected a rewrite fraction in (0, 1].')
                arm['fraction'] = fraction
                _integer(arm['attempts'], 'attempts')
                if (arm['attempts'] == 1) != (arm['threshold'] is None):
                    raise ValueError('A rewrite threshold is required exactly when attempts > 1.')
                if arm['threshold'] is not None:
                    arm['threshold'] = _number(arm['threshold'], 'threshold')
            arms.append(arm)
        if len({a['label'] for a in arms}) != len(arms):
            raise ValueError('Arm labels must be unique.')
        return tuple(arms)

    @staticmethod
    def _adaptation(raw, arms):
        needed = any(a['method'] in DIGITAL + ('calibration',) for a in arms)
        if raw is None:
            if needed:
                raise ValueError('Digital and calibration arms require adaptation settings.')
            return None
        if not isinstance(raw, dict) or set(raw) != ADAPTATION_KEYS:
            raise ValueError('Expected exactly the adaptation fields.')
        value = dict(raw)
        _label(value['corpus'], 'adaptation corpus')
        for key in ('corpus_sha256', 'cohort_sha256'):
            _hash(value[key], f'adaptation {key}')
        if value['split'] not in ADAPTATION_SPLITS:
            raise ValueError('Adaptation uses the training split only.')
        for key in ('sequences', 'seed', 'probe_sequences', 'steps', 'batch'):
            _integer(value[key], f'adaptation {key}')
        if value['probe_sequences'] > value['sequences'] or value['batch'] > value['sequences']:
            raise ValueError('Probe and batch sizes must fit in the adaptation cohort.')
        return value
