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
REWRITE = {'events', 'fraction', 'attempts', 'threshold'}
LORA = {'events', 'rank', 'learning_rate'}
ARM_KEYS = {
    'none': set(),
    'calibration': {'events'},                    # per-column affine readout fit (A1)
    'layernorm': {'events', 'learning_rate'},     # analog-MLP LayerNorm affine (A2, A3)
    'lora': LORA,                                 # digital LoRA on each analog matrix (A4, A5, A7)
    'lora_oracle': LORA,                          # fresh LoRA trained at each evaluation age, evaluated after it (O1)
    'lora_rescaled': LORA | {'reads'},            # A4 rescaled in time with beta, kappa estimated from reads (A6)
    'lora_clean': LORA,                           # LoRA trained on the clean target weights (no-drift diagnostic)
    'rewrite': REWRITE,                           # P&V reset of the largest observed |e| devices (A8, A9)
    'precompensated': REWRITE,                    # A9 with history-regressed target offsets (A10)
    'rewrite_oracle': REWRITE,                    # reset of the largest hidden future |e| devices (O2)
    'learning': REWRITE | {'step', 'rounds', 'batch_sequences', 'ideal'},  # gradient-step writes (diagnostic)
    'learning_newton': REWRITE | {'rounds', 'batch_sequences', 'ideal'},   # exp-003 closed loop (A11)
}
# Optional on these methods: twin = True holds the released state (no drift) for reads and evaluation.
TWIN = ('none', 'lora', 'learning', 'learning_newton')
DIGITAL = ('layernorm', 'lora', 'lora_oracle', 'lora_rescaled', 'lora_clean', 'learning', 'learning_newton')
WRITES = ('rewrite', 'precompensated', 'rewrite_oracle', 'learning', 'learning_newton')
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
    evaluation_start: int = 0
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
        # evaluation_start (the cohort's first sequence) is optional and defaults to 0.
        keys = set(raw) | {'evaluation_start'} if isinstance(raw, dict) else None
        if keys not in (names, legacy):
            raise ValueError('Expected exactly the drift-regimes fields.')
        value = {'evaluation_start': 0, **raw}
        if type(value['evaluation_start']) is not int or not 0 <= value['evaluation_start'] < 2**31:
            raise ValueError('Invalid evaluation_start.')
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
        value['arms'] = cls._arms(value['arms'], times)
        value['adaptation'] = cls._adaptation(value['adaptation'], value['arms'])
        return cls(**value)

    @staticmethod
    def _arms(raw, ages):
        horizon = ages[-1]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError('Expected a nonempty arm list.')
        arms = []
        for item in raw:
            if not isinstance(item, dict) or item.get('method') not in ARM_KEYS:
                raise ValueError('Unknown drift-regimes arm method.')
            keys = {'label', 'method'} | ARM_KEYS[item['method']]
            optional = {'twin'} | ({'score'} if item['method'] == 'learning_newton' else set())
            if (not keys <= set(item) <= keys | optional
                    or ('twin' in item and item['method'] not in TWIN)):
                raise ValueError(f"Expected exactly the {item['method']} arm fields.")
            if item.get('score', 'newton') not in ('newton', 'fisher'):
                raise ValueError('Expected a newton or fisher learning score.')
            arm = dict(item)
            _label(arm['label'], 'arm label')
            if 'twin' in arm and type(arm['twin']) is not bool:
                raise ValueError('Expected boolean twin.')
            for key in ('events', 'reads'):
                if key not in arm:
                    continue
                if not isinstance(arm[key], (list, tuple)) or not arm[key]:
                    raise ValueError(f'Expected nonempty arm {key}.')
                oracle = arm['method'] == 'lora_oracle'
                times = tuple(_number(t, 'event age', positive=not oracle) for t in arm[key])
                if tuple(sorted(set(times))) != times:
                    raise ValueError(f'Arm {key} must increase strictly.')
                if oracle and not set(times) <= set(ages):
                    raise ValueError('Oracle events are evaluation ages.')
                if not oracle and times[-1] >= horizon:
                    raise ValueError(f'Arm {key} must lie strictly inside the horizon.')
                arm[key] = times
            if 'reads' in arm and (set(arm['reads']) & set(arm['events']) or arm['reads'][0] <= arm['events'][0]):
                raise ValueError('Rescaling reads follow the training event and differ from it.')
            if 'learning_rate' in arm:
                arm['learning_rate'] = _number(arm['learning_rate'], 'learning_rate')
            if 'rank' in arm:
                _integer(arm['rank'], 'rank')
            if arm['method'] in ('learning', 'learning_newton'):
                if 'step' in arm:
                    arm['step'] = _number(arm['step'], 'step')
                _integer(arm['rounds'], 'rounds')
                _integer(arm['batch_sequences'], 'batch_sequences')
                if type(arm['ideal']) is not bool:
                    raise ValueError('Expected boolean ideal.')
                if arm['fraction'] >= 1:
                    raise ValueError('Analog learning writes a fraction of the devices.')
                if arm.get('twin') and (arm['attempts'] != 1 or arm['ideal']):
                    raise ValueError('A no-drift learning twin uses single unverified writes.')
            if arm['method'] in WRITES:
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
