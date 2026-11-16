"""Strict settings of the fault-recovery paper LR screen on a saved CMO DRN deployment.

One run reads one canonical fault-aware deployment and runs its declared arms in turn;
every arm restarts from that deployment. The recovery set is the WikiText-2 training
cohort with fixed teacher samples as targets; development is mean next-token KL
against the full FP32 teacher.
"""
from dataclasses import dataclass
import math

METHODS = ('naive_hold', 'hold', 'noisy', 'noiseless', 'child_d', 'lora', 'matched_noiseless', 'calibration')
LEARNING = ('noisy', 'noiseless', 'child_d', 'lora', 'matched_noiseless', 'calibration')


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'Expected a lowercase SHA-256 for {name}.')
    return value


def _positive_int(value, name, zero=False):
    if type(value) is not int or value < 0 or (value == 0 and not zero):
        raise ValueError(f'Expected a {"nonnegative" if zero else "positive"} integer {name}.')
    return value


@dataclass(frozen=True)
class Arm:
    label: str
    method: str
    learning_rate: float | None = None
    rank: int | None = None

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or not {'label', 'method'} <= set(raw) <= set(cls.__dataclass_fields__):
            raise ValueError('Expected arm fields label, method[, learning_rate, rank].')
        label, method = raw['label'], raw['method']
        if not isinstance(label, str) or not label or any(c in label for c in '/ '):
            raise ValueError('Expected a nonempty arm label without spaces or slashes.')
        if method not in METHODS:
            raise ValueError(f'Expected an arm method from {METHODS}.')
        lr, rank = raw.get('learning_rate'), raw.get('rank')
        if method in LEARNING:
            if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
                raise ValueError('Learning arms need a positive finite learning_rate.')
            lr = float(lr)
        elif lr is not None:
            raise ValueError('Hold arms take no learning rate.')
        if method == 'lora':
            _positive_int(rank, 'LoRA rank')
        elif rank is not None:
            raise ValueError('Only LoRA arms take a rank.')
        return cls(label, method, lr, rank)


@dataclass(frozen=True)
class EvaluationCohort:
    name: str
    corpus: str
    corpus_sha256: str
    split: str
    start: int
    sequences: int
    cohort_sha256: str

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError('Expected exactly the evaluation cohort fields.')
        for key in ('name', 'corpus', 'split'):
            if not isinstance(raw[key], str) or not raw[key] or any(c.isspace() or c in '/\\' for c in raw[key]):
                raise ValueError(f'Expected a plain evaluation cohort {key}.')
        _digest(raw['corpus_sha256'], 'evaluation corpus_sha256')
        _digest(raw['cohort_sha256'], 'evaluation cohort_sha256')
        _positive_int(raw['start'], 'evaluation start', zero=True)
        _positive_int(raw['sequences'], 'evaluation sequences')
        return cls(**raw)


@dataclass(frozen=True)
class FaultTuning:
    corpus: str
    corpus_sha256: str
    sequence_length: int
    recovery_split: str
    recovery_seed: int
    recovery_sequences: int
    recovery_sha256: str
    samples_file: str
    samples_sha256: str
    samples_count: int
    samples_seed: int
    development_split: str
    development_start: int
    development_sequences: int
    development_sha256: str
    steps: int
    batch: int
    microbatch: int
    evaluation_microbatch: int
    order_seed: int
    schedule: str
    development_every: int
    child_fraction: float
    fisher_batch: int
    lora_alpha: str
    lora_seed: int
    arms: tuple

    @property
    def evaluation_cohorts(self):
        return ()

    @classmethod
    def parse(cls, raw):
        required = set(FaultTuning.__dataclass_fields__)
        if not isinstance(raw, dict) or not required <= set(raw) <= required | {'evaluation_cohorts'}:
            raise ValueError(f'Expected fault-tuning fields {sorted(required)} and optional evaluation_cohorts.')
        v = dict(raw)
        for name in ('corpus_sha256', 'recovery_sha256', 'samples_sha256', 'development_sha256'):
            _digest(v[name], name)
        for name in ('corpus', 'samples_file', 'recovery_split', 'development_split'):
            if not isinstance(v[name], str) or not v[name] or '/' in v[name]:
                raise ValueError(f'Expected a plain name for {name}.')
        for name in ('sequence_length', 'recovery_sequences', 'samples_count', 'development_sequences', 'steps',
                     'batch', 'microbatch', 'evaluation_microbatch', 'fisher_batch'):
            _positive_int(v[name], name)
        for name in ('recovery_seed', 'samples_seed', 'development_start', 'order_seed', 'development_every',
                     'lora_seed'):
            _positive_int(v[name], name, zero=True)
        if v['batch'] % v['microbatch']:
            raise ValueError('Expected the microbatch to divide the batch.')
        if v['schedule'] != 'cosine_to_zero':
            raise ValueError('Expected the declared cosine_to_zero schedule (no warm-up).')
        if v['lora_alpha'] != 'rank':
            raise ValueError('Expected LoRA alpha = rank.')
        fraction = v['child_fraction']
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
            raise ValueError('Expected a Child-Tuning reserve fraction in (0, 1).')
        v['child_fraction'] = float(fraction)
        arms = v['arms']
        if not isinstance(arms, (list, tuple)) or not arms:
            raise ValueError('Expected a nonempty arm list.')
        v['arms'] = tuple(a if isinstance(a, Arm) else Arm.parse(a) for a in arms)
        labels = [a.label for a in v['arms']]
        if len(set(labels)) != len(labels):
            raise ValueError('Expected unique arm labels.')
        if 'evaluation_cohorts' in v:
            entries = v['evaluation_cohorts']
            if not isinstance(entries, list):
                raise ValueError('Expected an evaluation_cohorts list.')
            v['evaluation_cohorts'] = tuple(EvaluationCohort.parse(x) for x in entries)
            if len({x.name for x in v['evaluation_cohorts']}) != len(entries):
                raise ValueError('Expected unique evaluation cohort names.')
            return EvaluatedFaultTuning(**v)
        return cls(**v)


@dataclass(frozen=True)
class EvaluatedFaultTuning(FaultTuning):
    # Keep legacy asdict/config identities unchanged when the optional field is absent.
    evaluation_cohorts: tuple = ()
