"""Strict settings for a single initial-programming policy and its realizations."""
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path

POLICIES = tuple(f'P{i}' for i in range(7))


@dataclass(frozen=True)
class InitialProgramming:
    source_sha256: str
    model_path: str
    model_sha256: str
    cohort_sha256: str
    policy: str = 'P0'
    assignment_seeds: tuple = (271001, 271002, 271003)
    seed: int = 391030
    selection_seed: int = 491030
    validation_sequences: int = 64
    evaluation_seconds: tuple = (0., 10., 60., 300., 900., 1800., 3600.)
    initial_anchor_seconds: float = 2.
    verify_seconds: float = 62.
    rewrite_anchor_seconds: float = 64.
    release_seconds: float = 300.
    immediate_release_seconds: float = 5.
    retry_fraction: float = .2
    source_bounds: tuple = (51.6, 154.8)
    destination_bounds: tuple = (9., 88.2)
    boundary: str = 'clip_observation'
    canary: bool = False

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or set(raw) != {f.name for f in fields(cls)}:
            raise ValueError('Expected exactly the initial-programming fields.')
        value = dict(raw)
        for key in ('source_sha256', 'model_sha256', 'cohort_sha256'):
            h = value[key]
            if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError(f'Invalid {key}.')
        if not isinstance(value['model_path'], str) or not Path(value['model_path']).is_absolute():
            raise ValueError('Expected an absolute pinned model_path.')
        if value['policy'] not in POLICIES or value['boundary'] != 'clip_observation':
            raise ValueError('Unsupported programming policy or boundary.')
        if type(value['canary']) is not bool:
            raise ValueError('Expected boolean canary.')
        for key in ('seed', 'selection_seed', 'validation_sequences'):
            if type(value[key]) is not int or not 0 < value[key] < 2**31:
                raise ValueError(f'Invalid positive integer {key}.')
        for key in ('assignment_seeds', 'evaluation_seconds', 'source_bounds', 'destination_bounds'):
            if not isinstance(value[key], (list, tuple)) or not value[key]:
                raise ValueError(f'Expected nonempty {key}.')
            value[key] = tuple(value[key])
        seeds = value['assignment_seeds']
        if len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 < s < 2**31 for s in seeds):
            raise ValueError('Invalid realization seeds.')
        for key in ('evaluation_seconds', 'source_bounds', 'destination_bounds'):
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value[key]):
                raise ValueError(f'Invalid numbers in {key}.')
        for key in ('initial_anchor_seconds', 'verify_seconds', 'rewrite_anchor_seconds',
                    'release_seconds', 'immediate_release_seconds', 'retry_fraction'):
            x = value[key]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
                raise ValueError(f'Invalid {key}.')
        obj = cls(**value)
        if obj.source_bounds != (51.6, 154.8) or obj.destination_bounds != (9., 88.2):
            raise ValueError('This stage preserves the declared Flowers/crossbar coordinates.')
        if not (obj.initial_anchor_seconds + 3 <= obj.immediate_release_seconds < obj.verify_seconds
                < obj.rewrite_anchor_seconds <= obj.release_seconds - 3):
            raise ValueError('Invalid programming/read/release clock order.')
        if not 0 < obj.retry_fraction <= 1:
            raise ValueError('Invalid retry_fraction.')
        times = obj.evaluation_seconds
        if tuple(sorted(set(times))) != times or times[0] != 0 or times[-1] <= 0:
            raise ValueError('Evaluation ages must increase from zero.')
        if obj.release_seconds + times[-1] - obj.initial_anchor_seconds > 6242:
            raise ValueError('Device age exceeds the measured Flowers horizon.')
        return obj

    @property
    def release(self):
        return self.immediate_release_seconds if self.policy == 'P0' else self.release_seconds

    def calendar(self):
        # The same latent transition grid for every policy, including unused reads.
        return tuple(sorted({self.verify_seconds, self.rewrite_anchor_seconds}
                            | {r+t for r in (self.immediate_release_seconds, self.release_seconds)
                               for t in self.evaluation_seconds}))
