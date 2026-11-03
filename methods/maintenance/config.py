"""Strict scientific settings shared by the two native recovery stages."""
from dataclasses import asdict, dataclass, fields
import math
from hardware.drift.config import DriftLaw

ARMS = ('untreated', 'global_calibration', 'channel_calibration', 'full_refresh',
        'worst_refresh', 'random_refresh', 'stable_learning', 'random_learning', 'oracle_learning')


@dataclass(frozen=True)
class DriftRecovery:
    source_sha256: str
    law: DriftLaw = DriftLaw()
    persistent_fractions: tuple = (0., .9, 1.)
    assignment_seeds: tuple = (271002,)
    seed: int = 391029
    selection_seed: int = 491029
    selection_fraction: float = .1
    arms: tuple = ARMS
    protocols: tuple = ('single', 'repeated')
    write_budgets: tuple = (1., 10.)
    first_intervention_seconds: float = 3600.
    maintenance_interval_seconds: float = 604800.
    horizon_seconds: float = 2592000.
    followup_seconds: tuple = (60., 3600., 86400.)
    evaluation_seconds: tuple = (0., 3600., 7200., 21600., 86400., 90000., 172800.,
                                 604800., 1209600., 1814400., 2419200., 2592000.)
    training_sequences: int = 1024
    validation_sequences: int = 256
    calibration_sequences: int = 32
    batch_size: int = 32
    microbatch_size: int = 1
    max_updates: int = 128
    passes: int = 4
    learning_rates: tuple = (.0001, .0003, .001, .003)
    learning_rate: float | None = None
    no_drift_control: bool = True
    canary: bool = False
    curve_validation_sequences: int | None = None
    heldout_evaluation: str = 'all'
    execution: dict | None = None

    def to_dict(self):
        value = asdict(self)
        for key, default in (('curve_validation_sequences', None), ('heldout_evaluation', 'all'), ('execution', None)):
            if value[key] == default: value.pop(key)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in value.items()}

    def science_dict(self):
        value = self.to_dict()
        value.pop('execution', None)
        return value

    @classmethod
    def parse(cls, raw):
        optional = {'curve_validation_sequences', 'heldout_evaluation', 'execution'}
        required = {f.name for f in fields(cls)} - optional
        if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - optional:
            raise ValueError('Expected exactly the custom drift-recovery fields.')
        values = dict(raw)
        values = {'curve_validation_sequences': None, 'heldout_evaluation': 'all', 'execution': None, **values}
        if not isinstance(values['law'], dict) or set(values['law']) != set(DriftLaw.__dataclass_fields__):
            raise ValueError('Expected explicit custom drift law fields.')
        values['law'] = DriftLaw(**values['law'])
        digest = values['source_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Expected frozen HWA SHA-256.')
        defaults = cls(digest)
        for field in fields(cls):
            key, value = field.name, values[field.name]
            default = getattr(defaults, key)
            if isinstance(default, tuple):
                if not isinstance(value, (list, tuple)) or not value or len(set(value)) != len(value):
                    raise ValueError(f'Expected distinct nonempty {key}.')
                values[key] = tuple(value)
            elif isinstance(default, bool):
                if type(value) is not bool: raise ValueError(f'Expected boolean {key}.')
            elif type(default) is int:
                if type(value) is not int or value <= 0: raise ValueError(f'Expected positive integer {key}.')
            elif isinstance(default, float) or key == 'learning_rate':
                if key == 'learning_rate' and value is None: continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'Expected positive finite {key}.')
                values[key] = float(value)
        obj = cls(**values)
        if obj.curve_validation_sequences is not None and (type(obj.curve_validation_sequences) is not int
                or not 0 < obj.curve_validation_sequences <= obj.validation_sequences):
            raise ValueError('Curve validation must be a positive prefix of the selection cohort.')
        if obj.heldout_evaluation not in ('all', 'final'):
            raise ValueError('Unknown held-out evaluation schedule.')
        if obj.execution is not None:
            e = obj.execution
            if not isinstance(e, dict) or set(e) != {'phase', 'key', 'prepared_run', 'screen_runs'}:
                raise ValueError('Expected explicit recovery execution fields.')
            if e['phase'] not in ('prepare', 'screen', 'trajectory') or not isinstance(e['key'], str):
                raise ValueError('Unknown recovery execution phase/key.')
            if not isinstance(e['screen_runs'], list) or any(not isinstance(p, str) or not p.startswith('/') for p in e['screen_runs']):
                raise ValueError('Expected absolute native screen bundle paths.')
            if e['phase'] == 'prepare':
                if e['key'] or e['prepared_run'] is not None or e['screen_runs']:
                    raise ValueError('Preparation cannot consume other shards.')
            elif not isinstance(e['prepared_run'], str) or not e['prepared_run'].startswith('/'):
                raise ValueError('A shard requires an absolute prepared native bundle.')
        for key in ('persistent_fractions', 'write_budgets', 'followup_seconds', 'evaluation_seconds', 'learning_rates'):
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in getattr(obj, key)):
                raise ValueError(f'Expected finite numbers in {key}.')
        if any(not 0 <= r <= 1 for r in obj.persistent_fractions): raise ValueError('Invalid persistent variance fraction.')
        if any(type(s) is not int or not 0 < s < 2**31 for s in obj.assignment_seeds): raise ValueError('Invalid assignment seeds.')
        if not set(obj.arms) <= set(ARMS) or not set(obj.protocols) <= {'single', 'repeated'}:
            raise ValueError('Unknown recovery arm/protocol.')
        if not 0 < obj.selection_fraction <= 1 or obj.first_intervention_seconds >= obj.horizon_seconds:
            raise ValueError('Invalid mask fraction or recovery horizon.')
        if min(obj.write_budgets + obj.followup_seconds + obj.learning_rates) <= 0:
            raise ValueError('Expected positive budgets, follow-ups and learning rates.')
        if any(t < 0 or t > obj.horizon_seconds for t in obj.evaluation_seconds):
            raise ValueError('Evaluation ages must lie within the horizon.')
        if (obj.calibration_sequences > obj.training_sequences or obj.training_sequences < obj.batch_size
                or obj.batch_size % obj.microbatch_size):
            raise ValueError('Invalid calibration, training or microbatch counts.')
        if obj.canary and obj.learning_rate is None and obj.execution is None:
            raise ValueError('A timing canary requires an explicit LR, not a development screen.')
        return obj

    def intervention_times(self, protocol):
        times = [self.first_intervention_seconds]
        if protocol == 'repeated':
            while times[-1] + self.maintenance_interval_seconds < self.horizon_seconds:
                times.append(times[-1] + self.maintenance_interval_seconds)
        return tuple(times)

    def times(self, protocol):
        events = self.intervention_times(protocol)
        return tuple(sorted(set(self.evaluation_seconds) | {0., self.horizon_seconds} | set(events)
                            | {t+d for t in events for d in self.followup_seconds if t+d <= self.horizon_seconds}))
