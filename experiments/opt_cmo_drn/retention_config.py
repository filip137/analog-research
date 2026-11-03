"""Strict, inference-only scalar drift settings."""
from dataclasses import dataclass
import math
from training.output_calibration import METHODS
from training.power_law_drift import PowerLaw


def _increasing(values, positive):
    return (isinstance(values,(list,tuple)) and values and list(values)==sorted(set(values))
            and all(not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v)
                    and (v>0 if positive else v>=0) for v in values))


@dataclass(frozen=True)
class Calibration:
    """Closed-form MLP-output calibration of a selected subset of the parent's drift grid."""
    methods: tuple
    calibration_sequences: int
    seconds: tuple
    sigmas_us: tuple
    gain_bounds: tuple

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict) or set(raw)!=set(cls.__dataclass_fields__):
            raise ValueError('Expected exactly the DRN calibration fields.')
        methods=raw['methods']
        if (not isinstance(methods,(list,tuple)) or not methods or len(set(methods))!=len(methods)
                or any(m not in METHODS for m in methods)):
            raise ValueError(f'Expected unique calibration methods from {METHODS}.')
        if type(raw['calibration_sequences']) is not int or raw['calibration_sequences']<=0:
            raise ValueError('Expected positive calibration_sequences.')
        if not _increasing(raw['seconds'],True) or not _increasing(raw['sigmas_us'],False):
            raise ValueError('Expected increasing distinct calibrated ages and spreads.')
        bounds=raw['gain_bounds']
        if (not isinstance(bounds,(list,tuple)) or len(bounds)!=2 or not _increasing(bounds,True)
                or not bounds[0]<=1<=bounds[1]):
            raise ValueError('Expected two positive gain bounds containing one.')
        return cls(tuple(methods),raw['calibration_sequences'],tuple(float(t) for t in raw['seconds']),
                   tuple(float(s) for s in raw['sigmas_us']),tuple(float(b) for b in bounds))


@dataclass(frozen=True)
class Retention:
    seconds: tuple
    seed: int
    hwa_sha256: str
    power_law: PowerLaw
    calibration: Calibration | None = None

    @classmethod
    def parse(cls, raw):
        required=set(cls.__dataclass_fields__)-{'calibration'}
        if not isinstance(raw, dict) or set(raw)-{'calibration'}!=required:
            raise ValueError('Expected exactly the DRN retention fields.')
        times=raw['seconds']
        if (not isinstance(times,(list,tuple)) or not times
                or any(isinstance(t,bool) or not isinstance(t,(float,int)) or not math.isfinite(t) or t<=0 for t in times)
                or list(times)!=sorted(set(times))):
            raise ValueError('Expected distinct increasing positive drift ages in seconds.')
        if type(raw['seed']) is not int or not 0<raw['seed']<2**31:
            raise ValueError('Expected positive drift seed below 2**31.')
        h=raw['hwa_sha256']
        if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('Expected a pinned HWA SHA-256.')
        law=PowerLaw.parse(raw['power_law'])
        calibration=None if raw.get('calibration') is None else Calibration.parse(raw['calibration'])
        if calibration is not None and (not set(calibration.seconds)<=set(times)
                                        or not set(calibration.sigmas_us)<=set(law.sigmas_us)):
            raise ValueError('Expected calibrated ages and spreads from the drift grid.')
        return cls(tuple(float(t) for t in times),raw['seed'],h,law,calibration)
