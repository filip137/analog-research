"""core.cohorts — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
from hashlib import sha256
import numpy as np


def cohort_receipt(indices):
    value = np.asarray(indices, dtype="<i8")
    return {"count": len(value), "indices_sha256": sha256(value.tobytes()).hexdigest()}
