"""Ideal weak-inversion MOSFET law, Abbott & Machta (2026), Eq. (25).

This three-terminal behavioral device is not a foundry compact model. Its
matched shared-source bank realizes Eq. (31) with equal clamped drain voltages.
"""
THERMAL_VOLTAGE = .026
SLOPE_FACTOR = 1.5
THRESHOLD = .7
CURRENT_SCALE = 1e-8
TAIL_CURRENT = 1e-9  # < I0 even if one branch takes the entire tail
GATE_BIAS = .7
DRAIN_VOLTAGE = 3.
DEVICE = "mosfet_weak_inversion"


def subcircuit():
    return [
        "* Ideal subthreshold MOSFET: arXiv:2610.12571 Eq. (25), no parasitics/mismatch.",
        ".subckt mosweak drain gate source",
        (f"Bmoschannel drain source I={CURRENT_SCALE:.17g}"
         f"*exp((V(gate)-{THRESHOLD:.17g})/{SLOPE_FACTOR*THERMAL_VOLTAGE:.17g})"
         f"*(exp(-V(source)/{THERMAL_VOLTAGE:.17g})-exp(-V(drain)/{THERMAL_VOLTAGE:.17g}))"),
        ".ends mosweak",
    ]


def operating_metrics(source, gates, probabilities):
    """Check the paper's weak-inversion domain and independently evaluate Ids."""
    import numpy as np
    source = np.asarray(source)[...,None]
    gates, probabilities = np.asarray(gates), np.asarray(probabilities)
    current = CURRENT_SCALE*np.exp((gates-THRESHOLD)/(SLOPE_FACTOR*THERMAL_VOLTAGE))*(
        np.exp(-source/THERMAL_VOLTAGE)-np.exp(-DRAIN_VOLTAGE/THERMAL_VOLTAGE))
    return {
        "weak_inversion_margin":float(np.min(THRESHOLD+SLOPE_FACTOR*np.minimum(source,DRAIN_VOLTAGE)-gates)),
        "maximum_branch_current_over_i0":float(np.max(probabilities)*TAIL_CURRENT/CURRENT_SCALE),
        "current_law_error_over_tail":float(np.max(np.abs(current-TAIL_CURRENT*probabilities))/TAIL_CURRENT),
        "tail_fraction_error":float(np.max(np.abs(probabilities.sum(axis=-1)-1.))),
    }
