"""Programming layer: turns per-cell targets into IBM OM device endpoints.

``PROGRAMMERS`` selects the execution named by the config; none of them
sees which mapping produced the targets.
"""

from __future__ import annotations

from training.ibm_om.programming.compact import program_compact
from training.ibm_om.programming.ideal import program_mapped_target
from training.ibm_om.programming.pulse import program_pulse_resolved


PROGRAMMERS = {
    "mapped_target": program_mapped_target,
    "compact_endpoint": program_compact,
    "pulse_resolved": program_pulse_resolved,
}
