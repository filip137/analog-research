"""Frozen constants shared by the IBM OM layers."""

from __future__ import annotations


_STATE_VERSION = 2
_EXECUTIONS = ("mapped_target", "compact_endpoint", "pulse_resolved")
_CORRUPTION_POLICIES = ("counterfactual_repaired", "published")
_TARGET_MAPPINGS = (
    "literal_global",
    "dual_rail_quad_common_window",
    "differential_pair_common_window",
    "shared_reset_relative_quad",
    "raw_active_p90_quad",
)
_RESET_RELATIVE_MODES = ("continuous", "quantized_9_level")
_RAW_ACTIVE_MODES = ("continuous", "quantized_7_level")
_STREAMS = ("train", "evaluation")
_EVALUATION_STREAM_XOR = 0x4F4D5F50565F4556
_ARRAY_POPULATION_SCHEMA = "ebl.ibm_reram.om_array_population"
_ARRAY_POPULATION_SCHEMA_VERSION = 1
_ARRAY_SAMPLING_RECEIPT_SCHEMA = "ebl.ibm_reram.om_array_population_receipt"
_ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION = 1
_REQUIRED_AIHWKIT_VERSION = "1.1.0"
_MAX_EMPTY_COMMON_WINDOW_QUADS = 20
_MAX_EMPTY_COMMON_WINDOW_PAIRS = 20
IBM_RERAM_ENDPOINT_APPLICATION_POLICY = (
    "aihwkit_apparent_forward_persistent_update_state"
)

# Frozen by docs/ibm_om_raw_active_state_program_verify.md from the complete
# repaired development assignment (seed 84001).  These values define one
# array-wide coordinate; they must never be recomputed per cell or per later
# assignment.
IBM_OM_RAW_ACTIVE_COORDINATE_VERSION = (
    "ibm_om_raw_active_development_assignment_84001_v1"
)
IBM_OM_RAW_ACTIVE_A_MIN = -3.455834150314331
IBM_OM_RAW_ACTIVE_A_MAX = 2.5384607315063477
IBM_OM_RAW_ACTIVE_SCALE = 5.994294881820679
IBM_OM_RAW_ACTIVE_D90 = 0.10354409442884083
IBM_OM_RAW_ACTIVE_QUANTIZED_LEVELS = 7
IBM_OM_RAW_ACTIVE_TOLERANCE = 0.00791586015294392
IBM_OM_RAW_ACTIVE_CONDITIONING_QUIET_STEPS = 4
IBM_OM_RAW_ACTIVE_CONDITIONING_CHANGE_THRESHOLD = 2e-6
IBM_OM_RAW_ACTIVE_CONDITIONING_MAXIMUM_PULSES = 4096
