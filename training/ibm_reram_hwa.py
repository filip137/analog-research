"""IBM OM ReRAM hardware-aware training: historical import path.

The implementation lives in :mod:`training.ibm_om`, split into layers that
are each selected from the config:

* ``population`` / ``sampling`` - the fixed AIHWKit-sampled cell assignment
* ``topology`` - dual-rail quads and differential G+/G- pairs of the DRN
* ``characterization`` - controller-observed RESET commissioning
* ``mappings`` - intended DRN fractions to per-cell targets
* ``programming`` - per-cell targets to device endpoints
* ``readback`` - endpoints written into (and restored from) the DRN
* ``modifier`` - the DRN parameter modifier tying the layers together

This module re-exports every name it historically defined, so existing
imports keep working.  Monkeypatch the defining module instead of this one.
"""

from __future__ import annotations

from training.ibm_om.characterization import (  # facade re-export
    commission_ibm_reram_reset_relative_baselines,
    IbmReramResetCommissioning,
)
from training.ibm_om.constants import (  # facade re-export
    _ARRAY_POPULATION_SCHEMA,
    _ARRAY_POPULATION_SCHEMA_VERSION,
    _ARRAY_SAMPLING_RECEIPT_SCHEMA,
    _ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION,
    _CORRUPTION_POLICIES,
    _EVALUATION_STREAM_XOR,
    _EXECUTIONS,
    IBM_OM_RAW_ACTIVE_A_MAX,
    IBM_OM_RAW_ACTIVE_A_MIN,
    IBM_OM_RAW_ACTIVE_CONDITIONING_CHANGE_THRESHOLD,
    IBM_OM_RAW_ACTIVE_CONDITIONING_MAXIMUM_PULSES,
    IBM_OM_RAW_ACTIVE_CONDITIONING_QUIET_STEPS,
    IBM_OM_RAW_ACTIVE_COORDINATE_VERSION,
    IBM_OM_RAW_ACTIVE_D90,
    IBM_OM_RAW_ACTIVE_QUANTIZED_LEVELS,
    IBM_OM_RAW_ACTIVE_SCALE,
    IBM_OM_RAW_ACTIVE_TOLERANCE,
    IBM_RERAM_ENDPOINT_APPLICATION_POLICY,
    _MAX_EMPTY_COMMON_WINDOW_PAIRS,
    _MAX_EMPTY_COMMON_WINDOW_QUADS,
    _RAW_ACTIVE_MODES,
    _REQUIRED_AIHWKIT_VERSION,
    _RESET_RELATIVE_MODES,
    _STATE_VERSION,
    _STREAMS,
    _TARGET_MAPPINGS,
)
from training.ibm_om.mappings import (  # facade re-export
    map_ibm_reram_array_targets,
    validate_ibm_reram_target_mapping_preflight,
)
from training.ibm_om.mappings.base import (  # facade re-export
    _round_half_away_from_zero,
)
from training.ibm_om.mappings.raw_active import (  # facade re-export
    _raw_active_structural_cell_mask,
)
from training.ibm_om.modifier import (  # facade re-export
    build_ibm_reram_hwa_modifier,
    IbmReramHwaParameterModifier,
)
from training.ibm_om.plants import (  # facade re-export
    _ArrayControllerPort,
    _ArrayPlant,
    IbmReramPulsePlant,
    _RawActiveArrayControllerPort,
    _RawActiveArrayPlant,
)
from training.ibm_om.population import (  # facade re-export
    _ARRAY_POPULATION_FIELDS,
    IbmReramArrayPopulation,
    load_om_array_population,
    _native_seed,
    _npz_scalar,
    _population_fingerprint,
    save_om_array_population,
    _selected_array_population,
    _tensor_sha256,
    _tensor_summary,
)
from training.ibm_om.programming.base import (  # facade re-export
    _result_mapping,
)
from training.ibm_om.sampling import (  # facade re-export
    _artifact,
    _binding_layout,
    _ROOT,
    sample_om_array_population,
    sample_om_array_population_external,
    _sample_om_array_population_layout,
    sample_om_array_population_layout,
    sample_om_array_population_layout_external,
    _sample_tile_hidden,
)
from training.ibm_om.spec import (  # facade re-export
    IbmReramHwaConfig,
)
from training.ibm_om.topology import (  # facade re-export
    _canonical_differential_pair_layout,
    _normalize_dual_rail_layouts,
    _quad_axes,
    _validate_differential_pair_bindings,
)


__all__ = [
    "IBM_RERAM_ENDPOINT_APPLICATION_POLICY",
    "IBM_OM_RAW_ACTIVE_A_MAX",
    "IBM_OM_RAW_ACTIVE_A_MIN",
    "IBM_OM_RAW_ACTIVE_COORDINATE_VERSION",
    "IBM_OM_RAW_ACTIVE_D90",
    "IBM_OM_RAW_ACTIVE_SCALE",
    "IBM_OM_RAW_ACTIVE_TOLERANCE",
    "IbmReramArrayPopulation",
    "IbmReramHwaConfig",
    "IbmReramHwaParameterModifier",
    "IbmReramResetCommissioning",
    "build_ibm_reram_hwa_modifier",
    "commission_ibm_reram_reset_relative_baselines",
    "load_om_array_population",
    "map_ibm_reram_array_targets",
    "sample_om_array_population",
    "sample_om_array_population_external",
    "sample_om_array_population_layout",
    "sample_om_array_population_layout_external",
    "save_om_array_population",
    "validate_ibm_reram_target_mapping_preflight",
]
