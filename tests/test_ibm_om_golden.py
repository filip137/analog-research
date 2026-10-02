"""Bit-exact golden replay for the IBM OM HWA / recovery layering refactor.

The goldens were captured by ``python tests/ibm_om_golden_cases.py capture``
on the reference commit recorded in ``tests/golden/ibm_om/meta.json``.
"""

from __future__ import annotations

import importlib
import json

import pytest
import torch

import ibm_om_golden_cases as cases


_META_PATH = cases.GOLDEN_DIR / "meta.json"

# Every name imported from ``training.ibm_reram_hwa`` anywhere in the
# repository before the refactor, plus its ``__all__`` and the private
# function tests monkeypatch.  The facade must keep all of them importable.
IMPORT_SURFACE = (
    "IBM_OM_RAW_ACTIVE_A_MAX",
    "IBM_OM_RAW_ACTIVE_A_MIN",
    "IBM_OM_RAW_ACTIVE_COORDINATE_VERSION",
    "IBM_OM_RAW_ACTIVE_D90",
    "IBM_OM_RAW_ACTIVE_SCALE",
    "IBM_OM_RAW_ACTIVE_TOLERANCE",
    "IBM_RERAM_ENDPOINT_APPLICATION_POLICY",
    "IbmReramArrayPopulation",
    "IbmReramHwaConfig",
    "IbmReramHwaParameterModifier",
    "IbmReramPulsePlant",
    "IbmReramResetCommissioning",
    "_ARRAY_SAMPLING_RECEIPT_SCHEMA",
    "_ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION",
    "_REQUIRED_AIHWKIT_VERSION",
    "_RawActiveArrayPlant",
    "_artifact",
    "_population_fingerprint",
    "_raw_active_structural_cell_mask",
    "_sample_om_array_population_layout",
    "_sample_tile_hidden",
    "_selected_array_population",
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
)


def _meta() -> dict:
    return json.loads(_META_PATH.read_text(encoding="utf-8"))


def _require_matching_environment() -> None:
    if not _META_PATH.is_file():
        pytest.skip("IBM OM goldens have not been captured.")
    meta = _meta()
    if meta["torch"] != torch.__version__:
        pytest.skip(
            "IBM OM goldens were captured with torch "
            f"{meta['torch']}; this environment has {torch.__version__}."
        )


@pytest.mark.parametrize("name", sorted(cases.all_cases()))
def test_golden_case_is_bit_exact(name: str) -> None:
    _require_matching_environment()
    path = cases.golden_path(name)
    assert path.is_file(), f"missing golden {path}"
    expected = torch.load(path, weights_only=False)
    actual = cases.all_cases()[name]()
    cases.assert_same(expected, actual, path=name)


def test_golden_registry_matches_captured_cases() -> None:
    _require_matching_environment()
    assert _meta()["cases"] == sorted(cases.all_cases())


@pytest.mark.parametrize("name", IMPORT_SURFACE)
def test_facade_import_surface(name: str) -> None:
    module = importlib.import_module("training.ibm_reram_hwa")
    assert hasattr(module, name), name
