"""Bit-exact golden replay for the IBM OM HWA / recovery layering refactor.

The goldens were captured by ``python tests/ibm_om_golden_cases.py capture``
on the reference commit recorded in ``tests/golden/ibm_om/meta.json``.
"""

from __future__ import annotations

import json

import pytest
import torch

import ibm_om_golden_cases as cases


_META_PATH = cases.GOLDEN_DIR / "meta.json"

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
