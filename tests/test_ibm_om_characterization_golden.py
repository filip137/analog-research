"""Bit-exact golden replay for the characterization (per-cell RNG) stack."""

from __future__ import annotations

import json

import pytest
import torch

import ibm_om_characterization_golden_cases as cases


_META = cases.GOLDEN_DIR / "meta.json"


@pytest.mark.parametrize("name", sorted(cases.all_cases()))
def test_characterization_golden_is_bit_exact(name: str) -> None:
    if not _META.is_file():
        pytest.skip("Characterization goldens have not been captured.")
    meta = json.loads(_META.read_text(encoding="utf-8"))
    if meta["torch"] != torch.__version__:
        pytest.skip(f"Goldens were captured with torch {meta['torch']}.")
    fn, needs = cases.all_cases()[name]
    available = cases.requirements()
    missing = [need for need in needs if not available[need]]
    if missing:
        pytest.skip(f"Requires {', '.join(missing)}.")
    expected = torch.load(cases.golden_path(name), weights_only=False)
    cases.assert_same(expected, fn(), path=name)
