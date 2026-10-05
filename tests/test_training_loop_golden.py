"""Bit-exact golden replay of the MNIST DRN per-batch loops.

The goldens were captured by ``python tests/training_loop_golden_cases.py
capture`` on the reference commit recorded in
``tests/golden/training_loops/meta.json``, before those loops were routed
through ``training.core.engine``.
"""

from __future__ import annotations

import json

import pytest
import torch

import training_loop_golden_cases as cases


_META_PATH = cases.GOLDEN_DIR / "meta.json"


def _require_matching_environment() -> None:
    if not _META_PATH.is_file():
        pytest.skip("Training-loop goldens have not been captured.")
    meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
    if meta["torch"] != torch.__version__:
        pytest.skip(
            "Training-loop goldens were captured with torch "
            f"{meta['torch']}; this environment has {torch.__version__}."
        )


@pytest.mark.parametrize("name", sorted(cases.all_cases()))
def test_training_loop_golden_is_bit_exact(name: str) -> None:
    _require_matching_environment()
    path = cases.golden_path(name)
    assert path.is_file(), f"missing golden {path}"
    expected = torch.load(path, weights_only=False)
    actual = cases.all_cases()[name]()
    cases.assert_same(expected, actual, path=name)


def test_training_loop_golden_registry_matches_captured_cases() -> None:
    _require_matching_environment()
    meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
    assert meta["cases"] == sorted(cases.all_cases())
