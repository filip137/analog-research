"""Bit-exact golden replay of complete ``ebl train`` run bundles.

The goldens were captured by ``python tests/run_lifecycle_golden_cases.py
capture`` on the reference commit recorded in
``tests/golden/run_lifecycle/meta.json``, before the per-family ``run_train``
skeletons were moved onto the shared phase runner.  Every case runs in its
own single-threaded interpreter; cases run in parallel
(``EBL_GOLDEN_JOBS``, default 6).
"""

from __future__ import annotations

import functools
import json

import pytest
import torch

import run_lifecycle_golden_cases as cases


_META_PATH = cases.GOLDEN_DIR / "meta.json"


def _require_matching_environment() -> None:
    if not _META_PATH.is_file():
        pytest.skip("Run-lifecycle goldens have not been captured.")
    if not cases.inputs_available():
        pytest.skip("Run-lifecycle goldens need EBL_MNIST_ROOT and EBL_DEVICE_DATA.")
    meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
    if meta["torch"] != torch.__version__:
        pytest.skip(
            "Run-lifecycle goldens were captured with torch "
            f"{meta['torch']}; this environment has {torch.__version__}."
        )


@functools.lru_cache(maxsize=1)
def _replayed() -> dict[str, object]:
    def attempt(name: str) -> object:
        try:
            return cases.run_isolated(name)
        except RuntimeError as error:
            return error

    return cases._parallel(attempt, sorted(cases.all_cases()))


@pytest.mark.parametrize("name", sorted(cases.all_cases()))
def test_run_bundle_golden_is_bit_exact(name: str) -> None:
    _require_matching_environment()
    path = cases.golden_path(name)
    assert path.is_file(), f"missing golden {path}"
    actual = _replayed()[name]
    if isinstance(actual, Exception):
        raise actual
    expected = torch.load(path, weights_only=False)
    cases.assert_same(expected, actual, path=name)


def test_run_lifecycle_golden_registry_matches_captured_cases() -> None:
    _require_matching_environment()
    meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
    assert meta["cases"] == sorted(cases.all_cases())
