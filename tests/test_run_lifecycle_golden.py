"""Bit-exact golden replay of complete ``ebl train`` run bundles.

The goldens were captured by ``python tests/run_lifecycle_golden_cases.py
capture`` on the reference commit recorded in
``tests/golden/run_lifecycle/meta.json``, before the per-family ``run_train``
skeletons were moved onto the shared phase runner.

The replay runs real MNIST and measured-device training (about ten minutes
for all cases at twelve jobs), so it is opt-in::

    EBL_RUN_LIFECYCLE_GOLDENS=1 pytest tests/test_run_lifecycle_golden.py
    EBL_RUN_LIFECYCLE_GOLDENS=1 pytest tests/test_run_lifecycle_golden.py -k teacher

Only the selected cases run; each runs in its own single-threaded
interpreter and the selected cases run in parallel (``EBL_GOLDEN_JOBS``,
default 6).  ``python tests/run_lifecycle_golden_cases.py verify CASE...``
does the same without pytest.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import run_lifecycle_golden_cases as cases


_META_PATH = cases.GOLDEN_DIR / "meta.json"
_ENABLED = os.environ.get("EBL_RUN_LIFECYCLE_GOLDENS") == "1"
_CASE_TEST = "test_run_bundle_golden_is_bit_exact"


def _meta() -> dict:
    if not _META_PATH.is_file():
        pytest.skip("Run-lifecycle goldens have not been captured.")
    return json.loads(_META_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def replayed(request: pytest.FixtureRequest) -> dict[str, object]:
    """Replay, in parallel, exactly the cases this session selected."""

    if not cases.inputs_available():
        pytest.skip("Run-lifecycle goldens need EBL_MNIST_ROOT and EBL_DEVICE_DATA.")
    meta = _meta()
    if meta["torch"] != torch.__version__:
        pytest.skip(
            "Run-lifecycle goldens were captured with torch "
            f"{meta['torch']}; this environment has {torch.__version__}."
        )
    selected = sorted(
        item.callspec.params["name"]
        for item in request.session.items
        if item.originalname == _CASE_TEST
    )

    def attempt(name: str) -> object:
        try:
            return cases.run_isolated(name)
        except RuntimeError as error:
            return error

    return cases._parallel(attempt, selected)


@pytest.mark.skipif(
    not _ENABLED,
    reason="Slow run-bundle replay; set EBL_RUN_LIFECYCLE_GOLDENS=1 to run it.",
)
@pytest.mark.parametrize("name", sorted(cases.all_cases()))
def test_run_bundle_golden_is_bit_exact(name: str, replayed: dict[str, object]) -> None:
    path = cases.golden_path(name)
    assert path.is_file(), f"missing golden {path}"
    actual = replayed[name]
    if isinstance(actual, Exception):
        raise actual
    expected = torch.load(path, weights_only=False)
    cases.assert_same(expected, actual, path=name)


def test_run_lifecycle_golden_registry_matches_captured_cases() -> None:
    """Cheap and always on: every registered case has a captured golden."""

    meta = _meta()
    assert meta["cases"] == sorted(cases.all_cases())
    missing = [name for name in meta["cases"] if not cases.golden_path(name).is_file()]
    assert not missing, f"missing goldens: {missing}"
