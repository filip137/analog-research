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


def test_sqlite_source_normalization_keeps_scientific_values_exact():
    import copy
    expected = {
        "trajectories.sqlite3": {
            "metadata": {"rows": [["characterization", json.dumps({"sampler_source_sha256": "old", "seed": 17})]]},
            "trajectories": {"rows": [[1, 0.125]]},
        },
        "integrity_report.json": {"database_sha256": "old-database", "valid": True},
    }
    actual = copy.deepcopy(expected)
    actual["trajectories.sqlite3"]["metadata"]["rows"][0][1] = json.dumps({"sampler_source_sha256": "new", "seed": 17})
    actual["integrity_report.json"]["database_sha256"] = "new-database"
    cases.assert_same(expected, actual)
    actual["trajectories.sqlite3"]["trajectories"]["rows"][0][1] = 0.126
    with pytest.raises(AssertionError):
        cases.assert_same(expected, actual)
    actual["trajectories.sqlite3"]["trajectories"]["rows"][0][1] = 0.125
    actual["trajectories.sqlite3"]["metadata"]["rows"][0][1] = json.dumps({"sampler_source_sha256": "new", "seed": 18})
    with pytest.raises(AssertionError):
        cases.assert_same(expected, actual)
