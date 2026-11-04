from __future__ import annotations

import json
from pathlib import Path

import pytest

from ebl.execution.schema import (
    CampaignManifestError,
    CampaignSpec,
    load_campaign_manifest,
    topological_stages,
)


def _manifest(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "campaign_id": "comparison",
        "targets": [
            {
                "id": "base",
                "worktree": str(tmp_path / "base"),
                "python": "/usr/bin/python3",
            },
            {
                "id": "feature",
                "worktree": str(tmp_path / "feature"),
                "python": "/usr/bin/python3",
            },
        ],
        "stages": [
            {
                "id": "base_train",
                "case_id": "digits",
                "target": "base",
                "command": "train",
                "config": "base.json",
            },
            {
                "id": "feature_validate",
                "case_id": "digits",
                "target": "feature",
                "command": "validate",
                "config": "feature.json",
                "depends_on": ["base_train"],
                "inputs": {
                    "weights": {
                        "stage": "base_train",
                        "artifact_kind": "weights",
                    }
                },
            },
        ],
    }


def test_campaign_parses_and_orders_dependencies(tmp_path: Path) -> None:
    parsed = CampaignSpec.parse(_manifest(tmp_path), base_dir=tmp_path)
    assert [item.stage_id for item in topological_stages(parsed)] == [
        "base_train",
        "feature_validate",
    ]
    assert parsed.stages[1].inputs["weights"].stage == "base_train"


def test_campaign_accepts_generic_external_input_paths(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["stages"][0]["inputs"] = {
        "device_data": {"path": "devices.hdf5"},
        "teacher_weights": {"path": "teacher.pt"},
    }

    parsed = CampaignSpec.parse(manifest, base_dir=tmp_path)

    assert parsed.stages[0].inputs["device_data"].path == (
        tmp_path / "devices.hdf5"
    ).resolve()
    assert parsed.stages[0].inputs["teacher_weights"].path == (
        tmp_path / "teacher.pt"
    ).resolve()


def test_campaign_accepts_characterization_and_device_model_inputs(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["stages"][0]["command"] = "characterize"
    manifest["stages"][1]["inputs"]["device_model"] = {
        "path": "om-device-model.json"
    }

    parsed = CampaignSpec.parse(manifest, base_dir=tmp_path)

    assert parsed.stages[0].command == "characterize"
    assert parsed.stages[1].inputs["device_model"].path == (
        tmp_path / "om-device-model.json"
    ).resolve()


def test_campaign_rejects_unknown_external_input_key(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["stages"][0]["inputs"] = {
        "teacher_weight": {"path": "teacher.pt"}
    }

    with pytest.raises(ValueError, match="stage.inputs keys.*teacher_weights"):
        CampaignSpec.parse(manifest, base_dir=tmp_path)


def test_campaign_rejects_unknown_target(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["stages"][0]["target"] = "missing"
    with pytest.raises(ValueError, match="declared target"):
        CampaignSpec.parse(manifest, base_dir=tmp_path)


def test_campaign_rejects_cycles(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["stages"][0]["depends_on"] = ["feature_validate"]
    with pytest.raises(ValueError, match="acyclic"):
        CampaignSpec.parse(manifest, base_dir=tmp_path)


def test_stage_input_must_be_dependency(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["stages"][1]["depends_on"] = []
    with pytest.raises(ValueError, match="also appear in depends_on"):
        CampaignSpec.parse(manifest, base_dir=tmp_path)


def test_manifest_loader_resolves_all_relative_paths_explicitly(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest["targets"][0] = {
        "id": "base",
        "worktree": "base",
        "python": ".venv/bin/python",
    }
    manifest_path = tmp_path / "campaign.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    parsed = load_campaign_manifest(manifest_path)

    assert parsed.targets[0].worktree == (tmp_path / "base").resolve()
    assert parsed.targets[0].python == (
        tmp_path / "base" / ".venv/bin/python"
    ).resolve()
    assert parsed.stages[0].config == (tmp_path / "base.json").resolve()
    with pytest.raises(TypeError):
        parsed.stages[1].inputs["new"] = parsed.stages[1].inputs["weights"]


@pytest.mark.parametrize(
    "contents, message",
    [
        ('{"schema_version": 1,', "valid strict JSON"),
        (
            '{"schema_version": 1, "schema_version": 1}',
            "key to be unique",
        ),
        (
            (
                '{"schema_version": 1, "campaign_id": "x", '
                '"targets": [NaN], "stages": []}'
            ),
            "strict JSON syntax",
        ),
    ],
)
def test_manifest_loader_rejects_non_strict_json(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "campaign.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(CampaignManifestError, match=message):
        load_campaign_manifest(path)


def test_manifest_paths_must_be_strings(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["stages"][0]["config"] = 7
    with pytest.raises(ValueError, match="non-empty path string"):
        CampaignSpec.parse(manifest, base_dir=tmp_path)
