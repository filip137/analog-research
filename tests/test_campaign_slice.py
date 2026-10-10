import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "campaigns/opt-cmo-corruption/launch/slice_campaign.py"
spec = importlib.util.spec_from_file_location("slice_campaign", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_slice_includes_exact_dependency_closure_and_pinned_paths(tmp_path):
    (tmp_path / "stages").mkdir()
    stages = []
    for name, upstream in (("prepare", None), ("devices", "prepare"), ("hwa", "devices"), ("deploy", "hwa")):
        (tmp_path / "stages" / f"{name}.json").write_text(json.dumps({"stage": name}))
        stages.append({"id": name, "case_id": name, "target": "local", "command": "train",
                       "config": f"stages/{name}.json", "depends_on": [upstream] if upstream else [],
                       "inputs": {"weights": {"stage": upstream, "artifact_kind": "weights"}}
                       if upstream else {"teacher_weights": {"path": "teacher.bin"}}})
    raw = {"schema_version": 1, "campaign_id": "test", "targets": [{"id": "local", "worktree": ".", "python": "python"}],
           "stages": stages}
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps(raw))
    result, receipt = module.build_slice(manifest, ["hwa"])
    assert [item["id"] for item in result["stages"]] == ["prepare", "devices", "hwa"]
    assert result["stages"][2]["inputs"] == stages[2]["inputs"]
    assert result["stages"][0]["inputs"]["teacher_weights"]["path"] == str(tmp_path / "teacher.bin")
    assert result["stages"][2]["config"] == str(tmp_path / "stages/hwa.json")
    assert result["campaign_id"] == receipt["campaign_id"]
    assert module.build_slice(manifest, ["hwa"]) == (result, receipt)
    output = tmp_path / "slice.json"
    module.write_frozen(output, result)
    module.write_frozen(output, result)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        module.write_frozen(output, dict(result, campaign_id="changed"))
    with pytest.raises(ValueError, match="existing stage"):
        module.build_slice(manifest, ["missing"])
