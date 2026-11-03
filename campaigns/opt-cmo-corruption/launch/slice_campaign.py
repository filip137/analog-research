#!/usr/bin/env python3
"""Select native campaign stages with their complete dependency closure.

This changes execution placement only: stage configs and artifact references
remain exact. Each independently dispatched slice has its own campaign identity
and uses the normal ebl campaign runner's source/input/resume validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ebl.execution.schema import CampaignSpec


def build_slice(manifest: Path, selected: list[str]) -> tuple[dict, dict]:
    source = manifest.resolve()
    raw = json.loads(source.read_text())
    parsed = CampaignSpec.parse(raw, base_dir=source.parent)
    stages = {stage.stage_id: stage for stage in parsed.stages}
    if not selected or len(selected) != len(set(selected)) or set(selected) - stages.keys():
        raise ValueError("Expected unique existing stage IDs")
    closure = set()

    def include(name):
        if name not in closure:
            closure.add(name)
            for dependency in stages[name].depends_on:
                include(dependency)

    for name in selected:
        include(name)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    tag = hashlib.sha256(json.dumps([digest, sorted(selected)]).encode()).hexdigest()[:12]
    targets = {stages[name].target_id for name in closure}
    result = {"schema_version": raw["schema_version"],
              "campaign_id": f"{parsed.campaign_id}--{tag}",
              "targets": [{"id": item.target_id, "worktree": str(item.worktree), "python": str(item.python)}
                          for item in parsed.targets if item.target_id in targets], "stages": []}
    configs = {}
    for stage in parsed.stages:
        if stage.stage_id not in closure:
            continue
        if not stage.config.is_file():
            raise ValueError(f"Missing stage config: {stage.config}")
        configs[stage.stage_id] = hashlib.sha256(stage.config.read_bytes()).hexdigest()
        inputs = {role: ({"path": str(ref.path)} if ref.path is not None else
                         {"stage": ref.stage, "artifact_kind": ref.artifact_kind})
                  for role, ref in stage.inputs.items()}
        result["stages"].append({"id": stage.stage_id, "case_id": stage.case_id,
                                 "target": stage.target_id, "command": stage.command,
                                 "config": str(stage.config), "depends_on": list(stage.depends_on),
                                 "inputs": inputs})
    CampaignSpec.parse(result, base_dir=source.parent)
    receipt = {"schema": "campaign.slice.v1", "source_manifest": str(source),
               "source_manifest_sha256": digest, "selected": sorted(selected),
               "included": sorted(closure), "stage_config_sha256": configs,
               "campaign_id": result["campaign_id"]}
    return result, receipt


def write_frozen(path: Path, payload: dict) -> None:
    value = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            stream.write(value)
    except FileExistsError:
        if path.read_text() != value:
            raise ValueError(f"Refusing to overwrite a changed frozen slice: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("stages", nargs="+")
    args = parser.parse_args()
    result, receipt = build_slice(args.manifest, args.stages)
    write_frozen(args.output, result)
    write_frozen(args.output.with_suffix(".slice.json"), receipt)
    print(json.dumps({"manifest": str(args.output.resolve()), "campaign_id": result["campaign_id"],
                      "stages": receipt["included"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
