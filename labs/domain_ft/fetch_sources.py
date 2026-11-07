"""Download pinned dataset source files (repo + revision + SHA-256) for exp-001.

Usage: python labs/domain_ft/fetch_sources.py <data_root>
Writes <data_root>/sources/<key>/<file> and <data_root>/receipts/sources.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SOURCES = {
    "mnli": ("nyu-mll/glue", {
        "train": "mnli/train-00000-of-00001.parquet",
        "validation_matched": "mnli/validation_matched-00000-of-00001.parquet",
        "validation_mismatched": "mnli/validation_mismatched-00000-of-00001.parquet"}),
    "rct": ("armanc/pubmed-rct20k", {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}),
    "boolq": ("google/boolq", {"train": "data/train-00000-of-00001.parquet",
                               "validation": "data/validation-00000-of-00001.parquet"}),
    "casehold": ("coastalcph/lex_glue", {"train": "case_hold/train-00000-of-00001.parquet",
                                         "validation": "case_hold/validation-00000-of-00001.parquet",
                                         "test": "case_hold/test-00000-of-00001.parquet"}),
    "pubmed": ("ccdv/pubmed-summarization", {
        **{f"train{i}": f"document/train-0000{i}-of-00005.parquet" for i in range(5)},
        "validation": "document/validation-00000-of-00001.parquet",
        "test": "document/test-00000-of-00001.parquet"}),
    "code": ("codeparrot/codeparrot-clean-valid", {"all": "file-000000000054.json.gz"}),
    "arc_easy": ("allenai/ai2_arc", {"train": "ARC-Easy/train-00000-of-00001.parquet",
                                     "validation": "ARC-Easy/validation-00000-of-00001.parquet",
                                     "test": "ARC-Easy/test-00000-of-00001.parquet"}),
    "lambada": ("EleutherAI/lambada_openai", {"test": "default/test/default.parquet"}),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(root: Path) -> None:
    api = HfApi()
    receipt_path = root / "receipts" / "sources.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for key, (repo, files) in SOURCES.items():
        if key in receipt and all((root / v["path"]).exists() for v in receipt[key]["files"].values()):
            continue
        revision = api.dataset_info(repo).sha
        entry = {"repo": repo, "revision": revision, "files": {}}
        for name, remote in files.items():
            cached = Path(hf_hub_download(repo, remote, repo_type="dataset", revision=revision))
            target = root / "sources" / key / Path(remote).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
            entry["files"][name] = {"source": remote, "path": str(target.relative_to(root)),
                                    "sha256": sha256(target), "bytes": target.stat().st_size}
            print(key, name, entry["files"][name]["bytes"], flush=True)
        receipt[key] = entry
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
