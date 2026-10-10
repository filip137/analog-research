#!/usr/bin/env python3
"""source_record.py RUNS STAGE: print '<source path> <sha256>' of STAGE's complete run, or nothing."""
import json
from pathlib import Path
import sys

runs, stage = sys.argv[1:3]
for path in sorted(Path(runs, stage).glob("*/result.json"), reverse=True):
    result = json.loads(path.read_text())
    if result.get("status") == "complete":
        [item] = [a for a in result["artifacts"] if a["kind"] == "hwa_source"]
        print(path.parent / item["path"], item["sha256"])
        break
