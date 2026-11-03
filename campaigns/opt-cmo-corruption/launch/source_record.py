#!/usr/bin/env python3
"""source_record.py MANIFEST RUNS STAGE: resolve an unchanged completed HWA source."""
import json
import subprocess
import sys

manifest, runs, stage = sys.argv[1:4]
completed = subprocess.run([sys.executable, '-m', 'ebl', 'campaign', 'artifact', '--manifest', manifest,
                            '--output-dir', runs, '--stage', stage, '--kind', 'hwa_source'],
                           check=True, capture_output=True, text=True)
record = json.loads(completed.stdout)
print(record['path'], record['sha256'])
