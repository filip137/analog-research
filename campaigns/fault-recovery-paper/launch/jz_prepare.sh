#!/bin/bash -l
# On Jean Zay (login node): plan and dry-run each unit of the manifest.
# Usage: bash -l jz_prepare.sh <launch dir> <manifest.json> <unit> [<unit> ...]
set -uo pipefail
launch=$1
manifest=$2
shift 2
module purge
module load git/2.39.1 pytorch-gpu/py3/2.5.0 >/dev/null 2>&1
cd "$launch"
mkdir -p logs
for unit in "$@"; do
  if python worker.py "$manifest" "$unit" --prepare > "logs/prepare-$unit.log" 2>&1; then
    echo "prepared $unit"
  else
    echo "FAILED $unit"; tail -5 "logs/prepare-$unit.log"
  fi
done
