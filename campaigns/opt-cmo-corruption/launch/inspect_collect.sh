#!/bin/bash
# inspect_collect.sh <lifecycle_id> <source clone>: verify every bundle of the lifecycle here, then collect.
ID="$1"; SRC="$2"
. /etc/profile >/dev/null 2>&1
set -o pipefail
module purge >/dev/null 2>&1; module load pytorch-gpu/py3/2.5.0 >/dev/null 2>&1
cd "$SRC" || exit 2
python -m ebl runs inspect $(ls -d results/lifecycles/$ID/runs/*/*/) --verify-artifacts --require-complete > "results/lifecycles/$ID/inspect.txt" 2>&1
echo "inspect_exit=$? bundles=$(ls -d results/lifecycles/$ID/runs/*/*/ | wc -l)"
python -m workflow collect "results/lifecycles/$ID/runs" --output "results/lifecycles/$ID/collected" | tail -3
echo "collect_exit=$?"
