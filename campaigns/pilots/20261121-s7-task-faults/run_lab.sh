#!/bin/bash
# Native campaign launcher on a leased lab RTX 5090.
# run_lab.sh PINNED_WORKTREE INPUTS MANIFEST OUTPUT LIMIT_SECONDS [--resume]
set -euo pipefail
src=$1; corpora=$2; manifest=$3; out=$4; limit=$5; shift 5
py=/home/filip/miniconda3/envs/py312/bin/python
. "$src/campaigns/pilots/20261119-mapping-faults/gpu_env.sh"
export EBL_CORPUS_ROOT=$corpora OPT_DRN_RUN_ID=run
cd "$src"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
# EBL's worktree preflight fingerprints this exact commit and rejects dirty sources.
exec timeout --signal=TERM --kill-after=15s "${limit}s" "$py" -m ebl campaign run \
    --manifest "$manifest" --output-dir "$out" --fail-fast "$@"
