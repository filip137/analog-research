#!/bin/bash
# Plan one lifecycle on this host and run only the named stages: train_stages.sh <lifecycle_id> <stage> [<stage> ...]
# SRC (the clean source clone) must be set.
set -euo pipefail
ID="$1"; shift
BASE="$HOME/analog-research/opt-cmo-exp1"
: "${SRC:?Set SRC to the clean source clone.}"
PY="$HOME/miniconda3/envs/py312/bin/python"
export EBL_CORPUS_ROOT="$BASE/inputs/corpora"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$SRC"
echo "host=$(hostname) start=$(date -Is) commit=$(git rev-parse HEAD) id=$ID stages=$*"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
"$PY" -c "import torch; print('torch', torch.__version__)"
"$PY" -m ebl lifecycle plan "campaigns/opt-cmo-corruption/lifecycles/$ID.json" \
  --teacher-weights "$BASE/inputs/opt-125m-27dcfa74/pytorch_model.bin" \
  --output "results/lifecycles/$ID/plan" --device cuda:0 --cpu-threads 8 > /dev/null
stage_args=()
for stage in "$@"; do stage_args+=(--stage "$stage"); done
"$PY" -m ebl campaign run --manifest "results/lifecycles/$ID/plan/campaign.json" \
  --output-dir "results/lifecycles/$ID/runs" --resume "${stage_args[@]}"

echo "end=$(date -Is) id=$ID"
