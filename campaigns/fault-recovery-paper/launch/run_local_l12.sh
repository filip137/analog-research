#!/usr/bin/env bash
# exp-002 (12 MLPs): plan, run and inspect one l12 unit on a local GPU host (RTX 3090 nom-cool-1).
#
# usage: run_local_l12.sh UNIT ROOT [TAG]
#   UNIT  deploy | bscale | canary | s1 | s2 | s3   (lifecycles/opt125m-cmo-l12-fault-paper-tune-UNIT.json)
#   ROOT  run root; ROOT/inputs is EBL_CORPUS_ROOT (corpora, the 12-MLP source, the stored V100 draw,
#         the pinned deployment and B scale); plans and runs go to ROOT/{plans,runs}/TAG (TAG defaults to UNIT)
# Run from a clean checkout of a commit merged into main and pushed (no --allow-dirty).
set -euo pipefail
unit=$1
root=$(realpath "$2")
tag=${3:-$unit}
src=$(cd "$(dirname "$0")/../../.." && pwd)
py=${PYTHON:-$HOME/miniconda3/envs/py312/bin/python}
teacher=${TEACHER_WEIGHTS:-$HOME/analog-research/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin}
export EBL_CORPUS_ROOT=$root/inputs EBL_CAMPAIGN_STREAM=1
config=$src/campaigns/fault-recovery-paper/lifecycles/opt125m-cmo-l12-fault-paper-tune-$unit.json
cd "$src"
test -z "$(git status --porcelain)" || { echo "dirty checkout" >&2; exit 2; }
echo "{\"unit\": \"$unit\", \"tag\": \"$tag\", \"commit\": \"$(git rev-parse HEAD)\", \"host\": \"$(hostname)\", \"started\": \"$(date -Is)\"}"
"$py" -m ebl lifecycle plan "$config" --teacher-weights "$teacher" \
  --hwa-source "standard_hwa_model=$root/inputs/source-l12-ed53b6cb.pt" \
  --output "$root/plans/$tag" --device cuda:0 --python "$py"
"$py" -m ebl campaign run --manifest "$root/plans/$tag/campaign.json" --output-dir "$root/runs/$tag" --resume
run=$(dirname "$(ls -d "$root/runs/$tag"/*/stages/*/local/fault_tuning__*/attempt-*/runs/*/summary.json | tail -1)")
"$py" -m ebl runs inspect "$run" --verify-artifacts --require-complete
echo "{\"unit\": \"$unit\", \"tag\": \"$tag\", \"run\": \"$run\", \"finished\": \"$(date -Is)\"}"
