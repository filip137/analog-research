#!/bin/bash -l
# One Jean Zay job of the DRN fault-paper LR screen: draw-and-verify the fixed teacher
# samples on this GPU, then run one packed fault_tuning config through the public CLI.
# Usage (via jz_submit.sh): run.slurm <staged source dir> <output root> <config> [samples dir]
# Without [samples dir] the job draws the samples itself and refuses any tensor but df73ce66.
set -uo pipefail
module purge
module load pytorch-gpu/py3/2.5.0
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT=/lustre/fsn1/projects/rech/umg/ucy17uy
CORPORA=$ROOT/opt-cmo-exp1/inputs/corpora
TEACHER=$ROOT/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin
WEIGHTS=$ROOT/opt125m-cmo-drn-rates-v1/inputs/hwa-std3.pt
STATE=$ROOT/drn-fault-paper-lr-tuning-20261113/inputs/deployed-3pct-271001.pt
src=$1
out=$2
config=$3
samples=${4:-}
name=$(basename "$config" .json)
export OPT_DRN_SOURCE_SNAPSHOT=$src/source_snapshot.json
cd "$src"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "START $name job=${SLURM_JOB_ID:-none} $(date -u +%FT%TZ)"
if [ -z "$samples" ]; then
  samples=$out/corpora-${SLURM_JOB_ID:-local}
  mkdir -p "$samples"
  # The hwa runs that logged df73ce66 set no cuBLAS workspace; draw in the same environment.
  env -u CUBLAS_WORKSPACE_CONFIG EBL_CORPUS_ROOT=$CORPORA python campaigns/pilots/20261113-fault-paper-lr-tuning/draw_samples.py \
    "$config" "$TEACHER" "$samples" || { echo "SAMPLES_REFUSED $(date -u +%FT%TZ)"; exit 3; }
fi
ln -sf "$CORPORA/wikitext-2-raw-v1.pt" "$samples/wikitext-2-raw-v1.pt"
echo "SAMPLES_READY $samples $(date -u +%FT%TZ)"
mkdir -p "$out/$name"
EBL_CORPUS_ROOT=$samples OPT_DRN_RUN_ID=run python -m ebl train --config "$config" --output-dir "$out/$name" \
  --weights "$WEIGHTS" --teacher-weights "$TEACHER" --device-state "$STATE"
rc=$?
echo "END $name rc=$rc $(date -u +%FT%TZ)"
exit $rc
