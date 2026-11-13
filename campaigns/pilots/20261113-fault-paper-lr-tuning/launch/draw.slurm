#!/bin/bash -l
# V100 job: reproduce the fixed teacher-sample draw df73ce66 into a shared directory.
# Writes the file only if its tensor hash matches; exits nonzero otherwise, so jobs
# submitted with --dependency=afterok on this one never start on a different draw.
# Usage: draw.slurm <staged source dir> <config> <samples dir>
set -uo pipefail
module purge
module load pytorch-gpu/py3/2.5.0
export OMP_NUM_THREADS=10 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ROOT=/lustre/fsn1/projects/rech/umg/ucy17uy
src=$1
config=$2
samples=$3
cd "$src"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$samples"
# The hwa runs that logged df73ce66 set no cuBLAS workspace; draw in the same environment.
env -u CUBLAS_WORKSPACE_CONFIG EBL_CORPUS_ROOT=$ROOT/opt-cmo-exp1/inputs/corpora \
  python campaigns/pilots/20261113-fault-paper-lr-tuning/draw_samples.py \
  "$config" "$ROOT/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin" "$samples"
rc=$?
echo "DRAW rc=$rc $(date -u +%FT%TZ)"
exit $rc
