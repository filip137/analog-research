#!/bin/bash -l
# One Jean Zay job: run each exp-003 config in turn through the public CLI.
# Submit with jz_submit.sh (umg@v100, gpu_p13, qos_gpu-dev, v100-32g, 10 CPUs; no --mem flags).
# Usage: run.slurm <staged source dir> <output root> <config> [<config> ...]
set -uo pipefail
module purge
module load pytorch-gpu/py3/2.5.0
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ROOT=/lustre/fsn1/projects/rech/umg/ucy17uy
export EBL_CORPUS_ROOT=$ROOT/opt-cmo-exp1/inputs/corpora
TEACHER=$ROOT/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin
WEIGHTS=$ROOT/opt125m-cmo-drn-rates-v1/inputs/hwa-std3.pt
src=$1
out=$2
shift 2
export OPT_DRN_SOURCE_SNAPSHOT=$src/source_snapshot.json
cd "$src"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
status=0
for config in "$@"; do
  name=$(basename "$config" .json)
  mkdir -p "$out/$name"
  echo "START $name $(date -u +%FT%TZ)"
  python -m ebl train --config "$config" --output-dir "$out/$name" --weights "$WEIGHTS" \
    --teacher-weights "$TEACHER"
  rc=$?
  [ $rc -eq 0 ] || status=1
  echo "END $name rc=$rc $(date -u +%FT%TZ)"
done
exit $status
