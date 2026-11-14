#!/bin/bash -l
# One Jean Zay job: the fault-aware (partner-rewrite) deployment of the 6-MLP std3 DRN source on
# tuning map 271001 at one fault rate. Writes <output root>/<deploy name>/run/checkpoints/deployed.pt.
# Usage: deploy.slurm <staged source dir> <output root> <deploy config>
set -uo pipefail
module purge
if [ "${SLURM_JOB_PARTITION:-}" = gpu_p5 ]; then module load arch/a100; fi
module load pytorch-gpu/py3/2.5.0
threads=${SLURM_CPUS_PER_TASK:-10}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT=/lustre/fsn1/projects/rech/umg/ucy17uy
TEACHER=$ROOT/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin
WEIGHTS=$ROOT/opt125m-cmo-drn-rates-v1/inputs/hwa-std3.pt
src=$1
out=$2
config=$3
name=$(basename "$config" .json)
export OPT_DRN_SOURCE_SNAPSHOT=$src/source_snapshot.json
cd "$src"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "START $name job=${SLURM_JOB_ID:-none} $(date -u +%FT%TZ)"
mkdir -p "$out/$name"
EBL_CORPUS_ROOT=$ROOT/opt-cmo-exp1/inputs/corpora OPT_DRN_RUN_ID=run python -m ebl train --config "$config" \
  --output-dir "$out/$name" --weights "$WEIGHTS" --teacher-weights "$TEACHER"
rc=$?
test $rc -eq 0 && sha256sum "$out/$name/run/checkpoints/deployed.pt"
echo "END $name rc=$rc $(date -u +%FT%TZ)"
exit $rc
