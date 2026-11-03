#!/bin/bash -l
set -euo pipefail
module purge
# A100 nodes need their architecture module before the shared PyTorch module.
if [ "${SLURM_JOB_PARTITION:-}" = gpu_p5 ]; then module load arch/a100; fi
module load pytorch-gpu/py3/2.5.0
# Slurm executes a spool copy of this script. Resolve source from the explicit
# immutable manifest argument, never from $0 or the spool directory.
case "$1" in /*) ;; *) echo "Expected an absolute manifest path" >&2; exit 2 ;; esac
cd "$(dirname "$1")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 OPT_DRN_RUN_ID=run
export EBL_CORPUS_ROOT=/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1/inputs/corpora
export OPT_DRN_SOURCE_SNAPSHOT="$PWD/source-snapshot.json"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks=1 python -m workflow.slurm_recovery worker --manifest "$1" --mode "$2"
