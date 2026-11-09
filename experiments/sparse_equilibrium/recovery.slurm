#!/bin/bash -l
set -euo pipefail
module purge
module load pytorch-gpu/py3/2.5.0
case "$1" in /*) ;; *) echo "Expected absolute manifest path" >&2; exit 2 ;; esac
cd "$(dirname "$1")/.."
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
srun --ntasks=1 python -m experiments.sparse_equilibrium.launch worker --manifest "$1" --stage "$2"
