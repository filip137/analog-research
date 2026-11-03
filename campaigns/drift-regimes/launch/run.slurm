#!/bin/bash -l
# Submit: sbatch -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 \
#   --ntasks=1 --cpus-per-task=10 --hint=nomultithread -t <limit> -J dr-<group> \
#   -o <output>/slurm/dr-<group>-%j.out -e <output>/slurm/dr-<group>-%j.err run.slurm <manifest> <group>
# No --mem flags: Jean Zay rejects them.
set -euo pipefail
module purge
module load git/2.39.1
module load pytorch-gpu/py3/2.5.0
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
manifest=$1
group=$2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$(dirname "$manifest")/worker.py" "$manifest" "$group"
