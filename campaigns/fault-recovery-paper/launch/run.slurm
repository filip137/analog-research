#!/bin/bash -l
# One fault-tuning screen unit on a Jean Zay V100 (dev QoS). Resources come from jz_submit.sh:
#   sbatch -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 --ntasks=1 \
#     --cpus-per-task=10 --hint=nomultithread -t 01:30:00 --begin=... run.slurm <manifest> <unit>[+<unit>...]
# Units joined by '+' run one after another in the same job; each keeps its own output root.
# No --mem flags: Jean Zay rejects them.
set -euo pipefail
module purge
module load git/2.39.1
module load pytorch-gpu/py3/2.5.0
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
manifest=$1
units=$2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
status=0
for unit in ${units//+/ }; do
  python "$(dirname "$manifest")/worker.py" "$manifest" "$unit" || status=1
done
exit $status
