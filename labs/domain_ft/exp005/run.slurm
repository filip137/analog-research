#!/bin/bash -l
# sbatch -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 --ntasks=1 --cpus-per-task=10 \
#   --hint=nomultithread -t <limit> -J dft5-<n> -o <root>/slurm/%x-%j.out -e <root>/slurm/%x-%j.err \
#   run.slurm <root> <job_seconds> [check]
# No --mem flags: Jean Zay rejects them.
set -euo pipefail
module purge
module load pytorch-gpu/py3/2.5.0
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
root=$1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$root"
sha256sum -c --quiet inputs.sha256
if [ "${3:-}" = "check" ]; then
  cd "$root/code" && python labs/domain_ft/exp005/check_exp005.py "$root"
fi
python "$root/code/labs/domain_ft/exp005/jz_worker.py" "$root" "$2"
