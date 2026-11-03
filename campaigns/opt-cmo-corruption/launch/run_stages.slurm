#!/bin/bash
#SBATCH --account=umg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-dev
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1/logs/%x-%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1/logs/%x-%j.out
# Run named stages of one planned lifecycle on one V100:
#   sbatch --job-name=<name> --time=HH:MM:SS run_stages.slurm <lifecycle_id> [--input STAGE:ROLE=PATH ...] STAGE [STAGE ...]
set -euo pipefail
ID="$1"; shift
BASE=/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1
: "${SRC:?Submit with --export=ALL,SRC=<clean clone of the commit>}"
module purge
module load pytorch-gpu/py3/2.5.0
module load git/2.53.0
export EBL_CORPUS_ROOT="$BASE/inputs/corpora"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
cd "$SRC"
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) commit=$(git rev-parse HEAD) id=$ID args=$*"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
stage_args=()
while (($#)); do
  if [[ "$1" == --input ]]; then stage_args+=(--input "$2"); shift 2
  else stage_args+=(--stage "$1"); shift; fi
done
python -m ebl campaign run --manifest "results/lifecycles/$ID/plan/campaign.json" \
  --output-dir "results/lifecycles/$ID/runs" --resume "${stage_args[@]}"
echo "end=$(date -Is)"
