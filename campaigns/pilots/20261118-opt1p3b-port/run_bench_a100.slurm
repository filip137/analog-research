#!/bin/bash
#SBATCH --account=umg@a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1/logs/%x-%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1/logs/%x-%j.out
# Speed diagnostic of one OPT-1.3b HWA update (not a lifecycle stage):
#   sbatch --job-name=<name> --time=HH:MM:SS --export=ALL,SRC=<clean clone> run_bench_a100.slurm <lifecycle_id> [bench args]
set -euo pipefail
ID="$1"; shift
BASE=/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1
: "${SRC:?Submit with --export=ALL,SRC=<clean clone of the commit>}"
module purge
module load arch/a100
module load pytorch-gpu/py3/2.5.0
module load git/2.53.0
export EBL_CORPUS_ROOT="$BASE/inputs/corpora"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$SRC"
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) commit=$(git rev-parse HEAD) id=$ID args=$*"
mkdir -p "$BASE/bench"
python campaigns/pilots/20261118-opt1p3b-port/bench_hwa_step.py \
  "campaigns/pilots/20261118-opt1p3b-port/$ID.json" "$BASE/inputs/opt-1.3b-3f5c25d0/pytorch_model.bin" \
  "$BASE/bench/$ID-$SLURM_JOB_ID.json" "$@"
echo "end=$(date -Is)"
