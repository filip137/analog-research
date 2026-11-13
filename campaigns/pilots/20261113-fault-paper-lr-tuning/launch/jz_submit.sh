#!/bin/bash -l
# On Jean Zay: submit one V100 dev-queue job per packed config, held until the authorized start.
# Usage: bash -l jz_submit.sh <staged source dir> <output root> <begin> <time limit> <config name> [...]
# Prints "<config name> <job id>" per job.
set -euo pipefail
src=$1
out=$2
begin=$3
limit=$4
shift 4
mkdir -p "$out/slurm"
for name in "$@"; do
  config=$src/examples/opt_cmo_drn/fault_paper_tuning/$name.json
  test -f "$config"
  job=$(sbatch --parsable --begin="$begin" -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 \
    --ntasks=1 --cpus-per-task=10 --hint=nomultithread -t "$limit" -J "fpt-drn-${name#job-}" \
    -o "$out/slurm/$name-%j.out" -e "$out/slurm/$name-%j.err" \
    "$src/campaigns/pilots/20261113-fault-paper-lr-tuning/launch/run.slurm" "$src" "$out" "$config")
  echo "$name $job"
done
