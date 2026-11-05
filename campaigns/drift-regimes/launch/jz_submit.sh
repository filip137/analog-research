#!/bin/bash -l
# On Jean Zay: submit one dev-queue job per group; a group is a comma-separated list of config names.
# Usage: bash -l jz_submit.sh <staged source dir> <output root> <time limit> <group> [<group> ...]
# Prints "<first config> <job id>" per group.
set -euo pipefail
src=$1
out=$2
limit=$3
shift 3
mkdir -p "$out/slurm"
for group in "$@"; do
  configs=()
  IFS=, read -ra names <<< "$group"
  for name in "${names[@]}"; do configs+=("campaigns/drift-regimes/configs/exp-003/$name.json"); done
  tag=${names[0]}
  job=$(sbatch --parsable -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 --ntasks=1 \
    --cpus-per-task=10 --hint=nomultithread -t "$limit" -J "e3-$tag" \
    -o "$out/slurm/e3-$tag-%j.out" -e "$out/slurm/e3-$tag-%j.err" \
    "$src/campaigns/drift-regimes/launch/run.slurm" "$src" "$out" "${configs[@]}")
  echo "$tag $job"
done
