#!/bin/bash -l
# On Jean Zay: submit one dev-queue job per group for the drift-mapping campaign.
# A group is a comma-separated list of config names inside configs/<experiment>/.
# Usage: bash -l jz_submit.sh <staged source dir> <output root> <experiment> <time limit> <group> [<group> ...]
# Prints "<first config> <job id>" per group. Reuses the exp-003 run.slurm (public CLI, same inputs).
set -euo pipefail
src=$1
out=$2
experiment=$3
limit=$4
shift 4
mkdir -p "$out/slurm"
for group in "$@"; do
  configs=()
  IFS=, read -ra names <<< "$group"
  for name in "${names[@]}"; do
    path="campaigns/drift-mapping/configs/$experiment/$name.json"
    [ -f "$src/$path" ] || { echo "missing $path" >&2; exit 2; }
    configs+=("$path")
  done
  tag=${names[0]}
  job=$(sbatch --parsable -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 --ntasks=1 \
    --cpus-per-task=10 --hint=nomultithread -t "$limit" -J "dm-$tag" \
    -o "$out/slurm/dm-$tag-%j.out" -e "$out/slurm/dm-$tag-%j.err" \
    "$src/campaigns/drift-regimes/launch/run.slurm" "$src" "$out" "${configs[@]}")
  echo "$tag $job"
done
