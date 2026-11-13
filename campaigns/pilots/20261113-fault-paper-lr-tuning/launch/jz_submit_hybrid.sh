#!/bin/bash -l
# On Jean Zay: one V100 dev job draws and verifies the teacher samples; the packed configs then
# run on A100 dev jobs that start only after that draw succeeded (afterok).
# Usage: bash -l jz_submit_hybrid.sh <staged source dir> <output root> <begin> <time limit> <config name> [...]
# Prints "draw <job id>" and "<config name> <job id>" per job.
set -euo pipefail
src=$1
out=$2
begin=$3
limit=$4
shift 4
mkdir -p "$out/slurm"
dir=$src/examples/opt_cmo_drn/fault_paper_tuning
samples=$out/samples-v100
draw=$(sbatch --parsable --begin="$begin" -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 \
  --ntasks=1 --cpus-per-task=10 --hint=nomultithread -t 00:20:00 -J fpt-drn-draw \
  -o "$out/slurm/draw-%j.out" -e "$out/slurm/draw-%j.err" \
  "$src/campaigns/pilots/20261113-fault-paper-lr-tuning/launch/draw.slurm" "$src" "$dir/$1.json" "$samples")
echo "draw $draw"
for name in "$@"; do
  test -f "$dir/$name.json"
  job=$(sbatch --parsable --begin="$begin" --dependency=afterok:$draw -A umg@a100 -p gpu_p5 --qos=qos_gpu_a100-dev \
    -C a100 -N1 --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --hint=nomultithread -t "$limit" -J "fpt-drn-${name#job-}" \
    -o "$out/slurm/$name-%j.out" -e "$out/slurm/$name-%j.err" \
    "$src/campaigns/pilots/20261113-fault-paper-lr-tuning/launch/run.slurm" "$src" "$out" "$dir/$name.json" "$samples")
  echo "$name $job"
done
