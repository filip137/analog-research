#!/bin/bash -l
# On Jean Zay: submit the DRN fault-paper LR screens at 0, 1 and 5 % and the 3 % extensions.
# Per rate: one V100 dev deploy job, then five A100 dev screen jobs with afterok on it.
# The 3 % extensions run on the existing 3 % deployment right away.
# Usage: bash -l jz_submit.sh <staged source dir> <output root> <samples dir> <3 % deployment> [rate tags...]
# Prints "<name> <job id>" per job.
set -euo pipefail
src=$1
out=$2
samples=$3
state3=$4
shift 4
tags=${*:-r00 r01 r05}
dir=$src/examples/opt_cmo_drn/fault_paper_rates
launch=$src/campaigns/pilots/20261114-fault-paper-rates/launch
mkdir -p "$out/slurm"
test -f "$samples"/teacher-samples8-c919329d-df73ce66.pt
a100() {  # a100 <config name> <deployment> [dependency]
  sbatch --parsable ${3:+--dependency=afterok:$3} -A umg@a100 -p gpu_p5 --qos=qos_gpu_a100-dev -C a100 -N1 \
    --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --hint=nomultithread -t 02:00:00 -J "fpr-drn-$1" \
    -o "$out/slurm/$1-%j.out" -e "$out/slurm/$1-%j.err" "$launch/run.slurm" "$src" "$out" "$dir/$1.json" "$samples" "$2"
}
for tag in $tags; do
  deploy=$(sbatch --parsable -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 --ntasks=1 \
    --cpus-per-task=10 --hint=nomultithread -t 00:45:00 -J "fpr-drn-deploy-$tag" \
    -o "$out/slurm/deploy-$tag-%j.out" -e "$out/slurm/deploy-$tag-%j.err" \
    "$launch/deploy.slurm" "$src" "$out" "$dir/deploy-$tag.json")
  echo "deploy-$tag $deploy"
  for job in a-holds-noisy b-noiseless c-child-d d-lora16 e-lora64; do
    echo "$tag-$job $(a100 "$tag-$job" "$out/deploy-$tag/run/checkpoints/deployed.pt" "$deploy")"
  done
done
for job in x03-edges x03-child30; do
  echo "$job $(a100 "$job" "$state3")"
done
