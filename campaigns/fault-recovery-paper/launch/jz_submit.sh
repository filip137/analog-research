#!/bin/bash -l
# On Jean Zay: submit prepared units, one V100 dev-QoS job each, held until --begin.
# Usage: bash -l jz_submit.sh <launch dir> <manifest.json> <HH:MM:SS> <begin> <unit> [<unit> ...]
set -euo pipefail
launch=$1
manifest=$2
limit=$3
begin=$4
shift 4
cd "$launch"
output=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['output'])" "$manifest")
mkdir -p "$output/slurm"
for unit in "$@"; do
  test -f "$output/$unit/plan/campaign.json"
  id=$(sbatch --parsable -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 --ntasks=1 \
       --cpus-per-task=10 --hint=nomultithread -t "$limit" --begin="$begin" -J "fpt-$unit" \
       -o "$output/slurm/fpt-$unit-%j.out" -e "$output/slurm/fpt-$unit-%j.err" run.slurm "$launch/$manifest" "$unit")
  echo "$unit $id"
done
