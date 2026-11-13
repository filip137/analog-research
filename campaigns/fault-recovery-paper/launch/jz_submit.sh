#!/bin/bash -l
# On Jean Zay: submit prepared units, one V100 dev-QoS job each, held until --begin.
# Usage: bash -l jz_submit.sh <launch dir> <manifest.json> <HH:MM:SS> <begin> <job> [<job> ...]
# A job is one unit or several joined by '+' (run sequentially in one allocation).
# Optional environment: DEPEND=<job id> adds --dependency=afterok:<job id> (e.g. the V100 sample draw).
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
  for part in ${unit//+/ }; do test -f "$output/$part/plan/campaign.json"; done
  id=$(sbatch --parsable ${DEPEND:+--dependency=afterok:$DEPEND} -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 --ntasks=1 \
       --cpus-per-task=10 --hint=nomultithread -t "$limit" --begin="$begin" -J "fpt-$unit" \
       -o "$output/slurm/fpt-$unit-%j.out" -e "$output/slurm/fpt-$unit-%j.err" run.slurm "$launch/$manifest" "$unit")
  echo "$unit $id"
done
