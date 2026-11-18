#!/usr/bin/env bash
# cmo-drift-recovery (crossbar): plan, run and inspect one lifecycle on a lab GPU host (RTX 5090 trex/loulou/
# riri/fifi or RTX 3090), the native path of campaigns/fault-recovery-paper/launch/run_local_l12.sh.
#
# usage: run_lab.sh LIFECYCLE ROOT
#   LIFECYCLE  path relative to this checkout, e.g.
#              campaigns/pilots/20261118-cmo-drift-recovery/lifecycles/canary/cmo-drift-recovery-20261118-xb6-m4-canary.json
#   ROOT       run root; ROOT/inputs is EBL_CORPUS_ROOT and holds wikitext-2-raw-v1.pt, opt125m-synthetic-50m.pt,
#              teacher-samples8-n8192-51467dd7edb2.pt and the source (source-xb6-86522e23.pt or
#              source-l12-ed53b6cb.pt); plans and runs go to ROOT/{plans,runs}/<lifecycle id>.
# Run from a clean checkout of a commit merged into main and pushed (no --allow-dirty), in tmux.
# Refuses a GPU that already runs any compute process (policy: idle at start, never beside another user).
set -euo pipefail
config=$1
root=$(realpath "$2")
src=$(cd "$(dirname "$0")/../../.." && pwd)
py=${PYTHON:-$HOME/miniconda3/envs/py312/bin/python}
teacher=${TEACHER_WEIGHTS:-$HOME/analog-research/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin}
cd "$src"
test -z "$(git status --porcelain)" || { echo "dirty checkout" >&2; exit 2; }
busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
test -z "$busy" || { echo "GPU busy: $busy" >&2; exit 3; }
id=$("$py" -c "import json,sys; print(json.load(open(sys.argv[1]))['lifecycle_id'])" "$config")
layers=$("$py" -c "import json,sys; print(json.load(open(sys.argv[1]))['network']['analog_decoder_layers'])" "$config")
source=$root/inputs/$([ "$layers" = 12 ] && echo source-l12-ed53b6cb.pt || echo source-xb6-86522e23.pt)
"$py" - "$root/inputs" "$source" "$teacher" <<'EOF'
import hashlib, sys
from pathlib import Path
inputs, source, teacher = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pins = {inputs / 'wikitext-2-raw-v1.pt': '33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01',
        inputs / 'opt125m-synthetic-50m.pt': 'ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984',
        inputs / 'teacher-samples8-n8192-51467dd7edb2.pt': '9cba6cf3053413ec475de796ceff171f10b4f14447e685e98930eca3311739c5',
        teacher: '2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9',
        source: {'source-xb6-86522e23.pt': '86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028',
                 'source-l12-ed53b6cb.pt': 'ed53b6cbe2a7315f364291c7142ee2826c6ca4a44c21a2d5d25190b8c90569b0'}[source.name]}
for path, sha in pins.items():
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    if h.hexdigest() != sha:
        sys.exit(f'pinned input changed: {path} {h.hexdigest()}')
print('{"pins": "verified"}')
EOF
export EBL_CORPUS_ROOT=$root/inputs EBL_CAMPAIGN_STREAM=1
echo "{\"lifecycle\": \"$id\", \"commit\": \"$(git rev-parse HEAD)\", \"host\": \"$(hostname)\", \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader)\", \"started\": \"$(date -Is)\"}"
"$py" -m ebl lifecycle plan "$config" --teacher-weights "$teacher" --hwa-source "standard_hwa_model=$source" \
  --output "$root/plans/$id" --device cuda:0 --python "$py"
"$py" -m ebl campaign run --manifest "$root/plans/$id/campaign.json" --output-dir "$root/runs/$id" --resume
run=$(dirname "$(ls -d "$root/runs/$id"/*/stages/*/local/fault_tuning__*/attempt-*/runs/*/summary.json | tail -1)")
"$py" -m ebl runs inspect "$run" --verify-artifacts --require-complete
"$py" - "$config" "$run" <<'EOF'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('worker', 'campaigns/fault-recovery-paper/launch/worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
doc, rows = json.load(open(sys.argv[1])), json.load(open(sys.argv[2] + '/measurements.json'))
worker.validate_measurements(doc['fault_tuning'], rows)
summary = json.load(open(sys.argv[2] + '/summary.json'))
print(json.dumps({'validated': True, 'mean_teacher_kl': summary['drift']['mean_teacher_kl']}))
EOF
echo "{\"lifecycle\": \"$id\", \"run\": \"$run\", \"finished\": \"$(date -Is)\"}"
