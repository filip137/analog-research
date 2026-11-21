#!/bin/bash
# Lab (RTX 5090) mapping-faults cells (on-chip-learning #33, series 003 exp-001), evaluation only.
# run_lab.sh SOURCE OUTPUT CORPUS_DIR WEIGHTS TEACHER LIMIT_SECONDS CELL [CELL ...]
# Per cell: native deploy (`ebl train`, stage deploy), inspect, native fault_tuning on
# that deployed.pt, inspect, verify_cell.py. Same GPU locks, occupancy refusal and source/input
# hash checks as the 20261118 cmo-drift-recovery run_lab.sh. Never overwrites an attempt.
set -euo pipefail
src=$1; out=$2; corpora=$3; weights=$4; teacher=$5; limit=$6; shift 6
py=/home/filip/miniconda3/envs/py312/bin/python
pilot=$src/campaigns/pilots/20261119-mapping-faults
configs=${OPT_DRN_MAPPING_CONFIGS:-$src/examples/opt_cmo_drn/mapping_faults}
. "$pilot/gpu_env.sh"
export OPT_DRN_SOURCE_SNAPSHOT=$src/source_snapshot.json EBL_CORPUS_ROOT=$corpora OPT_DRN_RUN_ID=run
cd "$src"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$out"
for cell in "$@"; do
    dep=$configs/$cell-deploy.json
    tune=$configs/$cell.json
    # Exact trained DRN source for this depth, teacher and staged source files.
    $py - "$tune" "$weights" "$src/source_snapshot.json" "$teacher" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
c=json.load(open(sys.argv[1])); expected={6:'dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9',12:'6805b62950743838de89ed356cbaf5e92a98710cf87c3f7b11093be89c4a8a1a'}
if digest(sys.argv[2])!=expected[c['analog_layers']]: raise SystemExit('Wrong trained DRN source')
if digest(sys.argv[4])!='2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9': raise SystemExit('Wrong teacher')
r=json.load(open(sys.argv[3]))
for name,sha in r['files'].items():
    if digest(Path(sys.argv[3]).parent/name)!=sha: raise SystemExit('Staged source changed: '+name)
CHECK
    echo "CELL $cell start $(date -Is)"
    mkdir "$out/$cell-deploy"
    timeout --signal=TERM --kill-after=15s "${limit}s" $py -m ebl train --config "$dep" --output-dir "$out/$cell-deploy" --weights "$weights" --teacher-weights "$teacher"
    $py -m ebl runs inspect "$out/$cell-deploy/run" --verify-artifacts --require-complete
    mkdir "$out/$cell"
    timeout --signal=TERM --kill-after=15s "${limit}s" $py -m ebl train --config "$tune" --output-dir "$out/$cell" --weights "$weights" --teacher-weights "$teacher" --device-state "$out/$cell-deploy/run/checkpoints/deployed.pt"
    $py -m ebl runs inspect "$out/$cell/run" --verify-artifacts --require-complete
    $py "$pilot/verify_cell.py" "$dep" "$tune" "$out/$cell-deploy/run" "$out/$cell/run"
    echo "CELL $cell done $(date -Is)"
done
