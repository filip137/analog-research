#!/bin/bash
# Lab (RTX 5090) DRN drift-tuning run for cmo-drift-recovery exp-001 (deploys in-job).
# Same checks and native steps as the 20261116 drift token-budget run_lab.sh / run.slurm.
# run_lab.sh SOURCE OUTPUT CONFIG CORPUS_DIR WEIGHTS TEACHER LIMIT_SECONDS
# SOURCE is a `git archive` export with source_snapshot.json (make_receipt.py). Holds the
# shared-queue exclusive GPU lock and refuses an occupied GPU. Never overwrites an attempt.
set -euo pipefail
src=$1; out=$2; config=$3; corpora=$4; weights=$5; teacher=$6; limit=$7
py=/home/filip/miniconda3/envs/py312/bin/python
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
lock=$HOME/.cache/run-watch/shared-queue/gpus/$uuid.lock
mkdir -p "$(dirname "$lock")"
# The queue supervisor passes its locked open file description through exec.
# Reopening that same inode independently would block against our own supervisor.
if [[ -n ${RUN_WATCH_ATTEMPT:-} ]]; then
    [[ ${CUDA_VISIBLE_DEVICES:-} == "$uuid" ]] || { echo "Queue GPU mismatch"; exit 75; }
    gpu_fd=
    for fd in /proc/$$/fd/*; do
        if [[ $fd -ef $lock ]]; then gpu_fd=${fd##*/}; break; fi
    done
    [[ -n $gpu_fd ]] || { echo "Missing inherited GPU lock"; exit 75; }
else
    exec {gpu_fd}>>"$lock"
fi
flock -xn "$gpu_fd" || { echo "GPU lock held"; exit 75; }
if [ -n "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)" ]; then echo "GPU occupied"; exit 75; fi
export CUDA_VISIBLE_DEVICES=$uuid OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPT_DRN_SOURCE_SNAPSHOT=$src/source_snapshot.json EBL_CORPUS_ROOT=$corpora OPT_DRN_RUN_ID=run
cd "$src"
# Exact trained DRN source, teacher and staged source files.
$py - "$config" "$weights" "$src/source_snapshot.json" "$teacher" <<'CHECK'
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
receipt=Path(sys.argv[3]); r=json.load(open(receipt))
for name,sha in r['files'].items():
    if digest(receipt.parent/name)!=sha: raise SystemExit('Staged source changed: '+name)
CHECK
name=$(basename "$config" .json)
mkdir -p "$out"
mkdir "$out/$name"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
timeout --signal=TERM --kill-after=15s "${limit}s" $py -m ebl train --config "$config" --output-dir "$out/$name" --weights "$weights" --teacher-weights "$teacher"
$py -m ebl runs inspect "$out/$name/run" --verify-artifacts --require-complete
$py "$src/campaigns/pilots/20261118-cmo-drift-recovery/verify_unit.py" "$config" "$out/$name/run"
