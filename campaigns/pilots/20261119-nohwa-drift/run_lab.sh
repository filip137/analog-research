#!/bin/bash
# Lab (RTX 5090) launcher for exp-002 (on-chip-learning #33): no-HWA DRN source and drift.
# run_lab.sh MODE SOURCE OUTPUT CORPUS_DIR WEIGHTS TEACHER LIMIT_SECONDS NAME [NAME ...]
#   MODE prepare: examples/opt_cmo_drn/nohwa_drift/prepare/NAME.json (weights = payload)
#   MODE cell:    nohwa_drift/faults/NAME{-deploy,}.json: deploy, inspect, hold-only fault_tuning, inspect
#   MODE drift:   nohwa_drift/drift/NAME.json: drift_tuning stage (deploys in-job), inspect
# Same GPU locks, occupancy refusal and source/teacher/receipt checks as the 20261118 and
# 20261119-mapping-faults launchers; semantic verify.py after each bundle. Never overwrites.
set -euo pipefail
mode=$1; src=$2; out=$3; corpora=$4; weights=$5; teacher=$6; limit=$7; shift 7
py=/home/filip/miniconda3/envs/py312/bin/python
pilot=$src/campaigns/pilots/20261119-nohwa-drift
cfg=$src/examples/opt_cmo_drn/nohwa_drift
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
lock=$HOME/.cache/run-watch/shared-queue/gpus/$uuid.lock
mkdir -p "$(dirname "$lock")"
queue_lock=
if [[ -n ${RUN_WATCH_ATTEMPT:-} ]]; then
    [[ ${CUDA_VISIBLE_DEVICES:-} == "$uuid" ]] || { echo "Queue GPU mismatch"; exit 75; }
    queue_lock=${RUN_WATCH_WORKER_ROOT:-$HOME/.local/state/run-watch/queue-workers}/gpus/$uuid.lock
    queue_fd=
    for fd in /proc/$$/fd/*; do
        if [[ $fd -ef $queue_lock ]]; then queue_fd=${fd##*/}; break; fi
    done
    [[ -n $queue_fd ]] || { echo "Missing inherited GPU lock"; exit 75; }
    flock -xn "$queue_fd" || { echo "Queue GPU lock held"; exit 75; }
fi
if [[ ! $lock -ef $queue_lock ]]; then
    exec {legacy_fd}>>"$lock"
    flock -xn "$legacy_fd" || { echo "Legacy GPU lock held"; exit 75; }
fi
if [ -n "$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader)" ]; then echo "GPU occupied"; exit 75; fi
export CUDA_VISIBLE_DEVICES=$uuid OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPT_DRN_SOURCE_SNAPSHOT=$src/source_snapshot.json EBL_CORPUS_ROOT=$corpora OPT_DRN_RUN_ID=run
cd "$src"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$out"
train() { timeout --signal=TERM --kill-after=15s "${limit}s" $py -m ebl train "$@"; }
inspect() { $py -m ebl runs inspect "$1" --verify-artifacts --require-complete; }
for name in "$@"; do
    echo "UNIT $name start $(date -Is)"
    case $mode in
    prepare)
        c=$cfg/prepare/$name.json
        $py "$pilot/check_inputs.py" "$c" "$weights" "$teacher" "$src/source_snapshot.json"
        mkdir "$out/$name"
        train --config "$c" --output-dir "$out/$name" --weights "$weights" --teacher-weights "$teacher"
        inspect "$out/$name/run"
        $py "$pilot/verify.py" prepare "$c" "$out/$name/run" ;;
    cell)
        dep=$cfg/faults/$name-deploy.json; tune=$cfg/faults/$name.json
        $py "$pilot/check_inputs.py" "$tune" "$weights" "$teacher" "$src/source_snapshot.json"
        mkdir "$out/$name-deploy"
        train --config "$dep" --output-dir "$out/$name-deploy" --weights "$weights" --teacher-weights "$teacher"
        inspect "$out/$name-deploy/run"
        mkdir "$out/$name"
        train --config "$tune" --output-dir "$out/$name" --weights "$weights" --teacher-weights "$teacher" --device-state "$out/$name-deploy/run/checkpoints/deployed.pt"
        inspect "$out/$name/run"
        $py "$pilot/verify.py" cell "$dep" "$tune" "$out/$name-deploy/run" "$out/$name/run" ;;
    drift)
        c=$cfg/drift/$name.json
        $py "$pilot/check_inputs.py" "$c" "$weights" "$teacher" "$src/source_snapshot.json"
        mkdir "$out/$name"
        train --config "$c" --output-dir "$out/$name" --weights "$weights" --teacher-weights "$teacher"
        inspect "$out/$name/run"
        $py "$pilot/verify.py" drift "$c" "$out/$name/run" ;;
    *) echo "Unknown mode $mode"; exit 2 ;;
    esac
    echo "UNIT $name done $(date -Is)"
done
