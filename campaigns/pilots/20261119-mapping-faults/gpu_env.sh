# Source from native OPT DRN lab runners to lease the selected GPU and configure deterministic executions.
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
