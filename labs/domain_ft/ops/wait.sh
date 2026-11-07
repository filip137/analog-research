#!/usr/bin/env bash
# wait.sh <seconds>: return early if a worker dies, a run fails, or a queue drains
end=$(( $(date +%s) + $1 ))
base=$(for h in riri trex; do ssh -o BatchMode=yes -o ConnectTimeout=15 $h 'grep -h "rc=[^0]" ~/domain-ft-20261108/logs/worker-*.log 2>/dev/null | wc -l'; done | paste -sd+ | bc)
while [ $(date +%s) -lt $end ]; do
  sleep 60
  for h in riri trex; do
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $h 'tmux has-session -t =dft-w0 2>/dev/null && echo alive || echo dead; tail -1 ~/domain-ft-20261108/logs/worker-w0.log' 2>&1) || { echo "SSH_FAIL $h"; exit 0; }
    echo "$out" | head -1 | grep -q alive || { echo "WORKER_DEAD $h: $out"; exit 0; }
  done
  n=$(for h in riri trex; do ssh -o BatchMode=yes -o ConnectTimeout=15 $h 'grep -h "rc=[^0]" ~/domain-ft-20261108/logs/worker-*.log 2>/dev/null | wc -l'; done | paste -sd+ | bc)
  [ "$n" != "$base" ] && { echo "RUN_FAILED count $base -> $n"; exit 0; }
done
echo TIMEOUT
