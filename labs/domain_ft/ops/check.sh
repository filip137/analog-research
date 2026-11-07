#!/usr/bin/env bash
# check.sh: sync, extend queue, aggregate, status, health
cd /home/filip/analog-research/worktrees/hwa-domain-ft
P=/home/filip/miniconda3/envs/py312/bin/python
$P labs/domain_ft/make_queue.py 2>&1 | grep -v "^    " 
$P labs/domain_ft/aggregate.py
$P labs/domain_ft/status.py | tail -1
for h in riri trex; do echo "== $h"; ssh -o BatchMode=yes $h 'tmux has-session -t =dft-w0 && echo worker alive; tail -2 ~/domain-ft-20261108/logs/worker-w0.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; du -sh ~/domain-ft-20261108 | cut -f1; df -h /home | tail -1 | awk "{print \$5}"; grep "rc=[^0]" ~/domain-ft-20261108/logs/worker-*.log'; done
date
