#!/usr/bin/env bash
# Stage code snapshot, data and model to a GPU host: labs/domain_ft/stage.sh <host> [code-only]
set -euo pipefail
HOST=$1
WT=/home/filip/analog-research/worktrees/hwa-domain-ft
ART=/home/filip/analog-research/on-chip-learning/artifacts/domain-finetuning/prelim-20261108
HWA=/home/filip/analog-research/hwa/data
SNAP=/home/filip/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6
R=domain-ft-20261108
SSH="ssh -o BatchMode=yes"
$SSH "$HOST" "mkdir -p ~/$R/code/labs ~/$R/data/built ~/$R/data/hwa/sources ~/$R/model ~/$R/runs ~/$R/queue ~/$R/logs"
rsync -a -e "$SSH" --delete "$WT/networks" "$WT/core" "$WT/hardware" "$HOST:~/$R/code/"
rsync -a -e "$SSH" --delete --exclude BRIEF.md "$WT/labs/domain_ft" "$HOST:~/$R/code/labs/"
git -C "$WT" rev-parse HEAD | $SSH "$HOST" "cat > ~/$R/code/labs/domain_ft/COMMIT"
if [ "${2:-}" = "code-only" ]; then exit 0; fi
rsync -a -e "$SSH" "$ART/data/built/" "$HOST:~/$R/data/built/"
rsync -a -e "$SSH" "$ART/data/receipts" "$HOST:~/$R/data/"
rsync -a -e "$SSH" "$HWA/corpora/hellaswag-opt125m-v1.pt" "$HWA/corpora/sst2-opt125m-v1.pt" \
  "$HWA/corpora/piqa-opt125m-v1.pt" "$HWA/corpora/wikitext-2-raw-v1.pt" "$HOST:~/$R/data/hwa/"
rsync -a -e "$SSH" "$HWA/tasks/sources/piqa" "$HOST:~/$R/data/hwa/sources/"
rsync -aL -e "$SSH" "$SNAP/" "$HOST:~/$R/model/"
$SSH "$HOST" "cd ~/$R && sha256sum model/pytorch_model.bin data/built/*.pt data/hwa/*.pt > data/sha256.txt && du -sh ~/$R"
