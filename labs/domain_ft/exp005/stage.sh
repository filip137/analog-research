#!/usr/bin/env bash
# Stage the exp-005 root on Jean Zay: labs/domain_ft/exp005/stage.sh [code-only]
# Inputs, model and data are symlinked from the exp-002 root (same sha256 manifest); code is this worktree's snapshot.
set -euo pipefail
WT=/home/filip/analog-research/worktrees/hwa-domain-ft
R2=/lustre/fsn1/projects/rech/umg/ucy17uy/domain-ft-exp002-20261108
R=/lustre/fsn1/projects/rech/umg/ucy17uy/domain-ft-exp005-20261111
SSH="ssh -o BatchMode=yes"
if [ -n "$(git -C "$WT" status --porcelain -- labs/domain_ft)" ]; then echo "commit labs/domain_ft first" >&2; exit 1; fi
$SSH jean-zay "mkdir -p $R/code/labs $R/runs $R/deploy $R/queue $R/logs $R/slurm"
rsync -a -e "$SSH" --delete --exclude __pycache__ --exclude BRIEF.md "$WT/labs/domain_ft" "jean-zay:$R/code/labs/"
git -C "$WT" rev-parse HEAD | $SSH jean-zay "cat > $R/code/labs/domain_ft/COMMIT"
$SSH jean-zay "cp $R/code/labs/domain_ft/exp005/run.slurm $R/run.slurm"
if [ "${1:-}" = "code-only" ]; then exit 0; fi
$SSH jean-zay "cd $R && for d in inputs model data; do [ -e \$d ] || ln -s $R2/\$d \$d; done && cp $R2/inputs.sha256 . && touch queue/queue.txt && sha256sum -c --quiet inputs.sha256 && echo staged"
