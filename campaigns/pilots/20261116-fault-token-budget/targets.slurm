#!/bin/bash -l
# Minimal one-job target preparation; validates the historical prefix before publication.
set -euo pipefail
src=$1
out=$2
module purge
module load pytorch-gpu/py3/2.5.0
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$src"
python - "$src" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); receipt=json.loads((p/'source_snapshot.json').read_text())
for name,sha in receipt['files'].items():
    if hashlib.sha256((p/name).read_bytes()).hexdigest()!=sha:raise SystemExit('Source changed: '+name)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$src/campaigns/pilots/20261116-fault-token-budget/prepare_targets.py" \
  /lustre/fsn1/projects/rech/umg/ucy17uy/fault-paper-tuning-20261113/inputs/wikitext-2-raw-v1.pt \
  /lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin \
  "$out"
