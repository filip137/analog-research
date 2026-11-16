#!/bin/bash -l
# Native V100 DRN deploy/tuning wrapper. The same wrapper is used for canaries and full jobs.
# run.slurm SOURCE OUTPUT CONFIG CORPUS_DIR WEIGHTS [DEVICE_STATE]
set -euo pipefail
src=$1
out=$2
config=$3
corpora=$4
weights=$5
state=${6:-}
module purge
module load pytorch-gpu/py3/2.5.0
threads=${SLURM_CPUS_PER_TASK:-10}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPT_DRN_SOURCE_SNAPSHOT=$src/source_snapshot.json
export EBL_CORPUS_ROOT=$corpora OPT_DRN_RUN_ID=run
teacher=/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin
cd "$src"
# Check exact trained weights, separate from the config's crossbar mapping-origin hash.
python - "$config" "$weights" "$src/source_snapshot.json" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
c=json.load(open(sys.argv[1])); expected={6:'dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9',12:'6805b62950743838de89ed356cbaf5e92a98710cf87c3f7b11093be89c4a8a1a'}
if digest(sys.argv[2])!=expected[c['analog_layers']]: raise SystemExit('Wrong trained DRN source')
receipt=Path(sys.argv[3]); r=json.load(open(receipt))
for name,sha in r['files'].items():
    if digest(receipt.parent/name)!=sha: raise SystemExit('Staged source changed: '+name)
CHECK
name=$(basename "$config" .json)
# Do not overwrite a previous attempt. A retry must use a new OUTPUT root.
mkdir -p "$out"
mkdir "$out/$name"
args=(--config "$config" --output-dir "$out/$name" --weights "$weights" --teacher-weights "$teacher")
if [ -n "$state" ]; then args+=(--device-state "$state"); fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m ebl train "${args[@]}"
python -m ebl runs inspect "$out/$name/run" --verify-artifacts --require-complete
python "$src/campaigns/pilots/20261115-fault-tuning-completion/verify_unit.py" "$config" "$out/$name/run"
