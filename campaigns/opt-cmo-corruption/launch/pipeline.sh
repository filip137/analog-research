#!/bin/bash
# pipeline.sh: for one lifecycle trained as a chain on one host, ship each finished source to Jean Zay,
# verify its hash against the run record, submit its deployments (one dev job per arm), then wait for
# every job, inspect and collect on Jean Zay, inspect on the training host, and copy rows and metrics back.
#
# Required environment: HOST (training host), CLONE (clone dir on HOST), JCLONE (clone dir on Jean Zay),
#   ID (lifecycle id), ALIVE (remote command that succeeds while the training chain lives),
#   SHARED (Jean Zay job id of the lifecycle's prepare/devices job).
# Optional: EXTRA (comma list of other Jean Zay job ids to wait for), JZ_ROOT, REPO, STAGE_TMP,
#   SEEDS, DEPLOY_TIME, JOB_PREFIX.  Arguments: ARM ... (in training order).
# Never edit this file while an instance runs: bash reads scripts incrementally.
set -uo pipefail
: "${HOST:?}" "${CLONE:?}" "${JCLONE:?}" "${ID:?}" "${ALIVE:?}" "${SHARED:?}"
R=${JZ_ROOT:-/lustre/fsn1/projects/rech/umg/ucy17uy/opt-cmo-exp1}
JSRC=$R/$JCLONE
FBASE=analog-research/opt-cmo-exp1
FRUNS=$FBASE/$CLONE/results/lifecycles/$ID/runs
LOCAL=${REPO:-$HOME/analog-research/hwa}/results/lifecycles/$ID
TMP=${STAGE_TMP:-$HOME/.cache/opt-cmo-exp1/sources}
SEEDS=${SEEDS:-"271001 271002 271003 271004 271005"}
mkdir -p "$TMP" "$LOCAL/collected" "$LOCAL/hwa-metrics"
JOBS=$SHARED${EXTRA:+,$EXTRA}
rssh() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$@"; }
for ARM in "$@"; do
  echo "$(date -Is) [$ID] waiting for hwa__$ARM on $HOST"
  while true; do
    INFO=$(rssh $HOST "cd $FBASE/$CLONE && ~/miniconda3/envs/py312/bin/python ../launch/source_record.py results/lifecycles/$ID/plan/campaign.json results/lifecycles/$ID/runs hwa__$ARM" 2>/dev/null)
    [ -n "$INFO" ] && break
    if ! rssh $HOST "$ALIVE" >/dev/null 2>&1; then
      sleep 30
      INFO=$(rssh $HOST "cd $FBASE/$CLONE && ~/miniconda3/envs/py312/bin/python ../launch/source_record.py results/lifecycles/$ID/plan/campaign.json results/lifecycles/$ID/runs hwa__$ARM" 2>/dev/null)
      [ -n "$INFO" ] && break
      echo "$(date -Is) [$ID] FAIL: chain on $HOST ended without a complete hwa__$ARM"; exit 3
    fi
    sleep 120
  done
  SRCPATH=${INFO% *}; SHA=${INFO#* }
  NAME="$ID-$ARM-source.pt"
  echo "$(date -Is) [$ID] hwa__$ARM complete: $SRCPATH $SHA"
  scp -q "$HOST:$SRCPATH" "$TMP/$NAME" || { echo "FAIL scp from $HOST"; exit 4; }
  [ "$(sha256sum "$TMP/$NAME" | cut -c1-64)" = "$SHA" ] || { echo "FAIL local hash"; exit 4; }
  scp -q "$TMP/$NAME" "jean-zay:$R/sources/$NAME" || { echo "FAIL scp to JZ"; exit 4; }
  [ "$(rssh jean-zay "sha256sum $R/sources/$NAME" | cut -c1-64)" = "$SHA" ] || { echo "FAIL JZ hash"; exit 4; }
  STATE=$(rssh jean-zay "sacct -X -n -j $SHARED -o State" | tr -d ' ')
  case "$STATE" in
    COMPLETED) DEP="" ;;
    PENDING|RUNNING) DEP="--dependency=afterok:$SHARED" ;;
    *) echo "FAIL: shared job $SHARED is $STATE"; exit 5 ;;
  esac
  INPUTS=""; STAGES=""
  for S in $SEEDS; do
    INPUTS="$INPUTS --input deploy__${ARM}__$S:weights=$R/sources/$NAME"
    STAGES="$STAGES deploy__${ARM}__$S"
  done
  J=""
  for TRY in $(seq 1 24); do
    J=$(rssh jean-zay "cd $R && sbatch --parsable $DEP --export=ALL,SRC=$JSRC --job-name=${JOB_PREFIX:-x1e}-$ARM --time=${DEPLOY_TIME:-01:00:00} launch/run_stages.slurm $ID $INPUTS $STAGES" 2>&1)
    [[ "$J" =~ ^[0-9]+$ ]] && break
    echo "$(date -Is) [$ID] sbatch refused ($J); retrying in 5 min"; J=""; sleep 300
  done
  [ -n "$J" ] || { echo "FAIL sbatch for $ARM"; exit 6; }
  JOBS="$JOBS,$J"
  echo "$(date -Is) [$ID] submitted deployments of $ARM as Jean Zay job $J (hash verified)"
done
echo "$(date -Is) [$ID] waiting for Jean Zay jobs $JOBS"
while true; do
  STATES=$(rssh jean-zay "sacct -X -n -j $JOBS -o JobID,State" 2>/dev/null)
  if echo "$STATES" | grep -q -E "FAILED|CANCELLED|TIMEOUT|OUT_OF_ME|NODE_FAIL"; then echo "$STATES"; echo "[$ID] FAIL: a Jean Zay job failed"; exit 7; fi
  if [ -z "$STATES" ] || echo "$STATES" | grep -q -E "PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING"; then sleep 120; continue; fi
  echo "$STATES"; break
done
rssh jean-zay "bash $R/launch/inspect_collect.sh $ID $JSRC"
scp -q "jean-zay:$JSRC/results/lifecycles/$ID/collected/deployments.*" "$LOCAL/collected/"
scp -q "jean-zay:$JSRC/results/lifecycles/$ID/inspect.txt" "$LOCAL/collected/inspect-jean-zay.txt"
rssh $HOST "cd $FBASE/$CLONE && ~/miniconda3/envs/py312/bin/python -m ebl runs inspect \$(find results/lifecycles/$ID/runs -name result.json -printf '%h ') --verify-artifacts --require-complete > results/lifecycles/$ID/inspect.txt 2>&1; echo ${HOST}_inspect_exit=\$?"
scp -q "$HOST:$FBASE/$CLONE/results/lifecycles/$ID/inspect.txt" "$LOCAL/collected/inspect-$HOST.txt"
for ARM in "$@"; do
  INFO=$(rssh $HOST "cd $FBASE/$CLONE && ~/miniconda3/envs/py312/bin/python ../launch/source_record.py results/lifecycles/$ID/plan/campaign.json results/lifecycles/$ID/runs hwa__$ARM") || exit 8
  SRCPATH=${INFO% *}
  scp -q "$HOST:${SRCPATH%/checkpoints/*}/metrics.jsonl" "$LOCAL/hwa-metrics/$ARM.jsonl" || exit 8
done
ls -la "$LOCAL/collected" "$LOCAL/hwa-metrics"
echo "$(date -Is) [$ID] DONE"
