#!/bin/bash
# Run a list of drn_ft.py commands inside one GPU job with a shared deadline.
#   exp002_drn_job.sh <commands-file> <wall-seconds> [margin-seconds=420]
# Env: DRN_CODE (drn archive root, cwd for drn_ft.py), DRN_FT (path to drn_ft.py), DFT_LAB_DIR (labs/domain_ft),
#      PY (python). Each non-empty, non-# line of the commands file is the argument list of one drn_ft.py call.
# `train` lines get --deadline = start + wall - margin; a train that hits it writes resume.pt and exits 3,
# and the job stops there. Resubmit the same commands file: finished trainings (final.json done + state.pt)
# and finished deployment JSONs are skipped, and an interrupted training resumes from resume.pt.
set -u
CMDS=$1
WALL=$2
MARGIN=${3:-420}
START=$(date +%s)
DEADLINE=$((START + WALL - MARGIN))
cd "$DRN_CODE"
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue ;; esac
  # shellcheck disable=SC2086
  set -- $line
  echo "=== $(date -u +%FT%TZ) drn_ft.py $line" >&2
  if [ "$1" = train ]; then
    "$PY" "$DRN_FT" $line --deadline "$DEADLINE"
  else
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "deadline reached before: $line" >&2; exit 3; fi
    "$PY" "$DRN_FT" $line
  fi
  rc=$?
  echo "=== rc=$rc" >&2
  if [ $rc -eq 3 ]; then echo "RESUMABLE: $line" >&2; exit 3; fi
done < "$CMDS"
echo "ALL_DONE" >&2
