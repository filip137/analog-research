#!/bin/bash
# Local driver for exp-002 DRN chains on Jean Zay: drn_chain.sh <log> <chain>...
# Every 3 min: a chain with no queued/running job and no ALL_DONE in its newest slurm .err is (re)submitted
# (resume from resume.pt; finished steps are skipped) if fewer than 10 jobs are submitted. Exits (printing why)
# when every chain shows ALL_DONE, or when any step logged a non-0/3 rc, or a chain was submitted 5 times.
J=/lustre/fsn1/projects/rech/umg/ucy17uy/domain-ft-exp002-20261108
LOG=$1; shift
declare -A N
while true; do
  st=$(timeout 90 ssh -o BatchMode=yes jean-zay "bash -lc 'squeue -h -u \$USER -o %.80j | sed \"s/^ *//\"; echo ===; cd $J/slurm; for c in $*; do f=\$(ls -t dft2-drn-\$c-*.err 2>/dev/null | head -1); echo \"\$c \$(cat \${f:-/dev/null} 2>/dev/null | grep -c ALL_DONE) \$(cat \${f:-/dev/null} 2>/dev/null | grep -cE \"=== rc=[^03]\") \$f\"; done'" 2>/dev/null) || { sleep 60; continue; }
  names=$(echo "$st" | sed '/===/q' | grep -v ===); njobs=$(echo "$names" | grep -c .)
  alldone=1
  while read -r c done fail f; do
    [ -z "$c" ] && continue
    [ "$fail" != 0 ] && { echo "$(date +%T) STEP_FAILED $c $f" | tee -a $LOG; exit 1; }
    [ "$done" != 0 ] && continue
    alldone=0
    echo "$names" | grep -qx "dft2-drn-$c" && continue
    [ "$njobs" -ge 10 ] && continue
    N[$c]=$(( ${N[$c]:-0} + 1 )); [ ${N[$c]} -gt 5 ] && { echo "$(date +%T) TOO_MANY_SUBMITS $c" | tee -a $LOG; exit 1; }
    id=$(timeout 90 ssh -o BatchMode=yes jean-zay "cd $J && bash -lc 'sbatch --parsable -A umg@v100 -p gpu_p13 --qos=qos_gpu-dev -C v100-32g -N1 --gres=gpu:1 --ntasks=1 --cpus-per-task=10 --hint=nomultithread -t 02:00:00 -J dft2-drn-$c -o $J/slurm/%x-%j.out -e $J/slurm/%x-%j.err $J/run_drn.slurm $J $J/drn/cmds/$c.txt 7200'")
    echo "$(date +%T) submitted $c job $id" >> $LOG; njobs=$((njobs + 1))
  done < <(echo "$st" | sed '1,/===/d')
  [ $alldone = 1 ] && { echo "$(date +%T) ALL_CHAINS_DONE" | tee -a $LOG; exit 0; }
  sleep 180
done
