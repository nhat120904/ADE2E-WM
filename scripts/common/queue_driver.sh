#!/bin/bash
# Submit queued Slurm jobs so that this user never has more than MAX_JOBS jobs in Slurm (running + pending): user rule,
# 2026-11-01. Runs on the login node; it only calls squeue / sacct / sbatch once a minute.
#
# Each <queue dir>/*.lane file lists jobs, one per line, as sbatch arguments ('#' lines are comments). A lane submits its
# next line only after its previous job has left Slurm, so a lane holds at most one job; lines appended later are picked
# up. A lane containing the line '#stop_on_fail' stops at the first job that does not end COMPLETED. A line '@mark X'
# creates <queue dir>/X.mark once every earlier line of its lane has finished; a line '@wait X' holds its lane until then.
# State: <lane>.pos (lines submitted), <lane>.cur (job in Slurm), driver.log. Touch <queue dir>/STOP to end the driver.
#
# usage: nohup setsid scripts/common/queue_driver.sh <queue dir> > <queue dir>/driver.out 2>&1 &
set -u
Q=$(realpath "$1")
MAX_JOBS=${MAX_JOBS:-2}
cd /home/nhatnc129/nhat.nc/ade2e-wm
log() { echo "$(date -u +%FT%TZ) $*" >> "$Q/driver.log"; }
log "driver start pid $$ max_jobs $MAX_JOBS"
while [ ! -e "$Q/STOP" ]; do
  for L in "$Q"/*.lane; do
    [ -e "$L" ] || continue
    name=$(basename "$L" .lane)
    [ -e "$L.stopped" ] && continue
    cur=$(cat "$L.cur" 2>/dev/null || true)
    if [ -n "$cur" ]; then
      squeue -h -j "$cur" 2>/dev/null | grep -q . && continue
      st=$(sacct -j "$cur" -X -n -P --format=State 2>/dev/null | head -1)
      case "$st" in ""|PENDING*|RUNNING*|COMPLETING*|CONFIGURING*|REQUEUED*) continue ;; esac
      log "$name job $cur ended $st"
      : > "$L.cur"
      if [[ "$st" != COMPLETED* ]] && grep -qx '#stop_on_fail' "$L"; then
        log "$name stopped after $cur ($st)"; touch "$L.stopped"; continue
      fi
    fi
    pos=$(cat "$L.pos" 2>/dev/null || echo 0)
    line=$(grep -v '^#' "$L" | grep . | sed -n "$((pos + 1))p")
    [ -z "$line" ] && continue
    case "$line" in  # directives, no job: '@mark X' records that the lane got here, '@wait X' blocks until some lane marked X
      "@mark "*) touch "$Q/${line#@mark }.mark"; echo $((pos + 1)) > "$L.pos"; log "$name mark ${line#@mark }"; continue ;;
      "@wait "*) [ -e "$Q/${line#@wait }.mark" ] || continue
                 echo $((pos + 1)) > "$L.pos"; log "$name passed wait ${line#@wait }"; continue ;;
    esac
    n=$(squeue -h -u "$USER" 2>/dev/null | wc -l)
    [ "$n" -ge "$MAX_JOBS" ] && continue
    if jid=$(eval sbatch --parsable $line 2>>"$Q/driver.log"); then
      echo "$jid" > "$L.cur"; echo $((pos + 1)) > "$L.pos"; log "$name submitted line $((pos + 1)) as $jid: $line"
    else
      log "$name submit FAILED: $line"; touch "$L.stopped"
    fi
  done
  sleep 60
done
log "driver stop"
