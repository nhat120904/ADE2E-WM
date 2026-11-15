#!/bin/bash
#SBATCH --job-name=delete-old-copies
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Delete the old home copies data.__old / exp.__old after the move to /mnt/data (user request 2026-11-12). Verifies again
# first: a dry run must find no file of the old copy missing at the destination or newer than the destination's file
# (--update: files changed at the destination after the switch are expected and ignored). Deletes nothing otherwise.
set -uo pipefail
SRC=/home/nhatnc129/nhat.nc/ade2e-wm
DST=/mnt/data/nhatnc129/ade2e-wm
hostname; date; df -h $SRC | tail -1
bad=0
for d in data exp; do
  [ -d $SRC/$d.__old ] || { echo "$d.__old missing"; bad=1; continue; }
  [ "$(readlink $SRC/$d)" = "$DST/$d" ] || { echo "$d is not the symlink to $DST/$d"; bad=1; continue; }
  n=$(rsync -a --update --dry-run --itemize-changes $SRC/$d.__old/ $DST/$d/ | grep -v '^\.d' | tee /tmp/diff_$d.txt | wc -l)
  echo "$d: items of the old copy not at the destination: $n"; head -5 /tmp/diff_$d.txt
  [ "$n" = 0 ] || bad=1
done
date
if [ $bad != 0 ]; then echo "NOT deleting"; exit 1; fi
for d in data exp; do
  find $SRC/$d.__old -mindepth 1 -maxdepth 1 -print0 | xargs -0 -r -P 8 -n 1 rm -rf
  rmdir $SRC/$d.__old && echo "deleted $d.__old"
done
date; df -h $SRC | tail -1
