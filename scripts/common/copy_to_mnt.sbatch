#!/bin/bash
#SBATCH --job-name=copy-data-mnt
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Copy data/ and exp/ of the project to /mnt/data/nhatnc129/ade2e-wm/ (user request 2026-11-12). COPY ONLY: nothing is
# deleted here. The switch (rename the sources, symlink data/ and exp/ to the copies) and the deletion of the old copies
# happen later, after a final incremental pass and a check. Re-running resumes (rsync skips files already copied).
set -euo pipefail
SRC=/home/nhatnc129/nhat.nc/ade2e-wm
DST=/mnt/data/nhatnc129/ade2e-wm
hostname; date
mkdir -p "$DST"
test -w "$DST"
df -h "$DST" "$SRC"
cd "$SRC"
# parallel pass over the third-level directories (e.g. one per log under sensor_blobs/<split>/), then a full pass that
# picks up everything else (files at depth <= 3, anything written meanwhile)
find data exp -mindepth 3 -maxdepth 3 -type d -print0 \
  | xargs -0 -r -P 8 -I{} rsync -a --relative {} "$DST/"
echo "parallel pass done $(date)"
rsync -a data exp "$DST/"
echo "full pass done $(date)"
# check: a dry run must find nothing left to copy (files written by a running job after this point excepted)
left=$(rsync -a --dry-run --itemize-changes data exp "$DST/" | wc -l)
echo "items still differing: $left"
df -h "$DST"
date
