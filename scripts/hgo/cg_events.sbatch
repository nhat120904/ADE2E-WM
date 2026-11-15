#!/bin/bash
#SBATCH --job-name=cg-events
#SBATCH --partition=main
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Event labels for the redesigned compositional split (scripts/hgo/cg_events.py, docs/CG_SPLIT.md): navtest logs, then all
# 1,310 trainval logs, with nuPlan intersection labels. GT annotations only; CPU job.
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
date
python scripts/hgo/cg_events.py --split test --procs 12 --map
date
python scripts/hgo/cg_events.py --split trainval --procs 12 --map
date
