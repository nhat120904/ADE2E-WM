#!/bin/bash
#SBATCH --job-name=cg-samples
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Sample figures of the CG folds for reports (scripts/hgo/cg_samples.py; the nuPlan map needs more memory than the
# login node allows). usage: sbatch scripts/hgo/cg_samples.sbatch ["<extra args>"]
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
export OMP_NUM_THREADS=4
python scripts/hgo/cg_samples.py ${1:-}
