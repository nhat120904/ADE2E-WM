#!/bin/bash
#SBATCH --job-name=hazard-coverage
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Hazard coverage of the H32 slot state on navtest (scripts/hgo/hazard_coverage.py, METHOD_RATIONALE §8b). CPU only.
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
date
python scripts/hgo/hazard_coverage.py --run ${RUN:-exp/hgo/runs/H32_turn_vru_s0} --max-scenes ${MAX_SCENES:-0} --procs 16
date
