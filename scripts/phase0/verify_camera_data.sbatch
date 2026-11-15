#!/bin/bash
#SBATCH --job-name=p0-verify-cams
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
python $ADE2E_ROOT/scripts/phase0/verify_camera_data.py --data-root $OPENSCENE_DATA_ROOT --navsim-root $NAVSIM_DEVKIT_ROOT \
    --pilot-logs $ADE2E_ROOT/data/_lists/trainval_pilot_logs.txt --out $ADE2E_ROOT/docs/phase0/verify_camera_data.json \
    --workers $SLURM_CPUS_PER_TASK
