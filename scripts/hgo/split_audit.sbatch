#!/bin/bash
#SBATCH --job-name=hgo-audit
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# VRU interaction audit of the turn_vru split (docs/METHOD_RATIONALE.md §4b, tests 3-4): GT annotations of the navtest and
# pilot training logs -> data/_lists/hgo_vru_audit_{test,train}.csv and docs/hgo_vru_audit.md. CPU only.
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
python scripts/hgo/split_audit.py --sets test train --procs $SLURM_CPUS_PER_TASK
