#!/bin/bash
#SBATCH --job-name=cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# usage: sbatch [--job-name=..] scripts/common/run_cpu.sbatch '<bash commands>'   (CPU only; runs in $ADE2E_ROOT with the env)
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
export OMP_NUM_THREADS=4
bash -euo pipefail -c "$1"
