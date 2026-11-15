#!/bin/bash
#SBATCH --job-name=cmd
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=00:45:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# usage: sbatch [--job-name=..] scripts/common/run_cmd.sbatch '<bash commands>'   (runs in $ADE2E_ROOT with the env)
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash -euo pipefail -c "$1"
