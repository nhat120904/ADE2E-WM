#!/bin/bash
#SBATCH --job-name=hgo-train
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# optional PRE="<commands>" run first (e.g. building the split token list)
# usage: sbatch --job-name=hgo-H32-s0 --export=ALL,ARGS="--variant H32 --seed 0 --out exp/hgo/runs/H32_s0" scripts/hgo/train.sbatch
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if [ -n "${PRE:-}" ]; then bash -euo pipefail -c "$PRE"; fi
python scripts/hgo/train.py $ARGS --workers 10 --pdm-workers 6
