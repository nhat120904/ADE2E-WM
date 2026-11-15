#!/bin/bash
#SBATCH --job-name=p0-gpu-latency
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python $ADE2E_ROOT/scripts/phase0/gpu_latency_drivor.py
