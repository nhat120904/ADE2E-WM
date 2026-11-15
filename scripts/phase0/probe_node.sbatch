#!/bin/bash
#SBATCH --job-name=p0-probe
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Phase 0: check compute-node internet, filesystem and tools (no GPU requested).
hostname; date
timeout 20 curl -sI https://huggingface.co | head -1 || echo "NO_HF"
timeout 20 curl -sI https://github.com | head -1 || echo "NO_GITHUB"
timeout 20 curl -sI https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip | head -3 || echo "NO_S3"
df -h /home/nhatnc129/nhat.nc/ade2e-wm /tmp | cat
ls /home/nhatnc129/.local/bin/uv && /home/nhatnc129/.local/bin/uv --version
nproc; free -g | head -2
