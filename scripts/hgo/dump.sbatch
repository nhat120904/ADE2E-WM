#!/bin/bash
#SBATCH --job-name=hgo-dump
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# GPU part of the navtest evaluation: run HGO checkpoints on navtest (selected plan + all proposals + scorer outputs).
# The CPU part (official EPDMS, oracle / regret) is scripts/hgo/eval_cpu.sbatch, submitted after this job ends.
# usage: sbatch --export=ALL,NAMES="H32_turn_vru_s0" scripts/hgo/dump.sbatch
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
for N in $NAMES; do
  # one-log smoke first, so a code error fails in a minute instead of after the full pass
  python scripts/hgo/dump_plans.py --run exp/hgo/runs/$N --workers 14 --max-logs 1 --out navtest_plans_smoke.npz
  python scripts/hgo/dump_plans.py --run exp/hgo/runs/$N --workers 14
done
