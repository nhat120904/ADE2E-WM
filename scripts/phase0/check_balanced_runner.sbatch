#!/bin/bash
#SBATCH --job-name=p0-check-balanced
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Re-runs the 2-log navhard smoke test through run_pdm_balanced.py and requires a score CSV identical to
# the one produced by the plain devkit entry point (job 62539). Exits non-zero on any difference.
set -euo pipefail
ROOT=/home/nhatnc129/nhat.nc/ade2e-wm
REF=$(ls $ROOT/exp/drivor_nav2_navhard_smoke2logs/*/*.csv | head -1)
export LOGS_FILE=$ROOT/configs/subsets/navhard_smoke_2logs.txt EXP=drivor_nav2_navhard_smoke2logs_balanced
bash $ROOT/scripts/phase0/eval_drivor_navhard.sbatch
NEW=$(ls -t $ROOT/exp/$EXP/*/*.csv | head -1)
source $ROOT/scripts/phase0/env.sh
python $ROOT/scripts/phase0/compare_score_csv.py "$REF" "$NEW"
