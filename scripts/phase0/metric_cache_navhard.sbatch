#!/bin/bash
#SBATCH --job-name=p0-mcache-navhard
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Official NAVSIM v2.2 metric caching for navhard_two_stage, re-cached in this env (DrivoR#47).
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
CACHE=$NAVSIM_EXP_ROOT/metric_cache/navhard_two_stage_navsim2.2_np1.26.4
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=navhard_two_stage \
    metric_cache_path=$CACHE \
    worker.threads_per_node=$SLURM_CPUS_PER_TASK
find $CACHE -name '*.pkl' | wc -l
