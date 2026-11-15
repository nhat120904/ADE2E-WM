#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Official NAVSIM v2.2 metric caching for SPLIT (e.g. navtest), cached in this env (numpy 1.26.4).
set -euo pipefail
: "${SPLIT:?}"
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
CACHE=$NAVSIM_EXP_ROOT/metric_cache/${SPLIT}_navsim2.2_np1.26.4
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=$SPLIT metric_cache_path=$CACHE worker.threads_per_node=$SLURM_CPUS_PER_TASK
echo "cached: $(find $CACHE -name '*.pkl' | wc -l)"
