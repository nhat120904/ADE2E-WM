#!/bin/bash
#SBATCH --job-name=hgo-prep
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# CPU preparation for HGO training (docs/METHOD.md):
# (1) BEV semantic targets for the 238 pilot logs (train + val) and the navtest logs;
# (2) factor tables for the compositional splits; (3) official NAVSIM v2.2 metric cache of every frame of the pilot logs
# with 4 history and 10 future frames (all_scenes filter, frame_interval 1), so every training window gets PDM labels.
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
cd $ADE2E_ROOT
python -m pytest -q -p no:cacheprovider tests/hgo -rs 2>&1 | tail -3
python scripts/hgo/build_bev_cache.py --logs data/_lists/phase1_trainval_logs.txt --pkl-dir data/navsim_logs/trainval \
    --out exp/hgo/bev --procs $SLURM_CPUS_PER_TASK | tail -3
python scripts/hgo/build_bev_cache.py --logs data/_lists/phase1_test_logs.txt --pkl-dir data/navsim_logs/test \
    --out exp/hgo/bev --procs $SLURM_CPUS_PER_TASK | tail -3
python scripts/hgo/make_factors.py | tail -40
echo "bev files: $(ls exp/hgo/bev/*.npz | wc -l)"
CACHE=$NAVSIM_EXP_ROOT/metric_cache/train_hgo
LOGS="[$(sed '/^$/d; s/.*/"&"/' data/_lists/phase1_trainval_logs.txt | paste -sd,)]"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=trainval "train_test_split.scene_filter.log_names=$LOGS" train_test_split.scene_filter.frame_interval=1 \
    metric_cache_path=$CACHE worker.threads_per_node=$SLURM_CPUS_PER_TASK
echo "cached: $(find $CACHE -name '*.pkl' | wc -l)"
