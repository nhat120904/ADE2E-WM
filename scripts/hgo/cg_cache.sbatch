#!/bin/bash
#SBATCH --job-name=cg-cache
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Data for the CG split manifests of scripts/hgo/cg_split.py (docs/CG_SPLIT.md §9):
# (1) BEV semantic targets of the pool / extra-test logs not yet in exp/hgo/bev (cheap);
# (2) official NAVSIM v2.2 metric cache of the frames still missing one: extra test frames first (~5.5k), then the
#     training pool (~65k; the pilot frames are already in exp/metric_cache/train_hgo). Separate cache dirs per part.
# Pilot rate: ~8.8k frames / h on 16 CPUs (job 64414) -> ~8-9 h in total.
# usage: NAME=split_v1 sbatch scripts/hgo/cg_cache.sbatch   [PARTS="test train"]
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
cd $ADE2E_ROOT
NAME=${NAME:-split_v1}
L=data/_lists/cg/$NAME
TODO=$L/bev_todo_logs.txt
comm -23 <(cat $L/pool_logs.txt $L/test_extra_logs.txt | sort -u) <(ls exp/hgo/bev | sed 's/\.npz$//' | sort -u) > $TODO
echo "bev logs to build: $(wc -l < $TODO)"
if [ -s $TODO ]; then
    python scripts/hgo/build_bev_cache.py --logs $TODO --pkl-dir data/navsim_logs/trainval --out exp/hgo/bev \
        --procs $SLURM_CPUS_PER_TASK | tail -3
fi
SP="hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$ADE2E_ROOT/configs/navsim,file://$ADE2E_ROOT/configs/navsim_cg]"
for PART in ${PARTS:-test train}; do
    CACHE=$NAVSIM_EXP_ROOT/metric_cache/cg_${NAME}_${PART}
    echo "== $PART -> $CACHE  $(date)"
    python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
        train_test_split=trainval train_test_split/scene_filter=${NAME}_${PART} "$SP" \
        metric_cache_path=$CACHE worker.threads_per_node=$SLURM_CPUS_PER_TASK
    echo "cached $PART: $(find $CACHE -name 'metric_cache.pkl' | wc -l)  $(date)"
done
