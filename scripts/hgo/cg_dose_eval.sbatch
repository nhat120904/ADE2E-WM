#!/bin/bash
#SBATCH --job-name=cg-dose-eval
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Exposure-dose check (docs/CG_SPLIT.md §10.2): official NAVSIM v2.2 one-stage EPDMS of a DrivoR CG checkpoint on the
# fold's TRAINING windows selected by `scripts/hgo/cg_dose.py prep` (run once before, in a CPU job). Metric caches: the
# merged index over the pool written by cg_drivor_prep.py (exp/cg_drivor/train_metric_cache, absolute paths). CPU agent
# inference, unmodified devkit (as cg_drivor_eval.sbatch).
# usage: sbatch --export=ALL,CKPT=<absolute path of last.ckpt>,NAME=<run name>[,W=812,H=476] scripts/hgo/cg_dose_eval.sbatch
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
cd $ADE2E_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SPLIT=${SPLIT:-split_v1}; FOLD=${FOLD:-turn_vru}; W=${W:-812}; H=${H:-476}
CKPT=${CKPT:?checkpoint}; NAME=${NAME:?run name}
[ -f "$CKPT" ] || { echo "no checkpoint $CKPT"; exit 1; }
[ -f configs/navsim_cg/train_test_split/scene_filter/${SPLIT}_${FOLD}_dose.yaml ] || { echo "run cg_dose.py prep"; exit 1; }
OUT=exp/cg_eval/$NAME
mkdir -p $OUT
python scripts/phase0/run_pdm_balanced.py one_stage \
    "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$ADE2E_ROOT/configs/navsim,file://$ADE2E_ROOT/configs/navsim_cg]" \
    traffic_agents=non_reactive agent=drivor_nav2 agent.checkpoint_path=$CKPT "agent.config.image_size=[$W,$H]" \
    worker.threads_per_node=$SLURM_CPUS_PER_TASK \
    train_test_split=trainval train_test_split/scene_filter=${SPLIT}_${FOLD}_dose experiment_name=cg_eval/$NAME/dose \
    metric_cache_path=$ADE2E_ROOT/exp/cg_drivor/train_metric_cache
cp "$(ls -t $NAVSIM_EXP_ROOT/cg_eval/$NAME/dose/*/*.csv | awk 'NR==1')" $OUT/dose.csv
echo "rows: $(($(wc -l < $OUT/dose.csv) - 1))"
echo "=== done $(date -u +%T)"
