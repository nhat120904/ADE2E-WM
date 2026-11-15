#!/bin/bash
#SBATCH --job-name=hgo-smoke-v3
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# HGO v3 smoke (engineering check only, never a result): V3 on the main CG fold (withheld manifest + L2 holdout) for a
# few dozen steps of both stages on pool logs that have a metric cache (online PDM labels), logging time / memory per
# step; then the navtest dump path with the t-1 inputs on one log. Unit tests run separately (CPU job).
# usage: sbatch scripts/hgo/smoke_v3.sbatch   [--export=ALL,STEPS=45,BS=16,LOGS=6,THR=0.0 (THR=0 exercises the slots of an untrained detector)]
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
OUT=exp/hgo/runs/smoke_V3_${SLURM_JOB_ID}
mkdir -p $OUT
# pool logs of the pilot training set (navtrain tokens with a metric cache in exp/metric_cache/train_hgo)
comm -12 <(sort data/_lists/phase1_train_logs.txt) <(sort data/_lists/cg/split_v1/pool_logs.txt) | awk -v n=${LOGS:-6} 'NR<=n' > $OUT/logs.txt
python scripts/hgo/train.py --variant V3 --out $OUT --fold data/_lists/cg/split_v1/turn_vru --fold-variant withheld \
  --holdout-cell "turn|vru" --train-logs $OUT/logs.txt --max-steps ${STEPS:-45} --bs ${BS:-16} --workers 10 \
  --pdm-workers 6 --log-every 1 --pdm-props ${PROPS:-16} --val-batches 2 --max-logs ${LOGS:-6} --det-thr ${THR:-0.3}
python scripts/hgo/dump_plans.py --run $OUT --max-logs 1 --out smoke_plans.npz --raw-det 50
echo "=== done $(date -u +%T)"
