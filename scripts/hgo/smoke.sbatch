#!/bin/bash
#SBATCH --job-name=hgo-smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# HGO smoke: unit tests, then a few training steps per variant on 6 navtest logs (navtest metric cache available, so the
# online PDM labels can be timed). Engineering check only: training on test logs is never a result.
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m pytest -q -p no:cacheprovider tests/hgo -rs -s 2>&1 | tail -30
for V in ${VARIANTS:-H32 R64 O64}; do
  echo "=== $V $(date -u +%T)"
  python scripts/hgo/train.py --variant $V --out exp/hgo/runs/smoke_$V \
    --train-logs data/_lists/phase1_test_logs.txt --pkl-dir data/navsim_logs/test --sensors data/sensor_blobs/test \
    --tokens data/_lists/navtest_tokens.txt --metric-cache exp/metric_cache/navtest_navsim2.2_np1.26.4 --bev-dir "" \
    --max-logs 6 --max-steps ${STEPS:-15} --bs ${BS:-16} --workers 10 --pdm-workers ${PDM:-6} --log-every 1 --det-thr ${THR:-0.0} --pdm-props ${PROPS:-16} \
    --val-logs data/_lists/phase1_test_logs.txt --val-batches ${VALB:-2}
done
echo "=== done $(date -u +%T)"
