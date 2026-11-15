#!/bin/bash
#SBATCH --job-name=hgo-eval
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# CPU part of the navtest evaluation (needs <run>/navtest_plans.npz from scripts/hgo/dump.sbatch):
# official NAVSIM v2.2 one-stage EPDMS (non-reactive, unmodified devkit, precomputed-trajectory agent), then the
# proposal oracle / selection regret analysis (scripts/hgo/oracle_props.py).
# usage: sbatch --export=ALL,NAMES="H32_turn_vru_s0",TAG=main,SPLIT=turn_vru scripts/hgo/eval_cpu.sbatch
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
RUNS=$ADE2E_ROOT/exp/hgo/runs
cd $ADE2E_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for N in $NAMES; do
  echo "=== EPDMS $N $(date -u +%T)"
  python scripts/phase0/run_pdm_balanced.py one_stage "$ADE2E_HYDRA_SEARCHPATH" \
      train_test_split=navtest traffic_agents=non_reactive agent=precomputed \
      agent.plans_path=$RUNS/$N/navtest_plans.npz experiment_name=hgo_epdms/$TAG/$N \
      metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/navtest_navsim2.2_np1.26.4 \
      worker.threads_per_node=$SLURM_CPUS_PER_TASK
  cp "$(ls -t $NAVSIM_EXP_ROOT/hgo_epdms/$TAG/$N/*/*.csv | head -1)" $RUNS/$N/navtest_epdms_$TAG.csv
  echo "=== oracle $N $(date -u +%T)"
  python scripts/hgo/oracle_props.py --run $RUNS/$N --split ${SPLIT:-turn_vru} --workers $SLURM_CPUS_PER_TASK \
    || echo "oracle analysis failed for $N"
done
echo "=== done $(date -u +%T)"
