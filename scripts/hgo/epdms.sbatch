#!/bin/bash
#SBATCH --job-name=hgo-epdms
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Official NAVSIM v2.2 one-stage EPDMS on navtest (non-reactive) of HGO runs: dump the selected trajectories
# (scripts/hgo/dump_plans.py), then the unmodified devkit with the precomputed-trajectory agent.
# usage: sbatch --export=ALL,NAMES="H32_s0 R64_s0",TAG=main scripts/hgo/epdms.sbatch   (optional LOGS_FILE for a smoke)
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
RUNS=$ADE2E_ROOT/exp/hgo/runs
cd $ADE2E_ROOT
EXTRA=()
DUMP=()
if [ -n "${LOGS_FILE:-}" ]; then
  EXTRA+=("train_test_split.scene_filter.log_names=[$(sed '/^$/d; s/.*/"&"/' "$LOGS_FILE" | paste -sd,)]")
  DUMP+=(--logs "$LOGS_FILE")
fi
for N in $NAMES; do
  python scripts/hgo/dump_plans.py --run $RUNS/$N "${DUMP[@]}"
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for N in $NAMES; do
  echo "=== $N $(date -u +%T)"
  python scripts/phase0/run_pdm_balanced.py one_stage "$ADE2E_HYDRA_SEARCHPATH" \
      train_test_split=navtest traffic_agents=non_reactive agent=precomputed \
      agent.plans_path=$RUNS/$N/navtest_plans.npz experiment_name=hgo_epdms/$TAG/$N \
      metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/navtest_navsim2.2_np1.26.4 \
      worker.threads_per_node=$SLURM_CPUS_PER_TASK "${EXTRA[@]}"
  cp "$(ls -t $NAVSIM_EXP_ROOT/hgo_epdms/$TAG/$N/*/*.csv | head -1)" $RUNS/$N/navtest_epdms_$TAG.csv
done
echo "=== done $(date -u +%T)"
