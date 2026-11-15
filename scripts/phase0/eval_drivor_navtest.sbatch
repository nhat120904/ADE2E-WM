#!/bin/bash
#SBATCH --job-name=p0-eval-navtest
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Official NAVSIM v2.2 one-stage EPDMS on navtest (non-reactive log-replay agents, the devkit default)
# with the released DrivoR Nav2 checkpoint. CPU agent inference, one Ray task per log (run_pdm_balanced.py).
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CKPT=${CKPT:-$ADE2E_ROOT/checkpoints/drivor/drivor_Nav2_10epochs.pth}
EXP=${EXP:-drivor_nav2_navtest_full}
python $ADE2E_ROOT/scripts/phase0/run_pdm_balanced.py one_stage \
    "$ADE2E_HYDRA_SEARCHPATH" \
    train_test_split=navtest \
    traffic_agents=non_reactive \
    agent=drivor_nav2 \
    agent.checkpoint_path=$CKPT \
    experiment_name=$EXP \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/navtest_navsim2.2_np1.26.4 \
    worker.threads_per_node=$SLURM_CPUS_PER_TASK
