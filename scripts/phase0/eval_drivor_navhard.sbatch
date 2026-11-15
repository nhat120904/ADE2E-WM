#!/bin/bash
#SBATCH --job-name=p0-eval-drivor
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Official NAVSIM v2.2 run_pdm_score.py, one Ray task per log via run_pdm_balanced.py (CPU agent inference) on navhard_two_stage
# with the released DrivoR Nav2 checkpoint. Optional: LOGS_FILE=<file with one log name per line> restricts to a log subset.
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CKPT=${CKPT:-$ADE2E_ROOT/checkpoints/drivor/drivor_Nav2_10epochs.pth}
EXP=${EXP:-drivor_nav2_navhard_full}
EXTRA=()
if [ -n "${LOGS_FILE:-}" ]; then
  EXTRA+=("train_test_split.scene_filter.log_names=[$(sed '/^$/d; s/.*/"&"/' "$LOGS_FILE" | paste -sd,)]")
fi
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT
python $ADE2E_ROOT/scripts/phase0/run_pdm_balanced.py two_stage \
    "$ADE2E_HYDRA_SEARCHPATH" \
    train_test_split=navhard_two_stage \
    agent=drivor_nav2 \
    agent.checkpoint_path=$CKPT \
    experiment_name=$EXP \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/navhard_two_stage_navsim2.2_np1.26.4 \
    worker.threads_per_node=$SLURM_CPUS_PER_TASK \
    "${EXTRA[@]}"
