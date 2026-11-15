#!/bin/bash
#SBATCH --job-name=cg-drivor-eval
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Official NAVSIM v2.2 one-stage EPDMS of a DrivoR checkpoint trained on a CG fold (scripts/hgo/cg_drivor_train.sbatch),
# on the fold's evaluation subset (every cell / same_man / same_partner scene + 1,000 random man_none and other scenes,
# scripts/hgo/cg_drivor_prep.py): navtest part with the official navtest metric cache, CG-test-session part with the split's
# test cache. CPU agent inference, unmodified devkit (as scripts/hgo/cg_eval_drivor.sbatch), then per-group scores.
# usage: sbatch --export=ALL,CKPT=<last.ckpt>,NAME=<run name>[,W=812,H=476] scripts/hgo/cg_drivor_eval.sbatch
#        released checkpoint on the same subset (same scenes, same scoring context; reference for the exposed quality):
#        sbatch --export=ALL,CKPT=checkpoints/drivor/drivor_Nav2_10epochs.pth,NAME=drivor_pub_subset_turn_vru,W=1148,H=672 ...
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
cd $ADE2E_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SPLIT=${SPLIT:-split_v1}; FOLD=${FOLD:-turn_vru}; W=${W:-812}; H=${H:-476}
CKPT=${CKPT:?checkpoint}; NAME=${NAME:?run name}
# the evaluation subset comes from scripts/hgo/cg_drivor_prep.py, run once before (fixed scenes for every model)
for s in navtest cg_test; do
  [ -f configs/navsim_cg/train_test_split/scene_filter/${SPLIT}_${FOLD}_eval_$s.yaml ] || { echo "run cg_drivor_prep.py"; exit 1; }
done
OUT=exp/cg_eval/$NAME
mkdir -p $OUT
SP="hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$ADE2E_ROOT/configs/navsim,file://$ADE2E_ROOT/configs/navsim_cg]"
COMMON=(traffic_agents=non_reactive agent=drivor_nav2 agent.checkpoint_path=$CKPT "agent.config.image_size=[$W,$H]"
        worker.threads_per_node=$SLURM_CPUS_PER_TASK)
python scripts/phase0/run_pdm_balanced.py one_stage "$SP" "${COMMON[@]}" \
    train_test_split=navtest train_test_split/scene_filter=${SPLIT}_${FOLD}_eval_navtest experiment_name=cg_eval/$NAME/navtest \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/navtest_navsim2.2_np1.26.4
cp "$(ls -t $NAVSIM_EXP_ROOT/cg_eval/$NAME/navtest/*/*.csv | awk 'NR==1')" $OUT/navtest.csv
python scripts/phase0/run_pdm_balanced.py one_stage "$SP" "${COMMON[@]}" \
    train_test_split=trainval train_test_split/scene_filter=${SPLIT}_${FOLD}_eval_cg_test experiment_name=cg_eval/$NAME/cg_test \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/cg_split_v1_test_all
cp "$(ls -t $NAVSIM_EXP_ROOT/cg_eval/$NAME/cg_test/*/*.csv | awk 'NR==1')" $OUT/cg_test.csv
python scripts/hgo/cg_group_scores.py --fold data/_lists/cg/$SPLIT/$FOLD $OUT/navtest.csv $OUT/cg_test.csv \
    --out $OUT/groups_$FOLD.json | awk '/source +group/,0'
echo "=== done $(date -u +%T)"
