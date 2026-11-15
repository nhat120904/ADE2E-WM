#!/bin/bash
#SBATCH --job-name=cg-drivor
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=22:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# DrivoR baseline on a CG fold (step 1 of the go / no-go, docs/CG_SPLIT.md §10): the vendored DrivoR Nav2 agent and its
# NAVSIM-v2 README recipe (AdamW, base_lr 2e-4 at batch 64 scaled by sqrt, ref_num 4, ...), trained with the devkit
# run_training.py (scripts/hgo/cg_drivor_run_training.py: loss-dict logging of DrivoR's fork, periodic checkpoints,
# guarded resume, sample-count assert; train labels from the v2.2 metric cache via ade2e_wm.hgo.pdm, same as HGO) on the
# balanced, cache-filtered fold manifest (B-L1: expert windows only), 1 GPU, batch 16, no feature cache (features on the
# fly, nothing written to disk). Cheaper than the paper: image W x H (default 812 x 476) and EPOCHS (default 6), identical
# for the withheld and the exposed run, so the within-method gap stays valid; absolute scores are not comparable with the
# paper. Throughput (smoke 65453, Ray 16 / workers 8): 2.14 s / step, 84 % of it CPU PDM scoring -> ~18 h for 6 epochs.
# Run scripts/hgo/cg_drivor_prep.py once (CPU job) before; this job only reads its outputs. Resubmitting the same command
# resumes from the newest checkpoint of the run (refused if the arguments or manifests changed).
# usage: sbatch --export=ALL,VARIANT=withheld scripts/hgo/cg_drivor_train.sbatch
#        smoke (200 batches): sbatch --time=00:45:00 --export=ALL,VARIANT=withheld,SMOKE=1 scripts/hgo/cg_drivor_train.sbatch
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
cd $ADE2E_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
SPLIT=${SPLIT:-split_v1}; FOLD=${FOLD:-turn_vru}; VARIANT=${VARIANT:?withheld or exposed}
W=${W:-812}; H=${H:-476}; EPOCHS=${EPOCHS:-6}; SEED=${SEED:-0}
export NAVSIM_EXP_ROOT=$ADE2E_ROOT/exp/cg_drivor   # the agent reads $NAVSIM_EXP_ROOT/train_metric_cache
export DRIVOR_RAY_THREADS=${RAY_THREADS:-16}   # Ray pool of the online PDM scoring (DrivoR: 8; 16 = one sample per thread)
SFY=configs/navsim_cg/train_test_split/scene_filter/${SPLIT}_${FOLD}_${VARIANT}.yaml
N=$(cat $NAVSIM_EXP_ROOT/n_${SPLIT}_${FOLD}_withheld.txt)
[ "$N" = "$(cat $NAVSIM_EXP_ROOT/n_${SPLIT}_${FOLD}_exposed.txt)" ] || { echo "withheld / exposed sizes differ"; exit 1; }
export DRIVOR_EXPECT_N=$N DRIVOR_CKPT_EVERY=${CKPT_EVERY:-500}
# resume identity: manifests, log lists, agent / training configs, runner, agent + label source
ID=($SFY configs/navsim_cg/cg_logs/${SPLIT}.yaml configs/navsim/agent/drivor_nav2.yaml
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/training/default_training.yaml scripts/hgo/cg_drivor_run_training.py
    src/ade2e_wm/hgo/pdm.py $(find src/ade2e_wm/agents/drivor -name "*.py" | sort))
export DRIVOR_ID_FILES=$(IFS=,; echo "${ID[*]}")
NAME=drivor_${FOLD}_${VARIANT}_${W}x${H}_e${EPOCHS}_s${SEED}
LIMITS=(trainer.params.limit_val_batches=50)
if [ "${SMOKE:-0}" = 1 ]; then
  EPOCHS=1; NAME=smoke_${NAME}_${SLURM_JOB_ID}
  LIMITS=(trainer.params.limit_train_batches=200 trainer.params.limit_val_batches=5)
fi
# step budget: a mid-epoch resume would otherwise overshoot it (docs/reviews/drivor_cg_followup_2026-11-14.md)
STEPS=$(( EPOCHS * ( (N + 15) / 16 ) )); [ "${SMOKE:-0}" = 1 ] && STEPS=200
export DRIVOR_EXPECT_STEPS=$STEPS
export DRIVOR_RUN_DIR=$NAVSIM_EXP_ROOT/$NAME
echo "=== start $(date -u +%T) $NAME scenes=$N steps=$STEPS ray_threads=$DRIVOR_RAY_THREADS workers=${WORKERS:-8}"
# every output line gets a time stamp (the agent prints one line per 100 batches): step time from the log
python -u scripts/hgo/cg_drivor_run_training.py \
    "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$ADE2E_ROOT/configs/navsim,file://$ADE2E_ROOT/configs/navsim_cg]" \
    agent=drivor_nav2 +cg_logs=$SPLIT experiment_name=$NAME \
    train_test_split=trainval train_test_split/scene_filter=${SPLIT}_${FOLD}_${VARIANT} \
    cache_path=null use_cache_without_dataset=false force_cache_computation=false \
    trainer.params.max_epochs=$EPOCHS +trainer.params.max_steps=$STEPS trainer.params.strategy=auto "${LIMITS[@]}" \
    dataloader.params.prefetch_factor=1 dataloader.params.batch_size=16 dataloader.params.num_workers=${WORKERS:-8} \
    agent.lr_args.name=AdamW agent.lr_args.base_lr=0.0002 agent.num_gpus=1 agent.batch_size=16 \
    agent.progress_bar=false agent.scheduler_args.num_epochs=$EPOCHS agent.scheduler_args.dataset_size=$N \
    "agent.config.image_size=[$W,$H]" seed=$SEED 2>&1 | while IFS= read -r l; do printf '%(%T)T %s\n' -1 "$l"; done
echo "=== end $(date -u +%T)"
find $NAVSIM_EXP_ROOT/$NAME -name "*.ckpt" | sort
