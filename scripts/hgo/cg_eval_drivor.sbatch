#!/bin/bash
#SBATCH --job-name=cg-eval-drivor
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Step 0 of the CG go / no-go (headroom, not the exposure gap): the RELEASED DrivoR Nav2 checkpoint (trained on all of
# navtrain, so it has seen the held-out combinations and part of the CG-test sessions) scored with the official NAVSIM
# v2.2 one-stage EPDMS on the extra CG test frames of split v1 (CPU agent inference, unmodified devkit, as
# scripts/phase0/eval_drivor_navtest.sbatch); its existing navtest per-scene csv is reused. Then per test group of each
# fold (scripts/hgo/cg_group_scores.py).
# usage: sbatch scripts/hgo/cg_eval_drivor.sbatch
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
mkdir -p $RAY_TMPDIR
cd $ADE2E_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
L=data/_lists/cg/split_v1
OUT=exp/cg_eval/drivor_pub_split_v1
MC=$NAVSIM_EXP_ROOT/metric_cache/cg_split_v1_test_all
CKPT=$ADE2E_ROOT/checkpoints/drivor/drivor_Nav2_10epochs.pth
NAVTEST_CSV=$ADE2E_ROOT/exp/drivor_nav2_navtest_full/2026.10.28.12.55.37/2026.10.28.18.44.05.csv
mkdir -p $OUT $MC/metadata
# scene filter of all extra test frames, and one metric-cache index over the two caches that hold them (absolute paths)
python - <<'EOF'
from pathlib import Path
L = Path("data/_lists/cg/split_v1")
toks = L.joinpath("test_extra_tokens.txt").read_text().split()
logs = L.joinpath("test_extra_logs.txt").read_text().split()
y = ["_target_: navsim.common.dataclasses.SceneFilter", "_convert_: 'all'", "num_history_frames: 4",
     "num_future_frames: 10", "frame_interval: 1", "has_route: true", "max_scenes: null", "log_names:"]
y += [f"  - '{l}'" for l in logs] + ["tokens:"] + [f"  - '{t}'" for t in toks]
Path("configs/navsim_cg/train_test_split/scene_filter/split_v1_test_all.yaml").write_text("\n".join(y) + "\n")
want, rows = set(toks), {}
for d in ["exp/metric_cache/cg_split_v1_test", "exp/metric_cache/train_hgo"]:
    for p in sorted(Path(d, "metadata").glob("*.csv")):
        for line in p.read_text().splitlines()[1:]:
            t = line.split("/")[-2]
            if t in want and t not in rows:
                rows[t] = line
Path("exp/metric_cache/cg_split_v1_test_all/metadata/index.csv").write_text("file_name\n" + "\n".join(rows.values()) + "\n")
print(f"extra test tokens {len(toks)} logs {len(logs)} with metric cache {len(rows)}")
EOF
python scripts/phase0/run_pdm_balanced.py one_stage \
    "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$ADE2E_ROOT/configs/navsim,file://$ADE2E_ROOT/configs/navsim_cg]" \
    train_test_split=trainval train_test_split/scene_filter=split_v1_test_all \
    traffic_agents=non_reactive agent=drivor_nav2 agent.checkpoint_path=$CKPT \
    experiment_name=cg_eval/drivor_pub_extra metric_cache_path=$MC worker.threads_per_node=$SLURM_CPUS_PER_TASK
cp "$(ls -t $NAVSIM_EXP_ROOT/cg_eval/drivor_pub_extra/*/*.csv | awk 'NR==1')" $OUT/extra.csv
for F in turn_vru left_veh_opp; do
  python scripts/hgo/cg_group_scores.py --fold $L/$F $NAVTEST_CSV $OUT/extra.csv --out $OUT/groups_$F.json
done
echo "=== done $(date -u +%T)"
