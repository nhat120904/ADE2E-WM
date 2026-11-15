#!/bin/bash
# Pre-run checks of the DrivoR CG pilot (docs/CG_SPLIT.md §10); run inside a CPU job:
#   sbatch --job-name=cg-check scripts/common/run_cpu.sbatch 'bash scripts/hgo/check_cg_drivor_pilot.sh'
# 1. prep (balanced cached manifests, eval subset over cached scenes); 2. runner patches + resume guard (argv and an
# identity file changed); 3. Hydra composition of the real training command (max_steps = step budget);
# 4. cg_gap.py: identical runs -> gap 0; exposed = withheld + 5 points -> gap +5 with a zero-width interval; a rare group
# absent from some bootstrap draws -> those draws undefined, not 0; 5. shell syntax of the job scripts.
set -euo pipefail
F=data/_lists/cg/split_v1/turn_vru
echo "== 1. prep"
python scripts/hgo/cg_drivor_prep.py --split split_v1 --fold turn_vru
wc -l $F/train_withheld_cached.txt $F/train_exposed_cached.txt $F/balance_dropped.txt $F/eval_subset.txt
comm -12 <(sort $F/balance_dropped.txt) <(sort $F/train_exposed.txt) | wc -l | xargs echo "dropped tokens also in exposed (must be 0):"
echo "== 2. runner"
T=$(mktemp -d)
echo v1 > $T/id.txt
DRIVOR_RUN_DIR=$T/run DRIVOR_ID_FILES=configs/navsim_cg/cg_logs/split_v1.yaml,$T/id.txt python - $T <<'EOF'
import sys
TMP = sys.argv[1]
src = open("scripts/hgo/cg_drivor_run_training.py").read().split('if __name__ == "__main__":')[0]
sys.argv = ["x", "a=1"]
exec(src)
assert resume_checkpoint() is None
assert resume_checkpoint() is None          # same identity: accepted
for what in ("argv", "file"):
    if what == "argv":
        sys.argv = ["x", "a=2"]
    else:
        sys.argv = ["x", "a=1"]
        open(f"{TMP}/id.txt", "w").write("v2\n")
    try:
        resume_checkpoint(); raise AssertionError(f"{what} change accepted")
    except SystemExit as e:
        print(f"{what} change refused:", str(e).splitlines()[0])
print("patched:", DrivoRAgent.get_training_callbacks.__name__, wr.RayDistributedNoTorch.__init__.__name__,
      pl.Trainer.fit.__name__)
EOF
rm -rf $T
echo "== 3. hydra composition (trainer section of the real command)"
export NAVSIM_EXP_ROOT=$ADE2E_ROOT/exp/cg_drivor
N=$(cat $NAVSIM_EXP_ROOT/n_split_v1_turn_vru_withheld.txt); STEPS=$(( 6 * ( (N + 15) / 16 ) ))
echo "N=$N STEPS=$STEPS"
python scripts/hgo/cg_drivor_run_training.py \
    "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$ADE2E_ROOT/configs/navsim,file://$ADE2E_ROOT/configs/navsim_cg]" \
    agent=drivor_nav2 +cg_logs=split_v1 experiment_name=check train_test_split=trainval \
    train_test_split/scene_filter=split_v1_turn_vru_withheld cache_path=null \
    trainer.params.max_epochs=6 +trainer.params.max_steps=$STEPS trainer.params.strategy=auto \
    --cfg job --package trainer | grep -E "max_epochs|max_steps|precision|limit_"
echo "== 4. cg_gap"
S0=exp/drivor_nav2_navtest_full/2026.10.28.12.55.37/2026.10.28.18.44.05.csv
E0=exp/cg_eval/drivor_pub_split_v1/extra.csv
T=$(mktemp -d)
python - $S0 $E0 $T <<'EOF'
import sys, pandas as pd
s0, e0, T = sys.argv[1:]
for src, name in ((s0, "nt"), (e0, "cg")):
    d = pd.read_csv(src)
    d.to_csv(f"{T}/wh_{name}.csv", index=False)
    d = d.assign(score=d["score"] + 0.05, no_at_fault_collisions=d["no_at_fault_collisions"] + 0.05)
    d.to_csv(f"{T}/ex_{name}.csv", index=False)
EOF
echo "-- identical runs (gap must be 0)"
python scripts/hgo/cg_gap.py --fold $F --exposed $T/wh_nt.csv $T/wh_cg.csv --withheld $T/wh_nt.csv $T/wh_cg.csv \
  --n-boot 200 | awk '/^missing/ || /^cell|^same|^man_none|^other/'
echo "-- exposed = withheld + 5 points (gap must be +5, interval [5, 5], 200 valid draws)"
python scripts/hgo/cg_gap.py --fold $F --exposed $T/ex_nt.csv $T/ex_cg.csv --withheld $T/wh_nt.csv $T/wh_cg.csv \
  --n-boot 200 --out $T/gap.json | awk '/^cell|^same|^man_none|^other/'
python - $T/gap.json <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
for g in ("cell", "cell_in4", "cell_not_strict4", "other"):
    print(g, r[g]["gap_EPDMS"], r[g]["gap_EPDMS_ci_session"], r[g]["gap_EPDMS_valid_draws_session"])
print("did cell_in4 - other", r["did_EPDMS_cell_in4_minus_other"])
EOF
echo "-- rare group absent from some draws (must be NaN, not 0)"
python - <<'EOF'
import sys
sys.path.insert(0, "scripts/hgo")
import numpy as np, pandas as pd
from cg_gap import boot_means, ci
df = pd.DataFrame({"log": ["a", "b", "c", "d"], "d": [5.0, 5.0, 5.0, 5.0],
                   "slice_rare": [True, False, False, False], "slice_all": [True] * 4})
B = {"log": np.array([[0, 1, 2, 3], [0, 0, 1, 1], [1, 2, 3, 3]])}
m = boot_means(df, "d", "log", B, ["rare", "all"])
print("rare:", m["rare"].tolist(), "-> ci", ci(m["rare"]), "| did rare-all:", ci(m["rare"] - m["all"]))
assert np.isnan(m["rare"][2]) and ci(m["rare"]) == ([5.0, 5.0], 2)
EOF
rm -rf $T
echo "== 5. syntax"
for s in scripts/hgo/cg_drivor_train.sbatch scripts/hgo/cg_drivor_eval.sbatch; do bash -n $s && echo "ok $s"; done
echo "== done"
