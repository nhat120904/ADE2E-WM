"""Pick the OpenScene trainval camera splits for the Phase 1 pilot (seeded, uniform over the 200 splits).

Writes data/_lists/trainval_pilot_splits.txt and reports how many logs / navtrain scenes they cover.
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N, SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 40, 0
m = json.load(open(ROOT / "data/_lists/openscene_sensor_trainval_0-199.json"))
splits = sorted(int(k.rsplit("_", 1)[1]) for k in m)
chosen = sorted(random.Random(SEED).sample(splits, N))
nav = (ROOT / "third_party/navsim_v2/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml").read_text()
nav_logs = set(re.findall(r"^\s*-\s*'?([0-9]{4}\.[0-9.]+_veh-[0-9]+_[0-9_]+)'?\s*$", nav.split("log_names:")[1].split("tokens:")[0], re.M))
logs = {l for k in chosen for l in m[f"openscene_sensor_trainval_{k}"]}
(ROOT / "data/_lists/trainval_pilot_splits.txt").write_text("\n".join(map(str, chosen)) + "\n")
(ROOT / "data/_lists/trainval_pilot_logs.txt").write_text("\n".join(sorted(logs)) + "\n")
print(f"seed={SEED} splits={chosen}")
print(f"logs {len(logs)} / {len({l for v in m.values() for l in v})}; navtrain logs {len(logs & nav_logs)} / {len(nav_logs)}")
