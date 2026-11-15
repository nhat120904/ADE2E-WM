"""Official per-scene EPDMS csvs (NAVSIM v2.2 one-stage) averaged per group of the compositional split.

Groups: all, seen, held_out (data/_lists/hgo_split_<name>.json holdout on data/_lists/hgo_factors_test.csv), per maneuver.
Every csv is restricted to the tokens that are valid in all csvs given, so the rows compare the same scenes; the number of
scenes dropped per csv (missing or invalid, e.g. navtest scenes without an HGO window) is printed.

usage: python scripts/hgo/epdms_groups.py --split turn_vru NAME=path.csv [NAME=path.csv ...] [--out groups.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TERMS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
         "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
         "history_comfort", "two_frame_extended_comfort", "score"]
SHORT = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC", "EPDMS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="NAME=path.csv")
    ap.add_argument("--split", default="turn_vru")
    ap.add_argument("--factors", default="data/_lists/hgo_factors_test.csv")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    runs = {}
    for s in a.csvs:
        name, path = s.split("=", 1)
        d = pd.read_csv(ROOT / path if not Path(path).is_absolute() else path)
        d = d[d.token != "average_all_frames"].set_index("token")
        runs[name] = d
    valid = [set(d.index[d.valid.astype(str) == "True"]) for d in runs.values()]
    common = sorted(set.intersection(*valid))
    fac = pd.read_csv(ROOT / a.factors).set_index("token")
    split = json.loads((ROOT / f"data/_lists/hgo_split_{a.split}.json").read_text())
    common = [t for t in common if t in fac.index]
    f = fac.loc[common]
    held = np.zeros(len(f), bool)
    for h in split["holdout"]:
        m = np.ones(len(f), bool)
        for k, v in zip(["maneuver", "vru", "density"], h.split(":")):
            if v != "*":
                m &= (f[k] == v).to_numpy()
        held |= m
    groups = {"all": np.ones(len(f), bool), "seen": ~held, "held_out": held}
    for m in ["straight", "left", "right", "stop"]:
        groups[f"maneuver={m}"] = (f["maneuver"] == m).to_numpy()
    turn = f["maneuver"].isin(["left", "right"]).to_numpy()
    vru = (f["vru"] == "vru").to_numpy()
    groups["turn,no_vru"] = turn & ~vru   # seen parent factor (turn) without the held-out pairing
    groups["straight/stop,vru"] = ~turn & vru  # seen parent factor (VRU) without the held-out pairing
    res = {"split": a.split, "holdout": split["holdout"], "common_scenes": len(common),
           "dropped": {n: int(len(d) - len(common)) for n, d in runs.items()}, "groups": {}}
    print(json.dumps({k: res[k] for k in ["common_scenes", "dropped"]}))
    print("group".ljust(18), "n".rjust(6), " ".join(f"{n}:{s}".rjust(14) for n in runs for s in ["EPDMS"]))
    for g, m in groups.items():
        tok = [t for t, k in zip(common, m) if k]
        res["groups"][g] = {"scenes": len(tok)}
        for n, d in runs.items():
            r = d.loc[tok, TERMS].astype(float).mean()
            res["groups"][g][n] = {s: float(r[t]) for s, t in zip(SHORT, TERMS)}
        print(g.ljust(18), str(len(tok)).rjust(6),
              " ".join(f"{res['groups'][g][n]['EPDMS']:.4f}".rjust(14) for n in runs))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
