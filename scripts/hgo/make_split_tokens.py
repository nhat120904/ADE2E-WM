"""Training token list for a compositional split (docs/METHOD.md): every training window except the held-out factor
combinations; optionally keep only a seeded fraction of 'stop' windows (stationary ego, 36 % of the pilot windows).

usage: python scripts/hgo/make_split_tokens.py --name turn_vru --holdout left:vru right:vru [--stop-keep 0.3]
writes data/_lists/hgo_train_<name>.txt and data/_lists/hgo_split_<name>.json (definition + counts)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def held_mask(df, holdout):
    m = np.zeros(len(df), bool)
    for h in holdout:
        conds = dict(zip(["maneuver", "vru", "density"], h.split(":")))
        mm = np.ones(len(df), bool)
        for k, v in conds.items():
            if v != "*":
                mm &= (df[k] == v).to_numpy()
        m |= mm
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--holdout", nargs="*", default=[], help="maneuver:vru[:density], '*' = any")
    ap.add_argument("--stop-keep", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lists = ROOT / "data/_lists"
    tr = pd.read_csv(lists / "hgo_factors_train.csv")
    te = pd.read_csv(lists / "hgo_factors_test.csv")
    keep = ~held_mask(tr, a.holdout)
    if a.stop_keep < 1.0:
        rng = np.random.default_rng(a.seed)
        keep &= ~((tr["maneuver"] == "stop").to_numpy() & (rng.random(len(tr)) >= a.stop_keep))
    (lists / f"hgo_train_{a.name}.txt").write_text("\n".join(tr.loc[keep, "token"]) + "\n")
    th = held_mask(te, a.holdout)
    info = {"holdout": a.holdout, "stop_keep": a.stop_keep, "seed": a.seed, "train_windows": int(keep.sum()),
            "train_removed_holdout": int(held_mask(tr, a.holdout).sum()), "test_heldout": int(th.sum()),
            "test_seen": int((~th).sum())}
    (lists / f"hgo_split_{a.name}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
