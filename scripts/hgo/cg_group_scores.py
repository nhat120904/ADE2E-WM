"""Official per-scene EPDMS csvs (NAVSIM v2.2 one-stage) averaged per test group of a CG fold (docs/CG_SPLIT.md §5).

Groups (data/_lists/cg/<split>/<fold>/test_groups.csv): cell (held-out maneuver x partner), same_man, same_partner,
man_none, other; every row also split by source (navtest, cg_test = extra frames of the CG-test sessions). Means of the per-scene
EPDMS and of the main sub-scores, with 95 % CIs from a bootstrap over logs; differences cell - other group with a paired
bootstrap over logs. Scenes missing from every csv (no metric cache, invalid) are counted and dropped.

usage: python scripts/hgo/cg_group_scores.py --fold data/_lists/cg/split_v1/turn_vru a.csv b.csv [--out groups.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TERMS = {"EPDMS": "score", "NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
         "DDC": "driving_direction_compliance", "TTC": "time_to_collision_within_bound", "EP": "ego_progress",
         "LK": "lane_keeping", "TLC": "traffic_light_compliance"}
GROUPS = ["cell", "same_man", "same_partner", "man_none", "other"]


def boot_logs(df, col, n=1000, seed=0):
    """95 % CI of the scene mean, resampling logs."""
    g = df.groupby("log")[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    s, c = g["sum"].to_numpy()[idx].sum(1), g["count"].to_numpy()[idx].sum(1)
    return np.percentile(s / c, [2.5, 97.5]).round(2).tolist()


def boot_diff(a, b, col, n=1000, seed=0):
    """95 % CI of mean(a) - mean(b), resampling the union of logs (paired when the groups share logs)."""
    logs = np.array(sorted(set(a["log"]) | set(b["log"])))
    ga = a.groupby("log")[col].agg(["sum", "count"]).reindex(logs, fill_value=0)
    gb = b.groupby("log")[col].agg(["sum", "count"]).reindex(logs, fill_value=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(logs), (n, len(logs)))
    ma = ga["sum"].to_numpy()[idx].sum(1) / np.maximum(ga["count"].to_numpy()[idx].sum(1), 1)
    mb = gb["sum"].to_numpy()[idx].sum(1) / np.maximum(gb["count"].to_numpy()[idx].sum(1), 1)
    return np.percentile(ma - mb, [2.5, 97.5]).round(2).tolist()


def summary(df):
    out = {"scenes": len(df), "logs": int(df["log"].nunique())}
    for k, c in TERMS.items():
        out[k] = round(100 * float(df[c].mean()), 2) if len(df) else None
    if len(df) >= 5:
        out["EPDMS_ci"] = boot_logs(df, "pct")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("csvs", nargs="+", help="per-scene csvs of ONE model (e.g. navtest + extra test frames)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    groups = pd.read_csv(Path(a.fold) / "test_groups.csv")
    sc = pd.concat([pd.read_csv(p) for p in a.csvs])
    sc = sc[sc["token"].isin(groups["token"])].drop_duplicates("token")
    n_invalid = int((~sc["valid"].astype(bool)).sum())
    sc = sc[sc["valid"].astype(bool)]
    df = groups.merge(sc, on="token", how="inner")
    df["pct"] = 100 * df["score"]
    df["nc_pct"] = 100 * df["no_at_fault_collisions"]
    res = {"fold": a.fold, "csvs": a.csvs, "test_scenes": len(groups), "scored": len(df), "invalid": n_invalid,
           "missing": int(len(groups) - len(df) - n_invalid),
           "missing_by_group": (groups[~groups["token"].isin(df["token"])]["group"].value_counts().to_dict())}
    for src in ["all", "navtest", "cg_test"]:
        d = df if src == "all" else df[df["source"] == src]
        res[src] = {g: summary(d[d["group"] == g]) for g in GROUPS}
        cell = d[d["group"] == "cell"]
        res[src]["diff_cell_minus"] = {}
        for g in GROUPS[1:]:
            o = d[d["group"] == g]
            if len(cell) >= 5 and len(o) >= 5:
                res[src]["diff_cell_minus"][g] = {"EPDMS": round(float(cell["pct"].mean() - o["pct"].mean()), 2),
                                                  "ci": boot_diff(cell, o, "pct"),
                                                  "NC": round(float(cell["nc_pct"].mean() - o["nc_pct"].mean()), 2),
                                                  "NC_ci": boot_diff(cell, o, "nc_pct")}
    js = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(js)
    print(js)
    rows = []
    for src in ["all", "navtest", "cg_test"]:
        for g in GROUPS:
            s = res[src][g]
            rows.append([src, g, s["scenes"], s["logs"]] + [s[k] for k in TERMS] + [s.get("EPDMS_ci")])
    print(pd.DataFrame(rows, columns=["source", "group", "scenes", "logs"] + list(TERMS) + ["EPDMS_ci"]).to_string())
    for src in ["all", "navtest", "cg_test"]:
        for g, v in res[src]["diff_cell_minus"].items():
            print(f"{src:8s} cell - {g:12s} EPDMS {v['EPDMS']:+6.2f} {v['ci']}   NC {v['NC']:+6.2f} {v['NC_ci']}")


if __name__ == "__main__":
    main()
