"""Inputs of the DrivoR baseline on a CG fold (docs/CG_SPLIT.md; step 1 of the go / no-go: withheld vs exposed). Run inside
a job (scripts/hgo/cg_drivor_train.sbatch / cg_drivor_eval.sbatch); idempotent.

1. scene filters configs/navsim_cg/train_test_split/scene_filter/<split>_<fold>_<variant>.yaml (withheld, exposed): the
   fold manifest tokens that have a metric cache (the devkit dropped ~1 % of the new frames, no route), log_names = the
   pool logs; the pair is balanced to the same number of windows (surplus withheld-only controls dropped, seed 0;
   data/_lists/cg/<split>/<fold>/balance_dropped.txt) and written as the shared manifests train_<variant>_cached.txt;
   the number of scenes goes to exp/cg_drivor/n_<split>_<fold>_<variant>.txt (scheduler length, asserted at train time);
2. configs/navsim_cg/cg_logs/<split>.yaml (@package _global_): train_logs = the pool logs (the devkit's own log split
   would otherwise drop pool logs that it lists as val), val_logs = 10 pool logs: validation is monitoring on training
   scenes only; the run's last checkpoint is used, never the 'best' one;
3. exp/cg_drivor/train_metric_cache/metadata/index.csv: one index (absolute paths) over the metric caches of the pool;
   the agent reads $NAVSIM_EXP_ROOT/train_metric_cache, and the training job sets NAVSIM_EXP_ROOT=exp/cg_drivor;
4. evaluation subset of the fold's test scenes that have a metric cache (counts of the others in the prep json): every
   scene of cell / same_man / same_partner and --n-control random
   scenes (seed 0) of man_none and of other, split by source into scene filters <split>_<fold>_eval_{navtest,cg_test}
   and the token list data/_lists/cg/<split>/<fold>/eval_subset.txt.

usage: python scripts/hgo/cg_drivor_prep.py [--split split_v1] [--fold turn_vru]
"""

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SF = ROOT / "configs/navsim_cg/train_test_split/scene_filter"


def cache_index(dirs):
    out = {}
    for d in dirs:
        for p in sorted((ROOT / d / "metadata").glob("*.csv")):
            for line in p.read_text().splitlines()[1:]:
                out.setdefault(line.split("/")[-2], line)
    return out


def scene_filter(path, logs, tokens):
    y = ["_target_: navsim.common.dataclasses.SceneFilter", "_convert_: 'all'", "num_history_frames: 4",
         "num_future_frames: 10", "frame_interval: 1", "has_route: true", "max_scenes: null", "log_names:"]
    y += [f"  - '{l}'" for l in sorted(logs)] + ["tokens:"] + [f"  - '{t}'" for t in sorted(tokens)]
    path.write_text("\n".join(y) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split_v1")
    ap.add_argument("--fold", default="turn_vru")
    ap.add_argument("--caches", nargs="+", default=["exp/metric_cache/train_hgo", "exp/metric_cache/cg_split_v1_train"])
    ap.add_argument("--test-caches", nargs=2, default=["exp/metric_cache/navtest_navsim2.2_np1.26.4",
                                                       "exp/metric_cache/cg_split_v1_test_all"])
    ap.add_argument("--n-control", type=int, default=1000)
    ap.add_argument("--n-val-logs", type=int, default=10)
    a = ap.parse_args()
    L = ROOT / "data/_lists/cg" / a.split
    F = L / a.fold
    out = ROOT / "exp/cg_drivor"
    (out / "train_metric_cache/metadata").mkdir(parents=True, exist_ok=True)
    pool_logs = L.joinpath("pool_logs.txt").read_text().split()
    idx = cache_index(a.caches)
    info = {"metric_cache_entries": len(idx)}
    # 1. training scene filters. The cache filter drops different tokens from the two manifests; the pair must keep the
    # same number of windows (same steps, same scheduler length): drop, seed 0, the surplus from the withheld-only
    # tokens (by construction of cg_split.py the free controls that exposed swapped for the cell windows). The balanced
    # manifests train_<variant>_cached.txt are the shared manifests of every method on this fold.
    kept = {}
    for v in ("withheld", "exposed"):
        toks = F.joinpath(f"train_{v}.txt").read_text().split()
        kept[v] = sorted(t for t in toks if t in idx)
        info[v] = {"manifest": len(toks), "with_metric_cache": len(kept[v])}
    surplus = len(kept["withheld"]) - len(kept["exposed"])
    if surplus < 0:
        raise SystemExit(f"exposed has {-surplus} more cached windows than withheld: dropping exposed-only windows would "
                         "remove exposure; decide the rule before training")
    only_w = sorted(set(kept["withheld"]) - set(kept["exposed"]))
    dropped = sorted(random.Random(0).sample(only_w, surplus))
    kept["withheld"] = sorted(set(kept["withheld"]) - set(dropped))
    assert len(kept["withheld"]) == len(kept["exposed"])
    F.joinpath("balance_dropped.txt").write_text("".join(t + "\n" for t in dropped))
    info["balance_dropped_from_withheld"] = surplus
    for v in ("withheld", "exposed"):
        F.joinpath(f"train_{v}_cached.txt").write_text("\n".join(kept[v]) + "\n")
        scene_filter(SF / f"{a.split}_{a.fold}_{v}.yaml", pool_logs, kept[v])
        (out / f"n_{a.split}_{a.fold}_{v}.txt").write_text(f"{len(kept[v])}\n")
        info[v]["train"] = len(kept[v])
    # 3. metric cache index over the pool
    pool = set(L.joinpath("pool_tokens.txt").read_text().split())
    rows = [idx[t] for t in sorted(pool) if t in idx]
    (out / "train_metric_cache/metadata/index.csv").write_text("file_name\n" + "\n".join(rows) + "\n")
    info["pool_with_metric_cache"] = len(rows)
    # 2. log lists (Hydra global package)
    rng = random.Random(0)
    val = sorted(rng.sample(sorted(pool_logs), a.n_val_logs))
    (ROOT / "configs/navsim_cg/cg_logs").mkdir(parents=True, exist_ok=True)
    y = ["# @package _global_", "# written by scripts/hgo/cg_drivor_prep.py", "train_logs:"]
    y += [f"  - '{l}'" for l in sorted(pool_logs)] + ["val_logs:"] + [f"  - '{l}'" for l in val]
    (ROOT / f"configs/navsim_cg/cg_logs/{a.split}.yaml").write_text("\n".join(y) + "\n")
    # 4. evaluation subset, over the test scenes that have a metric cache (the official navtest cache for navtest, the
    # split's test cache for the CG-test sessions): every model is scored on the same fixed scenes
    test_idx = {"navtest": cache_index(a.test_caches[:1]), "cg_test": cache_index(a.test_caches[1:])}
    groups_all = list(csv.DictReader(open(F / "test_groups.csv")))
    groups = [r for r in groups_all if r["token"] in test_idx[r["source"]]]
    info["test_without_metric_cache"] = {g: sum(r["group"] == g for r in groups_all) - sum(r["group"] == g for r in groups)
                                         for g in ("cell", "same_man", "same_partner", "man_none", "other")}
    rng = random.Random(0)
    sub = []
    for g in ("cell", "same_man", "same_partner", "man_none", "other"):
        rows_g = [r for r in groups if r["group"] == g]
        if g in ("man_none", "other") and len(rows_g) > a.n_control:
            rows_g = rng.sample(rows_g, a.n_control)
        sub += rows_g
    F.joinpath("eval_subset.txt").write_text("\n".join(sorted(r["token"] for r in sub)) + "\n")
    for src in ("navtest", "cg_test"):
        rs = [r for r in sub if r["source"] == src]
        scene_filter(SF / f"{a.split}_{a.fold}_eval_{src}.yaml", {r["log"] for r in rs}, [r["token"] for r in rs])
        info[f"eval_{src}"] = len(rs)
    info["eval_by_group"] = {g: sum(r["group"] == g for r in sub) for g in ("cell", "same_man", "same_partner",
                                                                            "man_none", "other")}
    (out / f"prep_{a.split}_{a.fold}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
