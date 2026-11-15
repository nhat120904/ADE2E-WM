"""Manifests of the compositional split on NAVSIM (Protocol B, docs/CG_SPLIT.md). Labels only (GT annotations of the logs),
nothing here looks at a model.

Design (decided 12/11, docs/CG_SPLIT.md §9):
  sessions   the 171 trainval drive sessions (log name without the two trailing frame indices) are split, stratified by
             city and seeded, into CG-test sessions (--test-frac) and train sessions. navtest (46 other sessions) stays
             test. No session is in both, so a test scene never shares a drive (place, time, road users) with training.
  pool       the CG training pool of every method: navtrain tokens of the train sessions, plus every conflict frame
             (any maneuver x partner flag, inclusive 4 s / 10 s) of the train sessions not in navtrain, thinned to one
             frame per --gap frames per log, so that the atoms of each held-out pair keep support. Every window has 4
             history and 10 future frames (scene filter of the devkit and of the metric cache).
  test       navtest scenes + frames of the CG-test sessions (same window rule, thinned to one frame per --gap): all
             frames of a strict cell with a partner, plus up to --none-test frames per maneuver without a partner
             (controls). Groups per fold: cell (held-out), same_man (same maneuver, other partner), same_partner (same
             partner, other maneuver), man_none (same maneuver, no partner), other.
  fold       held-out cell (strict maneuver x crossing partner, cg_cells.label_cells). withheld = pool minus the inclusive purge (t0 or any of the
             next 8 frames in the cell under man4 or man10 and the 4 s or 10 s conflict); exposed = pool minus the
             same number of windows drawn at random among pool windows with no partner and outside every inclusive cell
             (same data budget, atoms untouched). Delta_exposure = S(exposed) - S(withheld) on the same test scenes,
             within each method.

usage: python scripts/hgo/cg_split.py [--test-frac 0.2] [--seed 0] [--folds turn|vru left|veh_opp]
writes data/_lists/cg/<--name>/ (sessions.json, pool_*.txt, val_logs.txt, test_extra_*.txt, cache_todo_*.txt, <fold>/train_*.txt,
<fold>/test_groups.csv, summary.json) and docs/cg_split_folds.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hgo"))
from cg_cells import EPDMS, MANEUVERS, PARTNERS, label_cells, md_table, power, purge_mask  # noqa: E402

NAVTRAIN = "third_party/navsim_v2/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"


def yaml_tokens(path):
    """tokens of a devkit scene filter yaml (plain list parse; the file is ~100k lines)."""
    sec, toks = None, []
    for line in (ROOT / path).read_text().splitlines():
        s = line.strip()
        if s.endswith(":") and not s.startswith("-"):
            sec = s[:-1]
        elif s.startswith("- ") and sec == "tokens":
            toks.append(s[2:].strip().strip("'\""))
    return toks


def window_ok(e):
    return (e["Hh"] >= 3) & (e["K"] >= 10)


def thin(df, gap):
    """Keep one frame per `gap` frames in every log (greedy in time)."""
    keep = []
    for _, g in df.sort_values(["log", "t"]).groupby("log", sort=False):
        last = -10 ** 9
        for i, t in zip(g.index.to_numpy(), g["t"].to_numpy()):
            if t - last >= gap:
                keep.append(i)
                last = t
    return df.loc[keep]


def cache_index(dirs):
    out = set()
    for d in dirs:
        csv = sorted((ROOT / d / "metadata").glob("*.csv"))
        if csv:
            out |= {Path(x.strip()).parent.name for x in csv[0].read_text().splitlines()[1:]}
    return out


MAN_SETS = {"turn": ["left", "right"]}  # merged fold (user 12/11: "Gộp rẽ × người đi bộ")


def fold_parts(f):
    """'left|vru' -> ('left', ['left'], 'vru'); 'turn|vru' or 'left+right|vru' -> one fold over both maneuvers."""
    m, p = f.split("|")
    return m, MAN_SETS.get(m, m.split("+")), p


def groups(df, ms, p):
    man = df["man"].isin(ms)
    return np.select([man & (df["partner"] == p),
                      man & (df["partner"] != "none"),
                      (df["partner"] == p),
                      man & (df["partner"] == "none")],
                     ["cell", "same_man", "same_partner", "man_none"], "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="split_v1")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pet", type=float, default=3.0)
    ap.add_argument("--tconf", type=float, default=6.0)
    ap.add_argument("--gap", type=int, default=2, help="thinning of added / extra-test frames, in frames (0.5 s)")
    ap.add_argument("--none-test", type=int, default=1500, help="control frames without a partner, per maneuver")
    ap.add_argument("--folds", nargs="+", default=["turn|vru", "left|veh_opp"],
                    help="maneuver|partner; maneuver 'turn' = left or right at an intersection, or 'a+b'")
    ap.add_argument("--metric-cache", nargs="*", default=["exp/metric_cache/train_hgo"])
    ap.add_argument("--out-md", default="docs/cg_split_folds.md")
    a = ap.parse_args()
    out = ROOT / "data/_lists/cg" / a.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    tv = label_cells(pd.read_parquet(ROOT / "data/_lists/cg/events_trainval.parquet").reset_index(drop=True), a.pet, a.tconf)
    te = pd.read_parquet(ROOT / "data/_lists/cg/events_test.parquet")
    navtest = set((ROOT / "data/_lists/navtest_tokens.txt").read_text().split())
    te = label_cells(te[te["token"].isin(navtest)].reset_index(drop=True), a.pet, a.tconf)
    navtrain = set(yaml_tokens(NAVTRAIN))
    tv["navtrain"] = tv["token"].isin(navtrain)
    incl = [f"in|{m}|{g}" for m in MANEUVERS for g in PARTNERS]
    tv["any_conflict"] = tv[incl].any(axis=1)
    assert not set(te["session"]) & set(tv["session"]), "navtest and trainval share a drive session"

    # 1. sessions
    sess = tv.groupby("session")["city"].first()
    test_s = []
    for city, g in sess.groupby(sess):
        s = np.sort(g.index.to_numpy())
        rng.shuffle(s)
        test_s += list(s[:max(1, round(a.test_frac * len(s)))])
    test_s = sorted(test_s)
    is_test = tv["session"].isin(test_s).to_numpy()
    ok = window_ok(tv).to_numpy()

    # 2. training pool
    base = tv[~is_test & ok & tv["navtrain"].to_numpy()]
    add = thin(tv[~is_test & ok & ~tv["navtrain"].to_numpy() & tv["any_conflict"].to_numpy()], a.gap)
    pool = pd.concat([base.assign(src="navtrain"), add.assign(src="conflict_add")]).sort_index()

    # 3. extra test frames from the CG-test sessions
    cand = tv[is_test & ok]
    xt = [thin(cand[cand["partner"] != "none"], a.gap)]
    for m in MANEUVERS:
        c = thin(cand[(cand["partner"] == "none") & (cand["man"] == m)], 2 * a.gap)
        xt.append(c.sample(min(a.none_test, len(c)), random_state=a.seed))
    xt = pd.concat(xt).sort_index()
    xt = xt[~xt.index.duplicated()]

    cached = cache_index(a.metric_cache)
    write = lambda name, items: (out / name).write_text("\n".join(sorted(set(items))) + "\n")
    write("pool_tokens.txt", pool["token"])
    write("pool_logs.txt", pool["log"])
    # validation logs (monitoring, warm-up stop, thresholds; added 13/11): logs of training sessions without any pool
    # window. Same drive sessions as the pool, never a CG-test session or navtest.
    write("val_logs.txt", sorted(set(tv.loc[~is_test, "log"]) - set(pool["log"])))
    write("test_extra_tokens.txt", xt["token"])
    write("test_extra_logs.txt", xt["log"])
    write("cache_todo_train.txt", pool.loc[~pool["token"].isin(cached), "token"])
    write("cache_todo_test.txt", xt.loc[~xt["token"].isin(cached), "token"])
    (out / "sessions.json").write_text(json.dumps(
        {"test_frac": a.test_frac, "seed": a.seed, "test_sessions": test_s,
         "test_by_city": {c: int(sess.loc[test_s].eq(c).sum()) for c in sorted(sess.unique())},
         "train_sessions": sorted(set(sess.index) - set(test_s))}, indent=1))

    # 4. folds
    scores = {}
    for k, path in EPDMS.items():  # per-scene navtest EPDMS of two reference models: rough power of each fold
        if (ROOT / path).exists():
            sc = pd.read_csv(ROOT / path)
            scores[k] = sc[sc["valid"] == True].set_index("token")["score"]  # noqa: E712
    pool_idx = pool.index.to_numpy()
    free = (pool["partner"] == "none").to_numpy() & ~pool["any_conflict"].to_numpy()
    rows = []
    for f in a.folds:
        m, ms, p = fold_parts(f)
        name = f.replace("|", "_").replace("+", "")
        fd = out / name
        fd.mkdir(exist_ok=True)
        tv[f"in|{f}"] = tv[[f"in|{x}|{p}" for x in ms]].any(axis=1)
        pool[f"in|{f}"] = tv.loc[pool_idx, f"in|{f}"].to_numpy()
        pm = purge_mask(tv, f"in|{f}")[pool_idx]
        is_cell = lambda df: df["man"].isin(ms) & (df["partner"] == p)
        withheld = pool[~pm]
        drop = rng.choice(np.nonzero(free & ~pm)[0], int(pm.sum()), replace=False)
        keep = np.ones(len(pool), bool)
        keep[drop] = False
        exposed = pool[keep]
        assert not is_cell(withheld).any() and not withheld[f"in|{f}"].any()
        assert len(exposed) == len(withheld) and is_cell(exposed).sum() == is_cell(pool).sum()
        write(f"{name}/train_withheld.txt", withheld["token"])
        write(f"{name}/train_exposed.txt", exposed["token"])
        tg = pd.concat([te.assign(source="navtest"), xt.assign(source="cg_test")])
        tg = tg.assign(group=groups(tg, ms, p))[["token", "log", "session", "source", "man", "partner", "group"]]
        tg.to_csv(fd / "test_groups.csv", index=False)
        cell = tg[tg["group"] == "cell"]
        n_nt, l_nt = int((cell["source"] == "navtest").sum()), cell.loc[cell["source"] == "navtest", "log"].nunique()
        sup_p = withheld[(withheld["partner"] == p) & ~withheld["man"].isin(ms)]
        r = {"fold": f, "test_cell_navtest": n_nt, "test_cell_cgtest": int((cell["source"] == "cg_test").sum()),
             "test_cell_logs": cell["log"].nunique(), "test_cell_sessions": cell["session"].nunique(),
             "purged": int(pm.sum()), "purge_pct": round(100 * pm.mean(), 2),
             "cell_in_exposed": int(is_cell(exposed).sum()),
             "support_partner": len(sup_p), "support_partner_logs": sup_p["log"].nunique(),
             "support_man": int((withheld["man"].isin(ms) & (withheld["partner"] != p)).sum()),
             "train_windows": len(withheld)}
        nt_cell = cell[cell["source"] == "navtest"]
        pw = power(scores, nt_cell["token"].to_numpy(), nt_cell["log"].to_numpy()) if len(scores) == 2 else {}
        if "se_paired" in pw:
            r["mde80_est"] = round(2.8 * max(pw["se_log_boot"] * np.sqrt(l_nt / r["test_cell_logs"]),
                                              pw["se_paired"] * np.sqrt(n_nt / len(cell))), 1)
        rows.append(r)
    folds = pd.DataFrame(rows)
    summary = {"name": a.name, "args": vars(a), "sessions_test": len(test_s), "sessions_train": int(sess.size - len(test_s)),
               "logs_test": int(tv.loc[is_test, "log"].nunique()), "logs_train": int(tv.loc[~is_test, "log"].nunique()),
               "pool_windows": len(pool), "pool_navtrain": len(base), "pool_conflict_add": len(add),
               "pool_logs": int(pool["log"].nunique()), "navtrain_in_test_sessions": int((is_test & tv["navtrain"]).sum()),
               "test_extra_frames": len(xt), "test_extra_logs": int(xt["log"].nunique()),
               "cache_todo_train": int((~pool["token"].isin(cached)).sum()),
               "cache_todo_test": int((~xt["token"].isin(cached)).sum()), "folds": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    # devkit scene filters of the frames whose metric cache is missing (scripts/hgo/cg_cache.sbatch); frames the devkit
    # drops (no route) get no cache and are removed from the manifests afterwards
    fdir = ROOT / "configs/navsim_cg/train_test_split/scene_filter"
    fdir.mkdir(parents=True, exist_ok=True)
    for part, df in (("test", xt), ("train", pool)):
        todo = df[~df["token"].isin(cached)]
        lines = ["_target_: navsim.common.dataclasses.SceneFilter", "_convert_: 'all'", "num_history_frames: 4",
                 "num_future_frames: 10", "frame_interval: 1", "has_route: true", "max_scenes: null", "log_names:"]
        lines += [f"  - '{x}'" for x in sorted(set(todo["log"]))] + ["tokens:"]
        lines += [f"  - '{x}'" for x in sorted(set(todo["token"]))]
        (fdir / f"{a.name}_{part}.yaml").write_text("\n".join(lines) + "\n")

    pc = pd.crosstab(pool["man"], pool["partner"])
    xc = pd.crosstab(xt["man"], xt["partner"])
    md = [f"# CG split manifests `{a.name}` (generated by scripts/hgo/cg_split.py)", "",
          f"Sessions: {len(test_s)} CG-test / {sess.size - len(test_s)} train of {sess.size} trainval sessions "
          f"(stratified by city, seed {a.seed}, test fraction {a.test_frac}); navtest = 46 further sessions. "
          f"Conflict: an agent whose trajectory crosses the ego path, PET < {a.pet} s, ego reaches the conflict point within "
          f"{a.tconf} s; left / right = turns inside an intersection (road bends = 'curve').", "",
          f"Training pool: {len(pool)} windows from {pool['log'].nunique()} logs = {len(base)} navtrain + {len(add)} "
          f"conflict frames added (one per {a.gap} frames). navtrain tokens dropped with the CG-test sessions: "
          f"{summary['navtrain_in_test_sessions']}.", "",
          f"Extra test frames (CG-test sessions): {len(xt)} from {xt['log'].nunique()} logs. Metric cache still to "
          f"build: {summary['cache_todo_train']} training + {summary['cache_todo_test']} test frames.", "",
          "## Pool (man x partner)", "", "```", pc.to_string(), "```", "",
          "## Extra test frames (man x partner)", "", "```", xc.to_string(), "```", "",
          "## Folds", "", md_table(folds), "",
          "mde80_est: rough, the SE of H32 − DrivoR on the navtest scenes of the fold (cg_cells.power) scaled by the number of test "
          "logs / scenes; unreliable when the navtest part is small."]
    (ROOT / a.out_md).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
