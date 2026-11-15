"""Exposure-dose check of the DrivoR CG pilot (docs/CG_SPLIT.md §10.2): does training on a window leave a measurable trace
in the official EPDMS / NC of that window? Both pilot checkpoints are scored on TRAINING windows of the fold:

  ex_only_cell    strict cell windows (turn at an intersection x crossing VRU) in exposed only: only exposed trained on them
  ex_only_purged  other windows of the inclusive purge (exposed only, not strict cell), sample
  wh_only         windows in withheld only (the free no-partner windows that exposed swapped for the purge), sample of the
                  same size as ex_only_cell: only withheld trained on them (mirror check)
  both_<group>    windows in both manifests (both trained on them), sample per test group: same_man, same_partner,
                  man_none, other

d = S(exposed) - S(withheld) per group, paired bootstrap over logs and over sessions, and differences in differences
d(ex_only_cell) - d(both_<g>), d(wh_only) - d(both_<g>). Groups use the same strict labels as the test groups
(cg_cells.label_cells, cg_split.groups).

usage (inside a job):
  python scripts/hgo/cg_dose.py prep [--fold data/_lists/cg/split_v1/turn_vru] [--n 500] [--seed 0]
  python scripts/hgo/cg_dose.py gap --exposed <csv> --withheld <csv> [--out f]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hgo"))
from cg_cells import label_cells  # noqa: E402
from cg_drivor_prep import scene_filter  # noqa: E402
from cg_gap import COLS, UNITS, boot_means, ci, load  # noqa: E402
from cg_split import fold_parts, groups  # noqa: E402

FOLD = "data/_lists/cg/split_v1/turn_vru"
SF = ROOT / "configs/navsim_cg/train_test_split/scene_filter"
BOTH = ["both_same_man", "both_same_partner", "both_man_none", "both_other"]
GROUPS = ["ex_only_cell", "ex_only_purged", "wh_only"] + BOTH


def prep(a):
    fold = ROOT / a.fold
    _, ms, p = fold_parts(fold.name.replace("_", "|", 1))
    tv = label_cells(pd.read_parquet(ROOT / "data/_lists/cg/events_trainval.parquet").reset_index(drop=True), 3.0, 6.0)
    tv = tv.drop_duplicates("token").set_index("token")
    tv["group"] = groups(tv, ms, p)
    ex = set(fold.joinpath("train_exposed_cached.txt").read_text().split())
    wh = set(fold.joinpath("train_withheld_cached.txt").read_text().split())
    assert not (ex | wh) - set(tv.index), "training tokens without event labels"
    rng = np.random.default_rng(a.seed)
    pick = lambda toks, n: sorted(rng.choice(sorted(toks), min(n, len(toks)), replace=False)) if toks else []
    g = tv["group"]
    ex_only, wh_only, both = ex - wh, wh - ex, ex & wh
    cell = {t for t in ex_only if g[t] == "cell"}
    sel = {"ex_only_cell": sorted(cell),
           "ex_only_purged": pick(ex_only - cell, a.n),
           "wh_only": pick(wh_only, len(cell))}
    for s in BOTH:
        sel[s] = pick({t for t in both if g[t] == s[5:]}, a.n)
    rows = [(t, s) for s, ts in sel.items() for t in ts]
    df = pd.DataFrame(rows, columns=["token", "group"])
    assert df["token"].is_unique
    df["log"] = tv.loc[df["token"], "log"].to_numpy()
    df["session"] = tv.loc[df["token"], "session"].to_numpy()
    df["man"] = tv.loc[df["token"], "man"].to_numpy()
    df["partner"] = tv.loc[df["token"], "partner"].to_numpy()
    out = fold / "dose"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "groups.csv", index=False)
    scene_filter(SF / f"{fold.parent.name}_{fold.name}_dose.yaml", set(df["log"]), df["token"])
    info = {"available": {"ex_only": len(ex_only), "wh_only": len(wh_only), "both": len(both), "ex_only_cell": len(cell)},
            "selected": {s: {"windows": int((df["group"] == s).sum()), "logs": int(df.loc[df["group"] == s, "log"].nunique())}
                         for s in GROUPS},
            "total": len(df)}
    (out / "prep.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def gap(a):
    g = pd.read_csv(ROOT / a.fold / "dose/groups.csv").set_index("token")
    (ex, ex_ok), (wh, wh_ok) = load(a.exposed), load(a.withheld)
    miss = {k: {s: int((~g.index.isin(list(ok)) & (g["group"] == s)).sum()) for s in GROUPS}
            for k, ok in (("exposed", ex_ok), ("withheld", wh_ok))}
    print("missing / invalid windows per run and group:", json.dumps(miss))
    if any(v for m in miss.values() for v in m.values()) and not a.allow_missing:
        raise SystemExit("missing or invalid windows (see above); pass --allow-missing to use the windows valid in both")
    toks = [t for t in g.index if t in ex_ok and t in wh_ok]
    df = g.loc[toks].copy()
    for s in GROUPS:
        df["slice_" + s] = df["group"] == s
    for k, c in COLS.items():
        df[f"ex_{k}"], df[f"wh_{k}"] = 100 * ex.loc[toks, c].to_numpy(), 100 * wh.loc[toks, c].to_numpy()
        df[f"d_{k}"] = df[f"ex_{k}"] - df[f"wh_{k}"]
    rng = np.random.default_rng(0)
    B = {u: rng.integers(0, df[u].nunique(), (a.n_boot, df[u].nunique())) for u in UNITS}
    boot = {(k, u): boot_means(df, f"d_{k}", u, B, GROUPS) for k in COLS for u in UNITS}
    res = {"exposed": a.exposed, "withheld": a.withheld, "windows": len(df), "missing": miss}
    for s in GROUPS:
        d = df[df["slice_" + s]]
        r = {"windows": len(d), "logs": int(d["log"].nunique()), "sessions": int(d["session"].nunique())}
        for k in COLS:
            r[f"exposed_{k}"], r[f"withheld_{k}"] = round(float(d[f"ex_{k}"].mean()), 2), round(float(d[f"wh_{k}"].mean()), 2)
            r[f"d_{k}"] = round(float(d[f"d_{k}"].mean()), 2)
            r[f"n_differ_{k}"] = int((d[f"d_{k}"] != 0).sum())
            for u in UNITS:
                r[f"d_{k}_ci_{u}"], r[f"d_{k}_valid_draws_{u}"] = ci(boot[(k, u)][s])
        res[s] = r
    for c in ("ex_only_cell", "ex_only_purged", "wh_only"):
        for s in BOTH:
            for k in COLS:
                key = f"did_{k}_{c}_minus_{s}"
                res[key] = {"value": round(res[c][f"d_{k}"] - res[s][f"d_{k}"], 2)}
                for u in UNITS:
                    res[key][f"ci_{u}"], res[key][f"valid_draws_{u}"] = ci(boot[(k, u)][c] - boot[(k, u)][s])
    js = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(js)
    print(js)
    for s in GROUPS:
        r = res[s]
        print(f"{s:15s} n={r['windows']:5d} logs={r['logs']:4d}  EPDMS ex {r['exposed_EPDMS']:6.2f} wh {r['withheld_EPDMS']:6.2f} "
              f"d {r['d_EPDMS']:+6.2f} log{r['d_EPDMS_ci_log']} sess{r['d_EPDMS_ci_session']}  |  NC ex {r['exposed_NC']:6.2f} "
              f"wh {r['withheld_NC']:6.2f} d {r['d_NC']:+6.2f} log{r['d_NC_ci_log']} sess{r['d_NC_ci_session']}")
    for c in ("ex_only_cell", "ex_only_purged", "wh_only"):
        for k in COLS:
            print(f"did {k:5s} {c} - both_*: " + "  ".join(
                f"{s[5:]} {res[f'did_{k}_{c}_minus_{s}']['value']:+.2f} {res[f'did_{k}_{c}_minus_{s}']['ci_log']}"
                for s in BOTH))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--fold", default=FOLD)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("gap")
    q.add_argument("--fold", default=FOLD)
    q.add_argument("--exposed", nargs="+", required=True)
    q.add_argument("--withheld", nargs="+", required=True)
    q.add_argument("--n-boot", type=int, default=2000)
    q.add_argument("--allow-missing", action="store_true")
    q.add_argument("--out")
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else gap(a)


if __name__ == "__main__":
    main()
