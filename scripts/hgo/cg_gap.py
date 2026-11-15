"""Exposure gap of one method on a CG fold (docs/CG_SPLIT.md §3, §10): Delta = S(exposed) - S(withheld) per test group on
the fold's fixed evaluation subset, with a paired bootstrap over logs and over drive sessions; and the difference in
differences Delta(cell) - Delta(group) (an exposure effect specific to the held-out cell, not a generic difference
between two training runs). EPDMS and NC.

Cell slices (fixed before any result, docs/CG_SPLIT.md §10): the cell label looks over the whole future (man10, conflict
within --tconf s), the planner and EPDMS over 4 s. cell_in4 = the same strict rule on the 4 s future (cg_cells.label_cells
with man4 / inter4 / petx4 / tconfx4): the composition is executed inside the scored horizon; cell_not_strict4 = the
other cell scenes (its complement; not all of them are shown to be anticipation: 79 have a turn and the 10 s conflict
within 4 s but fail the 4 s crossing test, docs/reviews/drivor_cg_followup_2026-11-14.md).

Bootstrap: a draw without any scene of a group is undefined (NaN), never a gap of 0; percentiles over the valid draws,
whose number is reported; a difference in differences uses the draws where both groups are defined.

Coverage: every subset scene must be scored valid by both runs; missing / invalid scenes are listed per run and group,
and the script stops unless --allow-missing (dropping hard scenes would bias the gap).

usage: python scripts/hgo/cg_gap.py --fold data/_lists/cg/split_v1/turn_vru \\
           --exposed exp/cg_eval/A/navtest.csv exp/cg_eval/A/cg_test.csv --withheld exp/cg_eval/B/navtest.csv ... [--out f]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cg_cells import PARTNERS, partner_flags
from cg_split import fold_parts

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ["cell", "cell_in4", "cell_not_strict4", "same_man", "same_partner", "man_none", "other"]
CELLS = ["cell", "cell_in4", "cell_not_strict4"]
COLS = {"EPDMS": "score", "NC": "no_at_fault_collisions"}
UNITS = ["log", "session"]


def load(paths):
    d = pd.concat([pd.read_csv(p) for p in paths]).drop_duplicates("token").set_index("token")
    return d, set(d.index[d["valid"].astype(bool)])


def strict4(e, ms, p, pet, tconf):
    """label_cells' strict cell rule on the 4 s future."""
    fx = partner_flags(e, "4", pet, tconf, "x")
    partner = np.select([fx[g] for g in PARTNERS], PARTNERS, "none")
    man = e["man4"].fillna("na")
    man = man.where(~man.isin(["left", "right"]) | (e["inter4"] == 1), "curve")
    return pd.Series(man.isin(ms).to_numpy() & (partner == p), index=e.index)


def ci(x):
    """95 % percentile interval over the defined draws (NaN = a draw without the group) and their number."""
    ok = x[np.isfinite(x)]
    return (np.percentile(ok, [2.5, 97.5]).round(2).tolist() if len(ok) else None), int(len(ok))


def boot_means(df, col, unit, B, groups):
    """bootstrap means of df[col] per group, resampling the units (index B: n_boot x n_units)."""
    units = np.array(sorted(df[unit].unique()))
    out = {}
    for g in groups:
        d = df[df["slice_" + g]]
        s = d.groupby(unit)[col].sum().reindex(units).fillna(0).to_numpy()
        n = d.groupby(unit)[col].count().reindex(units).fillna(0).to_numpy()
        num, den = s[B[unit]].sum(1), n[B[unit]].sum(1)
        out[g] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--exposed", nargs="+", required=True)
    ap.add_argument("--withheld", nargs="+", required=True)
    ap.add_argument("--subset", help="token list scored by both runs (default <fold>/eval_subset.txt)")
    ap.add_argument("--events", nargs="+", default=["data/_lists/cg/events_test.parquet",
                                                    "data/_lists/cg/events_trainval.parquet"])
    ap.add_argument("--pet", type=float, default=3.0)
    ap.add_argument("--tconf", type=float, default=6.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    fold = Path(a.fold)
    _, ms, p = fold_parts(fold.name.replace("_", "|", 1))
    g = pd.read_csv(fold / "test_groups.csv").set_index("token")
    subset = (Path(a.subset) if a.subset else fold / "eval_subset.txt").read_text().split()
    g = g.loc[subset]

    # cell slices
    cols = ["token", "man4", "inter4"] + [f"{k}x4_{q}" for k in ("pet", "tconf") for q in PARTNERS]
    cell_toks = g.index[g["group"] == "cell"]
    ev = pd.concat([pd.read_parquet(ROOT / f, columns=cols) for f in a.events]).drop_duplicates("token")
    ev = ev.set_index("token").reindex(cell_toks)
    assert ev["man4"].notna().any(), "no event labels for the cell scenes"
    in4 = strict4(ev, ms, p, a.pet, a.tconf)
    for s in GROUPS:
        g["slice_" + s] = g["group"] == s
    g["slice_cell_in4"] = False
    g.loc[cell_toks, "slice_cell_in4"] = in4.to_numpy()
    g["slice_cell_not_strict4"] = g["slice_cell"] & ~g["slice_cell_in4"]

    # coverage
    (ex, ex_ok), (wh, wh_ok) = load(a.exposed), load(a.withheld)
    cov = {}
    for name, ok in (("exposed", ex_ok), ("withheld", wh_ok)):
        miss = g.index[~g.index.isin(list(ok))]
        cov[name] = {s: int(g.loc[miss, "slice_" + s].sum()) for s in GROUPS}
        cov[name]["tokens"] = sorted(miss)
    n_miss = len(cov["exposed"]["tokens"]) + len(cov["withheld"]["tokens"])
    print("missing / invalid scenes per run and group:",
          json.dumps({k: {s: v[s] for s in GROUPS} for k, v in cov.items()}))
    if n_miss and not a.allow_missing:
        raise SystemExit("some subset scenes are missing or invalid in a run (see above); fix the evaluation or pass "
                         "--allow-missing (the gap then uses the scenes valid in both runs)")
    toks = [t for t in g.index if t in ex_ok and t in wh_ok]
    df = g.loc[toks].copy()
    for k, c in COLS.items():
        df[f"ex_{k}"], df[f"wh_{k}"] = 100 * ex.loc[toks, c].to_numpy(), 100 * wh.loc[toks, c].to_numpy()
        df[f"d_{k}"] = df[f"ex_{k}"] - df[f"wh_{k}"]

    rng = np.random.default_rng(0)
    B = {u: rng.integers(0, df[u].nunique(), (a.n_boot, df[u].nunique())) for u in UNITS}
    boot = {(k, u): boot_means(df, f"d_{k}", u, B, GROUPS) for k in COLS for u in UNITS}
    res = {"fold": a.fold, "scenes": len(df), "logs": int(df["log"].nunique()), "sessions": int(df["session"].nunique()),
           "exposed": a.exposed, "withheld": a.withheld, "coverage_missing": cov,
           "slice_rule": f"cell_in4: man4 in {ms} (left/right with inter4), first crossing partner at 4 s = {p} "
                         f"(petx4 < {a.pet}, tconfx4 <= {a.tconf})"}
    for s in GROUPS:
        d = df[df["slice_" + s]]
        r = {"scenes": len(d), "logs": int(d["log"].nunique()), "sessions": int(d["session"].nunique())}
        for k in COLS:
            r[f"exposed_{k}"] = round(float(d[f"ex_{k}"].mean()), 2)
            r[f"withheld_{k}"] = round(float(d[f"wh_{k}"].mean()), 2)
            r[f"gap_{k}"] = round(float(d[f"d_{k}"].mean()), 2)
            for u in UNITS:
                r[f"gap_{k}_ci_{u}"], r[f"gap_{k}_valid_draws_{u}"] = ci(boot[(k, u)][s])
        res[s] = r
    for c in CELLS:  # difference in differences: cell slice vs each control group
        dc = df[df["slice_" + c]]
        for s in GROUPS[3:]:
            d = df[df["slice_" + s]]
            for k in COLS:
                key = f"did_{k}_{c}_minus_{s}"
                res[key] = {"value": round(float(dc[f"d_{k}"].mean() - d[f"d_{k}"].mean()), 2)}
                for u in UNITS:
                    res[key][f"ci_{u}"], res[key][f"valid_draws_{u}"] = ci(boot[(k, u)][c] - boot[(k, u)][s])
    js = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(js)
    print(js)
    for s in GROUPS:
        r = res[s]
        print(f"{s:13s} n={r['scenes']:5d}  EPDMS ex {r['exposed_EPDMS']:6.2f} wh {r['withheld_EPDMS']:6.2f} "
              f"gap {r['gap_EPDMS']:+6.2f} log{r['gap_EPDMS_ci_log']} sess{r['gap_EPDMS_ci_session']}  |  "
              f"NC ex {r['exposed_NC']:6.2f} wh {r['withheld_NC']:6.2f} gap {r['gap_NC']:+6.2f} "
              f"log{r['gap_NC_ci_log']} sess{r['gap_NC_ci_session']}")


if __name__ == "__main__":
    main()
