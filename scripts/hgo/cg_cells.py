"""Candidate held-out cells for the redesigned compositional split (docs/CG_SPLIT.md), from the event labels of
scripts/hgo/cg_events.py. Counts only: nothing here trains or selects a model.

Atoms (labels only, GT annotations):
  maneuver  man10 (logged ego path over the contiguous future, <= 10 s; catches waiting before a turn), with man4 also used
            for the inclusive purge
  partner   vru        a pedestrian / bicycle conflict with the ego path: PET < --pet s and the ego reaches the conflict point
                         within --tconf s
            veh_cross  a crossing vehicle conflict (heading 45-135 deg at the conflict point), same thresholds
            veh_opp    an opposing-heading vehicle conflict (> 135 deg; oncoming traffic in a turn), same thresholds
            none       none of the three (a close lead vehicle is reported separately, it is not a partner here)
Cell = (maneuver, partner). For every candidate held-out cell (turn or straight x vru / veh_cross / veh_opp):
  test      navtest scenes in the cell (strict definition above), their logs, and the official per-scene EPDMS spread of two
            existing models (H32 pilot, DrivoR) -> standard error of a paired difference and the smallest difference
            detectable with 80 % power at alpha 0.05 (normal approximation; a log-level bootstrap SE as a check)
  purge     training windows removed by the inclusive rule: the window's t0 or any of its next 8 frames (the 4 s target
            and the JEPA horizons) is in the cell under man4 OR man10 and the 4 s OR 10 s conflict
  support   after the purge, training windows of the sibling cells (same maneuver with another partner; same partner with
            another maneuver): the atoms of the held-out pair must stay supported (SVIB / Wiedemer: compositional support)
Pools: 'pilot' = trainval logs with images on disk listed in phase1_train_logs.txt (what H32 trained on); 'all' = every
trainval log (images to download).

usage: python scripts/hgo/cg_cells.py [--pet 3] [--tconf 6]  ->  docs/cg_split_candidates.md, data/_lists/cg/candidates.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANEUVERS = ["left", "right", "straight"]
PARTNERS = ["vru", "veh_cross", "veh_opp"]
EPDMS = {"H32": "exp/hgo/runs/H32_turn_vru_s0/navtest_epdms_main.csv",
         "DrivoR": "exp/drivor_nav2_navtest_full/2026.10.28.12.55.37/2026.10.28.18.44.05.csv"}


def partner_flags(e, h, pet, tconf, x=""):
    return {g: (e[f"pet{x}{h}_{g}"] < pet) & (e[f"tconf{x}{h}_{g}"] <= tconf) for g in PARTNERS}


def label_cells(e, pet, tconf):
    """Strict cell of each frame = man x partner, used for test scenes, groups and support:
      man      man10, except that a left / right heading change outside the nuPlan intersection layer (inter10 != 1: a
               road bend, not a turn) becomes 'curve'
      partner  first group in PARTNERS order with an agent whose trajectory CROSSES the ego path (petx / tconfx of
               cg_events.py) with PET < pet and the ego at the conflict point within tconf s; else 'none'
    Inclusive membership in|<maneuver>|<partner> (purge): man4 OR man10 (no intersection test) x any agent near the path
    (pet / tconf, crossing or not) at 4 s OR 10 s, a superset of the strict cell, so the purge stays conservative.
    (Changed 12/11 after the sample figures: before, the strict partner was any agent within 2.5 / 3 m of the path,
    which counted oncoming traffic in the next lane and VRUs walking along the kerb.)"""
    f10, f4 = partner_flags(e, "10", pet, tconf), partner_flags(e, "4", pet, tconf)
    fx = partner_flags(e, "10", pet, tconf, "x")
    partner = np.select([fx[g] for g in PARTNERS], PARTNERS, "none")
    man = e["man10"].fillna("na")
    man = man.where(~man.isin(["left", "right"]) | (e["inter10"] == 1), "curve")
    e = e.assign(partner=partner, man=man, cell=man + "|" + partner)
    for m in MANEUVERS:
        man_any = (e["man10"] == m) | (e["man4"] == m)
        for g in PARTNERS:
            e[f"in|{m}|{g}"] = man_any & (f10[g] | f4[g])
    return e


def window_ok(e):
    return (e["Hh"] >= 3) & (e["K"] >= 8)


def purge_mask(e, col, ahead=8):
    """True for frames whose t0 or any of the next `ahead` frames (same log) is in the cell."""
    out = np.zeros(len(e), bool)
    for _, g in e.groupby("log", sort=False):
        idx = g.index.to_numpy()
        inc = g[col].to_numpy()
        order = np.argsort(g["t"].to_numpy())
        inc_s = inc[order]
        hit = inc_s.copy()
        for k in range(1, ahead + 1):
            hit[:-k] |= inc_s[k:]
        res = np.empty_like(hit)
        res[order] = hit
        out[e.index.get_indexer(idx)] = res
    return out


def power(scores, tokens, logs):
    """Paired-difference SE and minimal detectable difference (EPDMS points) between the two reference models."""
    if len(scores) < 2:
        return {}
    a, b = (s.reindex(tokens) for s in scores.values())
    ok = a.notna() & b.notna()
    d = (a - b)[ok] * 100
    n = int(ok.sum())
    if n < 5:
        return {"n_scored": n}
    se = float(d.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(0)
    lg = pd.Series(logs, index=tokens)[ok]
    by_log = d.groupby(lg.values)
    sums, cnts = by_log.sum().to_numpy(), by_log.size().to_numpy()
    boots = []
    for _ in range(1000):
        j = rng.integers(0, len(sums), len(sums))
        boots.append(sums[j].sum() / cnts[j].sum())
    out = {"n_scored": n, "sd_scene": round(float(np.nanstd(np.r_[a[ok], b[ok]]) * 100), 1),
           "se_paired": round(se, 2), "se_log_boot": round(float(np.std(boots)), 2)}
    out["mde80"] = round(2.8 * max(out["se_paired"], out["se_log_boot"]), 2)
    return out


def md_table(df):
    """DataFrame -> markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet", type=float, default=3.0)
    ap.add_argument("--tconf", type=float, default=6.0)
    ap.add_argument("--events-dir", default="data/_lists/cg")
    ap.add_argument("--out-md", default="docs/cg_split_candidates.md")
    a = ap.parse_args()
    d = ROOT / a.events_dir
    te = pd.read_parquet(d / "events_test.parquet")
    tv = pd.read_parquet(d / "events_trainval.parquet") if (d / "events_trainval.parquet").exists() else None
    navtest = set((ROOT / "data/_lists/navtest_tokens.txt").read_text().split())
    pilot_train = set((ROOT / "data/_lists/phase1_train_logs.txt").read_text().split())
    te = label_cells(te[te["token"].isin(navtest)].reset_index(drop=True), a.pet, a.tconf)
    scores = {}
    for k, p in EPDMS.items():
        if (ROOT / p).exists():
            s = pd.read_csv(ROOT / p)
            scores[k] = s[s["valid"] == True].set_index("token")["score"]  # noqa: E712
    md = ["# Candidate held-out cells for the compositional split (generated by scripts/hgo/cg_cells.py)", "",
          f"Partner conflict: PET < {a.pet} s and the ego reaches the conflict point within {a.tconf} s (logged path, "
          "contiguous future ≤ 10 s), the partner's trajectory crosses the ego path. Maneuver: man10, turns only inside an "
          "intersection (else 'curve'). Test = navtest scenes (strict cell). Purge = inclusive rule over "
          "t0 and the next 8 frames. Power: paired difference H32 − DrivoR on the test cell (official per-scene EPDMS); "
          "mde80 = 2.8 × max(SE paired, SE log bootstrap), in EPDMS points.", ""]
    md += ["## Strict cells on navtest (scenes / logs)", "", "```",
           pd.crosstab(te["man"], te["partner"]).to_string(), "```", ""]
    pools = {}
    if tv is not None:
        tv = tv[window_ok(tv)].reset_index(drop=True)
        tv = label_cells(tv, a.pet, a.tconf)
        pools = {"pilot": tv[tv["log"].isin(pilot_train) & tv["has_img"]].reset_index(drop=True), "all": tv}
        for name, p in pools.items():
            md += [f"## Strict cells, training pool '{name}': {len(p)} windows, {p['log'].nunique()} logs", "", "```",
                   pd.crosstab(p["man"], p["partner"]).to_string(), "```", ""]
    rows = []
    for m in MANEUVERS:
        for g in PARTNERS:
            cell = f"{m}|{g}"
            t = te[te["cell"] == cell]
            r = {"cell": cell, "test_scenes": len(t), "test_logs": t["log"].nunique(),
                 **power(scores, t["token"].to_numpy(), t["log"].to_numpy())}
            for name, p in pools.items():
                pm = purge_mask(p, f"in|{cell}")
                kept = p[~pm]
                r[f"{name}_purged"] = int(pm.sum())
                r[f"{name}_cell_left"] = int((kept["cell"] == cell).sum())  # must be 0
                r[f"{name}_same_man"] = int(((kept["man"] == m) & (kept["partner"] != g)).sum())
                r[f"{name}_same_partner"] = int(((kept["partner"] == g) & (kept["man"] != m)).sum())
                r[f"{name}_same_partner_logs"] = int(kept.loc[(kept["partner"] == g) & (kept["man"] != m), "log"].nunique())
            rows.append(r)
    c = pd.DataFrame(rows)
    (d / "candidates.csv").write_text(c.to_csv(index=False))
    md += ["## Candidate held-out cells", "", md_table(c), ""]
    (ROOT / a.out_md).write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
