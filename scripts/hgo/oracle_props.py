"""Where does planning lose points: proposals (generator) or selection (scorer)? (docs/METHOD.md, Evaluation)

For every navtest scene, every one of the 64 proposals dumped by scripts/hgo/dump_plans.py is graded with the PDM train
scorer (src/ade2e_wm/hgo/pdm.py: NC, DAC, EP, TTC, comfort, DDC, final; DrivoR's training metric on the official v2.2
metric cache, non-reactive). This is an approximation of EPDMS (no traffic-light, lane-keeping or extended-comfort
terms; DDC weighted, not multiplicative), used the same way for the oracle and for the selected proposal, so:
  oracle = best final over the 64 proposals        -> low: the generator has no good candidate
  regret = oracle - final of the selected proposal -> high: the scorer picks badly
plus no-safe-candidate rate, rank of the selection, per-scene Spearman, and AUC of the scorer's NC / DAC / DDC heads.
Groups: all, seen and held-out factor combinations of the split (data/_lists/hgo_split_<name>.json), per maneuver.

usage: python scripts/hgo/oracle_props.py --run exp/hgo/runs/H32_turn_vru_s0 --split turn_vru [--workers 16]
writes <run>/navtest_props_pdm.npz (token, pdm[N,64,7]) and <run>/navtest_oracle.json
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLS = ["nc", "dac", "ep", "ttc", "comfort", "ddc", "final"]


def grade(args):
    from ade2e_wm.hgo.pdm import proposal_scores
    path, poses = args
    try:
        return proposal_scores(path, poses)
    except Exception as e:  # recorded as NaN rows and counted
        return f"{type(e).__name__}: {e}"[:300]


def auc(score, label):
    """ROC AUC (rank form, ties averaged); NaN if one class is missing."""
    from scipy.stats import rankdata
    pos = label.sum()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def held_mask(df, holdout):
    m = np.zeros(len(df), bool)
    for h in holdout:
        mm = np.ones(len(df), bool)
        for k, v in zip(["maneuver", "vru", "density"], h.split(":")):
            if v != "*":
                mm &= (df[k] == v).to_numpy()
        m |= mm
    return m


def summarize(pdm, sel, sub, idx):
    from scipy.stats import spearmanr
    p, s, q = pdm[idx], sel[idx], sub[idx]
    final = p[..., 6]
    k = s.argmax(1)
    chosen = p[np.arange(len(p)), k]
    oracle = final.max(1)
    safe = (p[..., 0] * p[..., 1]).max(1) > 0  # some proposal without at-fault collision and off-road
    rank = (final > chosen[:, 6:7]).sum(1)  # proposals strictly better than the selected one
    rho = [spearmanr(si, fi)[0] for si, fi in zip(s, final) if fi.std() > 1e-6]
    out = {"scenes": int(len(p)),
           "selected_final": float(chosen[:, 6].mean()), "oracle_final": float(oracle.mean()),
           "regret": float((oracle - chosen[:, 6]).mean()),
           "mean_final_all_proposals": float(final.mean()),
           "no_safe_candidate": float(1 - safe.mean()),
           "selected_is_best": float((chosen[:, 6] >= oracle - 1e-6).mean()),
           "proposals_better_than_selected": float(rank.mean()),
           "spearman_sel_vs_final": float(np.mean(rho)) if rho else float("nan"),
           "selected": {c: float(chosen[:, i].mean()) for i, c in enumerate(COLS)},
           "oracle_proposal": {c: float(p[np.arange(len(p)), final.argmax(1), i].mean()) for i, c in enumerate(COLS)}}
    for i, c in [(0, "nc"), (1, "dac"), (5, "ddc"), (3, "ttc")]:  # scorer head i vs truth == 1, over all proposals
        out[f"auc_{c}"] = auc(q[..., i].ravel(), p[..., i].ravel() >= 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--plans", default="navtest_plans.npz")
    ap.add_argument("--split", default="turn_vru")
    ap.add_argument("--metric-cache", default="exp/metric_cache/navtest_navsim2.2_np1.26.4")
    ap.add_argument("--factors", default="data/_lists/hgo_factors_test.csv")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.chdir(ROOT)
    import pandas as pd
    z = np.load(a.run / a.plans)
    tokens, props, sel, sub = z["token"].tolist(), z["props"], z["sel"], z["sub"]
    csv = sorted((Path(a.metric_cache) / "metadata").glob("*.csv"))[0]
    mc = {Path(l.strip()).parent.name: l.strip() for l in csv.read_text().splitlines()[1:]}
    out_npz = a.run / "navtest_props_pdm.npz"
    t0 = time.time()
    if out_npz.exists() and np.load(out_npz)["token"].tolist() == tokens:
        pdm = np.load(out_npz)["pdm"]
        errors = []
    else:
        jobs = [(mc[t], props[i]) for i, t in enumerate(tokens) if t in mc]
        have = [i for i, t in enumerate(tokens) if t in mc]
        pdm = np.full((len(tokens), props.shape[1], 7), np.nan, np.float32)
        errors = []
        with mp.get_context("fork").Pool(a.workers) as pool:
            for j, r in zip(have, pool.imap(grade, jobs, chunksize=8)):
                if isinstance(r, str):
                    errors.append({"token": tokens[j], "error": r})
                else:
                    pdm[j] = r
        np.savez(out_npz, token=np.array(tokens), pdm=pdm)
    ok = ~np.isnan(pdm).any((1, 2))
    fac = pd.read_csv(a.factors).set_index("token").reindex(tokens)
    split = json.loads((ROOT / f"data/_lists/hgo_split_{a.split}.json").read_text())
    held = held_mask(fac.reset_index(), split["holdout"])
    labeled = fac["maneuver"].notna().to_numpy()  # navtest scenes outside the HGO factor windows: only in "all"
    groups = {"all": ok, "seen": ok & labeled & ~held, "held_out": ok & labeled & held}
    for m in ["straight", "left", "right", "stop"]:
        groups[f"maneuver={m}"] = ok & (fac["maneuver"] == m).to_numpy()
    turn = fac["maneuver"].isin(["left", "right"]).to_numpy()
    vru = (fac["vru"] == "vru").to_numpy()
    groups["turn,no_vru"] = ok & turn & ~vru
    groups["straight/stop,vru"] = ok & ~turn & vru
    res = {"run": str(a.run), "metric": "PDM train scorer (approximate EPDMS, see docstring)", "split": split,
           "dumped_scenes": len(tokens), "graded_scenes": int(ok.sum()), "grading_errors": len(errors),
           "errors_first": errors[:10], "seconds": round(time.time() - t0, 1),
           "groups": {g: summarize(pdm, sel, sub, np.nonzero(m)[0]) for g, m in groups.items() if m.any()}}
    (a.run / "navtest_oracle.json").write_text(json.dumps(res, indent=1))
    keys = ["scenes", "selected_final", "oracle_final", "regret", "mean_final_all_proposals", "no_safe_candidate",
            "selected_is_best", "spearman_sel_vs_final", "auc_nc", "auc_dac"]
    print(json.dumps({k: res[k] for k in ["dumped_scenes", "graded_scenes", "grading_errors", "seconds"]}))
    for g, r in res["groups"].items():
        print(g.ljust(18), " ".join(f"{k}={r[k]:.3f}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys))


if __name__ == "__main__":
    main()
