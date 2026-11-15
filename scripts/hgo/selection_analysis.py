"""Which part of the scorer loses the points? Offline analysis of the proposal selection on navtest (no retraining).

Inputs (from scripts/hgo/dump_plans.py and scripts/hgo/oracle_props.py):
  <run>/navtest_plans.npz      props, sel, raw head logits, per-slot risk logits (slot variants)
  <run>/navtest_props_pdm.npz  every proposal graded by DrivoR's PDM train scorer (nc, dac, ep, ttc, comfort, ddc, final)
The metric is that approximate PDM score (GT-privileged), not the official EPDMS. Everything here is a diagnostic on the
test split: rules or weights that look better here must be chosen again on a validation split before they are used.

1. selection rules, recomputed from the logits with DrivoR's weighted log score (the model's rule is checked to reproduce
   the dumped selection): current, NC head without slot risk, slot risk without NC head;
2. head swaps: one predicted term replaced by its PDM truth (and all of them) -> points lost by each head;
3. per-head discrimination over all proposals and within scenes (what selection needs), for NC: head, slot risk, both;
4. selected proposals with an at-fault collision: was a safe alternative available, how confident was the scorer;
5. sensitivity of the result to the NC / EP weights (diagnostic only);
6. does the scorer use the world model? future-swap test (every proposal scored with another proposal's imagined
   future, dump_plans.py --swap 1) and how much the imagined future depends on the action (registers, slots, slot motion).

v2 dumps (--rule v2 | mono | pmono, default: auto): the structured rule is turned into equivalent per-head logits
(NC = prod_k (1 - v_k sigmoid(pair_fault)), TTC = min_k (1 - sigmoid(pair_ttc)), context heads from ctx_logit), the
monolithic rules use logit_mono / logit_pmono; sections 1-5 then run unchanged (no slot-risk term, no future swap).

usage: python scripts/hgo/selection_analysis.py --run exp/hgo/runs/H32_turn_vru_s0 [--split turn_vru] [--rule v2]
writes <run>/navtest_selection.json (navtest_selection_<rule>.json for v2 dumps)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hgo"))
from oracle_props import COLS, auc, held_mask  # noqa: E402

HEAD = ["nc", "dac", "ep", "ttc", "comfort", "ddc"]  # order of logit[..., i] (SUB_SCORES)
EPS = 1e-6


def logsig(x):
    return -np.logaddexp(0.0, -x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def selection(t, w):
    """DrivoR's weighted log score from per-term values in [0, 1] (log terms) / probabilities (additive terms).
    t: dict of log_nc, log_dac, log_ddc (log-probabilities) and ttc, ep, comfort (probabilities), each (N, P)."""
    add = w["ttc"] * t["ttc"] + w["ep"] * t["ep"] + w["comfort"] * t["comfort"]
    return w["noc"] * t["log_nc"] + w["dac"] * t["log_dac"] + w["ddc"] * t["log_ddc"] + np.log(np.maximum(add, EPS))


def terms_pred(logit, risk_sum, use_head=True, use_risk=True):
    L = {k: logit[..., i] for i, k in enumerate(HEAD)}
    log_nc = np.zeros_like(L["nc"])
    if use_head:
        log_nc = log_nc + logsig(L["nc"])
    if use_risk and risk_sum is not None:
        log_nc = log_nc + risk_sum
    return {"log_nc": log_nc, "log_dac": logsig(L["dac"]), "log_ddc": logsig(L["ddc"]),
            "ttc": sigmoid(L["ttc"]), "ep": sigmoid(L["ep"]), "comfort": sigmoid(L["comfort"])}


def swap(t, pdm, key):
    """Replace one term by the PDM truth of every proposal."""
    t = dict(t)
    j = COLS.index({"log_nc": "nc", "log_dac": "dac", "log_ddc": "ddc"}.get(key, key))
    v = pdm[..., j]
    t[key] = np.log(np.maximum(v, EPS)) if key.startswith("log_") else v
    return t


def logit_of(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def v2_logits(z, rule):
    """Per-head logits (N,P,6, HEAD order) equivalent to a v2 selection rule, and the pair-fault logits (or None)."""
    if rule in ("mono", "pmono"):
        return z[f"logit_{rule}"].astype(np.float32), None
    pf, pt = z["pair_fault"].astype(np.float32), z["pair_ttc"].astype(np.float32)
    keep = pf > -29.0
    static = (z["slot_cls"][:, None, :] == 3) & keep
    r = np.clip(sigmoid(pf) * keep, 0, 1 - EPS)
    # E[NC] as in hgo.model.structured_selection: PDM takes the min over collisions of 0 (agent) / 0.5 (non-agent object)
    p_nc = np.exp(np.log1p(-r * (keep & ~static)).sum(-1)) * (0.5 + 0.5 * np.exp(np.log1p(-r * static).sum(-1)))
    ttc = np.where(keep, 1 - sigmoid(pt), 1.0).min(-1)
    ctx = z["ctx_logit"].astype(np.float32)  # dac, ddc, ep, comfort
    return np.stack([logit_of(p_nc), ctx[..., 0], ctx[..., 2], logit_of(ttc), ctx[..., 3], ctx[..., 1]], -1), pf


def pick(score, pdm):
    k = score.argmax(1)
    return pdm[np.arange(len(pdm)), k], k


def within_auc(score, label):
    """Mean over scenes that have both classes of the per-scene ROC AUC."""
    out = [auc(s, l) for s, l in zip(score, label) if 0 < l.sum() < len(l)]
    return float(np.mean(out)) if out else float("nan"), len(out)


def summarize_rule(score, pdm, idx):
    ch, _ = pick(score[idx], pdm[idx])
    return {"final": float(ch[:, 6].mean()), "nc": float(ch[:, 0].mean()), "dac": float(ch[:, 1].mean()),
            "ep": float(ch[:, 2].mean()), "ttc": float(ch[:, 3].mean()), "ddc": float(ch[:, 5].mean()),
            "collision_rate": float((ch[:, 0] < 1).mean()), "offroad_rate": float((ch[:, 1] < 1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--plans", default="navtest_plans.npz")
    ap.add_argument("--graded", default="navtest_props_pdm.npz")
    ap.add_argument("--split", default="turn_vru")
    ap.add_argument("--factors", default="data/_lists/hgo_factors_test.csv")
    ap.add_argument("--graded-plans", default="navtest_plans_eval65036.npz",
                    help="the dump whose proposals were graded (props must match the analysed dump)")
    ap.add_argument("--rule", default="auto", help="v2 dumps: v2 (structured) | mono | pmono; auto = the H32 logits when present, "
                    "else v2 when the dump has pair terms, else mono")
    a = ap.parse_args()
    import pandas as pd
    from scipy.stats import spearmanr
    z = np.load(a.run / a.plans)
    g = np.load(a.run / a.graded)
    tokens = z["token"].tolist()
    assert g["token"].tolist() == tokens, "graded proposals belong to another dump"
    rule = a.rule if a.rule != "auto" else ("h32" if "logit" in z.files else "v2" if "pair_fault" in z.files else "mono")
    pair = None
    if rule == "h32":
        logit, sel = z["logit"], z["sel"]
    else:
        logit, pair = v2_logits(z, rule)
        sel = z[f"sel_{rule}"] if f"sel_{rule}" in z.files else z["sel"]
    pdm = g["pdm"]
    ref = a.run / a.graded_plans
    props_diff = float(np.abs(np.load(ref)["props"] - z["props"]).max()) if ref.exists() else float("nan")
    print("max |props - graded props| (m):", props_diff)
    w = json.loads(str(z["score_w"]))
    risk = z["slot_risk"].astype(np.float32) if "slot_risk" in z.files and rule == "h32" else None  # (N,P,S), -30 = empty
    risk_sum = logsig(-risk).sum(-1) if risk is not None else None  # log prod(1 - sigma(r_k))
    ok = ~np.isnan(pdm).any((1, 2))

    fac = pd.read_csv(ROOT / a.factors).set_index("token").reindex(tokens)
    split = json.loads((ROOT / f"data/_lists/hgo_split_{a.split}.json").read_text())
    held = held_mask(fac.reset_index(), split["holdout"])
    labeled = fac["maneuver"].notna().to_numpy()
    groups = {"all": ok, "seen": ok & labeled & ~held, "held_out": ok & labeled & held}
    idx = {k: np.nonzero(m)[0] for k, m in groups.items()}

    # 0. the recomputed rule must reproduce the dumped selection
    t_cur = terms_pred(logit, risk_sum)
    s_cur = selection(t_cur, w)
    same = float((s_cur.argmax(1) == sel.argmax(1)).mean())
    n_ = np.arange(len(sel))
    mis = s_cur.argmax(1) != sel.argmax(1)  # bf16 autocast in the model vs fp64 here: near-tied proposals swap
    tie_gap = float(np.median((s_cur[n_, s_cur.argmax(1)] - s_cur[n_, sel.argmax(1)])[mis])) if mis.any() else 0.0
    res = {"run": str(a.run), "rule": rule, "metric": "PDM train scorer (approximate EPDMS), GT-privileged; diagnostic on navtest",
           "score_w": w, "scenes": {k: int(len(v)) for k, v in idx.items()},
           "recomputed_selection_matches_dump": same, "mismatch_median_score_gap_fp64": tie_gap,
           "max_abs_props_diff_vs_graded_m": props_diff,
           "oracle_final": {k: float(pdm[v, :, 6].max(1).mean()) for k, v in idx.items()}}

    # 1-2. rules and head swaps
    rules = {"current": s_cur}
    if risk_sum is not None:
        rules["nc_head_only"] = selection(terms_pred(logit, risk_sum, use_risk=False), w)
        rules["slot_risk_only"] = selection(terms_pred(logit, risk_sum, use_head=False), w)
    for key in ["log_nc", "log_dac", "log_ddc", "ttc", "ep", "comfort"]:
        rules[f"true_{key.replace('log_', '')}"] = selection(swap(t_cur, pdm, key), w)
    t_all = t_cur
    for key in ["log_nc", "log_dac", "log_ddc", "ttc", "ep", "comfort"]:
        t_all = swap(t_all, pdm, key)
    rules["true_all_terms"] = selection(t_all, w)
    # grouped swaps. The EP label is masked by the multiplicative metrics in both PDM scorers (progress * NC*DAC*DDC,
    # train_pdm_scorer.py:170, pdm_scorer.py:232), so 'true_ep' also tells the selector which proposals collide or leave
    # the road; 'true_ep_where_safe' uses the EP truth only where NC = DAC = DDC = 1 and keeps the prediction elsewhere.
    rules["true_safety_nc_ttc"] = selection(swap(swap(t_cur, pdm, "log_nc"), pdm, "ttc"), w)
    rules["true_context_dac_ddc"] = selection(swap(swap(t_cur, pdm, "log_dac"), pdm, "log_ddc"), w)
    mult_ok = (pdm[..., 0] >= 1) & (pdm[..., 1] >= 1) & (pdm[..., 5] >= 1)
    rules["true_ep_where_safe"] = selection(dict(t_cur, ep=np.where(mult_ok, pdm[..., 2], t_cur["ep"])), w)
    rules["dumped_selection"] = sel  # the model's own bf16-autocast score (near-ties may resolve differently)
    has_swap = "logit_swap" in z.files
    if has_swap:
        risk_sw = z["risk_swap"].astype(np.float32) if "risk_swap" in z.files else None
        rules["future_swapped"] = selection(terms_pred(z["logit_swap"], logsig(-risk_sw).sum(-1) if risk_sw is not None
                                                      else None), w)
    res["rules"] = {r: {k: summarize_rule(s, pdm, v) for k, v in idx.items()} for r, s in rules.items()}

    # 3. discrimination: over all proposals and within scenes
    P = pdm[ok]
    disc = {}
    nc_scores = {"nc_head": logsig(logit[ok][..., 0])}
    if risk_sum is not None:
        nc_scores["slot_risk"] = risk_sum[ok]
        nc_scores["nc_head_plus_risk"] = nc_scores["nc_head"] + risk_sum[ok]
    for name, sc in nc_scores.items():
        lab = P[..., 0] >= 1.0
        wa, n = within_auc(sc, lab)
        disc[name] = {"auc_all": auc(sc.ravel(), lab.ravel()), "auc_within_scene": wa, "scenes_with_both": n}
    for name, i, j in [("dac", 1, 1), ("ddc", 5, 5), ("ttc", 3, 3)]:
        sc, lab = logit[ok][..., i], P[..., j] >= 1.0
        wa, n = within_auc(sc, lab)
        disc[name] = {"auc_all": auc(sc.ravel(), lab.ravel()), "auc_within_scene": wa, "scenes_with_both": n}
    rho = [spearmanr(s, t)[0] for s, t in zip(logit[ok][..., 2], P[..., 2]) if t.std() > 1e-6]
    disc["ep"] = {"spearman_within_scene": float(np.nanmean(rho)), "scenes": len(rho)}
    m_ok = mult_ok[ok]  # EP ranking among proposals whose EP label is not masked by NC / DAC / DDC
    rho = [spearmanr(s[m], t[m])[0] for s, t, m in zip(logit[ok][..., 2], P[..., 2], m_ok) if m.sum() > 2 and t[m].std() > 1e-6]
    disc["ep_where_safe"] = {"spearman_within_scene": float(np.nanmean(rho)), "scenes": len(rho)}
    res["discrimination"] = disc

    # 4. selected proposals with an at-fault collision
    ch, k = pick(s_cur[ok], P)
    coll = ch[:, 0] < 1
    safe = (P[..., 0] >= 1) & (P[..., 1] >= 1)
    alt = safe.any(1)
    pnc = sigmoid(logit[ok][..., 0])
    if risk_sum is not None:
        pnc = pnc * np.exp(risk_sum[ok])
    rows = np.nonzero(coll)[0]
    best_safe = np.where(safe, P[..., 6], -1).argmax(1)
    res["selected_collisions"] = {
        "scenes": int(coll.sum()), "rate": float(coll.mean()),
        "with_safe_alternative": float(alt[rows].mean()) if len(rows) else float("nan"),
        "pred_nc_of_selected_mean": float(pnc[rows, k[rows]].mean()) if len(rows) else float("nan"),
        "pred_nc_of_best_safe_mean": float(pnc[rows, best_safe[rows]].mean()) if len(rows) else float("nan"),
        "true_ep_selected_mean": float(ch[rows, 2].mean()) if len(rows) else float("nan"),
        "true_ep_best_safe_mean": float(P[rows, best_safe[rows], 2].mean()) if len(rows) else float("nan"),
        "pred_ep_selected_mean": float(sigmoid(logit[ok][rows, k[rows], 2]).mean()) if len(rows) else float("nan"),
        "pred_ep_best_safe_mean": float(sigmoid(logit[ok][rows, best_safe[rows], 2]).mean()) if len(rows) else float("nan"),
    }
    if risk is not None or pair is not None:  # class of the slot with the highest risk on the selected proposal
        r = (risk if risk is not None else pair)[ok][np.arange(ok.sum()), k]  # (0 veh, 1 ped, 2 bike, 3 static)
        top = r.argmax(1)
        cls = z["slot_cls"][ok][np.arange(ok.sum()), top]
        res["selected_collisions"]["max_risk_slot_class_counts"] = {int(c): int((cls[rows] == c).sum())
                                                                    for c in np.unique(cls[rows])}
        res["selected_collisions"]["max_slot_risk_selected_mean"] = float(sigmoid(r[rows].max(1)).mean()) if len(rows) else float("nan")

    # 5. weight sensitivity (diagnostic on the test split; do not select weights here)
    sens = {}
    for m_nc in [1.0, 2.0, 4.0]:
        for m_ep in [1.0, 0.5]:
            ww = dict(w, noc=w["noc"] * m_nc, ep=w["ep"] * m_ep)
            sens[f"noc_x{m_nc:g}_ep_x{m_ep:g}"] = {k: summarize_rule(selection(t_cur, ww), pdm, v)["final"]
                                                  for k, v in idx.items()}
    res["weight_sensitivity_final"] = sens

    # 6. world-model usage
    if has_swap:
        s_sw = rules["future_swapped"]
        rho = [spearmanr(u, v)[0] for u, v in zip(s_cur[ok], s_sw[ok]) if u.std() > 1e-9 and v.std() > 1e-9]
        P_ = 1 / (1 + np.exp(-logit[ok]))
        Q_ = 1 / (1 + np.exp(-z["logit_swap"][ok]))
        res["future_swap"] = {
            "selection_changed": float((s_cur[ok].argmax(1) != s_sw[ok].argmax(1)).mean()),
            "spearman_scores_normal_vs_swapped": float(np.nanmean(rho)),
            "mean_abs_prob_change": {h: float(np.abs(P_[..., i] - Q_[..., i]).mean()) for i, h in enumerate(HEAD)},
            "dumped_sel_vs_recomputed_swap_check": float((z["sel_swap"][ok].argmax(1) == s_sw[ok].argmax(1)).mean())}
        if risk is not None:
            r0, r1 = risk_sum[ok], logsig(-z["risk_swap"].astype(np.float32)).sum(-1)[ok]
            res["future_swap"]["mean_abs_change_slot_risk_nc_prob"] = float(np.abs(np.exp(r0) - np.exp(r1)).mean())
        wsens = {}
        for k in ["spread_reg", "spread_slot", "hchange_reg", "hchange_slot", "motion_spread_m", "motion_disp_m"]:
            if k in z.files:
                wsens[k] = {gname: {"mean": float(np.nanmean(z[k][v])), "median": float(np.nanmedian(z[k][v]))}
                            for gname, v in idx.items()}
        res["wm_action_sensitivity"] = wsens

    (a.run / ("navtest_selection.json" if rule == "h32" else f"navtest_selection_{rule}.json")).write_text(
        json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ["scenes", "recomputed_selection_matches_dump", "mismatch_median_score_gap_fp64", "oracle_final"]}))
    print("rule".ljust(20), " ".join(f"{g}:final/coll".rjust(22) for g in idx))
    for r, v in res["rules"].items():
        print(r.ljust(20), " ".join(f"{v[g]['final']:.4f}/{v[g]['collision_rate']:.4f}".rjust(22) for g in idx))
    print(json.dumps(res["discrimination"], indent=1))
    print(json.dumps(res["selected_collisions"], indent=1))
    print(json.dumps(sens, indent=1))
    for k in ["future_swap", "wm_action_sensitivity"]:
        if k in res:
            print(k, json.dumps(res[k], indent=1))


if __name__ == "__main__":
    main()
