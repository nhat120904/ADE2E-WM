"""Perception diagnosis of a slot model (docs/METHOD_RATIONALE.md §8b): WHY hazardous agents are missing from the state,
and how much a different detection threshold would recover, on the validation logs (not navtest).

Input: a dump of scripts/hgo/dump_plans.py with --raw-det (det_box [N,K,7], det_cls, det_score: top-K CenterNet peaks
BEFORE the score threshold, ego frame), props [N,64,8,3], sel [N,64], slot_box / slot_cls / slot_risk (the kept slots).
For every scene, the v2 PDM labels (hgo.pdm.proposal_labels) of the 64 proposals against ALL GT tracks give the hazards
(a track with an at-fault collision or a TTC violation for >= 1 proposal), as scripts/hgo/hazard_coverage.py.

1. Cause of each hazard at the model's threshold (--thr, 0.3), checked in this order (class-aware, centre distance):
     covered           a kept slot within --gate m (2) of the same class
     out_of_range      t0 centre outside the perception area (--grid: pilot |x|, |y| <= 32 m; v3 x to 48 m)
     cut_by_selection  a same-class detection >= thr within the gate exists, but nearest-first selection dropped it
     below_threshold   a same-class detection within the gate exists with score < thr
     wrong_class       a detection >= thr within the gate exists, of another class
     localization      a same-class detection >= thr between the gate and 2 x gate
     absent            nothing within 2 x gate in the top-K peaks (any class, any score)
2. Threshold sweep (--sweep): for each threshold, selection is re-run (valid peaks, nearest first, <= --n-slot) and
   matched to GT (class-aware Hungarian, gated): hazard coverage, recall of all GT objects inside the grid per class and
   distance band, unmatched kept slots per scene (false positives), slots used, scenes at the slot cap.
3. Selected trajectory (argmax sel): share of scenes whose selected proposal is unsafe (at-fault collision), and among
   those, whether its at-fault objects are in the state (covered) or not (and why).
Rows are also split by the held-out cell of the checkpoint's training split (turn x VRU, hgo_factors_val.csv).

usage: python scripts/hgo/perception_diag.py --run exp/hgo/runs/H32_turn_vru_s0 --dump val_diag.npz [--procs 16]
writes <run>/perception_diag.json and <run>/perception_diag_hazards.csv
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hgo"))
AREA = {"pilot": (-32.0, 32.0, -32.0, 32.0), "v3": (-32.0, 48.0, -32.0, 32.0)}  # perception area per grid
CLS = {0: "vehicle", 1: "pedestrian", 2: "bicycle", 3: "static"}
BANDS = [(0, 10), (10, 20), (20, 32), (32, 1e9)]
G = {}


def init(cache_dir, pkl_dir, sensors, dump, gate, thr, sweep, n_slot, area):
    from train import metric_cache_paths
    z = np.load(ROOT / dump, allow_pickle=True)
    G.update(mc=metric_cache_paths(ROOT / cache_dir), pkl_dir=ROOT / pkl_dir, sensors=ROOT / sensors, gate=gate, thr=thr,
             area=area,
             sweep=sweep, n_slot=n_slot, props=z["props"], sel=z["sel"], det_box=z["det_box"], det_cls=z["det_cls"],
             det_score=z["det_score"])


def match(sb, sc, gb, gc, gate):
    """Kept slots (n,2; cls) vs GT (m,2; cls) -> GT index matched to each slot or -1 (gated, class-aware Hungarian)."""
    out = np.full(len(sb), -1)
    if not len(sb) or not len(gb):
        return out
    d = np.linalg.norm(sb[:, None] - gb[None], axis=-1)
    bad = (d > gate) | (sc[:, None] != gc[None])
    r, c = linear_sum_assignment(np.where(bad, 1e6, d))
    ok = ~bad[r, c]
    out[r[ok]] = c[ok]
    return out


def select(db, dc, ds, thr, n):
    """Model rule (hgo.model.select_slots): peaks with score >= thr, nearest first, at most n."""
    v = np.nonzero(ds >= thr)[0]
    v = v[np.argsort(np.hypot(db[v, 0], db[v, 1]), kind="stable")][:n]
    return v


def one_log(args):
    log, items = args
    from ade2e_wm.hgo.data import LogData
    from ade2e_wm.hgo.geometry import to_frame
    from ade2e_wm.hgo.pdm import proposal_labels
    ld = LogData(G["pkl_dir"] / f"{log}.pkl", G["sensors"])
    gate, thr = G["gate"], G["thr"]
    hz_rows, sweep_rows, sel_rows = [], [], []
    for i, tok, t, held in items:
        path = G["mc"].get(tok)
        if path is None:
            continue
        a = ld.anns[t]
        box = to_frame(np.linalg.inv(ld.e2g[t]) @ ld.l2g[t], a["box"])
        cls, track = a["cls"], a["track"]
        try:
            lab = proposal_labels(path, G["props"][i].astype(np.float64), list(track))
        except Exception:
            continue
        fault, ttc = lab["fault"] > 0, (lab["ttc_pair"] > 0) & lab["ttc_ok"][:, None]
        hazard = fault.any(0) | ttc.any(0)
        x0, x1, y0, y1 = G["area"]
        inside = (box[:, 0] >= x0) & (box[:, 0] <= x1) & (box[:, 1] >= y0) & (box[:, 1] <= y1)
        dist = np.hypot(box[:, 0], box[:, 1])
        ins = np.nonzero(inside)[0]
        db = G["det_box"][i].astype(np.float64)
        dc = G["det_cls"][i].astype(int)
        ds = G["det_score"][i].astype(np.float64)
        # --- 1. causes at the model threshold
        kept = select(db, dc, ds, thr, G["n_slot"])
        m = match(db[kept, :2], dc[kept], box[ins, :2], cls[ins], gate)
        covered = np.zeros(len(box), bool)
        covered[ins[m[m >= 0]]] = True
        dd = np.linalg.norm(db[:, None, :2] - box[None, :, :2], axis=-1) if len(box) else np.zeros((len(db), 0))
        cat_of = {}
        for k in np.nonzero(hazard)[0]:
            same, near, near2 = dc == cls[k], dd[:, k] <= gate, dd[:, k] <= 2 * gate
            if covered[k]:
                cat = "covered"
            elif not inside[k]:
                cat = "out_of_range"
            elif (same & near & (ds >= thr)).any():
                cat = "cut_by_selection"
            elif (same & near).any():
                cat = "below_threshold"
            elif (~same & near & (ds >= thr)).any():
                cat = "wrong_class"
            elif (same & near2 & (ds >= thr)).any():
                cat = "localization"
            else:
                cat = "absent"
            best = ds[same & near].max() if (same & near).any() else 0.0
            cat_of[k] = cat
            hz_rows.append({"token": tok, "cls": CLS.get(int(cls[k]), "?"), "dist": float(dist[k]), "x": float(box[k, 0]),
                            "y": float(box[k, 1]), "fault": bool(fault[:, k].any()), "category": cat,
                            "best_same_score": float(best), "held": held})
        # --- 2. threshold sweep
        for th in G["sweep"]:
            kp = select(db, dc, ds, th, G["n_slot"])
            mm = match(db[kp, :2], dc[kp], box[ins, :2], cls[ins], gate)
            cov = np.zeros(len(box), bool)
            cov[ins[mm[mm >= 0]]] = True
            n_valid = int((ds >= th).sum())
            row = {"token": tok, "thr": th, "kept": len(kp), "valid": n_valid, "fp": int((mm < 0).sum()),
                   "cap": int(n_valid > G["n_slot"]), "held": held,
                   "hz_n": int(hazard.sum()), "hz_cov": int((hazard & cov).sum())}
            for c, name in CLS.items():
                for lo, hi in BANDS[:3]:
                    sel_ = inside & (cls == c) & (dist >= lo) & (dist < hi)
                    row[f"gt_{name}_{lo}"] = int(sel_.sum())
                    row[f"tp_{name}_{lo}"] = int((sel_ & cov).sum())
                hs = hazard & (cls == c)
                row[f"hz_{name}"] = int(hs.sum())
                row[f"hzcov_{name}"] = int((hs & cov).sum())
            sweep_rows.append(row)
        # --- 3. selected trajectory
        j = int(np.argmax(G["sel"][i]))
        unsafe = bool(fault[j].any() or lab["fault_unmatched"][j] > 0)
        objs = np.nonzero(fault[j])[0]
        sel_rows.append({"token": tok, "held": held, "unsafe": unsafe, "n_fault_obj": len(objs),
                         "fault_covered": int(covered[objs].sum()), "fault_unmatched": bool(lab["fault_unmatched"][j] > 0),
                         "fault_classes": ",".join(CLS.get(int(cls[o]), "?") for o in objs),
                         "fault_causes": ",".join(cat_of.get(o, "?") for o in objs),
                         "fault_inside": int(inside[objs].sum())})
    return hz_rows, sweep_rows, sel_rows


def share(m):
    return round(float(np.mean(m)) * 100, 1) if len(m) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("exp/hgo/runs/H32_turn_vru_s0"))
    ap.add_argument("--dump", default="val_diag.npz")
    ap.add_argument("--metric-cache", default="exp/metric_cache/train_hgo")
    ap.add_argument("--pkl-dir", default="data/navsim_logs/trainval")
    ap.add_argument("--sensors", default="data/sensor_blobs/trainval")
    ap.add_argument("--events", default="data/_lists/cg/events_trainval.parquet")
    ap.add_argument("--factors", default="data/_lists/hgo_factors_val.csv")
    ap.add_argument("--gate", type=float, default=2.0)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--sweep", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    ap.add_argument("--n-slot", type=int, default=32)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--grid", default="pilot", choices=list(AREA), help="perception grid of the checkpoint")
    a = ap.parse_args()
    dump = a.run / a.dump
    z = np.load(ROOT / dump, allow_pickle=True)
    toks = [str(t) for t in z["token"]]
    ev = pd.read_parquet(ROOT / a.events, columns=["token", "log", "t"]).set_index("token")
    fac = pd.read_csv(ROOT / a.factors).set_index("token")
    held_of = lambda t: bool(t in fac.index and fac.at[t, "maneuver"] in ("left", "right") and fac.at[t, "vru"] == "vru")
    idx = [i for i, t in enumerate(toks) if t in ev.index]
    if a.max_scenes:
        idx = sorted(np.random.default_rng(0).choice(idx, min(a.max_scenes, len(idx)), replace=False))
    by_log = {}
    for i in idx:
        r = ev.loc[toks[i]]
        by_log.setdefault(r["log"], []).append((i, toks[i], int(r["t"]), held_of(toks[i])))
    res, t0 = [], time.time()
    with Pool(a.procs, initializer=init, initargs=(a.metric_cache, a.pkl_dir, a.sensors, str(dump), a.gate, a.thr,
                                                    a.sweep, a.n_slot, AREA[a.grid])) as p:
        for j, r in enumerate(p.imap_unordered(one_log, list(by_log.items()), chunksize=1)):
            res.append(r)
            print(f"log {j + 1}/{len(by_log)} {time.time() - t0:.0f} s", flush=True)
    hz = pd.DataFrame([x for r in res for x in r[0]])
    sw = pd.DataFrame([x for r in res for x in r[1]])
    sl = pd.DataFrame([x for r in res for x in r[2]])
    hz.to_csv(ROOT / a.run / "perception_diag_hazards.csv", index=False)
    cats = ["covered", "out_of_range", "cut_by_selection", "below_threshold", "wrong_class", "localization", "absent"]
    causes = lambda d: {c: share(d["category"] == c) for c in cats}
    out = {"dump": str(dump), "scenes": int(sl.shape[0]), "hazards": len(hz), "thr": a.thr, "gate": a.gate,
           "causes_all": causes(hz),
           "causes_by_class": {c: {"n": int((hz["cls"] == c).sum()), **causes(hz[hz["cls"] == c])} for c in CLS.values()},
           "causes_by_dist": {f"{lo}-{hi:g}": {"n": int(((hz["dist"] >= lo) & (hz["dist"] < hi)).sum()),
                                               **causes(hz[(hz["dist"] >= lo) & (hz["dist"] < hi)])} for lo, hi in BANDS},
           "pedestrian_by_held": {str(h): {"n": int(((hz["cls"] == "pedestrian") & (hz["held"] == h)).sum()),
                                           **causes(hz[(hz["cls"] == "pedestrian") & (hz["held"] == h)])}
                                  for h in (True, False)},
           "below_threshold_best_score_quantiles": hz.loc[hz["category"] == "below_threshold", "best_same_score"]
           .quantile([0.25, 0.5, 0.75]).round(3).tolist()}
    sweep = {}
    for th, g in sw.groupby("thr"):
        s = g.sum(numeric_only=True)
        d = {"hazard_coverage": round(100 * s["hz_cov"] / max(s["hz_n"], 1), 1),
             "fp_per_scene": round(float(g["fp"].mean()), 2), "slots_per_scene": round(float(g["kept"].mean()), 1),
             "scenes_at_cap_pct": round(100 * float(g["cap"].mean()), 1)}
        for name in CLS.values():
            d[f"hazard_cov_{name}"] = round(100 * s[f"hzcov_{name}"] / max(s[f"hz_{name}"], 1), 1)
            for lo, _ in BANDS[:3]:
                d[f"recall_{name}_{lo}"] = round(100 * s[f"tp_{name}_{lo}"] / max(s[f"gt_{name}_{lo}"], 1), 1)
        sweep[f"{th:g}"] = d
    out["sweep"] = sweep
    u = sl[sl["unsafe"]]
    has_obj = u[u["n_fault_obj"] > 0]
    out["selected"] = {"scenes": len(sl), "unsafe_pct": share(sl["unsafe"]),
                       "unsafe_with_fault_object": len(has_obj),
                       "of_those_all_fault_objects_in_state_pct": share(has_obj["fault_covered"] == has_obj["n_fault_obj"]),
                       "of_those_none_in_state_pct": share(has_obj["fault_covered"] == 0),
                       "of_those_object_outside_grid_pct": share(has_obj["fault_inside"] < has_obj["n_fault_obj"]),
                       "unsafe_by_held": {str(h): share(sl.loc[sl["held"] == h, "unsafe"]) for h in (True, False)},
                       "fault_classes_uncovered": has_obj.loc[has_obj["fault_covered"] == 0, "fault_classes"]
                       .value_counts().head(8).to_dict(),
                       "fault_causes": pd.Series(",".join(has_obj["fault_causes"]).split(","))
                       .value_counts().to_dict() if len(has_obj) else {}}
    (ROOT / a.run / "perception_diag.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
