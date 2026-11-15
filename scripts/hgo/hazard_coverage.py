"""Hazard coverage of the slot state (docs/METHOD_RATIONALE.md §8b): for the candidate pool of a trained slot model on
navtest, does the state contain the objects that make candidates unsafe, and if not, where are they lost?

For every scene of a dump (scripts/hgo/dump_plans.py: props (N,64,8,3), slot_box (N,S,7), slot_cls (N,S), slot_risk (N,64,S)
with -30 = empty slot), the v2 PDM labels (hgo.pdm.proposal_labels) are computed for the 64 proposals against ALL GT tracks of
the frame (not only the detection area). A hazard is a track with an at-fault collision (fault) or a TTC violation
(ttc_pair) for at least one proposal. Each hazard is put in one class:
  covered          matched to a kept slot (class-aware Hungarian on centre distance <= --gate m, gated before the
                   assignment, as hgo.losses.match_slots, against all GT objects of the detection area)
  out_of_range     its t0 centre is outside the detection area (|x| or |y| > 32 m in the t0 ego frame)
  cut_by_selection inside the area, not matched, the kept slots are full and all nearer than it (nearest-first selection
                   kept other detections instead)
  missed           inside the area, not matched otherwise (detector miss, localization error > gate, or below threshold)
Also: gt_rank (its rank by distance among GT objects of the area) -> share of hazards a nearest-k selection of perfect
detections would keep (k = 32, 64), and per-proposal coverage (the share of unsafe proposals none of whose at-fault objects
is matched, the quantity logged as unsafe_uncovered in training). GT annotations are used for labels and evaluation only.

usage: python scripts/hgo/hazard_coverage.py --run exp/hgo/runs/H32_turn_vru_s0 [--max-scenes 0] [--procs 16]
writes <run>/hazard_coverage.json, <run>/hazard_coverage.csv
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
DET_RANGE = 32.0
CLS_NAMES = {0: "vehicle", 1: "pedestrian", 2: "bicycle", 3: "static"}
G = {}


def init(cache_dir, pkl_dir, sensors, dump, gate):
    from train import metric_cache_paths
    G.update(mc=metric_cache_paths(ROOT / cache_dir), pkl_dir=ROOT / pkl_dir, sensors=ROOT / sensors, gate=gate)
    z = np.load(ROOT / dump, allow_pickle=True)
    G.update(props=z["props"], slot_box=z["slot_box"], slot_cls=z["slot_cls"], keep=(z["slot_risk"] > -29).any(1))


def match(sb, sc, gb, gc, gate):
    """Kept slots (n,2 xy; classes) vs GT objects (m,2; classes): GT index matched to each slot or -1 (gated, class-aware)."""
    out = np.full(len(sb), -1)
    if not len(sb) or not len(gb):
        return out
    d = np.linalg.norm(sb[:, None] - gb[None], axis=-1)
    bad = (d > gate) | (sc[:, None] != gc[None])
    r, c = linear_sum_assignment(np.where(bad, 1e6, d))
    ok = ~bad[r, c]
    out[r[ok]] = c[ok]
    return out


def one_log(args):
    log, items = args
    from ade2e_wm.hgo.data import LogData
    from ade2e_wm.hgo.geometry import to_frame
    from ade2e_wm.hgo.pdm import proposal_labels
    ld = LogData(G["pkl_dir"] / f"{log}.pkl", G["sensors"])
    rows, props_rows = [], []
    for i, tok, t in items:
        path = G["mc"].get(tok)
        if path is None:
            continue
        a = ld.anns[t]
        T = np.linalg.inv(ld.e2g[t]) @ ld.l2g[t]
        box = to_frame(T, a["box"])
        cls, track = a["cls"], a["track"]
        try:
            lab = proposal_labels(path, G["props"][i].astype(np.float64), list(track))
        except Exception as e:  # keep going; count the failure
            rows.append({"token": tok, "log": log, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        fault, ttc = lab["fault"] > 0, (lab["ttc_pair"] > 0) & lab["ttc_ok"][:, None]
        inside = (np.abs(box[:, 0]) <= DET_RANGE) & (np.abs(box[:, 1]) <= DET_RANGE)
        dist = np.hypot(box[:, 0], box[:, 1])
        ins = np.nonzero(inside)[0]
        rank = np.full(len(box), -1)
        rank[ins[np.argsort(dist[ins], kind="stable")]] = np.arange(len(ins))
        keep = G["keep"][i]
        sb, sc = G["slot_box"][i][keep, :2].astype(np.float64), G["slot_cls"][i][keep].astype(int)
        m = match(sb, sc, box[ins, :2], cls[ins], G["gate"])
        matched = np.zeros(len(box), bool)
        matched[ins[m[m >= 0]]] = True
        full = keep.sum() == keep.shape[0]
        far_kept = float(np.hypot(sb[:, 0], sb[:, 1]).max()) if len(sb) else 0.0
        for k in np.nonzero(fault.any(0) | ttc.any(0))[0]:
            if matched[k]:
                cat = "covered"
            elif not inside[k]:
                cat = "out_of_range"
            elif full and dist[k] > far_kept:
                cat = "cut_by_selection"
            else:
                cat = "missed"
            rows.append({"token": tok, "log": log, "cls": CLS_NAMES.get(int(cls[k]), str(cls[k])), "fault": bool(fault[:, k].any()),
                         "ttc": bool(ttc[:, k].any()), "n_props": int((fault[:, k] | ttc[:, k]).sum()), "dist": float(dist[k]),
                         "x": float(box[k, 0]), "y": float(box[k, 1]), "gt_rank": int(rank[k]), "category": cat})
        unsafe = fault.any(1) | (lab["fault_unmatched"] > 0)
        covered = (fault & matched[None]).any(1)
        props_rows.append({"token": tok, "unsafe": int(unsafe.sum()), "uncovered": int((unsafe & ~covered).sum()),
                           "kept_slots": int(keep.sum()), "n_tracks": len(track), "n_inside": int(inside.sum())})
    return rows, props_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("exp/hgo/runs/H32_turn_vru_s0"))
    ap.add_argument("--dump", default="navtest_plans.npz")
    ap.add_argument("--metric-cache", default="exp/metric_cache/navtest_navsim2.2_np1.26.4")
    ap.add_argument("--pkl-dir", default="data/navsim_logs/test")
    ap.add_argument("--sensors", default="data/sensor_blobs/test")
    ap.add_argument("--events", default="data/_lists/cg/events_test.parquet", help="token -> (log, frame index)")
    ap.add_argument("--gate", type=float, default=2.0)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    dump = a.run / a.dump
    z = np.load(ROOT / dump, allow_pickle=True)
    toks = [str(t) for t in z["token"]]
    ev = pd.read_parquet(ROOT / a.events, columns=["token", "log", "t"]).set_index("token")
    idx = [i for i, t in enumerate(toks) if t in ev.index]
    if a.max_scenes:
        idx = list(np.random.default_rng(0).choice(idx, min(a.max_scenes, len(idx)), replace=False))
    by_log = {}
    for i in idx:
        r = ev.loc[toks[i]]
        by_log.setdefault(r["log"], []).append((i, toks[i], int(r["t"])))
    with Pool(a.procs, initializer=init, initargs=(a.metric_cache, a.pkl_dir, a.sensors, str(dump), a.gate)) as p:
        res, t0 = [], time.time()
        for j, r in enumerate(p.imap_unordered(one_log, list(by_log.items()), chunksize=1)):
            res.append(r)
            print(f"log {j + 1}/{len(by_log)} done, {sum(len(x[1]) for x in res)} scenes, {time.time() - t0:.0f} s", flush=True)
    rows = [r for rs, _ in res for r in rs]
    prow = [r for _, ps in res for r in ps]
    cols = ["token", "log", "cls", "fault", "ttc", "n_props", "dist", "x", "y", "gt_rank", "category"]
    hz = pd.DataFrame([r for r in rows if "category" in r], columns=cols)
    hz[["fault", "ttc"]] = hz[["fault", "ttc"]].astype(bool)
    errors = [r for r in rows if "error" in r]
    err = len(errors)
    if errors:
        print("label errors (first 3):", errors[:3])
    hz.to_csv(ROOT / a.run / "hazard_coverage.csv", index=False)
    pp = pd.DataFrame(prow, columns=["token", "unsafe", "uncovered", "kept_slots", "n_tracks", "n_inside"])
    share = lambda m: round(float(m.mean()) * 100, 1) if len(m) else None
    out = {"dump": str(dump), "scenes": len(pp), "label_errors": err, "hazards": len(hz),
           "hazards_fault": int(hz["fault"].sum()), "hazards_ttc_only": int((~hz["fault"] & hz["ttc"]).sum()),
           "category_pct": {c: share(hz["category"] == c) for c in ["covered", "missed", "cut_by_selection", "out_of_range"]},
           "category_pct_fault": {c: share(hz.loc[hz["fault"], "category"] == c)
                                  for c in ["covered", "missed", "cut_by_selection", "out_of_range"]},
           "by_class": {c: {"n": int((hz["cls"] == c).sum()),
                            **{k: share(hz.loc[hz["cls"] == c, "category"] == k) for k in ["covered", "missed", "out_of_range"]}}
                        for c in sorted(hz["cls"].unique())},
           "gt_nearest_k_keeps_pct": {k: share((hz["gt_rank"] >= 0) & (hz["gt_rank"] < k)) for k in (16, 32, 64)},
           "unsafe_props": int(pp["unsafe"].sum()), "unsafe_uncovered_pct": share(pp["uncovered"].sum() / max(pp["unsafe"].sum(), 1) * np.ones(1)),
           "kept_slots_mean": round(float(pp["kept_slots"].mean()), 1), "tracks_inside_mean": round(float(pp["n_inside"].mean()), 1)}
    (ROOT / a.run / "hazard_coverage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
