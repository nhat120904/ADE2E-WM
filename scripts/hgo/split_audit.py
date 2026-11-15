"""Audit of the compositional split (docs/METHOD_RATIONALE.md §4b, tests 3-4). Do held-out 'turn x VRU' scenes contain a
real ego-VRU interaction, and do kept training windows contain the same interaction under another label (for example a car
waiting at a turn for crossing pedestrians is labelled 'stop' by the 4 s maneuver rule)? GT annotations only (labels,
never model inputs).

Per window t0 (the windows of data/_lists/hgo_factors_<set>.csv), in global coordinates:
  cmd            argmax of the t0 driving command one-hot (its order is read off the cmd x maneuver table in the summary)
  K              contiguous future frames (0.5 +- 0.1 s apart), at most 20 (10 s)
  man10          maneuver of the logged ego path at t0 + K/2 s with the thresholds of splits.py (stop / left / right / straight)
  n_vru          pedestrians and bicycles within 32 m of the ego centre at t0
  vru_dmin       min over those VRUs and 0..4 s of the same-time distance between VRU and ego centres (m)
  path4, pet4    closest approach (m) of a VRU position (0..4 s) to the ego centre path (0..4 s), and the smallest
                 |t_vru - t_ego| (s) over (VRU point, path point) pairs closer than --path-m (inf if none)
  path10, pet10  the same with VRU positions and ego path up to K frames (10 s): catches yielding (the ego passes later)
A conflict is path < --path-m and pet < a PET threshold (summary: 3 s and 1.5 s).

usage: python scripts/hgo/split_audit.py --sets test train [--logs-max N] [--procs 16] [--split turn_vru]
writes data/_lists/hgo_vru_audit_<set>.csv and docs/hgo_vru_audit.md (summary)
"""

import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PKL_DIR = {"train": "trainval", "val": "trainval", "test": "test"}
VRU_NAMES = ["pedestrian", "bicycle"]
REAR_TO_CENTER = 1.461  # ego rear axle -> box centre (hgo.geometry.EGO_REAR_TO_CENTER)
DT, H4, H10, NEAR_M = 0.5, 8, 20, 32.0


def maneuver(e2g, t, k):
    T = np.linalg.inv(e2g[t]) @ e2g[t + k]
    d, dyaw = float(np.hypot(T[0, 3], T[1, 3])), float(np.arctan2(T[1, 0], T[0, 0]))
    return "stop" if d < 2.0 else "left" if dyaw > np.deg2rad(20) else "right" if dyaw < -np.deg2rad(20) else "straight"


def path_stats(P, tp, C, tc, path_m):
    """VRU points P (n,2) at times tp, ego centre path C (m,2) at times tc: min point-to-path distance and the smallest
    |t_vru - t_ego| over pairs closer than path_m, the ego time interpolated along the segment."""
    if len(C) == 1:
        d = np.linalg.norm(P - C[0], axis=-1)[:, None]
        te = np.full_like(d, tc[0])
    else:
        A, AB = C[:-1], C[1:] - C[:-1]
        u = np.clip(((P[:, None] - A[None]) * AB[None]).sum(-1) / ((AB ** 2).sum(-1)[None] + 1e-9), 0.0, 1.0)
        d = np.linalg.norm(P[:, None] - (A[None] + u[..., None] * AB[None]), axis=-1)
        te = tc[:-1][None] + u * (tc[1:] - tc[:-1])[None]
    pet = np.where(d < path_m, np.abs(tp[:, None] - te), np.inf)
    return float(d.min()), float(pet.min())


def one(args):
    split, log, tokens, path_m = args
    frames = pickle.load(open(ROOT / f"data/navsim_logs/{split}/{log}.pkl", "rb"))
    index_of = {f["token"]: i for i, f in enumerate(frames)}
    F = len(frames)
    ts = np.array([f["timestamp"] for f in frames], np.int64)
    gap_ok = np.abs(np.diff(ts) / 1e6 - DT) <= 0.1
    e2g = np.stack([np.asarray(f["ego2global"], np.float64) for f in frames])
    ctr = (e2g @ np.array([REAR_TO_CENTER, 0.0, 0.0, 1.0]))[:, :2]
    vru = []  # per frame: track token -> global xy of pedestrians / bicycles
    for f in frames:
        an = f["anns"]
        m = np.isin(np.asarray(an["gt_names"]), VRU_NAMES)
        box = np.asarray(an["gt_boxes"], np.float64).reshape(-1, 7)[m]
        l2g = np.asarray(f["lidar2global"], np.float64)
        xy = (box[:, :3] @ l2g[:3, :3].T + l2g[:3, 3])[:, :2]
        vru.append(dict(zip(np.asarray(an["track_tokens"])[m].tolist(), xy)))
    rows = []
    for tok in tokens:
        t = index_of[tok]
        K = 0
        while K < H10 and t + K + 1 < F and gap_ok[t + K]:
            K += 1
        row = {"token": tok, "cmd": int(np.argmax(frames[t]["driving_command"])), "K": K,
               "man10": maneuver(e2g, t, K) if K >= H4 else None, "n_vru": 0, "vru_dmin": np.inf,
               "path4": np.inf, "pet4": np.inf, "path10": np.inf, "pet10": np.inf}
        near = [k for k, xy in vru[t].items() if np.hypot(*(xy - ctr[t])) <= NEAR_M]
        row["n_vru"] = len(near)
        k4 = min(H4, K)
        C4, C10 = ctr[t:t + k4 + 1], ctr[t:t + K + 1]
        tc4, tc10 = np.arange(k4 + 1) * DT, np.arange(K + 1) * DT
        for trk in near:
            ks = [k for k in range(K + 1) if trk in vru[t + k]]
            P = np.stack([vru[t + k][trk] for k in ks])
            tp = np.asarray(ks, np.float64) * DT
            m4 = np.asarray(ks) <= k4
            same = [np.hypot(*(vru[t + k][trk] - ctr[t + k])) for k in ks if k <= k4]
            row["vru_dmin"] = min(row["vru_dmin"], min(same))
            d4, p4 = path_stats(P[m4], tp[m4], C4, tc4, path_m)
            d10, p10 = path_stats(P, tp, C10, tc10, path_m)
            row["path4"], row["pet4"] = min(row["path4"], d4), min(row["pet4"], p4)
            row["path10"], row["pet10"] = min(row["path10"], d10), min(row["pet10"], p10)
        rows.append(row)
    return rows


def pct(m):
    return f"{100 * float(np.mean(m)):.1f}" if len(m) else "-"


def group_table(df, path_m):
    out = ["| maneuver (4 s) | VRU in corridor | N | VRU ≤ 5 m | conflict 4 s, PET < 3 s | PET < 1.5 s | conflict 10 s, PET < 3 s |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for (man, v), g in df.groupby(["maneuver", "vru"]):
        c4 = (g.path4 < path_m) & (g.pet4 < 3.0)
        c4s = (g.path4 < path_m) & (g.pet4 < 1.5)
        c10 = (g.path10 < path_m) & (g.pet10 < 3.0)
        out.append(f"| {man} | {v} | {len(g)} | {pct(g.vru_dmin <= 5)} | {pct(c4)} | {pct(c4s)} | {pct(c10)} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["test", "train"])
    ap.add_argument("--split", default="turn_vru")
    ap.add_argument("--path-m", type=float, default=2.5, help="VRU-to-ego-path distance that counts as on the path (m)")
    ap.add_argument("--logs-max", type=int, default=0, help="smoke test: first N logs per set")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out-md", default="docs/hgo_vru_audit.md")
    a = ap.parse_args()
    lists = ROOT / "data/_lists"
    md = ["# VRU interaction audit of the compositional split (generated by scripts/hgo/split_audit.py)", "",
          f"On the path: VRU centre within {a.path_m} m of the ego centre path. Conflict: on the path and |t_vru - t_ego| below "
          "the PET threshold. 4 s: VRU positions and ego path over 0-4 s; 10 s: both over the contiguous logged future (≤ 10 s).",
          "GT annotations; percentages of windows.", ""]
    train_used = set((lists / f"hgo_train_{a.split}.txt").read_text().split())
    for name in a.sets:
        fac = pd.read_csv(lists / f"hgo_factors_{name}.csv")
        logs = sorted(fac["log"].unique())[: a.logs_max or None]
        fac = fac[fac["log"].isin(logs)]
        jobs = [(PKL_DIR[name], log, g["token"].tolist(), a.path_m) for log, g in fac.groupby("log")]
        with Pool(a.procs) as p:
            rows = [r for rs in p.map(one, jobs) for r in rs]
        df = fac.merge(pd.DataFrame(rows), on="token")
        df["heldout"] = df["maneuver"].isin(["left", "right"]) & (df["vru"] == "vru")
        if name == "train":
            df["used"] = df["token"].isin(train_used)
        out = lists / (f"hgo_vru_audit_{name}.csv" if not a.logs_max else f"hgo_vru_audit_{name}_smoke.csv")
        df.to_csv(out, index=False)
        md += [f"## {name}: {len(df)} windows, {df['log'].nunique()} logs (`{out.relative_to(ROOT)}`)", ""]
        md += ["Driving command (argmax index) × 4 s maneuver:", "", "```",
               pd.crosstab(df["cmd"], df["maneuver"]).to_string(), "```", ""]
        md += group_table(df, a.path_m) + [""]
        if name == "train":
            kept = df[df["used"]]
            turn10 = kept["man10"].isin(["left", "right"]) & ~kept["maneuver"].isin(["left", "right"])
            c10 = (kept.path10 < a.path_m) & (kept.pet10 < 3.0)
            ho = df[df["heldout"]]
            md += [f"Kept training windows ({len(kept)}): turn within 10 s but labelled stop / straight at 4 s: "
                   f"{int(turn10.sum())}; of these with a VRU conflict over 10 s (PET < 3 s): {int((turn10 & c10).sum())}, "
                   f"of which with a VRU in the corridor at t0: {int((turn10 & c10 & (kept['vru'] == 'vru')).sum())}. "
                   f"Removed held-out windows: {len(ho)}, with a 4 s VRU conflict (PET < 3 s): "
                   f"{int(((ho.path4 < a.path_m) & (ho.pet4 < 3.0)).sum())}, with a 10 s conflict: "
                   f"{int(((ho.path10 < a.path_m) & (ho.pet10 < 3.0)).sum())}.", ""]
        if name == "test" and not a.logs_max:
            csvs = {"H32": ROOT / "exp/hgo/runs/H32_turn_vru_s0/navtest_epdms_main.csv",
                    "DrivoR": ROOT / "exp/drivor_nav2_navtest_full/2026.10.28.12.55.37/2026.10.28.18.44.05.csv"}
            md += ["Official EPDMS (per-scene csv, valid scenes) by group and 4 s conflict (PET < 3 s):", "",
                   "| group | conflict | N | " + " | ".join(csvs) + " |", "|---|---|---:|" + "---:|" * len(csvs)]
            sc = {k: pd.read_csv(v).query("valid == True").set_index("token")["score"] for k, v in csvs.items() if v.exists()}
            turn = df["maneuver"].isin(["left", "right"])
            conf = (df.path4 < a.path_m) & (df.pet4 < 3.0)
            for gname, gm in [("turn, VRU (held-out)", df["heldout"]), ("turn, no VRU", turn & (df["vru"] == "no_vru")),
                              ("straight / stop, VRU", ~turn & (df["vru"] == "vru"))]:
                for cname, cm in [("yes", conf), ("no", ~conf)]:
                    toks = df.loc[gm & cm, "token"]
                    vals = [s.reindex(toks).dropna() for s in sc.values()]
                    md.append(f"| {gname} | {cname} | {len(toks)} | " + " | ".join(f"{100 * v.mean():.2f}" for v in vals) + " |")
            md.append("")
    text = "\n".join(md)
    if not a.logs_max:
        (ROOT / a.out_md).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
