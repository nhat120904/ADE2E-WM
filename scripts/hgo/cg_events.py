"""Event labels for the redesigned compositional split (docs/CG_SPLIT.md). GT annotations only: labels and split
predicates, never model inputs.

For every frame t of every log of a split (trainval: all 1,310 logs, images on disk or not; test: the navtest logs), in
the global frame, using the contiguous logged future (frames 0.5 +- 0.1 s apart, at most 20 = 10 s):
  ego      man4 / man10: maneuver of the logged ego path over 4 s / over the whole contiguous future (thresholds of
           hgo/splits.py: stop if < 2 m travelled, left / right if the heading changes by > 20 deg, else straight);
           dyaw4, dyaw10 (deg), dist4, dist10 (m), lat4 (lateral offset at 4 s in the t0 frame, m), speed (m/s), cmd
           (argmax of the driving command: 0 left, 1 straight, 2 right, 3 unknown), city, session (drive session = log
           name without the two trailing frame indices), K (contiguous future frames), Hh (contiguous history frames, <= 3),
           has_img (camera folder of the log on disk)
  agents   relative to the ego centre path of the same horizon (4 s and 10 s), per group
             vru        pedestrian or bicycle
             veh_cross  vehicle crossing the ego path (heading 45-135 deg from the ego path at the conflict point)
             veh_opp    vehicle with opposing heading (> 135 deg; e.g. oncoming traffic in an unprotected turn)
             veh_same   vehicle with the same heading (< 45 deg; following, merging, cut-in)
           pet_<h>_<g>: smallest |t_agent - t_ego| over (agent position, ego path point) pairs closer than the group's
           radius (VRU 2.5 m, vehicle 3.0 m between centres; inf if none), tconf_<h>_<g>: the ego time of that pair (s),
           dpath_<h>_<g>: closest approach of any agent of the group to the path (m)
           petx_<h>_<g> / tconfx_<h>_<g>: the same over agents whose trajectory CROSSES the ego path (crossing(): positions
           on both sides of the path within 8 m; added 12/11 after the sample figures showed oncoming traffic in the
           next lane and VRUs walking along the kerb counted as conflicts)
           static_<h>: closest approach of a static object (cone, barrier, sign, generic object) to the path (m)
  counts   n_vru, n_veh: VRUs / vehicles within 32 m at t; lead: distance (m) to the nearest same-heading vehicle ahead in
           the ego lane (|y| < 1.8 m, 0 < x <= 40 m in the t0 ego frame), inf if none
  map      inter_f: ego centre in the nuPlan INTERSECTION layer at frame t (--map; NaN if the map query fails), inter4 /
           inter10: at any frame of 0..4 s / of the contiguous future

usage: python scripts/hgo/cg_events.py --split test [--logs-max 3] [--procs 16] [--map]
writes data/_lists/cg/events_<split>.parquet
"""

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REAR_TO_CENTER = 1.461   # ego rear axle -> box centre (hgo.geometry.EGO_REAR_TO_CENTER)
DT, H4, H10, H_HIST = 0.5, 8, 20, 3
NEAR_M = 50.0            # agents considered at t: within this distance of the ego centre
RADIUS = {"vru": 2.5, "veh": 3.0}
VRU = {"pedestrian", "bicycle"}
STATIC = {"traffic_cone", "barrier", "czone_sign", "generic_object"}
GROUPS = ["vru", "veh_cross", "veh_opp", "veh_same"]
MAPS = {}


def wrap180(a):
    return np.abs((a + 180.0) % 360.0 - 180.0)


def maneuver(d, dyaw):
    if d is None or np.isnan(d):
        return None
    return "stop" if d < 2.0 else "left" if dyaw > 20 else "right" if dyaw < -20 else "straight"


def in_intersection(city, xy):
    """Ego centre points (n,2) in the nuPlan INTERSECTION layer; NaN where the query fails."""
    try:
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        if city not in MAPS:
            MAPS[city] = get_maps_api(str(ROOT / "data/maps"), "nuplan-maps-v1.0", city)
        m = MAPS[city]
        return np.array([float(m.is_in_layer(Point2D(float(x), float(y)), SemanticMapLayer.INTERSECTION)) for x, y in xy])
    except Exception as e:  # keep the other labels; report once per process
        if city not in MAPS:
            print(f"map query failed for {city}: {type(e).__name__}: {e}", file=sys.stderr)
            MAPS[city] = None
        return np.full(len(xy), np.nan)


def path_conflict(P, valid, C, radius):
    """Agents' positions P (k,n,2) at times k*DT (valid (k,n)), ego centre path C (m,2) at times j*DT.
    Returns per agent: dmin (closest approach to the path), pet (smallest |t_agent - t_ego| over pairs closer than radius),
    tconf (ego time of that pair), kconf / sconf (agent frame / path segment index of that pair)."""
    k, n = P.shape[:2]
    if len(C) == 1:
        A, AB = C[:1], np.zeros((1, 2))
        te_a, te_b = np.zeros(1), np.zeros(1)
    else:
        A, AB = C[:-1], C[1:] - C[:-1]
        te_a, te_b = np.arange(len(C) - 1) * DT, np.arange(1, len(C)) * DT
    D = P[:, :, None, :] - A[None, None]                                  # k,n,s,2
    u = np.clip((D * AB[None, None]).sum(-1) / ((AB ** 2).sum(-1)[None, None] + 1e-9), 0.0, 1.0)
    d = np.linalg.norm(D - u[..., None] * AB[None, None], axis=-1)       # k,n,s
    d = np.where(valid[:, :, None], d, np.inf)
    te = te_a[None, None] + u * (te_b - te_a)[None, None]
    tp = (np.arange(k) * DT)[:, None, None]
    pet = np.where(d < radius, np.abs(tp - te), np.inf)
    flat = pet.transpose(1, 0, 2).reshape(n, -1)
    j = flat.argmin(1)
    kc, sc = np.unravel_index(j, (k, d.shape[2]))
    return (d.transpose(1, 0, 2).reshape(n, -1).min(1), flat[np.arange(n), j], te[kc, np.arange(n), sc], kc, sc)


def crossing(P, valid, C, max_off=8.0):
    """Per agent: does its trajectory cross the ego centre path C (m,2)? Signed lateral offset of each valid position
    to its nearest path segment (positions projecting beyond either end of the path are ignored); crossing = two
    positions within max_off m of the path on opposite sides. Excludes agents that only pass alongside (oncoming traffic
    in the next lane, VRUs walking along the kerb or behind the ego on its path)."""
    k, n = P.shape[:2]
    out = np.zeros(n, bool)
    if len(C) < 2:
        return out
    A, AB = C[:-1], C[1:] - C[:-1]
    L2 = (AB ** 2).sum(-1)
    seg_ok = L2 > 0.05 ** 2
    if not seg_ok.any():
        return out
    D = P[:, :, None, :] - A[None, None]                                  # k,n,s,2
    u_raw = (D * AB[None, None]).sum(-1) / np.maximum(L2, 1e-9)[None, None]
    u = np.clip(u_raw, 0.0, 1.0)
    dist = np.linalg.norm(D - u[..., None] * AB[None, None], axis=-1)
    dist = np.where(seg_ok[None, None], dist, np.inf)
    s = dist.argmin(-1)                                                   # k,n
    ii, jj = np.meshgrid(np.arange(k), np.arange(n), indexing="ij")
    ur = u_raw[ii, jj, s]
    first, last = np.nonzero(seg_ok)[0][[0, -1]]
    beyond = ((s == first) & (ur < 0)) | ((s == last) & (ur > 1))
    ab = AB[s]
    d = (ab[..., 0] * D[ii, jj, s, 1] - ab[..., 1] * D[ii, jj, s, 0]) / np.sqrt(np.maximum(L2[s], 1e-9))
    ok = valid & ~beyond & (np.abs(d) <= max_off)
    for j in range(n):
        dj = d[ok[:, j], j]
        out[j] = len(dj) > 1 and (dj > 0.3).any() and (dj < -0.3).any()
    return out


def one(args):
    split, log, use_map = args
    frames = pickle.load(open(ROOT / f"data/navsim_logs/{split}/{log}.pkl", "rb"))
    F = len(frames)
    city = frames[0]["map_location"]
    ts = np.array([f["timestamp"] for f in frames], np.int64)
    gap_ok = np.abs(np.diff(ts) / 1e6 - DT) <= 0.1
    e2g = np.stack([np.asarray(f["ego2global"], np.float64) for f in frames])
    ctr = (e2g @ np.array([REAR_TO_CENTER, 0.0, 0.0, 1.0]))[:, :2]
    eyaw = np.degrees(np.arctan2(e2g[:, 1, 0], e2g[:, 0, 0]))
    # tracks: global centre and heading per frame
    tid, cls_of = {}, []
    rec = []
    for f_i, f in enumerate(frames):
        an = f["anns"]
        names = np.asarray(an["gt_names"])
        box = np.asarray(an["gt_boxes"], np.float64).reshape(-1, 7)
        l2g = np.asarray(f["lidar2global"], np.float64)
        xy = (box[:, :3] @ l2g[:3, :3].T + l2g[:3, 3])[:, :2]
        yaw = np.degrees(box[:, 6] + np.arctan2(l2g[1, 0], l2g[0, 0]))
        for r, tok in enumerate(an["track_tokens"]):
            nm = str(names[r])
            g = "vru" if nm in VRU else "veh" if nm == "vehicle" else "static" if nm in STATIC else None
            if g is None:
                continue
            if tok not in tid:
                tid[tok] = len(tid)
                cls_of.append(g)
            rec.append((f_i, tid[tok], xy[r, 0], xy[r, 1], yaw[r]))
    NT = len(tid)
    POS = np.full((F, NT, 2), np.nan, np.float32)
    YAW = np.full((F, NT), np.nan, np.float32)
    if rec:
        r = np.asarray(rec, np.float64)
        fi, ti = r[:, 0].astype(int), r[:, 1].astype(int)
        POS[fi, ti] = r[:, 2:4]
        YAW[fi, ti] = r[:, 4]
    cls_of = np.asarray(cls_of)
    inside = in_intersection(city, ctr) if use_map else np.full(F, np.nan)
    has_img = (ROOT / f"data/sensor_blobs/{split}/{log}").is_dir()
    session = "_".join(log.split("_")[:-2])
    rows = []
    for t in range(F):
        K = 0
        while K < H10 and t + K + 1 < F and gap_ok[t + K]:
            K += 1
        Hh = 0
        while Hh < H_HIST and t - Hh - 1 >= 0 and gap_ok[t - Hh - 1]:
            Hh += 1
        E0inv = np.linalg.inv(e2g[t])
        row = {"token": frames[t]["token"], "log": log, "t": t, "session": session, "city": city, "K": K, "Hh": Hh,
               "has_img": has_img, "cmd": int(np.argmax(frames[t]["driving_command"])),
               "speed": float(np.hypot(*frames[t]["ego_dynamic_state"][:2])), "inter_f": inside[t],
               "inter4": float(np.nanmax(inside[t:t + min(H4, K) + 1])) if use_map else np.nan,
               "inter10": float(np.nanmax(inside[t:t + K + 1])) if use_map else np.nan}
        for h, name in ((H4, "4"), (K, "10")):
            if K >= H4:
                T = E0inv @ e2g[t + min(h, K)]
                d, dy = float(np.hypot(T[0, 3], T[1, 3])), float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
                row.update({f"man{name}": maneuver(d, dy), f"dyaw{name}": dy, f"dist{name}": d})
                if name == "4":
                    row["lat4"] = float(T[1, 3])
            else:
                row.update({f"man{name}": None, f"dyaw{name}": np.nan, f"dist{name}": np.nan})
                if name == "4":
                    row["lat4"] = np.nan
        # agents near the ego at t
        p0 = POS[t]
        near = np.where(~np.isnan(p0[:, 0]) & (np.hypot(*(p0 - ctr[t]).T) <= NEAR_M))[0]
        loc = (E0inv[:2, :2] @ (p0[near] - e2g[t][:2, 3]).T).T if len(near) else np.zeros((0, 2))
        cn = cls_of[near]
        dn = np.hypot(*(p0[near] - ctr[t]).T) if len(near) else np.zeros(0)
        row["n_vru"] = int(((cn == "vru") & (dn <= 32)).sum())
        row["n_veh"] = int(((cn == "veh") & (dn <= 32)).sum())
        ryaw0 = wrap180(YAW[t, near] - eyaw[t]) if len(near) else np.zeros(0)
        lead = (cn == "veh") & (loc[:, 0] > 0) & (loc[:, 0] <= 40) & (np.abs(loc[:, 1]) < 1.8) & (ryaw0 < 45)
        row["lead"] = float(loc[lead, 0].min()) if lead.any() else np.inf
        for h, name in ((H4, "4"), (K, "10")):
            hk = min(h, K)
            C = ctr[t:t + hk + 1]
            for g in GROUPS:
                row[f"pet{name}_{g}"], row[f"tconf{name}_{g}"], row[f"dpath{name}_{g}"] = np.inf, np.nan, np.inf
                row[f"petx{name}_{g}"], row[f"tconfx{name}_{g}"] = np.inf, np.nan
            st = near[cn == "static"]
            if len(st):
                dmin, *_ = path_conflict(POS[t][st][None].astype(np.float64), np.ones((1, len(st)), bool), C, 1e9)
                row[f"static{name}"] = float(dmin.min())
            else:
                row[f"static{name}"] = np.inf
            for base in ("vru", "veh"):
                idx = near[cn == base]
                if not len(idx):
                    continue
                P = POS[t:t + hk + 1, idx].astype(np.float64)
                valid = ~np.isnan(P[..., 0])
                P = np.nan_to_num(P)
                dmin, pet, tconf, kc, sc = path_conflict(P, valid, C, RADIUS[base])
                cross = crossing(P, valid, C)
                if base == "vru":
                    groups = {"vru": np.ones(len(idx), bool)}
                else:  # heading of the vehicle at its conflict point relative to the ego path there
                    seg = C[np.minimum(sc + 1, len(C) - 1)] - C[sc]
                    pyaw = np.where(np.hypot(*seg.T) > 0.1, np.degrees(np.arctan2(seg[:, 1], seg[:, 0])), eyaw[t + sc])
                    ay = YAW[t + kc, idx]
                    rel = wrap180(np.where(np.isnan(ay), YAW[t, idx], ay) - pyaw)
                    groups = {"veh_cross": (rel >= 45) & (rel <= 135), "veh_opp": rel > 135, "veh_same": rel < 45}
                for g, m in groups.items():
                    if not m.any():
                        continue
                    row[f"dpath{name}_{g}"] = float(dmin[m].min())
                    j = int(np.argmin(np.where(m, pet, np.inf)))
                    if np.isfinite(pet[j]) and m[j]:
                        row[f"pet{name}_{g}"], row[f"tconf{name}_{g}"] = float(pet[j]), float(tconf[j])
                    mx = m & cross
                    jx = int(np.argmin(np.where(mx, pet, np.inf)))
                    if mx[jx] and np.isfinite(pet[jx]):
                        row[f"petx{name}_{g}"], row[f"tconfx{name}_{g}"] = float(pet[jx]), float(tconf[jx])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["trainval", "test"])
    ap.add_argument("--logs-max", type=int, default=0, help="smoke: first N logs")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--map", action="store_true", help="intersection labels from the nuPlan map")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    logs = sorted(p.stem for p in (ROOT / f"data/navsim_logs/{a.split}").glob("*.pkl"))[: a.logs_max or None]
    with Pool(a.procs) as p:
        rows = [r for rs in p.imap_unordered(one, [(a.split, l, a.map) for l in logs], chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    out = Path(a.out) if a.out else ROOT / f"data/_lists/cg/events_{a.split}{'_smoke' if a.logs_max else ''}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{a.split}: {len(df)} frames, {df['log'].nunique()} logs -> {out}")


if __name__ == "__main__":
    main()
