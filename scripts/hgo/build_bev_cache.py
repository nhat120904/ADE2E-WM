"""BEV semantic targets for HGO (docs/METHOD.md): 64 x 64 cells of 1 m over |x|, |y| <= 32 m in each frame's ego frame
(row = x, col = y), TransFuser's classes: 0 none, 1 road, 2 walkway, 3 centerline, 4 static object, 5 vehicle,
6 pedestrian (later classes overwrite earlier ones). Map from the nuPlan maps on disk, objects from the GT boxes.

usage: python scripts/hgo/build_bev_cache.py --logs <list> --pkl-dir data/navsim_logs/trainval --out exp/hgo/bev --procs 16
Writes <out>/<log>.npz with token[F] and bev[F,64,64] uint8 for every frame of the log.
"""

import argparse
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

R, RANGE = 64, 32.0
STATIC = {"traffic_cone", "barrier", "czone_sign", "generic_object"}
VEHICLE = {"vehicle"}
PED = {"pedestrian", "bicycle"}
_maps = {}


def map_api(loc):
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    if loc not in _maps:
        _maps[loc] = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"], os.environ["NUPLAN_MAP_VERSION"], loc)
    return _maps[loc]


def to_px(xy_ego):
    """ego (N,2) -> cv2 points (N,2) as (col, row)."""
    return np.stack([xy_ego[:, 1] + RANGE, xy_ego[:, 0] + RANGE], 1)


def frame_bev(f, api):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map import SemanticMapLayer as L
    e2g = np.asarray(f["ego2global"], np.float64)
    g2e = np.linalg.inv(e2g)
    xy = lambda pts: (np.asarray(pts)[:, :2] @ g2e[:2, :2].T) + g2e[:2, 3]
    bev = np.zeros((R, R), np.uint8)
    centre = Point2D(e2g[0, 3], e2g[1, 3])
    objs = api.get_proximal_map_objects(centre, RANGE * 1.5, [L.LANE, L.INTERSECTION, L.WALKWAYS, L.LANE_CONNECTOR])
    for layers, val in [((L.LANE, L.INTERSECTION), 1), ((L.WALKWAYS,), 2)]:
        for layer in layers:
            for o in objs[layer]:
                ext = np.asarray(o.polygon.exterior.coords)
                cv2.fillPoly(bev, [np.round(to_px(xy(ext)) - 0.5).astype(np.int32)], val)
    for layer in (L.LANE, L.LANE_CONNECTOR):
        for o in objs[layer]:
            line = np.asarray(o.baseline_path.linestring.coords)
            cv2.polylines(bev, [np.round(to_px(xy(line)) - 0.5).astype(np.int32)], False, 3, 1)
    a = f["anns"]
    boxes = np.asarray(a["gt_boxes"], np.float64).reshape(-1, 7)
    names = np.asarray(a["gt_names"])
    l2e = np.asarray(f["lidar2ego"], np.float64)
    for cls_set, val in [(STATIC, 4), (VEHICLE, 5), (PED, 6)]:
        for b in boxes[np.isin(names, list(cls_set))]:
            c = l2e[:3, :3] @ b[:3] + l2e[:3, 3]
            yaw = b[6] + np.arctan2(l2e[1, 0], l2e[0, 0])
            dx = np.array([1, 1, -1, -1]) * b[3] / 2
            dy = np.array([1, -1, -1, 1]) * b[4] / 2
            pts = np.stack([c[0] + np.cos(yaw) * dx - np.sin(yaw) * dy, c[1] + np.sin(yaw) * dx + np.cos(yaw) * dy], 1)
            cv2.fillPoly(bev, [np.round(to_px(pts) - 0.5).astype(np.int32)], val)
    return bev


def one_log(args):
    log, pkl_dir, out = args
    dst = Path(out) / f"{log}.npz"
    if dst.exists():
        return log, -1
    frames = pickle.load(open(Path(pkl_dir) / f"{log}.pkl", "rb"))
    api = map_api(frames[0]["map_location"])
    bev = np.stack([frame_bev(f, api) for f in frames])
    np.savez_compressed(dst.with_suffix(".tmp.npz"), token=np.array([f["token"] for f in frames]), bev=bev)
    os.replace(dst.with_suffix(".tmp.npz"), dst)
    return log, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--pkl-dir", required=True)
    ap.add_argument("--out", default="exp/hgo/bev")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    logs = Path(a.logs).read_text().split()
    with Pool(a.procs) as p:
        for i, (log, n) in enumerate(p.imap_unordered(one_log, [(l, a.pkl_dir, a.out) for l in logs])):
            print(i, log, n, flush=True)


if __name__ == "__main__":
    main()
