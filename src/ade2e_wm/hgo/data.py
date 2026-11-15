"""HGO samples straight from OpenScene / NAVSIM logs (pkl) and camera JPEGs (docs/METHOD.md).

A sample is one frame t0 at 2 Hz. Model inputs (INPUT_KEYS): the four t0 images, camera calibration, ego status of
t0-1.5 s .. t0 (pose in the t0 ego frame, velocity, acceleration, driving_command); with prev=True (v3) also the four
t0-0.5 s images and their calibration with the camera poses expressed in the t0 ego frame (PREV_KEYS). Everything else is
a target or a tag; a model must only read input_keys(prev) (tests/hgo check this).
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import CAMS, CLASS_OF, scale_intrinsics, to_frame, yaw_of

H_HIST, N_POSES, DT = 4, 8, 0.5
M_MAX = 160          # GT boxes kept per frame (nearest first)
DET_RANGE = 32.0     # pilot detection area: |x|, |y| <= 32 m in the t0 ego frame (v3 passes its grid's area)
PILOT_AREA = (-DET_RANGE, DET_RANGE, -DET_RANGE, DET_RANGE)
INPUT_KEYS = ["img", "cam2ego", "K", "ego"]
PREV_KEYS = ["img_prev", "cam2ego_prev", "K_prev"]


def input_keys(prev=False):
    return INPUT_KEYS + (PREV_KEYS if prev else [])


class LogData:
    """One log: per-frame poses, calibration, annotations; camera files checked once against the directory listing."""

    def __init__(self, pkl: Path, sensor_root: Path):
        frames = pickle.load(open(pkl, "rb"))
        self.name = Path(pkl).stem
        self.sensor_root = Path(sensor_root)
        self.token = [f["token"] for f in frames]
        self.ts = np.array([f["timestamp"] for f in frames], np.int64)
        self.e2g = np.stack([np.asarray(f["ego2global"], np.float64) for f in frames])
        self.l2e = np.stack([np.asarray(f["lidar2ego"], np.float64) for f in frames])
        self.l2g = np.stack([np.asarray(f["lidar2global"], np.float64) for f in frames])
        self.ego_dyn = np.stack([np.asarray(f["ego_dynamic_state"], np.float32)[:4] for f in frames])
        self.cmd = np.stack([np.asarray(f["driving_command"], np.float32) for f in frames])
        self.map_location = frames[0]["map_location"]
        self.cams = [{c: f["cams"][c] for c in CAMS} for f in frames]
        self.anns = []
        for f in frames:
            a = f["anns"]
            names = np.asarray(a["gt_names"])
            keep = np.isin(names, list(CLASS_OF))
            self.anns.append({"box": np.asarray(a["gt_boxes"], np.float32).reshape(-1, 7)[keep],
                              "vel": np.asarray(a["gt_velocity_3d"], np.float32).reshape(-1, 3)[keep, :2],
                              "cls": np.array([CLASS_OF[n] for n in names[keep]], np.int64),
                              "track": np.asarray(a["track_tokens"])[keep]})
        present = {}
        for c in CAMS:
            d = self.sensor_root / self.name / c
            present[c] = set(os.listdir(d)) if d.is_dir() else set()
        self.cam_ok = np.array([all(fc[c]["data_path"] and Path(fc[c]["data_path"]).name in present[c] for c in CAMS)
                                for fc in self.cams])
        self.index_of = {t: i for i, t in enumerate(self.token)}

    def windows(self, horizons, tokens=None, need_future_images=True, check_gaps=True, need_prev_images=False):
        """t0 with H_HIST-1 earlier and N_POSES later frames, t0 cameras present, and cameras at t0 + h for every JEPA
        horizon h (frames) when need_future_images, and at t0-1 when need_prev_images. check_gaps: the frames must be
        0.5 +- 0.1 s apart (training); without it, frames are taken by index as NAVSIM's scene loader does (navtest
        dumping: every scene the devkit scores)."""
        F = len(self.token)
        gap_ok = np.abs(np.diff(self.ts) / 1e6 - DT) <= 0.1
        out = []
        for t in range(H_HIST - 1, F - N_POSES):
            if tokens is not None and self.token[t] not in tokens:
                continue
            if (check_gaps and not gap_ok[t - H_HIST + 1:t + N_POSES].all()) or not self.cam_ok[t]:
                continue
            if need_future_images and not all(self.cam_ok[t + h] for h in horizons):
                continue
            if need_prev_images and not self.cam_ok[t - 1]:
                continue
            out.append(t)
        return out


def load_image(path, wh):
    im = Image.open(path).convert("RGB")
    orig = im.size
    im = im.resize(tuple(wh))  # same call as DrivoR's feature builder
    return torch.from_numpy(np.asarray(im, np.uint8).copy()).permute(2, 0, 1), orig


def frame_cameras(ld: LogData, f, wh, with_images=True):
    """Images uint8 (4,3,H,W), cam2ego (4,4,4) of frame f in frame f's ego frame, K (4,3,3) scaled to wh."""
    imgs, c2e, Ks = [], [], []
    for c in CAMS:
        cd = ld.cams[f][c]
        if with_images:
            img, orig = load_image(ld.sensor_root / cd["data_path"], wh)
            imgs.append(img)
        else:
            orig = (1920, 1080)
        s2l = np.eye(4)
        s2l[:3, :3] = np.asarray(cd["sensor2lidar_rotation"])
        s2l[:3, 3] = np.asarray(cd["sensor2lidar_translation"])
        c2e.append(ld.l2e[f] @ s2l)
        Ks.append(scale_intrinsics(cd["cam_intrinsic"], orig, wh))
    out = {"cam2ego": np.stack(c2e).astype(np.float32), "K": np.stack(Ks).astype(np.float32)}
    if with_images:
        out["img"] = torch.stack(imgs)
    return out


def ego_status(ld: LogData, t):
    """NAVSIM EgoStatus of t-3..t0 as DrivoR builds it: (x, y, yaw) in the t0 ego frame, velocity, acceleration, command."""
    E0 = np.linalg.inv(ld.e2g[t])
    rows = []
    for f in range(t - H_HIST + 1, t + 1):
        T = E0 @ ld.e2g[f]
        rows.append(np.concatenate([[T[0, 3], T[1, 3], yaw_of(T)], ld.ego_dyn[f][:2], ld.ego_dyn[f][2:4], ld.cmd[f]]))
    return np.asarray(rows, np.float32)


def ego_plan(ld: LogData, t):
    """NAVSIM planning target: ego poses t0+0.5 .. 4 s in the t0 ego (rear-axle) frame."""
    E0 = np.linalg.inv(ld.e2g[t])
    out = np.zeros((N_POSES, 3), np.float32)
    for k in range(N_POSES):
        T = E0 @ ld.e2g[t + 1 + k]
        out[k] = [T[0, 3], T[1, 3], yaw_of(T)]
    return out


def prev_cameras(ld: LogData, t, wh):
    """Cameras of frame t-1 with their poses in the t0 ego frame (v3 input). If t-1 has no images (only possible when
    dumping without the gap / camera checks), the t0 cameras are repeated, i.e. a static scene."""
    f = t - 1 if t >= 1 and ld.cam_ok[t - 1] else t
    c = frame_cameras(ld, f, wh)
    T = (np.linalg.inv(ld.e2g[t]) @ ld.e2g[f]).astype(np.float32)
    return {"img_prev": c["img"], "cam2ego_prev": (T[None] @ c["cam2ego"]).astype(np.float32), "K_prev": c["K"]}


def gt_boxes(ld: LogData, t, area=PILOT_AREA, frame=None):
    """GT boxes of frame `frame` (default t) in the t0 ego frame, nearest M_MAX inside area = (x0, x1, y0, y1)."""
    f = t if frame is None else frame
    a = ld.anns[f]
    T = np.linalg.inv(ld.e2g[t]) @ ld.l2g[f]
    box, vel = to_frame(T, a["box"], a["vel"])
    x0, x1, y0, y1 = area
    inside = (box[:, 0] >= x0) & (box[:, 0] <= x1) & (box[:, 1] >= y0) & (box[:, 1] <= y1)
    idx = np.nonzero(inside)[0]
    idx = idx[np.argsort(np.hypot(box[idx, 0], box[idx, 1]), kind="stable")][:M_MAX]
    return box[idx], vel[idx], a["cls"][idx], a["track"][idx]


def pad_boxes(box, vel, cls):
    n = len(box)
    out = {"gt_box": np.zeros((M_MAX, 7), np.float32), "gt_vel": np.zeros((M_MAX, 2), np.float32),
           "gt_cls": np.full(M_MAX, -1, np.int64), "gt_mask": np.arange(M_MAX) < n}
    out["gt_box"][:n], out["gt_vel"][:n], out["gt_cls"][:n] = box, vel, cls
    return out


class HGODataset(Dataset):
    """train=True: t0 images + future images at each horizon (JEPA targets) + all targets.
    train=False: t0 images, plan target and tags only (navtest dumping; frames by index, no timestamp-gap check)."""

    def __init__(self, logs, pkl_dir, sensor_root, wh=(1148, 672), horizons=(2, 4, 8), tokens=None, train=True,
                 bev_dir=None, windows_filter=None, area=PILOT_AREA, prev=False):
        """area: detection area of the GT boxes (the model's perception grid); prev: t-1 cameras as inputs and, when
        training, the t-1 GT boxes in the t0 frame (gt_*_prev, slot-history detection targets)."""
        self.wh, self.horizons, self.train = tuple(wh), tuple(horizons), train
        self.area, self.prev = tuple(area), prev
        self.logs, self.index = [], []
        self.bev_dir = Path(bev_dir) if bev_dir else None
        for l in logs:
            ld = LogData(Path(pkl_dir) / f"{l}.pkl", sensor_root)
            ws = ld.windows(self.horizons, tokens, need_future_images=train, check_gaps=train,
                            need_prev_images=prev and train)
            if windows_filter is not None:
                ws = [t for t in ws if windows_filter(ld, t)]
            self.index += [(len(self.logs), t) for t in ws]
            self.logs.append(ld)
        self._bev = {}
        self.load_future_images = True  # False in the perception warm-up (JEPA targets unused): 4 instead of 16 images

    def __len__(self):
        return len(self.index)

    def bev(self, ld, t):
        if self.bev_dir is None:
            return None
        if ld.name not in self._bev:
            p = self.bev_dir / f"{ld.name}.npz"
            if not p.exists():
                self._bev[ld.name] = None
            else:
                z = np.load(p)
                self._bev[ld.name] = ({str(k): i for i, k in enumerate(z["token"])}, z["bev"])
        e = self._bev[ld.name]
        if e is None or ld.token[t] not in e[0]:
            return None
        return e[1][e[0][ld.token[t]]]

    def __getitem__(self, i):
        li, t = self.index[i]
        ld = self.logs[li]
        cam = frame_cameras(ld, t, self.wh)
        out = {"img": cam["img"], "cam2ego": cam["cam2ego"], "K": cam["K"], "ego": ego_status(ld, t),
               "traj": ego_plan(ld, t), "token": ld.token[t], "log": ld.name}
        if self.prev:
            out.update(prev_cameras(ld, t, self.wh))
        box, vel, cls, track = gt_boxes(ld, t, self.area)
        out.update(pad_boxes(box, vel, cls))
        out["gt_track"] = ",".join(track.tolist())  # track tokens of gt_box[:n] in order (per-pair PDM labels; a str)
        if not self.train:
            return out
        if self.prev:  # t-1 GT boxes in the t0 frame: targets of the t-1 detection pass (slot history)
            pb, pv, pc, _ = gt_boxes(ld, t, self.area, frame=t - 1)
            out.update({k + "_prev": v for k, v in pad_boxes(pb, pv, pc).items()})
        # future poses of every t0 box (t0 ego frame): motion and collision targets
        E0inv = np.linalg.inv(ld.e2g[t])
        fut = np.zeros((M_MAX, N_POSES, 3), np.float32)
        fut_ok = np.zeros((M_MAX, N_POSES), bool)
        pos = {tok: j for j, tok in enumerate(track)}
        for k in range(N_POSES):
            f = t + 1 + k
            a = ld.anns[f]
            T = E0inv @ ld.l2g[f]
            for r, tok in enumerate(a["track"]):
                j = pos.get(tok)
                if j is not None:
                    b = to_frame(T, a["box"][r:r + 1])[0]
                    fut[j, k] = [b[0], b[1], b[6]]
                    fut_ok[j, k] = True
        out["gt_fut"], out["gt_fut_ok"] = fut, fut_ok
        # JEPA targets: future images and, per t0 box, its GT box at each horizon in that frame's ego frame
        imgs, c2e, Ks = [], [], []
        hbox = np.zeros((len(self.horizons), M_MAX, 7), np.float32)
        hok = np.zeros((len(self.horizons), M_MAX), bool)
        for hi, h in enumerate(self.horizons):
            f = t + h
            if self.load_future_images:
                c = frame_cameras(ld, f, self.wh)
                imgs.append(c["img"]), c2e.append(c["cam2ego"]), Ks.append(c["K"])
            a = ld.anns[f]
            T = np.linalg.inv(ld.e2g[f]) @ ld.l2g[f]
            for r, tok in enumerate(a["track"]):
                j = pos.get(tok)
                if j is not None:
                    hbox[hi, j] = to_frame(T, a["box"][r:r + 1])[0]
                    hok[hi, j] = True
        if self.load_future_images:
            out["fut_img"], out["fut_cam2ego"], out["fut_K"] = torch.stack(imgs), np.stack(c2e), np.stack(Ks)
        out["fut_box"], out["fut_box_ok"] = hbox, hok
        # ego pose at each horizon in the t0 frame (to express future slot geometry in the t0 frame)
        out["fut_ego"] = np.stack([E0inv @ ld.e2g[t + h] for h in self.horizons]).astype(np.float32)
        b = self.bev(ld, t)
        out["bev"] = np.zeros((64, 64), np.int64) if b is None else b.astype(np.int64)
        out["bev_ok"] = b is not None
        return out
