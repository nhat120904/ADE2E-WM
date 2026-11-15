"""Phase 0 audit of OpenScene/NAVSIM temporal data.

Answers, from the released metadata and whatever sensor blobs are on disk:
  A. schema of a frame record (cameras, calibration, annotations, tracks);
  B. frame rate and camera/LiDAR timestamp alignment;
  C. persistence of agent track tokens over time, classes and agents per frame;
  D. availability of future camera frames on disk for scenes in the mini split;
  E. for navtrain, which future frames are covered by the navtrain sensor download
     (current + 3 history frames of every navtrain scene), computed from metadata only.

Nothing here invents fields: every statistic is computed only if the key exists, and
missing keys are reported as missing.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CAMS = ["CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0"]
DRIVOR_CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]


def describe(obj, depth=0, max_depth=3):
    """Compact recursive type/shape description of a frame record."""
    if isinstance(obj, dict):
        if depth >= max_depth:
            return f"dict[{len(obj)}]"
        return {str(k): describe(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return f"ndarray{tuple(obj.shape)}:{obj.dtype}"
    if isinstance(obj, (list, tuple)):
        inner = describe(obj[0], depth + 1, max_depth) if len(obj) else None
        return f"{type(obj).__name__}[{len(obj)}] of {inner}"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return f"{type(obj).__name__}={str(obj)[:60]}"
    return type(obj).__name__


def load_log(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def frame_stats(frames):
    """Timestamps, camera sync and track statistics for one log (list of frame dicts)."""
    out = {"n_frames": len(frames)}
    ts = np.array([f.get("timestamp", np.nan) for f in frames], dtype=np.float64)
    if np.isfinite(ts).all() and len(ts) > 1:
        out["dt_s"] = (np.diff(ts) / 1e6).tolist()
    # camera timestamp offsets relative to the frame (LiDAR) timestamp, when recorded
    offs = collections.defaultdict(list)
    for f in frames:
        for cam, c in (f.get("cams") or {}).items():
            if isinstance(c, dict) and "timestamp" in c and "timestamp" in f:
                offs[cam].append((c["timestamp"] - f["timestamp"]) / 1e3)  # ms
    out["cam_offsets_ms"] = {k: v for k, v in offs.items()}
    # tracks
    track_frames = collections.defaultdict(list)
    names_by_track = {}
    per_frame = []
    for i, f in enumerate(frames):
        anns = f.get("anns") or {}
        tt = anns.get("track_tokens")
        nm = anns.get("gt_names")
        if tt is None:
            continue
        per_frame.append(len(tt))
        for j, t in enumerate(tt):
            track_frames[t].append(i)
            if nm is not None:
                names_by_track[t] = nm[j]
    lengths, gaps = [], 0
    for t, idx in track_frames.items():
        idx = sorted(idx)
        lengths.append(len(idx))
        gaps += int(np.sum(np.diff(idx) > 1)) if len(idx) > 1 else 0
    out["agents_per_frame"] = per_frame
    out["track_lengths"] = lengths
    out["track_gaps"] = gaps
    out["track_classes"] = dict(collections.Counter(names_by_track.values()))
    return out


def summarize(values, name):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return f"{name}: none"
    q = np.percentile(v, [0, 5, 50, 95, 100])
    return f"{name}: n={v.size} mean={v.mean():.3f} min/p5/p50/p95/max={np.round(q, 3).tolist()}"


def audit_split(log_dir: Path, sensor_dir: Path | None, max_logs: int, report: dict):
    logs = sorted(log_dir.glob("*.pkl"))
    report["n_logs"] = len(logs)
    if not logs:
        return
    first = load_log(logs[0])
    report["record_schema"] = describe(first[0])
    report["record_keys"] = sorted(first[0].keys())
    dts, per_frame, lengths, gaps = [], [], [], 0
    cls = collections.Counter()
    offs = collections.defaultdict(list)
    fut_avail = collections.Counter()
    n_scenes = 0
    for lp in logs[:max_logs]:
        frames = load_log(lp)
        s = frame_stats(frames)
        dts += s.get("dt_s", [])
        per_frame += s["agents_per_frame"]
        lengths += s["track_lengths"]
        gaps += s["track_gaps"]
        cls.update(s["track_classes"])
        for k, v in s["cam_offsets_ms"].items():
            offs[k] += v
        if sensor_dir is not None:
            # scenes of 4 history + 8 future frames, as in NAVSIM; check image files of the 4 DrivoR cams
            for t in range(3, len(frames) - 8):
                ok_hist = all(
                    (sensor_dir / frames[t - h]["cams"][c]["data_path"]).exists()
                    for h in range(4) for c in DRIVOR_CAMS if c in frames[t - h].get("cams", {})
                )
                if not ok_hist:
                    continue
                n_scenes += 1
                for k in range(1, 9):
                    fr = frames[t + k]
                    if all((sensor_dir / fr["cams"][c]["data_path"]).exists() for c in DRIVOR_CAMS):
                        fut_avail[k] += 1
    report["frame_dt_s"] = summarize(dts, "dt")
    report["agents_per_frame"] = summarize(per_frame, "agents/frame")
    report["track_length_frames"] = summarize(lengths, "track length (frames)")
    report["track_gaps_total"] = gaps
    report["n_tracks"] = len(lengths)
    report["track_classes"] = dict(cls.most_common())
    report["cam_offsets_ms"] = {k: summarize(v, k) for k, v in offs.items()}
    if sensor_dir is not None:
        report["scenes_with_history_on_disk"] = n_scenes
        report["future_frame_availability"] = {
            f"t+{k}": (fut_avail[k] / n_scenes if n_scenes else None) for k in range(1, 9)
        }


def _navtrain_log(args):
    lp, tokset = args
    frames = load_log(lp)
    toks = [f["token"] for f in frames]
    idx = {t: i for i, t in enumerate(toks)}
    cur = [idx[t] for t in toks if t in tokset]
    have = set()
    for i in cur:
        have.update(range(max(0, i - 3), i + 1))
    res = np.zeros((len(cur), 10), dtype=bool)
    for r, i in enumerate(cur):
        for k in range(1, 11):
            res[r, k - 1] = (i + k) in have and (i + k) < len(toks)
    return lp.name, len(toks), len(cur), len(have), res


def audit_navtrain(log_dir: Path, filter_yaml: Path, workers: int, report: dict):
    s = filter_yaml.read_text()
    tokens = set(re.findall(r"^\s*-\s*'?([0-9a-f]{16})'?\s*$", s.split("tokens:")[1], re.M))
    log_names = re.findall(r"^\s*-\s*'?([^'\s]+)'?\s*$", s.split("log_names:")[1].split("tokens:")[0], re.M)
    paths = [log_dir / f"{n}.pkl" for n in log_names]
    missing = [p.name for p in paths if not p.exists()]
    paths = [p for p in paths if p.exists()]
    report["navtrain_tokens"] = len(tokens)
    report["navtrain_logs"] = len(log_names)
    report["navtrain_logs_missing_metadata"] = len(missing)
    allres, frames_total, frames_sensor = [], 0, 0
    with ProcessPoolExecutor(workers) as ex:
        for name, n, ncur, nhave, res in ex.map(_navtrain_log, [(p, tokens) for p in paths], chunksize=4):
            frames_total += n
            frames_sensor += nhave
            if ncur:
                allres.append(res)
    res = np.concatenate(allres) if allres else np.zeros((0, 10), bool)
    report["navtrain_scenes_found"] = int(res.shape[0])
    report["navtrain_frames_in_logs"] = frames_total
    report["navtrain_frames_with_sensors (current+3 history)"] = frames_sensor
    report["navtrain_future_frame_coverage"] = {f"t+{k+1}": float(res[:, k].mean()) for k in range(10)}
    report["navtrain_scenes_all_8_future_covered"] = float(res[:, :8].all(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--navsim-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-logs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    report = {}
    for split, sensors in [("mini", args.data_root / "sensor_blobs/mini"), ("test", None)]:
        d = args.data_root / "navsim_logs" / split
        if d.exists():
            report[split] = {}
            audit_split(d, sensors if (sensors and sensors.exists()) else None, args.max_logs, report[split])
    tv = args.data_root / "navsim_logs/trainval"
    if tv.exists():
        report["navtrain"] = {}
        audit_navtrain(
            tv,
            args.navsim_root / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
            args.workers,
            report["navtrain"],
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
