"""Verify the streamed camera data: per-frame availability of the 4 kept cameras, and scene coverage.

trainval pilot: for navtrain scenes in the pilot logs, share with all 4 history frames and all 8 future
frames on disk (the JEPA target horizon). test: share of navtest / navhard stage-one scenes whose
4 history frames are on disk. Also re-checks that no pilot log is a test log.
"""

import argparse
import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]


def filter_tokens(yaml_path: Path):
    s = yaml_path.read_text()
    logs = set(re.findall(r"^\s*-\s*'?([0-9]{4}\.[0-9.]+_veh-[0-9]+_[0-9_]+)'?\s*$", s.split("log_names:")[1].split("tokens:")[0], re.M))
    toks = set(re.findall(r"^\s*-\s*'?([0-9a-f]{16})'?\s*$", s.split("tokens:")[1], re.M))
    return logs, toks


def check_log(args):
    log_pkl, sensor_root, tokens, n_hist, n_fut = args
    frames = pickle.load(open(log_pkl, "rb"))
    have = [all((sensor_root / f["cams"][c]["data_path"]).exists() for c in CAMS) for f in frames]
    res = {"frames": len(frames), "frames_ok": sum(have), "scenes": 0, "hist_ok": 0, "hist_fut_ok": 0}
    for i, f in enumerate(frames):
        if f["token"] not in tokens:
            continue
        res["scenes"] += 1
        h = i - n_hist + 1 >= 0 and all(have[i - n_hist + 1 : i + 1])
        res["hist_ok"] += h
        res["hist_fut_ok"] += h and i + n_fut < len(frames) and all(have[i + 1 : i + n_fut + 1])
    return res


def run(log_dir, sensor_root, logs, tokens, n_fut, workers):
    jobs = [(log_dir / f"{l}.pkl", sensor_root, tokens, 4, n_fut) for l in sorted(logs)]
    tot = {"logs": len(jobs)}
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(check_log, jobs):
            for k, v in r.items():
                tot[k] = tot.get(k, 0) + v
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--navsim-root", type=Path, required=True)
    ap.add_argument("--pilot-logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sf = a.navsim_root / "navsim/planning/script/config/common/train_test_split/scene_filter"
    pilot = set(a.pilot_logs.read_text().split())
    test_logs = {p.stem for p in (a.data_root / "navsim_logs/test").glob("*.pkl")}
    rep = {"pilot_logs_in_test_split": len(pilot & test_logs)}
    nt_logs, nt_toks = filter_tokens(sf / "navtrain.yaml")
    rep["trainval_pilot_navtrain"] = run(a.data_root / "navsim_logs/trainval", a.data_root / "sensor_blobs/trainval",
                                         pilot & nt_logs, nt_toks, 8, a.workers)
    for name in ["navtest", "navhard_two_stage"]:
        logs, toks = filter_tokens(sf / f"{name}.yaml")
        rep[f"test_{name}"] = run(a.data_root / "navsim_logs/test", a.data_root / "sensor_blobs/test", logs & test_logs, toks, 0, a.workers)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
