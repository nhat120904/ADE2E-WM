"""Sample figures of the CG split folds (docs/CG_SPLIT.md §9) for reports. GT annotations only (labels, not model output).

Each sample row: front-left | front | front-right cameras at t0, and a bird's-eye view in the t0 ego frame (forward up):
map background from exp/hgo/bev (road / walkway / centre line, when built), ego box and logged path (past dashed, next
8 s with a dot per second), agents at t0 with their next 6 s, and the conflict partner of the row's category in red (the
agent of the group whose trajectory crosses the logged ego path with the smallest PET, as the strict cell of
cg_cells.py).

Per fold (data/_lists/cg/<split>/<fold>/test_groups.csv and train_withheld.txt):
  cell          test scenes of the held-out combination (navtest or CG-test sessions)
  same_man      training windows with the same maneuver and no partner
  same_partner  training windows with the same partner during another maneuver
Rows of cell / same_partner prefer clear conflicts (PET < --max-pet s, ego reaches the conflict in 0.5-4 s).
Outputs in --out: <fold>_overview.png (one row per category), <fold>_<category>.png (--n rows), rest.png (training
windows outside every fold), index.md (token, log, labels of every row).

usage: python scripts/hgo/cg_samples.py [--split-name split_v1] [--folds left_vru right_vru left_veh_opp] [--n 3]
       python scripts/hgo/cg_samples.py --tokens <token> [<token> ...]      (any trainval / navtest frame)
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hgo"))
from cg_events import crossing, path_conflict, wrap180  # noqa: E402

from ade2e_wm.hgo.data import LogData  # noqa: E402
from ade2e_wm.hgo.geometry import EGO_LENGTH, EGO_REAR_TO_CENTER, EGO_WIDTH, to_frame  # noqa: E402

DT = 0.5
MAN_VI = {"left": "rẽ trái ở giao lộ", "right": "rẽ phải ở giao lộ", "straight": "đi thẳng", "stop": "dừng / gần đứng yên",
          "curve": "đi theo đường cong", "turn": "rẽ trái / phải ở giao lộ"}
PART_VI = {"vru": "người đi bộ / xe đạp cắt ngang", "veh_opp": "xe ngược chiều", "veh_cross": "xe cắt ngang",
           "none": "không có đối tác xung đột"}
CAT_VI = {"cell": "TEST — tổ hợp bị giữ lại (model chưa từng thấy)",
          "same_man": "TRAIN — cùng hành động, không có đối tác",
          "same_partner": "TRAIN — cùng đối tác, hành động khác",
          "rest": "TRAIN — phần còn lại của dữ liệu"}
MAP_RGB = {0: (1, 1, 1), 1: (0.86, 0.86, 0.86), 2: (0.95, 0.91, 0.82), 3: (0.70, 0.70, 0.70)}
LOGS = {}


def log_data(split, log):
    if (split, log) not in LOGS:
        if len(LOGS) > 8:
            LOGS.clear()
        LOGS[(split, log)] = LogData(ROOT / f"data/navsim_logs/{split}/{log}.pkl", ROOT / f"data/sensor_blobs/{split}")
    return LOGS[(split, log)]


def box_corners(b):
    x, y, l, w, yaw = b[0], b[1], b[3], b[4], b[6]
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


def scene(split, log, t, group, fut=12, ego_fut=16):
    """Everything drawn for frame t of a log, in the t0 ego frame (x forward, y left)."""
    ld = log_data(split, log)
    F = len(ld.token)
    gap_ok = np.abs(np.diff(ld.ts) / 1e6 - DT) <= 0.1
    K = 0
    while K < 20 and t + K + 1 < F and gap_ok[t + K]:
        K += 1
    E0 = np.linalg.inv(ld.e2g[t])
    ctr = np.stack([(E0 @ ld.e2g[k] @ np.array([EGO_REAR_TO_CENTER, 0, 0, 1.0]))[:2] for k in range(max(0, t - 3), t + K + 1)])
    past, futp = ctr[: min(3, t) + 1], ctr[min(3, t):]
    boxes, tracks = {}, {}
    for k in range(t, t + min(K, max(fut, 20)) + 1):
        a = ld.anns[k]
        b = to_frame(E0 @ ld.l2g[k], a["box"])
        for j, tr in enumerate(a["track"]):
            tracks.setdefault(tr, {})[k - t] = b[j]
    t0 = {tr: v[0] for tr, v in tracks.items() if 0 in v}
    cls0 = dict(zip(ld.anns[t]["track"], ld.anns[t]["cls"]))
    partner, info = None, {}
    if group in ("vru", "veh_opp", "veh_cross"):
        want = (1, 2) if group == "vru" else (0,)
        cand = [tr for tr in t0 if cls0[tr] in want and np.hypot(*t0[tr][:2]) <= 50]
        if cand:
            P = np.zeros((K + 1, len(cand), 2))
            V = np.zeros((K + 1, len(cand)), bool)
            Y = np.full((K + 1, len(cand)), np.nan)
            for j, tr in enumerate(cand):
                for k, b in tracks[tr].items():
                    if k <= K:
                        P[k, j], V[k, j], Y[k, j] = b[:2], True, np.degrees(b[6])
            C = futp[: K + 1]
            dmin, pet, tconf, kc, sc = path_conflict(P, V, C, 2.5 if group == "vru" else 3.0)
            pet = np.where(crossing(P, V, C), pet, np.inf)  # partner = crossing agent, as the strict cell (cg_cells.py)
            if group != "vru":
                seg = C[np.minimum(sc + 1, len(C) - 1)] - C[sc]
                pyaw = np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
                ay = Y[kc, np.arange(len(cand))]
                rel = wrap180(np.where(np.isnan(ay), Y[0], ay) - pyaw)
                ok = rel > 135 if group == "veh_opp" else (rel >= 45) & (rel <= 135)
                pet = np.where(ok, pet, np.inf)
            j = int(np.argmin(pet))
            if np.isfinite(pet[j]):
                partner, info = cand[j], {"pet": float(pet[j]), "tconf": float(tconf[j])}
    bev, vmap = None, None
    try:
        vmap = vector_map(ld.map_location, ld.e2g[t])
    except Exception as e:  # fall back to the 1 m BEV cache
        print(f"map query failed ({type(e).__name__}: {e}); using exp/hgo/bev", file=sys.stderr)
        f = ROOT / f"exp/hgo/bev/{log}.npz"
        if f.exists():
            z = np.load(f)
            idx = np.nonzero(z["token"] == ld.token[t])[0]
            if len(idx):
                bev = z["bev"][idx[0]]
    cams = []
    for c in ("CAM_L0", "CAM_F0", "CAM_R0"):
        p = ld.sensor_root / ld.cams[t][c]["data_path"]
        im = cv2.imread(str(p))
        cams.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (640, 360)) if im is not None
                    else np.full((360, 640, 3), 230, np.uint8))
    return {"past": past, "fut": futp[: ego_fut + 1], "t0": t0, "cls0": cls0, "tracks": tracks, "partner": partner,
            "info": info, "bev": bev, "vmap": vmap, "cams": np.concatenate(cams, 1), "city": ld.map_location, "fut_n": fut}


MAPS = {}


def vector_map(city, e2g, radius=55.0):
    """nuPlan map polygons / centre lines around the ego, in the ego frame: {kind: [array (n,2)]}."""
    import os

    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map import SemanticMapLayer as L
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    if city not in MAPS:
        MAPS[city] = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"], os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"),
                                  city)
    g2e = np.linalg.inv(e2g)
    xy = lambda pts: np.asarray(pts)[:, :2] @ g2e[:2, :2].T + g2e[:2, 3]
    layers = [L.LANE, L.LANE_CONNECTOR, L.INTERSECTION, L.WALKWAYS, L.CROSSWALK]
    objs = MAPS[city].get_proximal_map_objects(Point2D(e2g[0, 3], e2g[1, 3]), radius, layers)
    out = {"road": [], "walkway": [], "crosswalk": [], "centre": []}
    for layer, kind in [(L.LANE, "road"), (L.LANE_CONNECTOR, "road"), (L.INTERSECTION, "road"), (L.WALKWAYS, "walkway"),
                        (L.CROSSWALK, "crosswalk")]:
        out[kind] += [xy(o.polygon.exterior.coords) for o in objs[layer]]
    for layer in (L.LANE, L.LANE_CONNECTOR):
        out["centre"] += [xy(o.baseline_path.linestring.coords) for o in objs[layer]]
    return out


def draw_bev(ax, s):
    U = lambda xy: (-np.asarray(xy)[..., 1], np.asarray(xy)[..., 0])  # plot coords: right = -y, up = x
    if s.get("vmap"):
        style = {"road": dict(fc="#dcdcdc", ec="none"), "walkway": dict(fc="#f2e8d0", ec="none"),
                 "crosswalk": dict(fc="#cfe3f5", ec="#8fb3d6", lw=0.5)}
        for kind in ("road", "walkway", "crosswalk"):
            for poly in s["vmap"][kind]:
                ax.add_patch(Polygon(np.c_[U(poly)], closed=True, zorder=0 if kind == "road" else 1, **style[kind]))
        for line in s["vmap"]["centre"]:
            ax.plot(*U(line), color="#b5b5b5", lw=0.5, ls=(0, (4, 3)), zorder=1)
    elif s["bev"] is not None:
        m = s["bev"][::-1, ::-1]
        rgb = np.ones(m.shape + (3,))
        for c, col in MAP_RGB.items():
            rgb[m == c] = col
        ax.imshow(rgb, extent=[-32, 32, -32, 32], origin="upper", interpolation="nearest", zorder=0)
    for tr, b in s["t0"].items():
        c = s["cls0"][tr]
        hit = tr == s["partner"]
        col = "#d62728" if hit else {0: "#555555", 1: "#ff7f0e", 2: "#ff7f0e", 3: "#222222"}[c]
        track = np.array([s["tracks"][tr][k][:2] for k in sorted(s["tracks"][tr]) if k <= s["fut_n"]])
        if c != 3 and len(track) > 1:
            ax.plot(*U(track), color=col, lw=2.2 if hit else 0.9, alpha=1 if hit else 0.6, zorder=3 if hit else 2)
        if c in (1, 2):
            ax.scatter(*U(b[:2]), s=40 if hit else 16, color=col, zorder=4, edgecolors="k" if hit else "none")
        else:
            ax.add_patch(Polygon(np.c_[U(box_corners(b))], closed=True, fc=col if hit else "none", ec=col,
                                 lw=1.5 if hit else 0.8, alpha=0.9, zorder=3))
    ego = box_corners(np.array([EGO_REAR_TO_CENTER, 0, 0, EGO_LENGTH, EGO_WIDTH, 0, 0]))
    ax.add_patch(Polygon(np.c_[U(ego)], closed=True, fc="#1f77b4", ec="k", lw=0.8, zorder=5))
    ax.plot(*U(s["past"]), "--", color="#1f77b4", lw=1.5, zorder=4)
    ax.plot(*U(s["fut"]), "-", color="#1f77b4", lw=2.5, zorder=4)
    ax.scatter(*U(s["fut"][2::2]), s=10, color="#1f77b4", zorder=5)
    ax.set_xlim(-24, 24)
    ax.set_ylim(-10, 38)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#1f77b4", lw=2.5, label="xe mình: quỹ đạo thật 8 s tới (chấm = 1 s)"),
         Line2D([], [], color="#d62728", lw=2.2, marker="o", label="đối tác xung đột (6 s tới)"),
         Line2D([], [], color="#ff7f0e", lw=1, marker="o", ms=4, ls="-", label="người đi bộ / xe đạp"),
         Line2D([], [], color="#555555", lw=1, label="xe khác"),
         Line2D([], [], color="#cfe3f5", lw=6, label="vạch qua đường"),
         Line2D([], [], color="#f2e8d0", lw=6, label="vỉa hè")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))


def render(rows, path, title=""):
    fig = plt.figure(figsize=(17, 3.3 * len(rows) + 0.8))
    gs = fig.add_gridspec(len(rows), 2, width_ratios=[16 / 3, 1.6], wspace=0.03, hspace=0.38)
    for i, r in enumerate(rows):
        s = r["scene"]
        a1, a2 = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        a1.imshow(s["cams"])
        w = s["cams"].shape[1] / 3
        a1.set_xticks([w / 2, 1.5 * w, 2.5 * w])
        a1.set_xticklabels(["camera trước-trái", "camera trước", "camera trước-phải"], fontsize=8)
        a1.set_yticks([])
        draw_bev(a2, s)
        a2.set_title("nhìn từ trên (hướng đi lên)", fontsize=8)
        a1.set_title(r["label"], fontsize=10, loc="left")
    if title:
        fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    legend(fig)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def label(cat, ev, s, group):
    man, part = ev.get("man"), ev.get("partner", "none")
    txt = f"{CAT_VI.get(cat, cat)}  |  {MAN_VI.get(man, man)} × {PART_VI.get(part, part)}"
    if s["info"]:
        txt += f"  |  PET {s['info']['pet']:.1f} s, xe tới điểm xung đột sau {s['info']['tconf']:.1f} s"
    return txt + f"\n{s['city']}  ·  {ev['log']}  ·  token {ev['token']}  ·  {ev.get('source', 'train')}"


def pick(df, n, rng, clear_group=None, max_pet=1.5):
    if clear_group:
        c = df[(df[f"petx10_{clear_group}"] < max_pet) & df[f"tconfx10_{clear_group}"].between(0.5, 4.0)]
        df = c if len(c) >= n else df
    if not len(df):
        return df
    # one row per log for variety
    df = df.sample(frac=1, random_state=int(rng.integers(1 << 31))).drop_duplicates("log")
    return df.head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-name", default="split_v1")
    ap.add_argument("--folds", nargs="+", default=["turn_vru", "left_veh_opp"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--rest", type=int, default=4)
    ap.add_argument("--max-pet", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", nargs="*", default=[])
    ap.add_argument("--out", default="docs/figures/cg_samples")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    sys.path.insert(0, str(ROOT / "scripts/hgo"))
    from cg_cells import PARTNERS, label_cells
    cols = ["token", "log", "t", "man4", "man10", "inter10"] + [f"{k}{h}_{g}" for k in ("pet", "tconf", "petx", "tconfx") for h in (4, 10) for g in PARTNERS if not (k.endswith("x") and h == 4)]
    keep = ["token", "log", "t", "man", "partner"] + [f"{k}10_{g}" for k in ("petx", "tconfx") for g in PARTNERS]
    load = lambda s: label_cells(pd.read_parquet(ROOT / f"data/_lists/cg/events_{s}.parquet", columns=cols), 3.0, 6.0)[keep]
    ev = pd.concat([load("trainval").assign(split="trainval"), load("test").assign(split="test")])
    ev = ev.drop_duplicates("token").set_index("token", drop=False)
    index = ["# Sample của các fold CG (`scripts/hgo/cg_samples.py`)", ""]

    def build(cat, r, group):
        s = scene(r["split"], r["log"], int(r["t"]), group)
        lab = label(cat, r, s, group)
        index.append(f"- {cat}: token `{r['token']}`, log `{r['log']}`, t={int(r['t'])}, {r['man']} × {r['partner']}"
                     + (f", PET {s['info']['pet']:.1f} s" if s["info"] else ""))
        return {"scene": s, "label": lab}

    if a.tokens:
        rows = [build("token", ev.loc[t], ev.loc[t, "partner"]) for t in a.tokens]
        render(rows, out / "tokens.png")
        print(out / "tokens.png")
        return
    d = ROOT / "data/_lists/cg" / a.split_name
    all_withheld = None
    for fold in a.folds:
        m, p = fold.split("_", 1)
        ms = {"turn": ["left", "right"]}.get(m, [m])
        tg = pd.read_csv(d / fold / "test_groups.csv")
        train = set((d / fold / "train_withheld.txt").read_text().split())
        all_withheld = train if all_withheld is None else all_withheld & train
        cell = ev.loc[tg.loc[tg["group"] == "cell", "token"]].assign(
            source=tg.set_index("token").loc[tg.loc[tg["group"] == "cell", "token"], "source"].to_numpy())
        trn = ev.loc[ev.index.isin(train)]
        cats = {"cell": (pick(cell, a.n, rng, p, a.max_pet), p),
                "same_man": (pick(trn[trn["man"].isin(ms) & (trn["partner"] == "none")], a.n, rng), None),
                "same_partner": (pick(trn[(trn["partner"] == p) & ~trn["man"].isin(ms + ["stop", "curve"])], a.n, rng, p,
                                      a.max_pet), p)}
        index += ["", f"## Fold {fold}: {MAN_VI[m]} × {PART_VI[p]}", ""]
        built = {}
        for cat, (df, g) in cats.items():
            built[cat] = [build(cat, r, g) for _, r in df.iterrows()]
            render(built[cat], out / f"{fold}_{cat}.png", f"Fold {MAN_VI[m]} × {PART_VI[p]}: {CAT_VI[cat]}")
            index.append(f"  → `{fold}_{cat}.png`")
        render([built[c][0] for c in cats if built[c]], out / f"{fold}_overview.png",
               f"Fold {MAN_VI[m]} × {PART_VI[p]}: test trên tổ hợp mới, train chỉ thấy từng thành phần")
        index.append(f"  → `{fold}_overview.png`")
        print("done", fold, flush=True)
    rest = ev.loc[ev.index.isin(all_withheld)]
    rest = rest[(rest["partner"] == "none") | (rest["man"] == "stop")]
    rows = [build("rest", r, None) for _, r in pick(rest, a.rest, rng).iterrows()]
    render(rows, out / "rest.png", CAT_VI["rest"])
    index += ["", "## Phần còn lại", "  → `rest.png`"]
    (out / "index.md").write_text("\n".join(index) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
