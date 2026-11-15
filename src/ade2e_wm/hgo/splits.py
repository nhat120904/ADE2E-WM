"""Compositional factors of a t0 frame for the HGO splits (docs/METHOD.md). Labels only (never model inputs).

maneuver  from the logged 4 s ego plan: stop (< 2 m travelled), left / right (heading change > 20 deg), straight
vru       a pedestrian or bicycle box at t0 in the ego corridor (x in [-2, 30] m, |y| < 8 m)
density   vehicle boxes within 32 m: low (<= 5), mid (6-15), high (>= 16)
"""

import numpy as np

from .data import LogData, ego_plan, gt_boxes

FACTORS = ["maneuver", "vru", "density"]


def factors(ld: LogData, t):
    plan = ego_plan(ld, t)
    d = float(np.hypot(*plan[-1, :2]))
    dyaw = float(plan[-1, 2])
    man = "stop" if d < 2.0 else "left" if dyaw > np.deg2rad(20) else "right" if dyaw < -np.deg2rad(20) else "straight"
    box, _, cls, _ = gt_boxes(ld, t)
    vru = bool((((cls == 1) | (cls == 2)) & (box[:, 0] >= -2) & (box[:, 0] <= 30) & (np.abs(box[:, 1]) < 8)).any())
    nveh = int(((cls == 0) & (np.hypot(box[:, 0], box[:, 1]) <= 32)).sum())
    dens = "low" if nveh <= 5 else "mid" if nveh <= 15 else "high"
    return {"maneuver": man, "vru": "vru" if vru else "no_vru", "density": dens, "n_vehicles": nveh}
